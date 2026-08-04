"""
FedChain :: Common utilities
============================

Small, dependency-light helpers shared by the trainer, blockchain, IPFS and
evaluation modules:

* logging bootstrap
* deterministic seeding
* CUDA detection with graceful CPU fallback
* SHA-256 fingerprinting of adapter bytes / files / directories
* size accounting and timing

Everything that touches ``torch`` imports it lazily so that the configuration
and audit layers stay importable on a machine without the deep-learning stack.
"""

from __future__ import annotations

import functools
import hashlib
import json
import logging
import os
import platform
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Tuple, Union

LOGGER = logging.getLogger(__name__)

PathLike = Union[str, os.PathLike]

#: Files that make up a PEFT adapter checkpoint, in the order we hash them.
ADAPTER_FILENAMES: Tuple[str, ...] = (
    "adapter_config.json",
    "adapter_model.safetensors",
    "adapter_model.bin",
)

_HASH_CHUNK = 1024 * 1024  # 1 MiB


# =============================================================================
# Logging
# =============================================================================
def setup_logging(level: str = "INFO", log_file: Optional[PathLike] = None) -> None:
    """Configure root logging once, with an optional mirrored file handler.

    Safe to call repeatedly: existing handlers are replaced rather than stacked,
    which keeps log lines from being duplicated when submodules re-initialise.
    """
    numeric_level = getattr(logging, str(level).upper(), logging.INFO)
    fmt = "%(asctime)s | %(levelname)-8s | %(name)-28s | %(message)s"
    datefmt = "%H:%M:%S"

    root = logging.getLogger()
    for handler in list(root.handlers):
        root.removeHandler(handler)
        try:
            handler.close()
        except Exception:  # pragma: no cover - handler already torn down
            pass

    stream_handler = logging.StreamHandler(stream=sys.stdout)
    stream_handler.setFormatter(logging.Formatter(fmt, datefmt=datefmt))
    root.addHandler(stream_handler)

    if log_file is not None:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_path, encoding="utf-8")
        file_handler.setFormatter(logging.Formatter(fmt, datefmt=datefmt))
        root.addHandler(file_handler)

    root.setLevel(numeric_level)

    # Third-party libraries are extremely chatty at INFO; keep the console useful.
    for noisy in ("urllib3", "web3", "filelock", "datasets", "httpx", "matplotlib"):
        logging.getLogger(noisy).setLevel(max(numeric_level, logging.WARNING))


# =============================================================================
# Reproducibility
# =============================================================================
def set_global_seed(seed: int = 42, deterministic: bool = False) -> int:
    """Seed python / numpy / torch RNGs. Returns the seed for convenience."""
    seed = int(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)

    try:
        import numpy as np

        np.random.seed(seed)
    except Exception:  # pragma: no cover - numpy always present in practice
        LOGGER.debug("numpy unavailable; skipping numpy seeding")

    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        if deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            try:
                torch.use_deterministic_algorithms(True, warn_only=True)
            except Exception:  # pragma: no cover - older torch
                pass
    except ImportError:
        LOGGER.debug("torch unavailable; skipping torch seeding")

    try:
        import transformers

        transformers.set_seed(seed)
    except Exception:
        pass

    LOGGER.debug("Global seed set to %d (deterministic=%s)", seed, deterministic)
    return seed


# =============================================================================
# Device management
# =============================================================================
@functools.lru_cache(maxsize=8)
def get_device(preference: str = "auto") -> Tuple[str, Dict[str, Any]]:
    """Resolve the compute device with a graceful CPU fallback.

    Parameters
    ----------
    preference:
        ``"auto"`` (detect), ``"cuda"`` (require, but fall back with a warning)
        or ``"cpu"`` (force).

    Returns
    -------
    (device_str, info_dict)
        ``device_str`` is ``"cuda"`` or ``"cpu"``. ``info_dict`` carries the GPU
        name / VRAM / capability used by the metrics report.
    """
    preference = (preference or "auto").lower()
    info: Dict[str, Any] = {
        "requested": preference,
        "torch_available": False,
        "cuda_available": False,
        "gpu_name": None,
        "gpu_total_memory_gb": None,
        "gpu_capability": None,
        "bf16_supported": False,
    }

    try:
        import torch
    except ImportError:
        LOGGER.warning("PyTorch is not installed - falling back to CPU stub mode.")
        return "cpu", info

    info["torch_available"] = True
    info["torch_version"] = torch.__version__

    if preference == "cpu":
        LOGGER.info("Device: CPU (forced by configuration).")
        return "cpu", info

    cuda_ok = False
    try:
        cuda_ok = bool(torch.cuda.is_available() and torch.cuda.device_count() > 0)
    except Exception as exc:  # pragma: no cover - broken driver install
        LOGGER.warning("CUDA probe failed (%s) - falling back to CPU.", exc)
        cuda_ok = False

    info["cuda_available"] = cuda_ok
    if not cuda_ok:
        if preference == "cuda":
            LOGGER.warning("CUDA was requested but is unavailable - falling back to CPU.")
        else:
            LOGGER.info("No CUDA device detected - running on CPU.")
        return "cpu", info

    try:
        props = torch.cuda.get_device_properties(0)
        info["gpu_name"] = props.name
        info["gpu_total_memory_gb"] = round(props.total_memory / (1024 ** 3), 2)
        info["gpu_capability"] = f"{props.major}.{props.minor}"
        info["cuda_version"] = torch.version.cuda
        info["bf16_supported"] = bool(getattr(torch.cuda, "is_bf16_supported", lambda: False)())
        LOGGER.info(
            "Device: CUDA -> %s (%.2f GB VRAM, sm_%s%s, bf16=%s)",
            info["gpu_name"],
            info["gpu_total_memory_gb"],
            props.major,
            props.minor,
            info["bf16_supported"],
        )
    except Exception as exc:  # pragma: no cover
        LOGGER.warning("Could not read CUDA device properties: %s", exc)

    return "cuda", info


def free_cuda_memory() -> None:
    """Best-effort VRAM reclamation - critical on a 4 GB card."""
    import gc

    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
            try:
                torch.cuda.reset_peak_memory_stats()
            except Exception:
                pass
    except ImportError:
        pass
    gc.collect()


def cuda_peak_memory_mb() -> Optional[float]:
    """Peak allocated VRAM since the last reset, in MB (``None`` on CPU)."""
    try:
        import torch

        if torch.cuda.is_available():
            return round(torch.cuda.max_memory_allocated() / (1024 ** 2), 2)
    except ImportError:
        pass
    return None


# =============================================================================
# Hashing
# =============================================================================
def sha256_bytes(payload: bytes) -> str:
    """SHA-256 of an in-memory buffer, lowercase hex."""
    return hashlib.sha256(payload).hexdigest()


def sha256_file(file_path: PathLike) -> str:
    """Streaming SHA-256 of a single file (constant memory)."""
    digest = hashlib.sha256()
    with open(file_path, "rb") as handle:
        for chunk in iter(lambda: handle.read(_HASH_CHUNK), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_directory(dir_path: PathLike, only: Optional[Iterable[str]] = None) -> str:
    """Deterministic SHA-256 over the contents of a directory.

    The digest folds in each file's POSIX-normalised relative path *and* its
    bytes, so a rename is as detectable as an edit. Iteration order is the
    sorted relative path, which makes the result stable across filesystems and
    operating systems (important: the paper's clients are simulated on Windows
    but the artefacts must verify on Linux nodes).
    """
    root = Path(dir_path)
    allow = set(only) if only is not None else None

    files = sorted(
        (p for p in root.rglob("*") if p.is_file()),
        key=lambda p: p.relative_to(root).as_posix(),
    )
    if allow is not None:
        files = [p for p in files if p.name in allow]

    digest = hashlib.sha256()
    for file_path in files:
        rel = file_path.relative_to(root).as_posix()
        digest.update(rel.encode("utf-8"))
        digest.update(b"\x00")
        with open(file_path, "rb") as handle:
            for chunk in iter(lambda: handle.read(_HASH_CHUNK), b""):
                digest.update(chunk)
        digest.update(b"\xff")
    return digest.hexdigest()


def sha256_path(target: PathLike, adapter_only: bool = True) -> str:
    """SHA-256 of a file *or* a directory, dispatching on what ``target`` is.

    When ``target`` is an adapter directory and ``adapter_only`` is True, only
    the canonical PEFT checkpoint files are folded in. That keeps the on-chain
    commitment stable even if the trainer drops incidental files
    (``README.md``, ``training_args.bin``, tokenizer copies, ...) next to them.
    """
    path = Path(target)
    if path.is_dir():
        only = ADAPTER_FILENAMES if adapter_only else None
        if only is not None and not any((path / name).exists() for name in only):
            only = None  # not an adapter dir after all - hash everything
        return sha256_directory(path, only=only)
    if path.is_file():
        return sha256_file(path)
    raise FileNotFoundError(f"Cannot hash missing path: {path}")


def sha256_any(target: Union[bytes, bytearray, memoryview, PathLike]) -> str:
    """SHA-256 of raw bytes or of whatever lives at a filesystem path."""
    if isinstance(target, (bytes, bytearray, memoryview)):
        return sha256_bytes(bytes(target))
    return sha256_path(target)


# =============================================================================
# Sizes and timing
# =============================================================================
def bytes_to_mb(num_bytes: Union[int, float]) -> float:
    """Bytes -> megabytes (MiB), rounded to 4 decimals."""
    return round(float(num_bytes) / (1024 ** 2), 4)


def dir_size_bytes(dir_path: PathLike) -> int:
    """Total size in bytes of every file under ``dir_path``."""
    root = Path(dir_path)
    if not root.exists():
        return 0
    return sum(p.stat().st_size for p in root.rglob("*") if p.is_file())


def path_size_bytes(target: PathLike) -> int:
    """Size in bytes of a file or of a whole directory tree."""
    path = Path(target)
    if path.is_file():
        return path.stat().st_size
    if path.is_dir():
        return dir_size_bytes(path)
    return 0


def format_duration(seconds: float) -> str:
    """Human-readable duration, e.g. ``1h 02m 03.4s``."""
    seconds = float(seconds)
    if seconds < 60:
        return f"{seconds:.2f}s"
    minutes, secs = divmod(seconds, 60)
    hours, minutes = divmod(int(minutes), 60)
    if hours:
        return f"{hours}h {minutes:02d}m {secs:04.1f}s"
    return f"{minutes}m {secs:04.1f}s"


class Timer:
    """Context manager measuring wall-clock seconds with a monotonic clock.

    >>> with Timer() as t:
    ...     pass
    >>> t.elapsed >= 0
    True
    """

    def __init__(self, label: Optional[str] = None, logger: Optional[logging.Logger] = None):
        self.label = label
        self.logger = logger or LOGGER
        self.start: float = 0.0
        self.elapsed: float = 0.0

    def __enter__(self) -> "Timer":
        self.start = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        self.elapsed = time.perf_counter() - self.start
        if self.label:
            self.logger.info("%s took %s", self.label, format_duration(self.elapsed))
        return False


# =============================================================================
# Misc
# =============================================================================
def write_json(path: PathLike, payload: Any, indent: int = 2) -> Path:
    """Serialise ``payload`` to JSON, creating parent directories as needed."""
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=indent, default=_json_default, ensure_ascii=False)
    return out_path


def _json_default(obj: Any) -> Any:
    """Fallback encoder for numpy scalars, Paths and other stragglers."""
    if isinstance(obj, Path):
        return str(obj)
    if hasattr(obj, "item"):  # numpy / torch scalars
        try:
            return obj.item()
        except Exception:
            pass
    if isinstance(obj, (set, frozenset)):
        return sorted(obj)
    return str(obj)


def describe_environment() -> Dict[str, Any]:
    """Snapshot of the runtime, embedded in every metrics report."""
    env: Dict[str, Any] = {
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "processor": platform.processor() or platform.machine(),
        "cpu_count": os.cpu_count(),
    }
    for module_name, key in (
        ("torch", "torch"),
        ("transformers", "transformers"),
        ("peft", "peft"),
        ("trl", "trl"),
        ("datasets", "datasets"),
        ("bitsandbytes", "bitsandbytes"),
        ("accelerate", "accelerate"),
        ("web3", "web3"),
    ):
        try:
            module = __import__(module_name)
            env[f"{key}_version"] = getattr(module, "__version__", "unknown")
        except Exception:
            env[f"{key}_version"] = None
    return env


def markdown_table(headers, rows) -> str:
    """Render a GitHub-flavoured Markdown table from headers + row tuples."""
    headers = [str(h) for h in headers]
    str_rows = [[("" if c is None else str(c)) for c in row] for row in rows]
    widths = [len(h) for h in headers]
    for row in str_rows:
        for idx, cell in enumerate(row):
            if idx < len(widths):
                widths[idx] = max(widths[idx], len(cell))

    def _line(cells):
        return "| " + " | ".join(c.ljust(widths[i]) for i, c in enumerate(cells)) + " |"

    out = [_line(headers), "|" + "|".join("-" * (w + 2) for w in widths) + "|"]
    out.extend(_line(row) for row in str_rows)
    return "\n".join(out)
