"""
FedChain :: Configuration loader
================================

YAML configuration with single-inheritance via an ``extends:`` key, deep
merging, ``${ENV_VAR}`` expansion and defensive numeric coercion.

    cfg = load_config("configs/exp4_fedchain.yaml")
    cfg.learning_rate        # attribute access
    cfg["learning_rate"]     # or dict access
    cfg.get("missing", 0)    # with defaults

Inheritance resolves relative to the *child* file's directory, so
``extends: base_config.yaml`` inside ``configs/exp2_fl.yaml`` picks up
``configs/base_config.yaml``. Chains are supported and cycles are detected.
"""

from __future__ import annotations

import copy
import logging
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Union

try:
    import yaml
except ImportError as exc:  # pragma: no cover - hard requirement
    raise ImportError(
        "PyYAML is required to load FedChain configs. Install it with: pip install pyyaml"
    ) from exc

LOGGER = logging.getLogger(__name__)

#: Repository root (parent of the ``utils`` package).
PROJECT_ROOT = Path(__file__).resolve().parent.parent

#: Key used for single inheritance. ``inherits`` is accepted as an alias.
_EXTENDS_KEYS = ("extends", "inherits", "base_config")

_ENV_PATTERN = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?::-([^}]*))?\}")

#: Keys coerced to float regardless of how PyYAML resolved them. PyYAML's
#: implicit float resolver rejects ``2e-4`` (unsigned exponent, no decimal
#: point) and silently yields the *string* "2e-4", which would crash the
#: optimiser deep inside HF Trainer with an unhelpful message.
_FLOAT_KEYS = frozenset(
    {
        "learning_rate",
        "lora_dropout",
        "weight_decay",
        "warmup_ratio",
        "max_grad_norm",
        "adam_beta1",
        "adam_beta2",
        "adam_epsilon",
    }
)

_INT_KEYS = frozenset(
    {
        "seed",
        "num_rounds",
        "local_epochs",
        "max_seq_length",
        "batch_size",
        "grad_accum_steps",
        "lora_r",
        "lora_alpha",
        "num_clients",
        "logging_steps",
        "max_steps",
        "eval_num_samples",
        "eval_batch_size",
        "eval_max_seq_length",
        "gen_num_samples",
        "gen_max_new_tokens",
        "gen_max_prompt_tokens",
        "eval_round_stride",
        "save_steps",
        "save_total_limit",
        "dataloader_num_workers",
        "blockchain_tx_timeout",
        "ipfs_timeout",
        "max_train_samples",
        "chain_id",
    }
)

_BOOL_KEYS = frozenset(
    {
        "enable_fl",
        "enable_blockchain",
        "enable_ipfs",
        "load_in_4bit",
        "bnb_4bit_use_double_quant",
        "gradient_checkpointing",
        "keep_model_loaded",
        "trust_remote_code",
        "use_chat_template",
        "deterministic",
        "eval_every_round",
        "eval_final",
        "eval_local_clients_every_round",
        "eval_loss_on_completion_only",
        "enable_generation_metrics",
        "enable_checkpointing",
        "enable_step_checkpoints",
        "log_global_model",
        "ipfs_roundtrip_aggregation",
        "verify_hash_on_download",
        "dry_run",
    }
)


class Config(dict):
    """A ``dict`` that also supports attribute access.

    Unknown attributes raise ``AttributeError`` (not ``KeyError``) so typos
    surface as normal Python errors rather than silent ``None`` propagation.
    """

    def __getattr__(self, name: str) -> Any:
        try:
            return self[name]
        except KeyError as exc:
            raise AttributeError(
                f"Config has no key {name!r}. Available keys: {sorted(self.keys())}"
            ) from exc

    def __setattr__(self, name: str, value: Any) -> None:
        self[name] = value

    def __delattr__(self, name: str) -> None:
        try:
            del self[name]
        except KeyError as exc:
            raise AttributeError(name) from exc

    def copy(self) -> "Config":  # type: ignore[override]
        return Config(copy.deepcopy(dict(self)))

    def to_dict(self) -> Dict[str, Any]:
        """Plain, JSON-serialisable dictionary copy."""
        return copy.deepcopy(dict(self))


# =============================================================================
# Merging / expansion / coercion
# =============================================================================
def deep_merge(base: Mapping[str, Any], override: Mapping[str, Any]) -> Dict[str, Any]:
    """Recursively merge ``override`` on top of ``base``.

    Nested mappings merge key-by-key; every other type (including lists) is
    replaced wholesale, which is what you want for e.g. ``client_files``.
    """
    merged: Dict[str, Any] = copy.deepcopy(dict(base))
    for key, value in override.items():
        if key in merged and isinstance(merged[key], Mapping) and isinstance(value, Mapping):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def _expand_env(value: Any) -> Any:
    """Expand ``${VAR}`` / ``${VAR:-default}`` inside string leaves."""
    if isinstance(value, str):

        def _sub(match: "re.Match[str]") -> str:
            var_name, default = match.group(1), match.group(2)
            return os.environ.get(var_name, default if default is not None else "")

        return _ENV_PATTERN.sub(_sub, value)
    if isinstance(value, dict):
        return {k: _expand_env(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_expand_env(v) for v in value]
    return value


def _coerce_types(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Force known numeric/boolean keys into their intended Python types."""
    for key in _FLOAT_KEYS & cfg.keys():
        value = cfg[key]
        if value is None or isinstance(value, bool):
            continue
        try:
            cfg[key] = float(value)
        except (TypeError, ValueError):
            raise ValueError(f"Config key {key!r} must be a number, got {value!r}") from None

    for key in _INT_KEYS & cfg.keys():
        value = cfg[key]
        if value is None or isinstance(value, bool):
            continue
        try:
            cfg[key] = int(value)
        except (TypeError, ValueError):
            raise ValueError(f"Config key {key!r} must be an integer, got {value!r}") from None

    for key in _BOOL_KEYS & cfg.keys():
        value = cfg[key]
        if isinstance(value, bool) or value is None:
            continue
        if isinstance(value, str):
            lowered = value.strip().lower()
            if lowered in {"true", "yes", "y", "1", "on"}:
                cfg[key] = True
            elif lowered in {"false", "no", "n", "0", "off", ""}:
                cfg[key] = False
            else:
                raise ValueError(f"Config key {key!r} must be a boolean, got {value!r}")
        else:
            cfg[key] = bool(value)
    return cfg


# =============================================================================
# Loading
# =============================================================================
def _read_yaml(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with open(path, "r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ValueError(f"Config {path} must contain a YAML mapping, got {type(data).__name__}")
    return data


def _resolve_chain(path: Path, seen: Optional[List[Path]] = None) -> Dict[str, Any]:
    """Load ``path`` and recursively merge whatever it extends."""
    seen = seen or []
    resolved = path.resolve()
    if resolved in seen:
        chain = " -> ".join(p.name for p in seen + [resolved])
        raise ValueError(f"Circular config inheritance detected: {chain}")
    seen = seen + [resolved]

    raw = _read_yaml(resolved)

    parent_ref: Optional[str] = None
    for key in _EXTENDS_KEYS:
        if raw.get(key):
            parent_ref = str(raw.pop(key))
            break
    # Drop any remaining inheritance aliases so they never reach the runtime dict.
    for key in _EXTENDS_KEYS:
        raw.pop(key, None)

    if parent_ref is None:
        return raw

    parent_path = Path(parent_ref)
    if not parent_path.is_absolute():
        candidates = [
            resolved.parent / parent_path,
            PROJECT_ROOT / parent_path,
            PROJECT_ROOT / "configs" / parent_path,
        ]
        for candidate in candidates:
            if candidate.exists():
                parent_path = candidate
                break
        else:
            raise FileNotFoundError(
                f"{resolved.name} extends {parent_ref!r}, which was not found in any of: "
                + ", ".join(str(c) for c in candidates)
            )

    LOGGER.debug("Config %s extends %s", resolved.name, parent_path.name)
    parent_cfg = _resolve_chain(parent_path, seen)
    return deep_merge(parent_cfg, raw)


def load_config(config_path: Union[str, os.PathLike], overrides: Optional[Mapping[str, Any]] = None) -> Config:
    """Load a FedChain experiment config.

    Parameters
    ----------
    config_path:
        Path to the experiment YAML (absolute, cwd-relative, or repo-relative).
    overrides:
        Optional mapping applied last (used for CLI flags). ``None`` values are
        ignored so unset argparse flags never clobber file values.
    """
    path = Path(config_path)
    if not path.exists():
        for candidate in (PROJECT_ROOT / path, PROJECT_ROOT / "configs" / path.name):
            if candidate.exists():
                path = candidate
                break
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    merged = _resolve_chain(path)
    merged = _expand_env(merged)

    if overrides:
        clean_overrides = {k: v for k, v in overrides.items() if v is not None}
        if clean_overrides:
            LOGGER.info("Applying %d CLI override(s): %s", len(clean_overrides), clean_overrides)
            merged = deep_merge(merged, clean_overrides)

    merged = _coerce_types(merged)

    merged.setdefault("exp_name", path.stem)
    merged.setdefault("config_path", str(path.resolve()))
    merged.setdefault("project_root", str(PROJECT_ROOT))

    cfg = Config(merged)
    validate_config(cfg)
    return cfg


def resolve_path(value: Union[str, os.PathLike], root: Optional[Union[str, os.PathLike]] = None) -> Path:
    """Resolve a possibly-relative config path against the repository root.

    Relative paths in the YAML (``data/client1.jsonl``) are interpreted
    relative to the repo root rather than the current working directory, so
    ``python main.py`` behaves identically no matter where it is launched from.
    """
    path = Path(value)
    if path.is_absolute():
        return path
    base = Path(root) if root is not None else PROJECT_ROOT
    return (base / path).resolve()


# =============================================================================
# Validation
# =============================================================================
def validate_config(cfg: Mapping[str, Any]) -> None:
    """Fail fast on structurally invalid experiment definitions."""
    errors: List[str] = []

    required = ("model_name", "seed", "num_rounds", "learning_rate", "lora_r", "lora_alpha")
    for key in required:
        if key not in cfg:
            errors.append(f"missing required key: {key!r}")

    if cfg.get("num_rounds", 1) < 1:
        errors.append("num_rounds must be >= 1")
    if cfg.get("batch_size", 1) < 1:
        errors.append("batch_size must be >= 1")
    if cfg.get("grad_accum_steps", 1) < 1:
        errors.append("grad_accum_steps must be >= 1")
    if cfg.get("lora_r", 1) < 1:
        errors.append("lora_r must be >= 1")

    targets = cfg.get("lora_target_modules")
    if targets is not None and (not isinstance(targets, Sequence) or isinstance(targets, str) or not targets):
        errors.append("lora_target_modules must be a non-empty list of module names")

    if cfg.get("enable_fl"):
        client_files = cfg.get("client_files")
        if not client_files:
            errors.append("enable_fl is true but client_files is empty")
        elif not isinstance(client_files, list):
            errors.append("client_files must be a list of dataset paths")
        else:
            declared = cfg.get("num_clients")
            if declared is not None and int(declared) != len(client_files):
                errors.append(
                    f"num_clients ({declared}) does not match len(client_files) ({len(client_files)})"
                )
    else:
        if not cfg.get("data_path"):
            errors.append("enable_fl is false but data_path is not set")

    if cfg.get("enable_ipfs") and not cfg.get("enable_blockchain"):
        LOGGER.warning(
            "enable_ipfs is true while enable_blockchain is false: CIDs will be produced "
            "but never anchored on-chain. This is a valid ablation, but it is not the "
            "FedChain configuration described in the paper."
        )

    if errors:
        raise ValueError(
            "Invalid FedChain configuration:\n  - " + "\n  - ".join(errors)
        )
