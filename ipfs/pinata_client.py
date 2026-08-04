"""
FedChain :: IPFS storage layer
==============================

``IPFSManager`` moves LoRA adapters between federated participants over
content-addressed storage, with three interchangeable backends:

============  =========================================================
``pinata``    Pinata pinning service (REST API, key pair or JWT)
``local``    A Kubo/go-ipfs daemon's HTTP API (default 127.0.0.1:5001)
``mock``      Local content-addressed store, used when neither is reachable
============  =========================================================

The backend is auto-detected once at construction time and reported in the
metrics so a run is never silently mislabelled. The mock backend is a genuine
content-addressed store, not a stub: it computes a **real CIDv0** (base58btc of
the sha2-256 multihash) so identical adapters deduplicate exactly as they would
on a live IPFS network, and offline runs still exercise the full
upload -> CID -> anchor -> download -> verify code path.

Adapters are PEFT *directories*; ``upload_adapter`` therefore packs a directory
into a deterministic ``.tar.gz`` before transfer and ``download_adapter``
unpacks it again, so the measured communication volume is the volume actually
moved over the wire.
"""

from __future__ import annotations

import hashlib
import io
import json
import logging
import os
import shutil
import tarfile
import tempfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from utils.common import bytes_to_mb, path_size_bytes, sha256_bytes, write_json

LOGGER = logging.getLogger(__name__)

PathLike = Union[str, os.PathLike]

try:  # pragma: no cover
    import requests

    REQUESTS_AVAILABLE = True
except Exception:
    requests = None  # type: ignore[assignment]
    REQUESTS_AVAILABLE = False


PINATA_API_ROOT = "https://api.pinata.cloud"
PINATA_AUTH_ENDPOINT = f"{PINATA_API_ROOT}/data/testAuthentication"
PINATA_PIN_FILE_ENDPOINT = f"{PINATA_API_ROOT}/pinning/pinFileToIPFS"

_B58_ALPHABET = "123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz"


# =============================================================================
# CID helpers
# =============================================================================
def base58_encode(payload: bytes) -> str:
    """Bitcoin-style base58 encoding (no checksum), used for CIDv0."""
    leading_zeros = len(payload) - len(payload.lstrip(b"\x00"))
    number = int.from_bytes(payload, "big")
    encoded = ""
    while number > 0:
        number, remainder = divmod(number, 58)
        encoded = _B58_ALPHABET[remainder] + encoded
    return "1" * leading_zeros + encoded


def compute_cidv0(payload: bytes) -> str:
    """Deterministic, well-formed CIDv0 for a byte payload.

    A CIDv0 is ``base58btc(<sha2-256 multihash>)`` where the multihash is
    ``0x12 0x20 || sha256(payload)``. The result is the canonical 46-character
    identifier beginning with ``Qm``.

    This is *not* byte-identical to what a real IPFS node would return for the
    same file, because Kubo chunks files and builds a UnixFS DAG whose root hash
    covers protobuf-wrapped links rather than the raw bytes. It is a valid,
    stable, collision-resistant identifier with the right shape and the right
    dedup semantics, which is what the offline benchmark needs.
    """
    multihash = b"\x12\x20" + hashlib.sha256(payload).digest()
    return base58_encode(multihash)


# =============================================================================
# Records
# =============================================================================
@dataclass
class TransferRecord:
    """One IPFS upload or download plus its systems metrics."""

    operation: str          # "upload" | "download"
    cid: str
    backend: str
    latency_sec: float
    size_bytes: int
    size_mb: float
    source: Optional[str] = None
    target: Optional[str] = None
    status: str = "success"
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# =============================================================================
# Manager
# =============================================================================
class IPFSManager:
    """Upload and retrieve adapter artefacts over IPFS (or a local mock store).

    Parameters
    ----------
    pinata_api_key / pinata_secret_key / pinata_jwt:
        Pinata credentials. Empty values fall back to the ``PINATA_API_KEY``,
        ``PINATA_SECRET_KEY`` and ``PINATA_JWT`` environment variables.
    local_api_url:
        Kubo HTTP API root, probed when Pinata credentials are absent.
    gateway_url:
        HTTP gateway used to resolve CIDs uploaded through Pinata.
    mock_dir:
        Directory backing the mock store. Defaults to
        ``<system temp>/ipfs_mock`` (``/tmp/ipfs_mock`` on Linux).
    force_mock:
        Skip probing entirely and use the mock store.
    """

    def __init__(
        self,
        pinata_api_key: str = "",
        pinata_secret_key: str = "",
        pinata_jwt: str = "",
        local_api_url: str = "http://127.0.0.1:5001",
        gateway_url: str = "https://gateway.pinata.cloud/ipfs",
        mock_dir: str = "",
        timeout: int = 120,
        probe_timeout: int = 5,
        force_mock: bool = False,
    ) -> None:
        self.pinata_api_key = (pinata_api_key or os.environ.get("PINATA_API_KEY", "")).strip()
        self.pinata_secret_key = (pinata_secret_key or os.environ.get("PINATA_SECRET_KEY", "")).strip()
        self.pinata_jwt = (pinata_jwt or os.environ.get("PINATA_JWT", "")).strip()
        self.local_api_url = (local_api_url or "http://127.0.0.1:5001").rstrip("/")
        self.gateway_url = (gateway_url or "https://gateway.pinata.cloud/ipfs").rstrip("/")
        self.timeout = int(timeout)
        self.probe_timeout = int(probe_timeout)

        self.mock_dir = Path(mock_dir) if mock_dir else Path(tempfile.gettempdir()) / "ipfs_mock"
        self.transfers: List[TransferRecord] = []
        self.backend: str = "mock"
        self.backend_note: Optional[str] = None
        # Staging area for tar archives of adapter directories.
        self._staging = Path(tempfile.gettempdir()) / "fedchain_ipfs_staging"

        if force_mock:
            self._enter_mock_mode("forced by caller")
        else:
            self._detect_backend()

    # -- properties ----------------------------------------------------------
    @property
    def is_mock(self) -> bool:
        return self.backend == "mock"

    # -- backend detection ---------------------------------------------------
    def _enter_mock_mode(self, reason: str) -> None:
        self.backend = "mock"
        self.backend_note = reason
        self.mock_dir.mkdir(parents=True, exist_ok=True)
        LOGGER.warning(
            "IPFSManager running in MOCK mode (%s). Artefacts are content-addressed "
            "into %s; CIDs are real CIDv0 digests but are not published to any network.",
            reason,
            self.mock_dir,
        )

    def _detect_backend(self) -> None:
        if not REQUESTS_AVAILABLE:
            self._enter_mock_mode("the `requests` package is not installed")
            return

        if self._has_pinata_credentials() and self._probe_pinata():
            self.backend = "pinata"
            self.backend_note = "authenticated against the Pinata API"
            LOGGER.info("IPFSManager backend: Pinata (%s)", PINATA_API_ROOT)
            return

        if self._probe_local_node():
            self.backend = "local"
            self.backend_note = f"Kubo HTTP API at {self.local_api_url}"
            LOGGER.info("IPFSManager backend: local IPFS daemon (%s)", self.local_api_url)
            return

        if self._has_pinata_credentials():
            self._enter_mock_mode("Pinata credentials were rejected and no local daemon answered")
        else:
            self._enter_mock_mode(
                f"no Pinata credentials and no IPFS daemon at {self.local_api_url}"
            )

    def _has_pinata_credentials(self) -> bool:
        return bool(self.pinata_jwt or (self.pinata_api_key and self.pinata_secret_key))

    def _pinata_headers(self) -> Dict[str, str]:
        if self.pinata_jwt:
            return {"Authorization": f"Bearer {self.pinata_jwt}"}
        return {
            "pinata_api_key": self.pinata_api_key,
            "pinata_secret_api_key": self.pinata_secret_key,
        }

    def _probe_pinata(self) -> bool:
        try:
            response = requests.get(
                PINATA_AUTH_ENDPOINT, headers=self._pinata_headers(), timeout=self.probe_timeout
            )
            if response.status_code == 200:
                return True
            LOGGER.warning(
                "Pinata authentication failed (HTTP %s): %s",
                response.status_code,
                response.text[:200],
            )
        except Exception as exc:
            LOGGER.warning("Could not reach the Pinata API: %s", exc)
        return False

    def _probe_local_node(self) -> bool:
        try:
            response = requests.post(
                f"{self.local_api_url}/api/v0/version", timeout=self.probe_timeout
            )
            if response.status_code == 200:
                LOGGER.debug("Local IPFS version: %s", response.text[:200])
                return True
        except Exception as exc:
            LOGGER.debug("No local IPFS daemon at %s (%s)", self.local_api_url, exc)
        return False

    # -- packing -------------------------------------------------------------
    def _pack(self, source: PathLike) -> Tuple[Path, bool]:
        """Return a single file to transfer, packing directories into tar.gz.

        The archive is built deterministically (sorted members, zeroed mtime /
        uid / gid) so that identical adapter contents always yield identical
        bytes, and therefore identical CIDs.
        """
        path = Path(source)
        if not path.exists():
            raise FileNotFoundError(f"Cannot upload missing path: {path}")
        if path.is_file():
            return path, False

        self._staging.mkdir(parents=True, exist_ok=True)
        archive_path = self._staging / f"{path.name}.tar.gz"

        members = sorted(
            (p for p in path.rglob("*") if p.is_file()),
            key=lambda p: p.relative_to(path).as_posix(),
        )
        # mtime=0 in GzipFile keeps the gzip header byte-identical across runs.
        with open(archive_path, "wb") as raw:
            import gzip

            with gzip.GzipFile(fileobj=raw, mode="wb", mtime=0) as gz:
                with tarfile.open(fileobj=gz, mode="w") as tar:  # type: ignore[arg-type]
                    for member_path in members:
                        info = tar.gettarinfo(
                            str(member_path), arcname=member_path.relative_to(path).as_posix()
                        )
                        info.mtime = 0
                        info.uid = info.gid = 0
                        info.uname = info.gname = ""
                        info.mode = 0o644
                        with open(member_path, "rb") as handle:
                            tar.addfile(info, handle)
        return archive_path, True

    @staticmethod
    def _unpack(archive_bytes: bytes, target_path: Path) -> None:
        """Extract a tar.gz payload into ``target_path``, rejecting path escapes."""
        target_path.mkdir(parents=True, exist_ok=True)
        resolved_root = target_path.resolve()
        with tarfile.open(fileobj=io.BytesIO(archive_bytes), mode="r:gz") as tar:
            safe_members = []
            for member in tar.getmembers():
                destination = (resolved_root / member.name).resolve()
                if not str(destination).startswith(str(resolved_root)):
                    raise ValueError(f"Refusing to extract member outside target: {member.name}")
                if member.issym() or member.islnk():
                    raise ValueError(f"Refusing to extract link member: {member.name}")
                safe_members.append(member)
            tar.extractall(path=resolved_root, members=safe_members)

    @staticmethod
    def _looks_like_gzip(payload: bytes) -> bool:
        return len(payload) >= 2 and payload[0] == 0x1F and payload[1] == 0x8B

    # -- upload --------------------------------------------------------------
    def upload_adapter(self, file_path: PathLike) -> Tuple[str, float, float]:
        """Pin an adapter file or directory to IPFS.

        Returns
        -------
        (cid, latency_sec, size_mb)
            ``cid`` is the content identifier, ``latency_sec`` the wall-clock
            transfer time (including packing), and ``size_mb`` the number of
            megabytes actually pushed over the wire.
        """
        overall_start = time.perf_counter()
        source = Path(file_path)

        try:
            payload_path, is_archive = self._pack(source)
            payload = payload_path.read_bytes()
            size_bytes = len(payload)
            display_name = f"{source.name}.tar.gz" if is_archive else source.name

            if self.backend == "pinata":
                cid = self._upload_pinata(payload, display_name)
            elif self.backend == "local":
                cid = self._upload_local(payload, display_name)
            else:
                cid = self._upload_mock(payload, display_name, is_archive)

            latency = time.perf_counter() - overall_start
            size_mb = bytes_to_mb(size_bytes)
            self.transfers.append(
                TransferRecord(
                    operation="upload",
                    cid=cid,
                    backend=self.backend,
                    latency_sec=round(latency, 6),
                    size_bytes=size_bytes,
                    size_mb=size_mb,
                    source=str(source),
                )
            )
            LOGGER.info(
                "[ipfs:%s] uploaded %s -> %s (%.3f MB in %.3fs)",
                self.backend,
                source.name,
                cid,
                size_mb,
                latency,
            )
            return cid, round(latency, 6), size_mb

        except Exception as exc:
            latency = time.perf_counter() - overall_start
            LOGGER.error("IPFS upload of %s failed: %s", source, exc)
            self.transfers.append(
                TransferRecord(
                    operation="upload",
                    cid="",
                    backend=self.backend,
                    latency_sec=round(latency, 6),
                    size_bytes=path_size_bytes(source),
                    size_mb=bytes_to_mb(path_size_bytes(source)),
                    source=str(source),
                    status="failed",
                    error=str(exc),
                )
            )
            raise

    def _upload_pinata(self, payload: bytes, name: str) -> str:
        files = {"file": (name, io.BytesIO(payload), "application/octet-stream")}
        data = {
            "pinataMetadata": json.dumps({"name": name, "keyvalues": {"project": "FedChain"}}),
            "pinataOptions": json.dumps({"cidVersion": 0}),
        }
        response = requests.post(
            PINATA_PIN_FILE_ENDPOINT,
            files=files,
            data=data,
            headers=self._pinata_headers(),
            timeout=self.timeout,
        )
        response.raise_for_status()
        cid = response.json().get("IpfsHash")
        if not cid:
            raise RuntimeError(f"Pinata response contained no IpfsHash: {response.text[:200]}")
        return cid

    def _upload_local(self, payload: bytes, name: str) -> str:
        files = {"file": (name, io.BytesIO(payload), "application/octet-stream")}
        response = requests.post(
            f"{self.local_api_url}/api/v0/add",
            files=files,
            params={"pin": "true", "cid-version": "0"},
            timeout=self.timeout,
        )
        response.raise_for_status()
        # Kubo streams newline-delimited JSON; the final object is the root.
        last_line = [line for line in response.text.strip().splitlines() if line.strip()][-1]
        cid = json.loads(last_line).get("Hash")
        if not cid:
            raise RuntimeError(f"IPFS add returned no Hash: {response.text[:200]}")
        return cid

    def _upload_mock(self, payload: bytes, name: str, is_archive: bool) -> str:
        self.mock_dir.mkdir(parents=True, exist_ok=True)
        cid = compute_cidv0(payload)
        blob_path = self.mock_dir / cid
        blob_path.write_bytes(payload)
        meta = {
            "cid": cid,
            "name": name,
            "size_bytes": len(payload),
            "sha256": sha256_bytes(payload),
            "is_archive": is_archive,
            "pinned_at": int(time.time()),
        }
        (self.mock_dir / f"{cid}.meta.json").write_text(
            json.dumps(meta, indent=2), encoding="utf-8"
        )
        return cid

    # -- download ------------------------------------------------------------
    def download_adapter(self, cid: str, target_path: PathLike) -> Tuple[float, int]:
        """Fetch a CID and materialise it at ``target_path``.

        If the payload is a gzip archive and ``target_path`` has no file
        suffix, it is treated as a directory and unpacked in place - which is
        what the aggregator wants for a PEFT adapter. Otherwise the raw bytes
        are written to ``target_path``.

        Returns
        -------
        (latency_sec, wire_bytes)
            ``wire_bytes`` is the size of the payload that actually crossed the
            network - i.e. post-compression, symmetric with what
            ``upload_adapter`` reports for the same artefact.
        """
        start = time.perf_counter()
        target = Path(target_path)

        try:
            if self.backend == "pinata":
                payload = self._download_gateway(cid)
            elif self.backend == "local":
                payload = self._download_local(cid)
            else:
                payload = self._download_mock(cid)

            treat_as_dir = self._looks_like_gzip(payload) and target.suffix == ""
            if treat_as_dir:
                self._unpack(payload, target)
            else:
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(payload)

            latency = time.perf_counter() - start
            size_bytes = len(payload)
            self.transfers.append(
                TransferRecord(
                    operation="download",
                    cid=cid,
                    backend=self.backend,
                    latency_sec=round(latency, 6),
                    size_bytes=size_bytes,
                    size_mb=bytes_to_mb(size_bytes),
                    target=str(target),
                )
            )
            LOGGER.info(
                "[ipfs:%s] downloaded %s -> %s (%.3f MB in %.3fs)",
                self.backend,
                cid,
                target,
                bytes_to_mb(size_bytes),
                latency,
            )
            return round(latency, 6), size_bytes

        except Exception as exc:
            latency = time.perf_counter() - start
            LOGGER.error("IPFS download of %s failed: %s", cid, exc)
            self.transfers.append(
                TransferRecord(
                    operation="download",
                    cid=cid,
                    backend=self.backend,
                    latency_sec=round(latency, 6),
                    size_bytes=0,
                    size_mb=0.0,
                    target=str(target),
                    status="failed",
                    error=str(exc),
                )
            )
            raise

    def _download_gateway(self, cid: str) -> bytes:
        response = requests.get(f"{self.gateway_url}/{cid}", timeout=self.timeout)
        response.raise_for_status()
        return response.content

    def _download_local(self, cid: str) -> bytes:
        response = requests.post(
            f"{self.local_api_url}/api/v0/cat", params={"arg": cid}, timeout=self.timeout
        )
        response.raise_for_status()
        return response.content

    def _download_mock(self, cid: str) -> bytes:
        blob_path = self.mock_dir / cid
        if not blob_path.exists():
            raise FileNotFoundError(f"CID {cid} is not present in the mock store at {self.mock_dir}")
        return blob_path.read_bytes()

    # -- metrics -------------------------------------------------------------
    def restore_transfers(self, transfers: List[Dict[str, Any]]) -> int:
        """Re-seed transfer records from a checkpoint after a crash.

        Transfers completed before the crash really happened and must keep
        contributing to the reported upload/download latency and volume, even
        though the artefacts they moved are not re-uploaded on resume.
        """
        restored = 0
        known_fields = {f for f in TransferRecord.__dataclass_fields__}  # type: ignore[attr-defined]
        for entry in transfers or []:
            try:
                self.transfers.append(
                    TransferRecord(**{k: v for k, v in entry.items() if k in known_fields})
                )
                restored += 1
            except Exception as exc:
                LOGGER.warning("Skipping unreadable checkpointed transfer: %s", exc)
        if restored:
            LOGGER.info("Restored %d IPFS transfer record(s) from the checkpoint.", restored)
        return restored

    def get_metrics_summary(self) -> Dict[str, Any]:
        """Aggregate transfer metrics across the whole run."""
        uploads = [t for t in self.transfers if t.operation == "upload" and t.status == "success"]
        downloads = [t for t in self.transfers if t.operation == "download" and t.status == "success"]

        def _stats(records: List[TransferRecord], prefix: str) -> Dict[str, Any]:
            latencies = [r.latency_sec for r in records]
            total_bytes = sum(r.size_bytes for r in records)
            return {
                f"num_{prefix}s": len(records),
                f"total_{prefix}_latency_sec": round(sum(latencies), 6),
                f"avg_{prefix}_latency_sec": round(sum(latencies) / len(latencies), 6) if latencies else 0.0,
                f"max_{prefix}_latency_sec": round(max(latencies), 6) if latencies else 0.0,
                f"total_{prefix}_mb": bytes_to_mb(total_bytes),
            }

        summary: Dict[str, Any] = {
            "backend": self.backend,
            "backend_note": self.backend_note,
            "gateway_url": self.gateway_url if self.backend == "pinata" else None,
            "local_api_url": self.local_api_url if self.backend == "local" else None,
            "mock_dir": str(self.mock_dir) if self.is_mock else None,
            "num_failed_transfers": sum(1 for t in self.transfers if t.status != "success"),
        }
        summary.update(_stats(uploads, "upload"))
        summary.update(_stats(downloads, "download"))
        summary["total_transfer_mb"] = round(
            summary["total_upload_mb"] + summary["total_download_mb"], 4
        )
        summary["total_transfer_latency_sec"] = round(
            summary["total_upload_latency_sec"] + summary["total_download_latency_sec"], 6
        )
        return summary

    def export_transfer_log(self, path: PathLike) -> Path:
        payload = {
            "summary": self.get_metrics_summary(),
            "transfers": [t.to_dict() for t in self.transfers],
        }
        out = write_json(path, payload)
        LOGGER.info("IPFS transfer log exported to %s", out)
        return out

    def cleanup_staging(self) -> None:
        """Delete the temporary tar staging area."""
        if self._staging.exists():
            shutil.rmtree(self._staging, ignore_errors=True)
