"""
FedChain :: Crash-safe checkpointing
====================================

A full benchmark sweep runs for 20-30 hours on a T600. A power cut, an OOM kill
or a dropped SSH session must not cost that time. This module provides the
durable state that makes every stage resumable.

Four levels of granularity cooperate:

======  ==============================  ==================================
Level   Mechanism                       Worst-case loss on a crash
======  ==============================  ==================================
1       HF Trainer step checkpoints     ``save_steps`` optimizer steps
        (``trainer/sft.py``)            (default 25, a few minutes)
2       Completed-adapter manifest      only the last unsaved optimizer steps
        (``trainer/sft.py``)
3       Per-client / per-round records  the client currently training
        (``CheckpointManager``)
4       Per-experiment metrics report   the experiment currently running
        (``main.py`` / ``run_all.sh``)
======  ==============================  ==================================

Durability
----------
``save()`` writes to a temporary file in the same directory, ``flush()`` +
``os.fsync()`` it, then ``os.replace()`` onto the target. ``os.replace`` is
atomic on POSIX and on Windows, so a crash mid-write leaves either the previous
complete checkpoint or the new complete one - never a truncated file.

Correctness
-----------
A checkpoint is only reused when its **configuration fingerprint** matches the
current run. Change the learning rate, the LoRA rank, the round count or the
client shards and the fingerprint changes, the stale checkpoint is archived
rather than silently reused, and the experiment restarts cleanly. Resuming
across a hyperparameter change would otherwise produce a result that looks
valid but corresponds to no single configuration - the worst possible failure
mode for a paper.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import shutil
import time
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Union

LOGGER = logging.getLogger(__name__)

PathLike = Union[str, os.PathLike]

#: Bump when the on-disk state layout changes incompatibly.
CHECKPOINT_VERSION = 3

#: Config keys that make a resumed run scientifically different from a fresh
#: one. Anything here is folded into the fingerprint.
FINGERPRINT_KEYS = (
    "model_name",
    "trust_remote_code",
    "attn_implementation",
    "seed",
    "num_rounds",
    "local_epochs",
    "learning_rate",
    "weight_decay",
    "warmup_ratio",
    "lr_scheduler_type",
    "max_grad_norm",
    "optim",
    "max_steps",
    "max_seq_length",
    "batch_size",
    "grad_accum_steps",
    "lora_r",
    "lora_alpha",
    "lora_dropout",
    "lora_target_modules",
    "lora_bias",
    "max_train_samples",
    "client_files",
    "data_path",
    "num_clients",
    "enable_fl",
    "enable_aggregation",
    "fedavg_weighted",
    "enable_blockchain",
    "enable_ipfs",
    "dry_run",
    "dry_run_layers",
    "dry_run_hidden",
    "use_chat_template",
    "deterministic",
    "gradient_checkpointing",
    "device",
    "dataloader_num_workers",
    "load_in_4bit",
    "bnb_4bit_quant_type",
    "bnb_4bit_use_double_quant",
    "bnb_4bit_compute_dtype",
    "eval_data_path",
    "eval_dataset_name",
    "eval_num_samples",
    "eval_batch_size",
    "eval_max_seq_length",
    "eval_every_round",
    "eval_final",
    "eval_loss_on_completion_only",
    "enable_generation_metrics",
    "gen_num_samples",
    "gen_max_new_tokens",
    "gen_max_prompt_tokens",
    "log_global_model",
    "ipfs_roundtrip_aggregation",
    "verify_hash_on_download",
)

DATA_FINGERPRINT_KEYS = frozenset({"client_files", "data_path", "eval_data_path"})


def _fingerprint_file(value: Any) -> Any:
    """Represent configured data paths by path plus content digest when present."""
    if isinstance(value, (list, tuple)):
        return [_fingerprint_file(item) for item in value]
    if not isinstance(value, (str, os.PathLike)):
        return value
    path = Path(value)
    if not path.is_absolute():
        path = Path(__file__).resolve().parent.parent / path
    path = path.resolve()
    result: Dict[str, Any] = {"path": str(path)}
    if path.is_file():
        digest = hashlib.sha256()
        with open(path, "rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        result["sha256"] = digest.hexdigest()
    else:
        result["missing"] = True
    return result


class CheckpointWriteError(RuntimeError):
    """Raised when durable progress cannot be written to disk."""


def compute_fingerprint(config: Mapping[str, Any]) -> str:
    """Stable SHA-256 over the run-defining subset of the configuration."""
    relevant = {}
    for key in FINGERPRINT_KEYS:
        if key in config:
            value = config[key]
            if key in DATA_FINGERPRINT_KEYS:
                relevant[key] = _fingerprint_file(value)
            else:
                relevant[key] = list(value) if isinstance(value, (list, tuple)) else value
    payload = json.dumps(relevant, sort_keys=True, default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _empty_state(exp_name: str, fingerprint: str, paradigm: str) -> Dict[str, Any]:
    now = int(time.time())
    return {
        "version": CHECKPOINT_VERSION,
        "exp_name": exp_name,
        "fingerprint": fingerprint,
        "paradigm": paradigm,
        "status": "in_progress",
        "created_at": now,
        "updated_at": now,
        "sessions": 0,
        "rounds": [],                 # completed round metric dicts
        "partial_round": None,        # {"round": int, "clients": {id: record}}
        "totals": {
            "training_time_sec": 0.0,
            "aggregation_time_sec": 0.0,
            "communication_bytes": 0,
            "wall_clock_sec": 0.0,
        },
        "global_adapter_path": None,
        "global_adapter_hash": None,
        "blockchain_receipts": [],
        "ipfs_transfers": [],
        "final_evaluation": None,
    }


class CheckpointManager:
    """Durable, fingerprint-guarded run state for one experiment.

    Parameters
    ----------
    checkpoint_path:
        JSON file backing the state, e.g. ``outputs/exp4_fedchain/checkpoint.json``.
    exp_name / config:
        Used for the fingerprint and for human-readable logging.
    enabled:
        ``False`` turns every method into a no-op, so callers need no branches.
    paradigm:
        ``"federated"`` or ``"centralized"``; recorded for diagnostics.
    """

    def __init__(
        self,
        checkpoint_path: PathLike,
        exp_name: str,
        config: Mapping[str, Any],
        enabled: bool = True,
        paradigm: str = "federated",
    ) -> None:
        self.path = Path(checkpoint_path)
        self.exp_name = exp_name
        self.enabled = bool(enabled)
        self.paradigm = paradigm
        self.fingerprint = compute_fingerprint(config)
        self.state: Dict[str, Any] = _empty_state(exp_name, self.fingerprint, paradigm)
        self.resumed = False
        self.stale_checkpoint_archived: Optional[Path] = None
        self._load_attempted = False

    # =========================================================================
    # Load / save
    # =========================================================================
    def load(self) -> bool:
        """Load prior state. Returns True when a usable checkpoint was resumed.

        A checkpoint is rejected (and archived) when it is unreadable, written
        by an incompatible version, or produced by a different configuration.
        """
        if self._load_attempted:
            return self.resumed
        self._load_attempted = True

        if not self.enabled:
            LOGGER.info("Checkpointing disabled; starting from scratch.")
            return False
        if not self.path.exists():
            if self._load_backup():
                return True
            LOGGER.info("No checkpoint at %s; starting a fresh run.", self.path)
            self._begin_fresh_session()
            return False

        try:
            with open(self.path, "r", encoding="utf-8") as handle:
                state = json.load(handle)
        except Exception as exc:
            LOGGER.warning("Checkpoint %s is unreadable (%s).", self.path, exc)
            if self._load_backup():
                return True
            LOGGER.warning("No usable backup exists; starting fresh.")
            self._archive("corrupt")
            self._begin_fresh_session()
            return False

        if state.get("version") != CHECKPOINT_VERSION:
            LOGGER.warning(
                "Checkpoint version %s != %s; starting fresh.",
                state.get("version"),
                CHECKPOINT_VERSION,
            )
            self._archive("version-mismatch")
            self._begin_fresh_session()
            return False

        if state.get("fingerprint") != self.fingerprint:
            LOGGER.warning(
                "=" * 74
                + "\nCONFIGURATION CHANGED since the last run of '%s'.\n"
                "The existing checkpoint was produced by a different setup and will NOT\n"
                "be reused - resuming across a hyperparameter change would silently mix\n"
                "two configurations into one result. Restarting this experiment.\n"
                + "=" * 74,
                self.exp_name,
            )
            self._archive("config-changed")
            self._begin_fresh_session()
            return False

        if state.get("status") == "completed":
            LOGGER.info("Checkpoint for '%s' is already marked completed.", self.exp_name)

        self.state = state
        self.state["sessions"] = int(state.get("sessions", 0)) + 1
        self.state["status"] = "in_progress"
        self.resumed = True
        self.save()

        LOGGER.info(
            "Resumed checkpoint %s | %d completed round(s), partial round=%s, session #%d",
            self.path.name,
            len(state.get("rounds", [])),
            (state.get("partial_round") or {}).get("round", "-"),
            self.state["sessions"],
        )
        return True

    @property
    def backup_path(self) -> Path:
        return self.path.with_name(self.path.name + ".bak")

    def _begin_fresh_session(self) -> None:
        self.state = _empty_state(self.exp_name, self.fingerprint, self.paradigm)
        self.state["sessions"] = 1
        self.save()

    def _load_backup(self) -> bool:
        """Recover the previous known-good generation if the primary is lost."""
        backup = self.backup_path
        if not backup.exists():
            return False
        try:
            with open(backup, "r", encoding="utf-8") as handle:
                state = json.load(handle)
            if state.get("version") != CHECKPOINT_VERSION:
                return False
            if state.get("fingerprint") != self.fingerprint:
                return False
        except Exception as exc:
            LOGGER.warning("Checkpoint backup %s is unreadable: %s", backup, exc)
            return False

        self.state = state
        self.state["sessions"] = int(state.get("sessions", 0)) + 1
        self.state["status"] = "in_progress"
        self.resumed = True
        LOGGER.warning("Recovered progress from checkpoint backup %s.", backup)
        # Do not let save() rotate an unreadable primary over the good backup.
        if self.path.exists():
            self._archive("corrupt-primary")
            # Recovery succeeded with the same fingerprint, so local step and
            # completion manifests remain eligible for reuse.
            self.stale_checkpoint_archived = None
        self.save()
        return True

    def restart(self, reason: str = "manual-restart") -> None:
        """Archive existing recovery state and begin a clean checkpointed run."""
        if not self.enabled:
            self._load_attempted = True
            return
        if self.path.exists():
            self._archive(reason)
        if self.backup_path.exists():
            try:
                target = self.backup_path.with_name(
                    f"{self.path.stem}.stale-{reason}-backup-{int(time.time())}.json"
                )
                os.replace(self.backup_path, target)
            except Exception as exc:
                raise CheckpointWriteError(
                    f"Could not archive checkpoint backup {self.backup_path}: {exc}"
                ) from exc
        # Also acts as a clean-restart marker when no checkpoint file existed
        # but completed adapter manifests are still present under output_root.
        if self.stale_checkpoint_archived is None:
            self.stale_checkpoint_archived = self.path.with_name(
                f"{self.path.stem}.stale-{reason}.json"
            )
        self.resumed = False
        self._load_attempted = True
        self._begin_fresh_session()

    def save(self, raise_on_error: bool = True) -> bool:
        """Atomically persist state and retain the previous known-good generation.

        Checkpoint failures are fatal by default. Continuing an expensive run
        after its failsafe stopped working would give a false sense of safety.
        Failure-reporting paths can use ``raise_on_error=False`` so they do not
        mask the original exception.
        """
        if not self.enabled:
            return True
        self.state["updated_at"] = int(time.time())
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = self.path.with_name(self.path.name + ".tmp")
        backup_tmp = self.backup_path.with_name(self.backup_path.name + ".tmp")
        try:
            with open(tmp_path, "w", encoding="utf-8") as handle:
                json.dump(self.state, handle, indent=2, default=str)
                handle.flush()
                os.fsync(handle.fileno())

            # Preserve the previous complete generation before replacing the
            # primary. The backup itself is staged and atomically replaced too.
            if self.path.exists():
                with open(self.path, "rb") as source, open(backup_tmp, "wb") as target:
                    shutil.copyfileobj(source, target)
                    target.flush()
                    os.fsync(target.fileno())
                os.replace(backup_tmp, self.backup_path)
            os.replace(tmp_path, self.path)
            return True
        except Exception as exc:
            LOGGER.error("Could not write checkpoint %s: %s", self.path, exc)
            for scratch in (tmp_path, backup_tmp):
                try:
                    scratch.unlink(missing_ok=True)
                except Exception:
                    pass
            if raise_on_error:
                raise CheckpointWriteError(
                    f"Could not persist crash-recovery checkpoint {self.path}: {exc}"
                ) from exc
            return False

    def _archive(self, reason: str) -> None:
        """Move an unusable checkpoint aside instead of deleting it."""
        try:
            target = self.path.with_name(f"{self.path.stem}.stale-{reason}-{int(time.time())}.json")
            os.replace(self.path, target)
            self.stale_checkpoint_archived = target
            LOGGER.info("Previous checkpoint archived as %s", target.name)
        except Exception as exc:
            LOGGER.debug("Could not archive stale checkpoint: %s", exc)

    def clear(self) -> None:
        """Delete the checkpoint (used after a fully successful run)."""
        try:
            self.path.unlink(missing_ok=True)
        except Exception as exc:
            LOGGER.debug("Could not remove checkpoint: %s", exc)

    # =========================================================================
    # Accessors used by the orchestrator
    # =========================================================================
    @property
    def completed_rounds(self) -> List[Dict[str, Any]]:
        return list(self.state.get("rounds", []))

    @property
    def num_completed_rounds(self) -> int:
        return len(self.state.get("rounds", []))

    @property
    def totals(self) -> Dict[str, Any]:
        return self.state.setdefault(
            "totals",
            {
                "training_time_sec": 0.0,
                "aggregation_time_sec": 0.0,
                "communication_bytes": 0,
                "wall_clock_sec": 0.0,
            },
        )

    @property
    def global_adapter_path(self) -> Optional[str]:
        return self.state.get("global_adapter_path")

    @property
    def global_adapter_hash(self) -> Optional[str]:
        return self.state.get("global_adapter_hash")

    @property
    def final_evaluation(self) -> Optional[Dict[str, Any]]:
        return self.state.get("final_evaluation")

    @property
    def is_completed(self) -> bool:
        return self.state.get("status") == "completed"

    def partial_clients(self, round_index: int) -> Dict[str, Any]:
        """Client records already finished for ``round_index`` (may be empty)."""
        partial = self.state.get("partial_round") or {}
        if partial.get("round") == round_index:
            return dict(partial.get("clients", {}))
        return {}

    # =========================================================================
    # Mutators (each persists immediately)
    # =========================================================================
    def record_client(
        self,
        round_index: int,
        client_id: str,
        record: Dict[str, Any],
        blockchain_receipts: Optional[List[Dict[str, Any]]] = None,
        ipfs_transfers: Optional[List[Dict[str, Any]]] = None,
    ) -> None:
        """Persist one finished client so a crash never repeats its training.

        The chain and IPFS ledgers are snapshotted here too, not only at round
        completion. A client that was anchored on-chain and pinned to IPFS
        before the crash is skipped on resume, so if its receipt were not
        captured now it would vanish from the totals - under-reporting gas,
        transaction count and transferred volume in the final report.
        """
        partial = self.state.get("partial_round") or {}
        if partial.get("round") != round_index:
            partial = {"round": round_index, "clients": {}}
        partial["clients"][client_id] = record
        self.state["partial_round"] = partial
        if blockchain_receipts is not None:
            self.state["blockchain_receipts"] = blockchain_receipts
        if ipfs_transfers is not None:
            self.state["ipfs_transfers"] = ipfs_transfers
        self.save()

    def complete_round(
        self,
        round_metrics: Dict[str, Any],
        global_adapter_path: Optional[str],
        totals: Mapping[str, Any],
        global_adapter_hash: Optional[str] = None,
        blockchain_receipts: Optional[List[Dict[str, Any]]] = None,
        ipfs_transfers: Optional[List[Dict[str, Any]]] = None,
    ) -> None:
        """Commit a finished round and clear the partial-round scratch state."""
        self.state.setdefault("rounds", []).append(round_metrics)
        self.state["partial_round"] = None
        self.state["global_adapter_path"] = global_adapter_path
        self.state["global_adapter_hash"] = global_adapter_hash
        self.state["totals"] = dict(totals)
        if blockchain_receipts is not None:
            self.state["blockchain_receipts"] = blockchain_receipts
        if ipfs_transfers is not None:
            self.state["ipfs_transfers"] = ipfs_transfers
        self.save()

    def record_final_evaluation(self, evaluation: Optional[Dict[str, Any]]) -> None:
        """Persist the final score so a crash afterwards never re-runs it."""
        self.state["final_evaluation"] = evaluation
        self.save()

    def mark_completed(self) -> None:
        self.state["status"] = "completed"
        self.save()

    def mark_failed(self, error: str) -> None:
        self.state["status"] = "failed"
        self.state["last_error"] = str(error)[:2000]
        self.save(raise_on_error=False)

    # =========================================================================
    # Diagnostics
    # =========================================================================
    def describe(self) -> Dict[str, Any]:
        """Summary embedded in the metrics report."""
        return {
            "enabled": self.enabled,
            "path": str(self.path),
            "resumed": self.resumed,
            "sessions": int(self.state.get("sessions", 0)),
            "completed_rounds": self.num_completed_rounds,
            "fingerprint": self.fingerprint[:16],
            "stale_checkpoint_archived": str(self.stale_checkpoint_archived)
            if self.stale_checkpoint_archived
            else None,
        }


# =============================================================================
# Adapter validation
# =============================================================================
def adapter_is_complete(adapter_path: Optional[PathLike]) -> bool:
    """True when ``adapter_path`` holds a fully written PEFT checkpoint."""
    if not adapter_path:
        return False
    path = Path(adapter_path)
    if not path.is_dir():
        return False
    if not (path / "adapter_config.json").exists():
        return False
    weights = (path / "adapter_model.safetensors", path / "adapter_model.bin")
    return any(w.exists() and w.stat().st_size > 0 for w in weights)


def adapter_matches_hash(adapter_path: PathLike, expected_hash: Optional[str]) -> bool:
    """Re-hash an adapter and compare against the digest recorded before the crash.

    This is the check that distinguishes "the adapter finished writing" from
    "the process died halfway through ``save_pretrained``". Without it a
    truncated safetensors file would be silently averaged into the global model.
    """
    if not expected_hash:
        return True  # nothing to compare against; the existence check stands
    try:
        from utils.common import sha256_path

        return sha256_path(adapter_path).lower() == str(expected_hash).lower()
    except Exception as exc:
        LOGGER.warning("Could not verify adapter %s: %s", adapter_path, exc)
        return False


def reusable_adapter(record: Mapping[str, Any]) -> bool:
    """Decide whether a checkpointed client record can be trusted on resume."""
    path = record.get("adapter_path")
    if not adapter_is_complete(path):
        LOGGER.info("Adapter %s is missing or incomplete; it will be retrained.", path)
        return False
    if not adapter_matches_hash(path, record.get("model_hash")):
        LOGGER.warning(
            "Adapter %s does not match its recorded SHA-256 (likely a crash during save); "
            "it will be retrained.",
            path,
        )
        return False
    return True
