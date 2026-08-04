"""
FedChain :: Federated orchestrator
==================================

``FederatedOrchestrator`` runs the multi-round federated loop and collects every
systems metric the paper reports.

Per round, for each client:

1. **Local training** - ``LocalTrainer`` fine-tunes a QLoRA adapter on that
   client's shard, warm-started from the previous round's *global* adapter.
2. **Publish** *(Exp 4)* - the adapter directory is packed and pinned to IPFS,
   yielding a CID.
3. **Anchor** *(Exp 3 & 4)* - SHA-256 of the adapter (and the CID when present)
   is written to the ``FedChainAudit`` contract.
4. **Retrieve + verify** *(Exp 4)* - the aggregator pulls each adapter back
   from IPFS by CID and re-hashes it against the on-chain commitment before
   allowing it into the average. A mismatch excludes that client from the round.
5. **Aggregate** - ``FedAvgAggregator`` averages the LoRA factors into the new
   global adapter, which is then anchored/pinned in turn.

Communication accounting
------------------------
``total_communication_mb`` is the volume that actually crosses a participant
boundary, so the four experiments stay comparable:

* Exp 2/3: ``N`` client uploads + ``N`` global broadcasts, sized from the
  adapter directories on disk.
* Exp 4: the byte counts measured by ``IPFSManager`` for every real transfer
  (upload and download), since the adapters are tar-gzipped in flight.
"""

from __future__ import annotations

import logging
import shutil
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from trainer.aggregation import FedAvgAggregator
from trainer.sft import LocalTrainer
from utils.checkpoint import (
    CheckpointManager,
    adapter_is_complete,
    adapter_matches_hash,
    reusable_adapter,
)
from utils.common import (
    bytes_to_mb,
    format_duration,
    free_cuda_memory,
    path_size_bytes,
    sha256_path,
)
from utils.config import resolve_path

LOGGER = logging.getLogger(__name__)


class FederatedOrchestrator:
    """Coordinates federated rounds across simulated clients.

    Parameters
    ----------
    config:
        The merged experiment configuration.
    trainer:
        A ``LocalTrainer``. Created from ``config`` when omitted.
    aggregator:
        A ``FedAvgAggregator``. Created uniform-weighted when omitted.
    blockchain_logger / ipfs_manager:
        Optional; supplied by ``main.py`` when the experiment enables them.
    evaluator:
        Optional ``Evaluator``; when present the global adapter is scored at the
        end of each round so the paper can plot loss-versus-round curves.
    """

    def __init__(
        self,
        config: Any,
        trainer: Optional[LocalTrainer] = None,
        aggregator: Optional[FedAvgAggregator] = None,
        blockchain_logger: Any = None,
        ipfs_manager: Any = None,
        evaluator: Any = None,
        checkpoint: Optional[CheckpointManager] = None,
    ) -> None:
        self.config = config
        self.trainer = trainer or LocalTrainer(config)
        self.aggregator = aggregator or FedAvgAggregator(weighted=False)
        self.blockchain = blockchain_logger
        self.ipfs = ipfs_manager
        self.evaluator = evaluator

        self.enable_blockchain = bool(self._get("enable_blockchain", False)) and self.blockchain is not None
        self.enable_ipfs = bool(self._get("enable_ipfs", False)) and self.ipfs is not None
        # Local-only ablation switch. Default True = ordinary FedAvg.
        self.enable_aggregation = bool(self._get("enable_aggregation", True))
        self.local_adapter_paths: Dict[str, Path] = {}
        self.log_global_model = bool(self._get("log_global_model", True))
        self.verify_on_download = bool(self._get("verify_hash_on_download", True))
        self.ipfs_roundtrip = bool(self._get("ipfs_roundtrip_aggregation", True))

        self.num_rounds = int(self._get("num_rounds", 3))
        self.client_files: List[Path] = [
            resolve_path(p) for p in self._get("client_files", []) or []
        ]
        self.num_clients = len(self.client_files)
        self.client_ids = [f"client_{i + 1}" for i in range(self.num_clients)]

        self.work_dir = resolve_path(self._get("output_root", "outputs")) / str(
            self._get("exp_name", "experiment")
        )
        self.work_dir.mkdir(parents=True, exist_ok=True)

        self.global_adapter_path: Optional[Path] = None
        self.round_metrics: List[Dict[str, Any]] = []
        self.total_communication_bytes: int = 0
        self.total_training_time: float = 0.0
        self.total_aggregation_time: float = 0.0
        self.prior_wall_clock: float = 0.0

        # Crash recovery. When omitted, a manager is created from the config so
        # the orchestrator is resumable even when driven directly.
        self.checkpoint = checkpoint or CheckpointManager(
            checkpoint_path=self.work_dir / "checkpoint.json",
            exp_name=str(self._get("exp_name", "experiment")),
            config=self.config,
            enabled=bool(self._get("enable_checkpointing", True)),
            paradigm="federated",
        )
        # A config change archives the old checkpoint; in that case any leftover
        # step checkpoints on disk belong to the abandoned setup too.
        self.allow_step_resume = self.checkpoint.enabled

        LOGGER.info(
            "FederatedOrchestrator ready | rounds=%d clients=%d blockchain=%s ipfs=%s checkpoint=%s",
            self.num_rounds,
            self.num_clients,
            self.enable_blockchain,
            self.enable_ipfs,
            self.checkpoint.enabled,
        )

    # -- config access -------------------------------------------------------
    def _get(self, key: str, default: Any = None) -> Any:
        try:
            return self.config.get(key, default)
        except AttributeError:
            return getattr(self.config, key, default)

    # =========================================================================
    # Public entry point
    # =========================================================================
    def run(self) -> Dict[str, Any]:
        """Execute every federated round and return the collected metrics.

        Safe to re-invoke after a crash: completed rounds are replayed from the
        checkpoint rather than recomputed, and the round that was in flight
        resumes at the first client that had not finished.
        """
        if self.num_clients == 0:
            raise ValueError("No client_files configured - cannot run a federated experiment.")

        self._check_shards()
        self._restore_from_checkpoint()

        overall_start = time.perf_counter()
        start_round = len(self.round_metrics) + 1

        if start_round > self.num_rounds:
            LOGGER.info("All %d round(s) already complete in the checkpoint.", self.num_rounds)
        elif start_round > 1:
            LOGGER.info(
                "Resuming at round %d/%d (%d round(s) restored from the checkpoint).",
                start_round,
                self.num_rounds,
                start_round - 1,
            )

        try:
            for round_index in range(start_round, self.num_rounds + 1):
                metrics = self.run_round(round_index)
                self.round_metrics.append(metrics)
                self._commit_round(metrics)
        except Exception as exc:
            self.checkpoint.mark_failed(str(exc))
            LOGGER.error(
                "Round loop aborted (%s). Progress is checkpointed at %s - "
                "re-run the same command to continue from here.",
                exc,
                self.checkpoint.path,
            )
            raise

        session_wall_clock = time.perf_counter() - overall_start
        total_wall_clock = self.prior_wall_clock + session_wall_clock
        self.checkpoint.totals["wall_clock_sec"] = total_wall_clock
        self.checkpoint.save()

        summary = self._build_summary(total_wall_clock)
        summary["session_wall_clock_sec"] = round(session_wall_clock, 4)

        LOGGER.info("=" * 78)
        LOGGER.info(
            "Federated run complete: %d round(s) in %s | comm=%.3f MB | train=%s%s",
            self.num_rounds,
            format_duration(total_wall_clock),
            summary["total_communication_mb"],
            format_duration(self.total_training_time),
            f" (resumed across {self.checkpoint.state.get('sessions', 0)} sessions)"
            if self.checkpoint.resumed
            else "",
        )
        LOGGER.info("=" * 78)
        return summary

    # =========================================================================
    # Checkpoint plumbing
    # =========================================================================
    def _restore_from_checkpoint(self) -> None:
        """Rehydrate round metrics, totals and external-service records."""
        if not self.checkpoint.load():
            # Either checkpointing is off, there was nothing to load, or the
            # config changed. In the last case leftover step checkpoints belong
            # to the abandoned configuration and must not be resumed into.
            self.allow_step_resume = bool(self.checkpoint.enabled) and (
                self.checkpoint.stale_checkpoint_archived is None
            )
            return

        self.round_metrics = self.checkpoint.completed_rounds
        totals = self.checkpoint.totals
        self.total_training_time = float(totals.get("training_time_sec", 0.0))
        self.total_aggregation_time = float(totals.get("aggregation_time_sec", 0.0))
        self.total_communication_bytes = int(totals.get("communication_bytes", 0))
        self.prior_wall_clock = float(totals.get("wall_clock_sec", 0.0))

        stored_global = self.checkpoint.global_adapter_path
        if stored_global and adapter_is_complete(stored_global) and adapter_matches_hash(
            stored_global, self.checkpoint.global_adapter_hash
        ):
            self.global_adapter_path = Path(stored_global)
            LOGGER.info("Restored global adapter: %s", stored_global)
        elif stored_global:
            raise RuntimeError(
                "Checkpointed global adapter is missing, incomplete, or corrupt: "
                f"{stored_global}. Refusing to silently continue from fresh weights."
            )

        # Chain and IPFS work already performed is real and must keep counting.
        if self.blockchain is not None and hasattr(self.blockchain, "restore_receipts"):
            self.blockchain.restore_receipts(self.checkpoint.state.get("blockchain_receipts", []))
        if self.ipfs is not None and hasattr(self.ipfs, "restore_transfers"):
            self.ipfs.restore_transfers(self.checkpoint.state.get("ipfs_transfers", []))

    def _commit_round(self, metrics: Dict[str, Any]) -> None:
        """Persist a finished round so it is never recomputed."""
        self.checkpoint.complete_round(
            round_metrics=metrics,
            global_adapter_path=str(self.global_adapter_path) if self.global_adapter_path else None,
            global_adapter_hash=(metrics.get("global_model") or {}).get("model_hash"),
            totals={
                "training_time_sec": self.total_training_time,
                "aggregation_time_sec": self.total_aggregation_time,
                "communication_bytes": self.total_communication_bytes,
                "wall_clock_sec": self.prior_wall_clock,
            },
            blockchain_receipts=self._blockchain_receipt_dicts(),
            ipfs_transfers=self._ipfs_transfer_dicts(),
        )

    def _blockchain_receipt_dicts(self) -> Optional[List[Dict[str, Any]]]:
        if self.blockchain is None:
            return None
        try:
            return [r.to_dict() for r in self.blockchain.receipts]
        except Exception:
            return None

    def _ipfs_transfer_dicts(self) -> Optional[List[Dict[str, Any]]]:
        if self.ipfs is None:
            return None
        try:
            return [t.to_dict() for t in self.ipfs.transfers]
        except Exception:
            return None

    def _check_shards(self) -> None:
        missing = [str(p) for p in self.client_files if not p.exists()]
        if missing:
            raise FileNotFoundError(
                "Missing client shard(s):\n  - "
                + "\n  - ".join(missing)
                + "\n\nRun `python data/prepare_data.py` to download and partition Dolly-15k."
            )

    # =========================================================================
    # One round
    # =========================================================================
    def run_round(self, round_index: int) -> Dict[str, Any]:
        """Train, publish, anchor, retrieve, verify and aggregate for one round."""
        LOGGER.info("")
        LOGGER.info("#" * 78)
        LOGGER.info("# ROUND %d / %d", round_index, self.num_rounds)
        LOGGER.info("#" * 78)

        round_start = time.perf_counter()
        round_dir = self.work_dir / f"round_{round_index}"
        round_dir.mkdir(parents=True, exist_ok=True)

        client_records: List[Dict[str, Any]] = []
        round_comm_bytes = 0
        round_training_time = 0.0

        # Client records already completed for this round before a crash.
        checkpointed = self.checkpoint.partial_clients(round_index)
        if checkpointed:
            LOGGER.info(
                "Checkpoint holds %d/%d client(s) for round %d: %s",
                len(checkpointed),
                self.num_clients,
                round_index,
                ", ".join(sorted(checkpointed)),
            )

        # ---------------- local training + publish + anchor -----------------
        for client_id, shard_path in zip(self.client_ids, self.client_files):
            saved = checkpointed.get(client_id)
            if saved is not None and reusable_adapter(saved):
                record = dict(saved)
                if record.get("stage") == "complete":
                    LOGGER.info(
                        "Skipping %s in round %d - already trained and verified "
                        "(%.1fs of work recovered).",
                        client_id,
                        round_index,
                        float(saved.get("training_time_sec", 0.0)),
                    )
                else:
                    LOGGER.info(
                        "Recovered trained adapter for %s in round %d; continuing at stage '%s'.",
                        client_id,
                        round_index,
                        record.get("stage", "trained"),
                    )
            else:
                record = self._run_client(round_index, client_id, shard_path, round_dir)
                # Persist before IPFS or blockchain work. A crash anywhere
                # after local optimisation can therefore never retrain it.
                self._persist_client(round_index, client_id, record)

            if record.get("stage") != "complete":
                record = self._finish_client(round_index, client_id, record)

            client_records.append(record)
            round_training_time += float(record.get("training_time_sec", 0.0))
            round_comm_bytes += int(record.get("upload_bytes", 0))

        # ---------------- retrieve (Exp 4) ----------------------------------
        aggregation_inputs, retrieval_bytes = self._collect_for_aggregation(
            round_index, client_records, round_dir
        )
        round_comm_bytes += retrieval_bytes

        if not aggregation_inputs:
            raise RuntimeError(
                f"Round {round_index}: no client adapter survived retrieval/verification. "
                "Aborting rather than producing an unsound global model."
            )

        # ---------------- aggregate -----------------------------------------
        global_dir = round_dir / "global"
        client_weights = [r["num_samples"] for r in client_records if r["included_in_aggregate"]]
        if self.enable_aggregation:
            aggregation_metrics = self.aggregator.aggregate_lora_adapters(
                aggregation_inputs,
                global_dir,
                client_weights=client_weights if any(client_weights) else None,
            )
            self.global_adapter_path = global_dir
        else:
            # Local-only ablation: no FedAvg step at all. Each client keeps its
            # own lineage (see `_init_adapter_for`), so this measures what a
            # participant achieves training alone on its own shard for the same
            # number of updates. It is the reference that shows whether
            # federation buys anything at all - without it, "FL costs +x% loss"
            # has no counterpart showing what isolation costs.
            aggregation_metrics = self._no_aggregation_metrics(
                aggregation_inputs, global_dir
            )
            self.global_adapter_path = Path(aggregation_metrics["global_adapter_path"])
        self.total_aggregation_time += aggregation_metrics["aggregation_time_sec"]

        # ---------------- publish + anchor the global model -----------------
        # With aggregation off there is no `global/` directory to publish; the
        # representative artefact is client 1's own adapter.
        publish_dir = Path(self.global_adapter_path or global_dir)
        global_record = self._publish_global(round_index, publish_dir)
        # Broadcast accounting for the new global adapter:
        #   without IPFS - the server pushes it to each of the N clients: N x size
        #   with IPFS    - the server pins it once, then each client pulls it:
        #                  1 x size (upload) + N x size (downloads)
        broadcast_bytes = global_record["upload_bytes"] * self.num_clients
        if self.enable_ipfs:
            broadcast_bytes = global_record["upload_bytes"] * (self.num_clients + 1)
        global_record["broadcast_bytes"] = broadcast_bytes
        round_comm_bytes += broadcast_bytes

        # ---------------- evaluate ------------------------------------------
        # Local-only has no aggregate worth scoring per round; every client is
        # scored once at the end by `_evaluate_local_only`.
        eval_metrics = (
            self._evaluate_round(round_index, publish_dir) if self.enable_aggregation else None
        )

        round_time = time.perf_counter() - round_start
        self.total_communication_bytes += round_comm_bytes
        self.total_training_time += round_training_time

        metrics: Dict[str, Any] = {
            "round": round_index,
            "round_latency_sec": round(round_time, 4),
            "training_time_sec": round(round_training_time, 4),
            "aggregation_time_sec": aggregation_metrics["aggregation_time_sec"],
            "communication_mb": bytes_to_mb(round_comm_bytes),
            "num_clients_included": len(aggregation_inputs),
            "clients": client_records,
            "aggregation": aggregation_metrics,
            "global_model": global_record,
            "global_adapter_path": str(publish_dir),
            "global_adapter_mb": bytes_to_mb(path_size_bytes(publish_dir)),
            "evaluation": eval_metrics,
        }

        LOGGER.info(
            "Round %d done in %s | comm=%.3f MB | %s",
            round_index,
            format_duration(round_time),
            metrics["communication_mb"],
            f"val_loss={eval_metrics['loss']:.4f} ppl={eval_metrics['perplexity']:.4f}"
            if eval_metrics
            else "evaluation skipped",
        )
        return metrics

    # -- step 1: one client ---------------------------------------------------
    def _sample_offset(self, round_index: int) -> int:
        """Start index into each client shard for ``round_index`` (1-based).

        Rounds consume *disjoint* windows: round 1 reads ``[0, C)``, round 2
        reads ``[C, 2C)``, and so on for a per-round cap of ``C``. Replaying the
        head of the shard every round would make an R-round federated run an
        R-epoch pass over ``C`` examples, so the federation would see R times
        less unique data than the centralized baseline it is compared to - and
        the reported "cost of federation" would really be a data-budget
        artefact. ``LocalTrainer.load_dataset`` wraps the window if a shard is
        too small to serve every round.
        """
        cap = self._get("max_train_samples", None)
        if not cap:
            return 0
        return (int(round_index) - 1) * int(cap)

    def _init_adapter_for(self, client_id: str) -> Optional[Path]:
        """Weights a client warm-starts from this round.

        Federated: the previous round's *global* adapter, which is what makes
        FedAvg iterative. Local-only: the client's own previous adapter, so the
        ablation never benefits from anyone else's data.
        """
        if self.enable_aggregation:
            return self.global_adapter_path
        return self.local_adapter_paths.get(client_id)

    def _no_aggregation_metrics(
        self, aggregation_inputs: List[Path], global_dir: Path
    ) -> Dict[str, Any]:
        """Stand-in for the FedAvg record when aggregation is disabled."""
        representative = str(aggregation_inputs[0]) if aggregation_inputs else str(global_dir)
        return {
            "num_clients": len(aggregation_inputs),
            "weighting": "none (local-only ablation)",
            "aggregation_time_sec": 0.0,
            "global_adapter_path": representative,
            "artifact_size_mb": bytes_to_mb(path_size_bytes(representative)),
            "note": (
                "Aggregation disabled: no averaging took place. The reported "
                "adapter is client 1's; per-client scores are in "
                "`per_client_evaluation`."
            ),
        }

    def _run_client(
        self, round_index: int, client_id: str, shard_path: Path, round_dir: Path
    ) -> Dict[str, Any]:
        adapter_dir = round_dir / client_id
        adapter_path, training_time = self.trainer.train_client(
            dataset_path=shard_path,
            output_dir=adapter_dir,
            init_adapter_path=self._init_adapter_for(client_id),
            client_id=f"{client_id}@r{round_index}",
            allow_resume=self.allow_step_resume,
            sample_offset=self._sample_offset(round_index),
        )
        if not self.enable_aggregation:
            self.local_adapter_paths[client_id] = Path(adapter_path)

        adapter_bytes = path_size_bytes(adapter_path)
        record: Dict[str, Any] = {
            "client_id": client_id,
            "round": round_index,
            "shard": str(shard_path),
            "adapter_path": str(adapter_path),
            "adapter_mb": bytes_to_mb(adapter_bytes),
            "training_time_sec": training_time,
            "num_samples": self._shard_sample_count(shard_path),
            "model_hash": None,
            "ipfs_cid": None,
            "ipfs_upload_latency_sec": 0.0,
            "ipfs_download_latency_sec": 0.0,
            "blockchain": None,
            "upload_bytes": 0,
            "download_bytes": 0,
            "integrity_verified": None,
            "included_in_aggregate": True,
            "ipfs_attempted": False,
            "blockchain_attempted": False,
            "stage": "trained",
        }

        # Hash before any external side effect and before returning to the
        # round loop, so the durable trained-stage record is self-validating.
        record["model_hash"] = sha256_path(adapter_path)
        return record

    def _finish_client(
        self, round_index: int, client_id: str, record: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Continue a trained client through publish/anchor without retraining."""
        adapter_path = Path(record["adapter_path"])
        adapter_bytes = path_size_bytes(adapter_path)

        # --- publish to IPFS (Exp 4) ---
        if self.enable_ipfs and not record.get("ipfs_attempted", False):
            try:
                cid, latency, size_mb = self.ipfs.upload_adapter(adapter_path)
                record["ipfs_cid"] = cid
                record["ipfs_upload_latency_sec"] = latency
                record["upload_bytes"] = int(size_mb * (1024 ** 2))
            except Exception as exc:
                LOGGER.error("IPFS upload failed for %s: %s", client_id, exc)
                record["upload_bytes"] = adapter_bytes
            finally:
                record["ipfs_attempted"] = True
                record["stage"] = "published"
                self._persist_client(round_index, client_id, record)
        elif not self.enable_ipfs:
            # No IPFS: the adapter still travels to the aggregator.
            record["upload_bytes"] = adapter_bytes
            record["ipfs_attempted"] = True
            record["stage"] = "published"
            self._persist_client(round_index, client_id, record)

        # --- anchor on-chain (Exp 3 & 4) ---
        if self.enable_blockchain and not record.get("blockchain_attempted", False):
            receipt = self.blockchain.log_model_update(
                round=round_index,
                client_id=client_id,
                adapter_bytes_or_file=adapter_path,
                ipfs_cid=record["ipfs_cid"] or "",
            )
            record["blockchain"] = receipt
            record["model_hash"] = receipt.get("model_hash") or record["model_hash"]
            record["blockchain_attempted"] = True
            record["stage"] = "anchored"
            self._persist_client(round_index, client_id, record)
        elif not self.enable_blockchain:
            record["blockchain_attempted"] = True

        record["stage"] = "complete"
        self._persist_client(round_index, client_id, record)

        return record

    def _persist_client(
        self, round_index: int, client_id: str, record: Dict[str, Any]
    ) -> None:
        """Durably snapshot a client plus external-service accounting."""
        self.checkpoint.record_client(
            round_index,
            client_id,
            record,
            blockchain_receipts=self._blockchain_receipt_dicts(),
            ipfs_transfers=self._ipfs_transfer_dicts(),
        )

    # -- step 2: retrieval ----------------------------------------------------
    def _collect_for_aggregation(
        self, round_index: int, client_records: List[Dict[str, Any]], round_dir: Path
    ) -> Tuple[List[Path], int]:
        """Return the adapter paths the aggregator should consume.

        With IPFS enabled and round-tripping on, every adapter is fetched back
        from the network by CID and verified against its anchored digest, which
        is the property the paper claims for FedChain. Otherwise the locally
        written directories are used directly.
        """
        if not (self.enable_ipfs and self.ipfs_roundtrip):
            return [Path(r["adapter_path"]) for r in client_records], 0

        retrieval_dir = round_dir / "_retrieved"
        retrieval_dir.mkdir(parents=True, exist_ok=True)

        paths: List[Path] = []
        retrieved_bytes = 0

        for record in client_records:
            cid = record.get("ipfs_cid")
            if not cid:
                LOGGER.warning(
                    "%s has no CID (upload failed); aggregating from its local copy instead.",
                    record["client_id"],
                )
                paths.append(Path(record["adapter_path"]))
                continue

            target = retrieval_dir / record["client_id"]
            try:
                latency, wire_bytes = self.ipfs.download_adapter(cid, target)
                record["ipfs_download_latency_sec"] = latency
                # Count what crossed the wire, not what it expanded to on disk.
                # The upload side is measured post-compression, so charging the
                # download the *unpacked* directory size would make the same
                # artefact cost ~27% more to fetch than it did to publish and
                # would inflate exp4's communication volume against exp2/exp3.
                record["download_bytes"] = wire_bytes
                record["unpacked_bytes"] = path_size_bytes(target)
                retrieved_bytes += wire_bytes
            except Exception as exc:
                LOGGER.error("IPFS download failed for %s (%s): %s", record["client_id"], cid, exc)
                paths.append(Path(record["adapter_path"]))
                continue

            if self.verify_on_download and record.get("model_hash"):
                actual = sha256_path(target)
                verified = actual.lower() == str(record["model_hash"]).lower()
                record["integrity_verified"] = verified
                if not verified:
                    LOGGER.error(
                        "INTEGRITY FAILURE for %s in round %d: on-chain %s != retrieved %s. "
                        "Excluding this contribution from the aggregate.",
                        record["client_id"],
                        round_index,
                        record["model_hash"],
                        actual,
                    )
                    record["included_in_aggregate"] = False
                    continue
                LOGGER.info("Integrity verified for %s (%s...)", record["client_id"], actual[:16])

            paths.append(target)

        return paths, retrieved_bytes

    # -- step 3: publish the global model -------------------------------------
    def _publish_global(self, round_index: int, global_dir: Path) -> Dict[str, Any]:
        record: Dict[str, Any] = {
            "round": round_index,
            "adapter_path": str(global_dir),
            "adapter_mb": bytes_to_mb(path_size_bytes(global_dir)),
            "model_hash": None,
            "ipfs_cid": None,
            "ipfs_upload_latency_sec": 0.0,
            "blockchain": None,
            "upload_bytes": path_size_bytes(global_dir),
            "broadcast_bytes": 0,
        }

        if self.enable_ipfs:
            try:
                cid, latency, size_mb = self.ipfs.upload_adapter(global_dir)
                record["ipfs_cid"] = cid
                record["ipfs_upload_latency_sec"] = latency
                record["upload_bytes"] = int(size_mb * (1024 ** 2))
            except Exception as exc:
                LOGGER.error("IPFS upload of the global adapter failed: %s", exc)

        if self.enable_blockchain and self.log_global_model:
            receipt = self.blockchain.log_model_update(
                round=round_index,
                client_id="server_global",
                adapter_bytes_or_file=global_dir,
                ipfs_cid=record["ipfs_cid"] or "",
            )
            record["blockchain"] = receipt
            record["model_hash"] = receipt.get("model_hash")
        else:
            record["model_hash"] = sha256_path(global_dir)

        return record

    # -- step 4: evaluation ---------------------------------------------------
    def _evaluate_round(self, round_index: int, global_dir: Path) -> Optional[Dict[str, Any]]:
        if self.evaluator is None or not bool(self._get("eval_every_round", True)):
            return None
        # The final round is scored by main.py; skipping it here avoids paying
        # for the same forward pass twice.
        if round_index == self.num_rounds and bool(self._get("eval_final", True)):
            LOGGER.debug("Deferring round %d evaluation to the final scoring pass.", round_index)
            return None
        try:
            free_cuda_memory()
            return self.evaluator.evaluate(str(global_dir))
        except Exception as exc:
            LOGGER.error("Round %d evaluation failed: %s", round_index, exc)
            return None

    # =========================================================================
    # Bookkeeping
    # =========================================================================
    def _shard_sample_count(self, shard_path: Path) -> int:
        """Number of JSONL records actually used from a shard this round."""
        cap = self._get("max_train_samples", None)
        try:
            with open(shard_path, "r", encoding="utf-8") as handle:
                total = sum(1 for line in handle if line.strip())
        except Exception:
            return int(cap) if cap else 0
        return min(total, int(cap)) if cap else total

    def _evaluate_local_only(self) -> Optional[Dict[str, Any]]:
        """Score every client's final adapter for the local-only ablation.

        Reporting a single client would be arbitrary, and with non-IID shards
        the spread across clients is itself a result: it is the variance that
        aggregation is supposed to remove.
        """
        if self.evaluator is None or self.enable_aggregation or not self.round_metrics:
            return None

        final_round = self.round_metrics[-1]
        per_client: List[Dict[str, Any]] = []
        for record in final_round.get("clients", []):
            try:
                free_cuda_memory()
                scores = self.evaluator.evaluate(
                    record["adapter_path"], label=f"local_only:{record['client_id']}"
                )
            except Exception as exc:
                LOGGER.error("Local-only evaluation failed for %s: %s", record["client_id"], exc)
                continue
            per_client.append({"client_id": record["client_id"], **scores})

        if not per_client:
            return None

        def spread(key: str) -> Dict[str, float]:
            values = [float(p[key]) for p in per_client if p.get(key) is not None]
            if not values:
                return {}
            mean = sum(values) / len(values)
            variance = sum((v - mean) ** 2 for v in values) / max(1, len(values) - 1)
            return {
                "mean": round(mean, 6),
                "std": round(variance ** 0.5, 6),
                "min": round(min(values), 6),
                "max": round(max(values), 6),
            }

        return {
            "per_client": per_client,
            "loss": spread("loss"),
            "perplexity": spread("perplexity"),
            "rouge_l": spread("rouge_l"),
            "bleu": spread("bleu"),
        }

    def _build_summary(self, total_wall_clock: float) -> Dict[str, Any]:
        last_eval = None
        for metrics in reversed(self.round_metrics):
            if metrics.get("evaluation"):
                last_eval = metrics["evaluation"]
                break

        summary: Dict[str, Any] = {
            "paradigm": "federated",
            "num_rounds": self.num_rounds,
            "num_clients": self.num_clients,
            "total_wall_clock_sec": round(total_wall_clock, 4),
            "total_training_time_sec": round(self.total_training_time, 4),
            "total_aggregation_time_sec": round(self.total_aggregation_time, 4),
            "total_communication_mb": bytes_to_mb(self.total_communication_bytes),
            "avg_round_latency_sec": round(
                sum(m["round_latency_sec"] for m in self.round_metrics) / max(1, len(self.round_metrics)),
                4,
            ),
            "global_adapter_path": str(self.global_adapter_path) if self.global_adapter_path else None,
            "global_adapter_mb": bytes_to_mb(path_size_bytes(self.global_adapter_path))
            if self.global_adapter_path
            else 0.0,
            "last_round_evaluation": last_eval,
            "checkpoint": self.checkpoint.describe(),
            "rounds": self.round_metrics,
        }

        integrity_checks = [
            r["integrity_verified"]
            for m in self.round_metrics
            for r in m["clients"]
            if r.get("integrity_verified") is not None
        ]
        if integrity_checks:
            summary["integrity_checks_total"] = len(integrity_checks)
            summary["integrity_checks_passed"] = sum(1 for v in integrity_checks if v)

        if not self.enable_aggregation:
            summary["paradigm"] = "local_only"
            summary["aggregation_enabled"] = False
            local_scores = self._evaluate_local_only()
            if local_scores:
                summary["per_client_evaluation"] = local_scores
                # Report the client mean as the headline number: no single
                # client's adapter is "the" model when nothing was aggregated.
                mean_scores = {
                    key: local_scores[key].get("mean")
                    for key in ("loss", "perplexity", "rouge_l", "bleu")
                    if local_scores.get(key)
                }
                if mean_scores:
                    summary["last_round_evaluation"] = {
                        "label": "local_only_mean",
                        "num_clients_averaged": len(local_scores["per_client"]),
                        **mean_scores,
                    }

        return summary

    def cleanup(self, remove_intermediate: bool = False) -> None:
        """Release the trainer and optionally delete per-round scratch adapters."""
        try:
            self.trainer.cleanup()
        except Exception as exc:  # pragma: no cover
            LOGGER.debug("Trainer cleanup raised: %s", exc)

        if remove_intermediate:
            for round_index in range(1, self.num_rounds):
                stale = self.work_dir / f"round_{round_index}"
                if stale.exists():
                    shutil.rmtree(stale, ignore_errors=True)
            LOGGER.info("Removed intermediate round artefacts under %s", self.work_dir)

        free_cuda_memory()
