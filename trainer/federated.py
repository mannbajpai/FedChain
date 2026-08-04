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
    ) -> None:
        self.config = config
        self.trainer = trainer or LocalTrainer(config)
        self.aggregator = aggregator or FedAvgAggregator(weighted=False)
        self.blockchain = blockchain_logger
        self.ipfs = ipfs_manager
        self.evaluator = evaluator

        self.enable_blockchain = bool(self._get("enable_blockchain", False)) and self.blockchain is not None
        self.enable_ipfs = bool(self._get("enable_ipfs", False)) and self.ipfs is not None
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

        LOGGER.info(
            "FederatedOrchestrator ready | rounds=%d clients=%d blockchain=%s ipfs=%s",
            self.num_rounds,
            self.num_clients,
            self.enable_blockchain,
            self.enable_ipfs,
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
        """Execute every federated round and return the collected metrics."""
        if self.num_clients == 0:
            raise ValueError("No client_files configured - cannot run a federated experiment.")

        self._check_shards()
        overall_start = time.perf_counter()

        for round_index in range(1, self.num_rounds + 1):
            metrics = self.run_round(round_index)
            self.round_metrics.append(metrics)

        total_wall_clock = time.perf_counter() - overall_start
        summary = self._build_summary(total_wall_clock)

        LOGGER.info("=" * 78)
        LOGGER.info(
            "Federated run complete: %d round(s) in %s | comm=%.3f MB | train=%s",
            self.num_rounds,
            format_duration(total_wall_clock),
            summary["total_communication_mb"],
            format_duration(self.total_training_time),
        )
        LOGGER.info("=" * 78)
        return summary

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

        # ---------------- local training + publish + anchor -----------------
        for client_id, shard_path in zip(self.client_ids, self.client_files):
            record = self._run_client(round_index, client_id, shard_path, round_dir)
            client_records.append(record)
            round_training_time += record["training_time_sec"]
            round_comm_bytes += record["upload_bytes"]

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
        aggregation_metrics = self.aggregator.aggregate_lora_adapters(
            aggregation_inputs,
            global_dir,
            client_weights=client_weights if any(client_weights) else None,
        )
        self.total_aggregation_time += aggregation_metrics["aggregation_time_sec"]
        self.global_adapter_path = global_dir

        # ---------------- publish + anchor the global model -----------------
        global_record = self._publish_global(round_index, global_dir)
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
        eval_metrics = self._evaluate_round(round_index, global_dir)

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
            "global_adapter_path": str(global_dir),
            "global_adapter_mb": bytes_to_mb(path_size_bytes(global_dir)),
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
    def _run_client(
        self, round_index: int, client_id: str, shard_path: Path, round_dir: Path
    ) -> Dict[str, Any]:
        adapter_dir = round_dir / client_id
        adapter_path, training_time = self.trainer.train_client(
            dataset_path=shard_path,
            output_dir=adapter_dir,
            init_adapter_path=self.global_adapter_path,
            client_id=f"{client_id}@r{round_index}",
        )

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
        }

        # --- publish to IPFS (Exp 4) ---
        if self.enable_ipfs:
            try:
                cid, latency, size_mb = self.ipfs.upload_adapter(adapter_path)
                record["ipfs_cid"] = cid
                record["ipfs_upload_latency_sec"] = latency
                record["upload_bytes"] = int(size_mb * (1024 ** 2))
            except Exception as exc:
                LOGGER.error("IPFS upload failed for %s: %s", client_id, exc)
                record["upload_bytes"] = adapter_bytes
        else:
            # No IPFS: the adapter still travels to the aggregator.
            record["upload_bytes"] = adapter_bytes

        # --- anchor on-chain (Exp 3 & 4) ---
        if self.enable_blockchain:
            receipt = self.blockchain.log_model_update(
                round=round_index,
                client_id=client_id,
                adapter_bytes_or_file=adapter_path,
                ipfs_cid=record["ipfs_cid"] or "",
            )
            record["blockchain"] = receipt
            record["model_hash"] = receipt.get("model_hash")
        else:
            # The digest is cheap and makes every run auditable after the fact.
            record["model_hash"] = sha256_path(adapter_path)

        return record

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
                latency = self.ipfs.download_adapter(cid, target)
                record["ipfs_download_latency_sec"] = latency
                downloaded = path_size_bytes(target)
                record["download_bytes"] = downloaded
                retrieved_bytes += downloaded
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
