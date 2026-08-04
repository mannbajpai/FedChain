#!/usr/bin/env python
"""
FedChain :: Unified experiment entry point
==========================================

Runs any of the four benchmark paradigms from a single YAML configuration::

    python main.py --config configs/exp1_sft.yaml        # centralized SFT
    python main.py --config configs/exp2_fl.yaml         # FedAvg
    python main.py --config configs/exp3_fl_bc.yaml      # FedAvg + blockchain
    python main.py --config configs/exp4_fedchain.yaml   # FedChain (+ IPFS)

Every run emits the same metric schema, so the four experiments drop straight
into one comparison table:

===============================  ==========================================
``validation_loss``              token-weighted NLL on the held-out split
``perplexity``                   exp(validation_loss)
``rouge_l`` / ``bleu``           instruction-following overlap (0-1 scale)
``training_time_sec``            wall-clock spent optimising weights
``communication_volume_mb``      total MB crossing a participant boundary
``adapter_size_mb``              size of the final LoRA artefact
``blockchain_tx_latency_sec``    cumulative on-chain anchoring latency
``blockchain_gas_used``          cumulative gas
``ipfs_upload_latency_sec``      cumulative pin latency
``ipfs_download_latency_sec``    cumulative retrieval latency
``aggregation_time_sec``         time spent inside FedAvg
``end_to_end_round_duration_sec``  mean wall-clock per federated round
===============================  ==========================================

Results are printed as a Markdown table and written to
``results/<exp_name>_metrics.json``.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Ensure the repository root is importable when main.py is invoked by path.
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from blockchain.logger import BlockchainLogger
from evaluation.eval_loss import Evaluator
from ipfs.pinata_client import IPFSManager
from trainer.aggregation import FedAvgAggregator
from trainer.federated import FederatedOrchestrator
from trainer.sft import LocalTrainer
from utils.common import (
    bytes_to_mb,
    describe_environment,
    format_duration,
    free_cuda_memory,
    get_device,
    markdown_table,
    path_size_bytes,
    set_global_seed,
    setup_logging,
    write_json,
)
from utils.config import load_config, resolve_path

LOGGER = logging.getLogger("fedchain")

#: The reported metric schema, in presentation order.
METRIC_SCHEMA: Tuple[Tuple[str, str, str], ...] = (
    ("validation_loss", "Validation Loss", "float4"),
    ("perplexity", "Perplexity", "float4"),
    ("rouge_l", "ROUGE-L", "float4"),
    ("bleu", "BLEU", "float4"),
    ("training_time_sec", "Training Time (s)", "float2"),
    ("communication_volume_mb", "Communication Volume (MB)", "float3"),
    ("adapter_size_mb", "Adapter Size (MB)", "float3"),
    ("blockchain_tx_latency_sec", "Blockchain Tx Latency (s)", "float4"),
    ("blockchain_gas_used", "Blockchain Gas Used", "int"),
    ("ipfs_upload_latency_sec", "IPFS Upload Latency (s)", "float4"),
    ("ipfs_download_latency_sec", "IPFS Download Latency (s)", "float4"),
    ("aggregation_time_sec", "Aggregation Time (s)", "float4"),
    ("end_to_end_round_duration_sec", "End-to-End Round Duration (s)", "float2"),
)


# =============================================================================
# CLI
# =============================================================================
def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="main.py",
        description="FedChain: auditable federated fine-tuning benchmark harness.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python main.py --config configs/exp1_sft.yaml\n"
            "  python main.py --config configs/exp4_fedchain.yaml --num-rounds 1\n"
            "  python main.py --config configs/exp4_fedchain.yaml --dry-run\n"
        ),
    )
    parser.add_argument(
        "--config",
        required=True,
        help="Path to the experiment YAML (e.g. configs/exp4_fedchain.yaml).",
    )
    parser.add_argument("--exp-name", dest="exp_name", default=None, help="Override the experiment name.")
    parser.add_argument("--seed", type=int, default=None, help="Override the random seed.")
    parser.add_argument("--num-rounds", dest="num_rounds", type=int, default=None, help="Override num_rounds.")
    parser.add_argument(
        "--max-train-samples",
        dest="max_train_samples",
        type=int,
        default=None,
        help="Override max_train_samples (per client, per round).",
    )
    parser.add_argument(
        "--device",
        choices=["auto", "cuda", "cpu"],
        default=None,
        help="Force the compute device (default: auto-detect with CPU fallback).",
    )
    parser.add_argument("--output-root", dest="output_root", default=None, help="Override the artefact root.")
    parser.add_argument("--results-dir", dest="results_dir", default=None, help="Override the results directory.")
    parser.add_argument(
        "--log-level",
        dest="log_level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default=None,
        help="Console log level.",
    )
    parser.add_argument(
        "--skip-eval",
        action="store_true",
        help="Skip evaluation entirely (systems metrics only).",
    )
    parser.add_argument(
        "--no-generation-metrics",
        action="store_true",
        help="Skip ROUGE-L / BLEU generation (loss and perplexity only).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help=(
            "Exercise the full pipeline with synthetic adapters and no model download. "
            "Useful for validating the blockchain/IPFS/aggregation paths without a GPU."
        ),
    )
    parser.add_argument(
        "--force-mock-chain",
        action="store_true",
        help="Force blockchain mock mode even when an RPC node is reachable.",
    )
    parser.add_argument(
        "--force-mock-ipfs",
        action="store_true",
        help="Force IPFS mock mode even when Pinata or a local daemon is reachable.",
    )
    return parser


def cli_overrides(args: argparse.Namespace) -> Dict[str, Any]:
    """Turn parsed flags into config overrides (``None`` values are dropped)."""
    overrides: Dict[str, Any] = {
        "exp_name": args.exp_name,
        "seed": args.seed,
        "num_rounds": args.num_rounds,
        "max_train_samples": args.max_train_samples,
        "device": args.device,
        "output_root": args.output_root,
        "results_dir": args.results_dir,
        "log_level": args.log_level,
    }
    if args.dry_run:
        overrides["dry_run"] = True
    if args.no_generation_metrics:
        overrides["enable_generation_metrics"] = False
    if args.skip_eval:
        overrides["eval_every_round"] = False
        overrides["eval_final"] = False
    return overrides


# =============================================================================
# Component construction
# =============================================================================
def build_blockchain_logger(cfg: Any, force_mock: bool = False) -> Optional[BlockchainLogger]:
    if not cfg.get("enable_blockchain", False):
        return None
    LOGGER.info("Initialising the blockchain audit layer ...")
    return BlockchainLogger(
        rpc_url=cfg.get("rpc_url", "http://127.0.0.1:8545"),
        contract_address=cfg.get("contract_address", ""),
        private_key=cfg.get("private_key", ""),
        chain_id=cfg.get("chain_id", None),
        contract_artifact=resolve_path(cfg.get("contract_artifact", "blockchain/artifacts/FedChainAudit.json")),
        contract_source=resolve_path(cfg.get("contract_source", "blockchain/contract.sol")),
        tx_timeout=int(cfg.get("blockchain_tx_timeout", 120)),
        force_mock=force_mock,
    )


def build_ipfs_manager(cfg: Any, force_mock: bool = False) -> Optional[IPFSManager]:
    if not cfg.get("enable_ipfs", False):
        return None
    LOGGER.info("Initialising the IPFS storage layer ...")
    return IPFSManager(
        pinata_api_key=cfg.get("pinata_api_key", ""),
        pinata_secret_key=cfg.get("pinata_secret_key", ""),
        pinata_jwt=cfg.get("pinata_jwt", ""),
        local_api_url=cfg.get("ipfs_local_api", "http://127.0.0.1:5001"),
        gateway_url=cfg.get("ipfs_gateway_url", "https://gateway.pinata.cloud/ipfs"),
        mock_dir=cfg.get("ipfs_mock_dir", ""),
        timeout=int(cfg.get("ipfs_timeout", 120)),
        force_mock=force_mock,
    )


def build_evaluator(cfg: Any) -> Optional[Evaluator]:
    if not (cfg.get("eval_every_round", True) or cfg.get("eval_final", True)):
        LOGGER.info("Evaluation disabled by configuration.")
        return None
    return Evaluator(cfg)


# =============================================================================
# Experiment pipelines
# =============================================================================
def run_centralized(cfg: Any, evaluator: Optional[Evaluator]) -> Dict[str, Any]:
    """Experiment 1: a single trainer over the pooled corpus."""
    LOGGER.info("=" * 78)
    LOGGER.info("PIPELINE: centralized supervised fine-tuning (no federation)")
    LOGGER.info("=" * 78)

    data_path = resolve_path(cfg.get("data_path", "data/centralized_full.jsonl"))
    work_dir = resolve_path(cfg.get("output_root", "outputs")) / cfg.get("exp_name", "exp1_sft")
    adapter_dir = work_dir / "global"

    trainer = LocalTrainer(cfg)
    overall_start = time.perf_counter()
    total_training_time = 0.0
    rounds: List[Dict[str, Any]] = []
    global_adapter: Optional[Path] = None

    try:
        num_rounds = int(cfg.get("num_rounds", 1))
        for round_index in range(1, num_rounds + 1):
            round_start = time.perf_counter()
            round_output = adapter_dir if num_rounds == 1 else work_dir / f"round_{round_index}"

            adapter_path, training_time = trainer.train_client(
                dataset_path=data_path,
                output_dir=round_output,
                init_adapter_path=global_adapter,
                client_id=f"centralized@r{round_index}",
            )
            total_training_time += training_time
            global_adapter = Path(adapter_path)

            round_eval = None
            if evaluator is not None and cfg.get("eval_every_round", True) and round_index < num_rounds:
                free_cuda_memory()
                try:
                    round_eval = evaluator.evaluate(str(global_adapter), label=f"round_{round_index}")
                except Exception as exc:
                    LOGGER.error("Round %d evaluation failed: %s", round_index, exc)

            rounds.append(
                {
                    "round": round_index,
                    "round_latency_sec": round(time.perf_counter() - round_start, 4),
                    "training_time_sec": training_time,
                    "aggregation_time_sec": 0.0,
                    "communication_mb": 0.0,
                    "global_adapter_path": str(global_adapter),
                    "global_adapter_mb": bytes_to_mb(path_size_bytes(global_adapter)),
                    "evaluation": round_eval,
                }
            )
    finally:
        trainer.cleanup()

    return {
        "paradigm": "centralized",
        "num_rounds": len(rounds),
        "num_clients": 1,
        "total_wall_clock_sec": round(time.perf_counter() - overall_start, 4),
        "total_training_time_sec": round(total_training_time, 4),
        "total_aggregation_time_sec": 0.0,
        # Nothing crosses a participant boundary in the centralized baseline.
        "total_communication_mb": 0.0,
        "avg_round_latency_sec": round(
            sum(r["round_latency_sec"] for r in rounds) / max(1, len(rounds)), 4
        ),
        "global_adapter_path": str(global_adapter) if global_adapter else None,
        "global_adapter_mb": bytes_to_mb(path_size_bytes(global_adapter)) if global_adapter else 0.0,
        "last_round_evaluation": rounds[-1]["evaluation"] if rounds else None,
        "rounds": rounds,
    }


def run_federated(
    cfg: Any,
    evaluator: Optional[Evaluator],
    blockchain_logger: Optional[BlockchainLogger],
    ipfs_manager: Optional[IPFSManager],
) -> Dict[str, Any]:
    """Experiments 2-4: the federated loop, with optional audit and storage."""
    LOGGER.info("=" * 78)
    LOGGER.info(
        "PIPELINE: federated learning | blockchain=%s | ipfs=%s",
        bool(cfg.get("enable_blockchain", False)),
        bool(cfg.get("enable_ipfs", False)),
    )
    LOGGER.info("=" * 78)

    orchestrator = FederatedOrchestrator(
        config=cfg,
        trainer=LocalTrainer(cfg),
        aggregator=FedAvgAggregator(weighted=False),
        blockchain_logger=blockchain_logger,
        ipfs_manager=ipfs_manager,
        evaluator=evaluator,
    )
    try:
        return orchestrator.run()
    finally:
        orchestrator.cleanup()


# =============================================================================
# Metric assembly
# =============================================================================
def collect_metrics(
    cfg: Any,
    run_summary: Dict[str, Any],
    final_eval: Optional[Dict[str, Any]],
    blockchain_logger: Optional[BlockchainLogger],
    ipfs_manager: Optional[IPFSManager],
) -> Dict[str, Any]:
    """Reduce the run into the flat, comparable metric schema."""
    chain_summary = blockchain_logger.get_metrics_summary() if blockchain_logger else None
    ipfs_summary = ipfs_manager.get_metrics_summary() if ipfs_manager else None

    evaluation = final_eval or run_summary.get("last_round_evaluation") or {}

    metrics: Dict[str, Any] = {
        "validation_loss": evaluation.get("loss"),
        "perplexity": evaluation.get("perplexity"),
        "rouge_l": evaluation.get("rouge_l"),
        "bleu": evaluation.get("bleu"),
        "training_time_sec": run_summary.get("total_training_time_sec", 0.0),
        "communication_volume_mb": run_summary.get("total_communication_mb", 0.0),
        "adapter_size_mb": run_summary.get("global_adapter_mb", 0.0),
        "blockchain_tx_latency_sec": chain_summary["total_latency_sec"] if chain_summary else 0.0,
        "blockchain_gas_used": chain_summary["total_gas_used"] if chain_summary else 0,
        "ipfs_upload_latency_sec": ipfs_summary["total_upload_latency_sec"] if ipfs_summary else 0.0,
        "ipfs_download_latency_sec": ipfs_summary["total_download_latency_sec"] if ipfs_summary else 0.0,
        "aggregation_time_sec": run_summary.get("total_aggregation_time_sec", 0.0),
        "end_to_end_round_duration_sec": run_summary.get("avg_round_latency_sec", 0.0),
    }
    return metrics


def log_metrics(metrics: Dict[str, Any]) -> None:
    """Explicitly log every metric in the schema, one line each."""
    LOGGER.info("-" * 78)
    LOGGER.info("COLLECTED METRICS")
    LOGGER.info("-" * 78)
    width = max(len(key) for key, _, _ in METRIC_SCHEMA)
    for key, _, kind in METRIC_SCHEMA:
        LOGGER.info("  %-*s : %s", width, key, format_metric(metrics.get(key), kind))
    LOGGER.info("-" * 78)


def format_metric(value: Any, kind: str) -> str:
    """Render a metric for the console and the Markdown table."""
    if value is None:
        return "n/a"
    try:
        if kind == "int":
            return f"{int(value):,}"
        if kind == "float2":
            return f"{float(value):.2f}"
        if kind == "float3":
            return f"{float(value):.3f}"
        return f"{float(value):.4f}"
    except (TypeError, ValueError):
        return str(value)


def render_markdown(cfg: Any, metrics: Dict[str, Any], context: Dict[str, Any]) -> str:
    """Build the stdout summary: run context + the metric table."""
    lines: List[str] = []
    lines.append("")
    lines.append(f"## FedChain Results - `{cfg.get('exp_name')}`")
    lines.append("")
    lines.append(f"_{cfg.get('exp_description', 'FedChain experiment')}_")
    lines.append("")

    context_rows = [
        ["Model", cfg.get("model_name")],
        ["Paradigm", context.get("paradigm")],
        ["Rounds x Clients", f"{context.get('num_rounds')} x {context.get('num_clients')}"],
        ["Device", context.get("device")],
        ["Blockchain", context.get("blockchain_mode") or "disabled"],
        ["IPFS", context.get("ipfs_backend") or "disabled"],
        ["Total wall clock", format_duration(context.get("total_wall_clock_sec", 0.0))],
    ]
    lines.append(markdown_table(["Setting", "Value"], context_rows))
    lines.append("")

    metric_rows = [
        [label, format_metric(metrics.get(key), kind)] for key, label, kind in METRIC_SCHEMA
    ]
    lines.append(markdown_table(["Metric", "Value"], metric_rows))
    lines.append("")
    return "\n".join(lines)


# =============================================================================
# Orchestration
# =============================================================================
def run_experiment(args: argparse.Namespace) -> Dict[str, Any]:
    cfg = load_config(args.config, overrides=cli_overrides(args))

    results_dir = resolve_path(cfg.get("results_dir", "results"))
    results_dir.mkdir(parents=True, exist_ok=True)
    exp_name = str(cfg.get("exp_name", "experiment"))

    setup_logging(cfg.get("log_level", "INFO"), log_file=results_dir / f"{exp_name}.log")

    LOGGER.info("=" * 78)
    LOGGER.info("FedChain :: %s", exp_name)
    LOGGER.info("%s", cfg.get("exp_description", ""))
    LOGGER.info("Config: %s", cfg.get("config_path"))
    LOGGER.info("=" * 78)

    seed = set_global_seed(int(cfg.get("seed", 42)), bool(cfg.get("deterministic", False)))
    device, device_info = get_device(cfg.get("device", "auto"))
    LOGGER.info("Seed=%d | device=%s", seed, device)

    if cfg.get("dry_run", False):
        LOGGER.warning(
            "DRY RUN: synthetic adapters are used and no base model is loaded. "
            "Accuracy metrics are placeholders; systems metrics are real."
        )

    blockchain_logger = build_blockchain_logger(cfg, force_mock=args.force_mock_chain)
    ipfs_manager = build_ipfs_manager(cfg, force_mock=args.force_mock_ipfs)
    evaluator = build_evaluator(cfg)

    overall_start = time.perf_counter()
    final_eval: Optional[Dict[str, Any]] = None

    try:
        if cfg.get("enable_fl", False):
            run_summary = run_federated(cfg, evaluator, blockchain_logger, ipfs_manager)
        else:
            run_summary = run_centralized(cfg, evaluator)

        # ---- final scoring of the global adapter --------------------------
        if evaluator is not None and cfg.get("eval_final", True):
            adapter_path = run_summary.get("global_adapter_path")
            if adapter_path:
                LOGGER.info("Final evaluation of %s ...", adapter_path)
                free_cuda_memory()
                try:
                    final_eval = evaluator.evaluate(adapter_path, label="final")
                except Exception as exc:
                    LOGGER.error("Final evaluation failed: %s", exc)
                    final_eval = None

        total_wall_clock = time.perf_counter() - overall_start
        metrics = collect_metrics(cfg, run_summary, final_eval, blockchain_logger, ipfs_manager)
        log_metrics(metrics)

        context = {
            "paradigm": run_summary.get("paradigm"),
            "num_rounds": run_summary.get("num_rounds"),
            "num_clients": run_summary.get("num_clients"),
            "device": device,
            "blockchain_mode": blockchain_logger.mode if blockchain_logger else None,
            "ipfs_backend": ipfs_manager.backend if ipfs_manager else None,
            "total_wall_clock_sec": round(total_wall_clock, 4),
        }

        report = build_report(
            cfg, metrics, context, run_summary, final_eval, blockchain_logger, ipfs_manager, device_info
        )

        json_path = results_dir / f"{exp_name}_metrics.json"
        write_json(json_path, report)
        LOGGER.info("Detailed JSON report written to %s", json_path)

        if blockchain_logger is not None:
            blockchain_logger.export_audit_trail(results_dir / f"{exp_name}_audit_trail.json")
        if ipfs_manager is not None:
            ipfs_manager.export_transfer_log(results_dir / f"{exp_name}_ipfs_transfers.json")

        print(render_markdown(cfg, metrics, context))
        return report

    finally:
        if evaluator is not None:
            evaluator.cleanup()
        if ipfs_manager is not None:
            ipfs_manager.cleanup_staging()
        if blockchain_logger is not None:
            blockchain_logger.close()
        free_cuda_memory()


def build_report(
    cfg: Any,
    metrics: Dict[str, Any],
    context: Dict[str, Any],
    run_summary: Dict[str, Any],
    final_eval: Optional[Dict[str, Any]],
    blockchain_logger: Optional[BlockchainLogger],
    ipfs_manager: Optional[IPFSManager],
    device_info: Dict[str, Any],
) -> Dict[str, Any]:
    """Assemble the full JSON report."""
    return {
        "experiment": {
            "name": cfg.get("exp_name"),
            "description": cfg.get("exp_description"),
            "config_path": cfg.get("config_path"),
            "paradigm": context.get("paradigm"),
            "enable_fl": bool(cfg.get("enable_fl", False)),
            "enable_blockchain": bool(cfg.get("enable_blockchain", False)),
            "enable_ipfs": bool(cfg.get("enable_ipfs", False)),
            "dry_run": bool(cfg.get("dry_run", False)),
            "timestamp_unix": int(time.time()),
        },
        "metrics": metrics,
        "context": context,
        "evaluation_detail": final_eval or run_summary.get("last_round_evaluation"),
        "run_summary": {k: v for k, v in run_summary.items() if k != "rounds"},
        "rounds": run_summary.get("rounds", []),
        "blockchain": blockchain_logger.get_metrics_summary() if blockchain_logger else None,
        "ipfs": ipfs_manager.get_metrics_summary() if ipfs_manager else None,
        "config": cfg.to_dict() if hasattr(cfg, "to_dict") else dict(cfg),
        "environment": {**describe_environment(), "device": device_info},
    }


def main(argv: Optional[List[str]] = None) -> int:
    args = build_arg_parser().parse_args(argv)
    # Bootstrap logging before the config is parsed so early failures are visible.
    setup_logging(args.log_level or "INFO")
    try:
        run_experiment(args)
        return 0
    except KeyboardInterrupt:
        LOGGER.warning("Interrupted by the user.")
        return 130
    except FileNotFoundError as exc:
        LOGGER.error("%s", exc)
        return 2
    except Exception as exc:
        LOGGER.exception("Experiment failed: %s", exc)
        return 1


if __name__ == "__main__":
    sys.exit(main())
