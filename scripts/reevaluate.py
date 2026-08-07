#!/usr/bin/env python
"""Re-score a finished adapter without retraining it.

Why this exists
---------------
Generation metrics are the expensive part of evaluation (~4.4 s per prompt on a
T600) and the cheap part of the *experiment*: they depend only on the finished
adapter, not on how it was produced. The baseline sweep scored 50 prompts, which
gives 95% confidence intervals spanning 5-17% of the reported value - enough to
show generation did not collapse, not enough to support any comparison between
arms.

Raising `gen_num_samples` and re-running the sweep would cost another 24 GPU-
hours to recompute numbers that are already correct. This script re-scores the
adapters that already exist, so tightening ROUGE-L / BLEU costs one evaluation
pass instead of one training run.

Typical use
-----------
Re-score one adapter at 250 prompts::

    python scripts/reevaluate.py \\
        --adapter outputs/smollm2-360m/seed_42/exp4_fedchain/round_3/global \\
        --config configs/exp4_fedchain.yaml --model smol \\
        --gen-num-samples 250

Re-score every final adapter of a sweep and write one JSON per arm::

    python scripts/reevaluate.py --sweep results/smollm2-360m \\
        --outputs-root outputs/smollm2-360m --model smol \\
        --gen-num-samples 250 --require-backend evaluate

`--require-backend evaluate` makes a missing `nltk` / `evaluate` install a hard
error rather than a silent fall back to the built-in implementation. Use it for
anything headed into a table: the baseline sweep fell back silently on all 18
runs, and the absolute ROUGE-L values it produced are not comparable with
published numbers.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evaluation.eval_loss import Evaluator  # noqa: E402
from utils.config import load_config  # noqa: E402
from utils.models import resolve_model  # noqa: E402

LOGGER = logging.getLogger("reevaluate")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Re-score finished adapters without retraining.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    target = parser.add_mutually_exclusive_group(required=True)
    target.add_argument(
        "--adapter",
        action="append",
        default=None,
        metavar="PATH",
        help="Adapter directory to score. Repeatable.",
    )
    target.add_argument(
        "--sweep",
        metavar="RESULTS_DIR",
        help=(
            "Score the final adapter of every *_metrics.json under RESULTS_DIR "
            "(recursively, so seed_*/ is picked up). Adapter paths are read from "
            "each report's run_summary."
        ),
    )

    parser.add_argument(
        "--config",
        default="configs/exp4_fedchain.yaml",
        help="Config supplying the evaluation settings (dataset, seq len, model).",
    )
    parser.add_argument(
        "--model",
        default=None,
        metavar="TIER_OR_ID",
        help="Model tier or HF id. Must match the model the adapter was trained on.",
    )
    parser.add_argument(
        "--gen-num-samples",
        type=int,
        default=250,
        help="Prompts decoded for ROUGE-L / BLEU (default: 250).",
    )
    parser.add_argument(
        "--eval-num-samples",
        type=int,
        default=None,
        help="Override the loss-sample count (default: whatever the config says).",
    )
    parser.add_argument(
        "--require-backend",
        default=None,
        choices=["evaluate", "builtin"],
        help=(
            "Fail instead of falling back when `evaluate` is unusable. Use "
            "'evaluate' for any number going into a table."
        ),
    )
    parser.add_argument(
        "--out",
        default=None,
        help="Write JSON here. With --sweep, a directory (one file per arm).",
    )
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default=None)
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING"])
    return parser


def discover_sweep_adapters(results_dir: Path) -> List[Dict[str, Any]]:
    """Find (arm, seed, adapter) triples from the metrics reports of a sweep.

    Reads the adapter path out of each report rather than guessing the layout,
    so a re-scored number is always traceable to the run that produced it.
    Local-only reports have no single global adapter - every client is its own
    model - so each client adapter is returned separately and the caller
    averages them, exactly as the training run does.
    """
    targets: List[Dict[str, Any]] = []
    for report_path in sorted(results_dir.rglob("*_metrics.json")):
        try:
            report = json.loads(report_path.read_text(encoding="utf-8"))
        except Exception as exc:
            LOGGER.warning("Skipping unreadable report %s: %s", report_path, exc)
            continue

        arm = report.get("experiment", {}).get("name") or report_path.stem
        seed = report.get("metrics", {}).get("seed")
        summary = report.get("run_summary") or {}

        if summary.get("aggregation_enabled") is False:
            rounds = report.get("rounds") or []
            if not rounds:
                continue
            for client in rounds[-1].get("clients", []):
                targets.append(
                    {
                        "arm": arm,
                        "seed": seed,
                        "client_id": client.get("client_id"),
                        "adapter": client.get("adapter_path"),
                        "report": str(report_path),
                    }
                )
            continue

        adapter = summary.get("global_adapter_path") or (
            report.get("evaluation_detail") or {}
        ).get("adapter_path")
        if adapter:
            targets.append(
                {"arm": arm, "seed": seed, "client_id": None,
                 "adapter": adapter, "report": str(report_path)}
            )
    return targets


def main() -> int:
    args = build_parser().parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s | %(levelname)-8s | %(name)-24s | %(message)s",
        datefmt="%H:%M:%S",
    )

    overrides: Dict[str, Any] = {
        "gen_num_samples": args.gen_num_samples,
        "enable_generation_metrics": args.gen_num_samples > 0,
    }
    if args.eval_num_samples is not None:
        overrides["eval_num_samples"] = args.eval_num_samples
    if args.device:
        overrides["device"] = args.device
    if args.require_backend == "evaluate":
        overrides["require_metric_backend"] = "evaluate"
    model_spec = resolve_model(args.model) if args.model else None
    if model_spec is not None:
        overrides["model_name"] = model_spec.hf_id
    # Never resume or write training artefacts from a scoring pass.
    overrides["dry_run"] = False

    config = load_config(args.config, overrides=overrides)

    if args.sweep:
        targets = discover_sweep_adapters(Path(args.sweep))
        if not targets:
            LOGGER.error("No adapters discovered under %s", args.sweep)
            return 1
    else:
        targets = [
            {"arm": None, "seed": None, "client_id": None, "adapter": a, "report": None}
            for a in args.adapter
        ]

    LOGGER.info(
        "Re-scoring %d adapter(s) at %d generation prompt(s). Model: %s",
        len(targets),
        args.gen_num_samples,
        config.get("model_name"),
    )

    evaluator = Evaluator(config)
    results: List[Dict[str, Any]] = []
    started = time.perf_counter()

    for index, target in enumerate(targets, start=1):
        adapter = target["adapter"]
        if not adapter or not Path(adapter).exists():
            LOGGER.error(
                "[%d/%d] adapter missing on disk: %s "
                "(outputs/ may have been pruned; re-run the arm or restore it)",
                index, len(targets), adapter,
            )
            results.append({**target, "error": "adapter not found"})
            continue

        label = target.get("arm") or Path(adapter).name
        if target.get("client_id"):
            label = f"{label}:{target['client_id']}"
        LOGGER.info("[%d/%d] %s", index, len(targets), label)

        try:
            scores = evaluator.evaluate(adapter, label=label)
        except Exception as exc:
            LOGGER.error("Evaluation failed for %s: %s", adapter, exc)
            results.append({**target, "error": str(exc)})
            continue
        results.append({**target, **scores})

    payload = {
        "experiment": {
            "name": "reevaluate",
            "description": "Generation metrics recomputed on finished adapters",
            "timestamp_unix": int(time.time()),
        },
        "context": {
            "config": args.config,
            "model_name": config.get("model_name"),
            "gen_num_samples": args.gen_num_samples,
            "eval_num_samples": config.get("eval_num_samples"),
            "require_metric_backend": config.get("require_metric_backend") or None,
            "total_latency_sec": round(time.perf_counter() - started, 2),
        },
        "results": results,
    }

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        LOGGER.info("Written: %s", out_path)
    else:
        print(json.dumps(payload, indent=2))

    backends = {r.get("generation_metric_backend") for r in results if r.get("generation_metric_backend")}
    if backends:
        LOGGER.info("Generation metric backend(s) used: %s", ", ".join(sorted(map(str, backends))))
        if "builtin" in backends and args.require_backend != "evaluate":
            LOGGER.warning(
                "At least one score used the built-in implementation. Absolute "
                "values are not comparable with published `evaluate` numbers - "
                "re-run with --require-backend evaluate for a paper table."
            )

    return 0 if any("error" not in r for r in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
