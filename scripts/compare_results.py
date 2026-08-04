#!/usr/bin/env python
"""
FedChain :: Results aggregator
==============================

Reads every ``results/<exp_name>_metrics.json`` produced by ``main.py`` and
emits the cross-experiment comparison the paper reports:

* ``results/comparison.md``  - Markdown table (paste straight into the paper)
* ``results/comparison.csv`` - same numbers, for plotting
* stdout                     - the Markdown table

Also surfaces the **overhead deltas** relative to the centralized baseline, and
flags any experiment whose blockchain or IPFS layer silently ran in mock mode -
those rows carry synthetic systems numbers and must not be reported as measured.

    python scripts/compare_results.py
    python scripts/compare_results.py --results-dir results --order exp1_sft,exp2_fl
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

#: Canonical experiment order for the comparison table.
DEFAULT_ORDER: Tuple[str, ...] = ("exp1_sft", "exp2_fl", "exp3_fl_bc", "exp4_fedchain")

SHORT_LABELS: Dict[str, str] = {
    "exp1_sft": "E1: Centralized SFT",
    "exp2_fl": "E2: FedAvg",
    "exp3_fl_bc": "E3: FL + Blockchain",
    "exp4_fedchain": "E4: FedChain",
}


def _load_schema() -> Tuple[Tuple[str, str, str], ...]:
    """Reuse main.py's schema so this script can never drift from the report."""
    try:
        from main import METRIC_SCHEMA  # type: ignore

        return tuple(METRIC_SCHEMA)
    except Exception:
        # Standalone fallback (e.g. pyyaml missing); keep in sync with main.py.
        return (
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


METRIC_SCHEMA = _load_schema()


def format_value(value: Any, kind: str) -> str:
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


def markdown_table(headers: Sequence[str], rows: Sequence[Sequence[str]]) -> str:
    headers = [str(h) for h in headers]
    str_rows = [[("" if c is None else str(c)) for c in row] for row in rows]
    widths = [len(h) for h in headers]
    for row in str_rows:
        for idx, cell in enumerate(row):
            if idx < len(widths):
                widths[idx] = max(widths[idx], len(cell))

    def line(cells: Sequence[str]) -> str:
        return "| " + " | ".join(str(c).ljust(widths[i]) for i, c in enumerate(cells)) + " |"

    out = [line(headers), "|" + "|".join("-" * (w + 2) for w in widths) + "|"]
    out.extend(line(row) for row in str_rows)
    return "\n".join(out)


def discover_reports(results_dir: Path, order: Sequence[str]) -> List[Tuple[str, Dict[str, Any]]]:
    """Load reports in the requested order, then any extras alphabetically."""
    found: Dict[str, Dict[str, Any]] = {}
    for path in sorted(results_dir.glob("*_metrics.json")):
        name = path.name[: -len("_metrics.json")]
        try:
            with open(path, "r", encoding="utf-8") as handle:
                found[name] = json.load(handle)
        except Exception as exc:
            print(f"WARNING: could not read {path.name}: {exc}", file=sys.stderr)

    ordered: List[Tuple[str, Dict[str, Any]]] = []
    for name in order:
        if name in found:
            ordered.append((name, found.pop(name)))
    ordered.extend(sorted(found.items()))
    return ordered


def integrity_warnings(name: str, report: Dict[str, Any]) -> List[str]:
    """Flag runs whose numbers are synthetic or otherwise not comparable."""
    warnings: List[str] = []
    experiment = report.get("experiment", {}) or {}
    chain = report.get("blockchain") or {}
    ipfs = report.get("ipfs") or {}

    if experiment.get("dry_run"):
        warnings.append(f"{name}: DRY RUN - accuracy metrics are placeholders, not measurements.")
    if experiment.get("enable_blockchain") and chain.get("mode") == "mock":
        warnings.append(
            f"{name}: blockchain ran in MOCK mode - gas and tx latency are modelled, not measured."
        )
    if experiment.get("enable_ipfs") and ipfs.get("backend") == "mock":
        warnings.append(
            f"{name}: IPFS ran in MOCK mode - transfer latency reflects local disk, not a network."
        )
    if chain.get("num_failed"):
        warnings.append(f"{name}: {chain['num_failed']} blockchain transaction(s) failed.")
    if ipfs.get("num_failed_transfers"):
        warnings.append(f"{name}: {ipfs['num_failed_transfers']} IPFS transfer(s) failed.")

    summary = report.get("run_summary", {}) or {}
    total = summary.get("integrity_checks_total")
    passed = summary.get("integrity_checks_passed")
    if total is not None and passed is not None and passed != total:
        warnings.append(f"{name}: only {passed}/{total} integrity checks passed.")
    return warnings


def build_overhead_rows(
    reports: List[Tuple[str, Dict[str, Any]]], baseline_key: str
) -> List[List[str]]:
    """Deltas versus the centralized baseline: the paper's 'cost of X' story."""
    baseline: Optional[Dict[str, Any]] = None
    for name, report in reports:
        if name == baseline_key:
            baseline = report.get("metrics", {})
            break
    if not baseline:
        return []

    rows: List[List[str]] = []
    for name, report in reports:
        if name == baseline_key:
            continue
        metrics = report.get("metrics", {}) or {}

        def delta(key: str) -> str:
            current, base = metrics.get(key), baseline.get(key)
            if current is None or base is None:
                return "n/a"
            diff = float(current) - float(base)
            if abs(float(base)) > 1e-12:
                return f"{diff:+.4f} ({diff / float(base) * 100:+.1f}%)"
            return f"{diff:+.4f}"

        def absolute(key: str, kind: str) -> str:
            return format_value(metrics.get(key), kind)

        rows.append(
            [
                SHORT_LABELS.get(name, name),
                delta("validation_loss"),
                delta("perplexity"),
                absolute("communication_volume_mb", "float3"),
                absolute("blockchain_gas_used", "int"),
                absolute("end_to_end_round_duration_sec", "float2"),
            ]
        )
    return rows


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Aggregate FedChain experiment results.")
    parser.add_argument("--results-dir", default=str(PROJECT_ROOT / "results"))
    parser.add_argument(
        "--order",
        default=",".join(DEFAULT_ORDER),
        help="Comma-separated experiment order for the table columns.",
    )
    parser.add_argument("--baseline", default="exp1_sft", help="Experiment used as the overhead baseline.")
    args = parser.parse_args(argv)

    results_dir = Path(args.results_dir)
    if not results_dir.exists():
        print(f"ERROR: results directory not found: {results_dir}", file=sys.stderr)
        return 2

    order = [s.strip() for s in args.order.split(",") if s.strip()]
    reports = discover_reports(results_dir, order)
    if not reports:
        print(f"ERROR: no *_metrics.json files in {results_dir}", file=sys.stderr)
        return 2

    names = [name for name, _ in reports]
    headers = ["Metric"] + [SHORT_LABELS.get(n, n) for n in names]

    rows: List[List[str]] = []
    for key, label, kind in METRIC_SCHEMA:
        row = [label]
        for _, report in reports:
            row.append(format_value((report.get("metrics") or {}).get(key), kind))
        rows.append(row)

    lines: List[str] = ["# FedChain - Experiment Comparison", ""]

    context_rows: List[List[str]] = []
    for field, label in (
        ("paradigm", "Paradigm"),
        ("num_rounds", "Rounds"),
        ("num_clients", "Clients"),
        ("device", "Device"),
        ("blockchain_mode", "Chain mode"),
        ("ipfs_backend", "IPFS backend"),
    ):
        row = [label]
        for _, report in reports:
            value = (report.get("context") or {}).get(field)
            row.append("-" if value in (None, "") else str(value))
        context_rows.append(row)

    model = None
    for _, report in reports:
        model = (report.get("config") or {}).get("model_name")
        if model:
            break
    if model:
        lines.append(f"**Model:** `{model}`")
        lines.append("")

    lines.append("## Run context")
    lines.append("")
    lines.append(markdown_table(headers, context_rows))
    lines.append("")
    lines.append("## Metrics")
    lines.append("")
    lines.append(markdown_table(headers, rows))
    lines.append("")

    overhead = build_overhead_rows(reports, args.baseline)
    if overhead:
        lines.append(f"## Overhead relative to `{args.baseline}`")
        lines.append("")
        lines.append(
            markdown_table(
                ["Experiment", "d Val. Loss", "d Perplexity", "Comm (MB)", "Gas", "Round (s)"],
                overhead,
            )
        )
        lines.append("")

    all_warnings: List[str] = []
    for name, report in reports:
        all_warnings.extend(integrity_warnings(name, report))
    if all_warnings:
        lines.append("## Warnings")
        lines.append("")
        for warning in all_warnings:
            lines.append(f"- {warning}")
        lines.append("")

    markdown = "\n".join(lines)
    md_path = results_dir / "comparison.md"
    md_path.write_text(markdown, encoding="utf-8")

    csv_path = results_dir / "comparison.csv"
    with open(csv_path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["metric"] + names)
        for key, label, _ in METRIC_SCHEMA:
            writer.writerow([key] + [(r.get("metrics") or {}).get(key) for _, r in reports])

    print(markdown)
    print(f"\nWritten: {md_path}")
    print(f"Written: {csv_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
