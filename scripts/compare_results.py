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
DEFAULT_ORDER: Tuple[str, ...] = (
    "exp0_local",
    "exp1_sft",
    "exp2_fl",
    "exp3_fl_bc",
    "exp4_fedchain",
    "exp5_noniid",
)

SHORT_LABELS: Dict[str, str] = {
    "exp0_local": "E0: Local-only",
    "exp1_sft": "E1: Centralized SFT",
    "exp2_fl": "E2: FedAvg",
    "exp3_fl_bc": "E3: FL + Blockchain",
    "exp4_fedchain": "E4: FedChain",
    "exp5_noniid": "E5: FedChain non-IID",
    # Ablation arms. B1 supplies the matched non-IID baselines that E5 lacks, so
    # its labels mirror E0/E1/E2 to make the paired tables read the same way.
    "ablationB_e0_noniid": "B1-E0: Local-only (non-IID)",
    "ablationB_e1_noniid": "B1-E1: Centralized (non-IID)",
    "ablationB_e2_noniid": "B1-E2: FedAvg (non-IID)",
    "ablationA_e0_rounds9": "A2: Local-only @ R=9",
    "ablationA_e2_rounds9": "A1: FedAvg @ R=9",
    "ablationA_e1_budget9": "A3: Centralized @ R=9 budget",
    "ablationA_e1_budget5": "A3: Centralized @ R=5 budget",
    "ablationD1_no_global_anchor": "D1: no global anchor",
    "ablationD2_no_roundtrip": "D2: no IPFS round-trip",
    "ablationD3_no_verify": "D3: no download verify",
}

#: Reports produced by the audit-layer scripts rather than a training run. They
#: carry none of METRIC_SCHEMA's keys, so listing them as table rows prints a
#: full line of "n/a" that reads as a failed run. They get their own section.
AUDIT_ONLY_EXPERIMENTS: Tuple[str, ...] = ("exp6_tamper", "exp7_scalability")

#: Tier directories kept for provenance but excluded from the ladder table.
#: `qwen-0.5b.leaky_backup` holds the VRAM-contaminated sweep whose timing
#: metrics are unusable (see ablation_study/10_two_model_results.md); rendering
#: it beside the clean tier invites a reader to average the two or quote the
#: wrong one. Keeping the directory but excluding it from the table is what
#: makes "do not quote this" enforceable rather than a note someone must recall.
EXCLUDED_TIER_MARKERS: Tuple[str, ...] = ("backup", "archive", "_old", "scratch")


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
            ("end_to_end_round_duration_sec", "Mean Round Duration (s)", "float2"),
            ("total_round_time_sec", "Total Round Time (s)", "float2"),
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


def format_aggregate(values: Sequence[Any], kind: str) -> str:
    """Format a metric across seeds as ``mean +- CI``, or plainly when it cannot vary.

    A single seed, or a metric that is byte-identical across seeds (gas and
    adapter size are deterministic given the config), renders as a bare value:
    printing "3,785,372 +- 0" adds noise and invites the reader to think there
    is sampling error where there is none.
    """
    numeric = [float(v) for v in values if v is not None]
    if not numeric:
        return "n/a"
    if len(numeric) == 1 or max(numeric) == min(numeric):
        return format_value(numeric[0], kind)
    stats = summarise_values(numeric)
    if stats is None:
        return "n/a"
    return f"{format_value(stats['mean'], kind)} +- {format_value(stats['ci95'], kind)}"


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


# =============================================================================
# Multi-seed aggregation
# =============================================================================
#: Two-sided 95% critical values of Student's t, indexed by degrees of freedom.
#: A seed sweep has 2-9 degrees of freedom, where the normal approximation
#: (1.96) understates the interval by 20-300% - at df=2 the true value is 4.303.
#: Reporting +-1.96 sigma from three seeds would overstate significance badly,
#: which is exactly the error a reviewer checks for.
_T_CRITICAL_95: Dict[int, float] = {
    1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365,
    8: 2.306, 9: 2.262, 10: 2.228, 12: 2.179, 15: 2.131, 20: 2.086, 30: 2.042,
}


def t_critical_95(degrees_of_freedom: int) -> float:
    """Two-sided 95% t critical value, conservative for unlisted df."""
    if degrees_of_freedom < 1:
        return float("nan")
    for df in sorted(_T_CRITICAL_95):
        if degrees_of_freedom <= df:
            return _T_CRITICAL_95[df]
    return 1.96


def summarise_values(values: Sequence[float]) -> Optional[Dict[str, float]]:
    """Mean, sample standard deviation and 95% CI half-width for a metric."""
    numeric = [float(v) for v in values if v is not None]
    if not numeric:
        return None
    n = len(numeric)
    mean = sum(numeric) / n
    if n == 1:
        return {"mean": mean, "std": 0.0, "n": 1, "ci95": float("nan")}
    variance = sum((v - mean) ** 2 for v in numeric) / (n - 1)
    std = variance ** 0.5
    return {
        "mean": mean,
        "std": std,
        "n": n,
        "ci95": t_critical_95(n - 1) * std / (n ** 0.5),
    }


def paired_delta(
    treatment: Sequence[float], baseline: Sequence[float]
) -> Optional[Dict[str, Any]]:
    """Paired per-seed difference (treatment - baseline) with a 95% CI.

    Pairing by seed is what makes 3-5 runs informative: seed-to-seed variation
    is shared by both arms and cancels, so the CI is over the *effect* rather
    than over the noisy absolute scores. If the interval spans zero, the
    experiment has not shown a difference and the paper must not claim one.
    """
    pairs = [
        (float(t), float(b))
        for t, b in zip(treatment, baseline)
        if t is not None and b is not None
    ]
    if len(pairs) < 2:
        return None
    differences = [t - b for t, b in pairs]
    stats = summarise_values(differences)
    if stats is None:
        return None
    significant = abs(stats["mean"]) > stats["ci95"]
    return {
        "mean_difference": stats["mean"],
        "ci95": stats["ci95"],
        "n_pairs": len(pairs),
        "significant_at_95": bool(significant),
    }


def discover_seed_reports(
    results_dir: Path, order: Sequence[str]
) -> Dict[str, List[Tuple[int, Dict[str, Any]]]]:
    """Collect ``seed_*/`` subdirectories into {experiment: [(seed, report)]}.

    Falls back to treating ``results_dir`` itself as one seed, so a single-seed
    tree still renders (with n=1 and no confidence intervals, which is the
    honest presentation of one run).
    """
    by_experiment: Dict[str, List[Tuple[int, Dict[str, Any]]]] = {}
    seed_dirs = sorted(p for p in results_dir.glob("seed_*") if p.is_dir())
    if not seed_dirs:
        seed_dirs = [results_dir]

    for seed_dir in seed_dirs:
        try:
            seed = int(seed_dir.name.split("_", 1)[1])
        except (IndexError, ValueError):
            seed = -1
        for name, report in discover_reports(seed_dir, order):
            actual = (report.get("metrics") or {}).get("seed")
            by_experiment.setdefault(name, []).append(
                (int(actual) if actual is not None else seed, report)
            )
    return by_experiment


def build_seed_tables(
    by_experiment: Dict[str, List[Tuple[int, Dict[str, Any]]]],
    order: Sequence[str],
    baseline_key: str,
    extra_baselines: Sequence[str] = (),
) -> List[str]:
    """Mean +- CI table over seeds, plus paired deltas against each baseline.

    More than one baseline matters here. Pairing everything against the
    centralized arm answers "what does federating cost?", but the question a
    reviewer asks first is "what does federating *buy*?" - and that needs the
    same experiments paired against the local-only arm instead. Reporting only
    the first pairing is how a 2.4%-of-the-gap result stays invisible.
    """
    names = [n for n in order if n in by_experiment]
    names.extend(sorted(n for n in by_experiment if n not in names))
    if not names:
        return []

    def values_for(name: str, key: str) -> List[float]:
        return [
            (report.get("metrics") or {}).get(key)
            for _, report in sorted(by_experiment[name], key=lambda pair: pair[0])
        ]

    # Accuracy metrics are the only ones a seed sweep is about; systems metrics
    # are deterministic given the configuration and are reported once.
    accuracy_keys = [
        ("validation_loss", "Validation Loss", 4),
        ("perplexity", "Perplexity", 4),
        ("rouge_l", "ROUGE-L", 4),
        ("bleu", "BLEU", 4),
    ]

    lines = ["## Accuracy across seeds (mean +- 95% CI)", ""]
    seed_counts = {name: len(by_experiment[name]) for name in names}
    lines.append(
        "_Seeds per experiment: "
        + ", ".join(f"{SHORT_LABELS.get(n, n)}={seed_counts[n]}" for n in names)
        + "._"
    )
    lines.append("")

    rows: List[List[str]] = []
    for key, label, places in accuracy_keys:
        row = [label]
        for name in names:
            stats = summarise_values(values_for(name, key))
            if stats is None:
                row.append("n/a")
            elif stats["n"] == 1:
                row.append(f"{stats['mean']:.{places}f} (n=1)")
            else:
                row.append(f"{stats['mean']:.{places}f} +- {stats['ci95']:.{places}f}")
        rows.append(row)
    lines.append(markdown_table(["Metric"] + [SHORT_LABELS.get(n, n) for n in names], rows))
    lines.append("")

    seen_baselines: List[str] = []
    for base in [baseline_key, *extra_baselines]:
        if base in seen_baselines or base not in by_experiment:
            continue
        seen_baselines.append(base)

        delta_rows: List[List[str]] = []
        for name in names:
            if name == base:
                continue
            for key, label, places in accuracy_keys[:2]:
                delta = paired_delta(values_for(name, key), values_for(base, key))
                if delta is None:
                    continue
                delta_rows.append(
                    [
                        SHORT_LABELS.get(name, name),
                        label,
                        f"{delta['mean_difference']:+.{places}f}",
                        f"+-{delta['ci95']:.{places}f}",
                        str(delta["n_pairs"]),
                        "yes" if delta["significant_at_95"] else "no",
                    ]
                )
        if not delta_rows:
            continue

        lines.append(f"## Paired difference vs `{base}` (per seed)")
        lines.append("")
        lines.append(
            markdown_table(
                ["Experiment", "Metric", "Mean diff", "95% CI", "Seeds", "Significant"],
                delta_rows,
            )
        )
        lines.append("")
        lines.append(
            "_'Significant' means the 95% CI of the paired difference excludes zero. "
            "A 'no' is a real result - it says the audit layer changed nothing "
            "measurable, which is the claim these experiments exist to support._"
        )
        if base == "exp0_local":
            lines.append("")
            lines.append(
                "_Against the local-only arm a *negative* difference means "
                "aggregation helped. Read it next to the same arm's distance from "
                "the centralized bound: a difference that is significant but a "
                "small fraction of that distance says federation is measurable, "
                "not that it is worthwhile._"
            )
        lines.append("")

    return lines


def build_trajectory_table(
    by_experiment: Dict[str, List[Tuple[int, Dict[str, Any]]]],
    order: Sequence[str],
) -> List[str]:
    """Per-round validation loss, averaged over seeds.

    The end-point alone cannot distinguish "converged" from "ran out of budget",
    and those license opposite conclusions. A curve still descending at the last
    round means the round count is a free parameter the result depends on.
    """
    names = [n for n in order if n in by_experiment]
    names.extend(sorted(n for n in by_experiment if n not in names))

    # round -> experiment -> [loss per seed]
    trajectory: Dict[int, Dict[str, List[float]]] = {}
    for name in names:
        for _, report in by_experiment[name]:
            for entry in report.get("rounds") or []:
                evaluation = entry.get("evaluation")
                if not evaluation or evaluation.get("loss") is None:
                    continue
                index = int(entry.get("round", 0))
                trajectory.setdefault(index, {}).setdefault(name, []).append(
                    float(evaluation["loss"])
                )

    if not trajectory:
        return []

    present = [n for n in names if any(n in per_round for per_round in trajectory.values())]
    rows: List[List[str]] = []
    for index in sorted(trajectory):
        row = [str(index)]
        for name in present:
            values = trajectory[index].get(name) or []
            row.append(f"{sum(values) / len(values):.4f}" if values else "-")
        rows.append(row)

    # Final-round scores live in `metrics`, not `rounds`: the last round is
    # deferred to the final scoring pass so the forward pass is not paid twice.
    final_row = ["final"]
    for name in present:
        finals = [
            (report.get("metrics") or {}).get("validation_loss")
            for _, report in by_experiment[name]
        ]
        finals = [float(v) for v in finals if v is not None]
        final_row.append(f"{sum(finals) / len(finals):.4f}" if finals else "-")
    rows.append(final_row)

    lines = ["## Validation loss by round (mean over seeds)", ""]
    lines.append(markdown_table(["Round"] + [SHORT_LABELS.get(n, n) for n in present], rows))
    lines.append("")
    lines.append(
        "_A curve still descending at the final round means the round count was "
        "budget-limited, not converged - any 'cost of federation' read off that "
        "end-point is a statement about the budget as much as about federation. "
        "Arms with no per-round rows evaluate only at the end; set "
        "`eval_local_clients_every_round: true` to give the local-only arm a curve._"
    )
    lines.append("")
    return lines


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
                absolute("total_round_time_sec", "float2"),
            ]
        )
    return rows


def build_across_models(results_root: Path, order: Sequence[str]) -> Optional[str]:
    """Table spanning every model tier that has results under ``results_root``.

    Each tier writes to ``results/<model_key>/``, so a ladder run
    (SmolLM2-360M -> Qwen2.5-0.5B -> Qwen2.5-1.5B) leaves one subdirectory per
    rung. This renders them as one table so the effect of model size on both
    accuracy and systems overhead is visible at a glance.
    """
    tiers: List[
        Tuple[
            str,
            str,
            List[Tuple[str, Dict[str, Any]]],
            Dict[str, List[Tuple[int, Dict[str, Any]]]],
            List[Tuple[str, Dict[str, Any]]],
        ]
    ] = []
    excluded: List[str] = []
    for sub in sorted(p for p in results_root.iterdir() if p.is_dir() and p.name != "logs"):
        if any(marker in sub.name.lower() for marker in EXCLUDED_TIER_MARKERS):
            excluded.append(sub.name)
            continue
        # A seed sweep writes the training runs to <tier>/seed_<N>/ and leaves
        # only the audit-layer reports at the tier root. Scanning the root alone
        # therefore finds exp6/exp7 and none of the experiments the table is
        # about, producing a full page of "n/a". Merge both layouts, preferring
        # the lowest seed as the representative run (deterministic, and the same
        # choice the per-tier report makes).
        reports = discover_reports(sub, order)
        audit = [(n, r) for n, r in reports if n in AUDIT_ONLY_EXPERIMENTS]
        reports = [(n, r) for n, r in reports if n not in AUDIT_ONLY_EXPERIMENTS]
        by_seed = discover_seed_reports(sub, order) if any(sub.glob("seed_*")) else {}
        by_seed = {n: runs for n, runs in by_seed.items() if n not in AUDIT_ONLY_EXPERIMENTS}
        if by_seed:
            seeded = {
                name: sorted(runs, key=lambda pair: pair[0])[0][1]
                for name, runs in by_seed.items()
            }
            merged = {name: report for name, report in reports}
            seeded.update({k: v for k, v in merged.items() if k not in seeded})
            ordered = [n for n in order if n in seeded]
            ordered.extend(sorted(n for n in seeded if n not in ordered))
            reports = [(n, seeded[n]) for n in ordered]
        if not reports and not audit:
            continue
        model_name = ""
        for _, report in reports + audit:
            model_name = (report.get("experiment") or {}).get("model_name") or (
                report.get("config") or {}
            ).get("model_name", "")
            if model_name:
                break
        tiers.append((sub.name, model_name, reports, by_seed, audit))

    if not tiers:
        return None

    lines: List[str] = ["# FedChain - Model Ladder Comparison", ""]
    lines.append(
        "Each row is one (model tier, experiment) pair. Tiers are separate runs "
        "with identical hyperparameters, so accuracy differences reflect model "
        "capacity and systems differences reflect adapter size."
    )
    lines.append("")

    lines.append("## Tiers")
    lines.append("")
    lines.append(
        markdown_table(
            ["Tier", "Model", "Seeds", "Experiments with results"],
            [
                [
                    key,
                    name or "-",
                    str(max((len(r) for r in by_seed.values()), default=1)),
                    ", ".join(n for n, _ in reports + audit),
                ]
                for key, name, reports, by_seed, audit in tiers
            ],
        )
    )
    lines.append("")

    columns = [
        ("validation_loss", "Val Loss", "float4"),
        ("perplexity", "PPL", "float4"),
        ("rouge_l", "ROUGE-L", "float4"),
        ("bleu", "BLEU", "float4"),
        ("training_time_sec", "Train (s)", "float2"),
        ("communication_volume_mb", "Comm (MB)", "float3"),
        ("adapter_size_mb", "Adapter (MB)", "float3"),
        ("blockchain_gas_used", "Gas", "int"),
        ("total_round_time_sec", "Total time (s)", "float2"),
    ]

    rows: List[List[str]] = []
    warnings: List[str] = []
    for tier_key, _, reports, by_seed, _ in tiers:
        for exp_name, report in reports:
            runs = by_seed.get(exp_name)
            if runs:
                # Aggregate over every seed rather than quoting the lowest one.
                # A point value from one seed cannot be told apart from noise,
                # and this table is the one most likely to be read in isolation.
                metric_sets = [(r.get("metrics") or {}) for _, r in runs]
            else:
                metric_sets = [report.get("metrics") or {}]
            rows.append(
                [tier_key, SHORT_LABELS.get(exp_name, exp_name), str(len(metric_sets))]
                + [
                    format_aggregate([m.get(key) for m in metric_sets], kind)
                    for key, _, kind in columns
                ]
            )
            for _, run_report in (runs or [(0, report)]):
                warnings.extend(
                    f"{tier_key}/{w}" for w in integrity_warnings(exp_name, run_report)
                )

    lines.append("## Metrics")
    lines.append("")
    lines.append(
        markdown_table(
            ["Tier", "Experiment", "n"] + [label for _, label, _ in columns], rows
        )
    )
    lines.append("")
    lines.append(
        "_`n` is the number of seeds. Values are mean +- 95% CI (Student's t) "
        "where they vary across seeds, and a bare figure where they are "
        "deterministic given the config. Quote the interval, not the mean alone._"
    )
    lines.append("")

    if excluded:
        lines.append(
            "_Excluded from this table (kept on disk for provenance, not evidence): "
            + ", ".join(f"`{name}`" for name in excluded)
            + "._"
        )
        lines.append("")

    audit_section = build_audit_section(tiers)
    if audit_section:
        lines.extend(audit_section)

    if warnings:
        lines.append("## Warnings")
        lines.append("")
        lines.extend(f"- {w}" for w in dict.fromkeys(warnings))
        lines.append("")

    return "\n".join(lines)


def build_audit_section(tiers: Sequence[Tuple[Any, ...]]) -> List[str]:
    """Render exp6/exp7 as their own tables instead of rows of ``n/a``.

    These are audit-layer experiments, not training runs: they carry detection
    rates and gas sweeps, none of METRIC_SCHEMA's keys. Rendered in the metrics
    table they print a full line of "n/a", which reads as a failed run when in
    fact they are the strongest systems evidence in the study.
    """
    tamper_rows: List[List[str]] = []
    scale_rows: List[List[str]] = []
    payload_rows: List[List[str]] = []

    for tier_key, _, _, _, audit in tiers:
        for exp_name, report in audit:
            if exp_name == "exp6_tamper":
                context = report.get("context") or {}
                for entry in report.get("summary") or []:
                    trials = entry.get("trials")
                    detected = entry.get("detected")
                    benign = entry.get("benign_control")
                    rate = entry.get("false_positive_rate" if benign else "detection_rate")
                    tamper_rows.append(
                        [
                            tier_key,
                            str(entry.get("attack", "-")),
                            "benign control" if benign else "attack",
                            f"{detected}/{trials}",
                            "-" if rate is None else f"{float(rate) * 100:.1f}%",
                            str(context.get("num_adapters", "-")),
                        ]
                    )
            elif exp_name == "exp7_scalability":
                for entry in report.get("clients_sweep") or []:
                    scale_rows.append(
                        [
                            tier_key,
                            str(entry.get("num_clients", "-")),
                            str(entry.get("transactions_per_round", "-")),
                            f"{int(entry.get('gas_per_round', 0)):,}",
                            f"{float(entry.get('gas_per_client', 0)):,.0f}",
                        ]
                    )
                for entry in report.get("payload_sweep") or []:
                    payload_rows.append(
                        [
                            tier_key,
                            str(entry.get("label", "-")),
                            f"{float(entry.get('adapter_mb', 0)):.4f}",
                            f"{int(entry.get('gas_used', 0)):,}",
                            str(entry.get("anchored_bytes", "-")),
                        ]
                    )

    if not (tamper_rows or scale_rows or payload_rows):
        return []

    lines = ["## Audit-layer experiments", ""]
    lines.append(
        "_Not training runs — they carry no loss or perplexity, so they are "
        "reported here rather than as blank rows above._"
    )
    lines.append("")
    if tamper_rows:
        lines.append("### E6 — tamper detection")
        lines.append("")
        lines.append(
            markdown_table(
                ["Tier", "Attack", "Kind", "Detected", "Rate", "Adapters"], tamper_rows
            )
        )
        lines.append("")
    if scale_rows:
        lines.append("### E7 — gas versus federation size")
        lines.append("")
        lines.append(
            markdown_table(
                ["Tier", "Clients", "Tx/round", "Gas/round", "Gas/client"], scale_rows
            )
        )
        lines.append("")
    if payload_rows:
        lines.append("### E7 — gas versus artefact size")
        lines.append("")
        lines.append(
            markdown_table(
                ["Tier", "Payload", "Adapter (MiB)", "Gas", "Anchored bytes"], payload_rows
            )
        )
        lines.append("")
    return lines


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Aggregate FedChain experiment results.")
    parser.add_argument("--results-dir", default=str(PROJECT_ROOT / "results"))
    parser.add_argument(
        "--across-models",
        action="store_true",
        help=(
            "Build results/comparison_across_models.md from every model-tier "
            "subdirectory (results/<model_key>/) instead of a single directory."
        ),
    )
    parser.add_argument(
        "--order",
        default=",".join(DEFAULT_ORDER),
        help="Comma-separated experiment order for the table columns.",
    )
    parser.add_argument("--baseline", default="exp1_sft", help="Experiment used as the overhead baseline.")
    parser.add_argument(
        "--extra-baselines",
        default="exp0_local",
        help=(
            "Comma-separated experiments to emit ADDITIONAL paired-difference "
            "tables against (default: exp0_local). Pairing against the "
            "centralized arm shows what federation costs; pairing against the "
            "local-only arm shows what it buys, which is the question a reviewer "
            "asks first. Pass '' to emit only the primary baseline."
        ),
    )
    parser.add_argument(
        "--seeds",
        action="store_true",
        help=(
            "Aggregate across seed_<N>/ subdirectories: report accuracy as "
            "mean +- 95%% CI and add paired per-seed deltas against the baseline. "
            "Use this for any accuracy claim - a single run cannot separate an "
            "effect from seed noise."
        ),
    )
    args = parser.parse_args(argv)

    results_dir = Path(args.results_dir)
    if not results_dir.exists():
        print(f"ERROR: results directory not found: {results_dir}", file=sys.stderr)
        return 2

    order = [s.strip() for s in args.order.split(",") if s.strip()]

    if args.across_models:
        markdown = build_across_models(results_dir, order)
        if markdown is None:
            print(
                f"ERROR: no model-tier subdirectories with results under {results_dir}",
                file=sys.stderr,
            )
            return 2
        out_path = results_dir / "comparison_across_models.md"
        out_path.write_text(markdown, encoding="utf-8")
        print(markdown)
        print(f"\nWritten: {out_path}")
        return 0

    seed_tables: List[str] = []
    if args.seeds:
        by_experiment = discover_seed_reports(results_dir, order)
        if not by_experiment:
            print(f"ERROR: no seed_*/ results under {results_dir}", file=sys.stderr)
            return 2
        extra = [s.strip() for s in (args.extra_baselines or "").split(",") if s.strip()]
        seed_tables = build_seed_tables(by_experiment, order, args.baseline, extra)
        seed_tables.extend(build_trajectory_table(by_experiment, order))
        # The per-experiment tables below describe one representative run; the
        # lowest seed is chosen so the choice is deterministic rather than
        # whichever run happened to sort first on disk.
        reports = [
            (name, sorted(runs, key=lambda pair: pair[0])[0][1])
            for name, runs in by_experiment.items()
        ]
        ordered_names = [n for n in order if n in dict(reports)]
        ordered_names.extend(sorted(n for n, _ in reports if n not in ordered_names))
        lookup = dict(reports)
        reports = [(n, lookup[n]) for n in ordered_names]
    else:
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
    if seed_tables:
        lines.append(
            "_Single representative seed. Accuracy with confidence intervals is "
            "in the next section - quote that, not this._"
        )
        lines.append("")
    lines.append(markdown_table(headers, rows))
    lines.append("")
    lines.append(
        "_Communication and adapter sizes are MiB (2^20 bytes). "
        "'Mean Round Duration' is per federated round, so it is not comparable "
        "across paradigms with different round counts - use 'Total Round Time'._"
    )
    lines.append("")

    lines.extend(seed_tables)

    overhead = build_overhead_rows(reports, args.baseline)
    if overhead:
        lines.append(f"## Overhead relative to `{args.baseline}`")
        lines.append("")
        lines.append(
            markdown_table(
                [
                    "Experiment",
                    "d Val. Loss",
                    "d Perplexity",
                    "Comm (MiB)",
                    "Gas",
                    "Total time (s)",
                ],
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
