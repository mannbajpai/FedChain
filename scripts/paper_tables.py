#!/usr/bin/env python3
"""FedChain :: paper-ready tables.

Emits exactly the tables a short (4-5 page) paper needs, in Markdown and LaTeX,
plus ``paper_numbers.json`` holding every quotable scalar so the prose can cite
figures without anyone re-deriving them by hand from a metrics file.

    python scripts/paper_tables.py                       # -> results/paper/
    python scripts/paper_tables.py --out results/paper   # explicit
    python scripts/paper_tables.py --check               # non-zero exit if a
                                                         # paper-blocking number
                                                         # is missing

Design notes that matter for correctness:

* **Generation metrics come only from ``results/<tier>/reeval250``.** The stored
  per-run ROUGE-L/BLEU mix two scorers - 30 main-table runs took the ``builtin``
  fallback while Ablation B1 used ``evaluate`` - so they are not comparable
  across tables. See ablation_study/06_ablation_results.md#e0.
* **Everything paired is paired per seed**, then averaged, never a difference of
  means. Seed variance is shared between arms and cancels.
* **Student's t at n=3 is 4.303**, not 1.96. Using the normal approximation here
  understates every interval by more than half.
* **Detection and false-positive rates get exact one-sided Clopper-Pearson
  bounds.** "100% detection" over 20 trials is a point estimate; the bound is
  what a reviewer will ask for, and it is 13.9% at n=20 versus 5.8% at n=50.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

# Two-sided 95% Student's t. n=3 is the study's standard; the rest are here so a
# re-run at more seeds does not silently fall back to a wrong constant.
T95 = {2: 12.706, 3: 4.303, 4: 3.182, 5: 2.776, 6: 2.571, 7: 2.447,
       8: 2.365, 9: 2.306, 10: 2.262}

TIERS = ("smollm2-360m", "qwen-0.5b")
TIER_LABEL = {"smollm2-360m": "SmolLM2-360M", "qwen-0.5b": "Qwen2.5-0.5B"}

# (arm key, results subdir, human label) for the two partitions.
IID_ARMS = {"e0": "exp0_local", "e1": "exp1_sft", "e2": "exp2_fl",
            "e3": "exp3_fl_bc", "e4": "exp4_fedchain", "e5": "exp5_noniid"}
NONIID_ARMS = {"e0": "ablationB_e0_noniid", "e1": "ablationB_e1_noniid",
               "e2": "ablationB_e2_noniid"}


# =============================================================================
# Statistics
# =============================================================================
def mean_ci(xs: Sequence[float]) -> Tuple[Optional[float], Optional[float]]:
    """Mean and half-width of the 95% t interval. ``(None, None)`` if empty."""
    xs = [x for x in xs if x is not None]
    if not xs:
        return None, None
    n = len(xs)
    m = sum(xs) / n
    if n < 2:
        return m, 0.0
    sd = (sum((x - m) ** 2 for x in xs) / (n - 1)) ** 0.5
    return m, T95.get(n, 1.96) * sd / math.sqrt(n)


def paired(a: Sequence[float], b: Sequence[float]) -> Tuple[Optional[float], Optional[float]]:
    """Mean and 95% CI of ``a - b``, paired element-wise (i.e. per seed)."""
    if not a or not b or len(a) != len(b):
        return None, None
    return mean_ci([x - y for x, y in zip(a, b)])


def significant(m: Optional[float], h: Optional[float]) -> Optional[bool]:
    if m is None or h is None:
        return None
    return abs(m) > h


def clopper_pearson_upper(failures: int, n: int, conf: float = 0.95) -> Optional[float]:
    """One-sided upper bound on a rate given ``failures`` out of ``n``.

    For the all-or-nothing cases this study reports (0 false positives, 0 missed
    attacks) the exact bound collapses to ``1 - (1-conf)**(1/n)``, which is what
    makes "0/20" so much weaker than "0/50": 13.9% versus 5.8%. Only that case is
    implemented, because it is the only one the data produces; anything else
    returns None rather than a wrong number.
    """
    if n <= 0 or failures != 0:
        return None
    return 1.0 - (1.0 - conf) ** (1.0 / n)


def linfit(xs: Sequence[float], ys: Sequence[float]) -> Dict[str, float]:
    """Least-squares ``y = a + b*x`` with R^2."""
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    b = sxy / sxx if sxx else 0.0
    a = my - b * mx
    ss_tot = sum((y - my) ** 2 for y in ys)
    ss_res = sum((y - (a + b * x)) ** 2 for x, y in zip(xs, ys))
    r2 = 1 - ss_res / ss_tot if ss_tot else 1.0
    return {"intercept": a, "slope": b, "r2": r2}


# =============================================================================
# Loading
# =============================================================================
class Study:
    """Every metrics file the paper draws on, loaded once."""

    def __init__(self, root: Path, seeds: Sequence[int]):
        self.root = root
        self.seeds = list(seeds)
        self.runs: Dict[Tuple[str, str, int], dict] = {}   # (tier, exp, seed)
        self.reeval: Dict[Tuple[str, str], List[dict]] = defaultdict(list)
        self.audit: Dict[Tuple[str, str], dict] = {}       # (tier, "e6"|"e7")
        self._load()

    def _load(self) -> None:
        for tier in TIERS:
            tdir = self.root / tier
            if not tdir.is_dir():
                continue
            for sub in ("", "ablation"):
                for seed in self.seeds:
                    sdir = tdir / sub / f"seed_{seed}" if sub else tdir / f"seed_{seed}"
                    if not sdir.is_dir():
                        continue
                    for f in sdir.glob("*_metrics.json"):
                        exp = f.name[: -len("_metrics.json")]
                        try:
                            self.runs[(tier, exp, seed)] = json.loads(f.read_text(encoding="utf-8"))
                        except Exception as exc:                     # pragma: no cover
                            print(f"  warn: unreadable {f}: {exc}", file=sys.stderr)

            # Generation metrics: reeval250 only. It is written either as a file
            # or as a directory of per-arm files depending on how reevaluate.py
            # was invoked, so accept both rather than making the caller care.
            for cand in (tdir / "reeval250", *sorted(tdir.glob("reeval*.json"))):
                if cand.is_file():
                    self._ingest_reeval(tier, cand)
                elif cand.is_dir():
                    for f in sorted(cand.glob("*.json")):
                        self._ingest_reeval(tier, f)
            for sub in ("ablation",):
                d = tdir / sub / "reeval250"
                if d.is_file():
                    self._ingest_reeval(tier, d)
                elif d.is_dir():
                    for f in sorted(d.glob("*.json")):
                        self._ingest_reeval(tier, f)

            for key, name in (("e6", "exp6_tamper_metrics.json"),
                              ("e7", "exp7_scalability_metrics.json")):
                f = tdir / name
                if f.is_file():
                    try:
                        self.audit[(tier, key)] = json.loads(f.read_text(encoding="utf-8"))
                    except Exception as exc:                        # pragma: no cover
                        print(f"  warn: unreadable {f}: {exc}", file=sys.stderr)

    def _ingest_reeval(self, tier: str, path: Path) -> None:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return
        for row in payload.get("results", []):
            arm = row.get("arm")
            if arm:
                self.reeval[(tier, arm)].append(row)

    # -- accessors ---------------------------------------------------------
    def loss(self, tier: str, exp: str) -> List[float]:
        """Validation loss per seed, in seed order. Short if a seed is missing."""
        out = []
        for s in self.seeds:
            d = self.runs.get((tier, exp, s))
            if d:
                v = (d.get("metrics") or {}).get("validation_loss")
                if v is not None:
                    out.append(v)
        return out

    def metric(self, tier: str, exp: str, key: str) -> List[float]:
        out = []
        for s in self.seeds:
            d = self.runs.get((tier, exp, s))
            if d:
                v = (d.get("metrics") or {}).get(key)
                if v is not None:
                    out.append(v)
        return out

    def gen(self, tier: str, arm: str, key: str) -> List[float]:
        """Generation metric per seed from reeval250, averaged within a seed.

        E0 contributes three adapters per seed (one per client); collapsing them
        to a seed-level mean first is what keeps the interval a between-seed
        interval rather than a mixture of two variance sources.
        """
        rows = self.reeval.get((tier, arm), [])
        if not rows:
            return []
        by_seed: Dict[Any, List[float]] = defaultdict(list)
        for r in rows:
            v = r.get(key)
            if v is not None:
                by_seed[r.get("seed")].append(v)
        return [sum(v) / len(v) for _, v in sorted(by_seed.items()) if v]

    def hashes(self, tier: str, exp: str, level: str = "client") -> Dict[Tuple, str]:
        """(seed, round, who) -> SHA-256 of that adapter.

        ``level='client'`` gives the per-client updates (3 per round); ``'global'``
        gives the aggregated model (1 per round). Both are worth reporting: the
        client level shows the audit layer does not perturb what each participant
        uploads, the global level shows it does not perturb what aggregation
        produces. The global hash is also the value anchored on-chain, so it is
        the one an external auditor can actually check.
        """
        out = {}
        for s in self.seeds:
            d = self.runs.get((tier, exp, s))
            if not d:
                continue
            for rnd in d.get("rounds", []):
                r = rnd.get("round")
                if level == "client":
                    for c in rnd.get("clients", []):
                        h = c.get("model_hash")
                        if h:
                            out[(s, r, c.get("client_id"))] = h
                else:
                    h = (rnd.get("global_model") or {}).get("model_hash")
                    if h:
                        out[(s, r, "global")] = h
        return out


# =============================================================================
# Formatting helpers
# =============================================================================
def f(v: Optional[float], p: int = 4) -> str:
    return "--" if v is None else f"{v:.{p}f}"


def pm(m: Optional[float], h: Optional[float], p: int = 4) -> str:
    if m is None:
        return "--"
    if not h:
        return f"{m:.{p}f}"
    return f"{m:.{p}f} ± {h:.{p}f}"


def pm_tex(m: Optional[float], h: Optional[float], p: int = 4) -> str:
    if m is None:
        return "--"
    if not h:
        return f"{m:.{p}f}"
    return f"${m:.{p}f} \\pm {h:.{p}f}$"


def pct(v: Optional[float], p: int = 1) -> str:
    return "--" if v is None else f"{v * 100:.{p}f}\\%"


def md_table(headers: Sequence[str], rows: Iterable[Sequence[str]]) -> str:
    rows = [list(map(str, r)) for r in rows]
    widths = [max(len(str(headers[i])), *(len(r[i]) for r in rows)) if rows
              else len(str(headers[i])) for i in range(len(headers))]
    out = ["| " + " | ".join(str(h).ljust(w) for h, w in zip(headers, widths)) + " |",
           "|" + "|".join("-" * (w + 2) for w in widths) + "|"]
    for r in rows:
        out.append("| " + " | ".join(c.ljust(w) for c, w in zip(r, widths)) + " |")
    return "\n".join(out)


def tex_table(headers: Sequence[str], rows: Iterable[Sequence[str]],
              caption: str, label: str, align: Optional[str] = None) -> str:
    rows = [list(map(str, r)) for r in rows]
    align = align or ("l" + "r" * (len(headers) - 1))
    esc = lambda s: (s.replace("±", "$\\pm$").replace("_", "\\_")
                      .replace("→", "$\\rightarrow$").replace("×", "$\\times$")
                      .replace("α", "$\\alpha$").replace("≤", "$\\leq$"))
    lines = [r"\begin{table}[t]", r"\centering", r"\small",
             f"\\caption{{{esc(caption)}}}", f"\\label{{{label}}}",
             f"\\begin{{tabular}}{{{align}}}", r"\toprule",
             " & ".join(f"\\textbf{{{esc(str(h))}}}" for h in headers) + r" \\",
             r"\midrule"]
    lines += [" & ".join(esc(c) for c in r) + r" \\" for r in rows]
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines)


# =============================================================================
# Tables
# =============================================================================
def table_main(st: Study, N: dict) -> Tuple[str, str, List[str]]:
    """T1 - the learning result: FedAvg's recovery of the isolation->pooled gap.

    This is the paper's motivation table. Each cell is a (tier, partition) pair;
    the last column is the quantity the study exists to produce.
    """
    headers = ["Model", "Partition", "Local-only (E0)", "Centralized (E1)",
               "FedAvg (E2)", "E0 − E2 (paired)", "Gap recovered"]
    rows, gaps, missing = [], {}, []

    for tier in TIERS:
        for part, arms, sub in (("IID", IID_ARMS, ""),
                                ("Dirichlet α=0.3", NONIID_ARMS, "ablation")):
            e0 = st.loss(tier, arms["e0"])
            e1 = st.loss(tier, arms["e1"])
            e2 = st.loss(tier, arms["e2"])
            if not (e0 and e1 and e2):
                rows.append([TIER_LABEL[tier], part, "--", "--", "--", "--",
                             "*(not run)*"])
                missing.append(f"{TIER_LABEL[tier]} / {part}")
                continue
            n = min(len(e0), len(e1), len(e2))
            e0, e1, e2 = e0[:n], e1[:n], e2[:n]
            gm, gh = paired(e0, e2)                     # FedAvg gain (positive = better)
            hm, hh = paired(e0, e1)                     # headroom to the bound
            frac = [(a - c) / (a - b) for a, b, c in zip(e0, e1, e2) if a != b]
            fm, fh = mean_ci(frac)
            sig = significant(gm, gh)
            rows.append([
                TIER_LABEL[tier], part,
                pm(*mean_ci(e0)), pm(*mean_ci(e1)), pm(*mean_ci(e2)),
                pm(gm, gh, 5) + ("" if sig else " (n.s.)"),
                f"{fm * 100:.1f}% ± {fh * 100:.1f}%" if fm is not None else "--",
            ])
            gaps[(tier, part)] = {
                "n_seeds": n,
                "e0": dict(zip(("mean", "ci"), mean_ci(e0))),
                "e1": dict(zip(("mean", "ci"), mean_ci(e1))),
                "e2": dict(zip(("mean", "ci"), mean_ci(e2))),
                "gain_e0_minus_e2": {"mean": gm, "ci": gh, "significant": sig},
                "headroom_e0_minus_e1": {"mean": hm, "ci": hh},
                "recovered_fraction": {"mean": fm, "ci": fh},
            }
    N["learning"] = {f"{t}|{p}": v for (t, p), v in gaps.items()}

    # The cross-partition growth factor, per tier. Unpaired across partitions
    # (different shard files), so it is a comparison of two paired estimates.
    growth = {}
    for tier in TIERS:
        a = gaps.get((tier, "IID"))
        b = gaps.get((tier, "Dirichlet α=0.3"))
        if not (a and b):
            continue
        ga, ha = a["gain_e0_minus_e2"]["mean"], a["gain_e0_minus_e2"]["ci"]
        gb, hb = b["gain_e0_minus_e2"]["mean"], b["gain_e0_minus_e2"]["ci"]
        growth[tier] = {
            "iid_gain": ga, "noniid_gain": gb,
            "factor": gb / ga if ga else None,
            "intervals_disjoint": (gb - hb) > (ga + ha),
            "recovered_iid": a["recovered_fraction"]["mean"],
            "recovered_noniid": b["recovered_fraction"]["mean"],
        }
    N["skew_growth"] = growth

    cap = ("Federated averaging against matched isolated and centralized baselines, "
           "at a matched 4{,}500-update budget (R=3). Loss is validation "
           "cross-entropy; lower is better. `Gap recovered' is "
           "(E0-E2)/(E0-E1) computed per seed. Intervals are 95\\% Student's t "
           "over 3 seeds.")
    return md_table(headers, rows), tex_table(headers, rows, cap, "tab:main"), missing


def table_hash_equality(st: Study, N: dict) -> Tuple[str, str, List[str]]:
    """T2 - the systems result: the audit layer is an exact no-op.

    Bit-identical artefacts, not a non-significant difference. This is the
    strongest claim in the paper and the one a reviewer can verify.
    """
    headers = ["Model", "Partition", "Comparison", "Client adapters",
               "Global models", "Δ val. loss"]
    rows, records, missing = [], [], []

    checks = [
        ("IID", "", [("exp2_fl", "exp3_fl_bc", "E2 vs E3 (chain)"),
                     ("exp3_fl_bc", "exp4_fedchain", "E3 vs E4 (+IPFS)")]),
        ("Dirichlet α=0.3", "ablation",
         [("ablationB_e2_noniid", "exp5_noniid", "E2 vs E4-equiv (chain+IPFS)")]),
    ]

    def compare(tier: str, left: str, right: str, level: str) -> Optional[Tuple[int, int]]:
        hl, hr = st.hashes(tier, left, level), st.hashes(tier, right, level)
        keys = sorted(set(hl) & set(hr))
        if not keys:
            return None
        return sum(1 for k in keys if hl[k] == hr[k]), len(keys)

    for tier in TIERS:
        for part, _sub, pairs in checks:
            for left, right, label in pairs:
                cl = compare(tier, left, right, "client")
                gl = compare(tier, left, right, "global")
                if cl is None and gl is None:
                    rows.append([TIER_LABEL[tier], part, label, "*(not run)*",
                                 "*(not run)*", "--"])
                    missing.append(f"{TIER_LABEL[tier]} / {part} / {label}")
                    continue
                fmt = lambda p: ("--" if p is None else
                                 (f"**{p[0]}/{p[1]}**" if p[0] == p[1]
                                  else f"**{p[0]}/{p[1]} MISMATCH**"))
                dm, _ = paired(st.loss(tier, left), st.loss(tier, right))
                rows.append([TIER_LABEL[tier], part, label, fmt(cl), fmt(gl),
                             "0.000000" if dm == 0 else f(dm, 6)])
                records.append({
                    "tier": tier, "partition": part, "comparison": label,
                    "client_adapters": cl[1] if cl else 0,
                    "client_identical": cl[0] if cl else 0,
                    "global_models": gl[1] if gl else 0,
                    "global_identical": gl[0] if gl else 0,
                    "delta_val_loss": dm,
                })
    # Distinct artefacts, as opposed to pairwise comparisons. One aggregated model
    # per (tier, partition, seed, round) is checked across every audited variant;
    # counting it once is the framing used elsewhere in the repo, and counting the
    # comparisons is what the table shows. Both are recorded so the paper cannot
    # end up quoting two different numbers for the same evidence.
    distinct_global: set = set()
    distinct_client: set = set()
    for tier in TIERS:
        for part, _sub, pairs in checks:
            for left, right, _label in pairs:
                for exp in (left, right):
                    for k in st.hashes(tier, exp, "global"):
                        distinct_global.add((tier, part, k))
                    for k in st.hashes(tier, exp, "client"):
                        distinct_client.add((tier, part, k))

    N["hash_equality"] = records
    N["hash_equality_total"] = {
        "client_comparisons": sum(r["client_adapters"] for r in records),
        "client_identical": sum(r["client_identical"] for r in records),
        "global_comparisons": sum(r["global_models"] for r in records),
        "global_identical": sum(r["global_identical"] for r in records),
        "distinct_client_adapters": len(distinct_client),
        "distinct_global_models": len(distinct_global),
        "all_identical": all(r["client_adapters"] == r["client_identical"]
                             and r["global_models"] == r["global_identical"]
                             for r in records),
    }
    cap = ("Artefact-level equality between the plain federated arm and the "
           "audited arms, at both the per-client and the aggregated level. Each "
           "adapter is compared by SHA-256 over its canonical serialization; the "
           "global digest is also the value anchored on-chain, so an external "
           "auditor can re-derive it. Equality is exact, so no statistical test "
           "is required or reported.")
    return md_table(headers, rows), tex_table(headers, rows, cap, "tab:hash"), missing


def table_systems(st: Study, N: dict) -> Tuple[str, str, List[str]]:
    """T3 - what the audit layer costs, decomposed into chain and transport."""
    headers = ["Model", "Arm", "Comm. (MiB)", "Δ comm.", "Gas", "Total time (s)",
               "Δ time"]
    rows, records = [], []
    arms = [("exp2_fl", "E2 FedAvg"), ("exp3_fl_bc", "E3 +chain"),
            ("exp4_fedchain", "E4 +chain+IPFS")]
    for tier in TIERS:
        base_c = base_t = None
        for exp, label in arms:
            comm, _ = mean_ci(st.metric(tier, exp, "communication_volume_mb"))
            gas, _ = mean_ci(st.metric(tier, exp, "blockchain_gas_used"))
            tm, th = mean_ci(st.metric(tier, exp, "total_round_time_sec"))
            if comm is None:
                continue
            if base_c is None:
                base_c, base_t = comm, tm
            dc = "--" if comm is None or not base_c else f"{(comm / base_c - 1) * 100:+.1f}%"
            dt = "--" if tm is None or not base_t else f"{(tm / base_t - 1) * 100:+.1f}%"
            rows.append([TIER_LABEL[tier], label, f(comm, 2), dc,
                         f"{gas:,.0f}" if gas is not None else "--",
                         pm(tm, th, 1), dt])
            records.append({"tier": tier, "arm": label, "comm_mib": comm,
                            "gas": gas, "total_time_sec": tm,
                            "comm_overhead_pct": (comm / base_c - 1) * 100 if base_c else None,
                            "time_overhead_pct": (tm / base_t - 1) * 100 if base_t else None})
    N["systems"] = records
    cap = ("Systems cost of the audit layer. Anchoring (E3) adds zero "
           "communication: its volume is byte-identical to the un-audited "
           "federated arm. The whole communication overhead arrives with IPFS "
           "transport (E4), which is an implementation choice rather than a cost "
           "of auditability.")
    return md_table(headers, rows), tex_table(headers, rows, cap, "tab:systems"), []


def table_tamper(st: Study, N: dict) -> Tuple[str, str, List[str]]:
    """T4 - E6. Detection with exact bounds, and the benign control."""
    headers = ["Model", "Perturbation", "Type", "Flagged", "Rate", "95% bound"]
    rows, records, missing = [], [], []
    for tier in TIERS:
        d = st.audit.get((tier, "e6"))
        if not d:
            missing.append(f"E6 at {TIER_LABEL[tier]}")
            continue
        for s in d.get("summary", []):
            n, det = s.get("trials", 0), s.get("detected", 0)
            benign = bool(s.get("benign_control"))
            if benign:
                bound = clopper_pearson_upper(det, n)
                bound_s = f"FPR ≤ {bound * 100:.1f}%" if bound is not None else "--"
            else:
                bound = clopper_pearson_upper(n - det, n)
                bound_s = f"miss ≤ {bound * 100:.1f}%" if bound is not None else "--"
            rows.append([TIER_LABEL[tier], s.get("attack", "?"),
                         "benign control" if benign else "attack",
                         f"{det}/{n}", f"{s.get('detection_rate', 0) * 100:.0f}%",
                         bound_s])
            records.append({"tier": tier, "perturbation": s.get("attack"),
                            "benign_control": benign, "trials": n, "flagged": det,
                            "bound_95": bound,
                            "mean_verify_latency_sec": s.get("mean_verify_latency_sec")})
    N["tamper"] = records
    lat = [r["mean_verify_latency_sec"] for r in records
           if r.get("mean_verify_latency_sec") is not None]
    N["verify_latency_sec_mean"] = sum(lat) / len(lat) if lat else None
    cap = ("Integrity checking against a hostile transport layer. Four "
           "perturbations must be flagged and one benign re-serialization must "
           "not. Bounds are exact one-sided 95\\% Clopper-Pearson: a perfect "
           "score over $n$ trials bounds the error rate at $1-0.05^{1/n}$, so "
           "trial count is what the claim rests on.")
    return md_table(headers, rows), tex_table(headers, rows, cap, "tab:tamper"), missing


def table_gas(st: Study, N: dict) -> Tuple[str, str, List[str]]:
    """T5 - E7. Anchoring cost against federation size and artefact size."""
    headers = ["Model", "Sweep", "Range", "Fit / spread", "Reading"]
    rows, records, missing = [], [], []
    for tier in TIERS:
        d = st.audit.get((tier, "e7"))
        if not d:
            missing.append(f"E7 at {TIER_LABEL[tier]}")
            continue
        cs = d.get("clients_sweep", [])
        if cs:
            ns = [c["num_clients"] for c in cs]
            gs = [c["gas_per_round"] for c in cs]
            fit = linfit(ns, gs)
            rows.append([TIER_LABEL[tier], "clients",
                         f"N = {min(ns)}–{max(ns)}",
                         f"gas = {fit['intercept']:,.0f} + {fit['slope']:,.0f}·N, R² = {fit['r2']:.6f}",
                         "linear in participants"])
            records.append({"tier": tier, "sweep": "clients", "n_min": min(ns),
                            "n_max": max(ns), **fit,
                            "gas_per_client_at_max": cs[-1].get("gas_per_client")})
        ps = d.get("payload_sweep", [])
        if ps:
            mb = [p["adapter_mb"] for p in ps]
            gas = [p["gas_used"] for p in ps]
            spread = (max(gas) - min(gas)) / min(gas) * 100 if min(gas) else 0.0
            anchored = {p.get("anchored_bytes") for p in ps}
            rows.append([TIER_LABEL[tier], "artefact size",
                         f"{min(mb):.2f}–{max(mb):.1f} MiB ({max(mb) / min(mb):.0f}×)",
                         f"gas spread {spread:.4f}%",
                         f"flat; {sorted(anchored)[0]} bytes anchored at every size"])
            records.append({"tier": tier, "sweep": "payload",
                            "mb_min": min(mb), "mb_max": max(mb),
                            "size_ratio": max(mb) / min(mb),
                            "gas_spread_pct": spread,
                            "anchored_bytes": sorted(anchored)})
    N["gas_scaling"] = records
    cap = ("Anchoring cost. Gas is linear in the number of participants and "
           "independent of artefact size, because what is anchored is a 32-byte "
           "digest rather than the model.")
    return md_table(headers, rows), tex_table(headers, rows, cap, "tab:gas"), missing


def table_generation(st: Study, N: dict) -> Tuple[str, str, List[str]]:
    """T6 - generation quality, single scorer, 250 samples. Supporting only."""
    headers = ["Model", "Partition", "Arm", "ROUGE-L", "BLEU"]
    rows, records, missing = [], [], []
    spec = [("IID", [("exp1_sft", "E1 Centralized"), ("exp2_fl", "E2 FedAvg"),
                     ("exp0_local", "E0 Local-only")]),
            ("Dirichlet α=0.3", [("ablationB_e1_noniid", "E1 Centralized"),
                                 ("ablationB_e2_noniid", "E2 FedAvg"),
                                 ("ablationB_e0_noniid", "E0 Local-only")])]
    for tier in TIERS:
        for part, arms in spec:
            for arm, label in arms:
                r, b = st.gen(tier, arm, "rouge_l"), st.gen(tier, arm, "bleu")
                if not r:
                    missing.append(f"reeval250 {TIER_LABEL[tier]} / {part} / {label}")
                    continue
                rows.append([TIER_LABEL[tier], part, label,
                             pm(*mean_ci(r)), pm(*mean_ci(b))])
                records.append({"tier": tier, "partition": part, "arm": arm,
                                "rouge_l": dict(zip(("mean", "ci"), mean_ci(r))),
                                "bleu": dict(zip(("mean", "ci"), mean_ci(b))),
                                "n_seeds": len(r)})
    N["generation"] = records
    cap = ("Generation quality over 250 greedy decodes, scored with a single "
           "backend (HF \\texttt{evaluate}). Reported as a collapse check "
           "supporting the loss result, not as an independent ordering: the "
           "intervals are between-seed at $n=3$ and do not respond to decode "
           "count.")
    return md_table(headers, rows), tex_table(headers, rows, cap, "tab:gen"), missing


# =============================================================================
# Main
# =============================================================================
def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results-dir", default="results", type=Path)
    ap.add_argument("--out", default=None, type=Path,
                    help="Output directory (default: <results-dir>/paper).")
    ap.add_argument("--seeds", default="42 43 44")
    ap.add_argument("--check", action="store_true",
                    help="Exit non-zero if a paper-blocking table is incomplete.")
    args = ap.parse_args(argv)

    # The tables carry U+2212, ±, α and × . Files are always written UTF-8, but
    # stdout inherits the console codepage, which is cp1252 on Windows and raises
    # on the first minus sign. Reconfigure rather than degrade the file content.
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):                        # pragma: no cover
            pass

    seeds = [int(s) for s in args.seeds.replace(",", " ").split()]
    out = args.out or (args.results_dir / "paper")
    out.mkdir(parents=True, exist_ok=True)

    st = Study(args.results_dir, seeds)
    if not st.runs:
        print(f"error: no metrics found under {args.results_dir}", file=sys.stderr)
        return 2

    N: Dict[str, Any] = {"seeds": seeds, "tiers": list(TIERS)}
    builders = [
        ("1", "Main result: does federating help?", table_main, True),
        ("2", "The audit layer is an exact no-op", table_hash_equality, True),
        ("3", "Systems cost of the audit layer", table_systems, False),
        ("4", "Tamper detection", table_tamper, True),
        ("5", "Anchoring cost scaling", table_gas, True),
        ("6", "Generation quality (supporting)", table_generation, False),
    ]

    md = ["# FedChain — paper tables",
          "",
          f"Generated from `{args.results_dir}` over seeds {seeds}.",
          "",
          "Every figure here is computed from the stored metrics files. Paired",
          "differences are per seed; intervals are 95% Student's *t*. Generation",
          "metrics come only from `reeval250` (single scorer) — never from the",
          "per-run values, which mix two scorers.",
          ""]
    tex = ["% FedChain -- paper tables. Generated by scripts/paper_tables.py.",
           "% Requires \\usepackage{booktabs}.", ""]
    blocking: List[str] = []

    for num, title, fn, is_blocking in builders:
        m, t, missing = fn(st, N)
        md += [f"## Table {num} — {title}", "", m, ""]
        tex += [t, ""]
        if missing:
            md += ["> **Incomplete:** " + "; ".join(missing), ""]
            if is_blocking:
                blocking += missing

    # Headline sentences, pre-composed so the prose quotes one source.
    md += ["## Quotable figures", ""]
    heads = []
    tot = N.get("hash_equality_total", {})
    if tot.get("client_comparisons"):
        parts = {(r["tier"], r["partition"]) for r in N.get("hash_equality", [])
                 if r["global_models"]}
        heads.append(
            f"- **{tot['client_identical']}/{tot['client_comparisons']} client-adapter** and "
            f"**{tot['global_identical']}/{tot['global_comparisons']} aggregated-model** "
            f"pairwise comparisons are bit-identical between audited and un-audited "
            f"federated training, over {len(parts)} (architecture, partition) settings.")
        heads.append(
            f"  Equivalently, in distinct artefacts: every one of "
            f"**{tot['distinct_global_models']} aggregated models** and "
            f"**{tot['distinct_client_adapters']} client adapters** is identical across "
            f"all audited variants. *(Quote one framing or the other, not both.)*")
    for tier, g in N.get("skew_growth", {}).items():
        if g.get("factor"):
            heads.append(
                f"- **{TIER_LABEL[tier]}**: FedAvg recovers "
                f"{g['recovered_iid'] * 100:.1f}% of the isolation→centralized gap under IID "
                f"and {g['recovered_noniid'] * 100:.1f}% under Dirichlet(0.3); the absolute "
                f"gain grows {g['factor']:.2f}× "
                f"({'disjoint' if g['intervals_disjoint'] else 'overlapping'} intervals).")
    fp = [r for r in N.get("tamper", []) if r.get("benign_control")]
    for r in fp:
        if r.get("bound_95") is not None:
            heads.append(f"- **{TIER_LABEL[r['tier']]}**: {r['flagged']}/{r['trials']} false "
                         f"positives on the benign control — FPR ≤ {r['bound_95'] * 100:.1f}% "
                         f"(one-sided 95%).")
    for r in N.get("gas_scaling", []):
        if r.get("sweep") == "clients":
            heads.append(f"- **{TIER_LABEL[r['tier']]}**: gas = "
                         f"{r['intercept']:,.0f} + {r['slope']:,.0f}·N, R² = {r['r2']:.6f}.")
    if N.get("verify_latency_sec_mean"):
        heads.append(f"- Mean integrity-check latency: "
                     f"**{N['verify_latency_sec_mean'] * 1000:.1f} ms** per artefact.")
    md += (heads or ["- *(nothing computable yet)*"]) + [""]

    # A hash divergence between an audited and an un-audited arm means the audit
    # layer perturbed training. ablation_study/05 names it first among the things
    # that would falsify the paper's central claim, so it is fatal here rather
    # than a row someone has to notice in a table.
    divergent = [r for r in N.get("hash_equality", [])
                 if (r["client_adapters"] and r["client_identical"] != r["client_adapters"])
                 or (r["global_models"] and r["global_identical"] != r["global_models"])]
    N["hash_divergence"] = divergent
    if divergent:
        md += ["## ⚠ HASH DIVERGENCE — STOP", "",
               "The audit layer produced artefacts that differ from plain federated",
               "training. This falsifies the paper's central claim and is a bug, not a",
               "measurement. Do not write the paper from these tables until it is",
               "understood.", ""]
        for r in divergent:
            md.append(f"- {TIER_LABEL[r['tier']]} / {r['partition']} / {r['comparison']}: "
                      f"{r['client_identical']}/{r['client_adapters']} client, "
                      f"{r['global_identical']}/{r['global_models']} global, "
                      f"Δ loss {f(r['delta_val_loss'], 6)}")
        md.append("")

    (out / "tables.md").write_text("\n".join(md), encoding="utf-8")
    (out / "tables.tex").write_text("\n".join(tex), encoding="utf-8")
    (out / "paper_numbers.json").write_text(json.dumps(N, indent=2), encoding="utf-8")

    print("\n".join(md))
    print(f"\nWritten: {out / 'tables.md'}")
    print(f"Written: {out / 'tables.tex'}")
    print(f"Written: {out / 'paper_numbers.json'}")

    if divergent:
        print("\nHASH DIVERGENCE (falsifies the central claim):", file=sys.stderr)
        for r in divergent:
            print(f"  - {TIER_LABEL[r['tier']]} / {r['partition']} / {r['comparison']}: "
                  f"{r['client_identical']}/{r['client_adapters']} client, "
                  f"{r['global_identical']}/{r['global_models']} global", file=sys.stderr)

    if blocking:
        print("\nINCOMPLETE (paper-blocking):", file=sys.stderr)
        for b in blocking:
            print(f"  - {b}", file=sys.stderr)

    if args.check:
        if divergent:
            return 2                    # distinct exit code: a bug, not a gap
        if blocking:
            return 1
        print("\nAll paper-blocking tables are complete, and every audited artefact "
              "is bit-identical to its un-audited counterpart.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
