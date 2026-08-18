#!/usr/bin/env python3
"""FedChain :: paper figures.

Renders every figure the paper can draw on, from the same stored metrics that
``scripts/paper_tables.py`` reads. Nothing here recomputes science: values come
from ``results/paper/paper_numbers.json`` where that file already has them, and
from the per-seed metrics only for the round-level trajectory, which the tables
do not carry.

    python scripts/paper_figures.py                 # -> paper/figures/
    python scripts/paper_figures.py --out somewhere

Each figure is written twice: PDF (vector, what goes in the paper) and PNG at
300 dpi (for review and for the repo README).

Design notes that matter:

* **Palette is Okabe-Ito**, validated colourblind-safe at the four slots used
  adjacently (worst all-pairs deuteranopia dE 11.0, normal-vision 15.6). Series
  identity is never carried by colour alone - every figure has a legend and the
  small-multiple panels are titled.
* **The 8-category partition figure uses a sequential ramp, not 8 hues.** Eight
  categorical hues cannot be separated safely; category share is a magnitude, so
  it gets one hue light-to-dark and printed values.
* **No figure titles.** Captions live in the LaTeX ``\\caption``. Axis labels and
  legends carry everything needed to read the panel.
* **No dual axes anywhere.** Where two measures of different scale belong in one
  figure they are separate panels.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.lines import Line2D

# --- palette -----------------------------------------------------------------
# Okabe-Ito. Slots are assigned in fixed order and never cycled.
BLUE, ORANGE, GREEN, VERM = "#0072B2", "#E69F00", "#009E73", "#D55E00"
SKY, PURPLE = "#56B4E9", "#CC79A7"
INK, MUTED, GRID = "#1a1a1a", "#5c5c5c", "#d8d8d8"

TIERS = ("smollm2-360m", "qwen-0.5b")
TIER_LABEL = {"smollm2-360m": "SmolLM2-360M", "qwen-0.5b": "Qwen2.5-0.5B"}
IID, NONIID = "IID", "Dirichlet α=0.3"

# Column widths for a two-column conference template, in inches.
COL, FULL = 3.4, 7.0


def style() -> None:
    sns.set_theme(context="paper", style="ticks")
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Nimbus Roman", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "font.size": 8,
        "axes.labelsize": 8,
        "axes.titlesize": 8.5,
        "legend.fontsize": 7,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "axes.edgecolor": MUTED,
        "axes.labelcolor": INK,
        "text.color": INK,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "axes.linewidth": 0.6,
        "grid.color": GRID,
        "grid.linewidth": 0.5,
        "lines.linewidth": 1.4,
        "legend.frameon": False,
        "figure.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
        "pdf.fonttype": 42,          # embed TrueType, not Type 3 - most venues require it
        "ps.fonttype": 42,
    })


def save(fig: plt.Figure, out: Path, name: str) -> None:
    out.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(out / f"{name}.{ext}")
    plt.close(fig)
    print(f"  wrote {name}.pdf / .png")


def vgrid(ax: plt.Axes, axis: str = "y") -> None:
    """Recessive grid on the value axis only."""
    ax.grid(True, axis=axis, linestyle="-", alpha=0.7, zorder=0)
    ax.set_axisbelow(True)
    sns.despine(ax=ax, trim=False)


# =============================================================================
# Data access
# =============================================================================
class Data:
    def __init__(self, results: Path):
        self.results = results
        self.N = json.loads((results / "paper" / "paper_numbers.json")
                            .read_text(encoding="utf-8"))

    def metrics(self, tier: str, exp: str, seed: int, sub: str = "") -> Optional[dict]:
        p = self.results / tier
        if sub:
            p = p / sub
        p = p / f"seed_{seed}" / f"{exp}_metrics.json"
        if not p.exists():
            return None
        return json.loads(p.read_text(encoding="utf-8"))

    def final_loss(self, tier: str, exp: str, sub: str = "") -> List[float]:
        out = []
        for seed in (42, 43, 44):
            d = self.metrics(tier, exp, seed, sub)
            if d:
                out.append(d["metrics"]["validation_loss"])
        return out

    def trajectory(self, tier: str, exp: str, sub: str = "") -> Optional[np.ndarray]:
        """Per-round validation loss, meaned over seeds.

        Round 1..R-1 are stored on the round record; the final round's score is
        the run-level ``validation_loss``. A round whose evaluation block is
        empty (eval stride, or the final round) falls back to that.
        """
        curves = []
        for seed in (42, 43, 44):
            d = self.metrics(tier, exp, seed, sub)
            if not d:
                continue
            pts = []
            for rd in d["rounds"]:
                ev = rd.get("evaluation") or {}
                pts.append(ev.get("loss"))
            pts[-1] = d["metrics"]["validation_loss"]
            if any(p is None for p in pts):
                continue
            curves.append(pts)
        return np.array(curves).mean(axis=0) if curves else None


def cp_bounds(k: int, n: int) -> tuple:
    """Exact one-sided 95% Clopper-Pearson bound on a 0/n or n/n outcome."""
    if k == n:
        return 0.05 ** (1 / n), 1.0          # rate is at least this
    if k == 0:
        return 0.0, 1 - 0.05 ** (1 / n)      # rate is at most this
    return k / n, k / n


# =============================================================================
# Figures
# =============================================================================
def fig_heterogeneity(root: Path, out: Path) -> None:
    """F1 - what the two partitions actually look like on disk.

    Category share is a magnitude, so it takes one sequential hue rather than
    eight categorical ones, and every cell prints its value.
    """
    mans = {
        "IID": json.loads((root / "data" / "manifest.json").read_text(encoding="utf-8")),
        "Dirichlet $\\alpha$=0.3": json.loads(
            (root / "data" / "dirichlet" / "manifest.json").read_text(encoding="utf-8")),
    }
    cats = sorted({c for m in mans.values() for p in m["partition_profile"]
                   for c in p["label_histogram"]})
    short = {c: c.replace("_qa", " QA").replace("_", " ") for c in cats}

    fig, axes = plt.subplots(1, 2, figsize=(FULL, 2.5),
                             gridspec_kw={"wspace": 0.16})
    for ax, (name, man) in zip(axes, mans.items()):
        profs = man["partition_profile"]
        mat = np.array([[100 * p["label_histogram"].get(c, 0) / p["num_records"]
                         for c in cats] for p in profs])
        ylab = [f"{p['client'].replace('client', 'Client ')}\n$n$={p['num_records']:,}"
                for p in profs]
        sns.heatmap(mat, ax=ax, cmap="Blues", vmin=0, vmax=55, annot=True,
                    fmt=".0f", annot_kws={"size": 6.5}, linewidths=1.4,
                    linecolor="white", cbar=False,
                    xticklabels=[short[c] for c in cats], yticklabels=ylab)
        ax.set_title(name, pad=6)
        ax.tick_params(axis="y", labelsize=6.5, rotation=0, length=0)
        ax.tick_params(axis="x", length=0)
        ax.set_xticklabels([short[c] for c in cats], fontsize=6, rotation=40,
                           ha="right", rotation_mode="anchor")
    fig.text(0.5, -0.20, "share of each client's shard in each Dolly task category (%)",
             ha="center", fontsize=7.5, color=MUTED)
    save(fig, out, "fig1_heterogeneity")


def fig_gap_recovery(d: Data, out: Path) -> None:
    """F2 - the main learning result, drawn as the quantity actually claimed.

    Each row is a (tier, partition) cell. The segment spans the isolation bound
    (E0) to the pooled bound (E1); FedAvg sits on it, and the labelled fraction
    is how much of that span it recovered.
    """
    cells = [(t, p) for t in TIERS for p in (IID, NONIID)]
    fig, ax = plt.subplots(figsize=(FULL, 2.35))
    ys = np.arange(len(cells))[::-1]

    for y, (tier, part) in zip(ys, cells):
        L = d.N["learning"][f"{tier}|{part}"]
        e0, e1, e2 = L["e0"]["mean"], L["e1"]["mean"], L["e2"]["mean"]
        frac = L["recovered_fraction"]
        ax.plot([e1, e0], [y, y], color=GRID, lw=5, solid_capstyle="round", zorder=1)
        ax.plot([e2, e0], [y, y], color=BLUE, lw=5, solid_capstyle="round",
                zorder=2, alpha=0.85)
        ax.errorbar(e2, y, xerr=L["e2"]["ci"], fmt="D", ms=5, color=BLUE,
                    ecolor=INK, elinewidth=0.9, capsize=2, mec="white", mew=0.8,
                    zorder=4)
        ax.plot(e1, y, "o", ms=5.5, color=GREEN, mec="white", mew=0.8, zorder=3)
        ax.plot(e0, y, "s", ms=5, color=VERM, mec="white", mew=0.8, zorder=3)
        ax.annotate(f"{frac['mean'] * 100:.1f}% $\\pm$ {frac['ci'] * 100:.1f}%",
                    xy=(e0, y), xytext=(6, 0), textcoords="offset points",
                    va="center", fontsize=7, color=INK)

    ax.set_yticks(ys)
    ax.set_yticklabels([f"{TIER_LABEL[t]}\n{p.replace('Dirichlet ', 'Dir. ')}"
                        for t, p in cells], fontsize=7)
    ax.set_xlabel("validation loss (lower is better)")
    ax.set_xlim(1.975, 2.115)
    ax.margins(y=0.18)
    vgrid(ax, axis="x")
    ax.legend(handles=[
        Line2D([], [], marker="o", ls="", color=GREEN, ms=5.5, mec="white",
               label="E1 centralized (upper bound)"),
        Line2D([], [], marker="D", ls="", color=BLUE, ms=5, mec="white",
               label="E2 FedAvg ($\\pm$95% CI)"),
        Line2D([], [], marker="s", ls="", color=VERM, ms=5, mec="white",
               label="E0 local-only (lower bound)"),
        Line2D([], [], color=BLUE, lw=5, alpha=0.85, label="gap recovered"),
    ], loc="lower left", ncol=2, columnspacing=1.1, handletextpad=0.5,
        bbox_to_anchor=(0.0, 1.0))
    save(fig, out, "fig2_gap_recovery")


def fig_trajectory(d: Data, out: Path) -> None:
    """F3 - the budget figure, which doubles as a picture of the no-op result.

    E2, E3 and E4 are bit-identical, so they are one line, labelled as such.
    Every curve is still descending at R=3.

    IID only, deliberately. The reference lines are the IID E0 and E1 arms, and
    overlaying the Dirichlet trajectory on them would confound the partition
    with federation - the same confound that makes E5 alone license no learning
    claim. The skewed cells are Fig. 2's job, where each is drawn against its
    own matched bounds.
    """
    fig, axes = plt.subplots(1, 2, figsize=(FULL, 2.5), sharex=True,
                             gridspec_kw={"wspace": 0.2})
    for ax, tier in zip(axes, TIERS):
        iid = d.trajectory(tier, "exp2_fl")
        x = np.arange(1, len(iid) + 1)
        ax.plot(x, iid, "-o", ms=4.5, color=BLUE, mec="white", mew=0.7,
                label="E2/E3/E4 FedAvg\n(bit-identical arms)", zorder=3)
        for exp, sub, col, lab in (("exp0_local", "", VERM, "E0 local-only"),
                                   ("exp1_sft", "", GREEN, "E1 centralized")):
            v = float(np.mean(d.final_loss(tier, exp, sub)))
            ax.axhline(v, color=col, ls=(0, (4, 2)), lw=1.0, zorder=2)
            ax.annotate(lab, xy=(1.02, v), xytext=(0, 2.5), textcoords="offset points",
                        fontsize=6.5, color=col)
        ax.set_title(TIER_LABEL[tier], pad=5)
        ax.set_xticks(x)
        ax.set_xlabel("federated round")
        vgrid(ax)
    axes[0].set_ylabel("validation loss")
    axes[0].legend(loc="upper right", fontsize=6.5, labelspacing=0.7)
    save(fig, out, "fig3_round_trajectory")


def fig_systems(d: Data, out: Path) -> None:
    """F4 - the cost decomposition: anchoring is free in bytes, IPFS is not."""
    arms = ["E2 FedAvg", "E3 +chain", "E4 +chain+IPFS"]
    colors = [GRID, BLUE, ORANGE]
    S = {(r["tier"], r["arm"]): r for r in d.N["systems"]}

    fig, axes = plt.subplots(1, 2, figsize=(FULL, 2.4),
                             gridspec_kw={"wspace": 0.28})
    w, xs = 0.26, np.arange(len(TIERS))

    for j, arm in enumerate(arms):
        off = (j - 1) * (w + 0.02)
        comm = [S[(t, arm)]["comm_mib"] for t in TIERS]
        gas = [S[(t, arm)]["gas"] / 1e6 for t in TIERS]
        axes[0].bar(xs + off, comm, w, color=colors[j], label=arm,
                    edgecolor="white", lw=0.8, zorder=3)
        axes[1].bar(xs + off, gas, w, color=colors[j], edgecolor="white",
                    lw=0.8, zorder=3)
        for x, t in zip(xs + off, TIERS):
            pct = S[(t, arm)]["comm_overhead_pct"]
            axes[0].annotate(f"{pct:+.1f}%", xy=(x, S[(t, arm)]["comm_mib"]),
                             xytext=(0, 2), textcoords="offset points",
                             ha="center", fontsize=6.2,
                             color=INK if pct else MUTED)
            # A zero bar draws nothing, and "FedAvg anchors nothing" is the
            # whole point of the panel - so it is labelled rather than absent.
            g = S[(t, arm)]["gas"]
            axes[1].annotate("0" if not g else f"{g / 1e6:.2f}M",
                             xy=(x, g / 1e6), xytext=(0, 2),
                             textcoords="offset points", ha="center",
                             fontsize=6.2, color=INK if g else MUTED)

    axes[0].set_ylabel("communication volume (MiB)")
    axes[0].set_ylim(0, 470)
    axes[1].set_ylabel("gas used (millions)")
    axes[1].set_ylim(0, 4.4)
    for ax in axes:
        ax.set_xticks(xs)
        ax.set_xticklabels([TIER_LABEL[t] for t in TIERS])
        vgrid(ax)
    axes[0].legend(loc="upper left", ncol=1, fontsize=6.8)
    save(fig, out, "fig4_systems_cost")


def fig_gas_scaling(d: Data, out: Path) -> None:
    """F5 - anchoring cost is linear in participants and flat in artefact size.

    One tier only: the sweep anchors 32-byte digests and never loads a model, so
    both tiers produce byte-identical numbers and plotting both would imply two
    independent measurements.
    """
    m = json.loads((d.results / "qwen-0.5b" / "exp7_scalability_metrics.json")
                   .read_text(encoding="utf-8"))
    fit = next(r for r in d.N["gas_scaling"]
               if r["tier"] == "qwen-0.5b" and r["sweep"] == "clients")

    fig, axes = plt.subplots(1, 2, figsize=(FULL, 2.4),
                             gridspec_kw={"wspace": 0.26})

    n = np.array([r["num_clients"] for r in m["clients_sweep"]])
    g = np.array([r["gas_per_round"] for r in m["clients_sweep"]]) / 1e6
    xs = np.linspace(0, 105, 100)
    axes[0].plot(xs, (fit["intercept"] + fit["slope"] * xs) / 1e6, "-",
                 color=MUTED, lw=1.0, zorder=2,
                 label=(f"gas $=$ {fit['intercept']:,.0f} $+$ {fit['slope']:,.0f}$N$\n"
                        f"$R^2$ = {fit['r2']:.6f}"))
    axes[0].plot(n, g, "o", ms=5, color=BLUE, mec="white", mew=0.8, zorder=3,
                 label="measured")
    axes[0].set_xlabel("participants $N$")
    axes[0].set_ylabel("gas per round (millions)")
    axes[0].set_xlim(0, 105)
    axes[0].legend(loc="upper left", fontsize=6.8)
    vgrid(axes[0])

    mb = np.array([r["adapter_mb"] for r in m["payload_sweep"]])
    gp = np.array([r["gas_used"] for r in m["payload_sweep"]]) / 1e3
    axes[1].plot(mb, gp, "-D", ms=4.5, color=ORANGE, mec="white", mew=0.8,
                 zorder=3, label="measured")
    axes[1].set_xscale("log")
    axes[1].set_xticks(mb)
    axes[1].set_xticklabels([f"{v:g}" for v in mb])
    axes[1].minorticks_off()
    axes[1].set_xlabel("artefact size (MiB, log scale)")
    axes[1].set_ylabel("gas per anchor (thousands)")
    axes[1].set_ylim(0, 620)
    axes[1].axhline(311.45, color=MUTED, ls=(0, (4, 2)), lw=0.9, zorder=2)
    axes[1].annotate("32 bytes anchored at every size;\n"
                     "spread 0.0077% over a 220$\\times$ range",
                     xy=(mb[1], 311.45), xytext=(0, 12),
                     textcoords="offset points", fontsize=6.5, color=INK)
    vgrid(axes[1])
    save(fig, out, "fig5_gas_scaling")


def fig_tamper(d: Data, out: Path) -> None:
    """F6 - detection and the benign control, drawn with their exact bounds.

    The point estimates are 100% and 0%; the content of the figure is the
    Clopper-Pearson bound at 50 trials, so the bars carry it explicitly.
    """
    order = ["bitflip", "scale", "substitute", "replay", "reserialize"]
    rows = {(r["tier"], r["perturbation"]): r for r in d.N["tamper"]}
    marks = {TIERS[0]: "o", TIERS[1]: "^"}

    # Bars are the wrong mark here: every outcome is 0 or 100, so a bar encodes
    # nothing the label does not, and the zero-height control bar would vanish.
    # What the figure is actually about is the interval around each outcome.
    fig, ax = plt.subplots(figsize=(FULL, 2.2))
    ys = np.arange(len(order))[::-1]

    for y, pert in zip(ys, order):
        ctrl = rows[(TIERS[0], pert)]["benign_control"]
        col = VERM if ctrl else BLUE
        lo, hi = cp_bounds(rows[(TIERS[0], pert)]["flagged"],
                           rows[(TIERS[0], pert)]["trials"])
        ax.plot([lo * 100, hi * 100], [y, y], color=col, lw=4.5, alpha=0.28,
                solid_capstyle="butt", zorder=2)
        for i, tier in enumerate(TIERS):
            r = rows[(tier, pert)]
            ax.plot(100 * r["flagged"] / r["trials"], y + (0.5 - i) * 0.22,
                    marks[tier], ms=5, color=col, mec="white", mew=0.8, zorder=4)
        r = rows[(TIERS[0], pert)]
        txt = (f"{r['flagged']}/{r['trials']} both tiers    "
               + (f"miss $\\leq$ {(1 - lo) * 100:.1f}%" if r["flagged"]
                  else f"FPR $\\leq$ {hi * 100:.1f}%"))
        ax.annotate(txt, xy=(112, y), va="center", ha="left", fontsize=6.5,
                    color=INK, annotation_clip=False)

    ax.axvline(107, color=GRID, lw=0.6, zorder=1)
    ax.set_yticks(ys)
    ax.set_yticklabels([f"{p}\n(benign control)" if p == "reserialize" else p
                        for p in order], fontsize=7)
    ax.set_xlabel("flagged by the integrity check (%)")
    ax.set_xlim(-4, 107)
    ax.set_xticks([0, 25, 50, 75, 100])
    ax.margins(y=0.16)
    vgrid(ax, axis="x")
    ax.legend(handles=[
        Line2D([], [], marker="o", ls="", color=BLUE, ms=5, mec="white",
               label="transport attack (must be flagged)"),
        Line2D([], [], marker="o", ls="", color=VERM, ms=5, mec="white",
               label="benign re-serialization (must not be)"),
        Line2D([], [], color=MUTED, lw=4.5, alpha=0.28,
               label="exact one-sided 95% interval, $n$=50"),
        Line2D([], [], marker=marks[TIERS[0]], ls="", color=MUTED, ms=5,
               mec="white", label=TIER_LABEL[TIERS[0]]),
        Line2D([], [], marker=marks[TIERS[1]], ls="", color=MUTED, ms=5,
               mec="white", label=TIER_LABEL[TIERS[1]]),
    ], loc="lower left", bbox_to_anchor=(0.0, 1.0), ncol=3, fontsize=6.5,
        handlelength=1.4, columnspacing=1.0)
    save(fig, out, "fig6_tamper_detection")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results-dir", default="results", type=Path)
    ap.add_argument("--out", default=Path("paper") / "figures", type=Path)
    args = ap.parse_args()

    root = Path(__file__).resolve().parent.parent
    results = args.results_dir if args.results_dir.is_absolute() else root / args.results_dir
    out = args.out if args.out.is_absolute() else root / args.out

    style()
    d = Data(results)
    print(f"Rendering to {out}")
    fig_heterogeneity(root, out)
    fig_gap_recovery(d, out)
    fig_trajectory(d, out)
    fig_systems(d, out)
    fig_gas_scaling(d, out)
    fig_tamper(d, out)
    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
