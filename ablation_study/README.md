# FedChain :: Ablation Study

This folder is the working record for the second experimental pass on FedChain.
It exists to carry the paper from "the system works" to "the system is worth
building", which the first pass established only halfway.

Read in order:

| # | Document | What it is |
|---|---|---|
| 01 | [Baseline design](01_baseline_design.md) | What the first pass ran and why each arm exists |
| 02 | [Baseline results](02_baseline_results.md) | Every number from all 3 seeds, as measured |
| 03 | [Baseline conclusions](03_baseline_conclusions.md) | What it proves, and what it conspicuously fails to prove |
| 04 | [Changes](04_changes.md) | Instrumentation fixes and config deltas applied before the new runs |
| 05 | [Ablation design](05_ablation_design.md) | The new experiments, with hypotheses fixed in advance — **carries a tier-calibration caveat** |
| 06 | [Ablation results](06_ablation_results.md) | **B1 and E are real data.** A, B2, B3, C, D, F are still `—` |
| 07 | [Ablation conclusions](07_ablation_conclusions.md) | **Written.** Which branch fired, and what the paper may and may not claim |
| 08 | [Shortcomings & roadmap](08_shortcomings_and_roadmap.md) | Every known weakness, ranked by how much it threatens the paper |
| 09 | [Run guide](09_run_guide.md) | **The commands to execute**, in order, on the GPU box |
| 10 | [Two-model results](10_two_model_results.md) | **Qwen2.5-0.5B vs SmolLM2-360M** — the motivation gap closes with scale |
| 11 | [Final status](11_final_status.md) | Evidence ledger, what each claim rests on, and the remaining run plan |
| 12 | [Paper plan](12_paper_plan.md) | **Start here now.** The 4–5 page route: what is novel, which table carries which claim, and the sentences that must not be written |

## Status

| Stage | State |
|---|---|
| Baseline run (smollm2-360m, 3 seeds, E0–E7) | **complete** — documented in 01–03 |
| Instrumentation changes C1–C7 | **all applied.** C3/C4 were finally exercised by Ablation E on 2026-08-13 |
| `qwen-0.5b` tier (3 seeds, E0–E5, E6–E7) | **complete and clean** — analysed in 10 |
| **Ablation B1** (α=0.3, 0.5B) | **complete** 2026-08-11/12 — `results/qwen-0.5b/ablation/` |
| **Ablation E** (re-score @250) | **complete** 2026-08-13 — `results/<tier>/reeval250` |
| **Ablation B1** (α=0.3, 360M) | **outstanding** — the last training run. `bash scripts/run_final.sh` |
| E6/E7 protocol parity at qwen | **outstanding** — minutes of compute, same script |
| Ablations A, B2, B3, C, D, F | **out of scope** for this paper — configs and runner blocks exist for a follow-up |

> **The programme is closed at two runs.** `bash scripts/run_final.sh` executes
> both, re-scores what they produce, and rebuilds every table, ending with a
> machine verdict (`COMPLETE` / `INCOMPLETE` / `HASH DIVERGENCE`). The paper is
> then written from `results/paper/` following [12](12_paper_plan.md).

**The motivation gap is closed, and skew widens it.** FedAvg recovered 2.4% of
the isolation→centralized gap at 360M and **34.0% ± 4.3% at 0.5B**; Ablation B1
then measured **41.5% ± 7.7%** under Dirichlet(0.3) at 0.5B, a **1.69×** larger
absolute gain with disjoint intervals, through the predicted mechanism —
isolated clients degrade 2.82× faster than the averaged model while the
centralized bound does not move at all.

The audit layer remains bit-identical to plain FedAvg at both scales and on both
partitions: **18/18 global-model hashes** under IID, **27/27 client adapter
hashes** under skew, with loss, perplexity, ROUGE-L and BLEU all equal to 6 dp.
Gas is byte-identical across tiers.

### Three corrections this pass produced

1. **The study's stated premise was a tier mismatch.** [05](05_ablation_design.md)
   motivates everything with "FedAvg recovers 2.4% — close to nothing", which is
   the *360M* figure, and calibrated every decision threshold on it. B1 ran at
   **0.5B**, where the IID baseline was already 34%. The branch structure
   therefore never had a live null case at the tier it was tested on. Recorded as
   a pre-registration defect, not rescaled —
   [06 §B.6](06_ablation_results.md#the-premise-correction).
2. **ROUGE-L/BLEU in `comparison.md` mix two scorers.** 30 main-table runs used
   the `builtin` fallback; B1's 6 used `evaluate`. Quote
   `results/<tier>/reeval250` and nothing else. Historical only — the
   `require_metric_backend` guard has been live since 2026-08-07 and the affected
   runs predate it.
3. **H-E1 was malformed.** Raising `gen_num_samples` 50 → 250 did not narrow the
   confidence intervals; three of four widened. The ±CI is a *between-seed*
   interval and sample count does not touch it. Tighter generation intervals
   need more **seeds**.

**What is still missing:** the non-IID contrast at 360M (the tier where FedAvg
did almost nothing under IID has no skewed arm), the α curve (B1 is one contrast,
not the ordering H-B1 predicted), and a convergence bound — every loss curve is
still descending at R=3. Full ledger in [11](11_final_status.md).

## The one-paragraph version

The audit layer is free and effective, at two model scales, two architectures and
two data partitions: E2/E3/E4 produce **bit-identical adapters** — 18/18
global-model SHA-256 hashes under IID, 27/27 client hashes under Dirichlet(0.3) —
so blockchain anchoring and IPFS transport cost exactly zero accuracy, add a
measured 0.45% wall-clock and **zero** communication attributable to anchoring
itself, and detect every tampered artefact on real trained adapters, at a gas cost
linear in clients (R² = 0.999994) and flat across a 220× model-size range. What
the 360M pass could not prove is that the thing being audited is worth doing —
FedAvg beat isolated training by only 2.4% of the distance to the centralized
bound. **The 0.5B tier settled that at 34.0%, and Ablation B1 raised it to 41.5%
under label skew.** What remains is scope, not existence: those are fixed-budget
numbers at R=3 with curves still descending, from a single α on a single tier.

## Provenance

All 360M numbers trace to `results/smollm2-360m/` (run 2026-08-04/05, E0 re-run
2026-08-12), all 0.5B numbers to `results/qwen-0.5b/` (2026-08-09/10), and all
B1 numbers to `results/qwen-0.5b/ablation/` (2026-08-11/12) — produced on a single
NVIDIA T600 (4 GB) under WSL2 with a live anvil chain (id 31337) and a local Kubo
IPFS daemon. Generation metrics quoted anywhere in these documents come from
`results/<tier>/reeval250` (2026-08-13). Nothing is estimated or reconstructed;
where a number is derived rather than read from a metrics file, the derivation is
shown.

Two directories are **not** evidence and must never be quoted:
`results/qwen-0.5b.leaky_backup/` (the VRAM-contaminated sweep, kept only for the
before/after comparison) and `results/_archive_prefix_20260804_222225/`. The
reporting tooling excludes both by name.
