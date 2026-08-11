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
| 05 | [Ablation design](05_ablation_design.md) | The new experiments, with hypotheses and predictions fixed in advance |
| 06 | [Ablation results](06_ablation_results.md) | **Empty template.** Filled as runs land. |
| 07 | [Ablation conclusions](07_ablation_conclusions.md) | **Empty template.** Decision rules pre-registered. |
| 08 | [Shortcomings & roadmap](08_shortcomings_and_roadmap.md) | Every known weakness, ranked by how much it threatens the paper |
| 09 | [Run guide](09_run_guide.md) | **The commands to execute**, in order, on the GPU box |
| 10 | [Two-model results](10_two_model_results.md) | **Qwen2.5-0.5B vs SmolLM2-360M** — the motivation gap closes with scale |
| 11 | [Final status](11_final_status.md) | **Start here.** Evidence ledger, what each claim rests on, and the remaining run plan |

## Status

| Stage | State |
|---|---|
| Baseline run (smollm2-360m, 3 seeds, E0–E7) | **complete** — documented in 01–03 |
| Instrumentation changes C1–C7 | C1, C5, C6, C7 applied; **C3/C4 implemented but never enabled in any run** |
| `qwen-0.5b` tier (3 seeds, E0–E5, E6–E7) | **complete and clean** — analysed in 10 |
| Ablation runs A–F | **not started** — see 05 |

**The motivation gap is closed.** FedAvg recovered 2.4% of the isolation→
centralized gap at 360M and **34.0% at 0.5B** — so the benefit of federating
grows with model capacity, and there is something worth auditing after all. The
audit layer remains bit-identical to plain FedAvg at both scales (**18/18
global-model hashes**), and gas is byte-identical across tiers.

The clean 0.5B sweep of 2026-08-10 resolved all four defects the previous pass
carried: the VRAM leak is fixed (per-client training time now flat at 1.02×, was
7.43×), E6 runs on real adapters, E5 exists at 0.5B, and no run resumed. **0.5B
timing metrics are now the tightest in the study.**

**One confound survives:** E5 is Dirichlet-sharded while E0/E1/E2 are IID, so no
non-IID *learning* claim is licensed yet — only the systems claim. Ablation B1
(~12 h) fixes it and is the highest-value outstanding run. Ablation A has moved
back up: it no longer rescues the motivation, but the loss curve is still
descending at R=3, so it bounds the 34.0% headline. Full ledger in
[11](11_final_status.md).

Documents 06 and 07 contain **no results**. They are pre-registered templates:
the tables, the hypotheses, and the decision rules are written down *before* the
runs so that the analysis cannot drift toward whatever the data happens to show.
Every cell is marked `—` until a real run fills it. Do not quote them until then.

## The one-paragraph version

The audit layer is free and effective, at two model scales and two
architectures: E2/E3/E4 produce **bit-identical adapters** — 18/18 global-model
SHA-256 hashes match — so blockchain anchoring and IPFS transport cost exactly
zero accuracy, add a measured 0.45% wall-clock, and detect every tampered
artefact on real trained adapters at 0% false positives, at a gas cost linear in
clients (R² = 0.999994) and flat across a 220× model-size range. What the 360M
pass could not prove is that the thing being audited is worth doing — FedAvg beat
isolated training by only 2.4% of the distance to the centralized bound. **The
0.5B tier settled that: 34.0%**, budget-matched and tightly measured. What
remains is scope, not existence: that 34% is an IID number at a 3-round budget,
and the non-IID arm still lacks matched baselines. Ablation B1 closes the second
gap in ~12 h; Ablation A bounds the first.

## Provenance

All 360M numbers trace to `results/smollm2-360m/` (run 2026-08-04/05) and all
0.5B numbers to `results/qwen-0.5b/` (run 2026-08-09/10), produced on a single
NVIDIA T600 (4 GB) under WSL2 with a live anvil chain (id 31337) and a local
Kubo IPFS daemon. Nothing in documents 01–03 or 10–11 is estimated or
reconstructed; where a number is derived rather than read directly from a metrics
file, the derivation is shown.

Two directories are **not** evidence and must never be quoted:
`results/qwen-0.5b.leaky_backup/` (the VRAM-contaminated sweep, kept only for the
before/after comparison) and `results/_archive_prefix_20260804_222225/`.
