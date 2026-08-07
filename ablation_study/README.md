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

## Status

| Stage | State |
|---|---|
| Baseline run (smollm2-360m, 3 seeds, E0–E7) | **complete** — documented in 01–03 |
| Instrumentation changes C1–C7 | **applied and verified** — see 04 |
| Ablation runs A–F | **not started** — needs a CUDA box; see 05 |
| `qwen-0.5b` tier | **not started** — needs a CUDA box |

The changes are in and tested (33 tests, `python -m pytest tests/`). What remains
is GPU time: this repo checkout has no CUDA device, no `trl`/`bitsandbytes`/`web3`,
and no anvil or IPFS daemon, so no training run can execute here. Every command
below is written to run on the machine that produced the baseline.

Documents 06 and 07 contain **no results**. They are pre-registered templates:
the tables, the hypotheses, and the decision rules are written down *before* the
runs so that the analysis cannot drift toward whatever the data happens to show.
Every cell is marked `—` until a real run fills it. Do not quote them until then.

## The one-paragraph version

The baseline pass proved the audit layer is free and effective: E2/E3/E4 produce
**bit-identical adapters**, so blockchain anchoring and IPFS transport cost
exactly zero accuracy, add <1% wall-clock, and detect 200/200 tampered artefacts
with 0/50 false positives, at a gas cost linear in clients and flat in model
size. What it did *not* prove is that the thing being audited is worth doing:
FedAvg beat isolated local training by 0.00085 nats — about 2.4% of the distance
to the centralized upper bound. The ablation study exists almost entirely to
close that gap, by testing whether the FedAvg advantage grows with **rounds**
(Ablation A) and with **data heterogeneity** (Ablation B), which are the two
regimes where federation is supposed to pay for itself.

## Provenance

All baseline numbers here trace to `results/smollm2-360m/`, produced on a single
NVIDIA T600 (4 GB) under WSL2 with a live anvil chain (id 31337) and a local
Kubo IPFS daemon. Nothing in documents 01–03 is estimated or reconstructed;
where a number is derived rather than read directly from a metrics file, the
derivation is shown.
