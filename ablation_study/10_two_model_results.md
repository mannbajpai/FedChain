# 10 — Two-model results: SmolLM2-360M vs Qwen2.5-0.5B

Qwen2.5-0.5B, 3 seeds (42/43/44), E0–E5 + E6/E7. Source: `results/qwen-0.5b/`,
run 2026-08-09 21:59 → 2026-08-10 21:21 UTC. All numbers read directly from the
metrics files.

> **This document was rewritten after the clean sweep of 2026-08-10.** The
> previous version described a run contaminated by a VRAM leak and carried four
> data-quality defects. **All four are now resolved** — the leak is fixed and
> verified, E5 was run at 0.5B, E6 ran on real adapters, and no run resumed. The
> contaminated tier is preserved at `results/qwen-0.5b.leaky_backup/` for
> comparison and must not be quoted. Section 4 below records what changed.

---

## The headline: the motivation gap closes with model scale

The 360M pass could not show that federating was worth doing — FedAvg beat
isolated training by 2.4% of the distance to the centralized bound. **At 0.5B it
recovers 34%.**

| | SmolLM2-360M | Qwen2.5-0.5B |
|---|---|---|
| E0 Local-only | 2.0236 ± 0.0011 | 2.0783 ± 0.0007 |
| E1 Centralized | 1.9884 ± 0.0006 | 2.0499 ± 0.0016 |
| E2/E3/E4 Federated | 2.0228 ± 0.0013 | 2.0686 ± 0.0009 |
| E5 FedChain non-IID | 2.0249 ± 0.0006 | 2.0723 ± 0.0028 |
| **E2 − E0** (what federation buys) | **−0.00085 ± 0.00022** | **−0.00964 ± 0.00093** |
| **as % of the E0→E1 gap** | **2.4%** | **34.0%** |
| E2 − E1 (what federation costs) | +0.0343 ± 0.0012 | +0.0187 ± 0.0018 |

All arms are budget-matched at **4,500 sample-updates**: E0 is 3 clients × 3
rounds × 500, E1 is 1 round × 3 shards × 1,500 pooled, E2–E4 is 3 rounds ×
3 clients × 500. The comparison is fair at the update level by construction, not
by accident ([base_config.yaml:89-90](../configs/base_config.yaml#L89-L90)).

Both gaps move in the right direction at the larger model: federation buys **11×
more** and costs **45% less**. This is the outcome
[03_baseline_conclusions.md](03_baseline_conclusions.md) listed as explanation 3
— *"the effect is real and small at this model scale"* — and flagged as testable
by climbing the model ladder. It is now tested, and it holds.

**Where this sits in the pre-registration.** It sits *outside* it. The branch
structure in [07](07_ablation_conclusions.md) names rounds (B1), heterogeneity
(B2), neither (B3) and sign-flip (B4). **Model scale is not one of the four
branches**, and the analysis protocol says to declare that rather than fit a new
branch. Declared here. The mitigating fact is that scale was named as a
competing explanation in document 03 *before* this run, so it is a pre-stated
alternative hypothesis rather than a post-hoc rescue — but it was never given a
decision rule, and it should be reported as a confirmed prediction from 03, not
as a planned ablation outcome.

**How far to push it.** Two points on a model-size axis is a direction, not a
law. 2.4% → 34.0% is a large, consistent, tight-interval move, but do not fit a
curve to it or claim monotonicity. The honest statement is: *the benefit of
federation grows with model capacity over the range tested, and at 0.5B it is
substantial.* A third rung (qwen-1.5b) turns a direction into a trend.

---

## The core claim reproduces exactly

E2, E3 and E4 produce **bit-identical adapters** at 0.5B, as they did at 360M.
Verified on the anchored SHA-256 of every global model:

| Tier | Rounds × seeds | E2 == E3 == E4 |
|---|---|---|
| smollm2-360m | 3 × 3 | **9/9 identical** |
| qwen-0.5b | 3 × 3 | **9/9 identical** |

**18/18 global-model hashes identical.** Paired difference on validation loss,
E3 − E2 and E4 − E2: **exactly 0.000000, sd 0.000000**, at both tiers.

The audit layer is now shown to be a no-op on the learning math at **two model
scales and two architectures** (Llama-style SmolLM2, 448 LoRA tensors; Qwen2.5
with GQA, 336 LoRA tensors). This is the strongest claim in the paper. Lead with
it — hash equality is a stronger statement than a non-significant loss
difference, and it pre-empts the "absence of evidence is not equivalence"
objection entirely.

---

## Systems cost is model-independent — measured, not asserted

| | SmolLM2-360M | Qwen2.5-0.5B | Δ |
|---|---|---|---|
| Trainable LoRA params | 8,683,520 | 8,798,208 | +1.3% |
| LoRA tensors | 448 | 336 | — |
| Adapter size (MiB) | 16.620 | 16.825 | +1.2% |
| **E3 gas** | **2,997,464** | **2,997,464** | **0** |
| **E4 gas** | **3,785,372** | **3,785,372** | **0** |
| E2 communication (MiB) | 299.18 | 302.86 | +1.2% |
| E4 communication (MiB) | 393.56 | 399.32 | +1.5% |
| E4 comm overhead vs E2 | +31.5% | +31.8% | — |

**Gas is byte-identical across tiers** — 3,785,372 for E4 and 2,997,464 for E3 in
all twelve runs at both scales, every seed. Communication tracks adapter size,
exactly as it should when only a 32-byte digest reaches the chain and the adapter
itself moves over IPFS.

**Chain integrity across the whole 0.5B sweep:** 12/12 transactions successful in
each of the nine chain-enabled runs, 9/9 integrity checks passed in every E4 and
E5 run, **0 IPFS transfer failures**, and `sessions: 1` with `resumed: false`
everywhere. No anomalies to log.

### E7 — the scaling law, now fitted rather than asserted

| N clients | Tx/round | Gas/round | Gas/client |
|---|---|---|---|
| 1 | 2 | 616,560 | 616,560 |
| 3 | 4 | 1,163,296 | 387,765 |
| 5 | 6 | 1,744,914 | 348,983 |
| 10 | 11 | 3,198,971 | 319,897 |
| 25 | 26 | 7,561,286 | 302,451 |
| 50 | 51 | 14,831,811 | 296,636 |

**OLS fit: `gas = 301,120 + 290,533·N`, R² = 0.999994.** The intercept is the
global-model anchor; the slope is the true marginal cost of admitting one more
participant, and it converges to 290,821 gas by N=10. Gas per *client* falls
monotonically — the fixed cost amortises — which is the deployability argument
stated as a number.

Payload sweep: **220× adapter-size range (0.22 → 49.05 MiB) moves gas by 24
units, 0.0077%.** Anchored payload is 32 bytes at every size. Model size does not
reach the chain, by construction and by measurement.

### Wall-clock overhead — now sourceable from 0.5B

With the leak fixed, the 0.5B tier gives the **tighter** timing evidence,
reversing the previous guidance in this document:

| Comparison | SmolLM2-360M | Qwen2.5-0.5B |
|---|---|---|
| E3 − E2 (anchoring only) | +11.8 s ± 102.3 (+0.25%) | −8.9 s ± 36.4 (−0.19%) |
| E4 − E2 (anchoring + IPFS) | +33.9 s ± 167.1 (+0.71%) | **+7.6 s ± 10.6 (+0.16%)** |
| Measured audit work | 21.99 s (0.46%) | 21.61 s (0.45%) |

Neither difference is significant at either tier — the audit layer's cost is not
distinguishable from run-to-run noise. **State it as an upper bound**, not as a
point estimate: at 0.5B the 95% interval bounds the end-to-end overhead below
**+0.4%**. The directly measured audit work (chain 0.22 s + IPFS upload 18.46 s +
download 2.92 s = 21.61 s against a 4,780 s round total) agrees at 0.45% and is
the number to quote, because it is measured rather than differenced.

---

## 4. The four data-quality problems are resolved

### 1. VRAM leak — **fixed and verified**

`paged_adamw_8bit` routes every parameter through
`bitsandbytes.optim.GlobalOptimManager`, a process-global singleton whose
bookkeeping dicts pinned the parameter tensors. `_release_optimizer_registries()`
in [trainer/sft.py](../trainer/sft.py) clears it. Measured on seed 42's E4, nine
consecutive client trainings:

| | Contaminated run | Clean run |
|---|---|---|
| Per-client training time, first → last | 449.5 s → 3,340.6 s | 447.1 s → 457.4 s |
| Ratio | **7.43×** | **1.02×** |
| Eval peak VRAM, r1 → final | 2,949 → 6,069 MB | **1,389.55 MB, constant** |
| E0 training-time spread across seeds | 2.4× | **1.00×** |

Training-time spread across seeds is now 1.00–1.01× on every arm. **0.5B timing
metrics are usable and are used above.**

### 2. E6 on synthetic adapters — **fixed**

`adapter_source` is now
`outputs/qwen-0.5b/seed_42/exp4_fedchain` — 12 real trained adapters, not
synthetic ones.

| | SmolLM2-360M | Qwen2.5-0.5B |
|---|---|---|
| Adapter source | 12 real E4 adapters | **12 real E4 adapters** |
| Trials per attack | 50 | **20** |
| Malicious detected | 200/200 (100%) | **80/80 (100%)** |
| False positives | 0/50 (0%) | **0/20 (0%)** |

Attacks are bitflip, scale, substitute and replay; `reserialize` is the benign
control. Detection is perfect at both tiers. **The remaining defect is protocol
parity, not validity:** 20 trials/attack instead of 50. One-sided 95% binomial
bounds on the zero-failure counts:

| | Miss rate (attacks) | False-positive rate (benign) |
|---|---|---|
| smollm2-360m | 200 trials → **≤ 1.5%** | 50 trials → **≤ 5.8%** |
| qwen-0.5b | 80 trials → **≤ 3.7%** | 20 trials → **≤ 13.9%** |

The FP claim is the weaker half at both tiers and is barely a claim at all at
0.5B. Re-run at 50 trials for parity; it costs minutes and tightens the 0.5B FP
bound from 13.9% to 5.8%.

### 3. No non-IID arm at 0.5B — **fixed, but see the caveat below**

E5 now exists for all three seeds at 0.5B, on `data/dirichlet/client*.jsonl`.
Gas, integrity and transfer counts are identical to E4. **Its learning result is
not yet interpretable** — see the confound in the next section.

### 4. Seed 44's resumed `exp3_fl_bc` — **gone**

Every run in the new sweep is `sessions: 1`, `resumed: false`, and E3 gas is
2,997,464 for all three seeds (it was 3,011,764 on the resumed run). No systems
row needs a footnote now.

---

## The one confound that survives: E5 has no matched baseline

E5 is Dirichlet(0.3)-sharded. E0, E1 and E2 are IID-sharded. Any E5 − E0 or
E5 − E2 difference therefore mixes **the partition change with the federation
change**, and cannot be attributed to either.

| | SmolLM2-360M | Qwen2.5-0.5B |
|---|---|---|
| E5 − E2 (reads as "non-IID penalty") | +0.00212 ± 0.00070 | +0.00364 ± 0.00345 |
| E5 − E0 (reads as "federation benefit, non-IID") | **+0.00127 ± 0.00049** | **−0.00600 ± 0.00341** |

Read naively these say federation *loses* to isolation under skew at 360M
(−3.6% of the gap) and wins at 0.5B (+21.2%) — a clean scale story. **Do not
report it.** Both arms of the comparison changed at once. The analysis protocol
in [05](05_ablation_design.md#analysis-protocol) rule 6 and
[EXPERIMENTS.md:23](../EXPERIMENTS.md#L23) both prohibit exactly this comparison,
and [C5](04_changes.md#c5--non-iid-baselines-for-e0-e1-and-e2) exists to fix it.

**What E5 does license today:** the audit layer behaves identically under label
skew — 12/12 transactions, 9/9 integrity checks, gas byte-identical to E4 at both
tiers. That is a systems claim and it is sound. The learning claim needs
Ablation B1, and B1 is now the single highest-value outstanding run.

Note also that E5 is *label* skew with **balanced quantities**:
`max_train_samples: 500` caps every client below the smallest Dirichlet shard
(4,798 / 2,715 / 6,998), so all three contribute 500 samples and FedAvg weights
come out uniform despite `fedavg_weighted: true`. Describe it that way in the
paper.

---

## Two statistical caveats a reviewer will find

**1. The seeds do not vary the data partition.** All three seeds read the same
`data/client*.jsonl` shards; the seed changes LoRA init, shuffling and dropout
only. The ±0.0009 intervals therefore measure *training noise*, not sampling
variability, and **understate the true uncertainty** on any claim about
partitions. Say so, or add a partition-reseeded arm. This is the most attackable
number in the paper.

**2. Three rounds is not converged.** Per-round validation loss, mean over seeds:

| Round | SmolLM2-360M | Qwen2.5-0.5B |
|---|---|---|
| 1 | 2.0834 | 2.1145 |
| 2 | 2.0388 (−0.0446) | 2.0792 (−0.0353) |
| 3 / final | 2.0228 (−0.0160) | 2.0686 (−0.0106) |

Both curves are still descending when the budget ends, with a per-round decay
ratio near 0.30–0.36. Note also that at round 1 the federated model is *worse*
than the local-only final (2.1145 vs 2.0783 at 0.5B) — FedAvg only overtakes
isolation by round 3. So **34.0% is the value at R=3, not at convergence**, and
whether it grows is exactly what Ablation A was designed to answer. A has not
been run. Do not describe 34.0% as the converged benefit; describe it as the
benefit at a matched 4,500-update budget.

---

## Claims status after two models

| Claim | Status | Evidence |
|---|---|---|
| Anchoring + IPFS cost zero accuracy | **solid at 2 scales** | 18/18 identical hashes, 12 runs |
| Gas independent of model size | **solid, measured** | 2,997,464 / 3,785,372 at both tiers |
| Communication tracks adapter size | **solid** | +1.2% adapter → +1.5% comm |
| Federation beats isolation | **supported at 0.5B** | 34.0% of the gap, ±0.00093 |
| Cost of federation shrinks with scale | **supported, 2 points** | 0.0343 → 0.0187 |
| Wall-clock overhead sub-1% | **solid at both tiers** | +0.16% ± 0.22% at 0.5B |
| 100% detection / 0% FP | **solid; FP interval weak at 0.5B** | E6, real adapters both tiers |
| Gas linear in N | **solid, fitted** | R² = 0.999994 |
| Non-IID robustness (systems) | **solid** | E5, 9/9 integrity both tiers |
| Non-IID robustness (learning) | **measured at 0.5B** (2026-08-12) | B1: 41.5% ± 7.7% of the gap at α=0.3 — [06 §B](06_ablation_results.md#b--data-heterogeneity) |
| Benefit at convergence | **not measured** | needs Ablation A |
| Generation metrics usable | **partially** (2026-08-13) | Ablation E: centralized-vs-rest only, at 0.5B only — [06 §E](06_ablation_results.md#e--evaluation-fidelity) |

---

## Stale artefacts — status as of 2026-08-13

1. ~~**`results/smollm2-360m/` E0 reports 299.202 MiB communication.**~~
   **Resolved 2026-08-12.** The 360M E0 arm was re-run post-[C1](04_changes.md#c1--fix-phantom-communication-accounting-in-the-local-only-arm)
   and now reports **0.000 MiB**, matching qwen-0.5b. The phantom bytes are gone
   from the cross-model table.
2. ~~**C3 and C4 implemented but never enabled.**~~ **Resolved 2026-08-13** by
   Ablation E — but not the way this document expected. Re-scoring at
   `gen_num_samples: 250` with `require_metric_backend: evaluate` produced usable
   numbers only for the *centralized-vs-rest* contrast at 0.5B, and **the
   intervals did not narrow** (they are between-seed intervals; sample count does
   not touch them). Two further findings, both in
   [06 §E](06_ablation_results.md#e--evaluation-fidelity):
   - **All 30 main-table runs used the `builtin` scorer while B1's 6 arms used
     `evaluate`.** ROUGE-L/BLEU in `results/*/comparison.md` are therefore *not
     comparable* between the main and ablation tables. Quote
     `results/<tier>/reeval250` and nothing else.
   - Those runs stored `require_metric_backend: ''`; the guard that makes a
     missing metric stack a hard failure landed 2026-08-07 and every run since
     has stored `'evaluate'`. **Historical contamination, not a live defect.**
3. **`results/qwen-0.5b.leaky_backup/`** — keep for the leak comparison, exclude
   from every table. Now enforced by the tooling, not by memory.
4. **E7 protocol parity** — **still open.** 360M swept to N=100, 0.5B stopped at
   N=50. `finish_study.sh` skipped this on 2026-08-13 because no chain was
   running and `anvil` was not on PATH. Minutes of compute once it is.

---

## What I'd do next, in order

Items 1, 2 and 4 of the previous list are **done** (B1, Ablation E, 360M E0
re-run). What remains, re-ranked:

1. **E6 at 50 trials and E7 to N=100 for qwen-0.5b** (~minutes) — protocol
   parity, and it tightens the false-positive bound from 13.9% to 5.8%. Blocked
   only on starting a local chain. **Cheapest outstanding item by far; do it
   first.**
2. **Ablation B1 at smollm2-360m** (~12 h) — the tier where FedAvg recovered only
   2.4% under IID has **no non-IID arm**, so the "skew rescues a near-zero
   baseline" contrast does not exist in this study. B1 at 0.5B started from an
   already-open 34% gap. This is now the highest-value *training* run, and it
   supersedes the previous ranking — see
   [06 §B.6](06_ablation_results.md#the-premise-correction).
3. **Ablation A at 0.5B** (~37 h) — the trajectory is unconverged, so 34.0% and
   41.5% are lower bounds of unknown tightness on claims the paper leads with.
4. **Ablation D** (~3.3 h, 1 seed) — attributes the +31.8% communication
   overhead. E3 already establishes that anchoring contributes **zero** of it and
   the whole of it arrives with IPFS; D2 splits it *within* IPFS.
5. **qwen-1.5b** — the 5.2 GB peak that made it hopeless was the leak; a clean
   run holds 1.39 GB at 0.5B. Turns the scale trend from two points into three.
6. **Ablations B2/B3 (α ∈ {0.1, 1.0})** — B1 gives one contrast, which is a
   direction. These make it a curve and are what H-B1 actually predicted.
