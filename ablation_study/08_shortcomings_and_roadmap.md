# 08 — Shortcomings and roadmap

Every known weakness, ranked by how much it threatens the paper. Each entry
states the problem, **how a reviewer will phrase the objection**, the fix, and
the cost.

Severity key: **S1** = threatens acceptance · **S2** = weakens a claim ·
**S3** = presentation.

> **Revised 2026-08-13.** S1-1, S1-2, S1-3 and S2-5 are **resolved**; S2-1 and
> S2-3 are downgraded. Resolved entries are kept with their resolution recorded
> rather than deleted, because the paper's limitations section is written from
> this file and a reviewer may raise a defect that was fixed rather than absent.
> Two **new** entries — S1-4 and S2-6 — came out of the resolutions.

---

## ~~S1-1~~ — Federation barely beats doing nothing · **RESOLVED**

**The problem, as it stood.** `E2 − E0 = −0.00085 ± 0.00022` val loss at
smollm2-360m: FedAvg over three clients recovered 2.4% of the
isolation→centralized gap.

**Resolution.** Not by an ablation but by the **model-ladder tier**. At
qwen-0.5b the same quantity is `−0.00964 ± 0.00093` = **34.0% ± 4.3%** of the
gap, and Ablation B1 raised it to **41.5% ± 7.7%** under Dirichlet(0.3). The
question "what does the system buy?" now has a measured answer at 0.5B.

**What still needs saying in the paper.** These are fixed-budget numbers at R=3
with the loss curves still descending, and the 360M tier remains near-null —
which is a scale result worth reporting honestly rather than burying. Ablation A
would bound the first caveat.

---

## ~~S1-2~~ — The non-IID arm has no matched baseline · **RESOLVED**

**The problem, as it stood.** E5 was the only Dirichlet run, so every E5−E0 and
E5−E2 difference confounded the partition with federation — prohibited by the
repo's own design doc.

**Resolution.** **Ablation B1**, run at qwen-0.5b on 2026-08-11/12: E0, E1 and E2
on the same α=0.3 shards, 3 seeds. Results in
[06 §B](06_ablation_results.md#b--data-heterogeneity). The comparison is now
matched, and it carries a partition-invariance control — E1 reads 2.0499 (IID)
against 2.0492 (α=0.3), i.e. the centralized bound does not move when the corpus
is re-split, which is what validates the Dirichlet shards.

**Residual.** B1 ran at **0.5B only**. See S1-4.

---

## S1-4 — The skew result exists at one tier, and not the one that motivated it

**The problem.** The argument for testing heterogeneity was that FedAvg recovered
only 2.4% of the gap — a *360M* figure. B1 was executed at **qwen-0.5b**, where
FedAvg already recovered 34% before any skew was added. So the study shows skew
*widening an open margin*, never *rescuing a null one*. The tier where the null
actually occurred has no non-IID arm at all.

Worse, [05](05_ablation_design.md)'s decision thresholds were calibrated on the
360M numbers and never re-derived, so B1's pre-registered "nothing is happening"
branch was never live at the tier it ran on. That is a pre-registration defect,
recorded in
[07](07_ablation_conclusions.md#before-the-branches-the-thresholds-were-calibrated-at-the-wrong-tier).

**How a reviewer will phrase it.**
> *The paper motivates the non-IID experiment by observing that federation barely
> helps, then runs it at a scale where federation already helps considerably. The
> heterogeneity claim rests on a single α at a single model size.*

**The fix.** Run B1 at smollm2-360m (~12 h), and re-derive the thresholds for
that tier before starting. B2/B3 (α ∈ {0.1, 1.0}) turn the one contrast into the
ordering H-B1 actually predicted; configs and runner blocks now exist for both.

**Cost.** 12 h for the tier contrast, +32 h for the α curve. **The 12 h is the
higher-value half.**

---

## ~~S1-3~~ — Local-only reports phantom communication · **RESOLVED**

**Resolution.** [C1](04_changes.md#c1--fix-phantom-communication-accounting-in-the-local-only-arm)
landed 2026-08-06 and the 360M E0 arm — the last one still carrying the stale
figure — was re-run 2026-08-12. **Both tiers now report 0.000 MiB for E0.** The
regression test asserting `total_communication_mb == 0.0` whenever
`aggregation_enabled is False` enforces it rather than leaving it to memory. The
original entry is kept below because the cost analysis rests on the communication
metric, and a reviewer may probe it.

**The problem, as it stood.** E0 reported 299.20 MiB of communication with no server, no
aggregation and no recipient — the same volume as E2, which actually federates.
Traced exactly to [trainer/federated.py:396-409](../trainer/federated.py#L396-L409):
publish-and-broadcast accounting runs outside the `enable_aggregation` guard, so
client 1's own adapter is billed as a global-model upload and then broadcast to
three clients who never requested it. Every client's `download_bytes` is `0`.

**How a reviewer will phrase it.**
> *Why does the no-aggregation baseline transmit the same 299 MiB as the
> federated system? Either the baseline is not what it claims, or the
> communication metric is not measuring communication.*

Both readings are damaging, and the second casts doubt on the +31.5% number that
the cost analysis rests on.

**The fix.** [C1](04_changes.md#c1--fix-phantom-communication-accounting-in-the-local-only-arm),
plus a regression test asserting `total_communication_mb == 0.0` whenever
`aggregation_enabled is False`.

**Cost.** ~1 h of code. E0's column becomes 0.00 MiB. No other arm moves.

---

## S2-1 — Generation metrics cannot support any claim · **PARTIALLY RESOLVED**

**The problem.** At `gen_num_samples: 50`, ROUGE-L and BLEU CIs span 5–17% of
their values. The rows are in the tables and invite comparison they cannot
support.

**Resolution.** Ablation E re-scored every adapter at 250 samples on the
`evaluate` backend. Outcome, from
[06 §E](06_ablation_results.md#e--evaluation-fidelity):

- At **qwen-0.5b**, E1 (centralized) significantly beats E0, E2 and E5, and the
  ROUGE-L ordering now agrees with the loss ordering. That contrast is quotable.
- At **360M**, nothing separates.
- **The contrast the motivation needs is still unresolved**: E2 − E0 under skew
  is +0.0163 on loss (significant) and −0.0008 ± 0.0044 on ROUGE-L (nothing).

**Standing position.** Loss and perplexity are the accuracy result. ROUGE-L/BLEU
@250 appear as a supporting collapse check with sample count and backend stated,
quoted for the centralized-vs-rest gap only, and **only from
`results/<tier>/reeval250`** (see S2-5).

---

## S2-6 — Generation confidence intervals do not respond to sample count

**The problem.** [05](05_ablation_design.md) predicted (H-E1) that going from 50
to 250 generation samples would narrow the CIs by `sqrt(5) = 2.24×`. Measured
ratios: **0.08×, 0.43×, 0.63×, 1.86× — three of four widened.**

The prediction confounded two different variances. The reported ±CI is a
**between-seed** interval at n=3; `gen_num_samples` controls Monte-Carlo error
*inside* each seed's point estimate, which the interval does not measure. The
interval is floored by seed variance at any sample count.

**How a reviewer will phrase it.**
> *The authors report ±CI on ROUGE-L over three seeds and describe 250 samples as
> improving precision. Those intervals are dominated by seed variance; the sample
> count is not what is being averaged over.*

**The fix.** State plainly what the interval is over. Where tighter generation
intervals are needed, add **seeds**, not samples. Do not describe the 50 → 250
change as having tightened anything — describe it as de-noising the per-seed
estimates, which is what made the qwen ordering coherent.

**Cost.** Free to disclose. Additional seeds cost a full matrix run each.

---

## S2-2 — The +31.5% communication overhead is unattributed · **DOWNGRADED**

**The problem.** E4 costs 399.3 MiB against E2's 302.9 (qwen-0.5b). One number,
no decomposition, so it reads as an intrinsic cost of auditability.

**Half-resolved from arms already in hand.** **E3 records 302.86 MiB —
byte-identical to E2.** Anchoring therefore adds *zero* communication, and the
entire +31.8% arrives with IPFS transport. That is the part of the objection that
actually bites, and it can be answered today without Ablation D.

**What remains.** The split *within* IPFS — client uploads vs the aggregator's
download leg vs the global model's round-trip. Until D2 runs, say "the overhead
is IPFS transport, decomposition pending" rather than leaving it unattributed.

**Cost.** 3.3 h, systems-only, one seed.

---

## S2-3 — Single model tier · **RESOLVED**

**The problem, as it stood.** Everything was `smollm2-360m`, the pipeline
shakedown tier.

**Resolution.** `qwen-0.5b` was run in full (E0–E7, 3 seeds) and is now the tier
the paper leads with; 360M is the second rung. The two-point ladder is what
closed S1-1.

**What still needs saying.** **Two points is a direction, not a law** — no
monotonicity claim and no fitted curve across tiers. The audit-layer conclusions
are genuinely scale-free and should be stated as such: hash equality is a
property of the commitment scheme, and E7 shows gas is flat across a 220×
artefact-size range. A third rung (`qwen-1.5b`, ~24 h) would turn the direction
into a trend; the 5.2 GB peak that previously made it infeasible was the VRAM
leak, now fixed (1.39 GB at 0.5B).

---

## S2-4 — Three seeds, and they share one partition

**The problem.** `t = 4.303` at n=3. Every interval is wide by construction.
**And all three seeds read the same shard files** — the seed varies LoRA init,
shuffling and dropout, never the partition. Every ±CI in this study is therefore
a *training-noise* interval, not a sampling interval, and understates true
uncertainty on any partition-dependent claim.

**Assessment.** The partition half is now the more serious half, because the
headline B1 result *is* partition-dependent: 41.5% ± 7.7% at α=0.3 rests on one
Dirichlet draw. A different draw at the same α would give a different number, and
nothing in the study bounds that spread.

**How a reviewer will phrase it.**
> *The non-IID result is reported with a confidence interval over three seeds,
> but all three use the same Dirichlet partition. The interval says nothing about
> sensitivity to the partition, which is the variable the experiment is about.*

**The fix.** State it explicitly next to the first CI, or add a
partition-reseeded arm — regenerate the α=0.3 shards under 2–3 different data
seeds and re-run E0/E2. That is the honest version of "3 seeds" for a
heterogeneity claim.

**Note.** Ablation A3 uses 2 seeds (`t = 12.706`). Those intervals are ~3× wider
and must be labelled preliminary. See also **S2-6** — for generation metrics,
seeds are the *only* lever on interval width.

---

## S2-5 — ROUGE backend silently substituted · **RESOLVED, with a residue**

**The problem.** `nltk` was absent, so runs fell back from the `evaluate` library
to the built-in ROUGE implementation.

**It was worse than "consistent across runs".** The original entry assumed the
fallback was uniform, so internal comparisons would still hold. Auditing all 45
stored runs on 2026-08-13 showed it was **not** uniform: the 30 main-table runs
took the fallback and **Ablation B1's 6 arms did not**. ROUGE-L and BLEU are
therefore not comparable between the main and ablation `comparison.md` tables —
which is how E5 and B1-E2, whose adapters are bit-identical, came to print
different ROUGE-L (0.2276 vs 0.2340).

**Resolution.** `require_metric_backend: "evaluate"` has been in
`base_config.yaml` since 2026-08-07 and makes a missing metric stack a hard
failure at the first evaluation; the contaminated runs stored `''` and predate
it. Every run since stores `'evaluate'`. **Historical contamination, not a live
defect.** `results/<tier>/reeval250` re-scores every adapter on one backend, and
on it B1-E2 and E5 agree to 6 dp.

**Residue.** The ROUGE-L/BLEU rows in every shipped `comparison.md` are
unquotable and should be regenerated from `reeval250` or dropped. Minor: the
import-failure path logs at `INFO` while the scoring-failure path logs at
`WARNING` — align them, since the log is the last line of defence if the guard is
ever unset.

**Cost.** Zero GPU. Report regeneration only.

---

## S3-1 — The E0 comparison is asymmetric and unlabelled

E0's headline number is the **mean of 3 separately-evaluated models**; E2's is
**one aggregated model**. Comparing them answers the right question — "what does
a participant get?" — but the asymmetry is invisible in the table. Fix via
[C7](04_changes.md#c7--record-the-local-only-comparison-semantics-explicitly):
footnote it and report the per-client spread alongside the mean.

## S3-2 — `fedavg_weighted` differs between E5 and E2–E4

Looks like a confound; is not. `max_train_samples: 500` caps every client below
the smallest Dirichlet shard, so observed weights are `[0.333, 0.333, 0.333]` in
both. [exp5_noniid.yaml:47-59](../configs/exp5_noniid.yaml#L47-L59) documents
this precisely. **The fix is to the paper, not the config:** describe E5 as
*label* skew with balanced quantities, not quantity skew.

## S3-3 — E6 runs on a single seed, and the 0.5B tier is under-powered

E6 uses seed-42 adapters. At 360M the source is an archived E4 run
(`results/_archive_prefix_20260804_222225/`); at 0.5B it is the live
`outputs/qwen-0.5b/seed_42/exp4_fedchain`.

**The statistical half is the part that matters, and it is a promotion from
"untidy provenance".** The 0.5B tier ran **20 trials per attack**, the 360M tier
50. A benign-control result of 0/20 bounds the false-positive rate at only
**13.9%** (one-sided 95% binomial) — so *the paper cannot claim a 0% FP rate at
0.5B*. 0/50 would bound it at 5.8%.

**The fix.** Re-run E6 at 50 trials and E7 to N=100 at qwen, for protocol parity
with the 360M tier. **Minutes of compute** — it is the cheapest outstanding item
in the entire study and the only one still blocking a stated claim. It needs a
running chain; `finish_study.sh --only audit` does it once `anvil` is up.

## S3-4 — Chain latency is devnet latency

Anvil at ~0.124 s/tx. The **gas** figures transfer to a real deployment; the
latency does not. State this rather than letting the sub-1% wall-clock claim rest
on a local node. A short discussion of what the gas costs on a real L2 would
strengthen the deployability argument at no experimental cost.

---

## Scope limits to state in the paper (not defects)

These are honest boundaries. Naming them is cheaper than having them found.

- **Byzantine clients are out of scope.** A malicious client that anchors a
  poisoned adapter it genuinely trained passes integrity checking by
  construction — the hash matches. That is Byzantine robustness (Krum, trimmed
  mean, norm clipping), orthogonal to provenance.
- **Privacy is out of scope.** Anchoring a digest leaks nothing, but LoRA
  updates are not differentially private.
- **The aggregator is assumed honest.** E6's threat model covers a hostile
  transport layer, not a hostile aggregator.
- **Single dataset.** Dolly-15k only. The 8-category label structure is what
  makes the Dirichlet partition meaningful; a second dataset would test whether
  the heterogeneity result generalises.
- **Fixed budget, not convergence.** Every accuracy number is measured at
  R=3 / 4,500 updates, with the loss curves still descending (decay ratio ≈ 0.31
  per round). This is a scope limit rather than a defect *provided it is
  labelled*; unlabelled, it becomes S1-level. Ablation A is what would bound it.

---

## Roadmap

Phases 0, 1, 3 (the E half) and 5 are **done** as of 2026-08-13.

| Phase | Work | Cost | Unblocks | Status |
|---|---|---|---|---|
| **0** | Apply C1–C4 | ~1 day of code | Everything | **done** |
| **1** | Ablation B1 at 0.5B | 12 h | S1-2 | **done** 08-12 |
| **3a** | Ablation E (re-score @250) | 6 h | S2-1, and surfaced S2-5/S2-6 | **done** 08-13 |
| **5** | `qwen-0.5b` replication | ~3× matrix | S1-1, S2-3 | **done** |
| **1b** | **E6 @ 50 trials + E7 to N=100 at qwen** | **minutes** | **S3-3 — the 0% FP claim** | **outstanding** |
| **1c** | **Ablation B1 at 360M** | 12 h | S1-4 | outstanding |
| **2** | Ablation A (round sweep) | 37 h | the fixed-budget caveat | outstanding |
| **3b** | Ablation D | 3 h | the rest of S2-2 | outstanding |
| **4** | Ablations B2/B3, C | 53 h | the α curve, drift robustness | outstanding |
| **6** | `qwen-1.5b` third rung | ~9× matrix | turns a direction into a trend | outstanding |

**Critical path to a submittable paper: phase 1b, minutes.** Everything else that
was blocking has landed. 1b is blocked only on starting a local chain, and until
it runs the false-positive claim at 0.5B has to be stated as ≤13.9% rather
than 0%.

**Recommended before submission:** 1b + 1c + 2 + 3b ≈ **52 h**. That closes the
cross-tier gap in the skew claim (S1-4), bounds the headline number, and finishes
the cost decomposition — the three things a reviewer is most likely to ask for.

**Free, and worth doing regardless:** regenerate or drop the ROUGE-L/BLEU rows in
every `comparison.md` (S2-5), and state next to the first confidence interval
that the seeds share a partition (S2-4) and that the interval over generation
metrics does not respond to sample count (S2-6).
