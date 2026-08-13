# 06 — Ablation results

Data only. Interpretation lives in [07](07_ablation_conclusions.md); the
hypotheses being checked were fixed in [05](05_ablation_design.md) before any of
these runs started.

> **Two of six blocks have been executed: B1 and E.** Rows marked `not started`
> below are still `—` and must not be quoted. Rows marked `done` are measured and
> quotable.

## Run status

| Block | Runs | Tier | Seeds | Status | Date | Metrics path |
|---|---|---|---|---|---|---|
| A1 | E2 @ R=9, IID | — | 42,43,44 | not started | — | — |
| A2 | E0 @ R=9, IID | — | 42,43,44 | not started | — | — |
| A3 | E1 @ 7.5k / 13.5k samples | — | 42,43 | not started | — | — |
| **B1** | **E0/E1/E2 @ α=0.3** | **qwen-0.5b** | **42,43,44** | **done** | **2026-08-11 → 08-12** | `results/qwen-0.5b/ablation/` |
| B2 | E0/E1/E2/E4 @ α=0.1 | — | 42,43,44 | not started | — | — |
| B3 | E0/E1/E2/E4 @ α=1.0 | — | 42,43,44 | not started | — | — |
| C | `local_epochs` ∈ {1,2,4} | — | 42,43,44 | not started | — | — |
| D1 | `log_global_model: false` | — | 42 | not started | — | — |
| D2 | `ipfs_roundtrip_aggregation: false` | — | 42 | not started | — | — |
| D3 | `verify_hash_on_download: false` | — | 42 | not started | — | — |
| **E** | **Re-eval @ 250 gen samples** | **both** | **42,43,44** | **done** | **2026-08-13** | `results/<tier>/reeval250` |
| F | `num_clients` ∈ {3,5,10} | — | 42,43,44 | not started | — | — |

**Tier note.** B1 ran at **qwen-0.5b**, not at 360M. Every B1 number below must
therefore be read against the *qwen-0.5b* IID row, never against the 360M one.
This is the single most common way to misread this document — see
[the premise correction](#the-premise-correction) below.

**Prerequisite gate.** C1–C4 from [04](04_changes.md) were applied before B1 and
E. C3 (metric backend) is the one that bites: see
[§E.0](#e0--the-backend-split-read-this-before-any-generation-number).

---

## A — Round count

**Not run.** Every cell below is `—`. The italicised R=3 row is the measured
baseline carried forward from [10](10_two_model_results.md), for reference only.

### A.1 Loss trajectory (qwen-0.5b, mean over 3 seeds)

| Round | E0 Local-only | E2 FedAvg | E2 − E0 | E1 (budget-matched) | E2 − E1 |
|---|---|---|---|---|---|
| 1 | — | *2.11453* | — | — | — |
| 2 | — | *2.07922* | — | — | — |
| 3 | *2.07830* | *2.06864* | *−0.00964* | *2.04994* | *+0.01870* |
| 5 | — | — | — | — | — |
| 7 | — | — | — | — | — |
| 9 | — | — | — | — | — |

E0 has no per-round curve: it evaluates only at the end. `eval_local_clients_every_round: true`
gives it one, and A2 must set it.

### A.2 Paired differences at each round count

| R | E2 − E0 (± CI) | Significant | E2 − E1 (± CI) | Significant |
|---|---|---|---|---|
| 3 | *−0.00964 ± 0.00093* | *yes* | *+0.01870 ± 0.00180* | *yes* |
| 5 | — | — | — | — |
| 7 | — | — | — | — |
| 9 | — | — | — | — |

### A.3 Prediction check

| Hypothesis | Predicted | Observed | Held? |
|---|---|---|---|
| H-A1: `\|E2−E0\|` ≥ 0.005 at R=9 | ≥ 0.005 | — | not tested |
| H-A2: E2 @ R=9 ≈ 2.014 | 2.014 | — | not tested |
| H-A3: `E2−E1` stays flat ≈ 0.034 | flat | — | not tested |

**Standing evidence that A is still needed.** Both curves are descending at R=3.
Per-round deltas at qwen-0.5b: IID `−0.0353, −0.0106`; α=0.3 `−0.0357, −0.0113`.
Decay ratio ≈ 0.30–0.32. Every accuracy number in this study is a **fixed-budget**
number at 4,500 updates.

---

## B — Data heterogeneity

### B.1 Accuracy by partition (R=3, mean ± 95% CI over 3 seeds)

**qwen-0.5b** — the tier B1 ran at:

| α | E0 Local-only | E1 Centralized | E2 FedAvg | E4-equiv (chain+IPFS) | E2 − E0 (paired) | Significant |
|---|---|---|---|---|---|---|
| 0.1 | — | — | — | — | — | — |
| **0.3** | **2.0885 ± 0.0016** | **2.0492 ± 0.0042** | **2.0723 ± 0.0028** | **2.0723 ± 0.0028** † | **−0.01626 ± 0.00128** | **yes** |
| 1.0 | — | — | — | — | — | — |
| IID | 2.0783 ± 0.0007 | 2.0499 ± 0.0016 | 2.0686 ± 0.0009 | 2.0686 ± 0.0009 | −0.00964 ± 0.00093 | yes |

† E5 from the baseline. It is now a *matched* arm: B1-E2 and E5 are the same
configuration apart from the chain and IPFS flags, and they are bit-identical
([§B.5](#b5--hash-equality-under-skew)).

**smollm2-360m** — IID only; **no non-IID ablation exists at this tier**:

| α | E0 Local-only | E1 Centralized | E2 FedAvg | E2 − E0 (paired) | Significant |
|---|---|---|---|---|---|
| 0.3 | — | — | — | — | — |
| IID | 2.0236 ± 0.0011 | 1.9884 ± 0.0006 | 2.0228 ± 0.0013 | −0.00085 ± 0.00022 | yes |

### B.2 Fraction of the isolation→centralized gap recovered by FedAvg

The quantity the study exists to produce. Computed per seed, then averaged:
`(E0 − E2) / (E0 − E1)`.

| Tier | Partition | E0 − E2 (FedAvg gain) | E0 − E1 (headroom) | Recovered |
|---|---|---|---|---|
| qwen-0.5b | IID | +0.00964 ± 0.00093 | +0.02836 ± 0.00101 | **34.0% ± 4.3%** |
| qwen-0.5b | **α = 0.3** | **+0.01626 ± 0.00128** | **+0.03932 ± 0.00479** | **41.5% ± 7.7%** |
| smollm2-360m | IID | +0.00085 ± 0.00022 | +0.03518 ± 0.00098 | 2.4% |

**Gain growth, IID → α=0.3, at qwen-0.5b: 0.00964 → 0.01626 = 1.69×.** The
intervals are disjoint (`0.01626 − 0.00128 = 0.01498 > 0.00964 + 0.00093 = 0.01057`).
This comparison is *unpaired across partitions* — the two arms read different
shard files — so it is a comparison of two paired estimates, not a paired
estimate itself.

### B.3 Shard sizes and the round cap

Read from the B1 run logs, not from the design doc:

| α | client1 | client2 | client3 | Union | Max rounds @ 500/round |
|---|---|---|---|---|---|
| 0.1 | — | — | — | — | — |
| **0.3** | **4,798** | **2,715** | **6,998** | **14,511** | **5** |
| 1.0 | — | — | — | — | — |
| IID | 4,837 | 4,837 | 4,837 | 14,511 | 9 |

**The unions are equal at 14,511.** The Dirichlet repartition redistributes the
same corpus, so it changes *who holds what*, not *how much exists*. R=3 used
windows `[0,500) [500,1000) [1000,1500)`; all three fit inside client2's 2,715.

**Partition-invariance control.** E1 trains on the pooled union (1,500 per shard
= 4,500 samples) and should therefore be indifferent to how that corpus was
split. Measured: **2.0499 ± 0.0016 (IID) vs 2.0492 ± 0.0042 (α=0.3)** — a
difference of 0.0007 against intervals of ±0.0016 and ±0.0042. The centralized
bound is partition-invariant, as it must be. This validates the Dirichlet
shards: had the repartition altered task difficulty, E1 would have moved.

### B.4 Prediction check

| Hypothesis | Predicted | Observed | Held? |
|---|---|---|---|
| H-B1: `\|E2−E0\|` ordered α=0.1 > 0.3 > 1.0 > IID | monotone in α | α=0.3 (0.01626) > IID (0.00964), disjoint intervals; α=0.1 and α=1.0 not run | **partially — direction confirmed on the one contrast available** |
| H-B2: E0 degrades faster than E2 as α falls | yes | IID→α=0.3: E0 **+0.01026**, E2 **+0.00364**, E1 **−0.00070**. E0 degrades **2.82×** faster | **yes** |
| H-B3: effect weaker than the FL literature reports (label skew only, quantity skew removed) | yes | All three clients contributed exactly 500/round; weights uniform. 41.5% recovery is well short of the near-total FedAvg advantages reported under joint label+quantity skew | **yes** |

H-B2 is the mechanism claim and it came through cleanly: under skew the isolated
clients lose ground while the averaged model barely moves, and the centralized
bound does not move at all.

### B.5 Hash equality under skew

The audit-layer no-op claim, re-tested on the non-IID partition. B1-E2 (no chain,
no IPFS) against E5 (chain + IPFS), identical in every other respect:

| Check | Result |
|---|---|
| Client adapter SHA-256, 3 seeds × 3 rounds × 3 clients | **27/27 identical** |
| Validation loss, per seed | identical to 6 dp (2.073369 / 2.072359 / 2.071123) |
| Perplexity, per seed | identical to 6 dp |
| ROUGE-L and BLEU @250, per seed | identical to 6 dp |
| Communication volume | 302.86 vs 399.31 MiB — **differs, as designed** (IPFS download leg) |
| Checkpoint fingerprint | differs — it hashes the config, which includes the chain/IPFS flags |

Only the two quantities that *should* differ do. The zero-accuracy-cost result
is no longer an IID-only result.

### <a id="the-premise-correction"></a>B.6 The premise correction

[05](05_ablation_design.md#what-this-study-has-to-fix) motivates the whole study
with "E2 − E0 = −0.00085 ± 0.00022, i.e. FedAvg recovers 2.4% of the gap," and
the header comment in `ablationB_e2_noniid.yaml` repeated it. **That is the
smollm2-360m figure.** B1 ran at qwen-0.5b, where the IID recovery was already
**34.0%**.

The corrected statement of what B1 found:

> At qwen-0.5b, FedAvg's recovery of the isolation→centralized gap rises from
> **34.0% ± 4.3% (IID) to 41.5% ± 7.7% (Dirichlet α=0.3)**, with the absolute
> gain growing 1.69× and the intervals disjoint.

Not "from nothing to something." The tier where FedAvg genuinely did almost
nothing — 360M, 2.4% — **has no non-IID arm**, so the contrast that would show
skew rescuing a near-zero baseline does not exist in this study. Running B1 at
360M is the missing piece; it is the highest-value single remaining run.

---

## C — Local epochs

**Not run.** The `local_epochs: 1` rows are the measured baseline, carried
forward. qwen-0.5b throughout.

| Partition | `local_epochs` | E2 val loss | E2 − E0 | E2/E3/E4 hashes identical? |
|---|---|---|---|---|
| IID | 1 | *2.0686 ± 0.0009* | *−0.00964 ± 0.00093* | *yes — 18/18 global, both tiers* |
| IID | 2 | — | — | — |
| IID | 4 | — | — | — |
| α=0.3 | 1 | *2.0723 ± 0.0028* | *−0.01626 ± 0.00128* | *yes — 27/27 client ([§B.5](#b5--hash-equality-under-skew))* |
| α=0.3 | 2 | — | — | — |
| α=0.3 | 4 | — | — | — |

H-C2 (hash equality survives every epoch setting) now has **two** partitions
confirmed at E=1 rather than one. It still has only one epoch setting.

---

## D — Audit-layer cost decomposition

**Not run.** Reference rows are measured (qwen-0.5b, mean over 3 seeds).

### D.1 Systems cost by variant

| Variant | Tx | Gas | Comm (MiB) | Δ comm vs E2 | IPFS up (s) | IPFS down (s) | Total round time (s) |
|---|---|---|---|---|---|---|---|
| E2 FedAvg (reference) | *0* | *0* | *302.86* | *—* | *0* | *0* | *4,779.7 ± 44.7* |
| E3 FL + chain | *12* | *2,997,464* | *302.86* | *0.0%* | *0* | *0* | *4,770.8 ± 34.3* |
| E4 FedChain (baseline) | *12* | *3,785,372* | *399.32 ± 0.01* | *+31.8%* | *18.04* | *2.85* | *4,787.4 ± 36.9* |
| D1 `log_global_model: false` | — | — | — | — | — | — | — |
| D2 `ipfs_roundtrip_aggregation: false` | — | — | — | — | — | — | — |
| D3 `verify_hash_on_download: false` | — | — | — | — | — | — | — |

E3 already isolates one half of the answer: **anchoring alone adds zero
communication** (302.86 MiB, byte-identical to E2). The entire +31.8% therefore
arrives with IPFS, not with the chain. D2 is what attributes it *within* IPFS.

### D.2 Attribution of the +31.8% communication overhead

| Component | MiB/round | Share of overhead |
|---|---|---|
| Client adapter uploads to IPFS | — | — |
| Aggregator downloads for FedAvg | — | — |
| Global model upload (pin) | — | — |
| Global model broadcast (client pulls) | — | — |
| **Total** | — | **100%** |

### D.3 Prediction check

| Hypothesis | Predicted | Observed | Held? |
|---|---|---|---|
| H-D1: D1 gas ≈ 2.84M (12→9 tx) | 2.84M | — | not tested |
| H-D2: D2 comm returns toward 302.9 MiB | ~303 | — | not tested |
| H-D3: D3 wall-clock saving immeasurable | ~0 | *supported indirectly*: E6 mean verify latency is **13.1 ms** per artefact against a ~4,780 s round | not tested directly |
| H-D4: all variants bit-identical to E4 | identical | — | not tested |

---

## E — Evaluation fidelity

**Done.** 2026-08-13, `scripts/reevaluate.py`, 39 adapter evaluations at
qwen-0.5b (13.3 h) and 24 at smollm2-360m (9.8 h). No retraining.

### <a id="e0--the-backend-split-read-this-before-any-generation-number"></a>E.0 The backend split — read this before any generation number

[eval_loss.py:592-601](../evaluation/eval_loss.py#L592-L601) falls back to a
self-contained scorer when the `evaluate` library cannot be imported, and records
which one it used in `evaluation_detail.generation_metric_backend`. Auditing all
45 stored runs:

| Backend | Runs | Which | Stored `require_metric_backend` |
|---|---|---|---|
| `builtin` | 30 | **every main-table run** — E1–E5, both tiers, 3 seeds | `''` |
| `evaluate` | 6 | B1's E1/E2 arms only | `'evaluate'` |
| *(none recorded)* | 9 | E0, which reports a `local_only_mean` over its 3 clients | mixed |

**The guard already exists and works.** `require_metric_backend: "evaluate"`
turns a missing metric stack into a hard failure at the first evaluation
([eval_loss.py:649-655](../evaluation/eval_loss.py#L649-L655)), and it landed in
`base_config.yaml` on 2026-08-07 (`f175959`). The contaminated runs stored
`require_metric_backend: ''` — they were launched from a checkout that predated
it. Every run since, B1 included, stored `'evaluate'` and got it. **This is a
historical contamination, not a live defect**, and it cannot recur on the current
configs.

**Consequence.** ROUGE-L and BLEU as printed in `results/*/comparison.md` are
**not comparable between the main table and the B1 table**: they were produced by
two different scorers. This is exactly what the E5-vs-B1-E2 discrepancy was —
0.2276 against 0.2340 on *bit-identical weights*. Re-scored on one backend the
two agree to 6 dp ([§B.5](#b5--hash-equality-under-skew)).

**Rule: quote generation metrics from `results/<tier>/reeval250` and nowhere
else.** The `comparison.md` ROUGE-L and BLEU rows should be regenerated from it
or dropped.

### E.1 ROUGE-L and BLEU @250, `evaluate` backend, mean ± 95% CI over seeds

**qwen-0.5b:**

| Arm | ROUGE-L | BLEU | Val loss |
|---|---|---|---|
| E1 Centralized | **0.2737 ± 0.0132** | **0.0612 ± 0.0049** | 2.0499 ± 0.0016 |
| E2 / E3 / E4 | 0.2564 ± 0.0117 | 0.0517 ± 0.0057 | 2.0686 ± 0.0009 |
| E0 Local-only | 0.2527 ± 0.0127 | 0.0501 ± 0.0057 | 2.0783 ± 0.0007 |
| E5 / B1-E2 (α=0.3) | 0.2455 ± 0.0110 | 0.0488 ± 0.0097 | 2.0723 ± 0.0028 |
| B1-E1 (α=0.3) | 0.2702 ± 0.0130 | 0.0583 ± 0.0074 | 2.0492 ± 0.0042 |
| B1-E0 (α=0.3) | 0.2463 ± 0.0107 | 0.0495 ± 0.0074 | 2.0885 ± 0.0016 |

**smollm2-360m:**

| Arm | ROUGE-L | BLEU | Val loss |
|---|---|---|---|
| E1 Centralized | 0.2477 ± 0.0022 | 0.0587 ± 0.0033 | 1.9884 ± 0.0006 |
| E5 (α=0.3) | 0.2457 ± 0.0160 | 0.0569 ± 0.0115 | 2.0249 ± 0.0006 |
| E0 Local-only | 0.2435 ± 0.0070 | 0.0594 ± 0.0021 | 2.0236 ± 0.0011 |
| E2 / E3 / E4 | 0.2417 ± 0.0135 | 0.0579 ± 0.0113 | 2.0228 ± 0.0013 |

At qwen-0.5b the ROUGE-L ordering now **matches the loss ordering**
(E1 > E2 > E0 > E5). At 360M it does not, and nothing there separates.

### E.2 Does 250 samples buy narrower intervals?

The clean test needs the same backend at both counts, which only the B1 arms
have (`evaluate` at 50 and at 250):

| Arm | ROUGE-L @50 | ROUGE-L @250 | CI ratio | BLEU @50 | BLEU @250 | CI ratio |
|---|---|---|---|---|---|---|
| B1-E1 | 0.2630 ± 0.0082 | 0.2702 ± 0.0130 | **0.63×** | 0.0406 ± 0.0137 | 0.0583 ± 0.0074 | 1.86× |
| B1-E2 | 0.2336 ± 0.0009 | 0.2455 ± 0.0110 | **0.08×** | 0.0449 ± 0.0042 | 0.0488 ± 0.0097 | 0.43× |

**The intervals did not narrow; three of four widened.** H-E1 predicted 2.24×
narrowing and is falsified — but the prediction was malformed, not the run. The
reported ±CI is a **between-seed** interval at n=3. Raising `gen_num_samples`
shrinks the Monte-Carlo error *inside* each seed's point estimate; it does
nothing to the seed-to-seed spread that the CI is actually measuring. The
interval is floored by seed variance at any sample count.

Sharpening generation CIs requires **more seeds**, not more generation samples.
What 250 samples does buy is a less noisy point estimate per seed — which is why
the ordering became coherent at qwen in E.1.

### E.3 Between-arm significance @250 (paired per seed)

vs E1 Centralized:

| Tier | Contrast | ΔROUGE-L | Significant |
|---|---|---|---|
| qwen-0.5b | E0 − E1 | −0.0210 ± 0.0093 | **yes** |
| qwen-0.5b | E2 − E1 | −0.0173 ± 0.0102 | **yes** |
| qwen-0.5b | E5 − E1 | −0.0282 ± 0.0241 | **yes** (marginal) |
| smollm2-360m | E0 − E1 | −0.0042 ± 0.0085 | no |
| smollm2-360m | E2 − E1 | −0.0060 ± 0.0130 | no |
| smollm2-360m | E5 − E1 | −0.0020 ± 0.0166 | no |

FedAvg vs local-only — the contrast the motivation rests on:

| Tier | Partition | ΔROUGE-L (E2 − E0) | Significant | ΔBLEU | Significant |
|---|---|---|---|---|---|
| qwen-0.5b | IID | +0.0037 ± 0.0016 | **yes** | +0.0016 ± 0.0050 | no |
| qwen-0.5b | α=0.3 | −0.0008 ± 0.0044 | **no** | −0.0006 ± 0.0112 | no |

### E.4 Prediction check

| Hypothesis | Predicted | Observed | Held? |
|---|---|---|---|
| H-E1: CIs narrow ~2.24× | 2.24× | 0.08×–1.86×; three of four widened | **no** — prediction confounded within-run sampling noise with between-seed variance |
| H-E2: no ordering becomes significant at 250 | none significant | **qwen-0.5b: E1 significantly beats E0, E2 and E5.** 360M: nothing significant | **no at qwen-0.5b, yes at 360M** |

**Net.** Generation metrics are usable at qwen-0.5b for the *centralized-vs-rest*
contrast, and only there. They cannot resolve E2 − E0 under skew — the contrast
the motivation actually needs — where loss says +0.0163 (significant) and ROUGE-L
says −0.0008 ± 0.0044 (nothing). Report loss and perplexity as the accuracy
result; report ROUGE-L/BLEU @250 as a supporting collapse check with the sample
count and backend stated.

---

## F — Federation size

**Not run.** The 3-client row is measured (qwen-0.5b).

| Clients | E0 val loss | E2 val loss | E2 − E0 | Gas | Comm (MiB) |
|---|---|---|---|---|---|
| 3 | *2.0783 ± 0.0007* | *2.0686 ± 0.0009* | *−0.00964 ± 0.00093* | *3,785,372* | *399.32* |
| 5 | — | — | — | *1,744,914 /round* † | — |
| 10 | — | — | — | *3,198,971 /round* † | — |

† From E7's chain-only sweep, which already covers the *audit* cost of scale.
F covers the *learning* side.

---

## Anomaly log

| Date | Run | Anomaly | Resolution |
|---|---|---|---|
| 2026-08-13 | E6/E7 re-run at qwen | `finish_study.sh` skipped both — no chain at `127.0.0.1:8545` and `anvil` not on PATH | **Open.** Item 3 of [11](11_final_status.md#remaining-work) is still outstanding. Start anvil, re-run. Minutes of compute. |
| 2026-08-13 | all main-table runs | Generation metrics computed with the `builtin` scorer while B1 used `evaluate`, making ROUGE-L/BLEU incomparable between the two tables | Diagnosed ([§E.0](#e0--the-backend-split-read-this-before-any-generation-number)). **Historical only** — those runs stored `require_metric_backend: ''`, predating the guard added 2026-08-07. Superseded by `reeval250`. Residual nit: the import-failure path logs at `INFO` ([eval_loss.py:597-600](../evaluation/eval_loss.py#L597-L600)) where the scoring-failure path logs at `WARNING`; align them if the guard is ever unset. |
| 2026-08-13 | B1 design premise | `ablationB_e2_noniid.yaml` and [05](05_ablation_design.md) quoted the 360M IID recovery (2.4%) as the premise for a run executed at qwen-0.5b (34.0%) | Corrected in [§B.6](#the-premise-correction) and in the config header. The measured result is unaffected; only its framing was. |
| — | B1, E | No failed transactions, no resumes, no OOM, no shard-window overrun. `sessions: 1` on all 9 B1 runs. | — |
