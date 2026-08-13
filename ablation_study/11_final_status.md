# 11 — Final status: what is proven, what is not, what remains

Written 2026-08-11 after the clean Qwen2.5-0.5B sweep; **revised 2026-08-13**
after Ablation B1, Ablation E, and the 360M E0 re-run. This is the decision
document. [10](10_two_model_results.md) holds the baseline numbers,
[06](06_ablation_results.md) the ablation numbers, [07](07_ablation_conclusions.md)
what they license; this holds the ledger and the remaining work.

---

## The experiment matrix, as actually executed

| | E0 | E1 | E2 | E3 | E4 | E5 | E6 | E7 |
|---|---|---|---|---|---|---|---|---|
| **smollm2-360m** | 3 seeds ✔ | 3 seeds | 3 seeds | 3 seeds | 3 seeds | 3 seeds | real, 50×5 | N≤100 |
| **qwen-0.5b** | 3 seeds | 3 seeds | 3 seeds | 3 seeds | 3 seeds | 3 seeds | real, 20×5 ⚠ | N≤50 ⚠ |

✔ = re-run 2026-08-12 post-C1; the 299.202 MiB artefact is gone (now 0.000).
⚠ = protocol parity with the 360M tier still outstanding (50 trials, N≤100).
**39 training runs, 4 audit runs, zero failures, zero resumes.**

| Ablation | A rounds | B1 α=0.3 | B2 α=0.1 | B3 α=1.0 | C epochs | D cost | E gen-eval | F clients |
|---|---|---|---|---|---|---|---|---|
| Status | not run | **done (0.5B)** | not run | not run | not run | not run | **done** | not run |

**Two of six ablation blocks are executed.** B1 ran at qwen-0.5b on 2026-08-11/12;
Ablation E (re-scoring, no retraining) on 2026-08-13. [06](06_ablation_results.md)
and [07](07_ablation_conclusions.md) are now written from real data and are
quotable for those two blocks; every other block's rows are still `—`.

What first closed the motivation gap was **not** an ablation but the model-ladder
tier — an outcome outside the pre-registered branch structure, declared as such in
[10](10_two_model_results.md#the-headline-the-motivation-gap-closes-with-model-scale).
B1 then added label skew on top of an already-open gap. A related defect surfaced
in the process: [05](05_ablation_design.md)'s decision thresholds were calibrated
on the 360M numbers and never re-derived when the study moved to 0.5B, so the
"nothing is happening" branch was never live at the tier B1 ran on
([07](07_ablation_conclusions.md#before-the-branches-the-thresholds-were-calibrated-at-the-wrong-tier)).

---

## Evidence ledger

Every claim the paper could make, what it rests on, and whether it holds.

### Tier 1 — solid, lead with these

| Claim | Evidence | Strength |
|---|---|---|
| The audit layer is an exact no-op on the learning math | **18/18 global-model SHA-256 hashes identical** across E2/E3/E4, 3 rounds × 3 seeds × 2 tiers, IID. **Plus 27/27 client adapter hashes identical** between B1-E2 and E5 under Dirichlet(0.3) — loss, perplexity, ROUGE-L and BLEU all equal to 6 dp. Paired loss difference exactly 0.000000, sd 0.000000. | Cryptographic identity, not a statistical test. Now demonstrated on **two data partitions**, not one. |
| Anchoring adds zero communication | E3 (chain, no IPFS) records **302.86 MiB — byte-identical to E2**. The whole +31.8% arrives with IPFS transport, none with the chain. | Directly measured; needs no ablation. |
| Anchoring cost is independent of model size | E3 gas = 2,997,464 and E4 gas = 3,785,372, **byte-identical in all 12 chain-enabled runs at both tiers**. E7 payload sweep: 220× adapter-size range moves gas 0.0077%; 32 bytes anchored at every size. | Measured on two architectures. |
| Anchoring cost is linear in participants | `gas = 301,120 + 290,533·N`, **R² = 0.999994**, N ∈ {1,3,5,10,25,50}. Marginal cost converges to 290,821 by N=10; gas *per client* falls monotonically. | Fitted, not asserted. |
| Tampering is detected | 100% detection on bitflip / scale / substitute / replay, on **real trained adapters at both tiers**: 200/200 at 360M (miss rate ≤ 1.5%), 80/80 at 0.5B (≤ 3.7%). Benign reserialization never flagged. | Detection side is solid; see Tier 3 for the FP side. |
| End-to-end overhead is sub-1% | E4 − E2 = **+0.16% ± 0.22%** at 0.5B (not significant); directly measured audit work 21.61 s against a 4,780 s round = **0.45%**. Both tiers agree. | Quote the measured 0.45%, and the interval as an upper bound. |
| The system runs reliably | 12/12 transactions in every chain run, 9/9 integrity checks in every E4/E5 run, 0 IPFS failures, `sessions: 1` everywhere. | No anomaly log entries. |

### Tier 2 — supported, but scope the wording

| Claim | Evidence | The scoping the claim needs |
|---|---|---|
| Federation beats isolated training | E2 − E0 = **−0.00964 ± 0.00093** at 0.5B = **34.0% ± 4.3%** of the isolation→centralized gap; −0.00085 ± 0.00022 = 2.4% at 360M. Budget-matched at 4,500 updates. | **"at a matched 4,500-update budget, R=3"** — not "at convergence". The curve is still descending (§ below). |
| Federation helps **more** under label skew | **B1**: recovery rises 34.0% ± 4.3% (IID) → **41.5% ± 7.7%** (α=0.3) at 0.5B; absolute gain 0.00964 → 0.01626 = **1.69×**, intervals disjoint. Mechanism confirmed (H-B2): E0 degrades **2.82×** faster than E2 while E1 is unmoved. | **One α contrast, not a curve.** α=0.1 and α=1.0 not run, so no monotonicity claim. Label skew only — the sample cap removes quantity skew. |
| The Dirichlet repartition is valid | E1 is partition-invariant: **2.0499 ± 0.0016 (IID) vs 2.0492 ± 0.0042 (α=0.3)**, unions equal at 14,511 records. | A control, and a strong one — quote it when the skew result is challenged. |
| The benefit grows with model capacity | 2.4% → 34.0%, tight intervals, consistent direction. | **Two points is a direction, not a law.** No monotonicity claim, no fitted curve. |
| The cost of federating shrinks with scale | E2 − E1: 0.0343 → 0.0187. | Same two-point caveat. |
| The audit layer is unaffected by data skew | E5 at both tiers: gas byte-identical to E4, 12/12 tx, 9/9 integrity. **And now B1's 27/27 hash equality**, which upgrades this from a systems claim to a learning-math one at 0.5B. | Systems claim at 360M; **cryptographic** at 0.5B. |

### Tier 3 — not supported today

| Claim | Why it fails | What fixes it |
|---|---|---|
| ~~Federation helps under non-IID~~ | **Resolved by B1 at 0.5B** — E0/E1/E2 now exist on the same Dirichlet(0.3) shards, so the comparison is matched. | done |
| Federation helps under non-IID **at 360M** | B1 ran only at 0.5B. The tier where FedAvg recovered 2.4% under IID has no non-IID arm, so the study cannot say whether skew rescues a near-zero baseline — which was the original motivation for testing skew at all. | **B1 at smollm2-360m** (~12 h) |
| `\|E2−E0\|` is monotone in α | One contrast (IID vs α=0.3) is a direction between two points. H-B1 predicted an ordering over four. | **Ablations B2, B3** |
| ROUGE-L / BLEU support any between-arm ordering | **Partially resolved by Ablation E.** At 0.5B, E1 significantly beats E0/E2/E5 @250 on `evaluate`. Nothing separates at 360M, and **the contrast the motivation needs — E2−E0 under skew — is not resolved**: loss says +0.0163 (sig), ROUGE-L says −0.0008 ± 0.0044 (nothing). Also: raising the sample count **does not narrow the CIs** (they are between-seed intervals). | Report loss/PPL as the accuracy result; generation metrics as a collapse check. More **seeds**, not more samples. |
| ROUGE-L / BLEU in `comparison.md` are comparable across tables | **No.** 30 main-table runs used the `builtin` scorer; B1's 6 used `evaluate`. Historical only — those runs stored `require_metric_backend: ''` and predate the guard added 2026-08-07. | Quote `results/<tier>/reeval250` only; regenerate or drop those rows. No code fix needed. |
| False-positive rate is 0% at 0.5B | 0/20 benign trials bounds the FPR at only **13.9%** (one-sided 95% binomial). The detection side is strong; the FP side at 0.5B is not. | Re-run E6 at 50 trials (minutes) → bound 5.8% |
| 34.0% / 41.5% is the benefit of federating | It is the benefit **at R=3**, and every loss curve is still descending (decay ratio ≈ 0.31). At round 1 the federated model is *worse* than local-only. | **Ablation A** |
| The +31.8% communication overhead is intrinsic to auditability | **Half-attributed without D**: E3 shows anchoring adds **zero** communication, so all of it is IPFS transport. The split *within* IPFS is still untested. | **Ablation D2** |
| The zero-cost property is hyperparameter-invariant | Now tested on **two partitions** but still at one `local_epochs`, one LoRA rank, one round count. The mechanism argument is strong (an out-of-band SHA-256 commitment cannot reach the optimizer) but it is an argument, not a measurement. | **Ablation C** |

---

## The three statistical soft spots

All cheap to disclose and expensive to be caught on.

**1. Seeds do not vary the data partition.** All three seeds read the same
`data/client*.jsonl` (or the same `data/dirichlet/client*.jsonl`); the seed
changes LoRA init, shuffling and dropout only. Every ±CI in this study is
therefore a **training-noise** interval, not a sampling interval, and understates
true uncertainty on any partition-dependent claim — which now includes the
headline B1 result. Either state this explicitly next to the first CI in the
paper, or add a partition-reseeded arm.

**2. R=3 is a budget, not a convergence point.** Per-round means:

| Round | smollm2-360m IID | qwen-0.5b IID | qwen-0.5b α=0.3 |
|---|---|---|---|
| 1 | 2.0834 | 2.1145 | 2.1193 |
| 2 | 2.0388 (−0.0446) | 2.0792 (−0.0353) | 2.0836 (−0.0357) |
| 3 | 2.0228 (−0.0160) | 2.0686 (−0.0106) | 2.0723 (−0.0113) |

Decay ratio ≈ 0.30–0.36 per round, still falling on all three curves. Every
accuracy number in the paper is a fixed-budget number and should be labelled one.

**3. Confidence intervals on generation metrics do not respond to sample count.**
Ablation E raised `gen_num_samples` 50 → 250 and the intervals **widened** in
three of four cases. The reported ±CI is a between-seed interval at n=3;
generation sampling noise sits *inside* each seed's estimate and is not what the
interval measures. Do not describe 250 samples as having tightened anything —
describe it as having de-noised the per-seed point estimates, which is what
actually made the qwen ordering coherent. Tighter generation intervals need
**more seeds**.

---

## Remaining work

Costs are per the [05 cost model](05_ablation_design.md#cost-model), on the T600.

**Done 2026-08-11 → 08-13:** B1 (0.5B), Ablation E, 360M E0 re-run, all tables
regenerated. Items 1, 2 and 4 of the previous list are closed.

> **Scope decision, 2026-08-14.** The paper is a short one (4–5 pages including
> references), so the programme is **closed at items 1 and 2 below**. Everything
> from item 3 down is deliberately out of scope and left for a follow-up; the
> configs and runner blocks exist and are validated, so none of it is blocked on
> authoring work. Both remaining runs are wired into a single command:
>
> ```bash
> bash scripts/run_final.sh          # ~13 h, unattended
> ```
>
> See [12_paper_plan.md](12_paper_plan.md) for what the resulting tables license.

| # | Run | Cost | What it buys | Blocking? |
|---|---|---|---|---|
| 1 | **E6 @ 50 trials + E7 to N=100**, qwen | minutes | Protocol parity across tiers; FP bound 13.9% → 5.8%. **Skipped on 08-13 — no chain running, `anvil` not on PATH.** | **Yes** — trivially cheap, and the only outstanding item that is |
| 2 | **B1 at smollm2-360m** — E0/E1/E2 on Dirichlet(0.3), 3 seeds | ~12 h | The skew contrast at the tier where FedAvg recovered only 2.4%. Without it the paper cannot say skew rescues a weak baseline — only that it widens an already-open one. | **Yes**, if the paper claims the skew effect generalises across scale |
| 3 | **A** — round sweep to R=9 at 0.5B | ~37 h | Bounds the 34.0%/41.5% claims the paper leads with | No, but strongly advisable |
| 4 | **D** — cost decomposition, 1 seed | ~3.3 h | Splits the +31.8% *within* IPFS. E3 already proves the chain contributes none of it. | No, but cheap and pre-empts a reviewer |
| 5 | **B2, B3** — α ∈ {0.1, 1.0} | ~32 h | Turns B1's one contrast into the α curve H-B1 actually predicted | No |
| 6 | **qwen-1.5b** tier | ~24 h | Third rung: turns a direction into a trend | No |
| 7 | C, F | ~40 h | Robustness breadth; C is the one that generalises the zero-cost claim off a single hyperparameter | No |

**Minimum to submission: item 1, minutes.** Everything else that was blocking is
done. Item 1 is blocked only on starting a local chain.

**Recommended: items 1–4, ≈ 52 h.** Adds the cross-tier skew contrast, the bound
on the headline number, and the cost attribution — the three things a reviewer is
most likely to ask for.

**Also do, no GPU needed:** regenerate or drop the ROUGE-L/BLEU rows in every
`comparison.md` — they mix two scorers and are the one part of the reporting
layer that is not quotable. `results/<tier>/reeval250` holds the replacement.

### Running item 1

`finish_study.sh` already covers it; it skipped on 08-13 only because no chain
was reachable. Start one first:

```bash
anvil --host 127.0.0.1 --port 8545 &        # or: bash infra.sh up
bash scripts/finish_study.sh --only audit   # E6 @ 50 trials + E7 to N=100
```

Steps are independent and idempotent — a completed step is skipped, and a
failure does not abort the ones after it. The script checks the metric stack
before spending GPU time, verifies shard sizes against the round count, and
regenerates every table at the end.

Items 2–7 are not in it: they are judgement calls, not mechanical follow-ups.
Commands are in [09_run_guide.md](09_run_guide.md); every config they need now
exists under `ablation_study/configs/`.

> **Gate for any non-IID run:** Dirichlet(0.3) shard sizes are **4,798 / 2,715 /
> 6,998**, union 14,511 — confirmed from the B1 run logs, equal to the IID union.
> The smallest caps the round count at `floor(2715/500) = 5`. Lower α skews
> further; regenerate the manifest and re-check before running B2.

---

## Reporting defects — fixed 2026-08-11

All three were in the presentation layer; no stored metric was affected.

1. **`comparison_across_models.md` reported a single seed.** `build_across_models`
   picked the lowest seed as representative, so the headline cross-model table
   showed seed-42 point values where the per-tier report correctly showed
   mean ± CI. **Fixed:** it now aggregates every seed, adds an `n` column, and
   prints `mean +- 95% CI` for metrics that vary while leaving deterministic
   ones (gas, adapter size) as bare figures.
2. **E6 and E7 rendered as all-`n/a` rows.** They carry none of the training-run
   metric keys, so every column printed `n/a` — which reads as "the run failed"
   when both succeeded. **Fixed:** they now have their own *Audit-layer
   experiments* section with detection rates and both gas sweeps.
3. **`qwen-0.5b.leaky_backup` appeared in the ladder table** beside the clean
   tier, inviting a reader to average the two. **Fixed:** tier directories whose
   names contain `backup`/`archive` are excluded and listed in a footnote, so
   "do not quote this" is enforced by the tooling rather than by memory.

~~Still outstanding: the 299.202 MiB E0 figure at 360M.~~ **Resolved 2026-08-12**
by the E0 re-run; the cross-model table now reads 0.000 MiB at both tiers.

## Reporting defect found 2026-08-13

**A metric-backend split contaminated the stored generation metrics.**
[eval_loss.py](../evaluation/eval_loss.py#L592-L601) falls back from the HF
`evaluate` scorer to a built-in implementation when `evaluate` cannot be
imported. All 30 main-table runs took the fallback; B1's 6 arms did not. ROUGE-L
and BLEU in `results/*/comparison.md` are therefore **not comparable between the
main and ablation tables** — which is how E5 and B1-E2 came to show different
ROUGE-L on bit-identical weights.

**Already guarded going forward.** `require_metric_backend: "evaluate"` makes a
missing metric stack a hard failure at the first evaluation, and it has been in
`base_config.yaml` since 2026-08-07 (`f175959`). The contaminated runs stored
`require_metric_backend: ''` — they predate it. B1 and everything after stored
`'evaluate'`. **No code fix is required**; this is a historical data-quality
issue, not a live defect.

**What it costs:** the ROUGE-L/BLEU rows in every `comparison.md` are
unquotable. `reeval250` is the replacement, and on that common scorer B1-E2 and
E5 agree to 6 dp. Regenerate those rows from it or drop them.

*Residual nit:* the import-failure path passes `level=LOGGER.info`
([eval_loss.py:597-600](../evaluation/eval_loss.py#L597-L600)) while the
scoring-failure path uses the `WARNING` default. Align them — it only matters if
someone unsets the guard, but that is exactly the situation where the log is the
last line of defence.

---

## Bottom line

The **systems contribution is finished and defensible**: verifiable provenance
for federated fine-tuning at bit-identical cost, demonstrated by 18/18 global
hash equality across two architectures under IID **and 27/27 client hash equality
under Dirichlet(0.3)**, with 100% tamper detection on real artefacts, sub-0.5%
measured overhead, zero communication attributable to anchoring itself, and
anchoring cost linear in participants (R² = 0.999994) and flat across a 220×
model-size range. Nothing outstanding threatens any of it.

The **learning contribution now stands on its own**. At qwen-0.5b, FedAvg
recovers **34.0% ± 4.3%** of the isolation→centralized gap on IID shards and
**41.5% ± 7.7%** under Dirichlet(0.3) — the absolute gain growing 1.69× with
disjoint intervals, through the predicted mechanism, against a matched baseline
on the same shards, with a partition-invariance control on the centralized bound.
That is a result, not a conjecture.

What it is not: converged (R=3, curves still descending), a curve in α (one
contrast), or demonstrated at 360M (no non-IID arm there). Those three qualifiers
must travel with every quotation of the number. The honest framing is *"federation
already helps at 0.5B, and label skew widens the margin"* — **not** *"federation
barely helps until you add skew"*, which was the premise the study was designed
around and which the 0.5B data does not support.
