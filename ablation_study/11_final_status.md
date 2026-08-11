# 11 — Final status: what is proven, what is not, what remains

Written 2026-08-11, after the clean Qwen2.5-0.5B sweep of 2026-08-10. This is the
decision document. [10](10_two_model_results.md) holds the numbers; this holds
what they license and what is still missing.

---

## The experiment matrix, as actually executed

| | E0 | E1 | E2 | E3 | E4 | E5 | E6 | E7 |
|---|---|---|---|---|---|---|---|---|
| **smollm2-360m** | 3 seeds ⚠ | 3 seeds | 3 seeds | 3 seeds | 3 seeds | 3 seeds | real, 50×5 | N≤100 |
| **qwen-0.5b** | 3 seeds | 3 seeds | 3 seeds | 3 seeds | 3 seeds | 3 seeds | real, 20×5 | N≤50 |

⚠ = pre-C1 instrumentation; its communication column reads 299.202 MiB and is an
artefact. **36 training runs, 4 audit runs, zero failures, zero resumes.**

| Ablation | A rounds | B1 α=0.3 | B2 α=0.1 | B3 α=1.0 | C epochs | D cost | E gen-eval | F clients |
|---|---|---|---|---|---|---|---|---|
| Status | not run | **not run** | not run | not run | not run | not run | **not run** | not run |

**No ablation from [05](05_ablation_design.md) has been executed.** What closed
the motivation gap was not an ablation but the model-ladder tier — an outcome
that sits outside the pre-registered branch structure and is declared as such in
[10](10_two_model_results.md#the-headline-the-motivation-gap-closes-with-model-scale).
Documents [06](06_ablation_results.md) and [07](07_ablation_conclusions.md)
remain untouched templates and must not be quoted.

---

## Evidence ledger

Every claim the paper could make, what it rests on, and whether it holds.

### Tier 1 — solid, lead with these

| Claim | Evidence | Strength |
|---|---|---|
| The audit layer is an exact no-op on the learning math | **18/18 global-model SHA-256 hashes identical** across E2/E3/E4, 3 rounds × 3 seeds × 2 tiers. Paired loss difference exactly 0.000000, sd 0.000000. | Cryptographic identity, not a statistical test. Nothing stronger is available. |
| Anchoring cost is independent of model size | E3 gas = 2,997,464 and E4 gas = 3,785,372, **byte-identical in all 12 chain-enabled runs at both tiers**. E7 payload sweep: 220× adapter-size range moves gas 0.0077%; 32 bytes anchored at every size. | Measured on two architectures. |
| Anchoring cost is linear in participants | `gas = 301,120 + 290,533·N`, **R² = 0.999994**, N ∈ {1,3,5,10,25,50}. Marginal cost converges to 290,821 by N=10; gas *per client* falls monotonically. | Fitted, not asserted. |
| Tampering is detected | 100% detection on bitflip / scale / substitute / replay, on **real trained adapters at both tiers**: 200/200 at 360M (miss rate ≤ 1.5%), 80/80 at 0.5B (≤ 3.7%). Benign reserialization never flagged. | Detection side is solid; see Tier 3 for the FP side. |
| End-to-end overhead is sub-1% | E4 − E2 = **+0.16% ± 0.22%** at 0.5B (not significant); directly measured audit work 21.61 s against a 4,780 s round = **0.45%**. Both tiers agree. | Quote the measured 0.45%, and the interval as an upper bound. |
| The system runs reliably | 12/12 transactions in every chain run, 9/9 integrity checks in every E4/E5 run, 0 IPFS failures, `sessions: 1` everywhere. | No anomaly log entries. |

### Tier 2 — supported, but scope the wording

| Claim | Evidence | The scoping the claim needs |
|---|---|---|
| Federation beats isolated training | E2 − E0 = **−0.00964 ± 0.00093** at 0.5B = **34.0%** of the isolation→centralized gap; −0.00085 ± 0.00022 = 2.4% at 360M. Budget-matched at 4,500 updates. | **"at a matched 4,500-update budget, R=3"** — not "at convergence". The curve is still descending (§ below). |
| The benefit grows with model capacity | 2.4% → 34.0%, tight intervals, consistent direction. | **Two points is a direction, not a law.** No monotonicity claim, no fitted curve. |
| The cost of federating shrinks with scale | E2 − E1: 0.0343 → 0.0187. | Same two-point caveat. |
| The audit layer is unaffected by data skew | E5 at both tiers: gas byte-identical to E4, 12/12 tx, 9/9 integrity. | This is a **systems** claim only. |

### Tier 3 — not supported today

| Claim | Why it fails | What fixes it |
|---|---|---|
| Federation helps under non-IID | **E5 is Dirichlet-sharded; E0/E1/E2 are IID-sharded.** Every E5−E0 and E5−E2 difference confounds partition with federation. Prohibited by [05 rule 6](05_ablation_design.md#analysis-protocol) and [EXPERIMENTS.md:23](../EXPERIMENTS.md#L23). | **Ablation B1** |
| ROUGE-L / BLEU support any between-arm ordering | All 36 runs used `gen_num_samples: 50` and the **builtin** backend. C3 and C4 were implemented but never enabled. Absolute values are not the standard implementation and CIs span 5–17% of the value. | **Ablation E** (no retraining) |
| False-positive rate is 0% at 0.5B | 0/20 benign trials bounds the FPR at only **13.9%** (one-sided 95% binomial). The detection side is strong; the FP side at 0.5B is not. | Re-run E6 at 50 trials (minutes) → bound 5.8% |
| 34.0% is the benefit of federating | It is the benefit **at R=3**, and both loss curves are still descending. At round 1 the federated model is *worse* than local-only. | **Ablation A** |
| The +31.8% communication overhead is intrinsic to auditability | Unattributed. It is very likely the global model's IPFS round-trip — an implementation choice — but that is untested. | **Ablation D** |
| The zero-cost property is hyperparameter-invariant | Tested at one `local_epochs`, one LoRA rank, one round count. The mechanism argument is strong (an out-of-band SHA-256 commitment cannot reach the optimizer) but it is an argument, not a measurement. | **Ablation C** |

---

## The two statistical soft spots

Both are cheap to disclose and expensive to be caught on.

**1. Seeds do not vary the data partition.** All three seeds read the same
`data/client*.jsonl`; the seed changes LoRA init, shuffling and dropout only.
Every ±CI in this study is therefore a **training-noise** interval, not a
sampling interval, and understates true uncertainty on any partition-dependent
claim. Either state this explicitly next to the first CI in the paper, or add a
partition-reseeded arm.

**2. R=3 is a budget, not a convergence point.** Per-round means:

| Round | smollm2-360m | qwen-0.5b |
|---|---|---|
| 1 | 2.0834 | 2.1145 |
| 2 | 2.0388 (−0.0446) | 2.0792 (−0.0353) |
| 3 | 2.0228 (−0.0160) | 2.0686 (−0.0106) |

Decay ratio ≈ 0.30–0.36 per round, still falling. Every accuracy number in the
paper is a fixed-budget number and should be labelled one.

---

## Remaining work

Costs are per the [05 cost model](05_ablation_design.md#cost-model), on the T600.

| # | Run | Cost | What it buys | Blocking? |
|---|---|---|---|---|
| 1 | **B1** — E0/E1/E2 on Dirichlet(0.3) at 0.5B, 3 seeds | ~12 h | The only path to a non-IID *learning* claim. Converts E5 from decoration to result. | **Yes** — otherwise cut every non-IID learning sentence |
| 2 | **E** — re-score existing adapters @ 250 gen samples, `evaluate` backend | ~6 h, **no training** | Usable ROUGE-L/BLEU, or a defensible demotion to a collapse check | **Yes** — otherwise cut two table rows |
| 3 | **E6 @ 50 trials + E7 to N=100**, qwen | minutes | Protocol parity across tiers; FP bound 14% → 5.8% | **Yes** — trivially cheap |
| 4 | **360M E0 re-run** | ~3.3 h | Removes the 299.202 MiB artefact from the headline cross-model table | **Yes** unless footnoted |
| 5 | **A** — round sweep to R=9 at 0.5B | ~37 h | Bounds the 34.0% claim the paper leads with | No, but strongly advisable |
| 6 | **D** — cost decomposition, 1 seed | ~3.3 h | Attributes +31.8% comm to transport choice, not to auditability | No, but cheap and pre-empts a reviewer |
| 7 | **qwen-1.5b** tier | ~24 h | Third rung: turns a direction into a trend | No |
| 8 | C, B2, B3, F | ~70 h | Robustness breadth | No |

**Minimum to submission: items 1–4, ≈ 21 h.** That is one overnight run plus a
morning. It removes every Tier-3 caveat that currently forces a claim to be cut.

**Recommended: 1–6, ≈ 62 h.** Adds the bound on the headline number and the cost
attribution — the two things a reviewer is most likely to ask for.

### Running items 1–4

All four are wired into one unattended script:

```bash
bash scripts/finish_study.sh                # everything outstanding, in order
bash scripts/finish_study.sh --dry-run      # rehearse without running
bash scripts/finish_study.sh --only b1      # or a single step
```

Steps are independent and idempotent — a completed step is skipped, and a
failure does not abort the ones after it, so an overnight run delivers whatever
it can and reports the rest. It checks the metric stack before spending any GPU
time, verifies the Dirichlet shards are large enough for R=3, and regenerates
every table at the end, including B1's paired non-IID comparison.

Items 5–7 are deliberately not in it: they are judgement calls that depend on
what B1 returns, not mechanical follow-ups. Commands for them are in
[09_run_guide.md](09_run_guide.md).

B1's configs already exist as `ablation_study/configs/ablationB_e0_noniid.yaml`,
`ablationB_e1_noniid.yaml`, `ablationB_e2_noniid.yaml`.

> **Gate for B1:** Dirichlet(0.3) shard sizes are 4,798 / 2,715 / 6,998. The
> smallest caps the round count at `floor(2715/500) = 5`. Regenerate the manifest
> and re-check before running.

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

Still outstanding: the **299.202 MiB E0 figure** at 360M, which is a stale
measurement rather than a rendering problem and needs the re-run (item 4 above).

---

## Bottom line

The **systems contribution is finished and defensible**: verifiable provenance
for federated fine-tuning at bit-identical cost, demonstrated by 18/18 hash
equality across two architectures, with 100% tamper detection on real artefacts,
sub-0.5% measured overhead, and anchoring cost linear in participants
(R² = 0.999994) and flat across a 220× model-size range. Nothing outstanding
threatens any of it.

The **learning contribution is one run short**. Federation beating isolation by
34% of the centralized gap at 0.5B is a real, tightly-measured, budget-matched
result — but it is an IID result at a fixed 3-round budget, and the non-IID arm
that would generalise it currently has no matched baseline. Ablation B1 is 12
hours and is the difference between "federation helps under skew" being a claim
and being a conjecture.
