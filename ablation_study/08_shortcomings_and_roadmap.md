# 08 — Shortcomings and roadmap

Every known weakness, ranked by how much it threatens the paper. Each entry
states the problem, **how a reviewer will phrase the objection**, the fix, and
the cost.

Severity key: **S1** = threatens acceptance · **S2** = weakens a claim ·
**S3** = presentation.

---

## S1-1 — Federation barely beats doing nothing

**The problem.** `E2 − E0 = −0.00085 ± 0.00022` val loss. FedAvg over three
clients recovers 2.4% of the isolation→centralized gap. The paper argues
"federated learning is valuable, and we make it auditable for almost free"; the
second clause is proved to an unusually high standard, the first is close to
false on this evidence.

**How a reviewer will phrase it.**
> *Table 1 shows local-only at 2.0236 and FedAvg at 2.0228. Three clients
> training in isolation get essentially the same model as three clients running
> the proposed federated protocol. What does the system buy?*

**The fix.** Ablations A (rounds) and B (heterogeneity) — see
[05](05_ablation_design.md). If neither rescues it, Branch 3 in
[07](07_ablation_conclusions.md) reframes the contribution around the audit
layer, which stands on its own.

**Cost.** 37 h (A) + 12 h (B1). **Do not submit without this.**

---

## S1-2 — The non-IID arm has no matched baseline

**The problem.** E5 is the only Dirichlet run. E0–E4 all use IID shards,
verified from every config block. `comparison.md` reports E5's paired difference
against **IID E1**, conflating the partition change with the federation change.
The repo's own design doc already forbids this
([EXPERIMENTS.md:23](../EXPERIMENTS.md#L23)):

> Report Exp 5 against an Exp 0/1/2 rerun on the same non-IID shards, not
> against the IID numbers.

**How a reviewer will phrase it.**
> *The non-IID result is compared against IID baselines, so the reported
> difference confounds the partition with everything else. There is no evidence
> that federation helps under skew, which is the regime the paper motivates.*

**The fix.** Ablation B1: E0, E1, E2 on the existing α=0.3 shards.

**Cost.** 11.9 h. **Cheapest high-value fix in the study** — the shards already
exist and no code change is needed beyond the three configs in
[`configs/`](configs/).

---

## S1-3 — Local-only reports phantom communication

**The problem.** E0 reports 299.20 MiB of communication with no server, no
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

## S2-1 — Generation metrics cannot support any claim

**The problem.** At `gen_num_samples: 50`, ROUGE-L and BLEU CIs span 5–17% of
their values (E5's ROUGE-L is 0.2454 ± 0.0419). The rows are in the tables and
invite comparison they cannot support.

**How a reviewer will phrase it.**
> *The ROUGE-L confidence intervals overlap completely across all six arms. What
> is the reader supposed to conclude from this row?*

**The fix.** Ablation E — re-score existing adapters at 250 samples, no
retraining. If H-E2 holds and nothing becomes significant, demote both metrics
to a stated collapse check.

**Cost.** 6 h, no retraining.

---

## S2-2 — The +31.5% communication overhead is unattributed

**The problem.** E4 costs 393.6 MiB against E2's 299.2. The report gives one
number and no decomposition, so it reads as an intrinsic cost of auditability.
It probably is not — it is likely the global model's IPFS round-trip, which is an
implementation choice.

**How a reviewer will phrase it.**
> *A 31.5% communication increase is not "negligible overhead". Which part is
> the cost of auditing and which is the cost of your particular transport
> design?*

**The fix.** Ablation D. If H-D2 holds, the paper can state that the anchoring
itself is ~free and the transport overhead is a tunable design choice, with D1/D2
as evidence.

**Cost.** 3.3 h, systems-only, one seed.

---

## S2-3 — Single model tier

**The problem.** Everything is `smollm2-360m`, which the repo itself designates
as the pipeline shakedown tier. The paper configuration is `qwen-1.5b`.

**How a reviewer will phrase it.**
> *All results are on a 360M model. The accuracy conclusions may not survive at
> the scale the paper claims to target.*

**Mitigation already in hand.** The audit-layer conclusions are scale-free: hash
equality is a property of the commitment scheme, not the model, and E7 shows gas
is independent of artefact size across a 220× range. Say this explicitly — it
converts a weakness into a demonstrated property.

**The fix.** Replicate the reduced matrix (E0/E1/E2/E4, 3 seeds) at
`qwen-0.5b`, then `qwen-1.5b`.

**Cost.** ~3× and ~9× the 360M matrix respectively. Plan around it.

---

## S2-4 — Three seeds

**The problem.** `t = 4.303` at n=3. Every interval is wide by construction.

**Assessment.** Not currently the binding constraint — the effect sizes are.
E2−E0 is significant at 3 seeds; the issue is that it is *small*, and more seeds
would not change that. Raising to 5 seeds is worth doing only after Ablations A
and B establish which comparisons matter.

**Note.** Ablation A3 uses 2 seeds (`t = 12.706`). Those intervals are ~3× wider
and must be labelled preliminary.

---

## S2-5 — ROUGE backend silently substituted

**The problem.** `nltk` was absent, so every run fell back from the `evaluate`
library to the built-in ROUGE implementation. Consistent across runs, so internal
comparisons hold — but the absolute values are not the standard implementation
and must not be compared cross-paper.

**The fix.** [C3](04_changes.md#c3--install-nltk-or-pin-the-metric-implementation-explicitly).
Either install `nltk`, or state the built-in implementation in the paper. Both
are defensible; silently falling back is not.

**Cost.** Minutes, plus re-evaluation (folded into Ablation E).

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

## S3-3 — E6 runs on a single seed and an archived adapter set

E6 uses seed 42 adapters from an archived E4 run
(`results/_archive_prefix_20260804_222225/`). 250 trials is plenty of statistical
power for a 100%/0% result, but the provenance is untidy. Re-run against the
current E4 outputs and state the source. Cost: minutes.

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

---

## Roadmap

| Phase | Work | Cost | Unblocks |
|---|---|---|---|
| **0** | Apply C1–C4 | ~1 day of code | Everything — the metrics are wrong without C1 |
| **1** | Ablation B1 (non-IID baselines) | 12 h | S1-2 |
| **2** | Ablation A (round sweep) | 37 h | S1-1 |
| **3** | Ablations D + E | 9 h | S2-1, S2-2 |
| **4** | Ablations B2/B3, C | 53 h | The α curve, drift robustness |
| **5** | `qwen-0.5b` replication | ~3× matrix | S2-3 |
| **6** | `qwen-1.5b` paper run | ~9× matrix | The headline configuration |

**Critical path to a submittable paper:** Phase 0 → 1 → 2 → 3 ≈ **58 h of GPU
plus a day of code**. Phases 4–6 strengthen; phases 0–3 are what stand between
the current results and a paper whose motivation survives review.

**If time is very short:** Phase 0, then A1+A2+B1 at 2 seeds ≈ 26 h. That
answers "does federation ever beat isolation?" — the only open question that
threatens acceptance — and everything else can be labelled future work.
