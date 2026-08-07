# 07 — Ablation conclusions

> ## ⚠ NOT YET WRITTEN
>
> **No ablation has been run.** This document contains the *decision rules*,
> fixed in advance, and the shape of the conclusions that each possible outcome
> would license. The conclusions themselves are `—` until
> [06](06_ablation_results.md) is filled.
>
> The point of writing this before the data is to make the analysis
> non-negotiable afterwards. If a result lands outside every branch below, say so
> explicitly rather than fitting a new branch to it.

---

## The question the study exists to answer

Does federated learning, at this scale, produce a model materially better than
what a participant gets training alone — and therefore is there anything worth
auditing?

**Baseline answer: barely.** `E2 − E0 = −0.00085 ± 0.00022`, which is 2.4% of
the isolation→centralized gap.

**Post-ablation answer:** `—`

---

## Branch 1 — Rounds explain it *(H-A1 holds)*

**Trigger:** `|E2 − E0|` grows monotonically in R and exceeds 0.005 at R=9.

**Then the paper says:** federated averaging's advantage is cumulative and the
3-round configuration was under-trained. R=9 becomes the headline configuration;
the round sweep becomes a figure; the audit-layer results carry over unchanged
(they are round-invariant by construction, and Ablation C's hash check confirms
it).

**Required in the write-up:** the full trajectory figure, and an explicit note
that the original 3-round result was budget-limited rather than converged.

**Observed:** `—`

---

## Branch 2 — Heterogeneity explains it *(H-B1 holds, H-A1 does not)*

**Trigger:** `|E2 − E0|` stays < 0.002 at R=9, but exceeds 0.01 at α=0.1.

**Then the paper says:** on IID data, isolated training is already near-optimal
because each client's shard is an unbiased sample of the target distribution —
FedAvg has nothing to add. Under label skew, averaging recovers what isolation
loses, and that is the regime federation is deployed in. The IID arm becomes a
control demonstrating the ceiling, not the headline.

**Required in the write-up:** the α sweep as the primary motivation figure; an
explicit statement that this is **label skew with balanced quantities** (per
H-B3), since `max_train_samples` caps below the smallest shard; and non-IID
baselines throughout, never IID-vs-non-IID comparisons.

**Observed:** `—`

---

## Branch 3 — Neither explains it

**Trigger:** `|E2 − E0|` < 0.002 at R=9 *and* < 0.002 at α=0.1.

**Then the paper says, honestly:** at 360M parameters with a 4.07% LoRA adapter,
FedAvg over three clients does not measurably beat isolated training on this
task. The contribution is reframed around the audit layer, which is where the
evidence is strongest anyway:

> *Verifiable provenance for federated fine-tuning at zero accuracy cost —
> demonstrated by bit-identical artefacts — 100% tamper detection at 0% false
> positives, sub-1% wall-clock overhead, and anchoring cost linear in
> participants and independent of model size.*

That is a complete systems contribution and does not require FedAvg to win. The
learning result becomes a scoped negative finding, reported rather than buried,
with the model-scale caveat stated (Ablation F / qwen-1.5b would be the natural
follow-up).

**This branch is not a failure.** It is the most likely single outcome given the
baseline, and a paper that reports it cleanly is more defensible than one that
quietly ships the 2.4% number in a table and hopes nobody computes it.

**Observed:** `—`

---

## Branch 4 — Isolation wins

**Trigger:** `E2 − E0` changes sign with more rounds or more skew.

**Then the paper says:** report it, prominently. A well-controlled negative
result about FedAvg at small scale — with matched update *and* data budgets,
which most FL papers do not do — is a contribution. The audit-layer results are
unaffected: the system anchors and verifies whatever the aggregation rule
produces.

**Observed:** `—`

---

## Sub-conclusions, independent of which branch fires

### Audit-layer zero-cost claim

**Rule:** the claim survives iff E2/E3/E4 adapters remain bit-identical under
**every** configuration tested (Ablations A, C, D). Any single divergence
invalidates it and indicates a bug, not a measurement.

**Observed:** `—`

### Communication overhead attribution

**Rule:** the +31.5% is attributed by Ablation D. If H-D2 holds and D2 returns
communication to ~299 MiB, then the overhead is the global model's IPFS
round-trip — an **implementation choice**, not a cost of auditability — and the
paper must say so, because it is a much weaker claim against the design than an
unattributed +31.5%.

**Observed:** `—`

### Verification cost

**Rule:** if H-D3 holds (D3 saves nothing measurable), report verification as
free. This is a useful negative: it says integrity checking is not what costs,
which strengthens the deployability argument.

**Observed:** `—`

### Generation metrics

**Rule:** if H-E2 holds and no ordering is significant even at 250 samples,
report loss and perplexity as the accuracy result and demote ROUGE-L/BLEU to a
collapse check with the sample count stated. Do not report them as a table row
that implies a comparison the data cannot support.

**Observed:** `—`

---

## Claims table — to be completed

| Claim | Baseline status | Post-ablation status | Evidence |
|---|---|---|---|
| Anchoring + IPFS cost zero accuracy | solid (bit-identical) | — | — |
| Zero-cost holds across hyperparameters | untested | — | Ablation C |
| Systems overhead <1% wall clock | solid | — | — |
| Comm overhead attributable to transport choice | unattributed | — | Ablation D |
| Verification is free | untested | — | Ablation D3 |
| 100% tamper detection, 0% FP | solid | — | — |
| Gas linear in N, flat in model size | solid | — | — |
| Federation beats isolation | **not supported** | — | Ablations A, B |
| Non-IID result vs matched baseline | **not measured** | — | Ablation B1 |
| Generation metrics usable | **no** | — | Ablation E |
| Holds at paper scale (1.5B) | untested | — | future |

---

## Writing order for the paper, once this is filled

1. **Lead the evaluation with hash equality**, not the loss table. It is a
   stronger claim, it is verifiable by a reviewer, and it pre-empts the
   "non-significance is not equivalence" objection that
   [EXPERIMENTS.md:114-122](../EXPERIMENTS.md#L114-L122) correctly anticipates.
2. **Then the motivation figure** — whichever of Branch 1/2/3 fired.
3. **Then the cost table**, with Ablation D's attribution, so the +31.5% is
   presented as a decomposed and understood number.
4. **Then E6 and E7**, which need no revision.
5. **Then the limitations section**, drawn from [08](08_shortcomings_and_roadmap.md).
   Write it before the conclusion, not after — the limitations here are known
   and specific, and naming them is cheaper than having them found.
