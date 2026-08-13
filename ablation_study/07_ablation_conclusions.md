# 07 — Ablation conclusions

Written 2026-08-13, after B1 and E. The decision rules below were fixed in
[05](05_ablation_design.md) before any run; the numbers are in
[06](06_ablation_results.md). Where a result lands outside the pre-registered
structure it is declared as such rather than fitted to a branch.

> **Status: two of six blocks executed (B1, E).** Ablations A, B2, B3, C, D and F
> are not run. Conclusions below are scoped to what B1 and E can carry.

---

## The question the study exists to answer

Does federated learning, at this scale, produce a model materially better than
what a participant gets training alone — and therefore is there anything worth
auditing?

**Answer: yes at qwen-0.5b, and more so under label skew than under IID.**

| | Recovery of the isolation→centralized gap |
|---|---|
| smollm2-360m, IID | 2.4% |
| qwen-0.5b, IID | 34.0% ± 4.3% |
| qwen-0.5b, **Dirichlet α=0.3** | **41.5% ± 7.7%** |

---

## Before the branches: the thresholds were calibrated at the wrong tier

[05](05_ablation_design.md) fixed its triggers around
`E2 − E0 = −0.00085 ± 0.00022` — the **smollm2-360m** figure — and set "nothing
is happening" at `|E2 − E0| < 0.002`. B1 was executed at **qwen-0.5b**, where the
IID baseline is `−0.00964`, already five times that threshold before any ablation
ran.

So the branch structure never had a live "nothing is happening" case at the tier
the ablation was run on. This is a pre-registration defect, not a data problem —
the thresholds were never re-derived when the study moved tiers. It is recorded
here rather than quietly rescaled, and the branches are evaluated below on their
*direction* triggers, which do transfer, rather than on their absolute cut-offs,
which do not.

The corrected premise is in [06 §B.6](06_ablation_results.md#the-premise-correction).
Any sentence in the paper of the form "FedAvg barely helps, so we tested skew"
must not be written: at the tier the skew test was run on, FedAvg already
recovered a third of the gap.

---

## Branch 1 — Rounds explain it *(H-A1)*

**Trigger:** `|E2 − E0|` grows monotonically in R and exceeds 0.005 at R=9.

**Observed:** `not tested`. Ablation A was not run.

**What is known.** Both loss curves are still descending at R=3. Per-round deltas
at qwen-0.5b are `−0.0353, −0.0106` (IID) and `−0.0357, −0.0113` (α=0.3), a decay
ratio near 0.31. Every accuracy figure in this study is a **fixed-budget** number
at 4,500 updates, R=3 — not a converged one. The 34.0% and 41.5% recoveries are
therefore statements about a budget as much as about federation, and the paper
must label them that way.

Branch 1 remains open and is the largest single unknown.

---

## Branch 2 — Heterogeneity explains it *(H-B1)*

**Trigger:** `|E2 − E0|` exceeds 0.01 under skew.

**Observed: fires, on the one contrast that was run.**

| Evidence | Value |
|---|---|
| `\|E2 − E0\|` at α=0.3, qwen-0.5b | **0.01626 ± 0.00128** — above the 0.01 trigger |
| Growth from IID | 0.00964 → 0.01626 = **1.69×**, intervals disjoint |
| Recovery of the centralized gap | 34.0% → **41.5%** |
| H-B2 mechanism: E0 degrades faster than E2 as α falls | E0 **+0.01026**, E2 **+0.00364** — E0 degrades **2.82×** faster |
| Control: E1 is partition-invariant | 2.0499 (IID) vs 2.0492 (α=0.3) — unmoved, as it must be |

**What the paper may say:** under label skew, isolated clients lose ground while
the averaged model largely holds its position, and the centralized bound does not
move. That is the predicted mechanism (H-B2), observed directly, with a clean
control confirming the repartition did not change task difficulty.

**What the paper may not say:** that `|E2 − E0|` is *monotone in α*. H-B1 ordered
four points; **one contrast was run**. α=0.1 and α=1.0 were not, so the result is
a direction between two partitions, not a curve. Nor may the effect be described
as large: `max_train_samples: 500` caps every client below the smallest shard, so
all three contributed equally and **quantity skew is removed** (H-B3 holds). This
is label skew with balanced quantities — the standard Dirichlet benchmark, and
weaker than setups that skew both.

---

## Branch 3 — Neither explains it

**Trigger:** `|E2 − E0| < 0.002` under both more rounds and more skew.

**Observed: ruled out at qwen-0.5b.** `|E2 − E0|` is 0.00964 (IID) and 0.01626
(α=0.3), both significant, both far above 0.002.

It is **not ruled out at smollm2-360m**, where the IID figure is 0.00085 and no
non-IID arm exists. The branch that closed at 0.5B is still open at 360M, and the
study contains no evidence either way.

Note what actually closed this branch: **model scale**, which was not one of the
pre-registered explanations. [10](10_two_model_results.md) declares that outcome
as sitting outside the branch structure, and it still does. Ablation B1 then
added skew on top of an already-open gap.

---

## Branch 4 — Isolation wins

**Trigger:** `E2 − E0` changes sign.

**Observed: ruled out.** The sign favours FedAvg in every measured cell — both
tiers, both partitions, all three seeds, significant everywhere. No configuration
tested produced isolation beating aggregation on validation loss.

---

## Sub-conclusions

### Audit-layer zero-cost claim

**Rule:** survives iff E2/E3/E4 adapters remain bit-identical under every
configuration tested.

**Observed: survives, and is now broader than it was.**

| Configuration | Evidence |
|---|---|
| IID, R=3, both tiers | 18/18 global-model SHA-256 identical across E2/E3/E4 |
| **α=0.3, R=3, qwen-0.5b** | **27/27 client adapter SHA-256 identical** between B1-E2 and E5; loss, perplexity, ROUGE-L and BLEU all identical to 6 dp |

The claim is no longer IID-only. It remains tested at **one** `local_epochs`, one
LoRA rank and one round count — Ablation C is what would generalise it, and the
mechanism argument (an out-of-band SHA-256 commitment cannot reach the optimizer)
is an argument, not a measurement.

### Communication overhead attribution

**Rule:** the +31.8% is attributed by Ablation D.

**Observed: half-attributed without D, from arms already in hand.** E3 (chain, no
IPFS) records **302.86 MiB — byte-identical to E2**. Anchoring therefore adds
*zero* communication, and the entire +31.8% arrives with IPFS transport. What
remains unattributed is the split *within* IPFS: client uploads versus the
aggregator's download leg versus the global model's round-trip. D2 is what
resolves it, and until it runs the paper should say "the overhead is IPFS
transport, decomposition pending" rather than leaving it unattributed.

### Verification cost

**Rule:** if D3 saves nothing measurable, report verification as free.

**Observed: supported indirectly, not tested.** E6 measures mean verification
latency at **13.1 ms per artefact** against a ~4,780 s round — roughly 3 parts per
million for 9 artefacts. D3 would confirm it directly; the E6 figure is already
strong enough to state as a measurement, provided it is attributed to E6 and not
to an ablation that did not run.

### Generation metrics

**Rule:** if no ordering is significant even at 250 samples, demote to a collapse
check.

**Observed: demote, with one exception.** At qwen-0.5b, E1 significantly beats
E0, E2 and E5 on ROUGE-L @250, and the ordering now matches the loss ordering. At
360M nothing separates. Critically, the contrast the motivation depends on —
E2 − E0 under skew — is **not** resolved by generation metrics: loss says
+0.0163 (significant), ROUGE-L says −0.0008 ± 0.0044 (nothing).

So: **loss and perplexity are the accuracy result.** ROUGE-L/BLEU @250 appear as
a supporting collapse check with sample count and backend stated, and the
centralized-vs-rest gap is the only ordering they are quoted for.

Two further findings from E, both of which change how the paper reports
uncertainty:

1. **H-E1 was malformed.** Raising `gen_num_samples` 50 → 250 did not narrow the
   intervals; three of four widened. The reported ±CI is a **between-seed**
   interval at n=3, and generation sample count does not touch seed-to-seed
   variance. Narrower generation CIs require **more seeds**. What 250 samples
   bought was a less noisy per-seed point estimate, which is why the qwen
   ordering became coherent.
2. **A scorer split contaminated the stored generation metrics.** All 30
   main-table runs used the `builtin` fallback; B1's 6 arms used `evaluate`
   ([06 §E.0](06_ablation_results.md#e0--the-backend-split-read-this-before-any-generation-number)).
   ROUGE-L/BLEU in `results/*/comparison.md` are therefore not comparable across
   those two tables. `reeval250` is the only self-consistent source. The
   contamination is **historical**: the `require_metric_backend: evaluate` guard
   landed 2026-08-07, the affected runs stored `''`, and every run since has
   stored `'evaluate'`. Nothing needs fixing in the code for it not to recur —
   but the paper must not quote the contaminated columns.

---

## Claims table

| Claim | Baseline status | Post-B1/E status | Evidence |
|---|---|---|---|
| Anchoring + IPFS cost zero accuracy | solid (bit-identical) | **solid, and now under skew** | 18/18 global IID + 27/27 client at α=0.3 |
| Zero-cost holds across hyperparameters | untested | **still untested** — one epoch setting, one rank, one R | Ablation C |
| Systems overhead <1% wall clock | solid | solid | 0.45% measured; E4−E2 +0.16% ± 0.22% (ns) |
| Comm overhead attributable to transport choice | unattributed | **half-attributed**: 0% from chain, 100% from IPFS; split within IPFS pending | E3 vs E2; Ablation D2 |
| Verification is free | untested | **supported** (13.1 ms/artefact) | E6; D3 would confirm |
| 100% tamper detection | solid | solid | 200/200 @360M, 80/80 @0.5B |
| 0% false-positive rate | **overstated at 0.5B** | **still overstated** — 0/20 bounds FPR at only 13.9% | Re-run E6 at 50 trials |
| Gas linear in N, flat in model size | solid | solid | R² = 0.999994; 220× size range moves gas 0.0077% |
| Federation beats isolation | not supported (at 360M) | **supported at qwen-0.5b**, 34.0% IID / 41.5% α=0.3 | E0/E1/E2 + B1 |
| Federation helps *more* under skew | not measured | **supported, one contrast** — 1.69×, disjoint intervals | B1 |
| Non-IID result vs matched baseline | not measured | **measured** | B1 |
| Non-IID result at 360M | not measured | **still not measured** | B1 at 360M |
| Generation metrics usable | no | **partially** — centralized-vs-rest at 0.5B only | Ablation E |
| Holds at paper scale (1.5B) | untested | untested | future |

---

## What the paper can claim today

**Systems contribution — complete.** Verifiable provenance for federated
fine-tuning at bit-identical cost, demonstrated across two architectures and two
data partitions, with 100% tamper detection on real adapters, sub-0.5% measured
wall-clock overhead, anchoring cost linear in participants (R² = 0.999994) and
flat across a 220× model-size range, and zero communication attributable to
anchoring itself.

**Learning contribution — real, and scoped.** At qwen-0.5b under a matched
4,500-update budget, FedAvg recovers 34.0% ± 4.3% of the isolation→centralized
gap on IID shards and 41.5% ± 7.7% under Dirichlet(0.3), the absolute gain
growing 1.69× with disjoint intervals, via the predicted mechanism (isolated
clients degrade 2.82× faster than the averaged model while the centralized bound
is unmoved).

Every one of those numbers carries three qualifiers that must travel with it:
**fixed budget** (R=3, still descending), **label skew only** (quantity skew
removed by the sample cap), and **training-noise intervals** (all three seeds
share one partition, so the CIs understate uncertainty on any partition-dependent
claim).

## What the paper must not claim

- That `|E2 − E0|` is monotone in α. One contrast was run.
- That the non-IID result generalises to 360M. No such arm exists.
- Any ROUGE-L or BLEU ordering read from `comparison.md`. Wrong scorer split.
- A 0% false-positive rate at 0.5B. 0/20 bounds it at 13.9%, not 0%.
- That 34.0%/41.5% is "the benefit of federating". It is the benefit at R=3.
- That the +31.8% communication overhead is intrinsic to auditability. It is
  IPFS transport; anchoring adds zero.

---

## Writing order for the paper

1. **Lead the evaluation with hash equality**, not the loss table — 18/18 global
   under IID and 27/27 client under skew. Stronger than any statistical test,
   reviewer-verifiable, and it pre-empts the "non-significance is not
   equivalence" objection.
2. **Then the motivation.** Branch 2 fired: the figure is the two-partition
   contrast at qwen-0.5b, with the 360M tier alongside it as the low-capacity
   case. State plainly that the tier ladder — not an ablation — is what first
   opened the gap.
3. **Then the cost table**, with the E3-vs-E2 result up front: anchoring costs
   zero communication, IPFS transport costs +31.8%, decomposition pending.
4. **Then E6 and E7**, after the 50-trial re-run at qwen so the FP bound is 5.8%
   rather than 13.9% and the two tiers use one protocol.
5. **Then limitations**, from [08](08_shortcomings_and_roadmap.md), extended with
   the three items this document adds: the R=3 budget, the single-partition
   seeds, and the pre-registration threshold defect above.
