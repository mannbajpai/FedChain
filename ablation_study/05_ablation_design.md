# 05 — Ablation design

Six ablations. Hypotheses, predictions and decision rules are fixed **here**,
before any run, so the analysis in [07](07_ablation_conclusions.md) cannot drift
toward whatever the data happens to show.

## What this study has to fix

The baseline proved the audit layer is free and effective. It did not show that
the thing being audited is worth doing: **E2 − E0 = −0.00085 ± 0.00022**, i.e.
FedAvg recovers 2.4% of the isolation→centralized gap. Ablations **A** and **B**
exist to test the two explanations that would rescue that (too few rounds; IID
makes isolation nearly optimal). Everything else is supporting work.

> ### ⚠ Tier caveat added 2026-08-13 — read before using any threshold below
>
> **That 2.4% is the smollm2-360m figure**, and every decision threshold in this
> document was calibrated on it. The study then moved to **qwen-0.5b**, where the
> IID baseline is **−0.00964 ± 0.00093 = 34.0%** of the gap — five times the
> `|E2 − E0| < 0.002` cut-off that the "nothing is happening" branch uses.
>
> Ablation B1 was executed at qwen-0.5b. Its decision rules therefore never had a
> live null case, and its absolute triggers do **not** transfer. The *direction*
> triggers (does the gain grow as α falls? does E0 degrade faster than E2?) do
> transfer, and are what [07](07_ablation_conclusions.md) evaluates.
>
> This is recorded as a pre-registration defect rather than quietly rescaled. If
> further blocks are run, **re-derive the thresholds for the tier they run on**
> and record the derivation here before starting.

## Cost model

Measured from the baseline, per seed on the T600:

| Unit | Cost |
|---|---|
| Client training, 1 round | 437 s |
| Federated round (3 clients), training only | 1,311 s |
| Evaluation (500 loss + 50 gen) | 340 s |
| Generation, per sample | 4.4 s |
| Centralized training, per sample | 0.875 s |
| Full 3-round federated run | 4,750 s ≈ 79 min |

---

## Ablation A — Round count *(Tier 1, the critical one)*

**Question.** Does FedAvg's advantage over isolated training grow with rounds?

**Why.** The baseline loss curve was still descending when the budget ran out
(Δ r1→r2 = −0.0447, Δ r2→r3 = −0.0160, against a 0.0343 gap to centralized).
FedAvg's benefit is cumulative — each averaging step propagates information
between clients — so 3 rounds may be before the point where it separates from
isolation. Three rounds is where the compute budget ended, not where federated
training converges.

**Design.** Extend to **R = 9** on the IID shards. Shard size is 4,837 records,
so `9 × 500 = 4,500` fits without the window running off the end.

| Run | Arm | Rounds | Eval cadence | Seeds | Cost/seed |
|---|---|---|---|---|---|
| A1 | E2 FedAvg | 9 | every round | 3 | 4.1 h |
| A2 | E0 Local-only | 9 | stride 2 (r=1,3,5,7,9), per-client | 3 | 4.7 h |
| A3 | E1 Centralized | budget-matched at R=5 (7,500) and R=9 (13,500 samples) | final | 2 | 5.3 h |

A3 uses 2 seeds because E1 has by far the smallest seed variance (±0.0006).
**Total ≈ 37 h.** A1 and A2 are the load-bearing pair; A3 keeps the upper bound
budget-matched, without which the R=9 comparison is not a fair one.

**Pre-registered predictions.**

- **H-A1.** `|E2 − E0|` grows monotonically in R. Specifically ≥ 0.005 at R=9,
  versus 0.00085 at R=3.
- **H-A2.** E2 at R=9 lands near **2.014**. (Geometric extrapolation of the
  observed per-round deltas, ratio `0.01597/0.04470 = 0.357`, gives a remaining
  improvement of ~0.0089 from 2.0228.)
- **H-A3 — the risk.** Under budget matching E1 *also* improves with more data.
  If `E2 − E1` stays flat near 0.034 across R ∈ {3,5,7,9}, the honest reading is
  that federation carries a round-invariant cost at this scale, and H-A1 is the
  only thing that can still save the motivation.

**Decision rule.**

| Outcome | Conclusion for the paper |
|---|---|
| `\|E2−E0\|` grows with R and exceeds 0.005 at R=9 | Motivation holds. Report the curve, state that 3 rounds was under-trained, promote R=9 to the headline configuration. |
| `\|E2−E0\|` stays < 0.002 at R=9 | Rounds are not the explanation. Motivation now rests entirely on Ablation B. |
| `E2−E0` changes sign (isolation wins) | Report it. A negative result about FedAvg at small scale is publishable and honest; the audit-layer contribution stands regardless. |

---

## Ablation B — Data heterogeneity *(Tier 1)*

**Question.** Does FedAvg beat isolation when clients hold genuinely different
distributions?

**Why.** With 4,837 IID Dolly records per client, each client's shard is already
an unbiased sample of the target distribution, so averaging has little to add.
Federation is supposed to pay off under skew. **And the baseline cannot answer
this at all**: E5 is the only Dirichlet run, so there is no non-IID E0 to compare
against ([C5](04_changes.md#c5--non-iid-baselines-for-e0-e1-and-e2)).

**Design.** Dirichlet α ∈ {0.1, 0.3, 1.0}, plus the IID arm as α → ∞.

| Run | α | Arms | Seeds | Cost |
|---|---|---|---|---|
| B1 | 0.3 (shards exist) | E0, E1, E2 | 3 | 11.9 h |
| B2 | 0.1 (generate) | E0, E1, E2, E4 | 3 | 15.8 h |
| B3 | 1.0 (generate) | E0, E1, E2, E4 | 3 | 15.8 h |

**B1 is the single highest-value block in the study** — it is cheap, it closes
the gap [EXPERIMENTS.md:23](../EXPERIMENTS.md#L23) already flags, and it directly
tests the motivation.

**Hard constraint.** Dirichlet(0.3) shard sizes are 4,798 / 2,715 / 6,998. The
smallest caps the round count at `floor(2715/500) = 5`. Lower α will skew
further, so **regenerate the manifest and re-check the cap before every run**.
Any non-IID round sweep stops at 5 rounds, or `max_train_samples` drops.

**Pre-registered predictions.**

- **H-B1.** FedAvg's advantage grows as α falls: `|E2−E0|` ordered
  `α=0.1 > α=0.3 > α=1.0 > IID`.
- **H-B2.** E0 degrades faster than E2 as α falls, because isolated clients
  overfit their skewed slice while averaging regularises across them.
- **H-B3 — the caveat.** `max_train_samples: 500` caps every client below the
  smallest shard, so all clients contribute equally and **quantity skew is
  removed**. This is *label* skew with balanced quantities — the standard
  Dirichlet benchmark, but weaker than setups that also skew quantity. Expect a
  smaller effect than the FL literature reports, and say so in the paper.

**Decision rule.** If `|E2−E0|` at α=0.1 exceeds 0.01, the motivation is secured
and the paper's framing becomes "federation matters under heterogeneity, and we
make it auditable for free." If it stays under 0.002 at α=0.1, then at this model
scale FedAvg genuinely does not help, and the paper must be reframed around the
audit contribution alone.

---

## Ablation C — Local epochs *(Tier 2)*

**Question.** Does client drift appear, and does the audit layer's zero-cost
property survive it?

**Design.** `local_epochs ∈ {1, 2, 4}` at R=3, on both IID and α=0.3 shards.
Cost ≈ 21.6 h for 3 seeds.

**H-C1.** More local work → more client drift → E2 degrades at E=4 relative to
E=1 under skew, while possibly improving under IID (where drift is harmless
because all clients move toward the same optimum).

**H-C2.** E3/E4 remain **bit-identical** to E2 at every epoch setting. This is
the important one: it shows the zero-accuracy-cost result is not an artefact of
one hyperparameter choice.

---

## Ablation D — Audit-layer cost decomposition *(Tier 1, cheap)*

**Question.** Where exactly does the +31.5% communication overhead come from?

**Why.** The baseline reports E4's communication as a single number. A reviewer
will ask which part is intrinsic to auditability and which is an implementation
choice. Right now we cannot answer.

**Design.** Systems metrics only — run with `--skip-eval`, **1 seed** (systems
metrics are deterministic given the config). 3 rounds each.

| Run | Variant | Isolates | Cost |
|---|---|---|---|
| D1 | `log_global_model: false` | Global-model anchoring: 12 tx → 9 tx | 1.1 h |
| D2 | `ipfs_roundtrip_aggregation: false` | The aggregator's download leg | 1.1 h |
| D3 | `verify_hash_on_download: false` | Verification compute | 1.1 h |

**Pre-registered predictions.**

- **H-D1.** Gas drops from 3,785,372 to ~2.84M (3 of 12 transactions removed),
  and the global IPFS upload disappears.
- **H-D2.** Communication returns toward 299.2 MiB, confirming the entire +31.5%
  is the global model's round-trip rather than the anchoring itself.
- **H-D3.** Verification saves only hashing time (~13 ms/artefact per E6) and is
  **immeasurable** in wall clock. This variant exists precisely to show that
  verification is not the cost — a useful negative.
- **H-D4.** All three variants produce adapters **bit-identical** to E4. None of
  them touch the learning math, so any divergence is a bug.

---

## Ablation E — Evaluation fidelity *(Tier 1, no retraining)*

**Question.** Are the generation metrics usable at a defensible sample count?

**Design.** Re-score the **existing** baseline adapters at
`gen_num_samples ∈ {50, 250}` using `scripts/reevaluate.py`
([C4](04_changes.md#c4--raise-generation-sample-count-and-add-a-standalone-re-evaluation-path)).
No retraining: 6 arms × 3 seeds = 18 adapters × ~20 min ≈ **6 h**.

**H-E1.** CIs narrow by ~`sqrt(250/50) = 2.24×`; E5's ROUGE-L interval goes from
±0.042 to roughly ±0.019.

> **⚠ H-E1 was malformed, and was falsified on that account (2026-08-13).** The
> reported ±CI is a **between-seed** interval at n=3. `gen_num_samples` controls
> Monte-Carlo error *inside* each seed's point estimate, which the interval does
> not measure. Measured 50 → 250 CI ratios: 0.08×, 0.63×, 0.43×, 1.86× — three of
> four **widened**. Narrower generation intervals require more **seeds**. What
> 250 samples did buy was less noisy per-seed estimates, which is why the
> qwen-0.5b ROUGE-L ordering became coherent with the loss ordering. See
> [06 §E.2](06_ablation_results.md#e2-does-250-samples-buy-narrower-intervals).

**H-E2.** No between-arm ROUGE-L ordering becomes significant even at 250. If
so, the honest move is to report loss/perplexity as the accuracy result and
present generation metrics only as a collapse check.

Run this **after C3** (`nltk`) so the re-scored values use the intended backend.

---

## Ablation F — Federation size *(Tier 3)*

**Question.** Does the accuracy picture change with more clients?

**Design.** `num_clients ∈ {3, 5, 10}` at fixed total data (re-shard so the union
is constant). ~18 h for 3 seeds.

E7 already covers the *audit* cost of scale — gas is linear in N, flat in model
size. F covers the *learning* side, which is nice to have and not load-bearing.
Deprioritised accordingly.

---

## Priority and total cost

| Tier | Blocks | Cost | Buys |
|---|---|---|---|
| **1** | A, B1, D, E | **58 h** ≈ 2.4 days | The motivation, the cost decomposition, usable generation metrics |
| 2 | B2, B3, C | 53 h | The α curve, drift robustness |
| 3 | F, qwen-0.5b replication | 18 h + | Scale generality |

**Minimum viable subset**, if GPU time is short — A1 + A2 + B1 at 2 seeds each,
≈ **26 h**. That answers "does federation ever beat isolation, in rounds or in
skew?", which is the only question that currently threatens the paper. Report
2-seed results as preliminary and note the wider intervals.

## Analysis protocol

Fixed in advance:

1. **Pair everything.** Report `E_x − E_y` per seed with its own CI, not the
   difference of means. Seed variance is shared and cancels.
2. **Student's t at n=3 is 4.303.** At n=2 it is 12.706 — 2-seed results have
   intervals ~3× wider and must be labelled preliminary.
3. **Hash-equality first.** For every audit-layer comparison, report artefact
   hash equality before reporting loss. It is a stronger claim and it pre-empts
   the "absence of evidence is not equivalence" objection.
4. **No equivalence claims from non-significance.** If an equivalence claim is
   needed anywhere hashes cannot supply it, pre-register a margin and run a TOST.
5. **Generation metrics are a collapse check** until Ablation E says otherwise.
6. **Report the non-IID arm against non-IID baselines**, never against IID ones.

## What would falsify the paper's central claim

Stated in advance, so it is not rationalised later:

- Any adapter hash divergence between E2/E3/E4 under any configuration would
  mean the audit layer perturbs training. **Nothing survives that** — it is the
  core claim.
- A tamper attack that goes undetected, or a benign transformation that is
  flagged, would break the E6 result.
- Gas growing super-linearly in N, or at all in model size, would break the
  deployability argument.

None of these are expected — the mechanism is a SHA-256 commitment, and the
baseline evidence is strong — but they are the conditions under which the
contribution fails, and the paper is stronger for naming them.
