# 12 — Paper plan: 4–5 pages including references

Written 2026-08-14. The experimental programme is closed at two runs
([09 steps 1–2](09_run_guide.md)); this document fixes what the paper claims,
which table each claim is read from, and what must not be written.

Run everything first:

```bash
bash scripts/run_final.sh          # ~13 h, unattended
```

It ends with a machine verdict — `COMPLETE`, `INCOMPLETE` (a run is missing), or
`HASH DIVERGENCE` (a bug that falsifies the central claim). Write nothing until
it prints `COMPLETE`. Every number below then lives in
`results/paper/tables.md`, `tables.tex` and `paper_numbers.json`.

---

## The contribution, in one sentence

> Federated fine-tuning of an LLM can be given end-to-end verifiable provenance —
> on-chain commitments plus content-addressed transport — at **provably zero
> accuracy cost**, demonstrated not by a non-significant difference but by
> **bit-identical artefacts**, across two model scales and two data partitions.

## What is actually novel

Four things, in the order a reviewer will weigh them.

**N1 — Bit-identity as the evaluation standard.** The literature on
blockchain-audited FL argues negligible accuracy impact from a *statistically
insignificant* difference. That is the weakest possible form of the claim:
absence of evidence, at n=3, with wide intervals. This paper instead reports
SHA-256 equality over the trained artefacts. It is exact, it needs no
statistical test, it cannot be produced by an underpowered experiment, and a
reviewer can re-derive it. **Lead with this.** It is the cheapest paper-level
differentiator available and the data is already in hand.

**N2 — Canonical serialization is what makes an anchored commitment
reproducible.** PEFT stores `target_modules` in a `set`, and Python salts string
hashes per process, so two honest runs producing bit-identical *weights*
serialise `adapter_config.json` differently. Hashing raw bytes therefore makes
the anchored digest irreproducible: an auditor who retrains from the same seed
and data computes a different hash and concludes the artefact was tampered with.
Folding the config in as canonical JSON (sorted keys, sorted string lists) fixes
it — and the **benign-control column of the tamper experiment is the measurement
that proves it**, not an assertion. This is a small, concrete, checkable
systems insight and it is the kind of detail that separates a real
implementation from a diagram.

**N3 — The audit cost decomposes, and anchoring is the free part.** Anchoring
adds *zero* communication: E3's volume is byte-identical to un-audited FedAvg.
The entire +31.5–31.8% arrives with IPFS transport, which is an implementation
choice rather than a cost of auditability. Gas is linear in participants
(R² > 0.9999) and flat across a 220× artefact-size range, because what is
anchored is 32 bytes regardless of model size. Most papers report one aggregate
overhead number; decomposing it is what lets this one say the overhead is
*tunable*.

**N4 — A 2×2 on scale × skew with matched baselines.** The learning result is
reported against isolated *and* pooled bounds on the *same* shards, at a matched
4,500-update budget, with a partition-invariance control on the centralized arm.
That control is worth a sentence of its own: E1 is unmoved by re-partitioning
(2.0499 vs 2.0492 at 0.5B), which is what shows the Dirichlet split
redistributed the corpus rather than changing the task.

N1 and N2 are the durable contributions. N4 is a supporting result — do not let
it carry the paper, because it is a fixed-budget number at one α.

---

## Section plan and page budget

Four to five pages including references is roughly **3.5 pages of body**. That
budget does not fit six tables. Promote four, demote two to prose.

| § | Content | Pages | Tables |
|---|---|---|---|
| 1 | Introduction + contributions | 0.6 | — |
| 2 | Related work (FL auditing, blockchain+FL, LoRA/PEFT) | 0.4 | — |
| 3 | Design: commitment scheme, canonical hashing (N2), transport | 0.7 | *(1 small figure)* |
| 4 | Experimental setup: tiers, arms, budget matching, seeds | 0.4 | — |
| 5 | Results | 1.2 | T1, T2, T3, T4 |
| 6 | Limitations | 0.3 | — |
| 7 | Conclusion | 0.15 | — |
| — | References | 0.5–1.0 | — |

**Promote (from `results/paper/tables.md`):**

- **T1 — main result.** The 2×2. Compress to 5 columns by dropping E1 into the
  caption if space is short.
- **T2 — hash equality.** The paper's core evidence. Keep every row.
- **T3 — systems cost.** Trim to E2/E3/E4 at one tier plus a sentence saying the
  other tier agrees.
- **T4 — tamper detection.** Keep the benign-control row; it is N2's evidence.

**Demote to prose:**

- **T5 — gas scaling.** Two sentences plus the fitted equation inline:
  *"gas = 301,120 + 290,533·N (R² = 0.999994) over N ∈ {1…100}, and varies by
  0.0077% across a 220× artefact-size range."*
- **T6 — generation quality.** One sentence: ROUGE-L/BLEU at 250 decodes on a
  single scorer agree with the loss ordering at 0.5B and separate nothing at
  360M; reported as a collapse check. Full table to the repo.

A figure earns its place over a table only for the per-round loss trajectory
(showing the budget is not convergence) — and only if § 6 would otherwise have to
argue it in prose.

---

## Claim → evidence map

Every sentence in § 5 should be traceable to one row. If it is not in this table,
it is not in the paper.

| Claim | Source | Guard |
|---|---|---|
| Anchoring + IPFS cost zero accuracy | T2 — all client and global artefacts bit-identical | State it as exact equality, never as "not significant" |
| The commitment is reproducible across honest re-runs | T4 benign control, 0 false positives | Quote the **exact bound** (≤5.8% at 50 trials), not "0%" |
| Tampering is detected | T4 — four perturbations, 100% | Quote the miss bound alongside |
| Anchoring adds no communication | T3 — E3 comm byte-identical to E2 | The +31.5% is IPFS; say so in the same sentence |
| Anchoring cost is size-independent and N-linear | § 5 prose from T5 | — |
| Wall-clock overhead is sub-1% | T3 Δ time, plus 13 ms/artefact verify latency | Quote the measured latency; the Δ time interval spans zero |
| Federation beats isolation | T1, E0−E2 with CI | Always with "at a matched 4,500-update budget, R=3" |
| Skew widens the margin | T1, both partitions, per tier | One α. Never say "monotone in α" |
| The Dirichlet split is valid | T1 E1 column, unchanged across partitions | This is the control — give it a clause |

---

## Prohibited sentences

Each of these is false, unsupported, or contradicted by the study's own data.

1. **"0% false-positive rate."** 0/50 bounds it at 5.8%. Write the bound.
2. **"FedAvg barely helps, so we tested heterogeneity."** At 0.5B it already
   recovered 34% under IID. This was the study's original premise and the data
   does not support it — see [06 §B.6](06_ablation_results.md#the-premise-correction).
3. **"The benefit grows monotonically with skew / with scale."** Two points in
   each direction is a direction, not a law.
4. **Any ROUGE-L or BLEU value from a `comparison.md`.** Those columns mix two
   scorers. Only `results/paper/` Table 6 is single-scorer.
5. **"Generation metrics became more precise at 250 samples."** The intervals are
   between-seed and did not narrow; three of four widened.
6. **"E2 and E3 are statistically equivalent."** Non-significance is not
   equivalence — and the hash result makes the statistical framing unnecessary.
7. **"Converged."** Every loss curve is still descending at R=3.

---

## Limitations section (§ 6) — write it before the conclusion

Six sentences, drawn from [08](08_shortcomings_and_roadmap.md). Naming these is
cheaper than having them found.

1. **Fixed budget, not convergence.** R=3, 4,500 updates, curves still
   descending (decay ratio ≈ 0.31/round).
2. **Seeds share a partition.** All three seeds read the same shards, so every
   interval is a training-noise interval and understates uncertainty on the
   partition-dependent results.
3. **One α, label skew only.** `max_train_samples` caps below the smallest shard,
   so quantity skew is removed; this is the standard Dirichlet benchmark and
   weaker than setups that skew both.
4. **Small scale.** 360M and 0.5B, three clients, one dataset (Dolly-15k).
5. **Devnet latency.** Gas transfers to a real deployment; anvil's ~0.12 s/tx
   does not. Discuss what the gas costs on an L2.
6. **Out of scope by construction.** A malicious client that anchors a poisoned
   adapter it genuinely trained passes integrity checking — the hash matches.
   That is Byzantine robustness, orthogonal to provenance. Privacy likewise:
   anchoring a digest leaks nothing, but LoRA updates are not DP.

Item 6 is a strength if framed as a threat model rather than a gap. Say the
aggregator is honest and transport is not, and that this is exactly the boundary
the commitment scheme is designed for.

---

## Reference targets

Roughly 15–20 entries fits half a page in two-column format. Cover:
FedAvg and the FL baseline; non-IID/Dirichlet benchmark protocol; LoRA and
QLoRA; PEFT; the instruction-tuning dataset; two or three blockchain-for-FL
systems (the works this paper's evaluation standard is contrasted against);
content-addressed storage; and ROUGE/BLEU. Pull the exact model, dataset and
library versions from `paper_numbers.json` and the run context blocks so the
setup section is reproducible without guesswork.
