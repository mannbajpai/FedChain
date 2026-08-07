# 03 — Baseline conclusions

What the first pass established, what it failed to establish, and which of those
gaps threaten the paper.

## Established

### 1. The audit layer is free — provably, not statistically

E2, E3 and E4 produce **bit-identical adapters** (9/9 hashes, all 3 seeds). The
audit and transport layers do not perturb the learning math at all; they observe
it. The paired accuracy difference is exactly `0.00000`, and that is a
consequence of byte-equality rather than a statistical finding.

This matters for how the paper argues. [EXPERIMENTS.md:114-122](../EXPERIMENTS.md#L114-L122)
correctly notes that with 3 seeds you cannot claim *equivalence* from a
non-significant difference — absence of evidence is not evidence of absence, and
an equivalence claim would need a pre-registered margin and a TOST. **The hash
equality sidesteps that objection entirely.** Lead with it. A reviewer can
verify byte-equality; they cannot be argued into accepting a null result.

### 2. Systems cost is bounded and small

+0.71% wall clock for the full FedChain stack over plain FedAvg. Chain latency
is 0.03% of the run, IPFS 0.43%. The one real cost is **+31.5% communication**
(299.2 → 393.6 MiB), attributable to the global model's IPFS round-trip. Gas is
3.79M for 12 transactions. On the local devnet that is 0.0038 ETH; the number
that transfers to a real deployment is the gas, not the latency.

### 3. The audit layer actually works

200/200 malicious artefacts rejected across four attack classes, 0/50 false
positives on benign re-serialisation, ~13 ms and ~242k gas per verification.
The zero false-positive rate is the non-obvious part: it validates the canonical
-JSON hashing decision, without which the anchored commitment would be
unreproducible across processes and the whole scheme would be unusable by an
auditor.

### 4. The design scales the way the architecture predicts

Gas is **linear in client count** (47.6× for 100×; per-client cost asymptotes to
~294k) and **flat in artefact size** (311k gas from 0.22 MiB to 49 MiB, a 220×
range) because only a 32-byte digest is anchored. Client-side hashing at
~1.4 GiB/s is the only size-dependent term. These are the numbers that make a
deployability argument, and they are clean.

### 5. The result survives label skew

E5 − E4 = +0.00212 ± 0.00070 val loss. Significant, ~0.1%, and the integrity
layer behaves identically (9/9 checks, 12/12 tx, 0 failed transfers). Whatever
else is true, the audit layer is not IID-dependent.

## Not established — and this is the problem

### The motivation gap: federation barely beats doing nothing

**E2 − E0 = −0.00085 ± 0.00022 val loss.** FedAvg over three clients recovers
**2.4%** of the distance from isolated training to the centralized upper bound.

The paper's structure is "federated learning is valuable, and we make it
auditable for almost free." The second clause is proven to an unusually high
standard. The first clause is, on this evidence, close to false. A reviewer who
reads Table 1 carefully will notice that three clients training alone get
essentially the same model as three clients running FedAvg, and will ask why the
system exists.

Three candidate explanations, which the ablation must distinguish:

1. **Too few rounds.** The loss curve is still descending at round 3
   (Δ r2→r3 = −0.016 against a 0.034 gap to centralized). FedAvg's advantage
   over isolation is cumulative — each round of averaging propagates information
   between clients — so 3 rounds may simply be before the point where it shows.
2. **The IID split makes isolation nearly optimal.** With 4,837 IID Dolly
   records per client, each client's local data is already an unbiased sample of
   the target distribution. There is little for averaging to add. Federation is
   supposed to pay off when clients hold *different* distributions.
3. **The effect is real and small at this model scale.** A 360M model with a
   4.07% LoRA adapter over 500 samples/round may simply not have the capacity or
   the signal for aggregation to matter.

Explanations 1 and 2 are testable and are exactly what Ablations A and B target.
Explanation 3 is testable by moving up the model ladder.

### The non-IID arm has no matched baseline

E5 is the **only** run on the Dirichlet(0.3) shards. E0, E1, E2, E3 and E4 all
use the IID shards (`data/client*.jsonl`), verified from every config block.
[EXPERIMENTS.md:23](../EXPERIMENTS.md#L23) states the requirement explicitly:

> Report Exp 5 against an Exp 0/1/2 rerun on the same non-IID shards, not
> against the IID numbers.

That rerun was not done. `comparison.md` therefore reports E5's paired
difference against **IID E1**, which conflates the partition change with the
federation change. Worse, the single question that could rescue the motivation —
*does FedAvg beat local-only under skew?* — requires a non-IID E0, which does
not exist.

This is the highest-value cheap fix in the whole study: three runs (E0, E1, E2
on the Dirichlet shards) at ~79 min each per seed.

### Generation metrics are not usable

ROUGE-L and BLEU CIs run ±0.013 to ±0.042 on values of 0.24 and 0.04 — up to 17%
of the value. At `gen_num_samples: 50` these rows can support "generation did
not collapse" and nothing else. [EXPERIMENTS.md:123-127](../EXPERIMENTS.md#L123-L127)
already says so. They should either be raised to 200+ or removed from the tables;
leaving them in at 50 invites a reviewer to compute the interval themselves.

### Single model tier

Everything here is `smollm2-360m`, which the repo itself designates as the
pipeline shakedown tier. The paper configuration is `qwen-1.5b`. Nothing in the
audit-layer conclusions should change with scale — the hash-equality argument is
scale-free and E7 shows gas is size-independent — but every accuracy conclusion
is provisional until at least one larger tier is run.

## What transfers to the paper as-is

| Claim | Status | Evidence |
|---|---|---|
| Anchoring + IPFS cost zero accuracy | **solid** | bit-identical hashes, 3 seeds |
| Systems overhead <1% wall clock | **solid** | +0.71% vs FedAvg |
| Communication overhead +31.5% | **solid**, needs attribution | E4 vs E2; decomposed in Ablation D |
| 100% tamper detection, 0% FP | **solid** | 250 trials |
| Gas linear in N, flat in model size | **solid** | E7 sweeps |
| Audit layer survives non-IID | **solid** | E5 integrity + accuracy |
| Federation is worth auditing | **not supported** | E2 − E0 = 2.4% of the gap |
| E5 vs baseline under skew | **not measured** | no non-IID E0/E1/E2 |
| ROUGE-L / BLEU differences | **not usable** | CIs up to 17% of value |
| Results hold at paper scale | **untested** | 360M only |

The system contribution is in good shape. The empirical motivation for the
system is not, and that is what the ablation study is for.
