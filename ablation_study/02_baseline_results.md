# 02 — Baseline results

Everything below is read directly from `results/smollm2-360m/`. Derived values
show their derivation. Three seeds: 42, 43, 44. Student's *t* critical value at
n=3 is **4.303** (using 1.96 would understate every interval by more than half).

## Accuracy — per seed

| Arm | Metric | seed 42 | seed 43 | seed 44 | mean ± 95% CI |
|---|---|---|---|---|---|
| E0 Local-only | val loss | 2.02411 | 2.02334 | 2.02340 | 2.0236 ± 0.0011 |
| E1 Centralized | val loss | 1.98853 | 1.98818 | 1.98861 | **1.9884 ± 0.0006** |
| E2 FedAvg | val loss | 2.02336 | 2.02243 | 2.02250 | 2.0228 ± 0.0013 |
| E3 FL+chain | val loss | 2.02336 | 2.02243 | 2.02250 | 2.0228 ± 0.0013 |
| E4 FedChain | val loss | 2.02336 | 2.02243 | 2.02250 | 2.0228 ± 0.0013 |
| E5 non-IID | val loss | 2.02517 | 2.02478 | 2.02470 | 2.0249 ± 0.0006 |
| E0 | perplexity | 7.56940 | 7.56354 | 7.56402 | 7.5657 ± 0.0081 |
| E1 | perplexity | 7.30478 | 7.30219 | 7.30535 | **7.3041 ± 0.0042** |
| E2/E3/E4 | perplexity | 7.56370 | 7.55668 | 7.55722 | 7.5592 ± 0.0097 |
| E5 | perplexity | 7.57738 | 7.57447 | 7.57383 | 7.5752 ± 0.0047 |
| E0 | ROUGE-L | 0.24465 | 0.25377 | 0.24453 | 0.2477 ± 0.0132 |
| E1 | ROUGE-L | 0.22823 | 0.21909 | 0.23289 | 0.2267 ± 0.0174 |
| E2/E3/E4 | ROUGE-L | 0.24696 | 0.24883 | 0.23681 | 0.2442 ± 0.0161 |
| E5 | ROUGE-L | 0.22686 | 0.24962 | 0.25975 | 0.2454 ± 0.0419 |
| E0 | BLEU | 0.04036 | 0.04638 | 0.04261 | 0.0431 ± 0.0076 |
| E1 | BLEU | 0.04169 | 0.04172 | 0.04555 | 0.0430 ± 0.0055 |
| E2/E3/E4 | BLEU | 0.04248 | 0.03889 | 0.04791 | 0.0431 ± 0.0113 |
| E5 | BLEU | 0.03291 | 0.04279 | 0.05101 | 0.0422 ± 0.0225 |

> **Do not quote the ROUGE-L or BLEU rows.** At `gen_num_samples: 50` the
> intervals span 5–17% of the values (E5's ROUGE-L CI is ±0.0419 on 0.2454).
> They are a sanity check that generation did not collapse, nothing more.

## The central result — E2/E3/E4 are bit-identical

Not "statistically indistinguishable." **Identical artefacts.** All 9 client
adapter hashes (3 clients × 3 rounds) match exactly across E2, E3 and E4, in
every seed:

| Seed | E2 first client hash | E2 == E3 | E3 == E4 |
|---|---|---|---|
| 42 | `d77512b3abf209070686463ddad30a7fbd799e840e2e03ccf16c4cadb429a23d` | ✔ (9/9) | ✔ (9/9) |
| 43 | `782cdc8e10cc3cac…` | ✔ (9/9) | ✔ (9/9) |
| 44 | `a5de0bb5a769cf5f…` | ✔ (9/9) | ✔ (9/9) |

Consequently the paired accuracy differences are not merely small — they are
exactly zero:

| Comparison | Δ val loss | Δ perplexity | Significant |
|---|---|---|---|
| E3 − E2 | **+0.00000 ± 0.00000** | +0.00000 | no |
| E4 − E2 | **+0.00000 ± 0.00000** | +0.00000 | no |
| E4 − E3 | **+0.00000 ± 0.00000** | +0.00000 | no |

The accuracy tables agreeing to six decimals is a *consequence* of byte-equal
artefacts, not a coincidence requiring statistical defence. This is the
strongest form the "audit layer is free" claim can take, and the paper should
lead with the hashes rather than with the loss table.

## Paired differences

Pairing matters: seed-to-seed variation is shared by both arms and cancels, so
the CI is over the effect rather than over the noisier absolute scores.

### vs the centralized upper bound (E1)

| Arm | Δ val loss | Δ perplexity | Significant |
|---|---|---|---|
| E0 Local-only | +0.0352 ± 0.0010 | +0.2615 ± 0.0074 | yes |
| E2 FedAvg | +0.0343 ± 0.0012 | +0.2551 ± 0.0089 | yes |
| E3 FL+chain | +0.0343 ± 0.0012 | +0.2551 ± 0.0089 | yes |
| E4 FedChain | +0.0343 ± 0.0012 | +0.2551 ± 0.0089 | yes |
| E5 non-IID | +0.0364 ± 0.0008 | +0.2711 ± 0.0057 | yes |

### Comparisons the shipped report does not compute

Derived here from the per-seed metrics files:

| Comparison | Δ val loss | Δ perplexity | Significant | Reading |
|---|---|---|---|---|
| **E2 − E0** (does FedAvg beat isolation?) | **−0.00085 ± 0.00022** | −0.00646 ± 0.00163 | yes | significant but **negligible** |
| E5 − E4 (cost of label skew) | +0.00212 ± 0.00070 | +0.01602 ± 0.00526 | yes | real, small |

**E2 − E0 is the problem.** Aggregating across three clients recovers
`0.00085 / 0.0352 = 2.4%` of the distance between isolated training and the
centralized upper bound. Statistically it clears zero; practically, at this
scale, FedAvg does approximately nothing.

## Training trajectory — the curve has not converged

Per-round global-model validation loss (mean over 3 seeds):

| Arm | round 1 | round 2 | round 3 (final) | Δ r1→r2 | Δ r2→r3 |
|---|---|---|---|---|---|
| E2 / E3 / E4 | 2.08343 | 2.03873 | 2.02277 | −0.04470 | −0.01597 |
| E5 non-IID | 2.08487 | 2.04123 | 2.02490 | −0.04363 | −0.01633 |

The gap to centralized at round 3 is **0.0343** — larger than the last observed
per-round improvement (0.0160) but of the same order. The run was stopped while
still descending. Nothing here supports the implicit claim that 3 rounds is
where federated training lands; 3 rounds is where the compute budget ran out.

**Naive extrapolation** (geometric decay, ratio `0.01597/0.04470 = 0.357`):
remaining improvement `≈ 0.0089`, so a 9-round E2 would land near **2.014** —
still ~0.025 above the *current* E1. This is a prediction, not a result, and it
is recorded in [05](05_ablation_design.md) as such. Note that under budget
matching E1 also improves as rounds increase, so the gap may not close at all.

E0 has **no per-round trajectory** — local-only evaluates once, at the end, as
`local_only_mean` over the 3 isolated clients. That missing curve is why
"does the FedAvg advantage grow with rounds?" cannot be answered from this data.

## Systems metrics (mean over 3 seeds)

| Metric | E0 | E1 | E2 | E3 | E4 | E5 |
|---|---|---|---|---|---|---|
| Total round time (s) | 3971.70 | 3939.17 | 4737.74 | 4749.57 | 4771.59 | 4789.77 |
| Training time (s) | 3958.28 | 3938.14 | 3962.74 | 3976.09 | 3976.63 | 3960.17 |
| Communication (MiB) | 299.20 ⚠ | 0.00 | 299.18 | 299.18 | 393.56 | 393.51 |
| Adapter size (MiB) | 16.62 | 16.62 | 16.62 | 16.62 | 16.62 | 16.62 |
| Chain tx latency (s) | 0 | 0 | 0 | 1.52 | 1.50 | 1.50 |
| Chain gas | 0 | 0 | 0 | 2,997,464 | 3,785,372 | 3,785,372 |
| IPFS upload (s) | 0 | 0 | 0 | 0 | 17.47 | 17.48 |
| IPFS download (s) | 0 | 0 | 0 | 0 | 3.02 | 3.03 |
| Aggregation (s) | 0 | 0 | 0.248 | 0.246 | 0.242 | 0.263 |

### Audit-layer overhead, computed against E2

| | E3 (+chain) | E4 (+chain +IPFS) |
|---|---|---|
| Wall clock | +0.25% | **+0.71%** |
| Chain latency as share of run | 0.032% | 0.031% |
| IPFS latency as share of run | — | 0.43% |
| Communication | +0.0% | **+31.5%** (299.2 → 393.6 MiB) |
| Gas | 2,997,464 (12 tx) | 3,785,372 (+26.3%) ≈ 0.00379 ETH |

The only non-trivial cost is the **+31.5% communication in E4**: the global
model is pushed to IPFS and pulled back for the verified round-trip. Everything
else is sub-1%. Per-transaction: ~0.124 s, ~315k gas.

> ⚠ **E0's 299.20 MiB is an instrumentation artefact, not a measurement.**
> Traced exactly: per round the reporter counts 49.87 MiB of client `upload_bytes`
> plus 49.87 MiB of `global_model.broadcast_bytes` (= 99.73 MiB × 3 rounds =
> 299.20). But `aggregation_enabled: false`, every client's `download_bytes` is
> **0**, and `global_adapter_path` points at `client_1`'s own directory — there
> is no server, no global model and no recipient. E0's true communication volume
> is **0 MiB**, the same as E1. Fix specified in [04](04_changes.md).

## Integrity and reliability

| Seed | Arm | Integrity checks | Chain tx | IPFS transfers | Resumed |
|---|---|---|---|---|---|
| 42/43/44 | E3 | — | 12/12 ok, 0 failed | — | no |
| 42/43/44 | E4 | **9/9 passed** | 12/12 ok, 0 failed | 0 failed (12 up, 9 down) | no |
| 42/43/44 | E5 | **9/9 passed** | 12/12 ok, 0 failed | 0 failed (12 up, 9 down) | no |

No failed transactions, no failed transfers, no resumed checkpoints anywhere in
the matrix. Every run is a single clean session.

*Non-contaminating note:* seed 42's `exp0_local.log` shows an
`Interrupted by the user` abort at 22:21 followed by a clean restart at 22:33
(`resumed: false`, `sessions: 1`, 3/3 rounds). The aborted attempt did not feed
the reported numbers.

## E6 — Integrity under attack

Seed 42, live chain, 12 adapters, 50 trials per attack (**250 total**):

| Attack | Type | Trials | Detected | Rate | FP rate | Verify (ms) | Gas |
|---|---|---|---|---|---|---|---|
| bitflip | malicious | 50 | 50 | **100%** | — | 13.13 | 247,205 |
| scale (LoRA-B ×10) | malicious | 50 | 50 | **100%** | — | 12.74 | 242,298 |
| substitute | malicious | 50 | 50 | **100%** | — | 13.08 | 242,358 |
| replay | malicious | 50 | 50 | **100%** | — | 13.11 | 242,310 |
| reserialize | **benign control** | 50 | 0 | 0% | **0.0%** | 14.34 | 242,370 |

**200/200 malicious artefacts rejected. 0/50 benign re-serialisations flagged.**
Every trial also recorded `correct: true` — the classifier was right on all 250.
Victims were spread across `client_1` (63), `client_2` (62), `client_3` (68) and
`global` (57).

The false-positive column is the more interesting one: it is the evidence that
canonical-JSON hashing of `adapter_config.json` makes the anchored commitment
reproducible across processes despite Python's per-process string-hash salting.

## E7 — Scalability

205/205 transactions succeeded. Avg latency 0.124 s, avg gas 291,392.

### Cost vs federation size

| Clients | Tx/round | Gas/round | Gas/client | Chain latency (s) |
|---|---|---|---|---|
| 1 | 2 | 616,560 | 616,560 | 0.253 |
| 3 | 4 | 1,163,296 | 387,765 | 0.503 |
| 5 | 6 | 1,744,914 | 348,983 | 0.726 |
| 10 | 11 | 3,198,971 | 319,897 | 1.349 |
| 25 | 26 | 7,561,286 | 302,451 | 3.202 |
| 50 | 51 | 14,831,811 | 296,636 | 6.267 |
| 100 | 101 | 29,372,873 | 293,729 | 12.573 |

Gas grows **47.6× for a 100× increase in clients** — linear in N with a fixed
per-round global-anchor term amortising away; per-client gas asymptotes to
~294k.

### Cost vs artefact size

| Payload | Adapter (MiB) | Gas | Hash (s) | Hash (MiB/s) | On-chain bytes |
|---|---|---|---|---|---|
| tiny | 0.223 | 311,439 | 0.0014 | 162.8 | 32 |
| small | 3.514 | 311,451 | 0.0031 | 1,129.9 | 32 |
| medium | 14.029 | 311,463 | 0.0104 | 1,350.5 | 32 |
| large | 49.050 | 311,451 | 0.0347 | 1,412.9 | 32 |

Adapter size grows **220× while gas changes 1.00×**. Only a 32-byte digest
reaches the chain, so anchoring cost is independent of model size. The sole
size-dependent term is client-side hashing, at ~1.4 GiB/s.

## Environment caveat

`nltk` was absent, so **every** run fell back from the HuggingFace `evaluate`
ROUGE implementation to the built-in one in `evaluation/eval_loss.py`:

```
WARNING | evaluation.eval_loss | The `evaluate` library failed to score
(... you need to install ['nltk'] ...); using the built-in implementation.
```

The fallback is consistent across all runs, so internal comparisons remain
valid, but the absolute ROUGE-L values are not from the standard implementation
and must not be compared against numbers from other papers.
