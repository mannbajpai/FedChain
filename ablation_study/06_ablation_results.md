# 06 — Ablation results

> ## ⚠ NO RESULTS YET
>
> **Every cell in this document is `—`. No ablation run has been executed.**
> The tables exist so that the reporting format is fixed before the data arrives.
> Do not quote, cite, or copy anything from this file into the paper until the
> corresponding run row in the status table below says `done`.

## Run status

| Block | Runs | Seeds | Status | Date | Metrics path |
|---|---|---|---|---|---|
| A1 | E2 @ R=9, IID | 42,43,44 | not started | — | — |
| A2 | E0 @ R=9, IID | 42,43,44 | not started | — | — |
| A3 | E1 @ 7.5k / 13.5k samples | 42,43 | not started | — | — |
| B1 | E0/E1/E2 @ α=0.3 | 42,43,44 | not started | — | — |
| B2 | E0/E1/E2/E4 @ α=0.1 | 42,43,44 | not started | — | — |
| B3 | E0/E1/E2/E4 @ α=1.0 | 42,43,44 | not started | — | — |
| C | `local_epochs` ∈ {1,2,4} | 42,43,44 | not started | — | — |
| D1 | `log_global_model: false` | 42 | not started | — | — |
| D2 | `ipfs_roundtrip_aggregation: false` | 42 | not started | — | — |
| D3 | `verify_hash_on_download: false` | 42 | not started | — | — |
| E | Re-eval @ 250 gen samples | 42,43,44 | not started | — | — |
| F | `num_clients` ∈ {3,5,10} | 42,43,44 | not started | — | — |

**Prerequisite gate.** C1–C4 from [04](04_changes.md) must be applied and
verified before any row above is marked `done`, because they change what the
metrics files mean. Record the commit SHA of the applied changes here: `—`

---

## A — Round count

### A.1 Loss trajectory (mean over seeds)

| Round | E0 Local-only | E2 FedAvg | E2 − E0 | E1 (budget-matched) | E2 − E1 |
|---|---|---|---|---|---|
| 1 | — | — | — | — | — |
| 2 | — | — | — | — | — |
| 3 | — | *2.02277 (baseline)* | *−0.00085* | *1.98844* | *+0.03433* |
| 4 | — | — | — | — | — |
| 5 | — | — | — | — | — |
| 6 | — | — | — | — | — |
| 7 | — | — | — | — | — |
| 8 | — | — | — | — | — |
| 9 | — | — | — | — | — |

*Italicised row = measured baseline, carried forward from [02](02_baseline_results.md) for reference.*

### A.2 Paired differences at each round count

| R | E2 − E0 (± CI) | Significant | E2 − E1 (± CI) | Significant |
|---|---|---|---|---|
| 3 | *−0.00085 ± 0.00022* | *yes* | *+0.0343 ± 0.0012* | *yes* |
| 5 | — | — | — | — |
| 7 | — | — | — | — |
| 9 | — | — | — | — |

### A.3 Prediction check

| Hypothesis | Predicted | Observed | Held? |
|---|---|---|---|
| H-A1: `\|E2−E0\|` ≥ 0.005 at R=9 | ≥ 0.005 | — | — |
| H-A2: E2 @ R=9 ≈ 2.014 | 2.014 | — | — |
| H-A3: `E2−E1` stays flat ≈ 0.034 | flat | — | — |

---

## B — Data heterogeneity

### B.1 Accuracy by partition (R=3, mean ± 95% CI)

| α | E0 Local-only | E1 Centralized | E2 FedAvg | E4 FedChain | E2 − E0 | Significant |
|---|---|---|---|---|---|---|
| 0.1 | — | — | — | — | — | — |
| 0.3 | — | — | — | *2.0249 ± 0.0006* † | — | — |
| 1.0 | — | — | — | — | — | — |
| IID | *2.0236 ± 0.0011* | *1.9884 ± 0.0006* | *2.0228 ± 0.0013* | *2.0228 ± 0.0013* | *−0.00085 ± 0.00022* | *yes* |

† Baseline E5. Note it is the **E4-equivalent** arm (chain + IPFS), not E2, and
it currently has no matched non-IID E0/E1/E2 — which is what B1 supplies.

### B.2 Shard-size caps (regenerate and re-check before each run)

| α | client1 | client2 | client3 | Max rounds @ 500/round |
|---|---|---|---|---|
| 0.1 | — | — | — | — |
| 0.3 | *4,798* | *2,715* | *6,998* | *5* |
| 1.0 | — | — | — | — |
| IID | *4,837* | *4,837* | *4,837* | *9* |

### B.3 Prediction check

| Hypothesis | Predicted | Observed | Held? |
|---|---|---|---|
| H-B1: `\|E2−E0\|` ordered α=0.1 > 0.3 > 1.0 > IID | monotone | — | — |
| H-B2: E0 degrades faster than E2 as α falls | yes | — | — |
| H-B3: effect weaker than literature (label skew only) | yes | — | — |

---

## C — Local epochs

| Partition | `local_epochs` | E2 val loss | E2 − E0 | E2/E3/E4 hashes identical? |
|---|---|---|---|---|
| IID | 1 | *2.0228 ± 0.0013* | *−0.00085* | *yes (9/9)* |
| IID | 2 | — | — | — |
| IID | 4 | — | — | — |
| α=0.3 | 1 | — | — | — |
| α=0.3 | 2 | — | — | — |
| α=0.3 | 4 | — | — | — |

---

## D — Audit-layer cost decomposition

### D.1 Systems cost by variant (seed 42, `--skip-eval`)

| Variant | Tx | Gas | Comm (MiB) | Δ comm vs E2 | IPFS up (s) | IPFS down (s) | Wall clock (s) |
|---|---|---|---|---|---|---|---|
| E2 FedAvg (reference) | *0* | *0* | *299.18* | *—* | *0* | *0* | *4,737.7* |
| E4 FedChain (baseline) | *12* | *3,785,372* | *393.56* | *+31.5%* | *17.47* | *3.02* | *4,771.6* |
| D1 `log_global_model: false` | — | — | — | — | — | — | — |
| D2 `ipfs_roundtrip_aggregation: false` | — | — | — | — | — | — | — |
| D3 `verify_hash_on_download: false` | — | — | — | — | — | — | — |

### D.2 Attribution of the +31.5% communication overhead

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
| H-D1: D1 gas ≈ 2.84M (12→9 tx) | 2.84M | — | — |
| H-D2: D2 comm returns toward 299.2 MiB | ~299 | — | — |
| H-D3: D3 wall-clock saving immeasurable | ~0 | — | — |
| H-D4: all variants bit-identical to E4 | identical | — | — |

---

## E — Evaluation fidelity

| Arm | ROUGE-L @ 50 | ROUGE-L @ 250 | CI @ 50 | CI @ 250 | BLEU @ 250 | Backend |
|---|---|---|---|---|---|---|
| E0 | *0.2477* | — | *±0.0132* | — | — | — |
| E1 | *0.2267* | — | *±0.0174* | — | — | — |
| E2/E3/E4 | *0.2442* | — | *±0.0161* | — | — | — |
| E5 | *0.2454* | — | *±0.0419* | — | — | — |

*Baseline column used the built-in ROUGE backend (`nltk` absent). After
[C3](04_changes.md#c3--install-nltk-or-pin-the-metric-implementation-explicitly)
the @250 column should read `evaluate`; if it still reads `builtin`, C3 did not
take and the run is invalid.*

| Hypothesis | Predicted | Observed | Held? |
|---|---|---|---|
| H-E1: CIs narrow ~2.24× | 2.24× | — | — |
| H-E2: no ordering becomes significant | none | — | — |

---

## F — Federation size

| Clients | E0 val loss | E2 val loss | E2 − E0 | Gas | Comm (MiB) |
|---|---|---|---|---|---|
| 3 | *2.0236* | *2.0228* | *−0.00085* | *3,785,372* | *393.56* |
| 5 | — | — | — | — | — |
| 10 | — | — | — | — | — |

---

## Anomaly log

Record anything unexpected here as it happens — failed transactions, resumed
checkpoints, hash mismatches, OOM events, shard-window overruns. An empty log at
the end of the study is itself a claim, so do not leave it empty by neglect.

| Date | Run | Anomaly | Resolution |
|---|---|---|---|
| — | — | — | — |
