# FedChain - Experiment Comparison

**Model:** `HuggingFaceTB/SmolLM2-360M-Instruct`

## Run context

| Metric       | B1-E0: Local-only (non-IID) | B1-E1: Centralized (non-IID) | B1-E2: FedAvg (non-IID) |
|--------------|-----------------------------|------------------------------|-------------------------|
| Paradigm     | local_only                  | centralized                  | federated               |
| Rounds       | 3                           | 1                            | 3                       |
| Clients      | 3                           | 1                            | 3                       |
| Device       | cuda                        | cuda                         | cuda                    |
| Chain mode   | -                           | -                            | -                       |
| IPFS backend | -                           | -                            | -                       |

## Metrics

_Single representative seed. Accuracy with confidence intervals is in the next section - quote that, not this._

| Metric                    | B1-E0: Local-only (non-IID) | B1-E1: Centralized (non-IID) | B1-E2: FedAvg (non-IID) |
|---------------------------|-----------------------------|------------------------------|-------------------------|
| Validation Loss           | 2.0303                      | 1.9880                       | 2.0252                  |
| Perplexity                | 7.6167                      | 7.3010                       | 7.5774                  |
| ROUGE-L                   | 0.2266                      | 0.2364                       | 0.2306                  |
| BLEU                      | 0.0346                      | 0.0394                       | 0.0314                  |
| Training Time (s)         | 3973.33                     | 3769.62                      | 3905.22                 |
| Communication Volume (MB) | 0.000                       | 0.000                        | 299.182                 |
| Adapter Size (MB)         | 16.622                      | 16.622                       | 16.620                  |
| Blockchain Tx Latency (s) | 0.0000                      | 0.0000                       | 0.0000                  |
| Blockchain Gas Used       | 0                           | 0                            | 0                       |
| IPFS Upload Latency (s)   | 0.0000                      | 0.0000                       | 0.0000                  |
| IPFS Download Latency (s) | 0.0000                      | 0.0000                       | 0.0000                  |
| Aggregation Time (s)      | 0.0000                      | 0.0000                       | 0.2236                  |
| Mean Round Duration (s)   | 1329.32                     | 3770.85                      | 1596.66                 |
| Total Round Time (s)      | 3987.95                     | 3770.85                      | 4789.98                 |

_Communication and adapter sizes are MiB (2^20 bytes). 'Mean Round Duration' is per federated round, so it is not comparable across paradigms with different round counts - use 'Total Round Time'._

## Accuracy across seeds (mean +- 95% CI)

_Seeds per experiment: B1-E0: Local-only (non-IID)=3, B1-E1: Centralized (non-IID)=3, B1-E2: FedAvg (non-IID)=3._

| Metric          | B1-E0: Local-only (non-IID) | B1-E1: Centralized (non-IID) | B1-E2: FedAvg (non-IID) |
|-----------------|-----------------------------|------------------------------|-------------------------|
| Validation Loss | 2.0302 +- 0.0006            | 1.9885 +- 0.0012             | 2.0249 +- 0.0006        |
| Perplexity      | 7.6157 +- 0.0049            | 7.3046 +- 0.0089             | 7.5752 +- 0.0047        |
| ROUGE-L         | 0.2311 +- 0.0146            | 0.2400 +- 0.0401             | 0.2499 +- 0.0452        |
| BLEU            | 0.0383 +- 0.0101            | 0.0392 +- 0.0029             | 0.0407 +- 0.0227        |

## Paired difference vs `ablationB_e1_noniid` (per seed)

| Experiment                  | Metric          | Mean diff | 95% CI   | Seeds | Significant |
|-----------------------------|-----------------|-----------|----------|-------|-------------|
| B1-E0: Local-only (non-IID) | Validation Loss | +0.0417   | +-0.0013 | 3     | yes         |
| B1-E0: Local-only (non-IID) | Perplexity      | +0.3110   | +-0.0099 | 3     | yes         |
| B1-E2: FedAvg (non-IID)     | Validation Loss | +0.0364   | +-0.0017 | 3     | yes         |
| B1-E2: FedAvg (non-IID)     | Perplexity      | +0.2706   | +-0.0129 | 3     | yes         |

_'Significant' means the 95% CI of the paired difference excludes zero. A 'no' is a real result - it says the audit layer changed nothing measurable, which is the claim these experiments exist to support._

## Paired difference vs `ablationB_e0_noniid` (per seed)

| Experiment                   | Metric          | Mean diff | 95% CI   | Seeds | Significant |
|------------------------------|-----------------|-----------|----------|-------|-------------|
| B1-E1: Centralized (non-IID) | Validation Loss | -0.0417   | +-0.0013 | 3     | yes         |
| B1-E1: Centralized (non-IID) | Perplexity      | -0.3110   | +-0.0099 | 3     | yes         |
| B1-E2: FedAvg (non-IID)      | Validation Loss | -0.0053   | +-0.0006 | 3     | yes         |
| B1-E2: FedAvg (non-IID)      | Perplexity      | -0.0405   | +-0.0044 | 3     | yes         |

_'Significant' means the 95% CI of the paired difference excludes zero. A 'no' is a real result - it says the audit layer changed nothing measurable, which is the claim these experiments exist to support._

## Validation loss by round (mean over seeds)

| Round | B1-E2: FedAvg (non-IID) |
|-------|-------------------------|
| 1     | 2.0849                  |
| 2     | 2.0412                  |
| final | 2.0249                  |

_A curve still descending at the final round means the round count was budget-limited, not converged - any 'cost of federation' read off that end-point is a statement about the budget as much as about federation. Arms with no per-round rows evaluate only at the end; set `eval_local_clients_every_round: true` to give the local-only arm a curve._

## Overhead relative to `ablationB_e1_noniid`

| Experiment                  | d Val. Loss     | d Perplexity    | Comm (MiB) | Gas | Total time (s) |
|-----------------------------|-----------------|-----------------|------------|-----|----------------|
| B1-E0: Local-only (non-IID) | +0.0423 (+2.1%) | +0.3156 (+4.3%) | 0.000      | 0   | 3987.95        |
| B1-E2: FedAvg (non-IID)     | +0.0372 (+1.9%) | +0.2763 (+3.8%) | 299.182    | 0   | 4789.98        |
