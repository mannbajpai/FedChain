# FedChain - Experiment Comparison

**Model:** `Qwen/Qwen2.5-0.5B-Instruct`

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
| Validation Loss           | 2.0891                      | 2.0476                       | 2.0734                  |
| Perplexity                | 8.0779                      | 7.7493                       | 7.9516                  |
| ROUGE-L                   | 0.2484                      | 0.2659                       | 0.2340                  |
| BLEU                      | 0.0395                      | 0.0398                       | 0.0430                  |
| Training Time (s)         | 4666.74                     | 3925.53                      | 3978.37                 |
| Communication Volume (MB) | 0.000                       | 0.000                        | 302.865                 |
| Adapter Size (MB)         | 16.827                      | 16.827                       | 16.825                  |
| Blockchain Tx Latency (s) | 0.0000                      | 0.0000                       | 0.0000                  |
| Blockchain Gas Used       | 0                           | 0                            | 0                       |
| IPFS Upload Latency (s)   | 0.0000                      | 0.0000                       | 0.0000                  |
| IPFS Download Latency (s) | 0.0000                      | 0.0000                       | 0.0000                  |
| Aggregation Time (s)      | 0.0000                      | 0.0000                       | 0.2028                  |
| Mean Round Duration (s)   | 1560.68                     | 3926.69                      | 1564.55                 |
| Total Round Time (s)      | 4682.03                     | 3926.69                      | 4693.65                 |

_Communication and adapter sizes are MiB (2^20 bytes). 'Mean Round Duration' is per federated round, so it is not comparable across paradigms with different round counts - use 'Total Round Time'._

## Accuracy across seeds (mean +- 95% CI)

_Seeds per experiment: B1-E0: Local-only (non-IID)=3, B1-E1: Centralized (non-IID)=3, B1-E2: FedAvg (non-IID)=3._

| Metric          | B1-E0: Local-only (non-IID) | B1-E1: Centralized (non-IID) | B1-E2: FedAvg (non-IID) |
|-----------------|-----------------------------|------------------------------|-------------------------|
| Validation Loss | 2.0885 +- 0.0016            | 2.0492 +- 0.0042             | 2.0723 +- 0.0028        |
| Perplexity      | 8.0736 +- 0.0128            | 7.7619 +- 0.0326             | 7.9429 +- 0.0222        |
| ROUGE-L         | 0.2434 +- 0.0179            | 0.2630 +- 0.0082             | 0.2336 +- 0.0009        |
| BLEU            | 0.0412 +- 0.0041            | 0.0406 +- 0.0137             | 0.0449 +- 0.0042        |

## Paired difference vs `ablationB_e1_noniid` (per seed)

| Experiment                  | Metric          | Mean diff | 95% CI   | Seeds | Significant |
|-----------------------------|-----------------|-----------|----------|-------|-------------|
| B1-E0: Local-only (non-IID) | Validation Loss | +0.0393   | +-0.0048 | 3     | yes         |
| B1-E0: Local-only (non-IID) | Perplexity      | +0.3117   | +-0.0375 | 3     | yes         |
| B1-E2: FedAvg (non-IID)     | Validation Loss | +0.0231   | +-0.0059 | 3     | yes         |
| B1-E2: FedAvg (non-IID)     | Perplexity      | +0.1811   | +-0.0460 | 3     | yes         |

_'Significant' means the 95% CI of the paired difference excludes zero. A 'no' is a real result - it says the audit layer changed nothing measurable, which is the claim these experiments exist to support._

## Paired difference vs `ablationB_e0_noniid` (per seed)

| Experiment                   | Metric          | Mean diff | 95% CI   | Seeds | Significant |
|------------------------------|-----------------|-----------|----------|-------|-------------|
| B1-E1: Centralized (non-IID) | Validation Loss | -0.0393   | +-0.0048 | 3     | yes         |
| B1-E1: Centralized (non-IID) | Perplexity      | -0.3117   | +-0.0375 | 3     | yes         |
| B1-E2: FedAvg (non-IID)      | Validation Loss | -0.0163   | +-0.0013 | 3     | yes         |
| B1-E2: FedAvg (non-IID)      | Perplexity      | -0.1306   | +-0.0099 | 3     | yes         |

_'Significant' means the 95% CI of the paired difference excludes zero. A 'no' is a real result - it says the audit layer changed nothing measurable, which is the claim these experiments exist to support._

## Validation loss by round (mean over seeds)

| Round | B1-E2: FedAvg (non-IID) |
|-------|-------------------------|
| 1     | 2.1193                  |
| 2     | 2.0836                  |
| final | 2.0723                  |

_A curve still descending at the final round means the round count was budget-limited, not converged - any 'cost of federation' read off that end-point is a statement about the budget as much as about federation. Arms with no per-round rows evaluate only at the end; set `eval_local_clients_every_round: true` to give the local-only arm a curve._

## Overhead relative to `ablationB_e1_noniid`

| Experiment                  | d Val. Loss     | d Perplexity    | Comm (MiB) | Gas | Total time (s) |
|-----------------------------|-----------------|-----------------|------------|-----|----------------|
| B1-E0: Local-only (non-IID) | +0.0415 (+2.0%) | +0.3286 (+4.2%) | 0.000      | 0   | 4682.03        |
| B1-E2: FedAvg (non-IID)     | +0.0258 (+1.3%) | +0.2022 (+2.6%) | 302.865    | 0   | 4693.65        |
