# FedChain - Experiment Comparison

**Model:** `HuggingFaceTB/SmolLM2-360M-Instruct`

## Run context

| Metric       | E0: Local-only | E1: Centralized SFT | E2: FedAvg | E3: FL + Blockchain | E4: FedChain | E5: FedChain non-IID |
|--------------|----------------|---------------------|------------|---------------------|--------------|----------------------|
| Paradigm     | local_only     | centralized         | federated  | federated           | federated    | federated            |
| Rounds       | 3              | 1                   | 3          | 3                   | 3            | 3                    |
| Clients      | 3              | 1                   | 3          | 3                   | 3            | 3                    |
| Device       | cuda           | cuda                | cuda       | cuda                | cuda         | cuda                 |
| Chain mode   | -              | -                   | -          | live                | live         | live                 |
| IPFS backend | -              | -                   | -          | -                   | local        | local                |

## Metrics

_Single representative seed. Accuracy with confidence intervals is in the next section - quote that, not this._

| Metric                    | E0: Local-only | E1: Centralized SFT | E2: FedAvg | E3: FL + Blockchain | E4: FedChain | E5: FedChain non-IID |
|---------------------------|----------------|---------------------|------------|---------------------|--------------|----------------------|
| Validation Loss           | 2.0241         | 1.9885              | 2.0234     | 2.0234              | 2.0234       | 2.0252               |
| Perplexity                | 7.5694         | 7.3048              | 7.5637     | 7.5637              | 7.5637       | 7.5774               |
| ROUGE-L                   | 0.2489         | 0.2282              | 0.2470     | 0.2470              | 0.2470       | 0.2269               |
| BLEU                      | 0.0378         | 0.0417              | 0.0425     | 0.0425              | 0.0425       | 0.0329               |
| Training Time (s)         | 3845.38        | 3881.22             | 3881.72    | 3938.45             | 3931.27      | 3970.74              |
| Communication Volume (MB) | 0.000          | 0.000               | 299.182    | 299.182             | 393.517      | 393.500              |
| Adapter Size (MB)         | 16.622         | 16.622              | 16.620     | 16.620              | 16.620       | 16.620               |
| Blockchain Tx Latency (s) | 0.0000         | 0.0000              | 0.0000     | 1.5056              | 1.4903       | 1.4843               |
| Blockchain Gas Used       | 0              | 0                   | 0          | 2,997,464           | 3,785,372    | 3,785,372            |
| IPFS Upload Latency (s)   | 0.0000         | 0.0000              | 0.0000     | 0.0000              | 17.8007      | 17.6427              |
| IPFS Download Latency (s) | 0.0000         | 0.0000              | 0.0000     | 0.0000              | 2.7466       | 2.9360               |
| Aggregation Time (s)      | 0.0000         | 0.0000              | 0.2505     | 0.2518              | 0.2018       | 0.3016               |
| Mean Round Duration (s)   | 1287.21        | 3882.27             | 1553.91    | 1572.89             | 1578.41      | 1597.35              |
| Total Round Time (s)      | 3861.63        | 3882.27             | 4661.72    | 4718.67             | 4735.22      | 4792.06              |

_Communication and adapter sizes are MiB (2^20 bytes). 'Mean Round Duration' is per federated round, so it is not comparable across paradigms with different round counts - use 'Total Round Time'._

## Accuracy across seeds (mean +- 95% CI)

_Seeds per experiment: E0: Local-only=3, E1: Centralized SFT=3, E2: FedAvg=3, E3: FL + Blockchain=3, E4: FedChain=3, E5: FedChain non-IID=3._

| Metric          | E0: Local-only   | E1: Centralized SFT | E2: FedAvg       | E3: FL + Blockchain | E4: FedChain     | E5: FedChain non-IID |
|-----------------|------------------|---------------------|------------------|---------------------|------------------|----------------------|
| Validation Loss | 2.0236 +- 0.0011 | 1.9884 +- 0.0006    | 2.0228 +- 0.0013 | 2.0228 +- 0.0013    | 2.0228 +- 0.0013 | 2.0249 +- 0.0006     |
| Perplexity      | 7.5657 +- 0.0081 | 7.3041 +- 0.0042    | 7.5592 +- 0.0097 | 7.5592 +- 0.0097    | 7.5592 +- 0.0097 | 7.5752 +- 0.0047     |
| ROUGE-L         | 0.2519 +- 0.0126 | 0.2267 +- 0.0174    | 0.2442 +- 0.0161 | 0.2442 +- 0.0161    | 0.2442 +- 0.0161 | 0.2454 +- 0.0419     |
| BLEU            | 0.0408 +- 0.0076 | 0.0430 +- 0.0055    | 0.0431 +- 0.0113 | 0.0431 +- 0.0113    | 0.0431 +- 0.0113 | 0.0422 +- 0.0225     |

## Paired difference vs `exp1_sft` (per seed)

| Experiment           | Metric          | Mean diff | 95% CI   | Seeds | Significant |
|----------------------|-----------------|-----------|----------|-------|-------------|
| E0: Local-only       | Validation Loss | +0.0352   | +-0.0010 | 3     | yes         |
| E0: Local-only       | Perplexity      | +0.2615   | +-0.0074 | 3     | yes         |
| E2: FedAvg           | Validation Loss | +0.0343   | +-0.0012 | 3     | yes         |
| E2: FedAvg           | Perplexity      | +0.2551   | +-0.0089 | 3     | yes         |
| E3: FL + Blockchain  | Validation Loss | +0.0343   | +-0.0012 | 3     | yes         |
| E3: FL + Blockchain  | Perplexity      | +0.2551   | +-0.0089 | 3     | yes         |
| E4: FedChain         | Validation Loss | +0.0343   | +-0.0012 | 3     | yes         |
| E4: FedChain         | Perplexity      | +0.2551   | +-0.0089 | 3     | yes         |
| E5: FedChain non-IID | Validation Loss | +0.0364   | +-0.0008 | 3     | yes         |
| E5: FedChain non-IID | Perplexity      | +0.2711   | +-0.0057 | 3     | yes         |

_'Significant' means the 95% CI of the paired difference excludes zero. A 'no' is a real result - it says the audit layer changed nothing measurable, which is the claim these experiments exist to support._

## Paired difference vs `exp0_local` (per seed)

| Experiment           | Metric          | Mean diff | 95% CI   | Seeds | Significant |
|----------------------|-----------------|-----------|----------|-------|-------------|
| E1: Centralized SFT  | Validation Loss | -0.0352   | +-0.0010 | 3     | yes         |
| E1: Centralized SFT  | Perplexity      | -0.2615   | +-0.0074 | 3     | yes         |
| E2: FedAvg           | Validation Loss | -0.0009   | +-0.0002 | 3     | yes         |
| E2: FedAvg           | Perplexity      | -0.0065   | +-0.0016 | 3     | yes         |
| E3: FL + Blockchain  | Validation Loss | -0.0009   | +-0.0002 | 3     | yes         |
| E3: FL + Blockchain  | Perplexity      | -0.0065   | +-0.0016 | 3     | yes         |
| E4: FedChain         | Validation Loss | -0.0009   | +-0.0002 | 3     | yes         |
| E4: FedChain         | Perplexity      | -0.0065   | +-0.0016 | 3     | yes         |
| E5: FedChain non-IID | Validation Loss | +0.0013   | +-0.0005 | 3     | yes         |
| E5: FedChain non-IID | Perplexity      | +0.0096   | +-0.0037 | 3     | yes         |

_'Significant' means the 95% CI of the paired difference excludes zero. A 'no' is a real result - it says the audit layer changed nothing measurable, which is the claim these experiments exist to support._

_Against the local-only arm a *negative* difference means aggregation helped. Read it next to the same arm's distance from the centralized bound: a difference that is significant but a small fraction of that distance says federation is measurable, not that it is worthwhile._

## Validation loss by round (mean over seeds)

| Round | E2: FedAvg | E3: FL + Blockchain | E4: FedChain | E5: FedChain non-IID |
|-------|------------|---------------------|--------------|----------------------|
| 1     | 2.0834     | 2.0834              | 2.0834       | 2.0849               |
| 2     | 2.0388     | 2.0388              | 2.0388       | 2.0412               |
| final | 2.0228     | 2.0228              | 2.0228       | 2.0249               |

_A curve still descending at the final round means the round count was budget-limited, not converged - any 'cost of federation' read off that end-point is a statement about the budget as much as about federation. Arms with no per-round rows evaluate only at the end; set `eval_local_clients_every_round: true` to give the local-only arm a curve._

## Overhead relative to `exp1_sft`

| Experiment           | d Val. Loss     | d Perplexity    | Comm (MiB) | Gas       | Total time (s) |
|----------------------|-----------------|-----------------|------------|-----------|----------------|
| E0: Local-only       | +0.0356 (+1.8%) | +0.2646 (+3.6%) | 0.000      | 0         | 3861.63        |
| E2: FedAvg           | +0.0348 (+1.8%) | +0.2589 (+3.5%) | 299.182    | 0         | 4661.72        |
| E3: FL + Blockchain  | +0.0348 (+1.8%) | +0.2589 (+3.5%) | 299.182    | 2,997,464 | 4718.67        |
| E4: FedChain         | +0.0348 (+1.8%) | +0.2589 (+3.5%) | 393.517    | 3,785,372 | 4735.22        |
| E5: FedChain non-IID | +0.0366 (+1.8%) | +0.2726 (+3.7%) | 393.500    | 3,785,372 | 4792.06        |
