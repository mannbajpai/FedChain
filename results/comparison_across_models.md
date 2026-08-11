# FedChain - Model Ladder Comparison

Each row is one (model tier, experiment) pair. Tiers are separate runs with identical hyperparameters, so accuracy differences reflect model capacity and systems differences reflect adapter size.

## Tiers

| Tier         | Model                               | Seeds | Experiments with results                                                                             |
|--------------|-------------------------------------|-------|------------------------------------------------------------------------------------------------------|
| qwen-0.5b    | Qwen/Qwen2.5-0.5B-Instruct          | 3     | exp0_local, exp1_sft, exp2_fl, exp3_fl_bc, exp4_fedchain, exp5_noniid, exp6_tamper, exp7_scalability |
| smollm2-360m | HuggingFaceTB/SmolLM2-360M-Instruct | 3     | exp0_local, exp1_sft, exp2_fl, exp3_fl_bc, exp4_fedchain, exp5_noniid, exp6_tamper, exp7_scalability |

## Metrics

| Tier         | Experiment           | n | Val Loss         | PPL              | ROUGE-L          | BLEU             | Train (s)         | Comm (MB)        | Adapter (MB) | Gas       | Total time (s)    |
|--------------|----------------------|---|------------------|------------------|------------------|------------------|-------------------|------------------|--------------|-----------|-------------------|
| qwen-0.5b    | E0: Local-only       | 3 | 2.0783 +- 0.0007 | 7.9908 +- 0.0053 | 0.2455 +- 0.0149 | 0.0437 +- 0.0080 | 4067.99 +- 24.72  | 0.000            | 16.827       | 0         | 4084.76 +- 25.02  |
| qwen-0.5b    | E1: Centralized SFT  | 3 | 2.0499 +- 0.0016 | 7.7673 +- 0.0127 | 0.2466 +- 0.0230 | 0.0497 +- 0.0079 | 4015.72 +- 48.40  | 0.000            | 16.827       | 0         | 4017.08 +- 48.54  |
| qwen-0.5b    | E2: FedAvg           | 3 | 2.0686 +- 0.0009 | 7.9141 +- 0.0070 | 0.2407 +- 0.0111 | 0.0418 +- 0.0040 | 4068.84 +- 63.55  | 302.864          | 16.825       | 0         | 4779.73 +- 44.69  |
| qwen-0.5b    | E3: FL + Blockchain  | 3 | 2.0686 +- 0.0009 | 7.9141 +- 0.0070 | 0.2407 +- 0.0111 | 0.0418 +- 0.0040 | 4062.63 +- 72.38  | 302.864          | 16.825       | 2,997,464 | 4770.84 +- 34.25  |
| qwen-0.5b    | E4: FedChain         | 3 | 2.0686 +- 0.0009 | 7.9141 +- 0.0070 | 0.2407 +- 0.0111 | 0.0418 +- 0.0040 | 4058.95 +- 56.72  | 399.321 +- 0.013 | 16.825       | 3,785,372 | 4787.35 +- 36.88  |
| qwen-0.5b    | E5: FedChain non-IID | 3 | 2.0723 +- 0.0028 | 7.9429 +- 0.0222 | 0.2278 +- 0.0010 | 0.0451 +- 0.0053 | 3955.88 +- 63.77  | 399.302 +- 0.032 | 16.825       | 3,785,372 | 4683.00 +- 49.60  |
| smollm2-360m | E0: Local-only       | 3 | 2.0236 +- 0.0011 | 7.5657 +- 0.0081 | 0.2477 +- 0.0132 | 0.0431 +- 0.0076 | 3958.28 +- 385.12 | 299.202          | 16.622       | 0         | 3971.70 +- 385.39 |
| smollm2-360m | E1: Centralized SFT  | 3 | 1.9884 +- 0.0006 | 7.3041 +- 0.0042 | 0.2267 +- 0.0174 | 0.0430 +- 0.0055 | 3938.14 +- 196.43 | 0.000            | 16.622       | 0         | 3939.17 +- 196.36 |
| smollm2-360m | E2: FedAvg           | 3 | 2.0228 +- 0.0013 | 7.5592 +- 0.0097 | 0.2442 +- 0.0161 | 0.0431 +- 0.0113 | 3962.74 +- 246.07 | 299.182          | 16.620       | 0         | 4737.74 +- 291.99 |
| smollm2-360m | E3: FL + Blockchain  | 3 | 2.0228 +- 0.0013 | 7.5592 +- 0.0097 | 0.2442 +- 0.0161 | 0.0431 +- 0.0113 | 3976.09 +- 151.32 | 299.182          | 16.620       | 2,997,464 | 4749.57 +- 219.88 |
| smollm2-360m | E4: FedChain         | 3 | 2.0228 +- 0.0013 | 7.5592 +- 0.0097 | 0.2442 +- 0.0161 | 0.0431 +- 0.0113 | 3976.63 +- 99.98  | 393.559 +- 0.093 | 16.620       | 3,785,372 | 4771.59 +- 125.57 |
| smollm2-360m | E5: FedChain non-IID | 3 | 2.0249 +- 0.0006 | 7.5752 +- 0.0047 | 0.2454 +- 0.0419 | 0.0422 +- 0.0225 | 3960.17 +- 106.69 | 393.507 +- 0.042 | 16.620       | 3,785,372 | 4789.77 +- 208.85 |

_`n` is the number of seeds. Values are mean +- 95% CI (Student's t) where they vary across seeds, and a bare figure where they are deterministic given the config. Quote the interval, not the mean alone._

_Excluded from this table (kept on disk for provenance, not evidence): `_archive_prefix_20260804_222225`, `qwen-0.5b.leaky_backup`._

## Audit-layer experiments

_Not training runs — they carry no loss or perplexity, so they are reported here rather than as blank rows above._

### E6 — tamper detection

| Tier         | Attack      | Kind           | Detected | Rate   | Adapters |
|--------------|-------------|----------------|----------|--------|----------|
| qwen-0.5b    | bitflip     | attack         | 20/20    | 100.0% | 12       |
| qwen-0.5b    | scale       | attack         | 20/20    | 100.0% | 12       |
| qwen-0.5b    | substitute  | attack         | 20/20    | 100.0% | 12       |
| qwen-0.5b    | replay      | attack         | 20/20    | 100.0% | 12       |
| qwen-0.5b    | reserialize | benign control | 0/20     | 0.0%   | 12       |
| smollm2-360m | bitflip     | attack         | 50/50    | 100.0% | 12       |
| smollm2-360m | scale       | attack         | 50/50    | 100.0% | 12       |
| smollm2-360m | substitute  | attack         | 50/50    | 100.0% | 12       |
| smollm2-360m | replay      | attack         | 50/50    | 100.0% | 12       |
| smollm2-360m | reserialize | benign control | 0/50     | 0.0%   | 12       |

### E7 — gas versus federation size

| Tier         | Clients | Tx/round | Gas/round  | Gas/client |
|--------------|---------|----------|------------|------------|
| qwen-0.5b    | 1       | 2        | 616,560    | 616,560    |
| qwen-0.5b    | 3       | 4        | 1,163,296  | 387,765    |
| qwen-0.5b    | 5       | 6        | 1,744,914  | 348,983    |
| qwen-0.5b    | 10      | 11       | 3,198,971  | 319,897    |
| qwen-0.5b    | 25      | 26       | 7,561,286  | 302,451    |
| qwen-0.5b    | 50      | 51       | 14,831,811 | 296,636    |
| smollm2-360m | 1       | 2        | 616,560    | 616,560    |
| smollm2-360m | 3       | 4        | 1,163,296  | 387,765    |
| smollm2-360m | 5       | 6        | 1,744,914  | 348,983    |
| smollm2-360m | 10      | 11       | 3,198,971  | 319,897    |
| smollm2-360m | 25      | 26       | 7,561,286  | 302,451    |
| smollm2-360m | 50      | 51       | 14,831,811 | 296,636    |
| smollm2-360m | 100     | 101      | 29,372,873 | 293,729    |

### E7 — gas versus artefact size

| Tier         | Payload | Adapter (MiB) | Gas     | Anchored bytes |
|--------------|---------|---------------|---------|----------------|
| qwen-0.5b    | tiny    | 0.2228        | 311,439 | 32             |
| qwen-0.5b    | small   | 3.5145        | 311,451 | 32             |
| qwen-0.5b    | medium  | 14.0288       | 311,463 | 32             |
| qwen-0.5b    | large   | 49.0503       | 311,451 | 32             |
| smollm2-360m | tiny    | 0.2228        | 311,439 | 32             |
| smollm2-360m | small   | 3.5145        | 311,451 | 32             |
| smollm2-360m | medium  | 14.0288       | 311,463 | 32             |
| smollm2-360m | large   | 49.0503       | 311,451 | 32             |
