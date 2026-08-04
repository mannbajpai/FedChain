# FedChain - Model Ladder Comparison

Each row is one (model tier, experiment) pair. Tiers are separate runs with identical hyperparameters, so accuracy differences reflect model capacity and systems differences reflect adapter size.

## Tiers

| Tier         | Model                               | Experiments with results                     |
|--------------|-------------------------------------|----------------------------------------------|
| smollm2-360m | HuggingFaceTB/SmolLM2-360M-Instruct | exp1_sft, exp2_fl, exp3_fl_bc, exp4_fedchain |

## Metrics

| Tier         | Experiment          | Val Loss | PPL    | ROUGE-L | BLEU   | Train (s) | Comm (MB) | Adapter (MB) | Gas       | Round (s) |
|--------------|---------------------|----------|--------|---------|--------|-----------|-----------|--------------|-----------|-----------|
| smollm2-360m | E1: Centralized SFT | 1.9883   | 7.3031 | 0.2106  | 0.0397 | 4745.05   | 0.000     | 16.622       | 0         | 4746.22   |
| smollm2-360m | E2: FedAvg          | 2.0236   | 7.5656 | 0.2338  | 0.0448 | 4169.39   | 299.181   | 16.620       | 0         | 1666.57   |
| smollm2-360m | E3: FL + Blockchain | 2.0236   | 7.5656 | 0.2338  | 0.0448 | 4005.62   | 299.181   | 16.620       | 2,997,464 | 1612.55   |
| smollm2-360m | E4: FedChain        | 2.0236   | 7.5656 | 0.2338  | 0.0448 | 4119.08   | 424.969   | 16.620       | 3,785,372 | 1701.29   |
