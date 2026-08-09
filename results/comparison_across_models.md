# FedChain - Model Ladder Comparison

Each row is one (model tier, experiment) pair. Tiers are separate runs with identical hyperparameters, so accuracy differences reflect model capacity and systems differences reflect adapter size.

## Tiers

| Tier         | Model | Experiments with results      |
|--------------|-------|-------------------------------|
| qwen-0.5b    | -     | exp6_tamper, exp7_scalability |
| smollm2-360m | -     | exp6_tamper, exp7_scalability |

## Metrics

| Tier         | Experiment       | Val Loss | PPL | ROUGE-L | BLEU | Train (s) | Comm (MB) | Adapter (MB) | Gas | Total time (s) |
|--------------|------------------|----------|-----|---------|------|-----------|-----------|--------------|-----|----------------|
| qwen-0.5b    | exp6_tamper      | n/a      | n/a | n/a     | n/a  | n/a       | n/a       | n/a          | n/a | n/a            |
| qwen-0.5b    | exp7_scalability | n/a      | n/a | n/a     | n/a  | n/a       | n/a       | n/a          | n/a | n/a            |
| smollm2-360m | exp6_tamper      | n/a      | n/a | n/a     | n/a  | n/a       | n/a       | n/a          | n/a | n/a            |
| smollm2-360m | exp7_scalability | n/a      | n/a | n/a     | n/a  | n/a       | n/a       | n/a          | n/a | n/a            |
