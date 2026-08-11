# FedChain - Model Ladder Comparison

Each row is one (model tier, experiment) pair. Tiers are separate runs with identical hyperparameters, so accuracy differences reflect model capacity and systems differences reflect adapter size.

## Tiers

| Tier                   | Model                               | Experiments with results                                                                             |
|------------------------|-------------------------------------|------------------------------------------------------------------------------------------------------|
| qwen-0.5b              | Qwen/Qwen2.5-0.5B-Instruct          | exp0_local, exp1_sft, exp2_fl, exp3_fl_bc, exp4_fedchain, exp5_noniid, exp6_tamper, exp7_scalability |
| qwen-0.5b.leaky_backup | Qwen/Qwen2.5-0.5B-Instruct          | exp0_local, exp1_sft, exp2_fl, exp3_fl_bc, exp4_fedchain, exp6_tamper, exp7_scalability              |
| smollm2-360m           | HuggingFaceTB/SmolLM2-360M-Instruct | exp0_local, exp1_sft, exp2_fl, exp3_fl_bc, exp4_fedchain, exp5_noniid, exp6_tamper, exp7_scalability |

## Metrics

| Tier                   | Experiment           | Val Loss | PPL    | ROUGE-L | BLEU   | Train (s) | Comm (MB) | Adapter (MB) | Gas       | Total time (s) |
|------------------------|----------------------|----------|--------|---------|--------|-----------|-----------|--------------|-----------|----------------|
| qwen-0.5b              | E0: Local-only       | 2.0780   | 7.9883 | 0.2484  | 0.0402 | 4060.64   | 0.000     | 16.827       | 0         | 4077.55        |
| qwen-0.5b              | E1: Centralized SFT  | 2.0492   | 7.7614 | 0.2481  | 0.0468 | 3999.11   | 0.000     | 16.827       | 0         | 4000.45        |
| qwen-0.5b              | E2: FedAvg           | 2.0686   | 7.9136 | 0.2457  | 0.0430 | 4051.51   | 302.864   | 16.825       | 0         | 4771.08        |
| qwen-0.5b              | E3: FL + Blockchain  | 2.0686   | 7.9136 | 0.2457  | 0.0430 | 4033.07   | 302.864   | 16.825       | 2,997,464 | 4755.29        |
| qwen-0.5b              | E4: FedChain         | 2.0686   | 7.9136 | 0.2457  | 0.0430 | 4038.21   | 399.325   | 16.825       | 3,785,372 | 4777.59        |
| qwen-0.5b              | E5: FedChain non-IID | 2.0734   | 7.9516 | 0.2276  | 0.0427 | 3931.26   | 399.314   | 16.825       | 3,785,372 | 4661.56        |
| qwen-0.5b              | exp6_tamper          | n/a      | n/a    | n/a     | n/a    | n/a       | n/a       | n/a          | n/a       | n/a            |
| qwen-0.5b              | exp7_scalability     | n/a      | n/a    | n/a     | n/a    | n/a       | n/a       | n/a          | n/a       | n/a            |
| qwen-0.5b.leaky_backup | E0: Local-only       | 2.0780   | 7.9883 | 0.2484  | 0.0402 | 6188.29   | 0.000     | 16.827       | 0         | 6202.50        |
| qwen-0.5b.leaky_backup | E1: Centralized SFT  | 2.0492   | 7.7614 | 0.2481  | 0.0468 | 4052.48   | 0.000     | 16.827       | 0         | 4053.60        |
| qwen-0.5b.leaky_backup | E2: FedAvg           | 2.0686   | 7.9136 | 0.2457  | 0.0430 | 10581.86  | 302.864   | 16.825       | 0         | 11482.67       |
| qwen-0.5b.leaky_backup | E3: FL + Blockchain  | 2.0686   | 7.9136 | 0.2457  | 0.0430 | 10494.16  | 302.864   | 16.825       | 2,997,464 | 11370.19       |
| qwen-0.5b.leaky_backup | E4: FedChain         | 2.0686   | 7.9136 | 0.2457  | 0.0430 | 11019.64  | 399.325   | 16.825       | 3,785,372 | 11965.60       |
| qwen-0.5b.leaky_backup | exp6_tamper          | n/a      | n/a    | n/a     | n/a    | n/a       | n/a       | n/a          | n/a       | n/a            |
| qwen-0.5b.leaky_backup | exp7_scalability     | n/a      | n/a    | n/a     | n/a    | n/a       | n/a       | n/a          | n/a       | n/a            |
| smollm2-360m           | E0: Local-only       | 2.0241   | 7.5694 | 0.2446  | 0.0404 | 3805.71   | 299.202   | 16.622       | 0         | 3818.94        |
| smollm2-360m           | E1: Centralized SFT  | 1.9885   | 7.3048 | 0.2282  | 0.0417 | 3881.22   | 0.000     | 16.622       | 0         | 3882.27        |
| smollm2-360m           | E2: FedAvg           | 2.0234   | 7.5637 | 0.2470  | 0.0425 | 3881.72   | 299.182   | 16.620       | 0         | 4661.72        |
| smollm2-360m           | E3: FL + Blockchain  | 2.0234   | 7.5637 | 0.2470  | 0.0425 | 3938.45   | 299.182   | 16.620       | 2,997,464 | 4718.67        |
| smollm2-360m           | E4: FedChain         | 2.0234   | 7.5637 | 0.2470  | 0.0425 | 3931.27   | 393.517   | 16.620       | 3,785,372 | 4735.22        |
| smollm2-360m           | E5: FedChain non-IID | 2.0252   | 7.5774 | 0.2269  | 0.0329 | 3970.74   | 393.500   | 16.620       | 3,785,372 | 4792.06        |
| smollm2-360m           | exp6_tamper          | n/a      | n/a    | n/a     | n/a    | n/a       | n/a       | n/a          | n/a       | n/a            |
| smollm2-360m           | exp7_scalability     | n/a      | n/a    | n/a     | n/a    | n/a       | n/a       | n/a          | n/a       | n/a            |
