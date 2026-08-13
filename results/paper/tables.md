# FedChain — paper tables

Generated from `results` over seeds [42, 43, 44].

Every figure here is computed from the stored metrics files. Paired
differences are per seed; intervals are 95% Student's *t*. Generation
metrics come only from `reeval250` (single scorer) — never from the
per-run values, which mix two scorers.

## Table 1 — Main result: does federating help?

| Model        | Partition       | Local-only (E0) | Centralized (E1) | FedAvg (E2)     | E0 − E2 (paired)  | Gap recovered |
|--------------|-----------------|-----------------|------------------|-----------------|-------------------|---------------|
| SmolLM2-360M | IID             | 2.0236 ± 0.0011 | 1.9884 ± 0.0006  | 2.0228 ± 0.0013 | 0.00085 ± 0.00022 | 2.4% ± 0.7%   |
| SmolLM2-360M | Dirichlet α=0.3 | --              | --               | --              | --                | *(not run)*   |
| Qwen2.5-0.5B | IID             | 2.0783 ± 0.0007 | 2.0499 ± 0.0016  | 2.0686 ± 0.0009 | 0.00964 ± 0.00093 | 34.0% ± 4.3%  |
| Qwen2.5-0.5B | Dirichlet α=0.3 | 2.0885 ± 0.0016 | 2.0492 ± 0.0042  | 2.0723 ± 0.0028 | 0.01626 ± 0.00128 | 41.5% ± 7.7%  |

> **Incomplete:** SmolLM2-360M / Dirichlet α=0.3

## Table 2 — The audit layer is an exact no-op

| Model        | Partition       | Comparison                  | Client adapters | Global models | Δ val. loss |
|--------------|-----------------|-----------------------------|-----------------|---------------|-------------|
| SmolLM2-360M | IID             | E2 vs E3 (chain)            | **27/27**       | **9/9**       | 0.000000    |
| SmolLM2-360M | IID             | E3 vs E4 (+IPFS)            | **27/27**       | **9/9**       | 0.000000    |
| SmolLM2-360M | Dirichlet α=0.3 | E2 vs E4-equiv (chain+IPFS) | *(not run)*     | *(not run)*   | --          |
| Qwen2.5-0.5B | IID             | E2 vs E3 (chain)            | **27/27**       | **9/9**       | 0.000000    |
| Qwen2.5-0.5B | IID             | E3 vs E4 (+IPFS)            | **27/27**       | **9/9**       | 0.000000    |
| Qwen2.5-0.5B | Dirichlet α=0.3 | E2 vs E4-equiv (chain+IPFS) | **27/27**       | **9/9**       | 0.000000    |

> **Incomplete:** SmolLM2-360M / Dirichlet α=0.3 / E2 vs E4-equiv (chain+IPFS)

## Table 3 — Systems cost of the audit layer

| Model        | Arm            | Comm. (MiB) | Δ comm. | Gas       | Total time (s) | Δ time |
|--------------|----------------|-------------|---------|-----------|----------------|--------|
| SmolLM2-360M | E2 FedAvg      | 299.18      | +0.0%   | 0         | 4737.7 ± 292.0 | +0.0%  |
| SmolLM2-360M | E3 +chain      | 299.18      | +0.0%   | 2,997,464 | 4749.6 ± 219.9 | +0.2%  |
| SmolLM2-360M | E4 +chain+IPFS | 393.56      | +31.5%  | 3,785,372 | 4771.6 ± 125.6 | +0.7%  |
| Qwen2.5-0.5B | E2 FedAvg      | 302.86      | +0.0%   | 0         | 4779.7 ± 44.7  | +0.0%  |
| Qwen2.5-0.5B | E3 +chain      | 302.86      | +0.0%   | 2,997,464 | 4770.8 ± 34.2  | -0.2%  |
| Qwen2.5-0.5B | E4 +chain+IPFS | 399.32      | +31.8%  | 3,785,372 | 4787.3 ± 36.9  | +0.2%  |

## Table 4 — Tamper detection

| Model        | Perturbation | Type           | Flagged | Rate | 95% bound    |
|--------------|--------------|----------------|---------|------|--------------|
| SmolLM2-360M | bitflip      | attack         | 50/50   | 100% | miss ≤ 5.8%  |
| SmolLM2-360M | scale        | attack         | 50/50   | 100% | miss ≤ 5.8%  |
| SmolLM2-360M | substitute   | attack         | 50/50   | 100% | miss ≤ 5.8%  |
| SmolLM2-360M | replay       | attack         | 50/50   | 100% | miss ≤ 5.8%  |
| SmolLM2-360M | reserialize  | benign control | 0/50    | 0%   | FPR ≤ 5.8%   |
| Qwen2.5-0.5B | bitflip      | attack         | 20/20   | 100% | miss ≤ 13.9% |
| Qwen2.5-0.5B | scale        | attack         | 20/20   | 100% | miss ≤ 13.9% |
| Qwen2.5-0.5B | substitute   | attack         | 20/20   | 100% | miss ≤ 13.9% |
| Qwen2.5-0.5B | replay       | attack         | 20/20   | 100% | miss ≤ 13.9% |
| Qwen2.5-0.5B | reserialize  | benign control | 0/20    | 0%   | FPR ≤ 13.9%  |

## Table 5 — Anchoring cost scaling

| Model        | Sweep         | Range                | Fit / spread                             | Reading                               |
|--------------|---------------|----------------------|------------------------------------------|---------------------------------------|
| SmolLM2-360M | clients       | N = 1–100            | gas = 299,069 + 290,702·N, R² = 0.999999 | linear in participants                |
| SmolLM2-360M | artefact size | 0.22–49.1 MiB (220×) | gas spread 0.0077%                       | flat; 32 bytes anchored at every size |
| Qwen2.5-0.5B | clients       | N = 1–50             | gas = 301,120 + 290,533·N, R² = 0.999994 | linear in participants                |
| Qwen2.5-0.5B | artefact size | 0.22–49.1 MiB (220×) | gas spread 0.0077%                       | flat; 32 bytes anchored at every size |

## Table 6 — Generation quality (supporting)

| Model        | Partition       | Arm            | ROUGE-L         | BLEU            |
|--------------|-----------------|----------------|-----------------|-----------------|
| SmolLM2-360M | IID             | E1 Centralized | 0.2477 ± 0.0022 | 0.0587 ± 0.0033 |
| SmolLM2-360M | IID             | E2 FedAvg      | 0.2417 ± 0.0135 | 0.0579 ± 0.0113 |
| SmolLM2-360M | IID             | E0 Local-only  | 0.2435 ± 0.0070 | 0.0594 ± 0.0021 |
| Qwen2.5-0.5B | IID             | E1 Centralized | 0.2737 ± 0.0132 | 0.0612 ± 0.0049 |
| Qwen2.5-0.5B | IID             | E2 FedAvg      | 0.2564 ± 0.0117 | 0.0517 ± 0.0057 |
| Qwen2.5-0.5B | IID             | E0 Local-only  | 0.2527 ± 0.0127 | 0.0501 ± 0.0057 |
| Qwen2.5-0.5B | Dirichlet α=0.3 | E1 Centralized | 0.2702 ± 0.0130 | 0.0583 ± 0.0074 |
| Qwen2.5-0.5B | Dirichlet α=0.3 | E2 FedAvg      | 0.2455 ± 0.0110 | 0.0488 ± 0.0097 |
| Qwen2.5-0.5B | Dirichlet α=0.3 | E0 Local-only  | 0.2463 ± 0.0107 | 0.0495 ± 0.0074 |

> **Incomplete:** reeval250 SmolLM2-360M / Dirichlet α=0.3 / E1 Centralized; reeval250 SmolLM2-360M / Dirichlet α=0.3 / E2 FedAvg; reeval250 SmolLM2-360M / Dirichlet α=0.3 / E0 Local-only

## Quotable figures

- **135/135 client-adapter** and **45/45 aggregated-model** pairwise comparisons are bit-identical between audited and un-audited federated training, over 3 (architecture, partition) settings.
  Equivalently, in distinct artefacts: every one of **36 aggregated models** and **108 client adapters** is identical across all audited variants. *(Quote one framing or the other, not both.)*
- **Qwen2.5-0.5B**: FedAvg recovers 34.0% of the isolation→centralized gap under IID and 41.5% under Dirichlet(0.3); the absolute gain grows 1.69× (disjoint intervals).
- **SmolLM2-360M**: 0/50 false positives on the benign control — FPR ≤ 5.8% (one-sided 95%).
- **Qwen2.5-0.5B**: 0/20 false positives on the benign control — FPR ≤ 13.9% (one-sided 95%).
- **SmolLM2-360M**: gas = 299,069 + 290,702·N, R² = 0.999999.
- **Qwen2.5-0.5B**: gas = 301,120 + 290,533·N, R² = 0.999994.
- Mean integrity-check latency: **13.3 ms** per artefact.
