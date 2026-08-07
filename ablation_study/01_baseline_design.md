# 01 — Baseline experiment design

The first pass. Source of record: `results/smollm2-360m/`, three seeds (42, 43, 44).

## The claim under test

> Anchoring federated model updates on-chain and moving them over IPFS gives
> verifiable provenance at negligible accuracy cost and bounded systems cost.

That sentence decomposes into four independent claims, each needing its own arm:

| Claim | Evidence required | Arm |
|---|---|---|
| Federating is a reasonable thing to do here | a lower bound (isolation) and an upper bound (pooling) | E0, E1 |
| The audit layer costs no accuracy | FL with and without it, over several seeds | E2 vs E3, E4 |
| The audit layer *does something* | attacks it must catch, benign churn it must not flag | E6 |
| The design is deployable | cost as a function of federation size and model size | E7 |
| It survives the regime FL is deployed in | non-IID partition | E5 |

## Arms

| # | Config | Setup | Isolates |
|---|---|---|---|
| E0 | `configs/exp0_local.yaml` | 3 clients, **aggregation disabled** | What a participant gets alone. Lower bound. |
| E1 | `configs/exp1_sft.yaml` | Centralized on the **pooled union** of client windows | Upper bound: same data, no federation. |
| E2 | `configs/exp2_fl.yaml` | FedAvg, trusted aggregator | Accuracy cost of federating. |
| E3 | `configs/exp3_fl_bc.yaml` | E2 + on-chain SHA-256 anchoring | Cost of the audit trail alone. |
| E4 | `configs/exp4_fedchain.yaml` | E3 + IPFS transport, verified round-trip | Cost of decentralised storage. |
| E5 | `configs/exp5_noniid.yaml` | E4 on Dirichlet(0.3) label-skewed shards | Whether the result survives skew. |
| E6 | `scripts/tamper_experiment.py` | Adversarial, 250 trials | Detection rate and false-positive rate. |
| E7 | `scripts/scalability_experiment.py` | Systems sweep | Gas/latency vs client count and adapter size. |

E2, E3 and E4 are **designed to be identical** in accuracy. They differ only in
audit and transport, which never touch the learning math. Divergence would mean
the audit layer is corrupting updates. The numbers that must differ are gas,
transaction latency, IPFS latency and communication volume.

## Model and training

| | |
|---|---|
| Base model | `HuggingFaceTB/SmolLM2-360M-Instruct` |
| Quantization | 4-bit NF4, double quant, compute dtype auto (bf16) |
| Adapter | LoRA `r=16`, `alpha=32`, `dropout=0.05`, bias none |
| Target modules | `q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj` |
| Trainable params | 8,683,520 / 213,218,240 = **4.07%** |
| Sequence length | 512 |
| Batch | micro-batch 1 × grad-accum 8 = effective 8 |
| Optimiser | `paged_adamw_8bit`, lr 2.0e-4, cosine, warmup 0.03, clip 0.3 |
| Local epochs | 1 |
| Adapter artefact | 16.62 MiB |

`smollm2-360m` is the shakedown tier of the model ladder. The paper
configuration is `qwen-1.5b`; `qwen-0.5b` sits between them.

## Data and budget matching

Dolly-15k (15,011 records). `eval_500.jsonl` is the head of the raw dataset and
is excluded from every training shard, so there is no train/eval leakage.

**IID partition** (`data/manifest.json`): 3 shards × 4,837 records.
**Dirichlet(0.3) partition** (`data/dirichlet/manifest.json`): 4,798 / 2,715 / 6,998 records,
label-skewed over Dolly's 8 categories.

The arms are matched on **two** axes:

- **Updates** — federated: `3 rounds × 3 clients × 500 = 4500`.
  Centralized: `1 round × 3 shards × 1500 = 4500`.
- **Data** — federated round `r` reads window `[(r-1)·500, r·500)` of each shard,
  so 3 rounds cover `client_k[0:1500]` and the union over `k` is 4,500 unique
  records. E1 pools exactly that union.

The second axis is the one that is easy to get wrong. If every round replayed
the head of the shard, a 3-round federated run would be 3 epochs over 500
examples and would see a third of the unique data the centralized arm sees; the
measured "cost of federation" would then mostly be a data-budget artefact.
**Verified in the logs** — `exp4_fedchain.log`, seed 42:

```
Training client_1@r1 | shard=client1.jsonl | epochs=1.00 | window=[0, 500)
Training client_1@r2 | shard=client1.jsonl | epochs=1.00 | window=[500, 1000)
Training client_1@r3 | shard=client1.jsonl | epochs=1.00 | window=[1000, 1500)
```

and E1 pools correctly rather than slicing `centralized_full.jsonl`:

```
Pooled centralized corpus: 3 shard(s), 1500 samples each
Training centralized@r1 | shard=client1.jsonl, client2.jsonl, client3.jsonl | epochs=1.00 | window=[0, 1500)
```

## Evaluation protocol

| | |
|---|---|
| Loss / perplexity | 500 held-out samples, full-sequence NLL (matches the SFT objective) |
| ROUGE-L / BLEU | 50 greedy generations, 128 new tokens, 384-token prompt cap |
| Cadence | every round + final |
| E0's reported metric | `local_only_mean` — the **mean over the 3 isolated clients**, final round only |

## Infrastructure

| | |
|---|---|
| GPU | NVIDIA T600 Laptop, 4 GB, capability 7.5, bf16 supported |
| Host | WSL2, Linux 6.18, Python 3.12.3, 12 CPU |
| Stack | torch 2.5.1+cu121, transformers 5.14.1, peft 0.20.0, trl 1.9.2, bitsandbytes 0.50.0, web3 7.16.0 |
| Chain | anvil, chain id 31337, `http://127.0.0.1:8545`, contract `FedChainAudit` |
| IPFS | Kubo HTTP API, `http://127.0.0.1:5001` |
| Peak VRAM | 1.16–2.24 GB observed |

## Threat model (E6)

The aggregator is honest; transport is not. A client anchors `H(θ_k)` on-chain
and publishes `θ_k` to IPFS. An adversary controlling storage or the network
path substitutes a different artefact before retrieval. **The adversary cannot
rewrite the chain.**

Four attacks that must be caught — `bitflip`, `scale` (LoRA-B boosting, i.e.
model replacement), `substitute`, `replay` — and one benign control that must
**not** be flagged: `reserialize`, which rewrites `adapter_config.json` with its
keys and `target_modules` in a different order.

The control is load-bearing, not a formality. PEFT stores `target_modules` in a
`set` and Python salts string hashes per process, so two honest runs producing
bit-identical weights serialise that file differently. Hashing raw bytes would
make the anchored commitment unreproducible — an auditor retraining from the
same seed and data would compute a different digest and conclude tampering.
`utils/common.sha256_directory` therefore folds `adapter_config.json` in as
canonical JSON (sorted keys, sorted string lists). E6's false-positive column is
the evidence that this works.

Explicitly **out of scope**: a malicious client that anchors a poisoned adapter
it genuinely trained (the hash matches, so integrity checking is silent by
construction — that is Byzantine robustness, orthogonal to provenance), and
privacy (anchoring a digest leaks nothing, but LoRA updates are not DP).

## Reproducing

```bash
./infra.sh                                                       # anvil + IPFS + contract
./run_all.sh --model smol --seeds "42 43 44" --audit-experiments
```
