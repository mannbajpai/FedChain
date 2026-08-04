# FedChain

**Auditable Federated Fine-Tuning of Large Language Models using Blockchain and IPFS**

Reference implementation and benchmark harness for four decentralized LLM
fine-tuning paradigms, built to run end-to-end on consumer hardware
(NVIDIA T600, 4 GB VRAM).

| # | Experiment | Config | FL | Chain | IPFS | Isolates |
|---|-----------|--------|----|-------|------|----------|
| 1 | Centralized SFT | `configs/exp1_sft.yaml` | – | – | – | accuracy upper bound, zero overhead |
| 2 | Federated (FedAvg) | `configs/exp2_fl.yaml` | ✔ | – | – | cost of federation |
| 3 | FL + Blockchain | `configs/exp3_fl_bc.yaml` | ✔ | ✔ | – | cost of auditability |
| 4 | **FedChain** | `configs/exp4_fedchain.yaml` | ✔ | ✔ | ✔ | full decentralized system |

## Model & training setup

- `Qwen/Qwen2.5-1.5B-Instruct`, 4-bit NF4 (BitsAndBytes, double quantization)
- LoRA `r=16`, `alpha=32`, `dropout=0.05` on
  `q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj`
- Gradient checkpointing, `max_seq_length=512`, micro-batch 1, grad-accum 8
- PyTorch SDPA attention
- Automatic CUDA detection with graceful CPU fallback

### Model ladder

The 1.5B model sits at the 4 GB VRAM ceiling, so validate on smaller models
first. `--model` selects the tier; hyperparameters are identical across all
three, and each writes to its own `results/<key>/` and `outputs/<key>/`.

| `--model` | Model | Purpose |
|---|---|---|
| `smol` / `smollm2-360m` | `HuggingFaceTB/SmolLM2-360M-Instruct` | pipeline shakedown |
| `qwen-0.5b` | `Qwen/Qwen2.5-0.5B-Instruct` | preliminary results |
| `qwen-1.5b` | `Qwen/Qwen2.5-1.5B-Instruct` | the paper configuration |

```bash
./run_all.sh --model smol         # 1. prove the pipeline
./run_all.sh --model qwen-0.5b    # 2. check the numbers
./run_all.sh --model qwen-1.5b    # 3. the paper run
./run_all.sh --model all          # ...or all three, smallest first
```

`results/comparison_across_models.md` puts every tier in one table.
Any Hugging Face id works too (`--model Qwen/Qwen2.5-3B-Instruct`); run
`python utils/models.py --list` to see the ladder.

## Quick start

Full setup and run instructions live in **[GUIDE.md](GUIDE.md)**. The short version:

```bash
./infra.sh          # once per machine: Foundry (anvil) + Kubo (IPFS)
./run_all.sh        # everything else: venv, deps, nodes, data, all 4 experiments
```

`run_all.sh` is idempotent and ends by writing `results/comparison.md`.
Sanity-check first with `./run_all.sh --quick` (~15 min) or `./run_all.sh --dry-run` (~1 min).

**Crash-safe.** A 20–30 h sweep survives power cuts, OOM kills and reboots: just
re-run the same command. Finished experiments are skipped, finished rounds and
clients are replayed from `outputs/<exp>/checkpoint.json`, and an interrupted
client restarts from its last optimizer-step checkpoint rather than from step 0.
Checkpointed adapters are re-hashed before reuse, so a half-written file from a
power cut is retrained instead of silently averaged in. See
[GUIDE.md](GUIDE.md#resuming-after-a-crash-or-reboot).

Running a single experiment manually:

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt

python data/prepare_data.py                          # download + partition Dolly-15k
python main.py --config configs/exp4_fedchain.yaml
python scripts/compare_results.py                    # build the comparison table
```

For authenticated Hugging Face downloads (higher Hub rate limits and no
unauthenticated-request warning), set `HF_TOKEN` before preparing data or
starting training:

```bash
export HF_TOKEN="hf_your_read_token"
./run_all.sh
```

PowerShell:

```powershell
$env:HF_TOKEN = "hf_your_read_token"
python data/prepare_data.py
python main.py --config configs/exp4_fedchain.yaml
```

The token is read only from the process environment; it is never placed in the
YAML config, checkpoint fingerprint, logs, or metrics report.

Validate the whole pipeline without a GPU or a model download:

```bash
python main.py --config configs/exp4_fedchain.yaml --dry-run
```

`--dry-run` substitutes structurally valid synthetic LoRA adapters. Training and
accuracy numbers become placeholders, but aggregation, hashing, IPFS transfer and
on-chain anchoring all execute for real.

## Crash recovery

Checkpointing is enabled by default. If training is interrupted by a reboot,
power loss, OOM kill, or closed terminal, run the **same command again**:

```bash
./run_all.sh
# or: python main.py --config configs/exp4_fedchain.yaml
```

Recovery happens at four boundaries:

- Hugging Face optimizer/adapter state is saved every `save_steps` (25 by
  default), so an interrupted client resumes near its last optimizer step.
- A completed adapter has an atomic, hash-verified client manifest, covering the
  small crash window between training completion and orchestration bookkeeping.
- Each trained/published/anchored client and each completed round is committed
  to `outputs/<exp_name>/checkpoint.json`; the previous checkpoint generation
  is retained as `checkpoint.json.bak`.
- A finished experiment's metrics report causes `run_all.sh` to skip it.

The checkpoint fingerprint includes training/evaluation settings and hashes of
the configured data files. If either changes, stale state is archived and is
not mixed into the new run. Use `--no-resume` for an intentional clean run, or
`--force` to archive a completed run's checkpoint and execute it again.

## Outputs

Each run prints a Markdown summary and writes:

```
results/<exp_name>_metrics.json          full report (metrics, per-round detail, config, environment)
results/<exp_name>_audit_trail.json      on-chain receipts + contract state   (exp 3, 4)
results/<exp_name>_ipfs_transfers.json   every upload/download with latency   (exp 4)
results/<exp_name>.log                   run log
outputs/<exp_name>/round_N/...           client and global adapters per round
outputs/<exp_name>/checkpoint.json       atomic crash-recovery state
outputs/<exp_name>/checkpoint.json.bak   previous known-good state
```

Reported metrics: `validation_loss`, `perplexity`, `rouge_l`, `bleu`,
`training_time_sec`, `communication_volume_mb`, `adapter_size_mb`,
`blockchain_tx_latency_sec`, `blockchain_gas_used`, `ipfs_upload_latency_sec`,
`ipfs_download_latency_sec`, `aggregation_time_sec`,
`end_to_end_round_duration_sec`.

## Offline / mock fallbacks

Nothing blocks a run. Each external dependency degrades to a clearly-labelled
substitute, and the mode is recorded in the JSON report:

| Missing | Fallback |
|---|---|
| CUDA | full-precision CPU execution (4-bit skipped) |
| RPC node / contract | in-process `MockChain` with a documented gas model (`mode: "mock"`) |
| Pinata keys / IPFS daemon | local content-addressed store with real CIDv0 digests (`backend: "mock"`) |
| `evaluate` library | built-in LCS ROUGE-L and smoothed corpus BLEU-4 |
| `trl.SFTTrainer` | `transformers.Trainer` over a pre-tokenised dataset |

## Layout

```
configs/       base_config.yaml + four experiment overlays (`extends:` inheritance)
blockchain/    FedChainAudit.sol + Web3 logger with mock mode
ipfs/          Pinata / Kubo / mock storage backends
trainer/       sft.py (QLoRA), aggregation.py (FedAvg), federated.py (orchestrator)
evaluation/    eval_loss.py (loss, perplexity, ROUGE-L, BLEU)
utils/         config inheritance, device detection, hashing, logging
data/          prepare_data.py
scripts/       compare_results.py (cross-experiment table)
main.py        CLI entry point (one experiment)
run_all.sh     autonomous driver (all experiments + aggregation)
infra.sh       one-time host setup (Foundry + Kubo)
GUIDE.md       complete setup, run and troubleshooting guide
```

## Notes on methodology

- **No train/eval leakage.** `prepare_data.py` holds out the first 500 Dolly-15k
  records as the fixed evaluation split *before* shuffling, and excludes them
  from every client shard.
- **Budget-matched experiments.** Exp 1 sees 4500 samples in one pass; Exp 2–4
  see 3 rounds × 3 clients × 500 = 4500 sample-updates. Losses are comparable.
- **Token-weighted validation loss.** Every scored token contributes equally, so
  perplexity is not skewed by answer length differences between adapters.
- **FedAvg on LoRA factors.** Averaging `A` and `B` independently approximates
  averaging the effective updates `B·A`; this is standard in federated LoRA work
  and keeps communication at `O(r(d_in+d_out))`. Warm-starting each round from
  the previous global adapter keeps the approximation tight.
  `FedAvgAggregator.aggregate_delta_w` computes the exact variant for ablations.
- **Deterministic artefacts.** Adapter directories are packed reproducibly
  (sorted members, zeroed mtime/uid/gid) so identical weights always yield
  identical SHA-256 digests and identical CIDs.
