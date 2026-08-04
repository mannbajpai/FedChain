# FedChain — Complete Setup & Run Guide

End-to-end instructions for reproducing every experiment in
*"FedChain: Auditable Federated Fine-Tuning of Large Language Models using
Blockchain and IPFS"* on a single Linux machine.

---

## Contents

1. [TL;DR — one command](#1-tldr--one-command)
2. [What you need](#2-what-you-need)
3. [What each script does](#3-what-each-script-does)
4. [Step-by-step manual setup](#4-step-by-step-manual-setup)
5. [Running the experiments](#5-running-the-experiments)
6. [Understanding the output](#6-understanding-the-output)
7. [Live vs. mock — the one check that matters](#7-live-vs-mock--the-one-check-that-matters)
8. [Time and hardware budget](#8-time-and-hardware-budget)
9. [Troubleshooting](#9-troubleshooting)
10. [Tuning the experiment](#10-tuning-the-experiment)
11. [Reproducibility notes for the paper](#11-reproducibility-notes-for-the-paper)

---

## 1. TL;DR — one command

You already ran `./infra.sh`. Everything else is one script:

```bash
cd fedchain
chmod +x run_all.sh
./run_all.sh
```

That single command creates the virtualenv, installs dependencies, starts
`anvil` and the IPFS daemon, compiles and stages the contract, downloads and
partitions Dolly-15k, runs a fast smoke test, executes all four experiments, and
writes `results/comparison.md`.

Before committing to the full run (many hours), do a 15-minute sanity pass:

```bash
./run_all.sh --quick        # 1 round, 20 samples per client, no ROUGE-L/BLEU
```

Or validate the plumbing in about a minute with no GPU and no model download:

```bash
./run_all.sh --dry-run
```

`run_all.sh` is **idempotent and non-destructive**. Re-running it reuses the
venv, reuses already-running nodes, and skips the dataset download if the shards
exist. Pass `--fresh` when you deliberately want to start over.

---

## 2. What you need

| Requirement | Notes |
|---|---|
| Linux (Ubuntu 22.04+ or WSL2) | `infra.sh` targets apt |
| Python 3.9+ (3.10/3.11 recommended) | `run_all.sh` verifies this |
| NVIDIA GPU, ≥4 GB VRAM | Tested on a T600. CPU works but is impractically slow |
| NVIDIA driver + CUDA 12.1 runtime | Change `FEDCHAIN_TORCH_INDEX` for other CUDA versions |
| ~25 GB free disk | Model weights, adapters, dataset, venv |
| Internet on first run | Model, dataset, pip packages, solc |

Optional, installed by `infra.sh`:

- **Foundry / anvil** — local EVM chain. Without it, experiments 3 and 4 use the
  mock chain.
- **Kubo / IPFS** — local IPFS node. Without it, experiment 4 uses the mock store.

Nothing is mandatory: every external dependency has a labelled fallback so a run
always completes. Whether it completed with *real* measurements is reported
explicitly — see [section 7](#7-live-vs-mock--the-one-check-that-matters).

---

## 3. What each script does

| Script | Purpose | Run it |
|---|---|---|
| `infra.sh` | Installs system packages, Foundry, Kubo. Generates `start_nodes.sh`. | Once per machine |
| `start_nodes.sh` | Starts `anvil` + `ipfs daemon` in the background. | Optional — `run_all.sh` does this |
| **`run_all.sh`** | **The autonomous driver: setup → infra → data → all 4 experiments → comparison table.** | Every time |
| `data/prepare_data.py` | Downloads Dolly-15k, holds out the eval split, partitions the rest. | Called by `run_all.sh` |
| `main.py` | Runs **one** experiment from one YAML config. | For individual runs |
| `scripts/compare_results.py` | Aggregates `results/*_metrics.json` into the comparison table. | Called by `run_all.sh` |

`infra.sh` deliberately covers only infrastructure — it installs no Python
packages and prepares no data. `run_all.sh` covers everything after it.

---

## 4. Step-by-step manual setup

Skip this if you used `run_all.sh`. Useful for debugging a specific stage.

### 4.1 Refresh your PATH

`infra.sh` installs Foundry into `~/.foundry/bin`, which is not on `PATH` in the
shell that ran it:

```bash
source ~/.bashrc
anvil --version && ipfs --version
```

If `anvil` is still not found:

```bash
echo 'export PATH="$HOME/.foundry/bin:$PATH"' >> ~/.bashrc && source ~/.bashrc
```

### 4.2 Start the nodes

```bash
./start_nodes.sh
```

`start_nodes.sh` backgrounds both daemons and does not report failures, so
verify them yourself:

```bash
curl -s -X POST -H 'Content-Type: application/json' \
  --data '{"jsonrpc":"2.0","method":"eth_chainId","params":[],"id":1}' \
  http://127.0.0.1:8545

curl -s -X POST http://127.0.0.1:5001/api/v0/version
```

Both must return JSON. If not, read `anvil_node.log` / `ipfs_node.log`.

### 4.3 Python environment

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt

python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

`run_all.sh` uses `.venv` by default; override with `FEDCHAIN_VENV=/path/to/env`.

The last line must print `True`. If it prints `False`, you installed a CPU-only
wheel — see [troubleshooting](#9-troubleshooting).

### 4.4 Dataset

```bash
python data/prepare_data.py
```

Produces, in `data/`:

| File | Contents |
|---|---|
| `eval_500.jsonl` | First 500 Dolly-15k records — the fixed evaluation split |
| `centralized_full.jsonl` | The remaining ~14.5k records, shuffled — Experiment 1 |
| `client1/2/3.jsonl` | Three disjoint I.I.D. partitions — Experiments 2–4 |
| `manifest.json` | Provenance record of the split |

The evaluation split is carved out **before** shuffling, so no evaluation record
appears in any client's training data.

### 4.5 Contract

Nothing to do in the normal case: `requirements.txt` includes `py-solc-x`, so
`blockchain/logger.py` compiles `contract.sol` and deploys it to anvil
automatically on first use. It needs internet once to fetch solc 0.8.20.

If that download is blocked, compile with the Foundry you already have:

```bash
forge build --contracts blockchain --out .forge-out
mkdir -p blockchain/artifacts
cp .forge-out/contract.sol/FedChainAudit.json blockchain/artifacts/
```

The logger reads Foundry's artifact format (`abi` + `bytecode.object`) directly.

To deploy once and reuse a fixed address across runs, put it in
`configs/exp3_fl_bc.yaml` and `configs/exp4_fedchain.yaml`:

```yaml
contract_address: "0x5FbDB2315678afecb367f032d93F642f64180aa3"
```

---

## 5. Running the experiments

### All four, autonomously

```bash
./run_all.sh
```

### A subset

```bash
./run_all.sh --experiments "3 4"
```

### One at a time

```bash
source .venv/bin/activate
python main.py --config configs/exp1_sft.yaml
python main.py --config configs/exp2_fl.yaml
python main.py --config configs/exp3_fl_bc.yaml
python main.py --config configs/exp4_fedchain.yaml
python scripts/compare_results.py
```

### The four paradigms

| # | Config | FL | Chain | IPFS | What it isolates |
|---|---|---|---|---|---|
| 1 | `exp1_sft.yaml` | – | – | – | Accuracy upper bound, zero overhead |
| 2 | `exp2_fl.yaml` | ✔ | – | – | Cost of federation |
| 3 | `exp3_fl_bc.yaml` | ✔ | ✔ | – | Cost of auditability |
| 4 | `exp4_fedchain.yaml` | ✔ | ✔ | ✔ | Full decentralized system |

### Useful `main.py` flags

```
--num-rounds N              Override the federated round count
--max-train-samples N       Samples per client per round
--no-generation-metrics     Skip ROUGE-L / BLEU (saves several minutes per eval)
--skip-eval                 Systems metrics only
--dry-run                   Synthetic adapters, no model download
--device cpu|cuda|auto      Force the compute device
--force-mock-chain          Ignore a live RPC node
--force-mock-ipfs           Ignore a live IPFS daemon
--log-level DEBUG           Verbose logging
```

`run_all.sh` forwards anything after `--` straight to `main.py`:

```bash
./run_all.sh -- --no-generation-metrics --log-level DEBUG
```

### Running unattended

```bash
nohup ./run_all.sh > run.out 2>&1 &
tail -f results/logs/run_all.log
```

Every run mirrors its full transcript to `results/logs/run_all.log`.

---

## 6. Understanding the output

### Files

```
results/comparison.md               Cross-experiment table  <- the paper table
results/comparison.csv              Same numbers, for plotting
results/<exp>_metrics.json          Full report: metrics, per-round detail, config, environment
results/<exp>_audit_trail.json      On-chain receipts + contract state       (exp 3, 4)
results/<exp>_ipfs_transfers.json   Every upload/download with latency       (exp 4)
results/<exp>.log                   Per-experiment log
results/logs/run_all.log            Full driver transcript
results/logs/anvil.log              Chain node output
results/logs/ipfs.log               IPFS daemon output
outputs/<exp>/round_N/client_K/     Per-client LoRA adapter for that round
outputs/<exp>/round_N/global/       Aggregated global adapter for that round
```

### The 13 reported metrics

| Metric | Meaning |
|---|---|
| `validation_loss` | Token-weighted NLL on the 500-sample held-out split |
| `perplexity` | `exp(validation_loss)` |
| `rouge_l` | LCS F1 vs. reference responses, 50 greedy generations (0–1) |
| `bleu` | Smoothed corpus BLEU-4 over the same 50 (0–1) |
| `training_time_sec` | Wall-clock spent optimizing weights, all clients, all rounds |
| `communication_volume_mb` | Total MB crossing a participant boundary |
| `adapter_size_mb` | On-disk size of the final LoRA artefact |
| `blockchain_tx_latency_sec` | Cumulative submit-to-confirmation latency |
| `blockchain_gas_used` | Cumulative gas across all anchoring transactions |
| `ipfs_upload_latency_sec` | Cumulative pin latency |
| `ipfs_download_latency_sec` | Cumulative retrieval latency |
| `aggregation_time_sec` | Time inside FedAvg |
| `end_to_end_round_duration_sec` | Mean wall-clock per federated round |

### How communication volume is counted

Only bytes that actually cross a participant boundary count, so the four
experiments stay comparable:

- **Exp 1:** `0` — nothing leaves the single trainer.
- **Exp 2 & 3:** `N` client uploads + `N` global broadcasts per round. The audit
  trail adds no payload, so exp 3 should match exp 2 almost exactly — that
  equality is the point of the ablation.
- **Exp 4:** the byte counts `IPFSManager` measured for every real transfer.
  Adapters are tar-gzipped in flight, and the aggregator fetches each one back
  by CID, so the number is higher and genuinely measured.

### Reading the comparison table

`comparison.md` has four sections: run context (rounds, clients, device, chain
and IPFS modes), the 13 metrics side by side, overhead deltas versus the
centralized baseline, and a **Warnings** section. Read the warnings first — that
is where dry runs, mock fallbacks, failed transactions and failed integrity
checks are surfaced.

---

## 7. Live vs. mock — the one check that matters

Every external dependency degrades gracefully. That keeps runs from crashing,
but it also means a run can *succeed* while producing synthetic systems numbers.
**Check this before reporting anything.**

```bash
grep -E '"mode"|"backend"' results/exp4_fedchain_metrics.json
```

You want:

```json
"mode": "live",       // blockchain: real anvil transactions
"backend": "local",   // IPFS: real Kubo daemon
```

`run_all.sh` prints this for you in its final summary, flagging any mock run
explicitly. `comparison.md` repeats it in the Warnings section.

| Fallback | Trigger | Consequence |
|---|---|---|
| CPU instead of CUDA | No GPU / driver | Correct results, ~50× slower |
| `mode: "mock"` | No RPC node, or no contract could be deployed | Gas and latency are modelled, not measured |
| `backend: "mock"` | No Pinata keys and no IPFS daemon | Transfer latency reflects local disk, not a network |
| Built-in ROUGE/BLEU | `evaluate` not installed | Slightly different numbers than the HF implementation |
| `transformers.Trainer` | `trl.SFTTrainer` unavailable | Same objective, different code path |

Mock mode is legitimate for testing and for machines without a chain — it is
just not a measurement.

### Using real Pinata instead of a local node

```bash
export PINATA_API_KEY="your-key"
export PINATA_SECRET_KEY="your-secret"
# or: export PINATA_JWT="your-jwt"
python main.py --config configs/exp4_fedchain.yaml
```

Credentials are read from the config first, then from the environment. Pinata
takes priority over a local daemon when both are available.

---

## 8. Time and hardware budget

On a T600 (4 GB) with the shipped defaults — 3 rounds × 3 clients × 500 samples:

| Stage | Rough cost |
|---|---|
| First model download | ~3 GB, one time |
| One client, 500 samples, 1 epoch | 25–45 min |
| One FL round (3 clients) | 1.5–2.5 h |
| Exp 2/3/4 (3 rounds) | 5–8 h each |
| Exp 1 (4500 samples, 1 pass) | 4–7 h |
| Evaluation (500 loss + 50 generations) | 10–25 min per call |
| **All four experiments** | **roughly 20–30 h** |

Measure your own machine before committing:

```bash
./run_all.sh --quick --experiments 2      # then extrapolate from the log
```

Ways to shorten a run:

```bash
./run_all.sh -- --no-generation-metrics                    # drop ROUGE-L/BLEU
./run_all.sh -- --max-train-samples 200                    # smaller shards
./run_all.sh -- --num-rounds 2                             # fewer rounds
```

Keep whatever you choose **identical across all four experiments**, or the
accuracy comparison becomes meaningless.

VRAM behaviour: the base model is released between clients and between
evaluations (`gc.collect()` + `torch.cuda.empty_cache()`), so peak usage is
bounded by one model + one adapter, not by the number of clients. Peak VRAM per
run is recorded as `peak_vram_mb` in the JSON report.

---

## 9. Troubleshooting

**`anvil: command not found`**
`~/.foundry/bin` is not on `PATH`. `run_all.sh` adds it automatically; for a
manual shell, `source ~/.bashrc` or add the export from §4.1.

**`torch.cuda.is_available()` is `False`**
You have a CPU-only wheel. Check `nvidia-smi` works, then reinstall with the
index matching your CUDA runtime:
```bash
pip uninstall -y torch
pip install torch --index-url https://download.pytorch.org/whl/cu121   # or cu118
```
For a different CUDA version: `FEDCHAIN_TORCH_INDEX=https://download.pytorch.org/whl/cu118 ./run_all.sh`

**`CUDA out of memory`**
Defaults already sit near the 4 GB floor. Next levers, in order:
`--max-train-samples 200`, then `max_seq_length: 256` in `configs/base_config.yaml`,
then `gen_max_new_tokens: 64`. Confirm nothing else is using the GPU (`nvidia-smi`).

**`bitsandbytes` import or CUDA-setup error**
`pip install -U bitsandbytes`. If it still fails, run with `--device cpu` to
confirm the rest of the pipeline works; 4-bit quantisation is skipped
automatically on CPU.

**Blockchain stuck in mock mode despite anvil running**
Check the reason recorded in the report:
```bash
python -c "import json;print(json.load(open('results/exp3_fl_bc_metrics.json'))['blockchain']['connection_note'])"
```
Usually either the RPC is unreachable, or no contract could be deployed — in
which case stage the artifact manually (§4.5).

**IPFS stuck in mock mode**
`curl -s -X POST http://127.0.0.1:5001/api/v0/version`. If that fails, check
`results/logs/ipfs.log`; a stale lock is cleared with
`rm ~/.ipfs/repo.lock` before restarting the daemon.

**`Missing client shard(s)`**
Run `python data/prepare_data.py`.

**Dataset download fails / gated**
`huggingface-cli login`, then retry. Dolly-15k is public, so this is usually a
network or proxy issue.

**A single experiment failed mid-run**
`run_all.sh` continues to the next one and reports failures in its summary.
Re-run just the failed one: `python main.py --config configs/exp3_fl_bc.yaml`,
then `python scripts/compare_results.py`.

**Ports 8545 / 5001 already in use**
Either reuse what is running (the script does this automatically), or
`pkill -f '^anvil'; pkill -f 'ipfs daemon'` and start fresh.

---

## 10. Tuning the experiment

All knobs live in `configs/`. `base_config.yaml` holds shared defaults; each
`expN_*.yaml` inherits it via `extends:` and overrides only what differs.

Frequently changed:

```yaml
num_rounds: 3               # federated rounds
local_epochs: 1             # epochs per client per round
max_train_samples: 500      # samples per client per round
learning_rate: 2.0e-4       # keep the 2.0e-4 form; PyYAML needs the dot and the sign
lora_r: 16                  # LoRA rank
lora_alpha: 32
max_seq_length: 512         # first thing to cut if VRAM is tight
enable_generation_metrics: true
gen_num_samples: 50         # prompts decoded for ROUGE-L / BLEU
```

Changing the client count means changing three things together — `num_clients`,
`client_files`, and the partition:

```bash
python data/prepare_data.py --num-clients 5
```
```yaml
num_clients: 5
client_files:
  - "data/client1.jsonl"
  # ... through client5.jsonl
```

The config loader validates that `num_clients` matches `len(client_files)` and
fails immediately if they disagree.

Adding a fifth experiment is a new YAML file plus one line in
`run_all.sh`'s `CONFIG_OF` map.

---

## 11. Reproducibility notes for the paper

- **No train/eval leakage.** The evaluation split is the first 500 Dolly-15k
  records, held out before shuffling and excluded from every client shard.
- **Budget-matched experiments.** Exp 1 sees 4500 samples in one pass; exps 2–4
  see 3 rounds × 3 clients × 500 = 4500 sample-updates. Losses are comparable.
- **Fixed seed.** `seed: 42` seeds Python, NumPy, PyTorch and the dataset
  shuffle. Set `deterministic: true` for cuDNN determinism at some speed cost.
- **Token-weighted validation loss.** Every scored token contributes equally, so
  perplexity is not skewed by answer-length differences between adapters.
- **FedAvg on LoRA factors.** `A` and `B` are averaged independently, which
  approximates averaging the effective updates `B·A`. This is standard in
  federated LoRA work and keeps communication at `O(r(d_in + d_out))`.
  Warm-starting each round from the previous global adapter keeps the
  approximation tight. `FedAvgAggregator.aggregate_delta_w` computes the exact
  variant if you want to quantify the gap for an ablation.
- **Deterministic artefacts.** Adapter directories are packed reproducibly
  (sorted members, zeroed mtime/uid/gid), so identical weights always produce
  identical SHA-256 digests and identical CIDs.
- **Verified provenance.** In experiment 4 every client adapter is fetched back
  from IPFS and re-hashed against its on-chain commitment before entering the
  average; a mismatch excludes that contribution. The pass/fail count is in
  `run_summary.integrity_checks_passed` and surfaced in `comparison.md`.
- **Environment capture.** Every report embeds Python, OS, GPU and library
  versions under `environment`, so a run is self-describing.

### Reporting checklist

- [ ] `comparison.md` Warnings section is empty (no dry runs, no mock modes)
- [ ] `"mode": "live"` and `"backend": "local"` in exps 3 and 4
- [ ] `integrity_checks_passed == integrity_checks_total` in exp 4
- [ ] All four experiments used the same `max_train_samples` and `num_rounds`
- [ ] `environment.device.gpu_name` matches the hardware you claim
