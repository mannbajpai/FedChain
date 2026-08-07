# 09 — Run guide

Everything in [04](04_changes.md) is applied and tested. What remains is GPU
time. This document is the execution order.

## Where these must run

**Not on this checkout.** The machine holding this repo has:

| Requirement | State here |
|---|---|
| CUDA device | **absent** — `torch 2.13.0+cpu`, `cuda.is_available() == False` |
| `trl` (SFT trainer) | **not installed** |
| `bitsandbytes` (4-bit NF4) | **not installed** — CUDA-only anyway |
| `web3` (chain anchoring) | **not installed** |
| anvil / IPFS daemon | **not on PATH** |

The baseline was produced at `/home/mann/FedChain` on WSL2 with an NVIDIA T600
(4 GB). Run everything below there, or on equivalent hardware.

What *was* verified here: config resolution, the C1/C2 accounting and cadence
logic under `--dry-run`, metric-backend enforcement, sweep discovery, report
generation, and the 33-test suite. The training loop itself needs the GPU box.

## Step 0 — Sync and sanity-check

```bash
git pull                                    # or copy the changed files across
pip install -r requirements.txt             # picks up nltk, which was missing
python -m pytest tests/ -q                  # expect 32 passed, 1 skipped
./infra.sh                                  # anvil + IPFS daemon + contract
python data/prepare_data.py                 # if data/*.jsonl are absent
```

Confirm the metric backend is now reachable — this is the check that would have
caught the silent fallback in the baseline:

```bash
python -c "import evaluate, nltk, rouge_score; print('metric stack OK')"
```

## Step 1 — Qwen2.5-0.5B main matrix *(the second model tier)*

This is the run that gives the paper two models.

```bash
./run_all.sh --model qwen-0.5b --seeds "42 43 44" --audit-experiments
```

**Cost.** The 360M matrix took ~24 GPU-hours. Qwen2.5-0.5B has ~1.4× the
parameters and a larger vocabulary, so budget **~35–40 GPU-hours** — call it two
days on the T600. `run_all.sh` checkpoints per client per round, so an
interruption resumes rather than restarts.

**Watch for.** The 4 GB card was at 2.24 GB peak on the 360M model. 0.5B should
fit, but if it OOMs the lever is `max_seq_length` (512 → 384) — *not*
`batch_size`, which is already 1, and not `grad_accum_steps`, which would change
the effective batch and break comparability with the 360M tier.

**Verify before trusting it** — the two checks the baseline could not pass:

```bash
# 1. Local-only must now report 0 MiB, not 299
python -c "
import json; d=json.load(open('results/qwen-0.5b/seed_42/exp0_local_metrics.json'))
print('comm:', d['metrics']['communication_volume_mb'], '| counted:', d['run_summary'].get('communication_counted'))"

# 2. The audit layer must still be a no-op: E2/E3/E4 bit-identical
python -c "
import json
h=lambda e:[c['model_hash'] for r in json.load(open(f'results/qwen-0.5b/seed_42/{e}_metrics.json'))['rounds'] for c in r['clients']]
a,b,c=h('exp2_fl'),h('exp3_fl_bc'),h('exp4_fedchain')
print('E2==E3:',a==b,'| E3==E4:',b==c,'| n:',len(a))"
```

If the second check fails, **stop** — the audit layer is perturbing training and
nothing else in the paper matters until that is understood.

## Step 2 — Ablation B1 *(non-IID baselines — cheapest high-value block)*

```bash
python data/prepare_data.py --partition dirichlet --alpha 0.3   # if absent
./ablation_study/run_ablation.sh --block B1 --model qwen-0.5b --seeds "42 43 44"
```

~12 h. Supplies the E0/E1/E2 arms on the Dirichlet shards that E5 currently
lacks, so the non-IID result stops being compared against IID baselines.

## Step 3 — Ablation A *(round sweep — the motivation fix)*

```bash
./ablation_study/run_ablation.sh --block A --model qwen-0.5b --seeds "42 43 44"
```

~37 h. The one that decides whether `E2 − E0` grows with rounds. Note the runner
warns if C2 is missing; it should not, now that the change is in.

## Step 4 — Ablation D *(audit decomposition — 3 h, one seed)*

```bash
./ablation_study/run_ablation.sh --block D --model qwen-0.5b --seeds "42"
```

Then confirm H-D4 — every variant must still produce E4-identical adapters.

## Step 5 — Ablation E *(generation metrics, no retraining)*

```bash
python scripts/reevaluate.py --sweep results/qwen-0.5b \
    --config configs/exp4_fedchain.yaml --model qwen-0.5b \
    --gen-num-samples 250 --require-backend evaluate \
    --out results/qwen-0.5b/reeval_250.json
```

~6 h, no training. `--require-backend evaluate` makes a missing dependency fail
immediately rather than silently reverting to the built-in implementation.

## Step 6 — Reports

```bash
python scripts/compare_results.py --results-dir results/qwen-0.5b --seeds
python scripts/compare_results.py --across-models --results-dir results
```

`comparison.md` now carries the paired difference against **both** `exp1_sft` and
`exp0_local`, plus the per-round trajectory table.
`comparison_across_models.md` is the two-model table for the paper.

## Optional — regenerate the 360M local-only arm

The three shipped `exp0_local` reports still carry the phantom 299.202 MiB; the
value was never measured, so it cannot be recomputed from the reports. If the
360M tier appears in the paper with a communication column, re-run just that arm:

```bash
./run_all.sh --model smol --experiments "0" --seeds "42 43 44" --force
```

~3.3 h. Afterwards `LocalOnlyCommunicationTests` enforces the invariant instead
of skipping. If the 360M tier is presented as the shakedown and only qwen-0.5b
carries the cost table, this is optional — but then say so explicitly rather than
printing a number known to be wrong.

## Recording results

Fill [06_ablation_results.md](06_ablation_results.md) as runs land — the run
status table first, then the prediction-check tables. Log anomalies in that
file's anomaly table as they happen. Only then write
[07_ablation_conclusions.md](07_ablation_conclusions.md), by working through the
branches already fixed there.

## Total

| Step | Cost | Cumulative |
|---|---|---|
| 1. qwen-0.5b matrix | ~35–40 h | 40 h |
| 2. Ablation B1 | ~12 h | 52 h |
| 3. Ablation A | ~37 h | 89 h |
| 4. Ablation D | ~3 h | 92 h |
| 5. Ablation E | ~6 h | 98 h |

~4 days of continuous GPU for the full programme. **If time is short, steps 1
and 2 alone** (~52 h) give the two-model study plus the non-IID baselines, which
covers the two gaps most likely to be raised in review.
