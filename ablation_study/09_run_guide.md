# 09 — Run guide

Execution order for what is left. Revised 2026-08-14.

> ## The whole remaining programme is one command
>
> ```bash
> bash scripts/run_final.sh          # ~13 h, unattended
> ```
>
> It runs the two outstanding experiments (steps 1 and 2 below), re-scores what
> they produce on a single metric backend, rebuilds every comparison table, and
> emits the paper tables — ending with a machine verdict:
>
> | Verdict | Meaning |
> |---|---|
> | `COMPLETE` | every paper-blocking table is filled and every audited artefact is bit-identical | 
> | `INCOMPLETE` | a run is missing; the summary names which |
> | `HASH DIVERGENCE` | the audit layer changed a trained artefact. **A bug, not a result** — it falsifies the central claim. Stop and diagnose. |
>
> `--dry-run` rehearses it, `--only audit,b1,reeval,tables,paper` runs a subset,
> `--force` redoes completed steps. Then write the paper from `results/paper/`
> following [12_paper_plan.md](12_paper_plan.md).
>
> Steps 3–6 further down are **out of scope for this paper** and kept for a
> follow-up; their configs and runner blocks exist and are validated.

## Where these must run

**Not on this checkout.** The machine holding this repo has:

| Requirement | State here |
|---|---|
| CUDA device | **absent** — `torch 2.13.0+cpu`, `cuda.is_available() == False` |
| `trl` (SFT trainer) | not installed |
| `bitsandbytes` (4-bit NF4) | not installed — CUDA-only anyway |
| `web3` (chain anchoring) | not installed |
| anvil / IPFS daemon | **not on PATH** |
| Trained adapters (`outputs/`) | **absent** — they live on the GPU box |

Everything was produced at `/home/mann/FedChain` on WSL2 with an NVIDIA T600
(4 GB). Run everything below there, or on equivalent hardware.

What *is* verified on this checkout: config resolution for all 26 experiment and
ablation configs, budget matching across every arm, the C1/C2 accounting and
cadence logic under `--dry-run`, sweep discovery, report generation, and the test
suite.

---

## What is already done

| Block | Status | Output |
|---|---|---|
| smollm2-360m matrix, 3 seeds | done | `results/smollm2-360m/` |
| qwen-0.5b matrix, 3 seeds | done | `results/qwen-0.5b/` |
| 360M E0 re-run (kills the 299.202 MiB artefact) | done 2026-08-12 | reports 0.000 MiB |
| **Ablation B1** (α=0.3 at 0.5B) | **done 2026-08-11/12** | `results/qwen-0.5b/ablation/` |
| **Ablation E** (re-score @250, `evaluate`) | **done 2026-08-13** | `results/<tier>/reeval250` |
| E6 / E7 at 360M | done | `results/smollm2-360m/exp6*, exp7*` |
| E6 / E7 at qwen-0.5b | **partial** — 20 trials, N≤50 | needs step 1 below |

Results are written up in [06](06_ablation_results.md); what they license is in
[07](07_ablation_conclusions.md).

---

## Step 0 — Sync and sanity-check

```bash
git pull
pip install -r requirements.txt
python -m pytest tests/ -q
./infra.sh                                  # anvil + IPFS daemon + contract
```

Confirm the metric stack is reachable. Every config now sets
`require_metric_backend: evaluate`, so a missing dependency is a hard failure at
the first evaluation rather than a silent fallback — which is exactly the defect
that contaminated the main-table generation metrics
([06 §E.0](06_ablation_results.md#e0--the-backend-split-read-this-before-any-generation-number)):

```bash
python -c "import evaluate, nltk, rouge_score; print('metric stack OK')"
```

---

## Step 1 — E6 @ 50 trials + E7 to N=100 at qwen *(minutes; do this first)*

The only outstanding item that blocks anything. `finish_study.sh` skipped it on
2026-08-13 because no chain was reachable and `anvil` was not on PATH.

```bash
anvil --host 127.0.0.1 --port 8545 &        # or: ./infra.sh
bash scripts/finish_study.sh --only audit
```

Buys protocol parity with the 360M tier and tightens the false-positive bound
from **13.9% → 5.8%**. Until it runs, the paper cannot claim a 0% FP rate at
0.5B — 0/20 benign trials does not support it.

---

## Step 2 — Ablation B1 at smollm2-360m *(~12 h; the highest-value training run)*

B1 ran only at 0.5B, where FedAvg already recovered 34% of the gap under IID. The
tier where it recovered **2.4%** has no non-IID arm, so the study cannot say
whether skew rescues a weak baseline — only that it widens an open one. See
[06 §B.6](06_ablation_results.md#the-premise-correction).

```bash
# shards already exist from the 0.5B run
./ablation_study/run_ablation.sh --block B1 --model smol --seeds "42 43 44"
```

**Re-derive the decision thresholds for this tier before you start** and record
them in [05](05_ablation_design.md). The existing ones were calibrated on the
360M IID baseline, which is what this run finally makes them applicable to.

---

## Step 3 — Ablation A *(round sweep; ~37 h)*

```bash
./ablation_study/run_ablation.sh --block A --model qwen-0.5b --seeds "42 43 44"
```

Bounds the 34.0% / 41.5% claims the paper leads with. Every loss curve in the
study is still descending at R=3 (decay ratio ≈ 0.31), so both figures are
fixed-budget lower bounds of unknown tightness.

A2 (local-only at R=9) must set `eval_local_clients_every_round: true` — the
baseline E0 arms evaluate only at the end, which is why
[06 §A.1](06_ablation_results.md#a1-loss-trajectory-qwen-05b-mean-over-3-seeds)
has no E0 column.

---

## Step 4 — Ablation D *(audit decomposition; ~3 h, one seed)*

```bash
./ablation_study/run_ablation.sh --block D --model qwen-0.5b --seeds "42"
```

Note what D no longer has to establish: **E3 already proves anchoring adds zero
communication** (302.86 MiB, byte-identical to E2), so the whole +31.8% is IPFS
transport. D2 splits it *within* IPFS. Afterwards confirm H-D4 — every variant
must still produce E4-identical adapters.

---

## Step 5 — Ablations B2 / B3 *(the α curve; ~32 h)*

B1 gives one contrast, which is a direction. H-B1 predicted an ordering over four
α values. Configs now exist for both.

```bash
# alpha = 0.1 -- NOTE the output dir; without it this overwrites B1's shards
python data/prepare_data.py --partition dirichlet --alpha 0.1 \
    --output-dir data/dirichlet_a01

# alpha = 1.0
python data/prepare_data.py --partition dirichlet --alpha 1.0 \
    --output-dir data/dirichlet_a10

for C in ablationB2_e0_alpha01 ablationB2_e1_alpha01 ablationB2_e2_alpha01 \
         ablationB3_e0_alpha10 ablationB3_e1_alpha10 ablationB3_e2_alpha10; do
  for S in 42 43 44; do
    python main.py --config ablation_study/configs/$C.yaml --model qwen --seed $S
  done
done
```

> **Two gates before running B2.** `prepare_data.py` writes non-IID shards to
> `data/<partition>/` — i.e. `data/dirichlet/` at *every* α — so omitting
> `--output-dir` destroys the shards B1 was run on and makes B1 unreproducible.
> And at α=0.1 the smallest shard may fall below the 1,500 records that R=3 × 500
> needs; read the sizes `prepare_data.py` prints, and if it does, lower
> `max_train_samples` for **all three arms of the triple** rather than dropping a
> round. Budget matching across arms is what makes the comparison fair.

**Check the control first.** E1 must read the same at every α (2.0499 at IID,
2.0492 at α=0.3). If it moves, the repartition changed the corpus rather than
redistributing it and the triple is invalid.

---

## Step 6 — Ablations C and F *(~40 h; breadth, not load-bearing)*

C generalises the zero-cost claim off a single hyperparameter — currently the
weakest leg of the strongest contribution. `--local-epochs` exists on `main.py`,
so the sweep runs without a code change:

```bash
for E in 1 2 4; do
  python main.py --config ablation_study/configs/ablationC_local_epochs.yaml \
      --model qwen --seed 42 --exp-name "ablationC_iid_e${E}" --local-epochs $E
done
```

F needs re-sharded data at constant union, which is what keeps it a
federation-size result rather than a data-quantity one:

```bash
python data/prepare_data.py --num-clients 5  --output-dir data/iid_n5
python data/prepare_data.py --num-clients 10 --output-dir data/iid_n10

for C in ablationF_clients5 ablationF_clients5_local \
         ablationF_clients10 ablationF_clients10_local; do
  python main.py --config ablation_study/configs/$C.yaml --model qwen --seed 42
done
```

The F configs already carry matched budgets — 5×3×300 and 10×3×150 both come to
4,500 updates, the same as every 3-client arm.

---

## Step 7 — Reports

```bash
python scripts/compare_results.py --results-dir results/qwen-0.5b --seeds
python scripts/compare_results.py --results-dir results/smollm2-360m --seeds
python scripts/compare_results.py --results-dir results/qwen-0.5b/ablation --seeds \
    --order ablationB_e0_noniid,ablationB_e1_noniid,ablationB_e2_noniid \
    --baseline ablationB_e1_noniid --extra-baselines ablationB_e0_noniid
python scripts/compare_results.py --across-models --results-dir results
```

> **Do not quote the ROUGE-L or BLEU rows from any `comparison.md`.** They mix
> two scorers: 30 main-table runs used `builtin`, B1's 6 used `evaluate`. Quote
> `results/<tier>/reeval250` instead, or regenerate those rows from it.

---

## Re-scoring after any new training block

Ablation E is not one-and-done — any new adapters need the same treatment before
their generation metrics are comparable:

```bash
python scripts/reevaluate.py --sweep results/qwen-0.5b/ablation \
    --config configs/exp4_fedchain.yaml --model qwen-0.5b \
    --gen-num-samples 250 --require-backend evaluate \
    --out results/qwen-0.5b/ablation/reeval250
```

Note what this does **not** buy: narrower confidence intervals. The reported ±CI
is a between-seed interval; sample count does not touch it, and the measured
50 → 250 ratios were 0.08×–1.86× ([06 §E.2](06_ablation_results.md#e2-does-250-samples-buy-narrower-intervals)).
It buys less noisy per-seed estimates. Tighter intervals need **more seeds**.

---

## Recording results

Update [06](06_ablation_results.md) as runs land — run-status table first, then
the prediction-check tables, then the anomaly log. Only then revisit
[07](07_ablation_conclusions.md), working through the branches fixed there.
Update the ledger in [11](11_final_status.md) last.

---

## Total remaining

| Step | Cost | Cumulative | Blocking? |
|---|---|---|---|
| 1. E6/E7 parity at qwen | minutes | — | **yes** |
| 2. B1 at 360M | ~12 h | 12 h | yes, if the skew claim is said to generalise |
| 3. Ablation A | ~37 h | 49 h | no, but it bounds a headline number |
| 4. Ablation D | ~3 h | 52 h | no |
| 5. B2 / B3 | ~32 h | 84 h | no |
| 6. C and F | ~40 h | 124 h | no |

**If time is short: step 1 alone unblocks submission.** Steps 1–2 (~12 h) are
what make the non-IID claim a cross-scale one rather than a single-tier one.
