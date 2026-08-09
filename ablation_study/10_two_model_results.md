# 10 — Two-model results: SmolLM2-360M vs Qwen2.5-0.5B

Qwen2.5-0.5B, 3 seeds (42/43/44), E0–E4 + E6/E7. ~30 hours on the T600.
Source: `results/qwen-0.5b/`. All numbers read directly from the metrics files.

---

## The headline: the motivation gap closes with model scale

The 360M pass could not show that federating was worth doing — FedAvg beat
isolated training by 2.4% of the distance to the centralized bound. **At 0.5B it
recovers 34%.**

| | SmolLM2-360M | Qwen2.5-0.5B |
|---|---|---|
| E0 Local-only | 2.0236 ± 0.0011 | 2.0783 ± 0.0007 |
| E1 Centralized | 1.9884 ± 0.0006 | 2.0499 ± 0.0016 |
| E2/E3/E4 Federated | 2.0228 ± 0.0013 | 2.0686 ± 0.0009 |
| **E2 − E0** (what federation buys) | **−0.00085 ± 0.00022** | **−0.00964 ± 0.00093** |
| **as % of the E0→E1 gap** | **2.4%** | **34.0%** |
| E2 − E1 (what federation costs) | +0.0343 ± 0.0012 | +0.0187 ± 0.0018 |

Both gaps move in the right direction at the larger model: federation buys **11×
more** and costs **45% less**. This is the outcome
[03_baseline_conclusions.md](03_baseline_conclusions.md) listed as explanation 3
— *"the effect is real and small at this model scale"* — and flagged as testable
by climbing the model ladder. It is now tested, and it holds.

**What this does to the paper.** The weakest claim in the 360M pass is no longer
weak. FedAvg over three clients demonstrably produces a materially better model
than isolated training, so there is something worth auditing. Lead the motivation
with the 0.5B numbers and present 360M as the shakedown tier it was designed to
be.

**How far to push it.** Two points on a model-size axis is a direction, not a
law. 2.4% → 34.0% is a large, consistent, tight-interval move, but do not fit a
curve to it or claim monotonicity. The honest statement is: *the benefit of
federation grows with model capacity over the range tested, and at 0.5B it is
substantial.* A third rung (qwen-1.5b) turns a direction into a trend.

---

## The core claim reproduces exactly

E2, E3 and E4 produce **bit-identical adapters** at 0.5B, as they did at 360M —
all 9 client hashes (3 clients × 3 rounds), every seed:

| Seed | First client hash | E2 == E3 | E3 == E4 |
|---|---|---|---|
| 42 | `03bc99594a4affad…` | ✔ 9/9 | ✔ 9/9 |
| 43 | `ebebb76b79c20a51…` | ✔ 9/9 | ✔ 9/9 |
| 44 | `d6aa3e63ab1978e7…` | ✔ 9/9 | ✔ 9/9 |

Paired difference E3 − E2 and E4 − E2: **exactly 0.00000**.

The audit layer is now shown to be a no-op on the learning math at **two model
scales and two architectures** (Llama-style SmolLM2 with 32 layers, Qwen2.5 with
24 layers and GQA). This is the strongest claim in the paper and it is now
scale-independent by demonstration, not just by argument.

---

## Systems cost is model-independent — measured, not asserted

The 360M pass argued from E7 that anchoring cost does not depend on model size.
The 0.5B run confirms it on a real second model:

| | SmolLM2-360M | Qwen2.5-0.5B | Δ |
|---|---|---|---|
| Trainable LoRA params | 8,683,520 | 8,798,208 | +1.3% |
| Trainable fraction | 4.07% | 2.72% | — |
| Adapter size (MiB) | 16.620 | 16.825 | +1.2% |
| **E3 gas** | **2,997,464** | **2,997,464** | **0** |
| **E4 gas** | **3,785,372** | **3,785,372** | **0** |
| E2 communication (MiB) | 299.18 | 302.86 | +1.2% |
| E4 communication (MiB) | 393.52 | 399.32 | +1.5% |

**Gas is byte-identical across tiers** — 3,785,372 for E4 in all six runs at both
scales. Communication tracks adapter size (+1.2%), exactly as it should when only
a 32-byte digest reaches the chain and the adapter itself moves over IPFS.

E7's sweeps are also byte-identical between tiers (616,560 → 1,163,296 →
1,744,914 → 3,198,971 → 7,561,286 → 14,831,811 gas for 1→50 clients; 311,439–
311,463 gas across a 220× payload range). 104/104 transactions succeeded.

**The C1 fix landed.** Local-only now reports **0.000 MiB** communication in all
three seeds, with `communication_counted: false` and
`reported_metric: local_only_mean`. The 360M tier still carries the old 299.202
figure — see below.

---

## Four data-quality problems

### 1. Timing metrics are unusable at 0.5B — VRAM leak

Peak VRAM climbed monotonically across the nine client trainings within each
experiment: 1591 → 2111 → 2633 → 3151 → 3671 → 4193 → 4711 → 5231 MB, roughly
**520 MB retained per client**, on a 4 GB card. Once past ~4 GB it thrashed into
host memory and step time went from 6.9 s/it to 60.5 s/it. Seed 44's `exp3_fl_bc`
died at the ninth training and was resumed.

The damage shows up as absurd seed-to-seed spread on identical work:

| Arm | seed 42 | seed 43 | seed 44 | spread |
|---|---|---|---|---|
| E0 training (s) | 6,188 | 15,054 | 13,330 | **2.4×** |
| E2 training (s) | 10,582 | 15,199 | 14,463 | 1.4× |

**Do not quote any wall-clock, training-time, or round-duration number from the
0.5B tier**, and do not compute the audit-layer wall-clock overhead from it. Gas,
communication volume, adapter size and the chain/IPFS latencies are unaffected —
they are deterministic given the config and do not depend on how long training
took. The <1% wall-clock overhead claim should be sourced from the 360M tier,
which ran clean.

Fixing the leak is a prerequisite for any timing claim at 0.5B or above.

### 2. E6 ran on synthetic adapters at 0.5B

```
no completed exp4 run found; tamper experiment uses synthetic adapters
```

| | SmolLM2-360M | Qwen2.5-0.5B |
|---|---|---|
| Adapter source | **12 real E4 adapters** | **synthetic** |
| Adapters | 12 | 3 |
| Trials per attack | 50 | 20 |
| Malicious detected | 200/200 (100%) | 80/80 (100%) |
| False positives | 0/50 (0%) | 0/20 (0%) |

The result is the same, but the 0.5B run does not demonstrate detection on real
trained artefacts. The runner looked for `outputs/qwen-0.5b/exp4_fedchain` while
the seed sweep wrote to `outputs/qwen-0.5b/seed_42/exp4_fedchain`. Re-run pointed
at the real path — it takes minutes:

```bash
python scripts/tamper_experiment.py \
    --adapter-root outputs/qwen-0.5b/seed_42/exp4_fedchain \
    --trials 50 --results-dir results/qwen-0.5b
```

Quote the 360M E6 numbers until this is redone.

### 3. No non-IID arm at 0.5B

`run_all.sh` defaults to experiments `0 1 2 3 4`; E5 was never run for qwen-0.5b.
The non-IID evidence rests entirely on the 360M tier — and there it still lacks
matched non-IID baselines (Ablation B1).

### 4. Seed 44's `exp3_fl_bc` is a resumed run

`sessions: 2`, gas 3,011,764 against 2,997,464 for the other two seeds (the
resume redeployed the contract and re-anchored), and `total_round_time` (6,902 s)
is *less* than `training_time` (12,489 s) because wall clock excludes the crashed
session. Its accuracy is unaffected — the adapter hashes match the other seeds'
pattern and E2/E3/E4 agree exactly — but its **systems** row should be excluded
or footnoted. Use seed 42 or 43 as the representative for E3 systems metrics.

---

## Claims status after two models

| Claim | Status | Evidence |
|---|---|---|
| Anchoring + IPFS cost zero accuracy | **solid at 2 scales** | bit-identical hashes, 6 runs |
| Gas independent of model size | **solid, measured** | 3,785,372 at both tiers |
| Communication tracks adapter size | **solid** | +1.2% adapter → +1.5% comm |
| Federation beats isolation | **now supported** | 34.0% of the gap at 0.5B |
| Cost of federation shrinks with scale | **supported, 2 points** | 0.0343 → 0.0187 |
| 100% detection / 0% FP | **solid at 360M**, synthetic at 0.5B | E6 |
| Gas linear in N | **solid, identical at both tiers** | E7 |
| Wall-clock overhead <1% | **solid at 360M only** | 0.5B timings contaminated |
| Non-IID robustness | **360M only, unmatched baselines** | E5 |
| Generation metrics usable | **no** | CIs still ~5–17% of value at 50 samples |

---

## What I'd do next, in order

1. **Re-run E6 against real 0.5B adapters** (minutes). Removes the only
   synthetic result in the paper.
2. **Fix the VRAM leak** (hours). Without it, no timing claim survives at 0.5B,
   and a 1.5B run will not complete at all — it OOM'd at 5.2 GB peak on a 4 GB
   card already.
3. **Ablation B1 at 0.5B** (~12 h). Non-IID baselines, now at the scale where
   federation demonstrably matters. Much more informative than at 360M.
4. **Re-run the 360M E0 arm** (~3.3 h) so its communication column stops printing
   299.202 — the cross-model table currently shows 0.000 for qwen-0.5b and
   299.202 for smollm2-360m, which invites exactly the wrong question.
5. **qwen-1.5b** once the leak is fixed — turns the scale trend from two points
   into three.

Ablation A (the round sweep) has dropped in priority. It existed to rescue the
motivation, and model scale already did that.
