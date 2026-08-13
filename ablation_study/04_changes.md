# 04 — Changes made before the ablation runs

Status: **applied and verified.** Each change is described against the code that
implements it, with the verification that was actually run.

Ordering matters: **C1–C4 are correctness fixes** (the baseline numbers are
wrong or unusable without them). **C5–C7 are enablers** (the ablations cannot be
run or analysed without them).

## Verification summary

| Change | Implemented in | Verified by | Result |
|---|---|---|---|
| C1 communication accounting | [trainer/federated.py](../trainer/federated.py) | `exp0_local --dry-run` | **299.202 → 0.000 MiB**; federated arm unchanged (15.893 MiB) |
| C2 per-round local eval + stride | [trainer/federated.py](../trainer/federated.py) | 5-round dry run at stride 2 | scored r2, r4; skipped r1, r3; deferred r5 to final |
| C3 metric-backend enforcement | [evaluation/eval_loss.py](../evaluation/eval_loss.py) | unit check both paths | permissive warns; `require_metric_backend: evaluate` raises |
| C4 re-evaluation entry point | [scripts/reevaluate.py](../scripts/reevaluate.py) | sweep discovery on real results | 24 targets found (E0 → 9 client adapters, others → 3 each) |
| C5 non-IID configs | [configs/](configs/) | config loader | all 10 resolve with expected values |
| C6 second baseline + trajectory | [scripts/compare_results.py](../scripts/compare_results.py) | regenerated `comparison.md` | E2−E0 = −0.0009 ± 0.0002 now emitted; trajectory table added |
| C7 local-only semantics | [trainer/federated.py](../trainer/federated.py) | dry run | `reported_metric: local_only_mean` + `communication_note` in report |

Regression guards live in `tests/test_experiment_validity.py`
(`LocalOnlyCommunicationTests`, `RoundEvaluationCadenceTests`). Suite: **32
passed, 1 skipped**. The skip is deliberate — see C1 below.

---

## C1 — Fix phantom communication accounting in the local-only arm

**Severity: correctness. Blocks publication of Table 1.**

**Problem.** E0 reports 299.20 MiB of communication despite having no server, no
aggregation and no recipient — the same volume as E2, which actually federates.
E1 correctly reports 0.

**Root cause.** [trainer/federated.py:396-409](../trainer/federated.py#L396-L409).
The publish-and-broadcast accounting runs unconditionally, outside the
`enable_aggregation` guard that protects the aggregation step at line 383:

```python
publish_dir = Path(self.global_adapter_path or global_dir)
global_record = self._publish_global(round_index, publish_dir)   # counts an upload
broadcast_bytes = global_record["upload_bytes"] * self.num_clients
if self.enable_ipfs:
    broadcast_bytes = global_record["upload_bytes"] * (self.num_clients + 1)
global_record["broadcast_bytes"] = broadcast_bytes
round_comm_bytes += broadcast_bytes                              # counts N downloads
```

With `enable_aggregation: false` there is no global model — `_no_aggregation_metrics`
([federated.py:477-493](../trainer/federated.py#L477-L493)) sets
`global_adapter_path` to **client 1's own directory**. So the code publishes
client 1's adapter as if it were a global model and then bills a broadcast of it
to 3 clients who never requested it. Arithmetic confirms this exactly:
49.87 MiB client uploads + 49.87 MiB phantom broadcast = 99.73 MiB/round ×
3 rounds = **299.20 MiB**, and every client's `download_bytes` is `0`.

**Fix applied.** A `counts_communication` flag (set from `enable_aggregation`)
now gates three sites: the client upload accumulator, `_finish_client`'s
`upload_bytes`, and the broadcast term. `_publish_global` still runs — its hash
is useful — but contributes no bytes.

**Measured effect.** E0 communication `299.202 → 0.000 MiB`, every client's
`upload_bytes` and the round's `broadcast_bytes` now `0`. The federated arms are
untouched: `exp2_fl` still reports 15.893 MiB and `exp4_fedchain` 20.349 MiB on
the same dry run, with 9/9 integrity checks and 12/12 transactions.

The report now also carries `communication_counted: false` and a
`communication_note` explaining the zero, so a reader cannot mistake it for
instrumentation that simply missed something.

**Guard.** `LocalOnlyCommunicationTests` asserts the invariant on every report
under `results/`. Reports written *before* this fix carry no
`communication_counted` key; rather than silently passing them, the test lists
them as stale and skips:

```
SKIPPED [1] no post-fix local-only reports to check (3 stale)
```

Those three were the shipped 360M `exp0_local` runs for seeds 42/43/44. Their
299.202 MiB could not be corrected without re-running the arm — the phantom bytes
were never measured, so there was nothing to recompute from.

> **Closed 2026-08-12.** The 360M E0 arm was re-run and now reports **0.000 MiB**,
> matching qwen-0.5b. No stale local-only report remains, so the guard is an
> enforced assertion rather than a skip.

---

## C2 — Per-round evaluation for the local-only arm

**Severity: blocks Ablation A.**

**Problem.** E0 evaluates once, at the end, reporting `local_only_mean` — the
mean over 3 isolated clients. There is no per-round trajectory, so the question
"does FedAvg's advantage over isolation grow with rounds?" is unanswerable.

**Root cause.** [trainer/federated.py:411-416](../trainer/federated.py#L411-L416):

```python
eval_metrics = (
    self._evaluate_round(round_index, publish_dir) if self.enable_aggregation else None
)
```

This is a deliberate cost decision, not an oversight — evaluating local-only per
round means 3 evaluations per round instead of 1.

**Fix applied.** Two new config keys, both defaulting to current behaviour:

- `eval_local_clients_every_round: false` — when true and aggregation is
  disabled, each client adapter is scored per round and `rounds[r].evaluation`
  carries the mean plus `per_client` detail and `spread`, in the same shape the
  federated arms use so trajectory tables need no special-casing.
- `eval_round_stride: 1` — score only rounds where `round % stride == 0`.

Both are also exposed on the CLI (`--eval-round-stride`), alongside two other
overrides the ablations need: `--local-epochs` and `--gen-num-samples`.

**Measured effect.** A 5-round dry run at stride 2 scored rounds 2 and 4, skipped
1 and 3, and deferred round 5 to the final pass (scoring it inline would pay for
the same forward pass twice):

```
Round 2 local-only mean over 3 client(s) | val_loss=2.0000 ppl=7.3891
Round 4 local-only mean over 3 client(s) | val_loss=2.0000 ppl=7.3891
```

Cost at stride 2 over 9 rounds: ~5,100 s of extra evaluation per seed, versus
~9,180 s at stride 1.

---

## C3 — Install `nltk`, or pin the metric implementation explicitly

**Severity: correctness of reported absolute values.**

**Problem.** Every baseline run logged:

```
WARNING | evaluation.eval_loss | The `evaluate` library failed to score
(... you need to install ['nltk'] ...); using the built-in implementation.
```

The fallback is consistent across runs so internal comparisons hold, but the
absolute ROUGE-L values are not the standard implementation and cannot be
compared to other papers.

**Correction to the original diagnosis.** `nltk>=3.8.1` was *already* pinned in
[requirements.txt:36](../requirements.txt#L36). It was simply not installed in
the run environment. Adding a dependency that is already declared would fix
nothing — the real defect is that a missing optional dependency silently changed
the metric implementation for all 18 runs and was only discoverable by grepping
the logs afterwards.

**Fix applied.** A `require_metric_backend` config key. Left empty (the default)
the fallback still happens, but the warning now states the consequence — that
absolute values are not comparable with published numbers — and points at the
`generation_metric_backend` field that records what was actually used. Set to
`evaluate`, a missing dependency raises at the **first** evaluation:

```
RuntimeError: require_metric_backend='evaluate' but the `evaluate` library
failed to score (...). Install the metric stack (`pip install evaluate
rouge-score nltk`) or unset require_metric_backend ...
```

That converts a footnote discovered after 24 GPU-hours into a failure that costs
minutes. `scripts/reevaluate.py --require-backend evaluate` sets it too.

**Use it for the paper run.** Add `require_metric_backend: "evaluate"` to the
config, or accept the built-in implementation and say so in the paper. Either is
defensible; falling back silently is not.

> **Outcome, 2026-08-13.** `require_metric_backend: "evaluate"` is now set in
> `base_config.yaml` (since `f175959`, 2026-08-07) and in every ablation config,
> so it is live for all future runs. Ablation E exercised it in anger.
>
> **The original diagnosis understated the damage in one respect.** "The fallback
> is consistent across runs so internal comparisons hold" turned out to be false
> once B1 ran on a machine that *did* have the metric stack: the 30 main-table
> runs used `builtin`, B1's 6 arms used `evaluate`, and the two sets are not
> comparable. E5 and B1-E2 have **bit-identical adapters** and printed ROUGE-L of
> 0.2276 and 0.2340 respectively — which is how the split was found.
>
> The affected reports are unfixable in place; `results/<tier>/reeval250`
> re-scores every adapter on one backend and is the only quotable source for
> generation metrics. See
> [06 §E.0](06_ablation_results.md#e0--the-backend-split-read-this-before-any-generation-number).

---

## C4 — Raise generation sample count and add a standalone re-evaluation path

**Severity: makes two table rows usable instead of decorative.**

**Problem.** At `gen_num_samples: 50`, ROUGE-L and BLEU CIs span 5–17% of their
values. The rows cannot support any claim.

**Fix, part 1.** Raise `gen_num_samples` to **250** in `base_config.yaml`.
Generation costs ~4.4 s/sample (218.9 s for 50), so 250 samples ≈ 1,100 s per
evaluation — acceptable at final-eval cadence, too expensive every round. Pair
with a separate `gen_num_samples_final: 250` / `gen_num_samples_round: 50` split
so per-round curves stay cheap.

**Fix, part 2.** Add `scripts/reevaluate.py` that loads an existing adapter
directory and re-scores it without retraining:

```bash
python scripts/reevaluate.py \
    --adapter outputs/smollm2-360m/seed_42/exp4_fedchain/round_3/global \
    --gen-num-samples 250 --out results/smollm2-360m/seed_42/exp4_reeval.json
```

This is what makes C3 and C4 affordable — the existing baseline adapters can be
re-scored for the cost of evaluation alone, with no retraining.

**Expected effect.** Generation CIs narrow by roughly `sqrt(250/50) = 2.24×`.
E5's ROUGE-L interval would go from ±0.042 to roughly ±0.019 — still wide, but
reportable.

---

## C5 — Non-IID baselines for E0, E1 and E2

**Severity: blocks the entire non-IID claim.**

**Problem.** E5 is the only run on Dirichlet shards. E0–E4 all use
`data/client*.jsonl` (IID), verified from every config block. `comparison.md`
therefore reports E5's paired difference against **IID E1**, conflating the
partition change with the federation change.
[EXPERIMENTS.md:23](../EXPERIMENTS.md#L23) already states the requirement:

> Report Exp 5 against an Exp 0/1/2 rerun on the same non-IID shards, not
> against the IID numbers.

**Fix.** Three new configs pointing at `data/dirichlet/`, provided in
[`configs/`](configs/): `ablationB_e0_noniid.yaml`, `ablationB_e1_noniid.yaml`,
`ablationB_e2_noniid.yaml`. E1's non-IID variant must pool the **Dirichlet**
shards, not the IID ones.

**Constraint discovered from the manifest.** Dirichlet(0.3) shard sizes are
4,798 / 2,715 / 6,998. The smallest shard caps the round count at
`floor(2715/500) = 5` rounds before the window runs off the end. Any non-IID
round sweep must stop at 5 rounds, or lower `max_train_samples`.

---

## C6 — Emit paired per-round differences in `compare_results.py`

**Severity: convenience, but prevents analysis errors.**

**Problem.** The E2 − E0 comparison — the single most important number for the
paper's motivation — is not computed anywhere. It had to be derived by hand from
the per-seed metrics files. `compare_results.py` only pairs against `exp1_sft`.

**Correction to the original diagnosis.** `--baseline` already existed. What was
missing was the ability to emit *both* pairings in one report, and any per-round
view at all.

**Fix applied.** `--extra-baselines` (default `exp0_local`) emits additional
paired-difference tables, and `build_trajectory_table` adds per-round validation
loss averaged over seeds. Regenerating `comparison.md` now yields, without any
hand computation:

| Experiment | Metric | Mean diff | 95% CI | Seeds | Significant |
|---|---|---|---|---|---|
| E2: FedAvg | Validation Loss | −0.0009 | ±0.0002 | 3 | yes |

…matching the hand-derived −0.00085 ± 0.00022, plus:

| Round | E2 | E3 | E4 | E5 |
|---|---|---|---|---|
| 1 | 2.0834 | 2.0834 | 2.0834 | 2.0849 |
| 2 | 2.0388 | 2.0388 | 2.0388 | 2.0412 |
| final | 2.0228 | 2.0228 | 2.0228 | 2.0249 |

The trajectory table notes that arms without per-round rows evaluate only at the
end — which is E0 and E1 today, and is what C2 fixes for E0.

---

## C7 — Record the local-only comparison semantics explicitly

**Severity: presentation, but a reviewer will ask.**

**Problem.** E0's headline number is the **mean of 3 separately-evaluated
models**; E2's is **one aggregated model**. Comparing them is the right thing to
do — "what does a participant get?" — but the asymmetry is invisible in the
table.

**Fix.** Carry `label: "local_only_mean"` and `num_clients_averaged` into
`comparison.md` as a footnote on the E0 column, and report the per-client spread
(already captured in `run_summary.per_client_evaluation`) alongside the mean.
For seed 42 the per-client losses were 2.0255 / … — that spread is itself
evidence about whether isolation is high-variance.

---

## Configuration deltas summary

| Key | Baseline | Ablation | Rationale |
|---|---|---|---|
| `gen_num_samples` | 50 | 250 (final), 50 (per-round) | C4 — usable CIs |
| `eval_local_clients_every_round` | *(new)* | true for Ablation A | C2 — E0 trajectory |
| `eval_round_stride` | *(new)* | 2 for 9-round runs | C2 — bound eval cost |
| `num_rounds` | 3 | 9 (IID), 5 (non-IID) | Ablation A; shard-size capped |
| `client_files` | `data/client*.jsonl` | `data/dirichlet/client*.jsonl` | C5 |
| `local_epochs` | 1 | 1, 2, 4 | Ablation C |
| `log_global_model` | true | false variant | Ablation D1 |
| `ipfs_roundtrip_aggregation` | true | false variant | Ablation D2 |
| `verify_hash_on_download` | true | false variant | Ablation D3 |

## Things deliberately **not** changed

- **`fedavg_weighted` in E5.** It is `true` in E5 and `false` in E2–E4, which
  looks like a confound. It is not: `max_train_samples: 500` caps every client
  below the smallest Dirichlet shard, so all clients contribute 500 samples and
  the weights come out uniform — the observed weights are `[0.333, 0.333, 0.333]`
  in both. [exp5_noniid.yaml:47-59](../configs/exp5_noniid.yaml#L47-L59) already
  documents this precisely. **The change needed is to the paper, not the config:**
  describe E5 as *label* skew with balanced quantities (the standard Dirichlet
  benchmark), not quantity skew.
- **Seed count.** Three seeds, `t = 4.303`. Raising to 5 would narrow every
  interval, but the intervals are not currently the binding constraint — the
  effect sizes are.
- **The hashing scheme.** E6's 0% false-positive rate says canonical-JSON
  hashing works. Leave it alone.
