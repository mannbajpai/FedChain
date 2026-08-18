# FedChain :: Experiment design

This document states what each experiment isolates, why it is in the paper, and
how to reproduce it. It exists because the most expensive mistakes in a
benchmark like this are not bugs — they are comparisons that run cleanly and
answer a different question than the one being claimed.

## The claim, and what supports it

> Anchoring federated model updates on-chain and moving them over IPFS gives
> verifiable provenance at negligible accuracy cost and bounded systems cost.

That sentence needs four separate pieces of evidence, and each one is a
different experiment:

| Claim | Needs | Experiment |
|---|---|---|
| Federation is a reasonable thing to do here | a lower bound (isolation) and an upper bound (pooling) | E0, E1 |
| The audit layer costs no accuracy | FL with and without it, over multiple seeds | E2 vs E3, E4 |
| The audit layer *does something* | attacks that it must catch, and benign churn it must not | E6 |
| The design is deployable | cost as a function of federation size and model size | E7 |
| The *systems* result holds where FL is hard | the non-IID regime | E5 |
| The *learning* result holds where FL is hard | non-IID arms for the lower and upper bounds too | **Ablation B1** |

That last row is a distinction worth keeping sharp. E5 is Dirichlet-sharded
while E0/E1/E2 are IID, so on its own **E5 licenses no learning claim** — every
E5−E0 or E5−E2 difference would confound the partition with federation. E5 alone
says the audit layer survives skew, which is a systems statement.
[Ablation B1](ablation_study/06_ablation_results.md#b--data-heterogeneity) supplies
the matched non-IID E0/E1/E2 and is what makes the learning claim sayable. It has
been run at **both** reported tiers, so the 2x2 over (scale, partition) is complete.

## Experiments

| # | Config | Paradigm | Isolates |
|---|---|---|---|
| E0 | `exp0_local.yaml` | 3 clients, **no aggregation** | What a participant gets training alone. The lower bound. |
| E1 | `exp1_sft.yaml` | Centralized on the **pooled union** of the client windows | The upper bound: same data, no federation. |
| E2 | `exp2_fl.yaml` | FedAvg, trusted aggregator | The accuracy cost of federating. |
| E3 | `exp3_fl_bc.yaml` | FedAvg + on-chain SHA-256 anchoring | The cost of the audit trail alone (gas, tx latency). |
| E4 | `exp4_fedchain.yaml` | E3 + IPFS transport with verified round-trip | The cost of decentralised storage. |
| E5 | `exp5_noniid.yaml` | E4 on Dirichlet(0.3) label-skewed shards | Whether the result survives the regime FL is deployed in. |
| E6 | `scripts/tamper_experiment.py` | Adversarial | Detection rate and false-positive rate of the audit layer. |
| E7 | `scripts/scalability_experiment.py` | Systems sweep | Gas and latency vs client count and adapter size. |

E2, E3 and E4 are expected to produce **identical** accuracy. That is the
result, not a bug: they differ only in the audit and transport layers, which do
not touch the learning math. Divergence between them would mean the audit layer
is corrupting updates. The numbers that must differ are gas, transaction
latency, IPFS latency and communication volume.

## Budget matching

The experiments are matched on **two** axes, and both matter.

**Updates.** `3 rounds x 3 clients x 500 samples = 4500` for the federated arm;
`1 round x 3 shards x 1500 samples = 4500` for the centralized arm.

**Data.** Federated round `r` reads window `[(r-1)*500, r*500)` of each client
shard, so three rounds cover `client_k[0:1500]`, and the union over `k` is 4500
unique records. E1 pools exactly that union.

The second axis is the one that is easy to get wrong. If every round replays
the head of the shard, a 3-round federated run is 3 epochs over 500 examples
and sees a third of the unique data the centralized baseline sees. The measured
"cost of federation" then mostly reflects the data budget, and it will not
reproduce at a different round count. The log line to check is:

```
Training client_1@r2 | shard=client1.jsonl | epochs=1.00 | window=[500, 1000)
```

A repeated `window=[0, 500)` across rounds means the comparison is invalid.

Similarly, E1 must pool the **client shards**, not the head of
`centralized_full.jsonl`. That file is the shuffled training pool and the client
shards are contiguous slices of it, so `centralized_full.jsonl[:4500]` is
literally client 1's partition — a "centralized" baseline that never sees
clients 2 and 3.

## Reproducing

```bash
# Infrastructure: anvil + IPFS daemon + contract artifact
./infra.sh

# Main result: three seeds, all six training experiments, plus the
# audit-layer experiments. This is what the paper's tables come from.
# Both tiers have been run; qwen-0.5b is the one the paper leads with.
./run_all.sh --model smol      --seeds "42 43 44" --audit-experiments
./run_all.sh --model qwen-0.5b --seeds "42 43 44" --audit-experiments

# Matched non-IID baselines (Ablation B1). Run at both reported tiers.
python data/prepare_data.py --partition dirichlet --alpha 0.3
./ablation_study/run_ablation.sh --block B1 --model qwen-0.5b    --seeds "42 43 44"
./ablation_study/run_ablation.sh --block B1 --model smollm2-360m --seeds "42 43 44"

# Generation metrics on one scorer at 250 samples. Required before any
# ROUGE-L/BLEU number is quoted -- see "Reading the output" below.
python scripts/reevaluate.py --sweep results/qwen-0.5b \
    --config configs/exp4_fedchain.yaml --model qwen-0.5b \
    --gen-num-samples 250 --require-backend evaluate \
    --out results/qwen-0.5b/reeval250
```

Audit-layer experiments standalone (no GPU, minutes):

```bash
python scripts/tamper_experiment.py --adapter-root outputs/smollm2-360m/exp4_fedchain --trials 50
python scripts/scalability_experiment.py --clients 1,3,5,10,25,50,100
```

## Reading the output

`results/<tier>/comparison.md` contains, in order:

1. **Metrics** — one representative seed. Systems metrics are deterministic
   given the config, so quote them from here. Do **not** quote accuracy here.
2. **Accuracy across seeds** — mean ± 95% CI, using Student's *t* (at three
   seeds the critical value is 4.303, not 1.96; using the normal approximation
   would understate the interval by more than half).
3. **Paired difference vs baseline** — the per-seed difference with its own CI.
   This is the number to quote for any "X costs Y" claim. Pairing matters:
   seed-to-seed variation is shared by both arms and cancels, so the CI is over
   the effect rather than over the noisy absolute scores.

A "Significant: no" on the E3/E4-vs-E2 rows is a **positive** result — it is the
audit layer failing to change anything measurable, which is the paper's claim.

> ### ⚠ Do not quote the ROUGE-L or BLEU rows from `comparison.md`
>
> They mix two scorers. `evaluation/eval_loss.py` falls back to a self-contained
> ROUGE/BLEU implementation when the HF `evaluate` library cannot be imported,
> and records which was used as `generation_metric_backend`. **All 30 main-table
> runs took the fallback; Ablation B1's 6 arms did not.** The absolute values are
> therefore not comparable between the two tables — and are not the standard
> implementation in the `builtin` case.
>
> This is historical: `require_metric_backend: "evaluate"` has been set in
> `base_config.yaml` since 2026-08-07 and makes a missing metric stack a hard
> failure, but the affected runs predate it.
>
> **Quote `results/<tier>/reeval250` instead** — every arm, one scorer, 250
> samples. On that common scorer, E5 and B1-E2 (bit-identical adapters) agree to
> 6 dp, which is the check that surfaced the split in the first place.

## Statistical reporting

With three seeds you can state:

- a paired difference and its 95% CI;
- whether that CI excludes zero.

You cannot state:

- that E2 and E3 are *equal* (absence of evidence is not equivalence — if you
  need an equivalence claim, pre-register a margin and run a TOST). For the audit
  layer this is moot: **artefact hash equality is a stronger claim than any
  statistical test**, and it is what the paper should lead with;
- that a ROUGE-L or BLEU difference is meaningful at `gen_num_samples: 50`.
  Fifty greedy generations is a sanity check, not a claim.

**And one thing that turned out to be wrong.** An earlier version of this section
advised raising `gen_num_samples` to 200+ "before putting a generation-quality
difference in a table", on the theory that the confidence intervals would narrow
by `sqrt(n)`. [Ablation E](ablation_study/06_ablation_results.md#e2-does-250-samples-buy-narrower-intervals)
tested that directly and it is false: at 50 → 250 the measured CI ratios were
0.08×, 0.43×, 0.63× and 1.86× — **three of four widened**.

The reason is that the reported ±CI is a **between-seed** interval at n=3.
Generation sample count controls the Monte-Carlo error *inside* each seed's point
estimate; the interval is measuring seed-to-seed spread, which it does not touch.
The interval is floored by seed variance at any sample count.

So: raising the sample count buys a less noisy point estimate per seed — worth
doing, and it is what made the qwen-0.5b ROUGE-L ordering agree with the loss
ordering — but **narrower generation intervals require more seeds, not more
samples.** Budget accordingly.

## Threat model for E6

The aggregator is honest; transport is not. A client anchors `H(theta_k)`
on-chain and publishes `theta_k` to IPFS. An adversary controlling storage or
the network path substitutes a different artefact before retrieval. The
adversary cannot rewrite the chain.

Under that model E6 measures four attacks that must be caught — `bitflip`,
`scale` (LoRA-B boosting, i.e. model replacement), `substitute`, `replay` — and
one benign control that must **not** be flagged: `reserialize`, which rewrites
`adapter_config.json` with its keys and `target_modules` in a different order.

The control is not a formality. PEFT stores `target_modules` in a `set` and
Python salts string hashes per process, so two honest runs that produce
bit-identical weights serialise that file differently. Hashing the raw bytes
would make the anchored commitment unreproducible: an auditor who retrains from
the same seed and data would compute a different digest and conclude the
artefact was tampered with. `utils/common.sha256_directory` therefore folds
`adapter_config.json` in as canonical JSON (sorted keys, sorted string lists)
rather than as raw bytes. E6's false-positive column is the evidence that this
works, and `tests/test_experiment_validity.py` guards it.

What E6 does **not** cover, and what the paper should say it does not cover:

- a malicious client that anchors a poisoned adapter it genuinely trained. The
  hash matches, so integrity checking is silent by construction. That is a
  Byzantine-robustness problem (Krum, trimmed mean, norm clipping), orthogonal
  to provenance.
- privacy. Anchoring a digest leaks nothing, but LoRA updates themselves are
  not differentially private.
