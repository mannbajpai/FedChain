#!/usr/bin/env bash
# =============================================================================
# FedChain :: ablation study runner
# -----------------------------------------------------------------------------
# Drives the ablation blocks defined in ablation_study/05_ablation_design.md, in
# priority order, writing to results/<tier>/ablation/seed_<N>/.
#
# This is deliberately a thin wrapper around main.py rather than a fork of
# run_all.sh: the ablations reuse the same infrastructure (anvil + IPFS), the
# same checkpointing, and the same metrics writer. Start the infrastructure with
# ./infra.sh first, exactly as for the baseline.
#
#   ./ablation_study/run_ablation.sh --block B1 --seeds "42 43 44"
#   ./ablation_study/run_ablation.sh --block D            # systems only, 1 seed
#   ./ablation_study/run_ablation.sh --block A --seeds "42 43" --dry-run
#
# Blocks:
#   A    round sweep       E2/E0 @ R=9, E1 budget-matched   ~37 h / 3 seeds
#   B1   non-IID baseline  E0/E1/E2 @ alpha=0.3             ~12 h / 3 seeds   DONE @ qwen-0.5b, smollm2-360m
#   B2   alpha sweep       E0/E1/E2 @ alpha=0.1             ~16 h / 3 seeds
#   B3   alpha sweep       E0/E1/E2 @ alpha=1.0             ~16 h / 3 seeds
#   C    local epochs      E in {1,2,4}                     ~22 h / 3 seeds
#   D    audit decomp      3 systems-only variants          ~3.3 h / 1 seed
#   F    federation size   N in {5,10}, constant union      ~18 h / 1 seed
#
# B1 is what makes the LEARNING claim sayable at a tier: E5 is Dirichlet-sharded
# while E0/E1/E2 are IID, so without a matched non-IID triple every E5-E0 gap
# confounds the partition with federation. A tier without B1 therefore carries a
# systems result only. llama-3.2-1b is the tier that still needs it:
#
#   ./ablation_study/run_ablation.sh --block B1 --model llama-3.2-1b --seeds "42 43 44"
#
# PREREQUISITE: changes C1-C4 in ablation_study/04_changes.md must be applied
# first. Block A2 in particular is inert without C2 - it will run, but it will
# produce a single end-point instead of the per-round curve the block exists to
# measure. The script warns but does not block, so that a deliberate
# without-C2 run is still possible.
# =============================================================================
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

CONFIG_DIR="ablation_study/configs"
PYTHON="${PYTHON:-python}"
MODEL="smol"
SEEDS="42 43 44"
BLOCK=""
EXTRA_ARGS=()

info() { printf '\033[0;36m[ablation]\033[0m %s\n' "$*"; }
warn() { printf '\033[0;33m[warn]\033[0m %s\n' "$*"; }
fail() { printf '\033[0;31m[fail]\033[0m %s\n' "$*" >&2; }

usage() {
    sed -n '2,37p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --block)  BLOCK="$2"; shift 2 ;;
        --model)  MODEL="$2"; shift 2 ;;
        --seeds)  SEEDS="$2"; shift 2 ;;
        --)       shift; while [[ $# -gt 0 ]]; do EXTRA_ARGS+=("$1"); shift; done ;;
        -h|--help) usage; exit 0 ;;
        *)        EXTRA_ARGS+=("$1"); shift ;;
    esac
done

[[ -z "$BLOCK" ]] && { fail "--block is required (A | B1 | B2 | B3 | C | D | F)"; usage; exit 2; }

# -- resolve the model alias to its canonical tier key -------------------------
# $MODEL is used for BOTH `main.py --model` and the results/outputs paths. main.py
# accepts aliases ("smol"), but the paths must use the canonical key, or a run
# launched as --model smol writes to results/smol/ablation/ while the rest of the
# 360M tier lives in results/smollm2-360m/. That split is silent and only shows up
# later as a comparison table with half the arms missing.
if RESOLVED="$("$PYTHON" utils/models.py --resolve "$MODEL" 2>/dev/null | cut -f1)" \
   && [[ -n "$RESOLVED" ]]; then
    [[ "$RESOLVED" != "$MODEL" ]] && info "resolved model alias '$MODEL' -> '$RESOLVED'"
    MODEL="$RESOLVED"
else
    warn "could not resolve '$MODEL' to a tier key; using it verbatim for paths."
fi

# -- metric-stack preflight ---------------------------------------------------
# This script calls main.py directly, so it does not pass through run_all.sh's
# environment step. With require_metric_backend set, a broken metric stack
# aborts at the first evaluation - which for the E0 arm is ~20 minutes into the
# block. Check it in seconds instead. See run_all.sh for the full rationale.
if ! "$PYTHON" - <<'PY'
import sys

required = ""
try:
    import yaml
    with open("configs/base_config.yaml") as fh:
        required = str((yaml.safe_load(fh) or {}).get("require_metric_backend", "") or "").strip().lower()
except Exception:
    pass
if required not in {"evaluate", "hf", "huggingface"}:
    sys.exit(0)

try:
    import evaluate as hf
    hf.load("rouge").compute(predictions=["the cat sat"], references=["the cat sat"])
    hf.load("bleu").compute(predictions=["the cat sat"], references=[["the cat sat"]])
except Exception as exc:
    print(f"interpreter: {sys.executable}")
    print(f"metric stack unusable: {type(exc).__name__}: {exc}")
    for name in ("nltk", "rouge_score", "evaluate"):
        try:
            module = __import__(name)
        except Exception as inner:
            print(f"  {name}: {type(inner).__name__}: {inner}")
        else:
            print(f"  {name} {getattr(module, '__version__', 'unknown')}")
    sys.exit(1)
PY
then
    fail "base_config.yaml requires the \`evaluate\` metric backend but it is unusable."
    fail "Activate the venv and install into THAT interpreter:"
    fail "    source .venv/bin/activate && python -m pip install -U nltk rouge-score evaluate"
    exit 1
fi
info "generation-metric stack ready"

# -- gated-repository preflight ----------------------------------------------
# Same reasoning as the metric-stack check above, and the same reasoning as
# run_all.sh's: meta-llama/* 401s without a token, and this script goes straight
# to main.py, so the failure would otherwise land ~20 minutes in at the first
# model load rather than here.
GATED_LINES="$("$PYTHON" utils/models.py --gated-list "$MODEL" 2>/dev/null || true)"
if [[ -n "$GATED_LINES" && -z "${HF_TOKEN:-${HUGGING_FACE_HUB_TOKEN:-}}" ]]; then
    fail "model '$MODEL' is a gated Hugging Face repository and no HF_TOKEN is set."
    fail "Accept the licence on the Hub, then: export HF_TOKEN=hf_..."
    exit 1
fi

# -- C2 presence check --------------------------------------------------------
# Block A2 measures a trajectory that only exists once C2 lands.
if [[ "$BLOCK" == "A" ]] && ! grep -q "eval_local_clients_every_round" trainer/federated.py 2>/dev/null; then
    warn "change C2 does not appear to be applied (trainer/federated.py has no"
    warn "eval_local_clients_every_round). Block A2 will produce an end-point,"
    warn "not a per-round curve. See ablation_study/04_changes.md."
fi

run_one() {
    local config="$1" seed="$2"; shift 2
    local name; name="$(basename "$config" .yaml)"
    local results_dir="results/$MODEL/ablation/seed_$seed"
    local output_root="outputs/$MODEL/ablation/seed_$seed"

    info "$name | seed $seed | $*"
    mkdir -p "$results_dir"

    "$PYTHON" main.py \
        --config "$CONFIG_DIR/$config" \
        --model "$MODEL" \
        --seed "$seed" \
        --results-dir "$results_dir" \
        --output-root "$output_root" \
        "$@" ${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"} \
        2>&1 | tee "$results_dir/${name}.log"
}

case "$BLOCK" in
    # -- A: round sweep ------------------------------------------------------
    # A1/A2 give the E2-vs-E0 trajectory; A3 keeps the upper bound budget-matched
    # at R=5 and R=9. Without A3 the R=9 comparison silently hands the federated
    # arm 3x the data of the centralized reference.
    A)
        for seed in $SEEDS; do
            run_one ablationA_e2_rounds9.yaml "$seed"
            run_one ablationA_e0_rounds9.yaml "$seed"
        done
        # A3: two seeds only - E1 has the smallest seed variance (+-0.0006).
        for seed in $(echo "$SEEDS" | cut -d' ' -f1,2); do
            run_one ablationA_e1_budget9.yaml "$seed"
            run_one ablationA_e1_budget9.yaml "$seed" \
                --max-train-samples 2500 --exp-name ablationA_e1_budget5
        done
        ;;

    # -- B1: the non-IID baseline triple -------------------------------------
    # The cheapest high-value block. Shards must exist first.
    B1)
        if [[ ! -f data/dirichlet/client1.jsonl ]]; then
            info "generating Dirichlet(0.3) shards"
            "$PYTHON" data/prepare_data.py --partition dirichlet --alpha 0.3
        fi
        warn "re-read data/dirichlet/manifest.json: the smallest shard caps the"
        warn "round count (2,715 records => 5 rounds at 500/round)."
        for seed in $SEEDS; do
            run_one ablationB_e0_noniid.yaml "$seed"
            run_one ablationB_e1_noniid.yaml "$seed"
            run_one ablationB_e2_noniid.yaml "$seed"
        done
        ;;

    # -- B2 / B3: the rest of the alpha sweep --------------------------------
    # B1 measured one contrast (IID vs alpha=0.3), which is a direction. H-B1
    # predicted an ordering over four alphas; these supply the other two.
    #
    # The --output-dir is load-bearing. prepare_data.py writes non-IID shards to
    # data/<partition>/ - i.e. data/dirichlet/ at EVERY alpha - so generating
    # without it destroys the shards B1 was run on and makes B1 unreproducible.
    B2|B3)
        if [[ "$BLOCK" == "B2" ]]; then
            alpha="0.1"; dir="data/dirichlet_a01"; suffix="alpha01"
        else
            alpha="1.0"; dir="data/dirichlet_a10"; suffix="alpha10"
        fi
        if [[ ! -f "$dir/client1.jsonl" ]]; then
            info "generating Dirichlet($alpha) shards into $dir"
            "$PYTHON" data/prepare_data.py --partition dirichlet \
                --alpha "$alpha" --output-dir "$dir"
        fi
        warn "gate: read $dir/manifest.json. R=3 needs the smallest shard to hold"
        warn "1,500 records. If it does not, lower max_train_samples in ALL THREE"
        warn "arms of this triple - never drop a round in one arm only."
        for seed in $SEEDS; do
            run_one "ablation${BLOCK}_e0_${suffix}.yaml" "$seed"
            run_one "ablation${BLOCK}_e1_${suffix}.yaml" "$seed"
            run_one "ablation${BLOCK}_e2_${suffix}.yaml" "$seed"
        done
        info "check the control first: E1 must read the same at every alpha"
        info "(2.0499 at IID, 2.0492 at alpha=0.3). If it moved, the repartition"
        info "changed the corpus rather than redistributing it - triple is invalid."
        ;;

    # -- F: federation size --------------------------------------------------
    # Constant union is what keeps this a federation-size result rather than a
    # data-quantity one: N=5 gets 300 samples/round and N=10 gets 150, so both
    # come to the same 4,500 sample-updates as every 3-client arm.
    F)
        for n in 5 10; do
            if [[ ! -f "data/iid_n${n}/client1.jsonl" ]]; then
                info "re-sharding the pool across $n clients"
                "$PYTHON" data/prepare_data.py --num-clients "$n" \
                    --output-dir "data/iid_n${n}"
            fi
        done
        seed="$(echo "$SEEDS" | cut -d' ' -f1)"
        for cfg in ablationF_clients5.yaml ablationF_clients5_local.yaml \
                   ablationF_clients10.yaml ablationF_clients10_local.yaml; do
            run_one "$cfg" "$seed"
        done
        info "read F against E2-E0 at N=3 (-0.00964 +- 0.00093), not against E2 alone."
        info "if N=10 underperforms, check the per-client curves before blaming"
        info "aggregation - 150 records/round may simply be too few to learn from."
        ;;

    # -- C: local epochs -----------------------------------------------------
    # Needs a --local-epochs flag on main.py (see the config header). Until then
    # this block errors rather than silently running everything at E=1.
    C)
        if ! grep -q '"--local-epochs"' main.py 2>/dev/null; then
            fail "main.py has no --local-epochs flag; block C would run every"
            fail "point at local_epochs=1. Add the flag or copy the config per"
            fail "epoch value. See ablation_study/configs/ablationC_local_epochs.yaml."
            exit 3
        fi
        for seed in $SEEDS; do
            for E in 1 2 4; do
                run_one ablationC_local_epochs.yaml "$seed" \
                    --local-epochs "$E" --exp-name "ablationC_iid_e${E}"
            done
        done
        ;;

    # -- D: audit-layer decomposition ----------------------------------------
    # Systems metrics are deterministic given the config, so one seed suffices.
    # --skip-eval removes ~340 s/round: accuracy is known to be bit-identical.
    D)
        seed="$(echo "$SEEDS" | cut -d' ' -f1)"
        for cfg in ablationD1_no_global_anchor.yaml \
                   ablationD2_no_roundtrip.yaml \
                   ablationD3_no_verify.yaml; do
            run_one "$cfg" "$seed" --skip-eval
        done
        info "verify H-D4: client adapter hashes must match the E4 baseline exactly."
        info "any divergence is a bug - none of these switches touch the learning math."
        ;;

    *)
        fail "unknown block: $BLOCK (expected A | B1 | B2 | B3 | C | D | F)"; exit 2 ;;
esac

info "block $BLOCK complete. Record results in ablation_study/06_ablation_results.md"
info "and log anything unexpected in that file's anomaly table."
