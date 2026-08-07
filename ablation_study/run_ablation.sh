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
#   B1   non-IID baseline  E0/E1/E2 @ alpha=0.3             ~12 h / 3 seeds
#   D    audit decomp      3 systems-only variants          ~3.3 h / 1 seed
#   C    local epochs      E in {1,2,4}                     ~22 h / 3 seeds
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
    sed -n '2,28p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
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

[[ -z "$BLOCK" ]] && { fail "--block is required (A | B1 | C | D)"; usage; exit 2; }

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
        fail "unknown block: $BLOCK (expected A | B1 | C | D)"; exit 2 ;;
esac

info "block $BLOCK complete. Record results in ablation_study/06_ablation_results.md"
info "and log anything unexpected in that file's anomaly table."
