#!/usr/bin/env bash
# =============================================================================
# FedChain :: finish the study
# -----------------------------------------------------------------------------
# Runs everything still outstanding after the clean qwen-0.5b sweep, in
# dependency order, unattended. Launch it and walk away.
#
#     bash scripts/finish_study.sh
#
# What it runs, and why each is here (see ablation_study/11_final_status.md):
#
#   1. E6 @ 50 trials + E7 to N=100 for qwen-0.5b       ~minutes
#      Protocol parity with the 360M tier. Tightens the false-positive bound
#      from 13.9% to 5.8%. Needs a live chain; everything else does not.
#
#   2. Ablation B1 - E0/E1/E2 on Dirichlet(0.3)          ~12 h
#      THE blocking run. E5 is Dirichlet-sharded while E0/E1/E2 are IID, so
#      every E5-E0 difference confounds partition with federation and no
#      non-IID *learning* claim is licensed. B1 supplies matched baselines.
#
#   3. Re-run the 360M E0 arm                            ~3.3 h
#      That tier predates the C1 fix and still reports 299.202 MiB of phantom
#      communication for an arm with no server. The bytes were never measured,
#      so the only correction is a re-run.
#
#   4. Ablation E - re-score at 250 generation samples   ~6 h, no retraining
#      All 36 runs used 50 samples and the built-in ROUGE backend, so the
#      generation rows support no between-arm claim. This re-scores finished
#      adapters; it does not retrain.
#
#   5. Regenerate every comparison table                 ~seconds
#
# Steps are independent and idempotent: a completed step is skipped unless
# --force, and a failing step does not abort the ones after it (so an overnight
# run still delivers whatever it can). The summary at the end reports each.
#
# Flags:
#   --model TIER      primary tier (default: qwen-0.5b)
#   --seeds "A B C"   seeds (default: 42 43 44)
#   --only STEP[,..]  run only these: audit,b1,e0-360m,reeval,tables
#   --skip STEP[,..]  skip these
#   --force           re-run steps whose outputs already exist
#   --dry-run         print the commands without running them
# =============================================================================
set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

MODEL="qwen-0.5b"
SEEDS="42 43 44"
BASELINE_MODEL="smollm2-360m"
ONLY=""
SKIP=""
FORCE=0
DRY_RUN=0
RPC_URL="${RPC_URL:-http://127.0.0.1:8545}"
LOG_DIR="results/logs"

C_STEP='\033[1;36m'; C_OK='\033[0;32m'; C_WARN='\033[0;33m'; C_FAIL='\033[0;31m'; C_OFF='\033[0m'
step() { printf "\n${C_STEP}==> %s${C_OFF}\n" "$*"; }
info() { printf '    %s\n' "$*"; }
ok()   { printf "    ${C_OK}[ok]${C_OFF} %s\n" "$*"; }
warn() { printf "    ${C_WARN}[warn]${C_OFF} %s\n" "$*"; }
fail() { printf "    ${C_FAIL}[fail]${C_OFF} %s\n" "$*" >&2; }

while [[ $# -gt 0 ]]; do
    case "$1" in
        --model)   MODEL="$2"; shift 2 ;;
        --seeds)   SEEDS="$2"; shift 2 ;;
        --only)    ONLY="$2"; shift 2 ;;
        --skip)    SKIP="$2"; shift 2 ;;
        --force)   FORCE=1; shift ;;
        --dry-run) DRY_RUN=1; shift ;;
        -h|--help) sed -n '2,44p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; exit 0 ;;
        *) fail "unknown argument: $1"; exit 2 ;;
    esac
done

wants() {
    local name="$1"
    [[ -n "$ONLY" && ",$ONLY," != *",$name,"* ]] && return 1
    [[ -n "$SKIP" && ",$SKIP," == *",$name,"* ]] && return 1
    return 0
}

mkdir -p "$LOG_DIR"
declare -a RESULTS=()
record() { RESULTS+=("$1|$2|$3"); }

# `run` executes a step, tees to a log, and times it. Never aborts the script:
# an overnight run should deliver every step it can, not stop at the first
# failure and waste the remaining hours.
run() {
    local name="$1" label="$2"; shift 2
    local log="$LOG_DIR/finish_${name}.log"
    info "-> $*"
    if [[ $DRY_RUN -eq 1 ]]; then record "$name" "dry-run" "$label"; return 0; fi
    local start; start=$(date +%s)
    if "$@" > >(tee "$log") 2>&1; then
        local secs=$(( $(date +%s) - start ))
        ok "$label done in $((secs / 60))m $((secs % 60))s  (log: $log)"
        record "$name" "done" "$label"
    else
        local secs=$(( $(date +%s) - start ))
        fail "$label FAILED after $((secs / 60))m  (log: $log)"
        record "$name" "FAILED" "$label"
    fi
}

# =============================================================================
# 0. Environment
# =============================================================================
step "[0/5] Environment"

if [[ -f ".venv/bin/activate" ]]; then
    # shellcheck disable=SC1091
    source .venv/bin/activate
    ok "activated .venv"
else
    warn "no .venv - using the ambient python3"
fi
PYTHON="$(command -v python || command -v python3)"
info "interpreter: $PYTHON"

# The metric stack is checked here rather than at first use because a fallback
# to the built-in ROUGE is silent and cost the baseline sweep all 36 runs'
# generation metrics. run_all.sh and run_ablation.sh carry the same guard.
# Skipped under --dry-run, which exists to rehearse the plan on any machine.
if [[ $DRY_RUN -eq 1 ]]; then
    warn "--dry-run: skipping the metric-stack check"
elif ! "$PYTHON" - <<'PY'
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
    fail "base_config.yaml requires the \`evaluate\` backend but it is unusable."
    fail "Install into the interpreter above:"
    fail "    source .venv/bin/activate && python -m pip install -U nltk rouge-score evaluate"
    exit 1
fi
ok "generation-metric stack ready"

# =============================================================================
# 1. Audit-layer parity (needs a live chain)
# =============================================================================
if wants audit; then
    step "[1/5] E6 tamper @ 50 trials + E7 scalability to N=100 ($MODEL)"

    rpc_up() {
        curl -s -m 3 -X POST -H 'Content-Type: application/json' \
            --data '{"jsonrpc":"2.0","method":"eth_chainId","params":[],"id":1}' \
            "$RPC_URL" 2>/dev/null | grep -q '"result"'
    }

    if ! rpc_up && command -v anvil >/dev/null; then
        info "starting anvil -> $LOG_DIR/anvil.log"
        nohup anvil --host 127.0.0.1 --port 8545 > "$LOG_DIR/anvil.log" 2>&1 &
        for _ in $(seq 1 40); do rpc_up && break; sleep 1; done
    fi

    if rpc_up; then
        ok "chain live at $RPC_URL"
        ADAPTER_ROOT=""
        for seed in $SEEDS; do
            candidate="outputs/$MODEL/seed_$seed/exp4_fedchain"
            [[ -d "$candidate" ]] && { ADAPTER_ROOT="$candidate"; break; }
        done
        if [[ -z "$ADAPTER_ROOT" ]]; then
            warn "no completed exp4 adapters under outputs/$MODEL/seed_*/exp4_fedchain"
            warn "E6 would fall back to synthetic adapters - skipping rather than"
            warn "producing a weaker result than the one already on disk."
            record audit "skipped" "E6/E7 (no real adapters found)"
        else
            info "E6 adapter source: $ADAPTER_ROOT"
            run audit_e6 "E6 tamper @ 50 trials" \
                "$PYTHON" scripts/tamper_experiment.py \
                    --adapter-root "$ADAPTER_ROOT" \
                    --trials 50 \
                    --results-dir "results/$MODEL"
            run audit_e7 "E7 scalability to N=100" \
                "$PYTHON" scripts/scalability_experiment.py \
                    --clients "1,3,5,10,25,50,100" \
                    --results-dir "results/$MODEL"
        fi
    else
        warn "no chain at $RPC_URL and anvil not available - skipping E6/E7."
        warn "These take minutes; run them separately once a node is up."
        record audit "skipped" "E6/E7 (no chain)"
    fi
else
    step "[1/5] Audit-layer parity (skipped)"
fi

# =============================================================================
# 2. Ablation B1 - the blocking run
# =============================================================================
if wants b1; then
    step "[2/5] Ablation B1: E0/E1/E2 on Dirichlet(0.3), $MODEL, seeds $SEEDS"

    if [[ ! -f "data/dirichlet/client1.jsonl" ]]; then
        warn "data/dirichlet/ shards are missing. Generate them first:"
        warn "    $PYTHON data/prepare_data.py --partition dirichlet --alpha 0.3"
        record b1 "skipped" "B1 (no Dirichlet shards)"
    else
        # The smallest Dirichlet(0.3) shard caps the round count. B1 runs at
        # R=3 to match E5, which needs 1,500 records; verify rather than trust.
        SMALLEST=$(wc -l < data/dirichlet/client2.jsonl)
        for f in data/dirichlet/client*.jsonl; do
            n=$(wc -l < "$f"); [[ $n -lt $SMALLEST ]] && SMALLEST=$n
        done
        info "smallest Dirichlet shard: $SMALLEST records (R=3 needs 1500)"
        if [[ $SMALLEST -lt 1500 ]]; then
            fail "smallest shard cannot support 3 rounds at 500 samples/round."
            fail "Lower max_train_samples or regenerate with a higher alpha."
            record b1 "FAILED" "B1 (shard too small)"
        else
            DONE_MARKER="results/$MODEL/ablation/seed_${SEEDS##* }/ablationB_e2_noniid_metrics.json"
            if [[ -f "$DONE_MARKER" && $FORCE -eq 0 ]]; then
                ok "B1 already complete ($DONE_MARKER) - use --force to re-run"
                record b1 "skipped" "B1 (already complete)"
            else
                run b1 "Ablation B1" \
                    bash ablation_study/run_ablation.sh \
                        --block B1 --model "$MODEL" --seeds "$SEEDS"
            fi
        fi
    fi
else
    step "[2/5] Ablation B1 (skipped)"
fi

# =============================================================================
# 3. Re-run the 360M local-only arm
# =============================================================================
if wants e0-360m; then
    step "[3/5] Re-run E0 on $BASELINE_MODEL (clears the 299.202 MiB artefact)"

    NEEDS_RERUN=0
    for seed in $SEEDS; do
        report="results/$BASELINE_MODEL/seed_$seed/exp0_local_metrics.json"
        if [[ -f "$report" ]] && ! grep -q '"communication_counted"' "$report"; then
            NEEDS_RERUN=1
        fi
    done

    if [[ $NEEDS_RERUN -eq 0 && $FORCE -eq 0 ]]; then
        ok "360M E0 already carries post-C1 instrumentation - nothing to do"
        record e0-360m "skipped" "360M E0 (already clean)"
    else
        run e0_360m "360M E0 re-run" \
            bash run_all.sh --model "$BASELINE_MODEL" --seeds "$SEEDS" \
                --experiments "0" --force --skip-smoke --skip-compare
    fi
else
    step "[3/5] 360M E0 re-run (skipped)"
fi

# =============================================================================
# 4. Ablation E - generation metrics at a defensible sample count
# =============================================================================
if wants reeval; then
    step "[4/5] Ablation E: re-score finished adapters @ 250 generation samples"

    for tier in "$MODEL" "$BASELINE_MODEL"; do
        [[ -d "results/$tier" ]] || { warn "no results/$tier - skipping"; continue; }
        out="results/$tier/reeval250"
        if [[ -d "$out" && $FORCE -eq 0 ]]; then
            ok "$tier already re-scored ($out) - use --force to redo"
            record "reeval_$tier" "skipped" "re-eval $tier (already done)"
            continue
        fi
        run "reeval_$tier" "re-eval $tier @250" \
            "$PYTHON" scripts/reevaluate.py \
                --sweep "results/$tier" \
                --model "$tier" \
                --gen-num-samples 250 \
                --require-backend evaluate \
                --out "$out"
    done
else
    step "[4/5] Ablation E (skipped)"
fi

# =============================================================================
# 5. Tables
# =============================================================================
if wants tables; then
    step "[5/5] Regenerating comparison tables"

    for tier in "$MODEL" "$BASELINE_MODEL"; do
        [[ -d "results/$tier" ]] || continue
        run "table_$tier" "per-tier table: $tier" \
            "$PYTHON" scripts/compare_results.py --results-dir "results/$tier" --seeds
    done

    # B1 pairs against its own arms: the whole point is a non-IID comparison
    # against non-IID baselines, never against the IID ones.
    if [[ -d "results/$MODEL/ablation" ]]; then
        run table_b1 "B1 paired table (non-IID vs non-IID)" \
            "$PYTHON" scripts/compare_results.py \
                --results-dir "results/$MODEL/ablation" --seeds \
                --order ablationB_e0_noniid,ablationB_e1_noniid,ablationB_e2_noniid \
                --baseline ablationB_e1_noniid \
                --extra-baselines ablationB_e0_noniid
    fi

    run table_ladder "model-ladder table" \
        "$PYTHON" scripts/compare_results.py --results-dir results --across-models
else
    step "[5/5] Tables (skipped)"
fi

# =============================================================================
# Summary
# =============================================================================
printf "\n%s\n" "======================================================================="
printf " FINISH-STUDY SUMMARY\n"
printf "%s\n" "======================================================================="
FAILED=0
for entry in "${RESULTS[@]+"${RESULTS[@]}"}"; do
    IFS='|' read -r name state label <<< "$entry"
    case "$state" in
        done)    printf "  ${C_OK}%-9s${C_OFF} %s\n" "done" "$label" ;;
        skipped) printf "  ${C_WARN}%-9s${C_OFF} %s\n" "skipped" "$label" ;;
        dry-run) printf "  %-9s %s\n" "dry-run" "$label" ;;
        *)       printf "  ${C_FAIL}%-9s${C_OFF} %s\n" "FAILED" "$label"; FAILED=1 ;;
    esac
done
[[ ${#RESULTS[@]} -eq 0 ]] && printf "  (nothing ran)\n"

cat <<EOF

 key artefacts:
   results/comparison_across_models.md            model-ladder table
   results/$MODEL/comparison.md                   per-tier, mean +- 95% CI
   results/$MODEL/ablation/comparison.md          B1: the non-IID paired result
   results/$MODEL/reeval250/                      generation metrics @250 samples
   $LOG_DIR/finish_*.log                          per-step transcripts

 next: read the B1 paired-difference table. Its E2-E0 row on matched Dirichlet
 shards is the number that converts E5 from a systems result into a learning
 one. Also diff B1-E2's adapter hashes against E5's - the configs are identical
 except the chain/IPFS flags, so they must match.
EOF

exit $FAILED
