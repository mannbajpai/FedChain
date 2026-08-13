#!/usr/bin/env bash
# =============================================================================
# FedChain :: the final run
# -----------------------------------------------------------------------------
# Runs the two experiments still outstanding, re-scores what they produce, and
# rebuilds every table the paper needs. Launch it and walk away.
#
#     bash scripts/run_final.sh
#
# Total: ~13 h, almost all of it step 2.
#
# WHAT IT RUNS, AND WHY EACH IS THE LAST THING MISSING
# ----------------------------------------------------
#   1. E6 @ 50 trials + E7 to N=100 at qwen-0.5b          ~minutes
#      The 0.5B tier ran E6 at 20 trials. A clean 0/20 on the benign control
#      bounds the false-positive rate at only 13.9% (one-sided 95%), so the
#      paper cannot write "0% false positives" - it has to write "<=13.9%",
#      which reads as a weak result for what is actually a perfect score. 0/50
#      bounds it at 5.8%. Needs a live chain; anvil is started automatically.
#
#   2. Ablation B1 at smollm2-360m                        ~12 h
#      B1 exists only at 0.5B, where FedAvg already recovered 34% of the
#      isolation->centralized gap under IID. At 360M it recovered 2.4%, and that
#      tier has no non-IID arm at all - so the study can currently say skew
#      *widens an open margin*, but not that it *rescues a weak one*. This run
#      completes the 2x2 (scale x skew) that Table 1 of the paper is built on.
#
#   3. Re-score the new 360M ablation adapters @250       ~2 h, no retraining
#      Generation metrics are only comparable when every arm used the same
#      scorer. Without this the new arms would carry `builtin` numbers next to
#      `evaluate` numbers - the exact defect that made the shipped ROUGE-L rows
#      unquotable.
#
#   4. Rebuild the comparison tables                      ~seconds
#   5. Build the paper tables and paper_numbers.json      ~seconds
#
# Steps are idempotent: a completed step is skipped unless --force, and a
# failure does not abort the ones after it, so an overnight run delivers
# whatever it can and reports the rest.
#
# Flags:
#   --only STEP[,..]   audit,b1,reeval,tables,paper
#   --skip STEP[,..]
#   --seeds "A B C"    default: 42 43 44
#   --force            re-run steps whose outputs already exist
#   --dry-run          print the plan without running anything
# =============================================================================
set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

AUDIT_TIER="qwen-0.5b"        # tier missing E6/E7 protocol parity
B1_TIER="smollm2-360m"        # tier missing the non-IID arm
SEEDS="42 43 44"
ONLY=""; SKIP=""; FORCE=0; DRY_RUN=0
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
        --only)    ONLY="$2"; shift 2 ;;
        --skip)    SKIP="$2"; shift 2 ;;
        --seeds)   SEEDS="$2"; shift 2 ;;
        --force)   FORCE=1; shift ;;
        --dry-run) DRY_RUN=1; shift ;;
        -h|--help) sed -n '2,52p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; exit 0 ;;
        *) fail "unknown argument: $1"; exit 2 ;;
    esac
done

wants() {
    [[ -n "$ONLY" && ",$ONLY," != *",$1,"* ]] && return 1
    [[ -n "$SKIP" && ",$SKIP," == *",$1,"* ]] && return 1
    return 0
}

mkdir -p "$LOG_DIR"
declare -a RESULTS=()
record() { RESULTS+=("$1|$2|$3"); }

run() {
    local name="$1" label="$2"; shift 2
    local log="$LOG_DIR/final_${name}.log"
    info "-> $*"
    if [[ $DRY_RUN -eq 1 ]]; then record "$name" "dry-run" "$label"; return 0; fi
    local start; start=$(date +%s)
    if "$@" > >(tee "$log") 2>&1; then
        local secs=$(( $(date +%s) - start ))
        ok "$label done in $((secs / 3600))h $(((secs % 3600) / 60))m  (log: $log)"
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

# Checked here rather than at first use. A silent fallback from `evaluate` to the
# built-in ROUGE is what made 36 runs' generation metrics unquotable; with
# require_metric_backend set it is now a hard failure, but that failure lands ~20
# minutes into the first evaluation. Cost seconds here instead.
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
    fail "    source .venv/bin/activate && python -m pip install -U nltk rouge-score evaluate"
    exit 1
fi
ok "generation-metric stack ready"

# =============================================================================
# 1. E6 @ 50 trials + E7 to N=100  (qwen-0.5b)
# =============================================================================
if wants audit; then
    step "[1/5] E6 tamper @ 50 trials + E7 to N=100 ($AUDIT_TIER)"

    rpc_up() {
        curl -s -m 3 -X POST -H 'Content-Type: application/json' \
            --data '{"jsonrpc":"2.0","method":"eth_chainId","params":[],"id":1}' \
            "$RPC_URL" 2>/dev/null | grep -q '"result"'
    }

    if [[ $DRY_RUN -eq 1 ]]; then
        info "(dry-run) would ensure a chain at $RPC_URL, then run E6/E7"
        record audit "dry-run" "E6 @ 50 + E7 to N=100"
    else
        if ! rpc_up; then
            if command -v anvil >/dev/null; then
                info "starting anvil -> $LOG_DIR/anvil.log"
                nohup anvil --host 127.0.0.1 --port 8545 > "$LOG_DIR/anvil.log" 2>&1 &
                for _ in $(seq 1 40); do rpc_up && break; sleep 1; done
            elif [[ -x ./infra.sh ]]; then
                info "anvil not on PATH - trying ./infra.sh"
                ./infra.sh > "$LOG_DIR/infra.log" 2>&1 &
                for _ in $(seq 1 60); do rpc_up && break; sleep 1; done
            fi
        fi

        if ! rpc_up; then
            fail "no chain at $RPC_URL, and neither anvil nor ./infra.sh could start one."
            fail "This step is minutes of compute and is the only one still blocking a"
            fail "stated claim (the 0.5B false-positive bound). Start a node and re-run:"
            fail "    bash scripts/run_final.sh --only audit"
            record audit "FAILED" "E6/E7 (no chain)"
        else
            ok "chain live at $RPC_URL"
            # Prefer real trained adapters. E6 on synthetic artefacts is a weaker
            # result than the one already on disk, so skip rather than overwrite.
            ADAPTER_ROOT=""
            for seed in $SEEDS; do
                cand="outputs/$AUDIT_TIER/seed_$seed/exp4_fedchain"
                [[ -d "$cand" ]] && { ADAPTER_ROOT="$cand"; break; }
            done
            if [[ -z "$ADAPTER_ROOT" ]]; then
                fail "no adapters under outputs/$AUDIT_TIER/seed_*/exp4_fedchain."
                fail "E6 would fall back to synthetic artefacts, which is weaker than"
                fail "the 20-trial result already stored. Not overwriting it."
                record audit "skipped" "E6/E7 (no real adapters)"
            else
                info "E6 adapter source: $ADAPTER_ROOT"
                run audit_e6 "E6 tamper @ 50 trials" \
                    "$PYTHON" scripts/tamper_experiment.py \
                        --adapter-root "$ADAPTER_ROOT" --trials 50 \
                        --results-dir "results/$AUDIT_TIER"
                run audit_e7 "E7 scalability to N=100" \
                    "$PYTHON" scripts/scalability_experiment.py \
                        --clients "1,3,5,10,25,50,100" \
                        --results-dir "results/$AUDIT_TIER"
            fi
        fi
    fi
else
    step "[1/5] Audit-layer parity (skipped)"
fi

# =============================================================================
# 2. Ablation B1 at 360M  -  the 2x2 completer
# =============================================================================
if wants b1; then
    step "[2/5] Ablation B1 on $B1_TIER: E0/E1/E2 @ Dirichlet(0.3), seeds $SEEDS"

    if [[ ! -f "data/dirichlet/client1.jsonl" ]]; then
        warn "data/dirichlet/ is missing; generating alpha=0.3 shards"
        if [[ $DRY_RUN -eq 0 ]]; then
            "$PYTHON" data/prepare_data.py --partition dirichlet --alpha 0.3
        fi
    fi

    if [[ ! -f "data/dirichlet/client1.jsonl" && $DRY_RUN -eq 0 ]]; then
        fail "could not obtain Dirichlet shards"
        record b1 "FAILED" "B1 @ 360M (no shards)"
    else
        # R=3 consumes 3 x 500 records from every shard. The smallest Dirichlet
        # shard is what binds, and it must be the SAME shard set the 0.5B run
        # used or the two tiers are not comparable.
        SMALLEST=999999
        if [[ -f "data/dirichlet/client1.jsonl" ]]; then
            for f in data/dirichlet/client*.jsonl; do
                n=$(wc -l < "$f"); [[ $n -lt $SMALLEST ]] && SMALLEST=$n
            done
            info "smallest Dirichlet shard: $SMALLEST records (R=3 needs 1500)"
        fi
        if [[ $SMALLEST -lt 1500 ]]; then
            fail "smallest shard cannot support 3 rounds at 500 samples/round."
            fail "Lower max_train_samples in ALL THREE B1 arms, or raise alpha."
            record b1 "FAILED" "B1 @ 360M (shard too small)"
        else
            LAST_SEED="${SEEDS##* }"
            MARKER="results/$B1_TIER/ablation/seed_${LAST_SEED}/ablationB_e2_noniid_metrics.json"
            if [[ -f "$MARKER" && $FORCE -eq 0 ]]; then
                ok "B1 @ $B1_TIER already complete - use --force to re-run"
                record b1 "skipped" "B1 @ 360M (already complete)"
            else
                # Pass the canonical tier key, never the "smol" alias:
                # run_ablation.sh uses --model for the results path as well as for
                # main.py, so an alias would scatter this run into results/smol/.
                run b1 "Ablation B1 @ $B1_TIER" \
                    bash ablation_study/run_ablation.sh \
                        --block B1 --model "$B1_TIER" --seeds "$SEEDS"
            fi
        fi
    fi
else
    step "[2/5] Ablation B1 @ 360M (skipped)"
fi

# =============================================================================
# 3. Re-score the new adapters at 250 generation samples
# =============================================================================
if wants reeval; then
    step "[3/5] Re-score $B1_TIER ablation adapters @250 (single scorer)"

    SWEEP="results/$B1_TIER/ablation"
    OUT="$SWEEP/reeval250"
    if [[ ! -d "$SWEEP" && $DRY_RUN -eq 0 ]]; then
        warn "no $SWEEP yet - step 2 must land first"
        record reeval "skipped" "re-eval 360M ablation (nothing to score)"
    elif [[ -e "$OUT" && $FORCE -eq 0 ]]; then
        ok "already re-scored ($OUT) - use --force to redo"
        record reeval "skipped" "re-eval 360M ablation (already done)"
    else
        run reeval "re-eval 360M ablation @250" \
            "$PYTHON" scripts/reevaluate.py \
                --sweep "$SWEEP" --model "$B1_TIER" \
                --gen-num-samples 250 --require-backend evaluate \
                --out "$OUT"
    fi
else
    step "[3/5] Re-scoring (skipped)"
fi

# =============================================================================
# 4. Comparison tables
# =============================================================================
if wants tables; then
    step "[4/5] Regenerating comparison tables"

    for tier in "$AUDIT_TIER" "$B1_TIER"; do
        [[ -d "results/$tier" ]] || continue
        run "table_$tier" "per-tier table: $tier" \
            "$PYTHON" scripts/compare_results.py --results-dir "results/$tier" --seeds
        # The non-IID arms pair against non-IID baselines, never the IID ones.
        if [[ -d "results/$tier/ablation" ]]; then
            run "table_ablation_$tier" "B1 paired table: $tier" \
                "$PYTHON" scripts/compare_results.py \
                    --results-dir "results/$tier/ablation" --seeds \
                    --order ablationB_e0_noniid,ablationB_e1_noniid,ablationB_e2_noniid \
                    --baseline ablationB_e1_noniid \
                    --extra-baselines ablationB_e0_noniid
        fi
    done

    run table_ladder "model-ladder table" \
        "$PYTHON" scripts/compare_results.py --results-dir results --across-models
else
    step "[4/5] Comparison tables (skipped)"
fi

# =============================================================================
# 5. Paper tables
# =============================================================================
if wants paper; then
    step "[5/5] Building paper tables"
    run paper "paper tables + paper_numbers.json" \
        "$PYTHON" scripts/paper_tables.py --results-dir results --seeds "$SEEDS"
else
    step "[5/5] Paper tables (skipped)"
fi

# =============================================================================
# Summary
# =============================================================================
printf "\n%s\n" "======================================================================="
printf " RUN-FINAL SUMMARY\n"
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

# The completeness gate. --check exits non-zero while any paper-blocking table
# still has a gap, so "did the study finish?" has a machine answer rather than
# depending on someone reading the tables carefully.
if [[ $DRY_RUN -eq 0 ]] && wants paper; then
    printf "\n"
    "$PYTHON" scripts/paper_tables.py --results-dir results --seeds "$SEEDS" \
        --check > /dev/null 2>"$LOG_DIR/final_paper_check.log"
    case $? in
        0) printf "  ${C_OK}COMPLETE${C_OFF}  every paper-blocking table is filled, and every\n"
           printf "            audited artefact is bit-identical to its un-audited twin.\n" ;;
        1) printf "  ${C_WARN}INCOMPLETE${C_OFF}  still missing:\n"
           sed -n 's/^  - /    - /p' "$LOG_DIR/final_paper_check.log"
           FAILED=1 ;;
        *) # Exit 2 is an artefact hash divergence between an audited and an
           # un-audited arm. That is the one outcome that falsifies the paper's
           # central claim, so it is called out separately from a missing run.
           printf "  ${C_FAIL}HASH DIVERGENCE${C_OFF}  the audit layer changed the trained artefacts.\n"
           printf "  ${C_FAIL}This is a bug, not a result. Do not write the paper from these tables.${C_OFF}\n"
           sed -n 's/^  - /    - /p' "$LOG_DIR/final_paper_check.log"
           FAILED=1 ;;
    esac
fi

cat <<EOF

 write the paper from these, in this order:
   results/paper/tables.md            all six tables, rendered
   results/paper/tables.tex           the same, booktabs-ready
   results/paper/paper_numbers.json   every quotable scalar, machine-readable

 supporting detail, if a reviewer asks:
   results/comparison_across_models.md              the model ladder
   results/<tier>/comparison.md                     per-tier, mean +- 95% CI
   results/<tier>/ablation/comparison.md            the non-IID paired result
   ablation_study/07_ablation_conclusions.md        what may and may not be claimed
   $LOG_DIR/final_*.log                             per-step transcripts

 do NOT quote ROUGE-L or BLEU from any comparison.md - those columns mix two
 scorers. Table 6 of results/paper/tables.md is the single-scorer version.
EOF

exit $FAILED
