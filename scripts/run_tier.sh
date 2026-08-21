#!/usr/bin/env bash
# =============================================================================
# FedChain :: one model tier, start to paper artefacts
# -----------------------------------------------------------------------------
# Takes a single rung of the model ladder from nothing to the point where its
# rows appear in every table and figure the paper draws on, unattended:
#
#     bash scripts/run_tier.sh --model llama-3.2-1b --seeds "42 43 44"
#
# This exists because "run the pipeline for a new model" is six commands whose
# ORDER and PARAMETERS matter, and getting either wrong produces a tier that
# looks finished and is not comparable with the tiers beside it:
#
#   1. sweep    E0-E5 x seeds                       ~14-20 h at 1B on a T600
#      The training arms. E5 needs the Dirichlet shards, so they are checked
#      before anything starts rather than 12 hours in.
#
#   2. b1       Ablation B1: E0/E1/E2 @ alpha=0.3   ~12-16 h at 1B
#      NOT optional. E5 is Dirichlet-sharded while E0/E1/E2 are IID, so on its
#      own E5 licenses no LEARNING claim - every E5-E0 gap would confound the
#      partition with federation. B1 supplies the matched non-IID baselines and
#      is what makes the 2x2 over (scale, partition) complete for this tier.
#
#   3. audit    E6 @ 50 trials, E7 to N=100         ~minutes, needs a chain
#      Deliberately NOT run via `run_all.sh --audit-experiments`, which uses 20
#      trials and stops at N=50. Both existing tiers were re-run at 50/100, and
#      a 20-trial false-positive bound (13.9%) tabulated beside a 50-trial one
#      (5.8%) is a protocol difference reported as a result.
#
#   4. reeval   re-score every arm @ 250 samples    ~6-8 h, no retraining
#      The per-run ROUGE-L/BLEU in the sweep may come from either scorer. Only
#      reeval250 is single-scorer, and it is the ONLY source paper_tables reads
#      for generation metrics. Without this step the tier's generation rows are
#      absent from the tables rather than wrong - which is the better failure,
#      but still a missing row.
#
#   5. tables   per-tier, B1 paired, cross-tier     ~seconds
#   6. paper    paper_tables.py + paper_figures.py  ~seconds
#
# Steps are independent and idempotent: a completed step is skipped unless
# --force, and a failing step does not abort the ones after it, so an overnight
# run delivers everything it can. The summary at the end reports each one.
#
# PREREQUISITE: ./infra.sh (anvil + IPFS + contract artifact). Steps 1 and 3
# need a live chain; 2, 4, 5 and 6 do not.
#
# Flags:
#   --model TIER      tier key, alias or Hugging Face id  (required)
#   --seeds "A B C"   seeds (default: 42 43 44)
#   --only STEP[,..]  run only these: sweep,b1,audit,reeval,tables,paper
#   --skip STEP[,..]  skip these
#   --force           re-run steps whose outputs already exist
#   --dry-run         print the plan without running anything
# =============================================================================
set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

MODEL=""
SEEDS="42 43 44"
ONLY=""
SKIP=""
FORCE=0
DRY_RUN=0
RPC_URL="${RPC_URL:-http://127.0.0.1:8545}"
LOG_DIR="results/logs"
ALPHA="0.3"

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
        -h|--help) sed -n '2,55p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; exit 0 ;;
        *) fail "unknown argument: $1"; exit 2 ;;
    esac
done

[[ -z "$MODEL" ]] && { fail "--model is required"; exit 2; }

wants() {
    local name="$1"
    [[ -n "$ONLY" && ",$ONLY," != *",$name,"* ]] && return 1
    [[ -n "$SKIP" && ",$SKIP," == *",$name,"* ]] && return 1
    return 0
}

mkdir -p "$LOG_DIR"
declare -a RESULTS=()
record() { RESULTS+=("$1|$2|$3"); }

# `run` executes a step, tees to a log and times it. It never aborts the script:
# an overnight run should deliver every step it can rather than stop at the
# first failure and waste the remaining hours.
run() {
    local name="$1" label="$2"; shift 2
    local log="$LOG_DIR/tier_${name}.log"
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
step "[0/6] Environment"

if [[ -f ".venv/bin/activate" ]]; then
    # shellcheck disable=SC1091
    source .venv/bin/activate
    ok "activated .venv"
else
    warn "no .venv - using the ambient python"
fi
PYTHON="$(command -v python || command -v python3)"
info "interpreter: $PYTHON"

# -- resolve the alias to its canonical tier key ------------------------------
# Every path below is derived from $TIER. main.py accepts aliases, but a run
# launched as --model llama writes to results/llama-3.2-1b/ while a path built
# from the alias points at results/llama/. That split is silent and only shows
# up later as a comparison table with half the arms missing.
if RESOLVED="$("$PYTHON" utils/models.py --resolve "$MODEL" 2>/dev/null | cut -f1)" \
   && [[ -n "$RESOLVED" ]]; then
    [[ "$RESOLVED" != "$MODEL" ]] && info "resolved '$MODEL' -> '$RESOLVED'"
    TIER="$RESOLVED"
else
    fail "could not resolve --model '$MODEL' to a tier key"
    "$PYTHON" utils/models.py --list
    exit 2
fi
HF_ID="$("$PYTHON" utils/models.py --resolve "$MODEL" 2>/dev/null | cut -f2)"
info "tier: $TIER  ($HF_ID)"
info "seeds: $SEEDS"

# -- gated-repository preflight ----------------------------------------------
if [[ -n "$("$PYTHON" utils/models.py --gated-list "$MODEL" 2>/dev/null || true)" \
      && -z "${HF_TOKEN:-${HUGGING_FACE_HUB_TOKEN:-}}" ]]; then
    fail "$HF_ID is a gated repository and no HF_TOKEN is exported."
    fail "Accept the licence at https://huggingface.co/$HF_ID with the same"
    fail "account, then:  export HF_TOKEN=hf_..."
    exit 2
fi

# -- metric-stack preflight ---------------------------------------------------
# A fallback to the built-in ROUGE is silent, and it cost the baseline sweep all
# 36 runs' generation metrics. run_all.sh and run_ablation.sh carry this same
# guard; it is repeated because --only reeval reaches neither of them.
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
    sys.exit(1)
PY
then
    fail "base_config.yaml requires the \`evaluate\` backend but it is unusable."
    fail "    python -m pip install -U nltk rouge-score evaluate"
    exit 1
fi
ok "generation-metric stack ready"

# -- Dirichlet shard guard ----------------------------------------------------
# Steps 1 (E5) and 2 (B1) both read data/dirichlet/. prepare_data.py writes
# EVERY alpha to that same directory, so a B2/B3 run at alpha=0.1 leaves shards
# here that are not the ones the other tiers were measured on - and nothing
# downstream would notice. The manifest is committed, so check against it.
check_shards() {
    local manifest="data/dirichlet/manifest.json"
    [[ -f "$manifest" ]] || { warn "no $manifest"; return 1; }
    "$PYTHON" - "$manifest" "$ALPHA" <<'PY'
import json, sys
from pathlib import Path

manifest, want_alpha = Path(sys.argv[1]), float(sys.argv[2])
meta = json.loads(manifest.read_text(encoding="utf-8"))
alpha = float(meta.get("dirichlet_alpha", -1))
if abs(alpha - want_alpha) > 1e-9:
    print(f"data/dirichlet/ holds alpha={alpha}, not {want_alpha}")
    sys.exit(1)
# Record counts, not just presence: a regeneration at a different seed produces
# the right alpha and the wrong shards, and B1 would then not be comparable
# with the B1 already run at the other tiers.
for entry in meta.get("partition_profile", []):
    shard = manifest.parent / f"{entry['client']}.jsonl"
    if not shard.is_file():
        print(f"missing shard {shard}")
        sys.exit(1)
    with open(shard, encoding="utf-8") as fh:
        lines = sum(1 for _ in fh)
    if lines != int(entry["num_records"]):
        print(f"{shard}: {lines} records, manifest says {entry['num_records']}")
        sys.exit(1)
smallest = min(int(e["num_records"]) for e in meta.get("partition_profile", [{"num_records": 0}]))
print(f"alpha={alpha}, smallest shard {smallest} records "
      f"({'ok' if smallest >= 1500 else 'TOO SMALL'} for 3 rounds at 500/round)")
sys.exit(0 if smallest >= 1500 else 1)
PY
}

NONIID_READY=0
if [[ $DRY_RUN -eq 1 ]]; then
    NONIID_READY=1
elif SHARD_MSG="$(check_shards)"; then
    ok "Dirichlet shards: $SHARD_MSG"
    NONIID_READY=1
else
    warn "Dirichlet shards unusable: ${SHARD_MSG:-not present}"
    info "generating Dirichlet(alpha=$ALPHA) shards"
    if "$PYTHON" data/prepare_data.py --partition dirichlet --alpha "$ALPHA" \
       && SHARD_MSG="$(check_shards)"; then
        ok "Dirichlet shards: $SHARD_MSG"
        NONIID_READY=1
    else
        fail "could not produce usable Dirichlet(alpha=$ALPHA) shards."
        fail "E5 and B1 will be skipped; the tier would carry a systems result only."
    fi
fi

# =============================================================================
# 1. Training sweep: E0-E5 across seeds
# =============================================================================
if wants sweep; then
    step "[1/6] Sweep: E0-E5, $TIER, seeds $SEEDS"

    if [[ $NONIID_READY -eq 1 ]]; then
        EXPERIMENTS="0 1 2 3 4 5"
    else
        EXPERIMENTS="0 1 2 3 4"
        warn "running without E5 - no Dirichlet shards"
    fi

    # The last seed's E5 (or E4) report is the completion marker: run_all.sh
    # walks seeds in order, so if it exists the sweep reached the end.
    LAST_SEED="${SEEDS##* }"
    LAST_EXP="exp5_noniid"; [[ $NONIID_READY -eq 1 ]] || LAST_EXP="exp4_fedchain"
    MARKER="results/$TIER/seed_$LAST_SEED/${LAST_EXP}_metrics.json"

    if [[ -f "$MARKER" && $FORCE -eq 0 ]]; then
        ok "sweep already complete ($MARKER) - use --force to re-run"
        record sweep "skipped" "sweep (already complete)"
    else
        SWEEP_ARGS=(--model "$TIER" --seeds "$SEEDS" --experiments "$EXPERIMENTS")
        [[ $NONIID_READY -eq 1 ]] && SWEEP_ARGS+=(--noniid "$ALPHA")
        [[ $FORCE -eq 1 ]] && SWEEP_ARGS+=(--force)
        # --audit-experiments is deliberately absent: step 3 runs E6/E7 at the
        # 50-trial / N=100 protocol the other tiers use.
        run sweep "E0-E5 sweep" bash run_all.sh "${SWEEP_ARGS[@]}"
    fi
else
    step "[1/6] Sweep (skipped)"
fi

# =============================================================================
# 2. Ablation B1: the matched non-IID baselines
# =============================================================================
if wants b1; then
    step "[2/6] Ablation B1: E0/E1/E2 on Dirichlet($ALPHA), $TIER, seeds $SEEDS"

    if [[ $NONIID_READY -eq 0 ]]; then
        fail "no usable Dirichlet shards - B1 cannot run"
        record b1 "FAILED" "B1 (no Dirichlet shards)"
    else
        MARKER="results/$TIER/ablation/seed_${SEEDS##* }/ablationB_e2_noniid_metrics.json"
        if [[ -f "$MARKER" && $FORCE -eq 0 ]]; then
            ok "B1 already complete ($MARKER) - use --force to re-run"
            record b1 "skipped" "B1 (already complete)"
        else
            run b1 "Ablation B1" \
                bash ablation_study/run_ablation.sh \
                    --block B1 --model "$TIER" --seeds "$SEEDS"
        fi
    fi
else
    step "[2/6] Ablation B1 (skipped)"
fi

# =============================================================================
# 3. Audit layer at the reported protocol
# =============================================================================
if wants audit; then
    step "[3/6] E6 tamper @ 50 trials + E7 scalability to N=100 ($TIER)"

    rpc_up() {
        curl -s -m 3 -X POST -H 'Content-Type: application/json' \
            --data '{"jsonrpc":"2.0","method":"eth_chainId","params":[],"id":1}' \
            "$RPC_URL" 2>/dev/null | grep -q '"result"'
    }

    if ! rpc_up && command -v anvil >/dev/null 2>&1; then
        info "starting anvil -> $LOG_DIR/anvil.log"
        nohup anvil --host 127.0.0.1 --port 8545 > "$LOG_DIR/anvil.log" 2>&1 &
        for _ in $(seq 1 40); do rpc_up && break; sleep 1; done
    fi

    if [[ $DRY_RUN -eq 0 ]] && ! rpc_up; then
        warn "no chain at $RPC_URL and anvil is unavailable - skipping E6/E7."
        warn "They take minutes; re-run with --only audit once a node is up."
        record audit "skipped" "E6/E7 (no chain)"
    else
        # E6 must perturb REAL adapters from this tier. Falling back to
        # synthetic ones would measure the hash function, not this tier's
        # artefacts, and the result would not be comparable with the tiers
        # beside it - so it is skipped rather than silently downgraded.
        ADAPTER_ROOT=""
        for seed in $SEEDS; do
            candidate="outputs/$TIER/seed_$seed/exp4_fedchain"
            [[ -d "$candidate" ]] && { ADAPTER_ROOT="$candidate"; break; }
        done

        if [[ -z "$ADAPTER_ROOT" && $DRY_RUN -eq 0 ]]; then
            warn "no adapters under outputs/$TIER/seed_*/exp4_fedchain"
            warn "E6 would have to use synthetic adapters - skipping instead."
            record audit "skipped" "E6 (no real adapters)"
        else
            # Under --dry-run there is nothing on disk to find, so show the
            # path the real run would use rather than an empty argument.
            [[ -z "$ADAPTER_ROOT" ]] && ADAPTER_ROOT="outputs/$TIER/seed_${SEEDS%% *}/exp4_fedchain"
            info "E6 adapter source: $ADAPTER_ROOT"
            E6_MARKER="results/$TIER/exp6_tamper_metrics.json"
            if [[ -f "$E6_MARKER" && $FORCE -eq 0 ]]; then
                ok "E6 already present - use --force to re-run"
                record audit_e6 "skipped" "E6 (already complete)"
            else
                run audit_e6 "E6 tamper @ 50 trials" \
                    "$PYTHON" scripts/tamper_experiment.py \
                        --adapter-root "$ADAPTER_ROOT" \
                        --trials 50 \
                        --results-dir "results/$TIER"
            fi
        fi

        E7_MARKER="results/$TIER/exp7_scalability_metrics.json"
        if [[ -f "$E7_MARKER" && $FORCE -eq 0 ]]; then
            ok "E7 already present - use --force to re-run"
            record audit_e7 "skipped" "E7 (already complete)"
        else
            run audit_e7 "E7 scalability to N=100" \
                "$PYTHON" scripts/scalability_experiment.py \
                    --clients "1,3,5,10,25,50,100" \
                    --results-dir "results/$TIER"
        fi
    fi
else
    step "[3/6] Audit layer (skipped)"
fi

# =============================================================================
# 4. Re-score generation metrics on one scorer
# =============================================================================
if wants reeval; then
    step "[4/6] Re-score every arm @ 250 generation samples ($TIER)"

    OUT="results/$TIER/reeval250"
    if [[ -e "$OUT" && $FORCE -eq 0 ]]; then
        ok "$OUT already present - use --force to redo"
        record reeval "skipped" "reeval250 (already done)"
    elif [[ ! -d "results/$TIER" && $DRY_RUN -eq 0 ]]; then
        warn "no results/$TIER - nothing to re-score"
        record reeval "skipped" "reeval250 (no results)"
    else
        # One --sweep pass covers the main arms AND the ablation arms, which is
        # the point: paper_tables reads generation metrics from here only, and a
        # table that mixed a reeval'd IID row with a per-run non-IID row would
        # be comparing two scorers down a single column.
        run reeval "re-eval @250" \
            "$PYTHON" scripts/reevaluate.py \
                --sweep "results/$TIER" \
                --model "$TIER" \
                --gen-num-samples 250 \
                --require-backend evaluate \
                --out "$OUT"
    fi
else
    step "[4/6] Re-score @250 (skipped)"
fi

# =============================================================================
# 5. Comparison tables
# =============================================================================
if wants tables; then
    step "[5/6] Comparison tables"

    if [[ -d "results/$TIER" || $DRY_RUN -eq 1 ]]; then
        run "table_tier" "per-tier table: $TIER" \
            "$PYTHON" scripts/compare_results.py --results-dir "results/$TIER" --seeds
    fi

    # B1 pairs against its OWN arms. The whole point is a non-IID comparison
    # against non-IID baselines, never against the IID ones.
    if [[ -d "results/$TIER/ablation" || $DRY_RUN -eq 1 ]]; then
        run table_b1 "B1 paired table (non-IID vs non-IID)" \
            "$PYTHON" scripts/compare_results.py \
                --results-dir "results/$TIER/ablation" --seeds \
                --order ablationB_e0_noniid,ablationB_e1_noniid,ablationB_e2_noniid \
                --baseline ablationB_e1_noniid \
                --extra-baselines ablationB_e0_noniid
    fi

    run table_ladder "model-ladder table" \
        "$PYTHON" scripts/compare_results.py --results-dir results --across-models
else
    step "[5/6] Tables (skipped)"
fi

# =============================================================================
# 6. Paper tables and figures
# =============================================================================
if wants paper; then
    step "[6/6] Paper tables and figures"

    # Both discover their tiers from results/, so this tier joins the existing
    # ones with no edit anywhere. --check is advisory here: it exits non-zero
    # while any paper-blocking cell is still missing, which is the normal state
    # until every step above has finished at every tier.
    run paper_tables "paper tables" \
        "$PYTHON" scripts/paper_tables.py --seeds "$SEEDS"
    run paper_figures "paper figures" \
        "$PYTHON" scripts/paper_figures.py
else
    step "[6/6] Paper artefacts (skipped)"
fi

# =============================================================================
# Summary
# =============================================================================
printf "\n%s\n" "======================================================================="
printf " RUN-TIER SUMMARY :: %s\n" "$TIER"
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
   results/$TIER/comparison.md              per-tier, mean +- 95% CI
   results/$TIER/ablation/comparison.md     B1: the non-IID paired result
   results/$TIER/reeval250                  generation metrics @250, one scorer
   results/comparison_across_models.md      the ladder, all tiers
   results/paper/tables.md                  paper tables (every tier found)
   paper/figures/                           paper figures (every tier found)
   $LOG_DIR/tier_*.log                      per-step transcripts

 first thing to read: the E3/E4-vs-E2 rows in results/$TIER/comparison.md must
 show identical accuracy, and the artefact hashes must match exactly. That is a
 stronger claim than any statistical test, and it is the one the paper leads
 with. A divergence there means the audit layer corrupted an update on this
 tier and nothing else in the run matters until it is explained.

 second: the B1 paired table's E2-E0 row on matched Dirichlet shards is what
 turns this tier's E5 from a systems result into a learning one.
EOF

exit $FAILED
