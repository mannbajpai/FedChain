#!/usr/bin/env bash
# =============================================================================
# FedChain :: One-command paper run
# -----------------------------------------------------------------------------
# Takes a freshly pulled repo to a complete, publishable result set:
#
#   ./run_paper.sh
#
# Stages, in order:
#   1. archive results produced before the validity fixes (once, guarded)
#   2. verify the pulled code - unit tests + budget-matching assertions
#   3. audit-layer experiments 6 and 7 (minutes, no GPU)
#   4. main sweep: experiments 0-4 x 3 seeds
#   5. non-IID sweep: experiments 0,1,2,5 x 3 seeds
#   6. print where everything landed
#
# Stage 2 is a hard gate. If the round windows are not advancing or the
# centralized baseline is not pooling, the sweep would burn ~40 hours producing
# numbers that cannot be compared - so the script stops there rather than
# starting it.
#
# Safe to re-run. Stage 1 never fires twice, and stages 4-5 resume from
# checkpoints, so after a crash, an OOM kill or a reboot just run it again.
#
# Run `./run_paper.sh --help` for options.
# =============================================================================
set -Eeuo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"

# --- defaults ----------------------------------------------------------------
MODEL="smol"
SEEDS="42 43 44"
ALPHA="0.3"
MAIN_EXPERIMENTS="0 1 2 3 4"
NONIID_EXPERIMENTS="0 1 2 5"
TAMPER_TRIALS=50
CLIENT_SWEEP="1,3,5,10,25,50,100"

DO_ARCHIVE=1
DO_VERIFY=1
DO_AUDIT=1
DO_MAIN=1
DO_NONIID=1
ASSUME_YES=0

ARCHIVE_MARKER="$REPO_ROOT/results/.preflight_archived"
LOG_DIR="$REPO_ROOT/results/logs"

# --- pretty output -----------------------------------------------------------
if [[ -t 1 ]]; then
    C_RESET=$'\033[0m'; C_BOLD=$'\033[1m'; C_RED=$'\033[31m'
    C_GREEN=$'\033[32m'; C_YELLOW=$'\033[33m'; C_BLUE=$'\033[34m'
else
    C_RESET=""; C_BOLD=""; C_RED=""; C_GREEN=""; C_YELLOW=""; C_BLUE=""
fi
stage() { printf '\n%s========================================================================%s\n' "$C_BOLD$C_BLUE" "$C_RESET"
          printf '%s STAGE %s%s\n' "$C_BOLD$C_BLUE" "$*" "$C_RESET"
          printf '%s========================================================================%s\n' "$C_BOLD$C_BLUE" "$C_RESET"; }
info()  { printf '    %s\n' "$*"; }
ok()    { printf '    %s[ok]%s %s\n'   "$C_GREEN"  "$C_RESET" "$*"; }
warn()  { printf '    %s[warn]%s %s\n' "$C_YELLOW" "$C_RESET" "$*"; }
fail()  { printf '    %s[fail]%s %s\n' "$C_RED"    "$C_RESET" "$*"; }
die()   { fail "$*"; exit 1; }

usage() {
    sed -n '2,25p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
    cat <<'EOF'

Options:
  --model TIER        Model tier (default: smol = SmolLM2-360M).
                      Use qwen-1.5b for the paper configuration.
  --seeds "42 43 44"  Seeds to sweep (default: 42 43 44). Fewer seeds widens
                      the confidence intervals; one seed cannot support an
                      accuracy claim at all.
  --alpha 0.3         Dirichlet concentration for the non-IID shards.
  --skip-archive      Do not move pre-fix results out of the way.
  --skip-verify       Skip the validity gate. Not recommended.
  --skip-audit        Skip experiments 6 and 7.
  --skip-main         Skip the IID sweep (experiments 0-4).
  --skip-noniid       Skip the non-IID sweep (experiment 5 and its baselines).
  -y, --yes           Do not prompt before the long GPU stages.
  -h, --help          This message.

Examples:
  ./run_paper.sh                          # everything, ~40 h on a T600
  ./run_paper.sh --seeds "42 43"          # ~27 h, wider intervals
  ./run_paper.sh --skip-main --skip-noniid  # just verify + audit, ~10 min
  ./run_paper.sh --model qwen-1.5b -y     # the paper tier, unattended
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --model)        MODEL="$2"; shift 2 ;;
        --seeds)        SEEDS="$2"; shift 2 ;;
        --alpha)        ALPHA="$2"; shift 2 ;;
        --skip-archive) DO_ARCHIVE=0; shift ;;
        --skip-verify)  DO_VERIFY=0;  shift ;;
        --skip-audit)   DO_AUDIT=0;   shift ;;
        --skip-main)    DO_MAIN=0;    shift ;;
        --skip-noniid)  DO_NONIID=0;  shift ;;
        -y|--yes)       ASSUME_YES=1; shift ;;
        -h|--help)      usage; exit 0 ;;
        *)              die "unknown option: $1 (try --help)" ;;
    esac
done

mkdir -p "$LOG_DIR"
RUN_STARTED=$SECONDS

# --- python ------------------------------------------------------------------
# Activate the venv run_all.sh builds, if it exists. Candidates are probed by
# actually executing them: `python3` resolves to a do-nothing shim on some
# systems, so `command -v` alone is not evidence that an interpreter works.
for activate in "$REPO_ROOT/.venv/bin/activate" "$REPO_ROOT/venv/bin/activate"; do
    if [[ -f "$activate" ]]; then
        # shellcheck disable=SC1091
        source "$activate"
        break
    fi
done

PYTHON=""
for candidate in python python3; do
    if "$candidate" -c 'import sys; sys.exit(0)' >/dev/null 2>&1; then
        PYTHON="$candidate"
        break
    fi
done
[[ -n "$PYTHON" ]] || die "no working python interpreter found (tried: python, python3)"

# Resolve the tier key so paths match what run_all.sh writes.
if ! MODEL_LINES="$("$PYTHON" utils/models.py --resolve-list "$MODEL" 2>&1)"; then
    echo "$MODEL_LINES"
    die "could not resolve --model '$MODEL'"
fi
MODEL_KEY="$(head -1 <<< "$MODEL_LINES" | cut -f1)"
[[ -n "$MODEL_KEY" ]] || die "could not resolve --model '$MODEL' (empty result)"
TIER_RESULTS="$REPO_ROOT/results/$MODEL_KEY"
TIER_OUTPUTS="$REPO_ROOT/outputs/$MODEL_KEY"

echo
info "repo    : $REPO_ROOT"
info "python  : $("$PYTHON" -V 2>&1)"
info "model   : $MODEL -> $MODEL_KEY"
info "seeds   : $SEEDS"
info "results : $TIER_RESULTS"

# =============================================================================
# 1. Archive pre-fix results
# =============================================================================
stage "1/6  Archive results produced before the validity fixes"

if [[ $DO_ARCHIVE -eq 0 ]]; then
    info "skipped (--skip-archive)"
elif [[ -f "$ARCHIVE_MARKER" ]]; then
    ok "already archived on $(cat "$ARCHIVE_MARKER") - not touching current results"
else
    # The guard matters: stages 3-5 write into these same paths, so an
    # unguarded archive step would move a half-finished sweep out of the way
    # every time the script was restarted after a crash.
    ARCHIVE_DIR="$REPO_ROOT/results/_archive_prefix_$(date +%Y%m%d_%H%M%S)"
    moved=0
    if compgen -G "$TIER_RESULTS/*_metrics.json" > /dev/null; then
        mkdir -p "$ARCHIVE_DIR/results"
        find "$TIER_RESULTS" -maxdepth 1 -type f \
             \( -name '*.json' -o -name '*.md' -o -name '*.csv' -o -name '*.txt' -o -name '*.log' \) \
             -exec mv -t "$ARCHIVE_DIR/results" {} +
        moved=1
        info "moved pre-fix reports to $ARCHIVE_DIR/results"
    fi
    if [[ -d "$TIER_OUTPUTS" ]]; then
        mkdir -p "$ARCHIVE_DIR"
        mv "$TIER_OUTPUTS" "$ARCHIVE_DIR/outputs_$MODEL_KEY"
        moved=1
        info "moved pre-fix adapters to $ARCHIVE_DIR/outputs_$MODEL_KEY"
        # Keep a pointer for stage 3: real adapters make the tamper experiment
        # concrete rather than synthetic.
        echo "$ARCHIVE_DIR/outputs_$MODEL_KEY" > "$REPO_ROOT/results/.archived_adapters"
    fi
    mkdir -p "$(dirname "$ARCHIVE_MARKER")"
    date -Is > "$ARCHIVE_MARKER"
    if [[ $moved -eq 1 ]]; then
        ok "pre-fix artefacts archived; they are kept, not deleted"
    else
        ok "nothing to archive - clean tree"
    fi
fi

# =============================================================================
# 2. Verify the pulled code
# =============================================================================
stage "2/6  Validity gate"

if [[ $DO_VERIFY -eq 0 ]]; then
    warn "skipped (--skip-verify) - the sweep may produce incomparable numbers"
else
    info "running unit tests ..."
    if "$PYTHON" -m unittest discover -s tests > "$LOG_DIR/verify_tests.log" 2>&1; then
        ok "unit tests pass ($(grep -oE 'Ran [0-9]+ tests' "$LOG_DIR/verify_tests.log" | tail -1))"
    else
        tail -30 "$LOG_DIR/verify_tests.log"
        die "unit tests failed - see $LOG_DIR/verify_tests.log"
    fi

    # --- budget matching on the real shards ---------------------------------
    # This is the assertion the whole comparison rests on: the union of the
    # per-round federated windows must equal the pooled centralized corpus.
    info "checking budget matching against the real data shards ..."
    if ! "$PYTHON" - > "$LOG_DIR/verify_budget.log" 2>&1 <<'PY'
import logging
import sys
from collections import Counter

sys.path.insert(0, ".")
logging.disable(logging.WARNING)
try:
    import datasets
    datasets.disable_progress_bars()
except Exception:
    pass

from utils.config import load_config, resolve_path
from trainer.sft import LocalTrainer

fed = load_config("configs/exp2_fl.yaml")
cen = load_config("configs/exp1_sft.yaml")

rounds = int(fed.get("num_rounds"))
per_round = int(fed.get("max_train_samples"))
shards = [resolve_path(p) for p in fed.get("client_files")]

central_paths = cen.get("data_path")
if not isinstance(central_paths, (list, tuple)):
    raise SystemExit(
        "exp1_sft.data_path is a single file. The centralized baseline must pool the\n"
        "client shards; a head slice of centralized_full.jsonl is client 1's partition."
    )
central_cap = int(cen.get("max_train_samples"))
if central_cap != rounds * per_round:
    raise SystemExit(
        f"budget mismatch: centralized takes {central_cap}/shard but federated covers "
        f"{rounds} x {per_round} = {rounds * per_round}/shard."
    )

trainer = LocalTrainer({**cen.to_dict(), "dry_run": True, "device": "cpu"})

class _Stub:            # avoid a Hub round-trip; slicing does not need a tokenizer
    eos_token = ""
    chat_template = None
trainer.load_tokenizer = lambda: _Stub()

# Multisets, not sets. Dolly-15k contains records that render to identical
# text (same instruction and response, differing only in metadata), and the
# shuffle can place copies in different shards. Set logic would report those
# as "repeated records" and fail a run whose windows are in fact disjoint.
# Counter equality still catches the real bug: if every round replayed the
# head of the shard, the federated side would hold 1500 records at count 3
# while the pooled side holds 4500 at count 1.
federated = Counter()
for r in range(rounds):
    for shard in shards:
        part = trainer.load_dataset(shard, max_samples=per_round, sample_offset=r * per_round)
        federated.update(part["text"])

pooled = Counter(
    trainer.load_dataset([resolve_path(p) for p in central_paths], max_samples=central_cap)["text"]
)

if federated != pooled:
    only_fed = federated - pooled
    only_pool = pooled - federated
    print(
        f"federated arm holds {sum(federated.values())} records ({len(federated)} distinct)\n"
        f"centralized arm holds {sum(pooled.values())} records ({len(pooled)} distinct)"
    )
    if sum(federated.values()) == sum(pooled.values()) and len(federated) < len(pooled):
        print(
            "\nSame total, fewer distinct records on the federated side: the rounds are\n"
            "re-reading the same window instead of advancing. Check that\n"
            "FederatedOrchestrator._sample_offset is present and that the training logs\n"
            "show window=[0, N), [N, 2N), [2N, 3N)."
        )
    for label, diff in (("only in federated", only_fed), ("only in centralized", only_pool)):
        if diff:
            example = next(iter(diff))
            print(f"\n{label}: {sum(diff.values())} record(s), e.g.\n  {example[:160]!r}")
    raise SystemExit(1)

duplicates = sum(count - 1 for count in pooled.values() if count > 1)
note = f" ({duplicates} duplicate record(s) in the corpus, counted consistently)" if duplicates else ""
print(
    f"OK  {sum(pooled.values())} records, identical in both arms "
    f"({rounds} rounds x {len(shards)} clients x {per_round}){note}"
)
PY
    then
        # Show the diagnosis, not the library chatter that surrounds it.
        grep -vE 'examples/s|Generating train split|Formatting prompts|Filter:|quantisation|^\s*$' \
            "$LOG_DIR/verify_budget.log" | sed 's/^/    /'
        echo
        info "full log: $LOG_DIR/verify_budget.log"
        die "budget-matching check failed - do NOT start the sweep"
    fi
    ok "$(grep '^OK' "$LOG_DIR/verify_budget.log" | sed 's/^OK  //')"

    # --- round windows advance in the real orchestrator ---------------------
    info "checking that federated rounds advance their data window ..."
    SMOKE_RES="$(mktemp -d)"
    "$PYTHON" main.py --config configs/exp2_fl.yaml --dry-run --force \
        --exp-name _verify --results-dir "$SMOKE_RES" --output-root "$SMOKE_RES/out" \
        > "$LOG_DIR/verify_windows.log" 2>&1 || true
    WINDOWS="$(grep -oE 'window=\[[0-9]+,' "$LOG_DIR/verify_windows.log" | sort -u | wc -l)"
    rm -rf "$SMOKE_RES"
    EXPECTED_ROUNDS="$("$PYTHON" -c "
import sys; sys.path.insert(0,'.')
from utils.config import load_config
print(int(load_config('configs/exp2_fl.yaml').get('num_rounds')))")"
    if [[ "$WINDOWS" -lt "$EXPECTED_ROUNDS" ]]; then
        grep -E 'window=' "$LOG_DIR/verify_windows.log" | head -10
        die "found $WINDOWS distinct window(s) across $EXPECTED_ROUNDS rounds - rounds are replaying the same samples"
    fi
    ok "$WINDOWS distinct round windows over $EXPECTED_ROUNDS rounds"
fi

# =============================================================================
# 3. Audit-layer experiments (no GPU)
# =============================================================================
stage "3/6  Audit-layer experiments (6: tamper detection, 7: scalability)"

mkdir -p "$TIER_RESULTS"
CHAIN_ARGS=()
if ! curl -s -m 3 -X POST "${FEDCHAIN_RPC_URL:-http://127.0.0.1:8545}" \
        -H 'Content-Type: application/json' \
        -d '{"jsonrpc":"2.0","method":"eth_chainId","params":[],"id":1}' >/dev/null 2>&1; then
    CHAIN_ARGS+=(--mock-chain)
    warn "no chain reachable - audit experiments will report estimated gas"
    warn "start one with ./infra.sh and re-run to get measured gas"
fi

if [[ $DO_AUDIT -eq 0 ]]; then
    info "skipped (--skip-audit)"
else
    TAMPER_ARGS=(--results-dir "$TIER_RESULTS" --trials "$TAMPER_TRIALS")
    ADAPTER_SRC=""
    [[ -f "$REPO_ROOT/results/.archived_adapters" ]] && ADAPTER_SRC="$(cat "$REPO_ROOT/results/.archived_adapters")/exp4_fedchain"
    [[ -d "$TIER_OUTPUTS/seed_${SEEDS%% *}/exp4_fedchain" ]] && ADAPTER_SRC="$TIER_OUTPUTS/seed_${SEEDS%% *}/exp4_fedchain"
    if [[ -n "$ADAPTER_SRC" && -d "$ADAPTER_SRC" ]]; then
        TAMPER_ARGS+=(--adapter-root "$ADAPTER_SRC")
        info "tamper trials use real adapters from $ADAPTER_SRC"
    else
        TAMPER_ARGS+=(--synthetic)
        info "no trained adapters yet - tamper trials use synthetic ones"
    fi

    if "$PYTHON" scripts/tamper_experiment.py "${TAMPER_ARGS[@]}" \
            ${CHAIN_ARGS[@]+"${CHAIN_ARGS[@]}"} \
            > "$TIER_RESULTS/exp6_tamper_stdout.txt" 2>&1; then
        ok "tamper detection: 100% caught, 0 false positives"
    else
        cat "$TIER_RESULTS/exp6_tamper_stdout.txt"
        # Not fatal to the sweep, but it invalidates the paper's core claim.
        fail "experiment 6 found undetected attacks or false positives"
        fail "this breaks the auditability claim - investigate before publishing"
    fi
    sed -n '/^| Attack/,/^$/p' "$TIER_RESULTS/exp6_tamper_stdout.txt" | sed 's/^/    /' || true

    if "$PYTHON" scripts/scalability_experiment.py --clients "$CLIENT_SWEEP" \
            --results-dir "$TIER_RESULTS" ${CHAIN_ARGS[@]+"${CHAIN_ARGS[@]}"} \
            > "$TIER_RESULTS/exp7_scalability_stdout.txt" 2>&1; then
        ok "scalability sweep: $TIER_RESULTS/exp7_scalability_metrics.json"
    else
        warn "scalability sweep failed (see $TIER_RESULTS/exp7_scalability_stdout.txt)"
    fi
fi

# =============================================================================
# 4-5. The long GPU stages
# =============================================================================
NUM_SEEDS="$(wc -w <<< "$SEEDS")"
if [[ $DO_MAIN -eq 1 || $DO_NONIID -eq 1 ]]; then
    # ~7.2 h per seed for 5 experiments, measured on a T600 at the 360M tier.
    EST_MAIN=0; EST_NONIID=0
    [[ $DO_MAIN -eq 1 ]]   && EST_MAIN=$(( NUM_SEEDS * 5 * 72 / 10 ))
    [[ $DO_NONIID -eq 1 ]] && EST_NONIID=$(( NUM_SEEDS * 4 * 72 / 10 ))
    EST_TOTAL=$(( EST_MAIN + EST_NONIID ))
    echo
    warn "the remaining stages are long: roughly ${EST_TOTAL} h at the ${MODEL_KEY} tier"
    info "  IID sweep     : $NUM_SEEDS seed(s) x experiments [$MAIN_EXPERIMENTS]   ~${EST_MAIN} h"
    info "  non-IID sweep : $NUM_SEEDS seed(s) x experiments [$NONIID_EXPERIMENTS] ~${EST_NONIID} h"
    info "  both resume from checkpoints, so an interruption costs at most one client"
    if [[ $ASSUME_YES -eq 0 && -t 0 ]]; then
        read -r -p "    Proceed? [y/N] " reply
        [[ "$reply" =~ ^[Yy]$ ]] || die "stopped before the long stages (re-run with -y to skip this prompt)"
    fi
fi

stage "4/6  IID sweep: experiments [$MAIN_EXPERIMENTS] x seeds [$SEEDS]"
if [[ $DO_MAIN -eq 0 ]]; then
    info "skipped (--skip-main)"
else
    ./run_all.sh --model "$MODEL" --seeds "$SEEDS" --experiments "$MAIN_EXPERIMENTS" \
        2>&1 | tee "$LOG_DIR/sweep_iid.log"
    ok "IID sweep complete"
fi

stage "5/6  Non-IID sweep: Dirichlet(alpha=$ALPHA), experiments [$NONIID_EXPERIMENTS]"
if [[ $DO_NONIID -eq 0 ]]; then
    info "skipped (--skip-noniid)"
else
    # --skip-infra/--skip-smoke: stage 4 already brought the node up and proved
    # the pipeline runs; repeating both would only add minutes.
    ./run_all.sh --model "$MODEL" --seeds "$SEEDS" --experiments "$NONIID_EXPERIMENTS" \
        --noniid "$ALPHA" --skip-smoke \
        2>&1 | tee "$LOG_DIR/sweep_noniid.log"
    ok "non-IID sweep complete"
fi

# =============================================================================
# 6. Summary
# =============================================================================
stage "6/6  Where everything landed"

ELAPSED=$(( SECONDS - RUN_STARTED ))
info "elapsed: $(( ELAPSED / 3600 ))h $(( (ELAPSED % 3600) / 60 ))m"
echo
info "Tables:"
[[ -f "$TIER_RESULTS/comparison.md" ]] \
    && ok "$TIER_RESULTS/comparison.md" \
    || warn "no comparison.md - the sweep did not finish"
[[ -f "$TIER_RESULTS/exp6_tamper_metrics.json" ]] \
    && ok "$TIER_RESULTS/exp6_tamper_metrics.json   (detection / false positives)"
[[ -f "$TIER_RESULTS/exp7_scalability_metrics.json" ]] \
    && ok "$TIER_RESULTS/exp7_scalability_metrics.json  (gas vs clients and model size)"
echo
info "Per-seed reports: $TIER_RESULTS/seed_*/"
echo
cat <<EOF
    Reading the results
    -------------------
    Quote accuracy from the "Paired difference vs exp1_sft" table, NOT from the
    metrics table above it - that one is a single representative seed.

    A "Significant: no" on the E3/E4-vs-E2 rows is the positive result: the
    audit layer changed nothing measurable. That is the paper's claim.

    Full design rationale and the pre-submission checklist: EXPERIMENTS.md
EOF
echo
