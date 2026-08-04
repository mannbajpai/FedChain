#!/usr/bin/env bash
# =============================================================================
# FedChain :: Autonomous experiment runner
# -----------------------------------------------------------------------------
# One command takes a freshly cloned repo (after ./infra.sh) to a finished
# comparison table:
#
#   ./run_all.sh
#
# It will, idempotently:
#   1. verify the toolchain and GPU
#   2. create ./venv and install Python dependencies
#   3. start anvil + the IPFS daemon if they are not already listening
#   4. compile + stage the FedChainAudit contract artifact
#   5. download and partition Dolly-15k (skipped if the shards already exist)
#   6. run a fast dry-run smoke test of the whole pipeline
#   7. run experiments 1-4 sequentially
#   8. aggregate everything into results/comparison.md
#
# Nothing here is destructive: existing nodes are reused and left running,
# existing data shards and results are kept unless you pass --fresh.
#
# Run `./run_all.sh --help` for options.
# =============================================================================
set -Eeuo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"

# --- defaults ----------------------------------------------------------------
RPC_URL="${FEDCHAIN_RPC_URL:-http://127.0.0.1:8545}"
IPFS_API="${FEDCHAIN_IPFS_API:-http://127.0.0.1:5001}"
VENV_DIR="${FEDCHAIN_VENV:-$REPO_ROOT/.venv}"
RESULTS_DIR="$REPO_ROOT/results"
LOG_DIR="$RESULTS_DIR/logs"
TORCH_INDEX="${FEDCHAIN_TORCH_INDEX:-https://download.pytorch.org/whl/cu121}"

EXPERIMENTS="1 2 3 4"
MODEL=""              # "" => use the model_name in the config (Qwen2.5-1.5B)
DO_SETUP=1
DO_INFRA=1
DO_DATA=1
DO_SMOKE=1
DO_COMPARE=1
KEEP_NODES=1          # nodes we start are left running by default
FRESH=0
FORCE_RERUN=0         # re-run experiments that already produced a metrics.json
EXTRA_ARGS=()
STARTED_ANVIL=0
STARTED_IPFS=0

# --- pretty output -----------------------------------------------------------
if [[ -t 1 ]]; then
    C_RESET=$'\033[0m'; C_BOLD=$'\033[1m'; C_RED=$'\033[31m'
    C_GREEN=$'\033[32m'; C_YELLOW=$'\033[33m'; C_BLUE=$'\033[34m'
else
    C_RESET=""; C_BOLD=""; C_RED=""; C_GREEN=""; C_YELLOW=""; C_BLUE=""
fi

step()  { printf '\n%s==> %s%s\n' "$C_BOLD$C_BLUE" "$*" "$C_RESET"; }
info()  { printf '    %s\n' "$*"; }
ok()    { printf '    %s[ok]%s %s\n'   "$C_GREEN"  "$C_RESET" "$*"; }
warn()  { printf '    %s[warn]%s %s\n' "$C_YELLOW" "$C_RESET" "$*"; }
fail()  { printf '    %s[fail]%s %s\n' "$C_RED"    "$C_RESET" "$*"; }

usage() {
    sed -n '2,22p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
    cat <<'EOF'

Options:
  --model TIER              Model tier to run. Default: whatever the config says
                            (Qwen2.5-1.5B). Accepts a key, an alias, a full
                            Hugging Face id, a list, or "all":
                              smollm2-360m | smol   HuggingFaceTB/SmolLM2-360M-Instruct
                              qwen-0.5b            Qwen/Qwen2.5-0.5B-Instruct
                              qwen-1.5b | qwen     Qwen/Qwen2.5-1.5B-Instruct
                              all                  the full ladder, smallest first
                            Artefacts are scoped to results/<key>/ and
                            outputs/<key>/, so tiers never overwrite each other.
  --experiments "1 2 3 4"   Which experiments to run (default: all four)
  --skip-setup              Do not create the venv or install dependencies
  --skip-infra              Do not start/check anvil or the IPFS daemon
  --skip-data               Do not run data/prepare_data.py
  --skip-smoke              Do not run the dry-run smoke test
  --skip-compare            Do not build results/comparison.md
  --stop-nodes              Stop any nodes this script started when it exits
  --fresh                   Delete previous outputs/ and results/ before running
  --quick                   Fast sanity pass: 1 round, 20 samples, no generation
  --dry-run                 Synthetic adapters, no model download (~1 min total)
  --force                   Re-run experiments that already have a metrics.json
  --no-resume               Ignore checkpoints; restart each experiment from scratch
  --                        Everything after this is passed through to main.py

Crash recovery:
  Re-running the exact same command after a crash, an OOM kill or a reboot
  continues where it stopped. Finished experiments are skipped, finished rounds
  and clients are replayed from outputs/<exp>/checkpoint.json, and an
  interrupted client restarts from its last step checkpoint - not from step 0.

Recommended ladder (validate cheaply, then scale up):
  ./run_all.sh --model smol           # 1. shakedown on SmolLM2-360M
  ./run_all.sh --model qwen-0.5b      # 2. preliminary results
  ./run_all.sh --model qwen-1.5b      # 3. the paper configuration
  ./run_all.sh --model all            # ...or all three back to back

Examples:
  ./run_all.sh                                   # full benchmark, config default model
  ./run_all.sh                                   # ...run again after a crash: resumes
  ./run_all.sh --model smol --quick              # fastest possible sanity pass
  ./run_all.sh --experiments "3 4"               # only the decentralized runs
  ./run_all.sh --force --experiments 2           # deliberately redo experiment 2
  HF_TOKEN=hf_... ./run_all.sh                    # authenticated Hub downloads
  ./run_all.sh -- --no-generation-metrics        # skip ROUGE-L/BLEU everywhere
EOF
}

# --- argument parsing --------------------------------------------------------
while [[ $# -gt 0 ]]; do
    case "$1" in
        --model)        MODEL="$2"; shift 2 ;;
        --experiments)  EXPERIMENTS="$2"; shift 2 ;;
        --skip-setup)   DO_SETUP=0;   shift ;;
        --skip-infra)   DO_INFRA=0;   shift ;;
        --skip-data)    DO_DATA=0;    shift ;;
        --skip-smoke)   DO_SMOKE=0;   shift ;;
        --skip-compare) DO_COMPARE=0; shift ;;
        --stop-nodes)   KEEP_NODES=0; shift ;;
        --fresh)        FRESH=1;      shift ;;
        --quick)
            EXTRA_ARGS+=(--num-rounds 1 --max-train-samples 20 --no-generation-metrics)
            shift ;;
        --dry-run)      EXTRA_ARGS+=(--dry-run); DO_SMOKE=0; shift ;;
        --force)        FORCE_RERUN=1; shift ;;
        --no-resume)    EXTRA_ARGS+=(--no-resume); shift ;;
        -h|--help)      usage; exit 0 ;;
        --)             shift; while [[ $# -gt 0 ]]; do EXTRA_ARGS+=("$1"); shift; done ;;
        *)              fail "Unknown option: $1"; echo; usage; exit 2 ;;
    esac
done

mkdir -p "$RESULTS_DIR" "$LOG_DIR"
RUN_LOG="$LOG_DIR/run_all.log"
# Mirror everything to a transcript so an unattended run is fully auditable.
exec > >(tee -a "$RUN_LOG") 2>&1

echo "======================================================================="
echo " FedChain :: autonomous benchmark run"
echo " started      : $(date -Is)"
echo " repo         : $REPO_ROOT"
echo " experiments  : $EXPERIMENTS"
echo " model(s)     : ${MODEL:-<config default: Qwen2.5-1.5B>}"
if [[ -n "${HF_TOKEN:-${HUGGING_FACE_HUB_TOKEN:-}}" ]]; then
    echo " hf hub       : authenticated via HF_TOKEN"
else
    echo " hf hub       : anonymous (export HF_TOKEN to authenticate)"
fi
echo " transcript   : $RUN_LOG"
echo "======================================================================="

cleanup() {
    local code=$?
    if [[ $KEEP_NODES -eq 0 ]]; then
        [[ $STARTED_ANVIL -eq 1 ]] && { pkill -f '^anvil' 2>/dev/null || true; info "stopped anvil"; }
        [[ $STARTED_IPFS  -eq 1 ]] && { pkill -f 'ipfs daemon' 2>/dev/null || true; info "stopped ipfs"; }
    fi
    exit $code
}
trap cleanup EXIT
trap 'fail "Aborted at line $LINENO"; exit 130' INT TERM

# =============================================================================
# 0. Fresh start
# =============================================================================
if [[ $FRESH -eq 1 ]]; then
    step "Clearing previous outputs and results (--fresh)"
    rm -rf "$REPO_ROOT/outputs"
    find "$RESULTS_DIR" -maxdepth 1 -type f -delete 2>/dev/null || true
    mkdir -p "$LOG_DIR"
    ok "cleared outputs/ and results/"
fi

# =============================================================================
# 1. Toolchain
# =============================================================================
step "[1/8] Checking the toolchain"

export PATH="$HOME/.foundry/bin:$PATH"

command -v python3 >/dev/null || { fail "python3 not found. Install Python 3.10+."; exit 1; }
PY_VERSION="$(python3 -c 'import sys; print("%d.%d" % sys.version_info[:2])')"
ok "python3 $PY_VERSION"
python3 -c 'import sys; sys.exit(0 if sys.version_info[:2] >= (3, 9) else 1)' \
    || { fail "Python 3.9+ is required (found $PY_VERSION)."; exit 1; }

if command -v anvil >/dev/null; then ok "anvil  $(anvil --version 2>/dev/null | head -1)"
else warn "anvil not on PATH - the blockchain layer will fall back to MOCK mode."; fi

if command -v ipfs >/dev/null; then ok "ipfs   $(ipfs --version 2>/dev/null)"
else warn "ipfs not on PATH - the IPFS layer will fall back to MOCK mode."; fi

if command -v nvidia-smi >/dev/null; then
    ok "GPU    $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1)"
else
    warn "nvidia-smi not found - training will run on CPU (very slow, but correct)."
fi

# =============================================================================
# 2. Python environment
# =============================================================================
if [[ $DO_SETUP -eq 1 ]]; then
    step "[2/8] Preparing the Python environment"
    if [[ ! -d "$VENV_DIR" ]]; then
        info "creating venv at $VENV_DIR"
        python3 -m venv "$VENV_DIR"
    else
        info "reusing existing venv at $VENV_DIR"
    fi
    # shellcheck disable=SC1091
    source "$VENV_DIR/bin/activate"
    python -m pip install --quiet --upgrade pip setuptools wheel

    if python -c 'import torch' 2>/dev/null; then
        ok "torch $(python -c 'import torch; print(torch.__version__)') already installed"
    else
        info "installing torch from $TORCH_INDEX (this takes a few minutes)"
        python -m pip install --quiet torch --index-url "$TORCH_INDEX" \
            || { warn "CUDA wheel install failed; falling back to the default index (CPU build)"
                 python -m pip install --quiet torch; }
    fi

    info "installing project requirements"
    python -m pip install --quiet -r requirements.txt
    ok "dependencies ready"

    python - <<'PY'
import torch
print(f"    [ok] torch {torch.__version__} | CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    p = torch.cuda.get_device_properties(0)
    print(f"    [ok] GPU: {p.name} ({p.total_memory / 1024**3:.2f} GB)")
PY
else
    step "[2/8] Python environment (skipped)"
    if [[ -f "$VENV_DIR/bin/activate" ]]; then
        # shellcheck disable=SC1091
        source "$VENV_DIR/bin/activate"
        ok "activated $VENV_DIR"
    else
        warn "no venv at $VENV_DIR - using the ambient python3"
    fi
fi

PYTHON="$(command -v python || command -v python3)"

# =============================================================================
# 3. Infrastructure
# =============================================================================
rpc_up() {
    curl -s -m 3 -X POST -H 'Content-Type: application/json' \
        --data '{"jsonrpc":"2.0","method":"eth_chainId","params":[],"id":1}' \
        "$RPC_URL" 2>/dev/null | grep -q '"result"'
}
ipfs_up() {
    curl -s -m 3 -X POST "$IPFS_API/api/v0/version" 2>/dev/null | grep -q '"Version"'
}
wait_for() {
    local label="$1" probe="$2" tries="${3:-40}" i
    for ((i = 1; i <= tries; i++)); do
        if "$probe"; then ok "$label is up (after ${i}s)"; return 0; fi
        sleep 1
    done
    warn "$label did not come up within ${tries}s"
    return 1
}

if [[ $DO_INFRA -eq 1 ]]; then
    step "[3/8] Starting local infrastructure"

    if rpc_up; then
        ok "anvil already listening on $RPC_URL - reusing it"
    elif command -v anvil >/dev/null; then
        info "starting anvil -> $LOG_DIR/anvil.log"
        nohup anvil --host 127.0.0.1 --port 8545 > "$LOG_DIR/anvil.log" 2>&1 &
        STARTED_ANVIL=1
        wait_for "anvil" rpc_up 40 || warn "see $LOG_DIR/anvil.log"
    else
        warn "anvil unavailable - experiments 3 and 4 will use the mock chain"
    fi

    if ipfs_up; then
        ok "IPFS daemon already listening on $IPFS_API - reusing it"
    elif command -v ipfs >/dev/null; then
        [[ -d "${IPFS_PATH:-$HOME/.ipfs}" ]] || { info "initialising the IPFS repo"; ipfs init >/dev/null 2>&1 || true; }
        info "starting ipfs daemon --offline -> $LOG_DIR/ipfs.log"
        nohup ipfs daemon --offline > "$LOG_DIR/ipfs.log" 2>&1 &
        STARTED_IPFS=1
        wait_for "IPFS daemon" ipfs_up 60 || warn "see $LOG_DIR/ipfs.log"
    else
        warn "ipfs unavailable - experiment 4 will use the mock store"
    fi
else
    step "[3/8] Infrastructure (skipped)"
fi

# =============================================================================
# 4. Contract artifact
# =============================================================================
step "[4/8] Preparing the FedChainAudit contract"
ARTIFACT="$REPO_ROOT/blockchain/artifacts/FedChainAudit.json"

if [[ -f "$ARTIFACT" ]]; then
    ok "artifact already staged at blockchain/artifacts/FedChainAudit.json"
elif command -v forge >/dev/null; then
    info "compiling blockchain/contract.sol with forge"
    if forge build --contracts blockchain --out "$REPO_ROOT/.forge-out" >/dev/null 2>&1 \
       && [[ -f "$REPO_ROOT/.forge-out/contract.sol/FedChainAudit.json" ]]; then
        mkdir -p "$(dirname "$ARTIFACT")"
        cp "$REPO_ROOT/.forge-out/contract.sol/FedChainAudit.json" "$ARTIFACT"
        ok "artifact staged (main.py will deploy it to anvil on first use)"
    else
        warn "forge build failed - falling back to py-solc-x at runtime"
    fi
else
    info "forge not available; main.py will compile via py-solc-x at runtime"
fi

# =============================================================================
# 5. Data
# =============================================================================
step "[5/8] Preparing the dataset"
SHARDS_PRESENT=1
for f in eval_500.jsonl centralized_full.jsonl client1.jsonl client2.jsonl client3.jsonl; do
    [[ -s "$REPO_ROOT/data/$f" ]] || SHARDS_PRESENT=0
done

if [[ $DO_DATA -eq 0 ]]; then
    info "skipped (--skip-data)"
elif [[ $SHARDS_PRESENT -eq 1 ]]; then
    ok "all shards already present in data/ - not re-downloading"
else
    info "downloading and partitioning Dolly-15k"
    "$PYTHON" data/prepare_data.py
    ok "data ready"
fi

# =============================================================================
# 6. Smoke test
# =============================================================================
if [[ $DO_SMOKE -eq 1 ]]; then
    step "[6/8] Dry-run smoke test (no GPU, no model download)"
    # --force so a leftover _smoke report from an earlier run cannot short-circuit
    # the check; the scratch artefacts are removed immediately afterwards.
    if "$PYTHON" main.py --config configs/exp4_fedchain.yaml --dry-run --force \
            --exp-name _smoke --log-level WARNING > "$LOG_DIR/smoke.log" 2>&1; then
        ok "full pipeline exercised: training -> IPFS -> chain -> FedAvg -> report"
        rm -f "$RESULTS_DIR/_smoke_metrics.json" "$RESULTS_DIR/_smoke_audit_trail.json" \
              "$RESULTS_DIR/_smoke_ipfs_transfers.json" "$RESULTS_DIR/_smoke.log"
        rm -rf "$REPO_ROOT/outputs/_smoke"
    else
        fail "smoke test failed - see $LOG_DIR/smoke.log"
        tail -20 "$LOG_DIR/smoke.log"
        exit 1
    fi
else
    step "[6/8] Smoke test (skipped)"
fi

# =============================================================================
# 7. Experiments
# =============================================================================
step "[7/8] Running experiments: $EXPERIMENTS"

declare -A CONFIG_OF=(
    [1]="configs/exp1_sft.yaml"
    [2]="configs/exp2_fl.yaml"
    [3]="configs/exp3_fl_bc.yaml"
    [4]="configs/exp4_fedchain.yaml"
)
declare -A NAME_OF=(
    [1]="exp1_sft" [2]="exp2_fl" [3]="exp3_fl_bc" [4]="exp4_fedchain"
)

# --- resolve the model ladder ------------------------------------------------
# utils/models.py is the single source of truth for keys, aliases and ids, so
# the shell never hard-codes a Hugging Face repository name.
MODEL_KEYS=()
MODEL_IDS=()
if [[ -n "$MODEL" ]]; then
    # Invoked as a script, not `-m utils.models`: running the file directly
    # bypasses utils/__init__.py, so resolving a model name never depends on
    # PyYAML or anything else being importable.
    if ! MODEL_LINES="$("$PYTHON" utils/models.py --resolve-list "$MODEL" 2>&1)"; then
        fail "could not resolve --model '$MODEL'"
        echo "$MODEL_LINES"
        exit 2
    fi
    while IFS=$'\t' read -r _key _id; do
        [[ -n "$_key" ]] || continue
        MODEL_KEYS+=("$_key")
        MODEL_IDS+=("$_id")
    done <<< "$MODEL_LINES"
    info "model ladder: ${MODEL_KEYS[*]}"
else
    MODEL_KEYS+=("")   # empty => keep the config's model_name and default paths
    MODEL_IDS+=("")
fi

SUCCEEDED=()
FAILED=()
SKIPPED=()
RESULT_DIRS=()

for mi in "${!MODEL_KEYS[@]}"; do
    model_key="${MODEL_KEYS[$mi]}"
    model_id="${MODEL_IDS[$mi]}"

    if [[ -n "$model_key" ]]; then
        exp_results_dir="$RESULTS_DIR/$model_key"
        exp_output_root="$REPO_ROOT/outputs/$model_key"
        MODEL_ARGS=(--model "$model_id"
                    --results-dir "$exp_results_dir"
                    --output-root "$exp_output_root")
        echo
        echo "#######################################################################"
        echo "# MODEL TIER: $model_key  ($model_id)"
        echo "#######################################################################"
    else
        exp_results_dir="$RESULTS_DIR"
        exp_output_root="$REPO_ROOT/outputs"
        MODEL_ARGS=()
    fi
    mkdir -p "$exp_results_dir"
    RESULT_DIRS+=("$exp_results_dir")

for n in $EXPERIMENTS; do
    cfg="${CONFIG_OF[$n]:-}"
    name="${NAME_OF[$n]:-}"
    if [[ -z "$cfg" ]]; then
        warn "no such experiment: $n (expected 1-4)"
        continue
    fi

    label="$name"
    [[ -n "$model_key" ]] && label="$model_key/$name"
    report="$exp_results_dir/${name}_metrics.json"
    ckpt="$exp_output_root/${name}/checkpoint.json"

    # Level 3 recovery: a completed experiment is never repeated.
    if [[ -s "$report" && $FORCE_RERUN -eq 0 ]]; then
        # A crash during an older/non-atomic report write may have left a
        # non-empty but truncated file. Only valid JSON proves completion.
        if "$PYTHON" -c 'import json,sys; json.load(open(sys.argv[1], encoding="utf-8"))' "$report" \
                >/dev/null 2>&1; then
            ok "$label already complete ($report) - skipping. Use --force to redo it."
            SUCCEEDED+=("$label")
            SKIPPED+=("$label")
            continue
        fi
        warn "$report is not valid JSON; main.py will archive it and resume from checkpoint"
    fi

    echo
    echo "-----------------------------------------------------------------------"
    echo " Experiment $n : $name"
    echo " config       : $cfg"
    [[ -n "$model_key" ]] && echo " model        : $model_id  [$model_key]"
    echo " started      : $(date -Is)"
    if [[ -s "$ckpt" ]]; then
        # Levels 1+2 recovery: report what will be reused before starting.
        "$PYTHON" - "$ckpt" <<'PY' || true
import json, sys
try:
    s = json.load(open(sys.argv[1], encoding="utf-8"))
except Exception:
    sys.exit(0)
partial = s.get("partial_round") or {}
clients = sorted((partial.get("clients") or {}).keys())
print(f" resume       : {len(s.get('rounds', []))} round(s) done"
      + (f", round {partial['round']} has {len(clients)} client(s) done: {', '.join(clients)}"
         if clients else "")
      + f" [status={s.get('status')}, session #{s.get('sessions', 0) + 1}]")
PY
    fi
    echo "-----------------------------------------------------------------------"

    started=$SECONDS
    RUN_ARGS=(--config "$cfg" ${MODEL_ARGS[@]+"${MODEL_ARGS[@]}"})
    [[ $FORCE_RERUN -eq 1 ]] && RUN_ARGS+=(--force)
    if "$PYTHON" main.py "${RUN_ARGS[@]}" ${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}; then
        elapsed=$((SECONDS - started))
        ok "$label finished in $((elapsed / 60))m $((elapsed % 60))s"
        SUCCEEDED+=("$label")
    else
        elapsed=$((SECONDS - started))
        fail "$label FAILED after $((elapsed / 60))m $((elapsed % 60))s (see $exp_results_dir/$name.log)"
        info "progress is checkpointed - re-run this script to continue from here"
        FAILED+=("$label")
    fi
done   # experiments

# Per-tier comparison table, built as soon as that tier finishes so a long
# ladder produces usable output without waiting for the largest model.
if [[ $DO_COMPARE -eq 1 ]]; then
    if compgen -G "$exp_results_dir/*_metrics.json" > /dev/null; then
        "$PYTHON" scripts/compare_results.py --results-dir "$exp_results_dir" \
            > "$exp_results_dir/comparison_stdout.txt" 2>&1 \
            && ok "comparison table: $exp_results_dir/comparison.md" \
            || warn "aggregation failed for ${model_key:-default}"
    fi
fi

done   # model tiers

# =============================================================================
# 8. Aggregate
# =============================================================================
if [[ $DO_COMPARE -eq 1 && ${#SUCCEEDED[@]} -gt 0 ]]; then
    step "[8/8] Comparison tables"
    for d in ${RESULT_DIRS[@]+"${RESULT_DIRS[@]}"}; do
        [[ -f "$d/comparison.md" ]] && info "$d/comparison.md"
    done
    # Cross-tier view: only meaningful once more than one model has results.
    if [[ ${#RESULT_DIRS[@]} -gt 1 ]] || compgen -G "$RESULTS_DIR/*/[a-z]*_metrics.json" > /dev/null; then
        "$PYTHON" scripts/compare_results.py --across-models --results-dir "$RESULTS_DIR" \
            && ok "cross-model table: $RESULTS_DIR/comparison_across_models.md" \
            || warn "cross-model aggregation failed"
    fi
else
    step "[8/8] Aggregation (skipped)"
fi

# =============================================================================
# Summary
# =============================================================================
echo
echo "======================================================================="
echo " RUN SUMMARY"
echo "======================================================================="
printf ' completed : %s\n' "${SUCCEEDED[*]:-none}"
printf ' failed    : %s\n' "${FAILED[*]:-none}"
printf ' skipped   : %s (already had results)\n' "${SKIPPED[*]:-none}"
echo

# Report whether the systems numbers are measured or synthetic - this decides
# whether the run is publishable as-is.
for label in ${SUCCEEDED[@]+"${SUCCEEDED[@]}"}; do
    if [[ "$label" == */* ]]; then
        report="$RESULTS_DIR/${label%%/*}/${label##*/}_metrics.json"
    else
        report="$RESULTS_DIR/${label}_metrics.json"
    fi
    [[ -f "$report" ]] || continue
    "$PYTHON" - "$report" <<'PY'
import json, sys
report = json.load(open(sys.argv[1], encoding="utf-8"))
exp, chain, ipfs = report["experiment"], report.get("blockchain"), report.get("ipfs")
bits = []
if exp["enable_blockchain"]:
    bits.append(f"chain={chain['mode'] if chain else '?'}")
if exp["enable_ipfs"]:
    bits.append(f"ipfs={ipfs['backend'] if ipfs else '?'}")
suffix = f"  [{', '.join(bits)}]" if bits else ""
flag = " <-- MOCK: systems metrics are synthetic" if "mock" in " ".join(bits) else ""
tier = exp.get("model_tier", "-")
print(f"   {tier:<14} {exp['name']:<16} loss={report['metrics']['validation_loss']}"
      f" ppl={report['metrics']['perplexity']}{suffix}{flag}")
PY
done

echo
echo " artefacts:"
for d in ${RESULT_DIRS[@]+"${RESULT_DIRS[@]}"}; do
    echo "   $d/comparison.md"
done
[[ -f "$RESULTS_DIR/comparison_across_models.md" ]] && \
    echo "   $RESULTS_DIR/comparison_across_models.md   model-ladder table"
echo "   <results-dir>/<exp>_metrics.json  full per-run report"
echo "   $LOG_DIR/                         transcripts and node logs"
if [[ $KEEP_NODES -eq 1 && ( $STARTED_ANVIL -eq 1 || $STARTED_IPFS -eq 1 ) ]]; then
    echo
    echo " note: anvil/ipfs were started by this script and are still running."
    echo "       stop them with:  pkill -f '^anvil'; pkill -f 'ipfs daemon'"
fi
echo " finished  : $(date -Is)"
echo "======================================================================="

[[ ${#FAILED[@]} -eq 0 ]] || exit 1
