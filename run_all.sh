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
DO_SETUP=1
DO_INFRA=1
DO_DATA=1
DO_SMOKE=1
DO_COMPARE=1
KEEP_NODES=1          # nodes we start are left running by default
FRESH=0
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
  --                        Everything after this is passed through to main.py

Examples:
  ./run_all.sh                                   # full benchmark
  ./run_all.sh --quick                           # ~15 min end-to-end check
  ./run_all.sh --experiments "3 4"               # only the decentralized runs
  ./run_all.sh -- --no-generation-metrics        # skip ROUGE-L/BLEU everywhere
EOF
}

# --- argument parsing --------------------------------------------------------
while [[ $# -gt 0 ]]; do
    case "$1" in
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
    if "$PYTHON" main.py --config configs/exp4_fedchain.yaml --dry-run \
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

SUCCEEDED=()
FAILED=()

for n in $EXPERIMENTS; do
    cfg="${CONFIG_OF[$n]:-}"
    name="${NAME_OF[$n]:-}"
    if [[ -z "$cfg" ]]; then
        warn "no such experiment: $n (expected 1-4)"
        continue
    fi

    echo
    echo "-----------------------------------------------------------------------"
    echo " Experiment $n : $name"
    echo " config       : $cfg"
    echo " started      : $(date -Is)"
    echo "-----------------------------------------------------------------------"

    started=$SECONDS
    if "$PYTHON" main.py --config "$cfg" ${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}; then
        elapsed=$((SECONDS - started))
        ok "$name finished in $((elapsed / 60))m $((elapsed % 60))s"
        SUCCEEDED+=("$name")
    else
        elapsed=$((SECONDS - started))
        fail "$name FAILED after $((elapsed / 60))m $((elapsed % 60))s (see $RESULTS_DIR/$name.log)"
        FAILED+=("$name")
    fi
done

# =============================================================================
# 8. Aggregate
# =============================================================================
if [[ $DO_COMPARE -eq 1 && ${#SUCCEEDED[@]} -gt 0 ]]; then
    step "[8/8] Building the comparison table"
    "$PYTHON" scripts/compare_results.py --results-dir "$RESULTS_DIR" || warn "aggregation failed"
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
echo

# Report whether the systems numbers are measured or synthetic - this decides
# whether the run is publishable as-is.
for name in ${SUCCEEDED[@]+"${SUCCEEDED[@]}"}; do
    report="$RESULTS_DIR/${name}_metrics.json"
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
print(f"   {exp['name']:<16} loss={report['metrics']['validation_loss']}"
      f" ppl={report['metrics']['perplexity']}{suffix}{flag}")
PY
done

echo
echo " artefacts:"
echo "   $RESULTS_DIR/comparison.md      cross-experiment table"
echo "   $RESULTS_DIR/comparison.csv     same numbers for plotting"
echo "   $RESULTS_DIR/<exp>_metrics.json full per-run report"
echo "   $LOG_DIR/                       transcripts and node logs"
if [[ $KEEP_NODES -eq 1 && ( $STARTED_ANVIL -eq 1 || $STARTED_IPFS -eq 1 ) ]]; then
    echo
    echo " note: anvil/ipfs were started by this script and are still running."
    echo "       stop them with:  pkill -f '^anvil'; pkill -f 'ipfs daemon'"
fi
echo " finished  : $(date -Is)"
echo "======================================================================="

[[ ${#FAILED[@]} -eq 0 ]] || exit 1
