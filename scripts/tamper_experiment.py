#!/usr/bin/env python
"""
FedChain :: Adversarial integrity experiment (Exp 6)
====================================================

Experiments 1-5 show that the audit layer costs almost nothing. They do **not**
show that it *does* anything: every integrity check in those runs passes,
because nothing ever attacks them. A paper whose contribution is auditable
federated learning has to demonstrate the audit catching something, and has to
report a false-positive rate for benign artefact churn. That is this script.

Threat model
------------
The aggregator is honest but the transport is not. A client anchors
``H(theta_k)`` on-chain, publishes ``theta_k`` to IPFS, and an adversary who
controls storage or the network path substitutes a different artefact before
the aggregator retrieves it. The adversary cannot rewrite the chain, so the
question is only whether the retrieved artefact is bound tightly enough to the
commitment for substitution to be detectable.

Attacks
-------
``bitflip``      flip a single bit in ``adapter_model.safetensors`` - the
                 weakest possible corruption, and the detection floor.
``scale``        multiply the LoRA B factors by a constant. This is the
                 classic model-replacement/boosting attack: it is semantically
                 devastating and leaves the file *structurally* valid, so
                 anything weaker than a content hash misses it.
``substitute``   serve a different client's adapter under the victim's CID.
``replay``       serve the victim's own adapter from an earlier round - every
                 byte was legitimately produced and signed at some point, which
                 defeats provenance schemes that check "is this a real adapter"
                 rather than "is this *the* adapter for this round".
``reserialize``  BENIGN control. Rewrite ``adapter_config.json`` with its keys
                 and ``target_modules`` in a different order, leaving the
                 weights untouched. A correct scheme must NOT flag this: PEFT
                 stores ``target_modules`` in a set and Python salts string
                 hashes per process, so honest re-serialisation reorders the
                 file. Flagging it would mean an honest client is randomly
                 rejected and an auditor cannot reproduce the commitment.

Every trial anchors through the real ``BlockchainLogger`` and verifies through
the same ``sha256_path`` the orchestrator uses, so the numbers describe the
deployed code path rather than a reimplementation of it.

Usage
-----
    # against artefacts a real run already produced
    python scripts/tamper_experiment.py \\
        --adapter-root outputs/smollm2-360m/exp4_fedchain \\
        --trials 20 --mock-chain

    # standalone, no prior run needed (synthetic adapters)
    python scripts/tamper_experiment.py --synthetic --trials 20 --mock-chain

Results land in ``results/<tier>/exp6_tamper_metrics.json`` plus a Markdown
table on stdout.
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import shutil
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from blockchain.logger import BlockchainLogger
from utils.common import markdown_table, setup_logging, sha256_path, write_json

ATTACKS = ("bitflip", "scale", "substitute", "replay", "reserialize")

#: Attacks that a correct scheme must flag. ``reserialize`` is deliberately
#: absent: it is the benign control, and detecting it is a false positive.
MALICIOUS_ATTACKS = ("bitflip", "scale", "substitute", "replay")


# =============================================================================
# Adapter discovery / synthesis
# =============================================================================
def discover_adapters(adapter_root: Path) -> List[Path]:
    """Find every client adapter directory under a completed run."""
    candidates = sorted(
        p.parent
        for p in adapter_root.rglob("adapter_model.safetensors")
        if "_retrieved" not in p.parts and "checkpoint-" not in p.parent.name
    )
    return candidates


def synthesise_adapters(target: Path, count: int, seed: int) -> List[Path]:
    """Write small but structurally valid PEFT adapters for a standalone run."""
    from trainer.sft import synthesize_adapter

    paths: List[Path] = []
    for index in range(count):
        path = target / f"client_{index + 1}"
        synthesize_adapter(path, seed=seed + index)
        paths.append(path)
    return paths


# =============================================================================
# Attacks
# =============================================================================
def attack_bitflip(victim: Path, rng: random.Random) -> str:
    weights = victim / "adapter_model.safetensors"
    payload = bytearray(weights.read_bytes())
    # Stay clear of the safetensors JSON header so the file remains loadable:
    # a corruption that merely makes the file unparseable would be caught by
    # any loader and would overstate what the hash contributes.
    header_len = int.from_bytes(payload[:8], "little") + 8
    position = rng.randrange(header_len, len(payload))
    payload[position] ^= 1 << rng.randrange(8)
    weights.write_bytes(bytes(payload))
    return f"flipped 1 bit at byte {position} of {len(payload)}"


def attack_scale(victim: Path, rng: random.Random, factor: float = 10.0) -> str:
    """Boost the LoRA B factors - the model-replacement attack."""
    try:
        from safetensors.torch import load_file, save_file
    except ImportError:
        return attack_bitflip(victim, rng) + " (safetensors unavailable; fell back to bitflip)"

    weights = victim / "adapter_model.safetensors"
    tensors = load_file(str(weights))
    touched = 0
    for key in list(tensors):
        if "lora_B" in key:
            tensors[key] = tensors[key] * factor
            touched += 1
    if not touched:  # dummy adapters may use another naming scheme
        key = sorted(tensors)[0]
        tensors[key] = tensors[key] * factor
        touched = 1
    save_file(tensors, str(weights))
    return f"scaled {touched} LoRA-B tensor(s) by {factor}x"


def attack_substitute(victim: Path, donor: Path, rng: random.Random) -> str:
    shutil.copy2(donor / "adapter_model.safetensors", victim / "adapter_model.safetensors")
    return f"served {donor.name}'s weights under the victim's commitment"


def attack_replay(victim: Path, earlier: Path, rng: random.Random) -> str:
    shutil.copy2(earlier / "adapter_model.safetensors", victim / "adapter_model.safetensors")
    return f"replayed a previously anchored adapter ({earlier.parent.name}/{earlier.name})"


def attack_reserialize(victim: Path, rng: random.Random) -> str:
    """Benign: rewrite adapter_config.json with a different member order."""
    config_path = victim / "adapter_config.json"
    if not config_path.exists():
        return "no adapter_config.json to reserialise"
    with open(config_path, "r", encoding="utf-8") as handle:
        config = json.load(handle)

    items = list(config.items())
    rng.shuffle(items)
    shuffled = dict(items)
    for key, value in shuffled.items():
        if isinstance(value, list) and all(isinstance(v, str) for v in value):
            reordered = list(value)
            rng.shuffle(reordered)
            shuffled[key] = reordered

    with open(config_path, "w", encoding="utf-8") as handle:
        json.dump(shuffled, handle, indent=1)
    return "reordered adapter_config.json keys and target_modules (weights untouched)"


# =============================================================================
# Trial driver
# =============================================================================
def run_trial(
    attack: str,
    adapters: List[Path],
    chain: BlockchainLogger,
    round_index: int,
    rng: random.Random,
    workspace: Path,
) -> Dict[str, Any]:
    """Anchor a clean adapter, attack the copy in flight, then verify."""
    victim_source = adapters[rng.randrange(len(adapters))]
    staging = workspace / f"trial_{round_index}"
    if staging.exists():
        shutil.rmtree(staging)
    shutil.copytree(victim_source, staging)

    # 1. Client anchors the honest artefact.
    anchor_start = time.perf_counter()
    receipt = chain.log_model_update(
        round=round_index,
        client_id=f"victim@{attack}",
        adapter_bytes_or_file=staging,
        ipfs_cid="",
    )
    anchor_latency = time.perf_counter() - anchor_start
    anchored_hash = receipt.get("model_hash") or sha256_path(staging)

    # 2. Adversary tampers with the artefact in transit.
    others = [a for a in adapters if a != victim_source] or adapters
    if attack == "bitflip":
        detail = attack_bitflip(staging, rng)
    elif attack == "scale":
        detail = attack_scale(staging, rng)
    elif attack == "substitute":
        detail = attack_substitute(staging, others[rng.randrange(len(others))], rng)
    elif attack == "replay":
        detail = attack_replay(staging, others[rng.randrange(len(others))], rng)
    elif attack == "reserialize":
        detail = attack_reserialize(staging, rng)
    else:
        raise ValueError(f"Unknown attack: {attack}")

    # 3. Aggregator re-hashes what it retrieved and compares with the chain.
    verify_start = time.perf_counter()
    accepted = chain.verify_artifact(staging, anchored_hash)
    verify_latency = time.perf_counter() - verify_start

    benign = attack not in MALICIOUS_ATTACKS
    # Correct outcome: malicious artefacts rejected, benign ones accepted.
    correct = accepted if benign else not accepted

    shutil.rmtree(staging, ignore_errors=True)
    return {
        "attack": attack,
        "benign": benign,
        "victim": victim_source.name,
        "detail": detail,
        "anchored_hash": anchored_hash,
        "accepted": accepted,
        "detected": not accepted,
        "correct": correct,
        "anchor_latency_sec": round(anchor_latency, 6),
        "verify_latency_sec": round(verify_latency, 6),
        "gas_used": receipt.get("gas_used", 0),
    }


def summarise(trials: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Per-attack detection / false-positive rates."""
    rows: List[Dict[str, Any]] = []
    for attack in ATTACKS:
        subset = [t for t in trials if t["attack"] == attack]
        if not subset:
            continue
        detected = sum(1 for t in subset if t["detected"])
        benign = subset[0]["benign"]
        rows.append(
            {
                "attack": attack,
                "benign_control": benign,
                "trials": len(subset),
                "detected": detected,
                "detection_rate": round(detected / len(subset), 4),
                "false_positive_rate": round(detected / len(subset), 4) if benign else None,
                "mean_verify_latency_sec": round(
                    sum(t["verify_latency_sec"] for t in subset) / len(subset), 6
                ),
                "mean_gas_used": int(sum(t["gas_used"] for t in subset) / len(subset)),
            }
        )
    return rows


def render(rows: List[Dict[str, Any]], context: Dict[str, Any]) -> str:
    table = markdown_table(
        ["Attack", "Type", "Trials", "Detected", "Detection rate", "Verify (ms)"],
        [
            [
                r["attack"],
                "benign control" if r["benign_control"] else "malicious",
                str(r["trials"]),
                str(r["detected"]),
                f"{r['detection_rate'] * 100:.1f}%",
                f"{r['mean_verify_latency_sec'] * 1000:.2f}",
            ]
            for r in rows
        ],
    )
    malicious = [r for r in rows if not r["benign_control"]]
    controls = [r for r in rows if r["benign_control"]]
    total_mal = sum(r["trials"] for r in malicious)
    caught = sum(r["detected"] for r in malicious)
    total_ctl = sum(r["trials"] for r in controls)
    false_pos = sum(r["detected"] for r in controls)

    lines = [
        "",
        "## FedChain - Integrity under attack (Exp 6)",
        "",
        f"_Chain mode: {context.get('chain_mode')} | adapters: {context.get('num_adapters')} | "
        f"seed: {context.get('seed')}_",
        "",
        table,
        "",
        f"**Detection:** {caught}/{total_mal} malicious artefacts rejected "
        f"({(caught / total_mal * 100) if total_mal else 0:.1f}%).",
        f"**False positives:** {false_pos}/{total_ctl} benign re-serialisations rejected "
        f"({(false_pos / total_ctl * 100) if total_ctl else 0:.1f}%).",
        "",
    ]
    return "\n".join(lines)


# =============================================================================
# CLI
# =============================================================================
def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Measure whether the FedChain audit layer detects tampering."
    )
    parser.add_argument(
        "--adapter-root",
        default=None,
        help="Root of a completed run (e.g. outputs/smollm2-360m/exp4_fedchain).",
    )
    parser.add_argument(
        "--synthetic",
        action="store_true",
        help="Generate throwaway adapters instead of using a completed run.",
    )
    parser.add_argument("--trials", type=int, default=20, help="Trials per attack. Default: 20.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--attacks",
        default=",".join(ATTACKS),
        help=f"Comma-separated subset of: {', '.join(ATTACKS)}.",
    )
    parser.add_argument("--rpc-url", default="http://127.0.0.1:8545")
    parser.add_argument(
        "--mock-chain",
        action="store_true",
        help="Use the in-process mock ledger (no node needed). Detection is a "
        "property of the hash, not of the chain, so results are identical; "
        "gas figures are then estimates rather than measurements.",
    )
    parser.add_argument("--results-dir", default="results", help="Where to write the report.")
    parser.add_argument("--exp-name", default="exp6_tamper")
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    args = build_arg_parser().parse_args(argv)
    setup_logging("WARNING")
    # A detected attack logs INTEGRITY FAILURE at ERROR level. Here that is the
    # expected outcome for most trials, so silence it - otherwise the run emits
    # hundreds of scary lines describing the system working correctly.
    logging.getLogger("blockchain.logger").setLevel(logging.CRITICAL)
    rng = random.Random(args.seed)

    attacks = [a.strip() for a in args.attacks.split(",") if a.strip()]
    unknown = [a for a in attacks if a not in ATTACKS]
    if unknown:
        print(f"ERROR: unknown attack(s): {', '.join(unknown)}", file=sys.stderr)
        return 2

    workspace = Path(tempfile.mkdtemp(prefix="fedchain_tamper_"))
    try:
        if args.synthetic or not args.adapter_root:
            adapters = synthesise_adapters(workspace / "source", count=3, seed=args.seed)
            source_label = "synthetic"
        else:
            root = Path(args.adapter_root)
            if not root.exists():
                print(f"ERROR: no such adapter root: {root}", file=sys.stderr)
                return 2
            adapters = discover_adapters(root)
            if len(adapters) < 2:
                print(
                    f"ERROR: found {len(adapters)} adapter(s) under {root}; need at least 2 "
                    "for the substitution and replay attacks. Pass --synthetic instead.",
                    file=sys.stderr,
                )
                return 2
            source_label = str(root)

        chain = BlockchainLogger(
            rpc_url=args.rpc_url,
            contract_address="",
            contract_artifact=PROJECT_ROOT / "blockchain/artifacts/FedChainAudit.json",
            contract_source=PROJECT_ROOT / "blockchain/contract.sol",
            force_mock=args.mock_chain,
        )

        trials: List[Dict[str, Any]] = []
        counter = 0
        for attack in attacks:
            for _ in range(args.trials):
                counter += 1
                trials.append(run_trial(attack, adapters, chain, counter, rng, workspace))

        rows = summarise(trials)
        context = {
            "chain_mode": chain.mode,
            "num_adapters": len(adapters),
            "adapter_source": source_label,
            "seed": args.seed,
            "trials_per_attack": args.trials,
        }

        results_dir = Path(args.results_dir)
        results_dir.mkdir(parents=True, exist_ok=True)
        report = {
            "experiment": {
                "name": args.exp_name,
                "description": "Detection rate of the on-chain audit layer under artefact tampering",
                "timestamp_unix": int(time.time()),
            },
            "context": context,
            "summary": rows,
            "trials": trials,
        }
        write_json(results_dir / f"{args.exp_name}_metrics.json", report)
        print(render(rows, context))
        print(f"Written: {results_dir / f'{args.exp_name}_metrics.json'}")

        chain.close()
        # Non-zero exit if the audit layer failed to do its job, so this is
        # usable as a regression gate and not only as a reporting script.
        malicious_missed = sum(
            r["trials"] - r["detected"] for r in rows if not r["benign_control"]
        )
        false_positives = sum(r["detected"] for r in rows if r["benign_control"])
        return 1 if (malicious_missed or false_positives) else 0
    finally:
        shutil.rmtree(workspace, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())
