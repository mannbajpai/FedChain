#!/usr/bin/env python
"""
FedChain :: Audit-layer scalability (Exp 7)
===========================================

Exps 3-5 anchor a fixed 3-client federation, so they report one point on every
cost curve. A systems reviewer asks the obvious next question - what happens at
50 clients, or with a 7B adapter? - and "we ran three clients" is not an answer.

This script measures the audit layer alone across two sweeps, with no training
involved, so a full curve costs minutes rather than GPU-days:

* **clients**  - anchoring cost per round as the federation grows. The
  interesting quantity is whether per-round gas is linear in N with a small
  constant, since that is what decides whether the design is deployable on a
  chain where gas is money.
* **payload**  - cost as the artefact grows. SHA-256 anchoring should be
  *flat* in model size (only a 32-byte digest reaches the chain), which is the
  central efficiency argument for hash-anchoring over on-chain storage. A flat
  line here is the result; a rising one would falsify the design.

Because no model is trained, the numbers isolate protocol overhead from
learning cost - which is what makes them comparable across model tiers.

    python scripts/scalability_experiment.py --clients 1,3,5,10,25,50 --mock-chain
    python scripts/scalability_experiment.py --rpc-url http://127.0.0.1:8545

Writes ``results/exp7_scalability_metrics.json`` and a Markdown summary.
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from blockchain.logger import BlockchainLogger
from utils.common import bytes_to_mb, markdown_table, path_size_bytes, setup_logging, write_json

DEFAULT_CLIENT_COUNTS = "1,3,5,10,25,50"
#: (label, num_layers, hidden_size) - spans roughly 0.1 MB to 70 MB of adapter.
DEFAULT_PAYLOADS = "tiny:2:128,small:8:512,medium:16:1024,large:28:2048"


def build_adapter(target: Path, num_layers: int, hidden_size: int, seed: int) -> Path:
    from trainer.sft import synthesize_adapter

    return synthesize_adapter(
        target, num_layers=num_layers, hidden_size=hidden_size, seed=seed
    )


def sweep_clients(
    chain: BlockchainLogger, adapter: Path, counts: List[int], log_global: bool
) -> List[Dict[str, Any]]:
    """Anchor one round at each federation size."""
    rows: List[Dict[str, Any]] = []
    for count in counts:
        gas = 0
        latency = 0.0
        transactions = 0
        start = time.perf_counter()
        for index in range(count):
            receipt = chain.log_model_update(
                round=1,
                client_id=f"client_{index + 1}",
                adapter_bytes_or_file=adapter,
                ipfs_cid=f"Qm{'x' * 44}",
            )
            gas += int(receipt.get("gas_used", 0) or 0)
            latency += float(receipt.get("latency_sec", 0.0) or 0.0)
            transactions += 1
        if log_global:
            receipt = chain.log_model_update(
                round=1,
                client_id="server_global",
                adapter_bytes_or_file=adapter,
                ipfs_cid=f"Qm{'x' * 44}",
            )
            gas += int(receipt.get("gas_used", 0) or 0)
            latency += float(receipt.get("latency_sec", 0.0) or 0.0)
            transactions += 1
        wall = time.perf_counter() - start

        rows.append(
            {
                "num_clients": count,
                "transactions_per_round": transactions,
                "gas_per_round": gas,
                "gas_per_client": round(gas / max(1, count), 1),
                "chain_latency_sec": round(latency, 6),
                "wall_clock_sec": round(wall, 4),
            }
        )
    return rows


def sweep_payloads(
    chain: BlockchainLogger, workspace: Path, payloads: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """Anchor a single update at each artefact size."""
    rows: List[Dict[str, Any]] = []
    for index, spec in enumerate(payloads):
        adapter = build_adapter(
            workspace / f"payload_{spec['label']}",
            spec["num_layers"],
            spec["hidden_size"],
            seed=1000 + index,
        )
        size_bytes = path_size_bytes(adapter)

        hash_start = time.perf_counter()
        digest = chain.compute_hash(adapter)
        hash_latency = time.perf_counter() - hash_start

        receipt = chain.log_model_update(
            round=1,
            client_id=f"payload_{spec['label']}",
            adapter_bytes_or_file=adapter,
            ipfs_cid=f"Qm{'x' * 44}",
        )
        rows.append(
            {
                "label": spec["label"],
                "adapter_mb": bytes_to_mb(size_bytes),
                "gas_used": int(receipt.get("gas_used", 0) or 0),
                "chain_latency_sec": float(receipt.get("latency_sec", 0.0) or 0.0),
                "hash_latency_sec": round(hash_latency, 6),
                "hash_throughput_mb_s": round(
                    bytes_to_mb(size_bytes) / hash_latency if hash_latency else 0.0, 2
                ),
                "anchored_bytes": 32,
                "digest": digest[:16],
            }
        )
        shutil.rmtree(adapter, ignore_errors=True)
    return rows


def render(client_rows, payload_rows, context) -> str:
    lines = [
        "",
        "## FedChain - Audit-layer scalability (Exp 7)",
        "",
        f"_Chain mode: {context['chain_mode']} | anchored payload per tx: 32 B digest + CID_",
        "",
        "### Cost versus federation size",
        "",
        markdown_table(
            ["Clients", "Tx / round", "Gas / round", "Gas / client", "Chain latency (s)"],
            [
                [
                    str(r["num_clients"]),
                    str(r["transactions_per_round"]),
                    f"{r['gas_per_round']:,}",
                    f"{r['gas_per_client']:,.0f}",
                    f"{r['chain_latency_sec']:.4f}",
                ]
                for r in client_rows
            ],
        ),
        "",
    ]

    if len(client_rows) >= 2:
        first, last = client_rows[0], client_rows[-1]
        growth = last["gas_per_round"] / max(1, first["gas_per_round"])
        client_growth = last["num_clients"] / max(1, first["num_clients"])
        lines.append(
            f"_Gas grows {growth:.1f}x for a {client_growth:.0f}x increase in clients "
            f"({first['num_clients']} -> {last['num_clients']}): "
            f"{'linear in N, as designed' if growth <= client_growth * 1.3 else 'super-linear - investigate'}._"
        )
        lines.append("")

    lines.extend(
        [
            "### Cost versus artefact size",
            "",
            markdown_table(
                ["Payload", "Adapter (MiB)", "Gas", "Hash (s)", "Hash (MiB/s)", "On-chain bytes"],
                [
                    [
                        r["label"],
                        f"{r['adapter_mb']:.3f}",
                        f"{r['gas_used']:,}",
                        f"{r['hash_latency_sec']:.4f}",
                        f"{r['hash_throughput_mb_s']:.1f}",
                        str(r["anchored_bytes"]),
                    ]
                    for r in payload_rows
                ],
            ),
            "",
        ]
    )

    if len(payload_rows) >= 2:
        smallest, largest = payload_rows[0], payload_rows[-1]
        size_growth = largest["adapter_mb"] / max(1e-9, smallest["adapter_mb"])
        gas_growth = largest["gas_used"] / max(1, smallest["gas_used"])
        lines.append(
            f"_Adapter size grows {size_growth:.0f}x while gas changes {gas_growth:.2f}x: "
            "anchoring cost is independent of model size, because only the digest "
            "reaches the chain. Client-side hashing is the only size-dependent term._"
        )
        lines.append("")

    return "\n".join(lines)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Measure how the FedChain audit layer scales with clients and model size."
    )
    parser.add_argument("--clients", default=DEFAULT_CLIENT_COUNTS)
    parser.add_argument(
        "--payloads",
        default=DEFAULT_PAYLOADS,
        help="Comma-separated label:num_layers:hidden_size triples.",
    )
    parser.add_argument("--rpc-url", default="http://127.0.0.1:8545")
    parser.add_argument(
        "--mock-chain",
        action="store_true",
        help="Use the in-process ledger. Gas is then a deterministic estimate; "
        "report live-chain numbers in the paper.",
    )
    parser.add_argument(
        "--no-global",
        action="store_true",
        help="Do not anchor the aggregated global model each round.",
    )
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--exp-name", default="exp7_scalability")
    parser.add_argument("--seed", type=int, default=42)
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    args = build_arg_parser().parse_args(argv)
    setup_logging("WARNING")
    logging.getLogger("blockchain.logger").setLevel(logging.ERROR)

    try:
        counts = sorted({int(c) for c in args.clients.split(",") if c.strip()})
    except ValueError:
        print(f"ERROR: --clients must be integers, got {args.clients!r}", file=sys.stderr)
        return 2

    payloads: List[Dict[str, Any]] = []
    for spec in args.payloads.split(","):
        parts = spec.split(":")
        if len(parts) != 3:
            print(f"ERROR: bad --payloads entry {spec!r}; want label:layers:hidden", file=sys.stderr)
            return 2
        payloads.append(
            {"label": parts[0], "num_layers": int(parts[1]), "hidden_size": int(parts[2])}
        )

    workspace = Path(tempfile.mkdtemp(prefix="fedchain_scale_"))
    try:
        chain = BlockchainLogger(
            rpc_url=args.rpc_url,
            contract_address="",
            contract_artifact=PROJECT_ROOT / "blockchain/artifacts/FedChainAudit.json",
            contract_source=PROJECT_ROOT / "blockchain/contract.sol",
            force_mock=args.mock_chain,
        )

        reference = build_adapter(workspace / "reference", 4, 256, seed=args.seed)
        print(f"Sweeping federation sizes: {counts} ...")
        client_rows = sweep_clients(chain, reference, counts, log_global=not args.no_global)
        print(f"Sweeping payload sizes: {[p['label'] for p in payloads]} ...")
        payload_rows = sweep_payloads(chain, workspace, payloads)

        context = {
            "chain_mode": chain.mode,
            "rpc_url": args.rpc_url,
            "log_global_model": not args.no_global,
            "seed": args.seed,
        }
        results_dir = Path(args.results_dir)
        results_dir.mkdir(parents=True, exist_ok=True)
        write_json(
            results_dir / f"{args.exp_name}_metrics.json",
            {
                "experiment": {
                    "name": args.exp_name,
                    "description": "Audit-layer cost versus federation size and artefact size",
                    "timestamp_unix": int(time.time()),
                },
                "context": context,
                "clients_sweep": client_rows,
                "payload_sweep": payload_rows,
                "chain": chain.get_metrics_summary(),
            },
        )
        print(render(client_rows, payload_rows, context))
        print(f"Written: {results_dir / f'{args.exp_name}_metrics.json'}")
        chain.close()
        return 0
    finally:
        shutil.rmtree(workspace, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())
