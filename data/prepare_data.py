"""
FedChain :: Dataset preparation
===============================

Downloads Databricks Dolly-15k and materialises every shard the benchmark needs:

===========================  =================================================
``eval_500.jsonl``           the **first 500 records** of Dolly-15k, held out
                             of all training shards (the fixed evaluation split)
``centralized_full.jsonl``   the remaining pool, shuffled - Experiment 1
``client1..3.jsonl``         three disjoint I.I.D. partitions - Experiments 2-4
===========================  =================================================

The evaluation split is carved out **before** shuffling and partitioning, so no
evaluation record ever appears in any client's training data. Without this,
validation loss and perplexity would be measured on samples the model has
already fitted, and the four experiments would not be comparable.

Shuffling uses a fixed seed so the client partitions are reproducible and
identically distributed - the I.I.D. setting the paper reports.

    python data/prepare_data.py
    python data/prepare_data.py --num-clients 5 --seed 7
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List, Optional

DATA_DIR = Path(__file__).resolve().parent

DEFAULT_DATASET = "databricks/databricks-dolly-15k"
DEFAULT_EVAL_SIZE = 500
DEFAULT_NUM_CLIENTS = 3
DEFAULT_SEED = 42


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Download and partition Dolly-15k for the FedChain experiments."
    )
    parser.add_argument("--dataset", default=DEFAULT_DATASET, help="Hugging Face dataset id.")
    parser.add_argument(
        "--eval-size",
        type=int,
        default=DEFAULT_EVAL_SIZE,
        help="Records held out for evaluation, taken from the head of the dataset.",
    )
    parser.add_argument(
        "--num-clients", type=int, default=DEFAULT_NUM_CLIENTS, help="Number of federated clients."
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="Shuffle seed (reproducibility).")
    parser.add_argument(
        "--output-dir", default=str(DATA_DIR), help="Directory the JSONL shards are written to."
    )
    return parser


def write_jsonl(dataset, path: Path) -> int:
    """Write a datasets.Dataset to JSONL, returning the record count."""
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with open(path, "w", encoding="utf-8") as handle:
        for record in dataset:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            count += 1
    return count


def main(argv: Optional[List[str]] = None) -> int:
    args = build_arg_parser().parse_args(argv)

    try:
        from datasets import load_dataset
    except ImportError:
        print(
            "ERROR: the `datasets` package is required.\n"
            "       Install the project requirements first:  pip install -r requirements.txt",
            file=sys.stderr,
        )
        return 1

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Downloading {args.dataset} ...")
    dataset = load_dataset(args.dataset, split="train")
    total = len(dataset)
    print(f"  {total} records available.")

    if args.eval_size >= total:
        print(
            f"ERROR: --eval-size ({args.eval_size}) must be smaller than the dataset ({total}).",
            file=sys.stderr,
        )
        return 2
    if args.num_clients < 1:
        print("ERROR: --num-clients must be at least 1.", file=sys.stderr)
        return 2

    # ---- 1. Hold out the evaluation split BEFORE shuffling -----------------
    # Taking the head keeps the split identical to "the first 500 samples of
    # Dolly-15k" regardless of the shuffle seed used for the training shards.
    eval_split = dataset.select(range(args.eval_size))
    train_pool = dataset.select(range(args.eval_size, total))
    print(f"  Held out {len(eval_split)} evaluation records; {len(train_pool)} remain for training.")

    # ---- 2. Shuffle the training pool for an I.I.D. partition --------------
    print(f"Shuffling the training pool with seed {args.seed} for an I.I.D. split ...")
    train_pool = train_pool.shuffle(seed=args.seed)

    # ---- 3. Write the evaluation and centralized shards --------------------
    written = {}
    written["eval_500.jsonl"] = write_jsonl(eval_split, out_dir / "eval_500.jsonl")
    written["centralized_full.jsonl"] = write_jsonl(train_pool, out_dir / "centralized_full.jsonl")

    # ---- 4. Partition into disjoint client shards --------------------------
    num_clients = args.num_clients
    pool_size = len(train_pool)
    shard_size = pool_size // num_clients
    print(f"Partitioning {pool_size} records across {num_clients} client(s) (~{shard_size} each) ...")

    for index in range(num_clients):
        start = index * shard_size
        # The last client absorbs the remainder so no record is dropped.
        end = pool_size if index == num_clients - 1 else start + shard_size
        shard_name = f"client{index + 1}.jsonl"
        written[shard_name] = write_jsonl(train_pool.select(range(start, end)), out_dir / shard_name)

    # ---- 5. Manifest -------------------------------------------------------
    manifest = {
        "dataset": args.dataset,
        "total_records": total,
        "eval_size": args.eval_size,
        "num_clients": num_clients,
        "seed": args.seed,
        "files": written,
        "note": (
            "eval_500.jsonl is the head of the raw dataset and is excluded from every "
            "training shard, so there is no train/eval leakage."
        ),
    }
    with open(out_dir / "manifest.json", "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)

    print("\nData preparation complete:")
    for name, count in written.items():
        print(f"  {name:<28} {count:>6} records")
    print(f"  {'manifest.json':<28} {'':>6}")
    print(f"\nFiles written to {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
