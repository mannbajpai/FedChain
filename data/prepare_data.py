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
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

DATA_DIR = Path(__file__).resolve().parent

DEFAULT_DATASET = "databricks/databricks-dolly-15k"
DEFAULT_EVAL_SIZE = 500
DEFAULT_NUM_CLIENTS = 3
DEFAULT_SEED = 42


def hf_auth_kwargs():
    """Authenticate Hub downloads from environment without exposing the token."""
    token = (
        os.environ.get("HF_TOKEN", "")
        or os.environ.get("HUGGING_FACE_HUB_TOKEN", "")
    ).strip()
    return {"token": token} if token else {}


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
        "--output-dir", default=None, help="Directory the JSONL shards are written to."
    )
    parser.add_argument(
        "--partition",
        choices=["iid", "dirichlet"],
        default="iid",
        help=(
            "How the training pool is split across clients. "
            "'iid' gives every client a uniform random slice (the easy case for "
            "FedAvg). 'dirichlet' skews the mixture of Dolly task categories per "
            "client, which is the standard non-IID benchmark protocol and the "
            "regime federation is actually deployed in. Default: iid."
        ),
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.3,
        help=(
            "Dirichlet concentration for --partition dirichlet. Lower is more "
            "skewed: 0.1 is near-pathological (clients hold almost disjoint "
            "categories), 0.3 is the common 'hard but learnable' setting, 10.0 "
            "is nearly IID. Default: 0.3."
        ),
    )
    parser.add_argument(
        "--label-key",
        default="category",
        help="Record field the non-IID skew is defined over. Default: category.",
    )
    return parser


def partition_iid(pool_size: int, num_clients: int, rng: Any) -> List[List[int]]:
    """Contiguous equal slices of the already-shuffled pool."""
    shard_size = pool_size // num_clients
    shards: List[List[int]] = []
    for index in range(num_clients):
        start = index * shard_size
        # The last client absorbs the remainder so no record is dropped.
        end = pool_size if index == num_clients - 1 else start + shard_size
        shards.append(list(range(start, end)))
    return shards


def partition_dirichlet(
    labels: List[str], num_clients: int, alpha: float, rng: Any
) -> List[List[int]]:
    """Label-skewed split: per category, draw client proportions from Dir(alpha).

    This is the standard non-IID construction (Hsu et al., 2019). For every
    label c we sample p_c ~ Dirichlet(alpha * 1_N) and deal that label's records
    to clients in those proportions, so client k ends up with a distinctive
    category mixture. Small alpha => clients hold nearly disjoint categories,
    which is precisely the regime where naive FedAvg degrades and where an
    audit layer has to prove it costs nothing on top of an already hard problem.

    Shard sizes are deliberately *not* equalised: uneven |D_k| is part of what
    makes the setting non-IID, and it is what sample-weighted FedAvg exists to
    handle.
    """
    by_label: Dict[str, List[int]] = {}
    for index, label in enumerate(labels):
        by_label.setdefault(label, []).append(index)

    shards: List[List[int]] = [[] for _ in range(num_clients)]
    for label in sorted(by_label):
        indices = by_label[label]
        rng.shuffle(indices)
        proportions = rng.dirichlet([alpha] * num_clients)
        # Cut points along this label's records, proportional to the draw.
        cuts = (proportions.cumsum() * len(indices)).astype(int)[:-1]
        for client_index, chunk in enumerate(_split_at(indices, cuts)):
            shards[client_index].extend(chunk)

    for shard in shards:
        rng.shuffle(shard)
    return shards


def _split_at(items: List[int], cuts: Any) -> List[List[int]]:
    """Split ``items`` at the given cut indices (numpy-free equivalent of split)."""
    chunks: List[List[int]] = []
    previous = 0
    for cut in list(cuts) + [len(items)]:
        cut = max(previous, min(int(cut), len(items)))
        chunks.append(items[previous:cut])
        previous = cut
    return chunks


def describe_partition(
    shards: List[List[int]], labels: List[str]
) -> List[Dict[str, Any]]:
    """Per-client size and label histogram - the table the paper prints."""
    summary: List[Dict[str, Any]] = []
    for index, shard in enumerate(shards):
        histogram: Dict[str, int] = {}
        for record_index in shard:
            histogram[labels[record_index]] = histogram.get(labels[record_index], 0) + 1
        summary.append(
            {
                "client": f"client{index + 1}",
                "num_records": len(shard),
                "label_histogram": dict(sorted(histogram.items())),
            }
        )
    return summary


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

    # Non-IID shards live in their own subdirectory so both partitions can
    # coexist: the paper reports IID and non-IID side by side, and regenerating
    # one must never silently overwrite the other's results.
    if args.output_dir:
        out_dir = Path(args.output_dir)
    elif args.partition == "iid":
        out_dir = DATA_DIR
    else:
        out_dir = DATA_DIR / args.partition
    out_dir.mkdir(parents=True, exist_ok=True)

    auth_kwargs = hf_auth_kwargs()
    auth_mode = "authenticated via HF_TOKEN" if auth_kwargs else "anonymous"
    print(f"Downloading {args.dataset} ({auth_mode}) ...")
    dataset = load_dataset(args.dataset, split="train", **auth_kwargs)
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

    try:
        import numpy as np

        rng: Any = np.random.default_rng(args.seed)
    except ImportError:
        if args.partition == "dirichlet":
            print(
                "ERROR: --partition dirichlet needs numpy. Install it with "
                "`pip install numpy`.",
                file=sys.stderr,
            )
            return 3
        rng = None

    labels = [str(record.get(args.label_key, "") or "unlabelled") for record in train_pool]

    if args.partition == "dirichlet":
        print(
            f"Partitioning {pool_size} records across {num_clients} client(s) "
            f"with Dirichlet(alpha={args.alpha}) skew over '{args.label_key}' ..."
        )
        shards = partition_dirichlet(labels, num_clients, args.alpha, rng)
    else:
        shard_size = pool_size // num_clients
        print(
            f"Partitioning {pool_size} records across {num_clients} client(s) "
            f"IID (~{shard_size} each) ..."
        )
        shards = partition_iid(pool_size, num_clients, rng)

    for index, shard_indices in enumerate(shards):
        shard_name = f"client{index + 1}.jsonl"
        written[shard_name] = write_jsonl(
            train_pool.select(shard_indices), out_dir / shard_name
        )

    partition_summary = describe_partition(shards, labels)
    print("\nPartition profile:")
    for entry in partition_summary:
        top = sorted(entry["label_histogram"].items(), key=lambda kv: -kv[1])[:3]
        top_text = ", ".join(f"{name}={count}" for name, count in top)
        print(f"  {entry['client']:<10} {entry['num_records']:>6} records  | top: {top_text}")

    # ---- 5. Manifest -------------------------------------------------------
    manifest = {
        "dataset": args.dataset,
        "total_records": total,
        "eval_size": args.eval_size,
        "num_clients": num_clients,
        "seed": args.seed,
        "partition": args.partition,
        "dirichlet_alpha": args.alpha if args.partition == "dirichlet" else None,
        "label_key": args.label_key if args.partition == "dirichlet" else None,
        "partition_profile": partition_summary,
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
