"""
FedChain :: Model tier registry
===============================

The benchmark is run as a **ladder**: validate the whole pipeline on a small
model, confirm the numbers behave sensibly, then move up. Each rung costs
roughly a quarter of the wall-clock of the next, so a broken config is caught in
minutes rather than after a 20-hour sweep.

    1. SmolLM2-360M-Instruct   shakedown - is the pipeline correct end to end?
    2. Qwen2.5-0.5B-Instruct   preliminary results at a usable scale
    3. Llama-3.2-1B-Instruct   a second model family at a larger scale
    4. Qwen2.5-1.5B-Instruct   the configuration reported in the paper

The ladder varies two things, and the ablation only separates them because they
vary independently. **Scale** moves 360M -> 0.5B -> 1B -> 1.5B. **Family** moves
Qwen2 -> Llama, and rung 3 is the rung that supplies it: at 1B it is both the
largest completed tier and the only non-Qwen one, so a Llama-vs-Qwen difference
has to be read against the 360M rung (SmolLM2, also Llama-architecture) rather
than attributed to size alone.

All four share the same LoRA target modules
(``q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj``): SmolLM2 and
Llama-3.2 are Llama-architecture and Qwen2.5 is Qwen2-architecture, and all three
expose that exact projection naming. Every tier also ships a chat template, so
``use_chat_template: true`` works unchanged. No per-tier config edits are needed.

One tier is **gated**: ``meta-llama/Llama-3.2-1B-Instruct`` requires accepting
Meta's licence on the Hub and an exported ``HF_TOKEN``. ``ModelSpec.gated``
records that so a caller can fail in the first second rather than after the
first experiment has already been scheduled.

Artefacts are scoped per tier (``results/<key>/``, ``outputs/<key>/``) so runs of
different sizes never overwrite each other's reports or checkpoints.

Usage::

    python main.py --config configs/exp2_fl.yaml --model smol
    ./run_all.sh --model qwen-0.5b
    ./run_all.sh --model all          # the full ladder, smallest first

    python utils/models.py --list
    python utils/models.py --resolve smol     # -> "smollm2-360m<TAB>HuggingFaceTB/..."

This module deliberately has **no third-party imports**, and shell callers invoke
it as a script (``python utils/models.py``) rather than ``-m utils.models`` so
that resolving a model name never depends on PyYAML or the rest of the package
being importable.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


@dataclass(frozen=True)
class ModelSpec:
    """One rung of the model ladder."""

    key: str                      #: filesystem-safe tag, e.g. "smollm2-360m"
    hf_id: str                    #: Hugging Face repository id
    params: str                   #: human-readable parameter count
    architecture: str             #: transformers architecture family
    purpose: str                  #: why this rung exists
    speed_hint: str               #: rough wall-clock relative to the 1.5B tier
    aliases: Tuple[str, ...] = field(default_factory=tuple)
    gated: bool = False           #: Hub repo needs a licence acceptance + HF_TOKEN

    @property
    def label(self) -> str:
        return f"{self.hf_id} ({self.params})"


#: Ordered smallest-first; `--model all` walks this list in order.
MODEL_TIERS: Tuple[ModelSpec, ...] = (
    ModelSpec(
        key="smollm2-360m",
        hf_id="HuggingFaceTB/SmolLM2-360M-Instruct",
        params="360M",
        architecture="llama",
        purpose="Pipeline shakedown: fastest full end-to-end validation.",
        speed_hint="roughly 4x faster than the 1.5B tier (approximate)",
        aliases=("smol", "smollm", "smollm2", "smollm2-360m-instruct", "360m", "1", "tier1"),
    ),
    ModelSpec(
        key="qwen-0.5b",
        hf_id="Qwen/Qwen2.5-0.5B-Instruct",
        params="0.5B",
        architecture="qwen2",
        purpose="Preliminary results at a usable scale before the full run.",
        speed_hint="roughly 3x faster than the 1.5B tier (approximate)",
        aliases=("qwen0.5b", "qwen2.5-0.5b", "qwen-0.5b-instruct", "0.5b", "500m", "2", "tier2"),
    ),
    ModelSpec(
        key="llama-3.2-1b",
        hf_id="meta-llama/Llama-3.2-1B-Instruct",
        params="1.2B",
        architecture="llama",
        purpose="Second model family (Llama, not Qwen2) at the largest completed scale.",
        speed_hint="roughly 1.5x faster than the 1.5B tier (approximate)",
        aliases=("llama", "llama3.2", "llama-3.2", "llama3.2-1b", "llama-1b",
                 "llama-3.2-1b-instruct", "1b", "3", "tier3"),
        gated=True,
    ),
    ModelSpec(
        key="qwen-1.5b",
        hf_id="Qwen/Qwen2.5-1.5B-Instruct",
        params="1.5B",
        architecture="qwen2",
        purpose="The configuration reported in the paper.",
        speed_hint="baseline; the 4 GB VRAM ceiling",
        aliases=("qwen", "qwen1.5b", "qwen2.5-1.5b", "qwen-1.5b-instruct", "1.5b", "4", "tier4", "full"),
    ),
)

#: Positional aliases ("1".."4", "tier1".."tier4") track ladder ORDER, not a
#: frozen identity. Inserting Llama-3.2-1B at rung 3 therefore moved
#: Qwen2.5-1.5B from "3" to "4". Nothing in the repo referenced the numeric
#: aliases when that happened, but a habit does not show up in a grep: prefer
#: the explicit keys (`--model llama-3.2-1b`) in anything you save or share.

DEFAULT_TIER_KEY = "qwen-1.5b"

_BY_NAME: Dict[str, ModelSpec] = {}
for _spec in MODEL_TIERS:
    _BY_NAME[_spec.key.lower()] = _spec
    _BY_NAME[_spec.hf_id.lower()] = _spec
    for _alias in _spec.aliases:
        _BY_NAME[_alias.lower()] = _spec


def slugify_model(hf_id: str) -> str:
    """Filesystem-safe tag for an arbitrary Hugging Face id.

    ``"mistralai/Mistral-7B-Instruct-v0.3"`` -> ``"mistral-7b-instruct-v0.3"``.
    Used for models outside the registry so their artefacts are still scoped.
    Registered ids never reach this function via :func:`model_key_for` - the
    registry key wins, so ``meta-llama/Llama-3.2-1B-Instruct`` resolves to
    ``llama-3.2-1b`` rather than to its slug ``llama-3.2-1b-instruct``.
    """
    tail = str(hf_id).strip().rstrip("/").split("/")[-1]
    slug = re.sub(r"[^a-zA-Z0-9._-]+", "-", tail).strip("-.").lower()
    return slug or "custom-model"


def resolve_model(name: Optional[str]) -> Optional[ModelSpec]:
    """Resolve a tier key, alias, or raw Hugging Face id to a ``ModelSpec``.

    Returns ``None`` for an empty/None input, meaning "keep whatever
    ``model_name`` the config already specifies".

    Unregistered ids containing ``"/"`` are accepted and given a derived key, so
    the ladder is a convenience rather than a whitelist.
    """
    if name is None:
        return None
    cleaned = str(name).strip()
    if not cleaned:
        return None

    found = _BY_NAME.get(cleaned.lower())
    if found is not None:
        return found

    if "/" in cleaned:
        return ModelSpec(
            key=slugify_model(cleaned),
            hf_id=cleaned,
            params="unknown",
            architecture="unknown",
            purpose="User-supplied model (not part of the standard ladder).",
            speed_hint="unknown",
        )

    raise ValueError(
        f"Unknown model {name!r}.\n"
        f"Known tiers: {', '.join(s.key for s in MODEL_TIERS)}\n"
        f"Aliases:     {', '.join(sorted(a for s in MODEL_TIERS for a in s.aliases))}\n"
        "Or pass a full Hugging Face id such as 'Qwen/Qwen2.5-3B-Instruct'."
    )


def resolve_model_list(names: Optional[str]) -> List[ModelSpec]:
    """Resolve a comma/space separated selection, or ``all`` for every tier.

    ``all`` yields the ladder smallest-first, which is the recommended order:
    a failure on the cheap rung costs minutes, not hours.
    """
    if names is None or not str(names).strip():
        return []
    text = str(names).strip()
    if text.lower() in {"all", "ladder", "every"}:
        return list(MODEL_TIERS)

    specs: List[ModelSpec] = []
    seen = set()
    for token in re.split(r"[,\s]+", text):
        if not token:
            continue
        spec = resolve_model(token)
        if spec is not None and spec.key not in seen:
            specs.append(spec)
            seen.add(spec.key)
    return specs


def model_key_for(hf_id: Optional[str]) -> str:
    """Tag for a model id, preferring the registry key when it is a known tier."""
    if not hf_id:
        return "unknown-model"
    found = _BY_NAME.get(str(hf_id).lower())
    return found.key if found else slugify_model(hf_id)


def describe_tiers() -> str:
    """Human-readable ladder listing for ``--help`` output."""
    lines = [
        f"{'key':<14} {'params':<8} {'model':<40} purpose",
        f"{'-' * 14} {'-' * 8} {'-' * 40} {'-' * 40}",
    ]
    for spec in MODEL_TIERS:
        gate = " [gated]" if spec.gated else ""
        lines.append(
            f"{spec.key:<14} {spec.params:<8} {spec.hf_id:<40} {spec.purpose}{gate}"
        )
    lines.append("")
    lines.append("Aliases: " + "; ".join(f"{s.key} = {', '.join(s.aliases)}" for s in MODEL_TIERS))
    gated = [s for s in MODEL_TIERS if s.gated]
    if gated:
        lines.append("")
        lines.append(
            "[gated] needs the licence accepted on huggingface.co and an exported "
            "HF_TOKEN: " + ", ".join(s.hf_id for s in gated)
        )
    return "\n".join(lines)


def _main(argv: Optional[List[str]] = None) -> int:
    """Tiny CLI so shell scripts share this single source of truth."""
    import argparse

    parser = argparse.ArgumentParser(description="FedChain model tier registry.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--list", action="store_true", help="Print the ladder.")
    group.add_argument("--resolve", metavar="NAME", help="Print '<key>\\t<hf_id>' for one model.")
    group.add_argument(
        "--resolve-list",
        metavar="NAMES",
        help="Print one '<key>\\t<hf_id>' line per model ('all' for the whole ladder).",
    )
    group.add_argument(
        "--gated-list",
        metavar="NAMES",
        help=(
            "Print one '<key>\\t<hf_id>' line for each GATED model in the "
            "selection, and nothing at all when none is gated. Shell callers use "
            "this to check HF_TOKEN before scheduling a sweep rather than after."
        ),
    )
    args = parser.parse_args(argv)

    try:
        if args.list:
            print(describe_tiers())
        elif args.resolve:
            spec = resolve_model(args.resolve)
            if spec is None:
                return 1
            print(f"{spec.key}\t{spec.hf_id}")
        elif args.gated_list is not None:
            # Empty output IS the "nothing gated" answer, so this exits 0 either
            # way: a non-zero status would be indistinguishable from a bad name.
            for spec in resolve_model_list(args.gated_list):
                if spec.gated:
                    print(f"{spec.key}\t{spec.hf_id}")
        else:
            specs = resolve_model_list(args.resolve_list)
            if not specs:
                return 1
            for spec in specs:
                print(f"{spec.key}\t{spec.hf_id}")
    except ValueError as exc:
        print(str(exc), file=__import__("sys").stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
