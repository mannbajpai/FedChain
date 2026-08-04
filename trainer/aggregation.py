"""
FedChain :: FedAvg over LoRA adapters
=====================================

``FedAvgAggregator`` merges the LoRA adapters produced by the participating
clients into a single global adapter for the next federated round.

Algorithm
---------
For every LoRA parameter tensor ``W`` present in all client checkpoints:

.. math::  W_{global} = \\sum_i p_i W_i,   \\qquad \\sum_i p_i = 1

With uniform weights this is the plain arithmetic mean
:math:`\\frac{1}{N}\\sum_i W_i`, which is the default and matches the FedAvg
formulation in the paper. Passing per-client sample counts switches to the
sample-weighted form of McMahan et al. (2017).

A note on correctness
---------------------
Averaging the factors ``A`` and ``B`` independently is **not** identical to
averaging the effective updates :math:`B_iA_i`, because
:math:`(\\frac{1}{N}\\sum B_i)(\\frac{1}{N}\\sum A_i) \\neq \\frac{1}{N}\\sum B_iA_i`
whenever the clients' factors are not aligned. This factor-wise approximation is
the standard practice in federated LoRA work (FedIT, FedPEFT, Sun et al. 2024)
because it keeps the communication cost at ``O(r(d_in + d_out))`` instead of
``O(d_in * d_out)``. Warm-starting each round from the previous *global* adapter,
as ``FederatedOrchestrator`` does, keeps the client factors close to one another
and keeps the approximation tight. The exact-but-expensive alternative is
available via ``aggregate_delta_w`` for ablation studies.

Accumulation is always performed in float32 and cast back to each tensor's
original dtype at the end, so averaging bf16/fp16 adapters does not compound
rounding error across rounds.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import time
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

from utils.common import bytes_to_mb, path_size_bytes

LOGGER = logging.getLogger(__name__)

PathLike = Union[str, os.PathLike]

SAFETENSORS_NAME = "adapter_model.safetensors"
BIN_NAME = "adapter_model.bin"
CONFIG_NAME = "adapter_config.json"

#: Substrings identifying trainable LoRA tensors inside a PEFT state dict.
LORA_KEY_MARKERS: Tuple[str, ...] = (
    "lora_A",
    "lora_B",
    "lora_embedding_A",
    "lora_embedding_B",
    "lora_magnitude_vector",  # DoRA
)


def is_lora_key(key: str) -> bool:
    """True when a state-dict key holds an aggregatable LoRA parameter."""
    return any(marker in key for marker in LORA_KEY_MARKERS)


class FedAvgAggregator:
    """Federated averaging of PEFT LoRA checkpoints.

    Parameters
    ----------
    weighted:
        When True, ``aggregate_lora_adapters`` honours the ``client_weights``
        argument (typically local sample counts). When False the mean is always
        uniform, which is the configuration reported in the paper.
    strict:
        When True, a key present in some clients but missing in others raises.
        When False (default) the aggregator averages over the intersection and
        logs precisely which keys were skipped.
    """

    def __init__(self, weighted: bool = False, strict: bool = False) -> None:
        self.weighted = weighted
        self.strict = strict
        self.history: List[Dict[str, Any]] = []

    # =========================================================================
    # Checkpoint IO
    # =========================================================================
    @staticmethod
    def _resolve_checkpoint(adapter_path: PathLike) -> Path:
        """Locate the weight file inside an adapter directory (or accept a file)."""
        path = Path(adapter_path)
        if path.is_file():
            return path
        if not path.is_dir():
            raise FileNotFoundError(f"Adapter path does not exist: {path}")

        for candidate in (path / SAFETENSORS_NAME, path / BIN_NAME):
            if candidate.exists():
                return candidate

        # Fall back to any single weight file the trainer may have produced.
        loose = sorted(list(path.glob("*.safetensors")) + list(path.glob("*.bin")))
        if loose:
            LOGGER.debug("Using non-standard adapter weight file %s", loose[0].name)
            return loose[0]

        raise FileNotFoundError(
            f"No adapter weights found in {path} (looked for {SAFETENSORS_NAME} / {BIN_NAME})"
        )

    @staticmethod
    def load_adapter_state(adapter_path: PathLike) -> Dict[str, Any]:
        """Load an adapter state dict from safetensors or a PyTorch pickle."""
        import torch

        checkpoint = FedAvgAggregator._resolve_checkpoint(adapter_path)

        if checkpoint.suffix == ".safetensors":
            from safetensors.torch import load_file

            state = load_file(str(checkpoint), device="cpu")
        else:
            try:
                state = torch.load(str(checkpoint), map_location="cpu", weights_only=True)
            except TypeError:  # torch < 2.0 has no weights_only
                state = torch.load(str(checkpoint), map_location="cpu")

        if not isinstance(state, dict):
            raise ValueError(f"{checkpoint} did not contain a state dict (got {type(state).__name__})")

        LOGGER.debug("Loaded %d tensors from %s", len(state), checkpoint)
        return state

    @staticmethod
    def save_adapter_state(
        state_dict: Dict[str, Any], output_dir: PathLike, use_safetensors: bool = True
    ) -> Path:
        """Persist a state dict as a PEFT-loadable adapter checkpoint."""
        import torch

        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        if use_safetensors:
            from safetensors.torch import save_file

            # PEFT requires contiguous tensors and the `format: pt` metadata key.
            contiguous = {k: v.contiguous() for k, v in state_dict.items()}
            target = out_dir / SAFETENSORS_NAME
            save_file(contiguous, str(target), metadata={"format": "pt"})
        else:
            target = out_dir / BIN_NAME
            torch.save(state_dict, str(target))

        return target

    # =========================================================================
    # Aggregation
    # =========================================================================
    def aggregate_lora_adapters(
        self,
        client_adapter_paths: Sequence[PathLike],
        global_output_path: PathLike,
        client_weights: Optional[Sequence[float]] = None,
    ) -> Dict[str, Any]:
        """Average client LoRA adapters into one global adapter.

        Parameters
        ----------
        client_adapter_paths:
            One adapter directory (or weight file) per participating client.
        global_output_path:
            Directory the aggregated adapter is written to. Receives
            ``adapter_model.safetensors`` plus a copy of ``adapter_config.json``
            so it can be reloaded with ``PeftModel.from_pretrained``.
        client_weights:
            Optional mixing coefficients (normalised internally). Ignored unless
            the aggregator was constructed with ``weighted=True``.

        Returns
        -------
        dict
            Aggregation metrics: tensor/parameter counts, timings, artefact size
            and the exact mixing weights used.
        """
        import torch

        start = time.perf_counter()
        paths = [Path(p) for p in client_adapter_paths]
        if not paths:
            raise ValueError("aggregate_lora_adapters() requires at least one client adapter.")

        out_dir = Path(global_output_path)
        out_dir.mkdir(parents=True, exist_ok=True)

        weights = self._normalise_weights(len(paths), client_weights)
        LOGGER.info(
            "FedAvg over %d client adapter(s) with weights %s",
            len(paths),
            [round(w, 4) for w in weights],
        )

        # ---- load -----------------------------------------------------------
        states: List[Dict[str, Any]] = []
        for path in paths:
            states.append(self.load_adapter_state(path))

        # Single client: nothing to average, but still normalise the output
        # layout so downstream rounds see a consistent artefact.
        lora_keys, skipped_keys = self._select_keys(states)
        if not lora_keys:
            raise ValueError(
                "No LoRA tensors were common to all client adapters. "
                f"Client 0 exposes keys such as: {list(states[0])[:5]}"
            )

        # ---- average --------------------------------------------------------
        aggregated: Dict[str, Any] = OrderedDict()
        total_params = 0
        for key in lora_keys:
            reference = states[0][key]
            accumulator = torch.zeros(reference.shape, dtype=torch.float32)
            for state, weight in zip(states, weights):
                tensor = state[key]
                if tensor.shape != reference.shape:
                    raise ValueError(
                        f"Shape mismatch for {key}: {tuple(reference.shape)} vs {tuple(tensor.shape)}. "
                        "All clients must share the same LoRA rank and target modules."
                    )
                accumulator.add_(tensor.to(torch.float32), alpha=float(weight))
            aggregated[key] = accumulator.to(reference.dtype)
            total_params += reference.numel()

        # Non-LoRA tensors (buffers, modules_to_save) are carried over verbatim
        # from the first client so the checkpoint stays loadable.
        carried = 0
        for key, tensor in states[0].items():
            if key not in aggregated:
                aggregated[key] = tensor.clone()
                carried += 1

        # ---- persist --------------------------------------------------------
        weight_file = self.save_adapter_state(aggregated, out_dir, use_safetensors=True)
        config_source = self._copy_adapter_config(paths, out_dir)

        elapsed = time.perf_counter() - start
        artifact_bytes = path_size_bytes(out_dir)

        metrics = {
            "num_clients": len(paths),
            "client_weights": [round(w, 6) for w in weights],
            "weighting": "sample-weighted" if self.weighted and client_weights else "uniform",
            "num_lora_tensors": len(lora_keys),
            "num_carried_tensors": carried,
            "num_skipped_keys": len(skipped_keys),
            "skipped_keys_sample": skipped_keys[:10],
            "total_lora_parameters": total_params,
            "aggregation_time_sec": round(elapsed, 4),
            "global_adapter_path": str(out_dir),
            "weight_file": str(weight_file),
            "adapter_config_source": str(config_source) if config_source else None,
            "artifact_size_mb": bytes_to_mb(artifact_bytes),
        }

        LOGGER.info(
            "FedAvg complete: %d LoRA tensors (%s params) merged in %.3fs -> %s (%.3f MB)",
            len(lora_keys),
            f"{total_params:,}",
            elapsed,
            out_dir,
            metrics["artifact_size_mb"],
        )
        if skipped_keys:
            LOGGER.warning(
                "%d key(s) were not common to all clients and were skipped: %s%s",
                len(skipped_keys),
                skipped_keys[:5],
                " ..." if len(skipped_keys) > 5 else "",
            )

        self.history.append(metrics)
        return metrics

    # =========================================================================
    # Helpers
    # =========================================================================
    def _normalise_weights(
        self, num_clients: int, client_weights: Optional[Sequence[float]]
    ) -> List[float]:
        if not self.weighted or client_weights is None:
            return [1.0 / num_clients] * num_clients

        if len(client_weights) != num_clients:
            raise ValueError(
                f"client_weights has {len(client_weights)} entries but {num_clients} adapters were given."
            )
        values = [max(0.0, float(w)) for w in client_weights]
        total = sum(values)
        if total <= 0:
            LOGGER.warning("All client weights were zero; falling back to a uniform average.")
            return [1.0 / num_clients] * num_clients
        return [v / total for v in values]

    def _select_keys(self, states: Sequence[Dict[str, Any]]) -> Tuple[List[str], List[str]]:
        """Intersect LoRA keys across clients; report the ones left out."""
        per_client_lora = [{k for k in state if is_lora_key(k)} for state in states]
        common = set(per_client_lora[0])
        for keys in per_client_lora[1:]:
            common &= keys

        union = set()
        for keys in per_client_lora:
            union |= keys
        skipped = sorted(union - common)

        if skipped and self.strict:
            raise ValueError(
                f"Client adapters disagree on {len(skipped)} LoRA key(s) and strict=True: {skipped[:10]}"
            )

        # Sorting keeps the output tensor order deterministic, which in turn
        # keeps the serialised bytes - and therefore the SHA-256 anchored
        # on-chain - reproducible for identical inputs.
        return sorted(common), skipped

    @staticmethod
    def _copy_adapter_config(client_paths: Sequence[Path], out_dir: Path) -> Optional[Path]:
        """Copy ``adapter_config.json`` from the first client that has one."""
        for path in client_paths:
            source_dir = path if path.is_dir() else path.parent
            config_path = source_dir / CONFIG_NAME
            if config_path.exists():
                destination = out_dir / CONFIG_NAME
                shutil.copyfile(config_path, destination)
                # The aggregated adapter is a fresh artefact, not a resumed run.
                try:
                    with open(destination, "r", encoding="utf-8") as handle:
                        config = json.load(handle)
                    config["inference_mode"] = False
                    with open(destination, "w", encoding="utf-8") as handle:
                        json.dump(config, handle, indent=2)
                except Exception as exc:  # pragma: no cover - malformed config
                    LOGGER.debug("Left adapter_config.json untouched (%s)", exc)
                LOGGER.debug("Copied adapter config from %s", config_path)
                return config_path

        LOGGER.warning(
            "No %s found among the client adapters; the aggregated checkpoint will "
            "not be loadable by PeftModel.from_pretrained until one is supplied.",
            CONFIG_NAME,
        )
        return None

    # =========================================================================
    # Exact (expensive) variant, for ablations
    # =========================================================================
    def aggregate_delta_w(
        self,
        client_adapter_paths: Sequence[PathLike],
        client_weights: Optional[Sequence[float]] = None,
    ) -> Dict[str, Any]:
        """Average the *effective* updates ``B_i @ A_i`` instead of the factors.

        Returns a mapping ``module_prefix -> averaged dense delta`` rather than a
        PEFT checkpoint, because the exact mean of low-rank products generally
        has rank ``N*r`` and cannot be written back into a rank-``r`` adapter
        without a truncated SVD. Provided so the paper can quantify the error
        introduced by the factor-wise approximation used in the main pipeline.
        """
        import torch

        paths = [Path(p) for p in client_adapter_paths]
        weights = self._normalise_weights(len(paths), client_weights)
        states = [self.load_adapter_state(p) for p in paths]

        prefixes = sorted(
            {key.rsplit(".lora_A", 1)[0] for key in states[0] if ".lora_A" in key}
        )

        deltas: Dict[str, Any] = {}
        for prefix in prefixes:
            a_key, b_key = f"{prefix}.lora_A.weight", f"{prefix}.lora_B.weight"
            if not all(a_key in s and b_key in s for s in states):
                continue
            accumulator = None
            for state, weight in zip(states, weights):
                delta = state[b_key].to(torch.float32) @ state[a_key].to(torch.float32)
                accumulator = delta * weight if accumulator is None else accumulator + delta * weight
            deltas[prefix] = accumulator

        LOGGER.info("Computed %d exact averaged delta-W matrices.", len(deltas))
        return deltas


def load_lora_tensors(adapter_path: PathLike) -> Dict[str, Any]:
    """Convenience helper: only the LoRA tensors of an adapter checkpoint."""
    state = FedAvgAggregator.load_adapter_state(adapter_path)
    return {k: v for k, v in state.items() if is_lora_key(k)}
