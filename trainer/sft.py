"""
FedChain :: Local QLoRA supervised fine-tuner
=============================================

``LocalTrainer`` trains one LoRA adapter on one client's data shard, under the
memory envelope of a 4 GB NVIDIA T600:

* Qwen2.5-1.5B-Instruct loaded in **4-bit NF4** (BitsAndBytes, double quant)
* LoRA r=16 / alpha=32 on all seven attention + MLP projections
* gradient checkpointing, micro-batch 1, grad-accum 8, max_seq_length 512
* PyTorch SDPA attention
* aggressive teardown (``del`` -> ``gc.collect()`` -> ``torch.cuda.empty_cache()``)
  after every client so three clients can run back-to-back without OOM

Robustness
----------
* **No GPU?** 4-bit quantisation is skipped and the model loads in fp32 on CPU,
  so the pipeline is unit-testable end-to-end without CUDA.
* **trl API drift?** ``SFTTrainer``/``SFTConfig`` keyword names have changed
  repeatedly across releases (``tokenizer``->``processing_class``,
  ``max_seq_length``->``max_length``). Every constructor call is filtered
  through :func:`_filter_kwargs`, which inspects the installed signature and
  drops anything unsupported. If ``trl`` is missing or refuses to build, the
  trainer falls back to a plain ``transformers.Trainer`` over a pre-tokenised
  dataset, which is objective-identical.
* **``dry_run``?** A structurally valid random adapter is emitted without ever
  touching the base model, which exercises aggregation, IPFS and the chain.
"""

from __future__ import annotations

import gc
import inspect
import json
import logging
import math
import os
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

from utils.checkpoint import adapter_is_complete, adapter_matches_hash, compute_fingerprint
from utils.common import (
    Timer,
    cuda_peak_memory_mb,
    format_duration,
    free_cuda_memory,
    get_device,
    hf_auth_kwargs,
    sha256_path,
)

LOGGER = logging.getLogger(__name__)

PathLike = Union[str, os.PathLike]

#: Prompt template used when the tokenizer exposes no chat template.
ALPACA_WITH_CONTEXT = (
    "Below is an instruction that describes a task, paired with an input that "
    "provides further context. Write a response that appropriately completes the request.\n\n"
    "### Instruction:\n{instruction}\n\n### Input:\n{context}\n\n### Response:\n"
)
ALPACA_NO_CONTEXT = (
    "Below is an instruction that describes a task. "
    "Write a response that appropriately completes the request.\n\n"
    "### Instruction:\n{instruction}\n\n### Response:\n"
)

SYSTEM_PROMPT = "You are a helpful assistant."


# =============================================================================
# Version-tolerant helpers
# =============================================================================
def _filter_kwargs(target: Callable[..., Any], kwargs: Dict[str, Any]) -> Dict[str, Any]:
    """Keep only the kwargs that ``target``'s installed signature accepts.

    Hugging Face and TRL rename constructor arguments between minor releases.
    Filtering against the live signature means this repository keeps working
    across the whole 4.4x-4.5x / trl 0.9-0.2x range instead of hard-failing on
    a ``TypeError`` deep inside a library.
    """
    try:
        signature = inspect.signature(target)
    except (TypeError, ValueError):  # builtins / C extensions
        return dict(kwargs)

    # Strictly named parameters only. Even when the target declares **kwargs it
    # may reject unknown names (HF Trainer raises on unexpected arguments), so
    # silently losing an optional setting is preferable to a hard TypeError.
    allowed = set(signature.parameters)
    kept = {k: v for k, v in kwargs.items() if k in allowed}
    dropped = sorted(set(kwargs) - set(kept))
    if dropped:
        LOGGER.debug(
            "Dropping kwargs unsupported by the installed %s: %s",
            getattr(target, "__name__", target),
            dropped,
        )
    return kept


def _first_supported(target: Callable[..., Any], names: Sequence[str]) -> Optional[str]:
    """Return the first name in ``names`` that ``target`` actually accepts."""
    try:
        params = set(inspect.signature(target).parameters)
    except (TypeError, ValueError):
        return names[0] if names else None
    for name in names:
        if name in params:
            return name
    return None


# =============================================================================
# Trainer
# =============================================================================
class LocalTrainer:
    """Trains a single client's QLoRA adapter.

    One instance is reusable across clients and rounds; the heavy base model is
    loaded lazily and (by default) released after every ``train_client`` call so
    that peak VRAM stays bounded on a 4 GB card.
    """

    def __init__(self, config: Any, device: Optional[str] = None) -> None:
        self.config = config
        self.cfg = config  # short alias used throughout

        preference = device or self._get("device", "auto")
        self.device, self.device_info = get_device(preference)
        self.use_cuda = self.device == "cuda"

        self.model_name: str = self._get("model_name", "Qwen/Qwen2.5-1.5B-Instruct")
        self.max_seq_length: int = int(self._get("max_seq_length", 512))
        self.dry_run: bool = bool(self._get("dry_run", False))
        self.keep_model_loaded: bool = bool(self._get("keep_model_loaded", False))

        # Intra-training crash recovery: HF Trainer writes optimizer + scheduler
        # + adapter state every `save_steps`, and a killed run resumes from the
        # last one instead of restarting the client from step 0.
        self.step_checkpoints: bool = bool(self._get("enable_step_checkpoints", True))
        self.save_steps: int = int(self._get("save_steps", 25))
        self.save_total_limit: int = int(self._get("save_total_limit", 2))

        # 4-bit only makes sense on CUDA: bitsandbytes has no CPU kernels for it.
        self.use_4bit: bool = bool(self._get("load_in_4bit", True)) and self.use_cuda
        if bool(self._get("load_in_4bit", True)) and not self.use_cuda:
            LOGGER.warning(
                "4-bit quantisation requested but no CUDA device is present; "
                "loading the model in full precision on CPU instead."
            )

        self._tokenizer: Any = None
        self._base_model: Any = None
        self.history: List[Dict[str, Any]] = []

        LOGGER.info(
            "LocalTrainer ready | model=%s device=%s 4bit=%s seq_len=%d dry_run=%s",
            self.model_name,
            self.device,
            self.use_4bit,
            self.max_seq_length,
            self.dry_run,
        )

    # -- config access -------------------------------------------------------
    def _get(self, key: str, default: Any = None) -> Any:
        try:
            return self.config.get(key, default)
        except AttributeError:
            return getattr(self.config, key, default)

    # -- dtype ---------------------------------------------------------------
    def _compute_dtype(self) -> Any:
        import torch

        requested = str(self._get("bnb_4bit_compute_dtype", "auto")).lower()
        if requested in {"float16", "fp16", "half"}:
            return torch.float16
        if requested in {"bfloat16", "bf16"}:
            return torch.bfloat16
        if requested in {"float32", "fp32"}:
            return torch.float32
        if not self.use_cuda:
            return torch.float32
        if bool(self.device_info.get("bf16_supported")):
            return torch.bfloat16
        return torch.float16

    # =========================================================================
    # Tokenizer / model
    # =========================================================================
    def load_tokenizer(self) -> Any:
        if self._tokenizer is not None:
            return self._tokenizer

        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            trust_remote_code=bool(self._get("trust_remote_code", False)),
            use_fast=True,
            **hf_auth_kwargs(),
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            LOGGER.info("Tokenizer had no pad_token; reusing eos_token (%r).", tokenizer.eos_token)
        # Causal LM training/eval expects right padding; left padding would
        # misalign the shifted labels.
        tokenizer.padding_side = "right"
        tokenizer.model_max_length = self.max_seq_length

        self._tokenizer = tokenizer
        return tokenizer

    def _quantization_config(self) -> Any:
        from transformers import BitsAndBytesConfig

        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type=str(self._get("bnb_4bit_quant_type", "nf4")),
            bnb_4bit_use_double_quant=bool(self._get("bnb_4bit_use_double_quant", True)),
            bnb_4bit_compute_dtype=self._compute_dtype(),
        )

    def load_base_model(self) -> Any:
        """Load (and cache) the quantised base model."""
        if self._base_model is not None:
            return self._base_model

        import torch
        from transformers import AutoModelForCausalLM

        load_kwargs: Dict[str, Any] = {
            "trust_remote_code": bool(self._get("trust_remote_code", False)),
            "attn_implementation": str(self._get("attn_implementation", "sdpa")),
            "low_cpu_mem_usage": True,
            **hf_auth_kwargs(),
        }

        compute_dtype = self._compute_dtype()
        # `torch_dtype` was renamed to `dtype` in transformers 4.56; pass
        # whichever the installed version documents.
        dtype_key = "dtype" if _accepts_dtype_kwarg() else "torch_dtype"
        load_kwargs[dtype_key] = compute_dtype

        if self.use_4bit:
            load_kwargs["quantization_config"] = self._quantization_config()
            load_kwargs["device_map"] = {"": 0}
        elif self.use_cuda:
            load_kwargs["device_map"] = {"": 0}

        LOGGER.info("Loading base model %s (4bit=%s, dtype=%s) ...", self.model_name, self.use_4bit, compute_dtype)
        with Timer("Base model load", LOGGER):
            try:
                model = AutoModelForCausalLM.from_pretrained(self.model_name, **load_kwargs)
            except Exception as exc:
                if "attn_implementation" in load_kwargs:
                    LOGGER.warning("SDPA attention rejected (%s); retrying with the default kernel.", exc)
                    load_kwargs.pop("attn_implementation")
                    model = AutoModelForCausalLM.from_pretrained(self.model_name, **load_kwargs)
                else:
                    raise

        if not self.use_cuda:
            model = model.to(torch.device("cpu"))

        model.config.use_cache = False  # incompatible with gradient checkpointing
        if getattr(model.config, "pretraining_tp", 1) != 1:
            model.config.pretraining_tp = 1

        self._base_model = model
        return model

    def _prepare_peft_model(self, base_model: Any, init_adapter_path: Optional[PathLike]) -> Any:
        """Attach a LoRA adapter, resuming from the global adapter when given."""
        from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training

        gradient_checkpointing = bool(self._get("gradient_checkpointing", True))

        if self.use_4bit:
            base_model = prepare_model_for_kbit_training(
                base_model, use_gradient_checkpointing=gradient_checkpointing
            )
        elif gradient_checkpointing:
            base_model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

        if gradient_checkpointing and hasattr(base_model, "enable_input_require_grads"):
            # Without this the checkpointed graph has no grad-requiring input
            # and backward silently produces no LoRA gradients.
            base_model.enable_input_require_grads()

        if init_adapter_path is not None and Path(init_adapter_path).exists():
            LOGGER.info("Resuming from the global adapter at %s", init_adapter_path)
            model = PeftModel.from_pretrained(base_model, str(init_adapter_path), is_trainable=True)
        else:
            lora_config = LoraConfig(
                r=int(self._get("lora_r", 16)),
                lora_alpha=int(self._get("lora_alpha", 32)),
                lora_dropout=float(self._get("lora_dropout", 0.05)),
                bias=str(self._get("lora_bias", "none")),
                task_type="CAUSAL_LM",
                target_modules=list(
                    self._get(
                        "lora_target_modules",
                        ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
                    )
                ),
            )
            model = get_peft_model(base_model, lora_config)

        trainable, total = _count_parameters(model)
        LOGGER.info(
            "LoRA attached: %s trainable / %s total parameters (%.4f%%)",
            f"{trainable:,}",
            f"{total:,}",
            100.0 * trainable / max(total, 1),
        )
        return model

    # =========================================================================
    # Data
    # =========================================================================
    def format_example(self, example: Dict[str, Any]) -> str:
        """Render one Dolly-style record as a single training string."""
        instruction = str(example.get("instruction", "") or "")
        context = str(example.get("context", "") or example.get("input", "") or "")
        response = str(example.get("response", "") or example.get("output", "") or "")

        if bool(self._get("use_chat_template", True)):
            tokenizer = self.load_tokenizer()
            chat_template = getattr(tokenizer, "chat_template", None)
            if chat_template:
                user_content = f"{instruction}\n\n{context}".strip() if context else instruction
                messages = [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_content},
                    {"role": "assistant", "content": response},
                ]
                try:
                    return tokenizer.apply_chat_template(messages, tokenize=False)
                except Exception as exc:  # pragma: no cover - malformed template
                    LOGGER.warning("apply_chat_template failed (%s); using the Alpaca template.", exc)

        template = ALPACA_WITH_CONTEXT if context else ALPACA_NO_CONTEXT
        prompt = template.format(instruction=instruction, context=context)
        eos = getattr(self.load_tokenizer(), "eos_token", "") or ""
        return prompt + response + eos

    def load_dataset(self, dataset_path: PathLike, max_samples: Optional[int] = None) -> Any:
        """Load a JSONL shard and add the rendered ``text`` column."""
        from datasets import load_dataset as hf_load_dataset

        path = Path(dataset_path)
        if not path.exists():
            raise FileNotFoundError(
                f"Dataset shard not found: {path}\n"
                "Run `python data/prepare_data.py` first to download and partition Dolly-15k."
            )

        dataset = hf_load_dataset("json", data_files=str(path), split="train")
        original_size = len(dataset)

        if max_samples is not None and max_samples > 0 and max_samples < original_size:
            dataset = dataset.select(range(max_samples))

        dataset = dataset.map(
            lambda example: {"text": self.format_example(example)},
            remove_columns=[c for c in dataset.column_names if c != "text"],
            desc="Formatting prompts",
        )
        dataset = dataset.filter(lambda row: bool(row["text"]) and row["text"].strip() != "")

        LOGGER.info(
            "Loaded %s: %d samples (of %d available) from %s",
            path.name,
            len(dataset),
            original_size,
            path.parent,
        )
        return dataset

    def _tokenize_dataset(self, dataset: Any) -> Any:
        """Pre-tokenise for the plain-``Trainer`` fallback path."""
        tokenizer = self.load_tokenizer()

        def _encode(batch: Dict[str, List[str]]) -> Dict[str, Any]:
            encoded = tokenizer(
                batch["text"],
                truncation=True,
                max_length=self.max_seq_length,
                padding=False,
            )
            encoded["labels"] = [list(ids) for ids in encoded["input_ids"]]
            return encoded

        return dataset.map(_encode, batched=True, remove_columns=["text"], desc="Tokenizing")

    # =========================================================================
    # Training arguments
    # =========================================================================
    def _training_kwargs(self, output_dir: Path, epochs: float, dataset_size: int) -> Dict[str, Any]:
        import torch

        compute_dtype = self._compute_dtype()
        optim = str(self._get("optim", "paged_adamw_8bit"))
        # `paged_*` and `*_8bit` optimisers are bitsandbytes CUDA kernels.
        if not self.use_cuda and ("8bit" in optim or "paged" in optim):
            LOGGER.info("Optimiser %s needs CUDA; falling back to adamw_torch on CPU.", optim)
            optim = "adamw_torch"

        batch_size = int(self._get("batch_size", 1))
        grad_accum = int(self._get("grad_accum_steps", 8))
        steps_per_epoch = max(1, math.ceil(dataset_size / max(1, batch_size * grad_accum)))
        total_steps = max(1, int(steps_per_epoch * epochs))

        kwargs: Dict[str, Any] = {
            "output_dir": str(output_dir),
            "num_train_epochs": float(epochs),
            "per_device_train_batch_size": batch_size,
            "gradient_accumulation_steps": grad_accum,
            "learning_rate": float(self._get("learning_rate", 2.0e-4)),
            "weight_decay": float(self._get("weight_decay", 0.0)),
            "warmup_ratio": float(self._get("warmup_ratio", 0.03)),
            "lr_scheduler_type": str(self._get("lr_scheduler_type", "cosine")),
            "max_grad_norm": float(self._get("max_grad_norm", 0.3)),
            "optim": optim,
            "logging_steps": int(self._get("logging_steps", 10)),
            "report_to": [],
            "seed": int(self._get("seed", 42)),
            "gradient_checkpointing": bool(self._get("gradient_checkpointing", True)),
            "gradient_checkpointing_kwargs": {"use_reentrant": False},
            "dataloader_num_workers": int(self._get("dataloader_num_workers", 0)),
            "dataloader_pin_memory": self.use_cuda,
            "remove_unused_columns": False,
            "disable_tqdm": False,
            "fp16": self.use_cuda and compute_dtype == torch.float16,
            "bf16": self.use_cuda and compute_dtype == torch.bfloat16,
        }

        if self.step_checkpoints:
            # Cap save_steps so short clients still get at least one mid-run
            # checkpoint, and so a 1-step client does not save every step.
            save_steps = max(1, min(self.save_steps, max(1, total_steps // 2)))
            kwargs["save_strategy"] = "steps"
            kwargs["save_steps"] = save_steps
            kwargs["save_total_limit"] = max(1, self.save_total_limit)
            kwargs["save_safetensors"] = True
        else:
            kwargs["save_strategy"] = "no"

        max_steps = int(self._get("max_steps", -1) or -1)
        if max_steps > 0:
            kwargs["max_steps"] = max_steps
            total_steps = max_steps

        LOGGER.info(
            "Optimisation plan: %d samples, effective batch %d, ~%d step(s) over %.2f epoch(s)",
            dataset_size,
            batch_size * grad_accum,
            total_steps,
            epochs,
        )
        return kwargs

    def _build_trainer(self, model: Any, dataset: Any, output_dir: Path, epochs: float) -> Tuple[Any, str]:
        """Build an SFTTrainer, falling back to ``transformers.Trainer``."""
        tokenizer = self.load_tokenizer()
        base_kwargs = self._training_kwargs(output_dir, epochs, len(dataset))

        trainer = self._try_build_sft_trainer(model, dataset, tokenizer, base_kwargs)
        if trainer is not None:
            return trainer, "trl.SFTTrainer"

        LOGGER.warning("Falling back to transformers.Trainer with a pre-tokenised dataset.")
        return self._build_hf_trainer(model, dataset, tokenizer, base_kwargs), "transformers.Trainer"

    def _try_build_sft_trainer(
        self, model: Any, dataset: Any, tokenizer: Any, base_kwargs: Dict[str, Any]
    ) -> Optional[Any]:
        try:
            from trl import SFTTrainer
        except Exception as exc:
            LOGGER.warning("trl.SFTTrainer is unavailable (%s).", exc)
            return None

        try:
            from trl import SFTConfig

            sft_kwargs = dict(base_kwargs)
            seq_len_key = _first_supported(SFTConfig, ["max_length", "max_seq_length"])
            if seq_len_key:
                sft_kwargs[seq_len_key] = self.max_seq_length
            sft_kwargs["dataset_text_field"] = "text"
            sft_kwargs["packing"] = False
            args = SFTConfig(**_filter_kwargs(SFTConfig, sft_kwargs))
        except Exception as exc:
            LOGGER.info("SFTConfig unavailable/rejected (%s); using TrainingArguments.", exc)
            from transformers import TrainingArguments

            try:
                args = TrainingArguments(**_filter_kwargs(TrainingArguments, base_kwargs))
            except Exception as inner:
                LOGGER.warning("Could not build TrainingArguments: %s", inner)
                return None

        trainer_kwargs: Dict[str, Any] = {"model": model, "args": args, "train_dataset": dataset}
        tokenizer_key = _first_supported(SFTTrainer, ["processing_class", "tokenizer"])
        if tokenizer_key:
            trainer_kwargs[tokenizer_key] = tokenizer
        # Older trl releases take these on the trainer rather than the config.
        trainer_kwargs["dataset_text_field"] = "text"
        trainer_kwargs["max_seq_length"] = self.max_seq_length
        trainer_kwargs["packing"] = False

        try:
            return SFTTrainer(**_filter_kwargs(SFTTrainer, trainer_kwargs))
        except Exception as exc:
            LOGGER.warning("SFTTrainer construction failed (%s).", exc)
            return None

    def _build_hf_trainer(
        self, model: Any, dataset: Any, tokenizer: Any, base_kwargs: Dict[str, Any]
    ) -> Any:
        from transformers import DataCollatorForLanguageModeling, Trainer, TrainingArguments

        tokenized = self._tokenize_dataset(dataset)
        args = TrainingArguments(**_filter_kwargs(TrainingArguments, base_kwargs))
        collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

        trainer_kwargs: Dict[str, Any] = {
            "model": model,
            "args": args,
            "train_dataset": tokenized,
            "data_collator": collator,
        }
        tokenizer_key = _first_supported(Trainer, ["processing_class", "tokenizer"])
        if tokenizer_key:
            trainer_kwargs[tokenizer_key] = tokenizer
        return Trainer(**_filter_kwargs(Trainer, trainer_kwargs))

    # =========================================================================
    # Public training API
    # =========================================================================
    def train_client(
        self,
        dataset_path: PathLike,
        output_dir: PathLike,
        init_adapter_path: Optional[PathLike] = None,
        client_id: str = "client",
        epochs: Optional[float] = None,
        max_samples: Optional[int] = None,
        allow_resume: bool = True,
    ) -> Tuple[str, float]:
        """Fine-tune one LoRA adapter on one shard.

        Parameters
        ----------
        dataset_path:
            JSONL shard for this client.
        output_dir:
            Directory the adapter is written to (created if absent).
        init_adapter_path:
            Global adapter from the previous federated round. When given,
            training resumes from those weights instead of a fresh init - this
            is what makes FedAvg iterative rather than three independent runs.
        client_id:
            Label used in logs and metrics.
        epochs / max_samples:
            Per-call overrides of ``local_epochs`` / ``max_train_samples``.
        allow_resume:
            When True (default) and a ``checkpoint-*`` directory from a killed
            run is present in ``output_dir``, training continues from it rather
            than restarting at step 0. Pass False to force a clean restart, e.g.
            after a configuration change invalidated the old state.

        Returns
        -------
        (adapter_path, training_time_sec)
        """
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        epochs = float(epochs if epochs is not None else self._get("local_epochs", 1))
        if max_samples is None:
            max_samples = self._get("max_train_samples", None)

        LOGGER.info("=" * 78)
        LOGGER.info("Training %s | shard=%s | epochs=%.2f", client_id, Path(dataset_path).name, epochs)
        LOGGER.info("=" * 78)

        resume_identity = self._training_identity(
            dataset_path=dataset_path,
            init_adapter_path=init_adapter_path,
            client_id=client_id,
            epochs=epochs,
            max_samples=max_samples,
        )
        manifest = self._read_training_manifest(out_dir)
        identity_matches = bool(
            manifest
            and manifest.get("resume_fingerprint") == resume_identity["resume_fingerprint"]
        )

        if allow_resume and identity_matches and manifest.get("status") == "completed":
            expected_hash = manifest.get("model_hash")
            if adapter_is_complete(out_dir) and adapter_matches_hash(out_dir, expected_hash):
                self._purge_step_checkpoints(out_dir, reason="completed adapter recovered")
                training_time = float(manifest.get("training_time_sec", 0.0))
                LOGGER.info(
                    "Recovered completed adapter for %s from %s; no training repeated.",
                    client_id,
                    out_dir,
                )
                self.history.append(
                    {
                        "client_id": client_id,
                        "dataset": str(dataset_path),
                        "training_time_sec": round(training_time, 4),
                        "adapter_path": str(out_dir),
                        "recovered_completed_adapter": True,
                    }
                )
                return str(out_dir), round(training_time, 4)

        # Step checkpoints are only trustworthy when the adjacent manifest
        # proves they were produced by this exact data/config/init-adapter
        # combination. This also makes --no-resume a genuine clean restart.
        if not allow_resume or not identity_matches:
            self._purge_step_checkpoints(out_dir, reason="clean restart requested")

        self._write_training_manifest(
            out_dir,
            {
                **resume_identity,
                "status": "in_progress",
                "started_at": int(time.time()),
            },
        )

        if self.dry_run:
            return self._train_dry_run(
                out_dir,
                client_id,
                init_adapter_path,
                dataset_path,
                epochs,
                resume_identity,
            )

        start = time.perf_counter()
        trainer = None
        peft_model = None
        train_metrics: Dict[str, Any] = {}
        backend = "unknown"

        try:
            dataset = self.load_dataset(dataset_path, max_samples=max_samples)
            if len(dataset) == 0:
                raise ValueError(f"Shard {dataset_path} produced zero usable training samples.")

            base_model = self.load_base_model()
            peft_model = self._prepare_peft_model(base_model, init_adapter_path)
            trainer, backend = self._build_trainer(peft_model, dataset, out_dir, epochs)

            resume_from = self._find_step_checkpoint(out_dir) if allow_resume else None
            if resume_from:
                LOGGER.info(
                    "Resuming %s from step checkpoint %s (crash recovery).",
                    client_id,
                    Path(resume_from).name,
                )
                result = trainer.train(resume_from_checkpoint=resume_from)
                resumed_from_step_checkpoint = True
            else:
                LOGGER.info("Starting local optimisation via %s ...", backend)
                result = trainer.train()
                resumed_from_step_checkpoint = False
            train_metrics = dict(getattr(result, "metrics", {}) or {})

            # `save_pretrained` on a PeftModel writes only the adapter
            # (adapter_config.json + adapter_model.safetensors), which is the
            # few-megabyte artefact that actually travels over IPFS.
            peft_model.save_pretrained(str(out_dir))

            training_time = time.perf_counter() - start
            model_hash = sha256_path(out_dir)
            self._write_training_manifest(
                out_dir,
                {
                    **resume_identity,
                    "status": "completed",
                    "num_samples": len(dataset),
                    "trainer_backend": backend,
                    "training_time_sec": round(training_time, 4),
                    "model_hash": model_hash,
                    "completed_at": int(time.time()),
                },
            )

            # Step checkpoints carry optimizer state and are far larger than the
            # adapter. Remove them now that the final artefact exists, otherwise
            # they would inflate adapter_size_mb, the IPFS payload and the
            # measured communication volume.
            self._purge_step_checkpoints(out_dir, reason="training completed")

            peak_mb = cuda_peak_memory_mb()
            LOGGER.info(
                "%s finished in %s | train_loss=%s | peak VRAM=%s MB",
                client_id,
                format_duration(training_time),
                _fmt(train_metrics.get("train_loss")),
                peak_mb if peak_mb is not None else "n/a",
            )

            self.history.append(
                {
                    "client_id": client_id,
                    "dataset": str(dataset_path),
                    "num_samples": len(dataset),
                    "epochs": epochs,
                    "backend": backend,
                    "training_time_sec": round(training_time, 4),
                    "train_loss": train_metrics.get("train_loss"),
                    "train_runtime_sec": train_metrics.get("train_runtime"),
                    "peak_vram_mb": peak_mb,
                    "adapter_path": str(out_dir),
                    "resumed_from_step_checkpoint": resumed_from_step_checkpoint,
                }
            )
            return str(out_dir), round(training_time, 4)

        finally:
            # Teardown order matters on 4 GB: drop the trainer's optimiser state
            # and the PEFT wrapper before touching the CUDA caching allocator.
            del trainer
            if peft_model is not None:
                del peft_model
            if not self.keep_model_loaded:
                self.unload_model()
            free_cuda_memory()

    def _train_dry_run(
        self,
        out_dir: Path,
        client_id: str,
        init_adapter_path: Optional[PathLike],
        dataset_path: PathLike,
        epochs: float,
        resume_identity: Dict[str, Any],
    ) -> Tuple[str, float]:
        """Emit a structurally valid random adapter without loading any model."""
        start = time.perf_counter()
        synthesize_adapter(
            out_dir,
            r=int(self._get("lora_r", 16)),
            lora_alpha=int(self._get("lora_alpha", 32)),
            lora_dropout=float(self._get("lora_dropout", 0.05)),
            target_modules=list(
                self._get(
                    "lora_target_modules",
                    ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
                )
            ),
            base_model_name=self.model_name,
            num_layers=int(self._get("dry_run_layers", 4)),
            hidden_size=int(self._get("dry_run_hidden", 256)),
            # `hash()` on str is salted per process; derive a stable offset
            # instead so dry runs stay reproducible across invocations.
            seed=int(self._get("seed", 42)) + _stable_offset(client_id),
            init_from=init_adapter_path,
        )
        elapsed = time.perf_counter() - start
        self._write_training_manifest(
            out_dir,
            {
                **resume_identity,
                "status": "completed",
                "num_samples": self._manifest_sample_count(dataset_path),
                "trainer_backend": "dry_run",
                "training_time_sec": round(elapsed, 4),
                "model_hash": sha256_path(out_dir),
                "completed_at": int(time.time()),
            },
        )
        LOGGER.info("[dry-run] Synthetic adapter for %s written to %s (%.3fs)", client_id, out_dir, elapsed)
        self.history.append(
            {
                "client_id": client_id,
                "dry_run": True,
                "training_time_sec": round(elapsed, 4),
                "adapter_path": str(out_dir),
            }
        )
        return str(out_dir), round(elapsed, 4)

    # =========================================================================
    # Per-client completion manifest
    # =========================================================================
    def _training_identity(
        self,
        dataset_path: PathLike,
        init_adapter_path: Optional[PathLike],
        client_id: str,
        epochs: float,
        max_samples: Optional[int],
    ) -> Dict[str, Any]:
        """Identity proving that an on-disk client artefact is safe to reuse."""
        dataset = Path(dataset_path).resolve()
        dataset_hash = sha256_path(dataset) if dataset.is_file() else None
        init_path = Path(init_adapter_path).resolve() if init_adapter_path else None
        init_hash = sha256_path(init_path) if init_path and init_path.exists() else None
        identity: Dict[str, Any] = {
            "checkpoint_version": 1,
            "config_fingerprint": compute_fingerprint(self.config),
            "client_id": client_id,
            "dataset": str(dataset),
            "dataset_hash": dataset_hash,
            "init_adapter_path": str(init_path) if init_path else None,
            "init_adapter_hash": init_hash,
            "epochs": float(epochs),
            "max_samples": int(max_samples) if max_samples is not None else None,
        }
        import hashlib

        payload = json.dumps(identity, sort_keys=True, default=str).encode("utf-8")
        identity["resume_fingerprint"] = hashlib.sha256(payload).hexdigest()
        return identity

    @staticmethod
    def _read_training_manifest(out_dir: Path) -> Dict[str, Any]:
        path = out_dir / "fedchain_client_meta.json"
        try:
            with open(path, "r", encoding="utf-8") as handle:
                value = json.load(handle)
            return value if isinstance(value, dict) else {}
        except (OSError, ValueError, TypeError):
            return {}

    @staticmethod
    def _write_training_manifest(out_dir: Path, payload: Dict[str, Any]) -> None:
        _write_client_metadata(out_dir, payload)

    @staticmethod
    def _manifest_sample_count(dataset_path: PathLike) -> int:
        try:
            with open(dataset_path, "r", encoding="utf-8") as handle:
                return sum(1 for line in handle if line.strip())
        except OSError:
            return 0

    # =========================================================================
    # Step-level crash recovery
    # =========================================================================
    @staticmethod
    def _find_step_checkpoint(out_dir: Path) -> Optional[str]:
        """Locate the newest usable ``checkpoint-N`` directory, if any."""
        if not out_dir.is_dir():
            return None

        try:
            from transformers.trainer_utils import get_last_checkpoint

            found = get_last_checkpoint(str(out_dir))
            if found:
                return found
        except Exception as exc:  # older/newer transformers, or an odd layout
            LOGGER.debug("get_last_checkpoint unavailable (%s); globbing instead.", exc)

        candidates = []
        for entry in out_dir.glob("checkpoint-*"):
            if not entry.is_dir():
                continue
            suffix = entry.name.split("-")[-1]
            if suffix.isdigit() and (entry / "trainer_state.json").exists():
                candidates.append((int(suffix), entry))
        if not candidates:
            return None
        return str(max(candidates)[1])

    @staticmethod
    def _purge_step_checkpoints(out_dir: Path, reason: str = "") -> int:
        """Delete ``checkpoint-*`` directories; returns how many were removed."""
        import shutil

        if not out_dir.is_dir():
            return 0
        removed = 0
        for entry in out_dir.glob("checkpoint-*"):
            if entry.is_dir():
                shutil.rmtree(entry, ignore_errors=True)
                removed += 1
        if removed:
            LOGGER.debug("Removed %d step checkpoint(s) from %s (%s)", removed, out_dir, reason)
        return removed

    # =========================================================================
    # Memory management
    # =========================================================================
    def unload_model(self) -> None:
        """Drop the cached base model and reclaim VRAM."""
        if self._base_model is not None:
            LOGGER.debug("Releasing the base model from memory.")
            del self._base_model
            self._base_model = None
        gc.collect()
        free_cuda_memory()

    def cleanup(self) -> None:
        """Release everything this trainer holds (model + tokenizer)."""
        self.unload_model()
        self._tokenizer = None
        gc.collect()
        free_cuda_memory()
        LOGGER.debug("LocalTrainer cleaned up.")

    def __enter__(self) -> "LocalTrainer":
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        self.cleanup()
        return False


# =============================================================================
# Module-level helpers
# =============================================================================
def _accepts_dtype_kwarg() -> bool:
    """True when the installed transformers prefers ``dtype`` over ``torch_dtype``."""
    try:
        import transformers

        parts = str(transformers.__version__).split(".")
        major, minor = int(parts[0]), int(parts[1])
        return (major, minor) >= (4, 56)
    except Exception:
        return False


def _count_parameters(model: Any) -> Tuple[int, int]:
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return trainable, total


def _fmt(value: Any) -> str:
    return f"{value:.4f}" if isinstance(value, (int, float)) else str(value)


def _stable_offset(text: str, modulo: int = 1000) -> int:
    """Process-independent small integer derived from a string."""
    import hashlib

    return int(hashlib.sha256(text.encode("utf-8")).hexdigest()[:8], 16) % modulo


def _write_client_metadata(out_dir: Path, payload: Dict[str, Any]) -> None:
    """Atomically write the per-client recovery/provenance sidecar."""
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "fedchain_client_meta.json"
    tmp_path = path.with_name(path.name + ".tmp")
    with open(tmp_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, default=str)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp_path, path)


def synthesize_adapter(
    output_dir: PathLike,
    r: int = 16,
    lora_alpha: int = 32,
    lora_dropout: float = 0.05,
    target_modules: Optional[Sequence[str]] = None,
    base_model_name: str = "Qwen/Qwen2.5-1.5B-Instruct",
    num_layers: int = 4,
    hidden_size: int = 256,
    seed: int = 42,
    init_from: Optional[PathLike] = None,
) -> Path:
    """Write a small but structurally valid PEFT adapter (``dry_run`` support).

    The tensor names mirror what PEFT emits for a Qwen2 model, so the resulting
    directory exercises the real aggregation, hashing, IPFS and blockchain code
    paths without downloading a 3 GB checkpoint. When ``init_from`` points at an
    existing adapter, its weights are perturbed rather than regenerated, which
    makes successive dry-run rounds behave like genuine local updates.
    """
    import torch
    from safetensors.torch import load_file, save_file

    target_modules = list(
        target_modules or ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    )
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    generator = torch.Generator().manual_seed(int(seed))

    state_dict: Dict[str, "torch.Tensor"] = {}
    previous: Dict[str, "torch.Tensor"] = {}
    if init_from is not None:
        previous_file = Path(init_from) / "adapter_model.safetensors"
        if previous_file.exists():
            previous = load_file(str(previous_file))

    for layer in range(num_layers):
        for module in target_modules:
            prefix = f"base_model.model.model.layers.{layer}.{_module_parent(module)}.{module}"
            a_key = f"{prefix}.lora_A.weight"
            b_key = f"{prefix}.lora_B.weight"
            if a_key in previous and b_key in previous:
                state_dict[a_key] = previous[a_key] + 0.01 * torch.randn(
                    previous[a_key].shape, generator=generator
                )
                state_dict[b_key] = previous[b_key] + 0.01 * torch.randn(
                    previous[b_key].shape, generator=generator
                )
            else:
                state_dict[a_key] = torch.randn(r, hidden_size, generator=generator) * 0.02
                state_dict[b_key] = torch.zeros(hidden_size, r)

    # `format: pt` metadata is required by PEFT's safetensors loader.
    save_file(state_dict, str(out_dir / "adapter_model.safetensors"), metadata={"format": "pt"})

    adapter_config = {
        "alpha_pattern": {},
        "auto_mapping": None,
        "base_model_name_or_path": base_model_name,
        "bias": "none",
        "fan_in_fan_out": False,
        "inference_mode": False,
        "init_lora_weights": True,
        "layers_pattern": None,
        "layers_to_transform": None,
        "loftq_config": {},
        "lora_alpha": lora_alpha,
        "lora_dropout": lora_dropout,
        "megatron_config": None,
        "megatron_core": "megatron.core",
        "modules_to_save": None,
        "peft_type": "LORA",
        "r": r,
        "rank_pattern": {},
        "revision": None,
        "target_modules": target_modules,
        "task_type": "CAUSAL_LM",
        "use_dora": False,
        "use_rslora": False,
    }
    with open(out_dir / "adapter_config.json", "w", encoding="utf-8") as handle:
        json.dump(adapter_config, handle, indent=2)

    return out_dir


def _module_parent(module_name: str) -> str:
    """Qwen2 places attention and MLP projections under different parents."""
    return "self_attn" if module_name in {"q_proj", "k_proj", "v_proj", "o_proj"} else "mlp"
