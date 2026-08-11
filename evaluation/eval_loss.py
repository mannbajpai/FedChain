"""
FedChain :: Validation loss, perplexity and generation quality
=============================================================

``Evaluator`` scores a global LoRA adapter on a fixed held-out split
(the first 500 Dolly-15k records) and reports:

* **Validation loss** - token-weighted negative log-likelihood

  .. math:: \\mathcal{L} = \\frac{\\sum_j \\text{NLL}_j}{\\sum_j T_j}

  The mean is token-weighted, not sample-weighted: every scored token
  contributes equally, so a batch of short answers cannot dominate a batch of
  long ones. This keeps the metric comparable across the four experiments even
  though their adapters produce different answer lengths.

* **Perplexity** - :math:`\\exp(\\mathcal{L})`

* **ROUGE-L** and **BLEU** - instruction-following overlap, measured by greedily
  decoding a small subset (50 prompts by default) and comparing against the
  reference responses. Uses the Hugging Face ``evaluate`` library when it is
  installed; otherwise a self-contained implementation of LCS-based ROUGE-L and
  smoothed corpus BLEU-4 is used, so this metric never disappears from the
  results table just because an optional dependency is missing. Both are
  reported on the 0-1 scale.

The same quantised base model as training is used (4-bit NF4 on CUDA, fp32 on
CPU), and the adapter is attached with ``PeftModel.from_pretrained``. The model
is torn down after every call so per-round evaluation does not accumulate VRAM
on a 4 GB card.
"""

from __future__ import annotations

import logging
import math
import os
import re
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

from utils.common import (
    Timer,
    cuda_peak_memory_mb,
    format_duration,
    free_cuda_memory,
    get_device,
    hf_auth_kwargs,
)
from utils.config import resolve_path

LOGGER = logging.getLogger(__name__)

PathLike = Union[str, os.PathLike]

SYSTEM_PROMPT = "You are a helpful assistant."

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

#: exp() of anything above this overflows float64; report inf instead of crashing.
_MAX_EXP_ARG = 709.0

#: Third-party modules `evaluate`'s rouge/bleu metric scripts import at load time.
_METRIC_STACK = ("nltk", "rouge_score", "evaluate")


def _diagnose_metric_stack() -> str:
    """Report which metric dependency actually failed to import, and where from.

    `evaluate` probes its metric script's dependencies with a bare
    ``importlib.import_module`` inside a ``try/except ImportError``, then raises
    a message naming the *declared* dependency:

        To be able to use evaluate-metric/rouge, you need to install the
        following dependencies['nltk'] using 'pip install nltk' for instance'

    That message is generated from the declaration, not from the failure, so it
    says ``['nltk']`` whether nltk is genuinely absent, installed into a
    different interpreter than the one running this process, or present but
    unimportable because one of *its* dependencies (regex, joblib, click, tqdm)
    is broken. All three read identically, and the real traceback is discarded.

    The baseline sweep lost generation metrics on all 36 runs to this, and
    `pip install nltk` reported "Requirement already satisfied" each time
    because it resolved against a different environment. So report the
    interpreter and the true per-module import result.
    """
    import importlib
    import sys

    parts = [f"interpreter={sys.executable}"]
    for name in _METRIC_STACK:
        try:
            module = importlib.import_module(name)
        except Exception as exc:  # ImportError, but a broken install can raise others
            parts.append(f"{name}=FAILED({type(exc).__name__}: {exc})")
        else:
            version = getattr(module, "__version__", "unknown")
            parts.append(f"{name}={version}")
    return "Metric stack: " + ", ".join(parts) + "."


class Evaluator:
    """Computes validation loss, perplexity, ROUGE-L, BLEU and latency.

    The base model is cached across calls when ``keep_model_loaded`` is true, so
    per-round evaluation does not repeatedly pay the load cost and only the
    adapter is swapped. On a 4 GB card the default is to release everything
    after each call instead.
    """

    def __init__(self, config: Any, device: Optional[str] = None) -> None:
        self.config = config

        preference = device or self._get("device", "auto")
        self.device, self.device_info = get_device(preference)
        self.use_cuda = self.device == "cuda"

        self.model_name: str = self._get("model_name", "Qwen/Qwen2.5-1.5B-Instruct")
        self.max_seq_length: int = int(self._get("eval_max_seq_length", self._get("max_seq_length", 512)))
        self.num_samples: int = int(self._get("eval_num_samples", 500))
        self.batch_size: int = int(self._get("eval_batch_size", 1))
        self.completion_only: bool = bool(self._get("eval_loss_on_completion_only", False))
        self.keep_model_loaded: bool = bool(self._get("keep_model_loaded", False))
        self.dry_run: bool = bool(self._get("dry_run", False))

        # --- generation-quality settings ---
        self.enable_generation_metrics: bool = bool(self._get("enable_generation_metrics", True))
        self.gen_num_samples: int = int(self._get("gen_num_samples", 50))
        self.gen_max_new_tokens: int = int(self._get("gen_max_new_tokens", 128))
        self.gen_max_prompt_tokens: int = int(self._get("gen_max_prompt_tokens", 384))

        self.use_4bit: bool = bool(self._get("load_in_4bit", True)) and self.use_cuda

        self._tokenizer: Any = None
        self._base_model: Any = None
        self._dataset: Any = None
        self._hf_rouge: Any = None
        self._hf_bleu: Any = None
        self._metric_backend: Optional[str] = None
        self.history: List[Dict[str, Any]] = []

        LOGGER.info(
            "Evaluator ready | %d loss samples | %d generation samples | seq_len=%d | device=%s",
            self.num_samples,
            self.gen_num_samples if self.enable_generation_metrics else 0,
            self.max_seq_length,
            self.device,
        )

    # -- config access -------------------------------------------------------
    def _get(self, key: str, default: Any = None) -> Any:
        try:
            return self.config.get(key, default)
        except AttributeError:
            return getattr(self.config, key, default)

    def _compute_dtype(self) -> Any:
        import torch

        requested = str(self._get("bnb_4bit_compute_dtype", "auto")).lower()
        if requested in {"float16", "fp16", "half"}:
            return torch.float16
        if requested in {"bfloat16", "bf16"}:
            return torch.bfloat16
        if requested in {"float32", "fp32"} or not self.use_cuda:
            return torch.float32
        return torch.bfloat16 if self.device_info.get("bf16_supported") else torch.float16

    # =========================================================================
    # Data
    # =========================================================================
    def load_eval_dataset(self) -> Any:
        """Load the fixed evaluation split, caching it across rounds.

        Resolution order:

        1. ``eval_data_path`` (written by ``data/prepare_data.py`` and held out
           of every client shard, so there is no train/eval leakage);
        2. the first ``eval_num_samples`` records of ``databricks/dolly-15k``
           straight from the Hub, if the local file is absent.
        """
        if self._dataset is not None:
            return self._dataset

        from datasets import load_dataset

        local_path = self._get("eval_data_path", "data/eval_500.jsonl")
        if local_path:
            candidate = resolve_path(local_path)
            if candidate.exists():
                dataset = load_dataset("json", data_files=str(candidate), split="train")
                if self.num_samples > 0 and len(dataset) > self.num_samples:
                    dataset = dataset.select(range(self.num_samples))
                LOGGER.info("Evaluation split: %d samples from %s", len(dataset), candidate.name)
                self._dataset = dataset
                return dataset
            LOGGER.warning(
                "%s not found; falling back to the first %d records of the Hub dataset. "
                "Those records may overlap the training shards - run "
                "`python data/prepare_data.py` to generate a properly held-out split.",
                candidate,
                self.num_samples,
            )

        dataset_name = self._get("eval_dataset_name", "databricks/databricks-dolly-15k")
        split = f"train[:{self.num_samples}]" if self.num_samples > 0 else "train"
        dataset = load_dataset(dataset_name, split=split, **hf_auth_kwargs())
        LOGGER.info("Evaluation split: %d samples from %s (%s)", len(dataset), dataset_name, split)
        self._dataset = dataset
        return dataset

    def _render(self, example: Dict[str, Any]) -> Tuple[str, str, str]:
        """Return ``(prompt, full_text, reference_response)`` for one record."""
        instruction = str(example.get("instruction", "") or "")
        context = str(example.get("context", "") or example.get("input", "") or "")
        response = str(example.get("response", "") or example.get("output", "") or "")

        tokenizer = self.load_tokenizer()
        if bool(self._get("use_chat_template", True)) and getattr(tokenizer, "chat_template", None):
            user_content = f"{instruction}\n\n{context}".strip() if context else instruction
            history = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_content},
            ]
            try:
                prompt = tokenizer.apply_chat_template(history, tokenize=False, add_generation_prompt=True)
                full = tokenizer.apply_chat_template(
                    history + [{"role": "assistant", "content": response}], tokenize=False
                )
                return prompt, full, response
            except Exception as exc:  # pragma: no cover
                LOGGER.debug("Chat template failed during evaluation (%s); using Alpaca.", exc)

        template = ALPACA_WITH_CONTEXT if context else ALPACA_NO_CONTEXT
        prompt = template.format(instruction=instruction, context=context)
        eos = getattr(tokenizer, "eos_token", "") or ""
        return prompt, prompt + response + eos, response

    # =========================================================================
    # Model
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
        tokenizer.padding_side = "right"
        self._tokenizer = tokenizer
        return tokenizer

    def _load_base_model(self) -> Any:
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
        load_kwargs[_dtype_kwarg_name()] = compute_dtype

        if self.use_4bit:
            from transformers import BitsAndBytesConfig

            load_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type=str(self._get("bnb_4bit_quant_type", "nf4")),
                bnb_4bit_use_double_quant=bool(self._get("bnb_4bit_use_double_quant", True)),
                bnb_4bit_compute_dtype=compute_dtype,
            )
            load_kwargs["device_map"] = {"": 0}
        elif self.use_cuda:
            load_kwargs["device_map"] = {"": 0}

        with Timer("Evaluation base model load", LOGGER):
            try:
                model = AutoModelForCausalLM.from_pretrained(self.model_name, **load_kwargs)
            except Exception as exc:
                if "attn_implementation" in load_kwargs:
                    LOGGER.warning("SDPA rejected during evaluation (%s); using the default kernel.", exc)
                    load_kwargs.pop("attn_implementation")
                    model = AutoModelForCausalLM.from_pretrained(self.model_name, **load_kwargs)
                else:
                    raise

        if not self.use_cuda:
            model = model.to(torch.device("cpu"))

        model.config.use_cache = True  # no gradient checkpointing during inference
        self._base_model = model
        return model

    def _attach_adapter(self, base_model: Any, adapter_path: Optional[PathLike]) -> Any:
        if adapter_path is None:
            LOGGER.info("Evaluating the base model (no adapter attached).")
            return base_model

        path = Path(adapter_path)
        if not path.exists():
            LOGGER.warning("Adapter %s does not exist; evaluating the base model instead.", path)
            return base_model

        from peft import PeftModel

        LOGGER.info("Attaching adapter %s", path)
        return PeftModel.from_pretrained(base_model, str(path), is_trainable=False)

    # =========================================================================
    # Evaluation
    # =========================================================================
    def evaluate(self, adapter_path: Optional[PathLike] = None, label: str = "") -> Dict[str, Any]:
        """Score ``adapter_path``.

        Returns
        -------
        dict
            ``loss``, ``perplexity``, ``rouge_l``, ``bleu``,
            ``eval_latency_sec`` (total, including generation),
            ``loss_latency_sec``, ``generation_latency_sec``, ``num_samples``,
            ``num_tokens``, ``adapter_path`` and ``peak_vram_mb``.
        """
        if self.dry_run:
            return self._evaluate_dry_run(adapter_path, label)

        import torch

        start = time.perf_counter()
        model = None
        try:
            dataset = self.load_eval_dataset()
            tokenizer = self.load_tokenizer()
            base_model = self._load_base_model()
            model = self._attach_adapter(base_model, adapter_path)
            model.eval()

            device = torch.device("cuda" if self.use_cuda else "cpu")

            # ---------------- 1. loss / perplexity ---------------------------
            loss_start = time.perf_counter()
            total_nll = 0.0
            total_tokens = 0
            skipped = 0

            LOGGER.info("Scoring %d evaluation samples for loss ...", len(dataset))
            with torch.no_grad():
                for index in range(len(dataset)):
                    example = dataset[index]
                    prompt, full_text, _ = self._render(example)

                    encoded = tokenizer(
                        full_text,
                        return_tensors="pt",
                        truncation=True,
                        max_length=self.max_seq_length,
                    )
                    input_ids = encoded["input_ids"].to(device)
                    attention_mask = encoded["attention_mask"].to(device)

                    if input_ids.shape[1] < 2:
                        skipped += 1
                        continue

                    labels = input_ids.clone()
                    if self.completion_only:
                        prompt_len = len(
                            tokenizer(prompt, truncation=True, max_length=self.max_seq_length)["input_ids"]
                        )
                        prompt_len = min(prompt_len, input_ids.shape[1] - 1)
                        labels[:, :prompt_len] = -100

                    # HF shifts internally: position 0 never contributes a term.
                    num_scored = int((labels[:, 1:] != -100).sum().item())
                    if num_scored == 0:
                        skipped += 1
                        continue

                    outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                    loss_value = float(outputs.loss.item())
                    if not math.isfinite(loss_value):
                        LOGGER.warning("Non-finite loss at sample %d; skipping it.", index)
                        skipped += 1
                        continue

                    # outputs.loss is the *mean* over scored tokens; undo the
                    # mean so the aggregate stays token-weighted.
                    total_nll += loss_value * num_scored
                    total_tokens += num_scored

                    if (index + 1) % 100 == 0:
                        running = total_nll / max(total_tokens, 1)
                        LOGGER.info(
                            "  %d/%d scored | running loss=%.4f ppl=%.4f",
                            index + 1,
                            len(dataset),
                            running,
                            _safe_exp(running),
                        )

            if total_tokens == 0:
                raise RuntimeError("Evaluation scored zero tokens - the split may be empty or malformed.")

            loss = total_nll / total_tokens
            perplexity = _safe_exp(loss)
            loss_latency = time.perf_counter() - loss_start

            # ---------------- 2. ROUGE-L / BLEU ------------------------------
            generation_metrics = self._compute_generation_metrics(model, tokenizer, dataset, device)

            latency = time.perf_counter() - start
            metrics: Dict[str, Any] = {
                "label": label or (Path(adapter_path).name if adapter_path else "base_model"),
                "adapter_path": str(adapter_path) if adapter_path else None,
                "loss": round(loss, 6),
                "perplexity": round(perplexity, 6),
                "rouge_l": generation_metrics["rouge_l"],
                "bleu": generation_metrics["bleu"],
                "eval_latency_sec": round(latency, 4),
                "loss_latency_sec": round(loss_latency, 4),
                "generation_latency_sec": generation_metrics["generation_latency_sec"],
                "num_samples": len(dataset) - skipped,
                "num_samples_skipped": skipped,
                "num_tokens": total_tokens,
                "num_generation_samples": generation_metrics["num_generation_samples"],
                "generation_metric_backend": generation_metrics["backend"],
                "completion_only": self.completion_only,
                "peak_vram_mb": cuda_peak_memory_mb(),
            }

            LOGGER.info(
                "Evaluation: loss=%.4f ppl=%.4f rougeL=%s bleu=%s over %s tokens in %s",
                loss,
                perplexity,
                _fmt(generation_metrics["rouge_l"]),
                _fmt(generation_metrics["bleu"]),
                f"{total_tokens:,}",
                format_duration(latency),
            )
            self.history.append(metrics)
            return metrics

        finally:
            # Drop the PEFT wrapper but keep the (expensive) base model unless
            # the config asks us to release everything.
            #
            # `base_model` is a local aliasing `self._base_model`. `unload()`
            # clears the attribute, but this local kept the model alive across
            # `free_cuda_memory()`, so one model's worth stayed resident per
            # call: evaluation peak VRAM climbed 1163 -> 1703 -> 2243 MB over
            # three rounds of the same 360M run. Same defect as `train_client`.
            if model is not None and model is not self._base_model:
                del model
            model = None
            base_model = None
            if not self.keep_model_loaded:
                self.unload()
            free_cuda_memory()

    # =========================================================================
    # Generation quality (ROUGE-L / BLEU)
    # =========================================================================
    def _compute_generation_metrics(
        self, model: Any, tokenizer: Any, dataset: Any, device: Any
    ) -> Dict[str, Any]:
        """Greedily decode a subset and score overlap against the references."""
        blank = {
            "rouge_l": None,
            "bleu": None,
            "generation_latency_sec": 0.0,
            "num_generation_samples": 0,
            "backend": None,
        }
        if not self.enable_generation_metrics or self.gen_num_samples <= 0:
            LOGGER.info("Generation metrics disabled; skipping ROUGE-L / BLEU.")
            return blank

        import torch

        subset_size = min(self.gen_num_samples, len(dataset))
        LOGGER.info("Generating %d responses for ROUGE-L / BLEU ...", subset_size)

        predictions: List[str] = []
        references: List[str] = []
        start = time.perf_counter()

        try:
            with torch.no_grad():
                for index in range(subset_size):
                    example = dataset[index]
                    prompt, _, reference = self._render(example)
                    if not reference.strip():
                        continue

                    encoded = tokenizer(
                        prompt,
                        return_tensors="pt",
                        truncation=True,
                        max_length=self.gen_max_prompt_tokens,
                    )
                    input_ids = encoded["input_ids"].to(device)
                    attention_mask = encoded["attention_mask"].to(device)

                    generated = model.generate(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        max_new_tokens=self.gen_max_new_tokens,
                        do_sample=False,          # greedy: deterministic across runs
                        num_beams=1,
                        temperature=None,
                        top_p=None,
                        top_k=None,
                        pad_token_id=tokenizer.pad_token_id,
                        eos_token_id=tokenizer.eos_token_id,
                    )
                    # Keep only the continuation, never the echoed prompt.
                    completion_ids = generated[0][input_ids.shape[1]:]
                    prediction = tokenizer.decode(completion_ids, skip_special_tokens=True).strip()

                    predictions.append(prediction)
                    references.append(reference.strip())

                    if (index + 1) % 10 == 0:
                        LOGGER.info("  generated %d/%d", index + 1, subset_size)

        except Exception as exc:
            LOGGER.error("Generation for ROUGE-L / BLEU failed: %s", exc)
            blank["generation_latency_sec"] = round(time.perf_counter() - start, 4)
            return blank

        latency = time.perf_counter() - start
        if not predictions:
            LOGGER.warning("No usable generations were produced; ROUGE-L / BLEU unavailable.")
            blank["generation_latency_sec"] = round(latency, 4)
            return blank

        scores = self._score_overlap(predictions, references)
        LOGGER.info(
            "Generation quality over %d samples: ROUGE-L=%.4f BLEU=%.4f (backend=%s, %s)",
            len(predictions),
            scores["rouge_l"],
            scores["bleu"],
            scores["backend"],
            format_duration(latency),
        )
        return {
            "rouge_l": round(scores["rouge_l"], 6),
            "bleu": round(scores["bleu"], 6),
            "generation_latency_sec": round(latency, 4),
            "num_generation_samples": len(predictions),
            "backend": scores["backend"],
        }

    def _score_overlap(self, predictions: Sequence[str], references: Sequence[str]) -> Dict[str, Any]:
        """ROUGE-L and BLEU via ``evaluate``, with a self-contained fallback."""
        hf_scores = self._score_with_evaluate(predictions, references)
        if hf_scores is not None:
            return hf_scores

        rouge_l = rouge_l_fmeasure(predictions, references)
        bleu = corpus_bleu(predictions, references)
        return {"rouge_l": rouge_l, "bleu": bleu, "backend": "builtin"}

    def _score_with_evaluate(
        self, predictions: Sequence[str], references: Sequence[str]
    ) -> Optional[Dict[str, Any]]:
        """Try the Hugging Face ``evaluate`` library; ``None`` if unusable."""
        if self._metric_backend == "builtin":
            return None
        try:
            import evaluate as hf_evaluate
        except Exception as exc:
            self._fallback_to_builtin(
                f"the `evaluate` library is unavailable ({exc}). {_diagnose_metric_stack()}",
                level=LOGGER.info,
            )
            return None

        try:
            if self._hf_rouge is None:
                self._hf_rouge = hf_evaluate.load("rouge")
            if self._hf_bleu is None:
                self._hf_bleu = hf_evaluate.load("bleu")

            rouge_result = self._hf_rouge.compute(
                predictions=list(predictions), references=list(references), use_stemmer=True
            )
            # `bleu` divides by zero when a prediction is empty; filter those out.
            paired = [(p, r) for p, r in zip(predictions, references) if p.strip()]
            if paired:
                bleu_result = self._hf_bleu.compute(
                    predictions=[p for p, _ in paired],
                    references=[[r] for _, r in paired],
                )
                bleu_value = float(bleu_result.get("bleu", 0.0))
            else:
                bleu_value = 0.0

            self._metric_backend = "evaluate"
            return {
                "rouge_l": float(rouge_result.get("rougeL", 0.0)),
                "bleu": bleu_value,
                "backend": "evaluate",
            }
        except Exception as exc:
            self._fallback_to_builtin(
                f"the `evaluate` library failed to score ({exc}). {_diagnose_metric_stack()}"
            )
            return None

    def _fallback_to_builtin(self, reason: str, level: Any = None) -> None:
        """Switch to the built-in ROUGE-L / BLEU, or refuse to.

        The fallback keeps a run alive when an optional dependency is missing,
        which is usually what you want. It is also how a whole benchmark ends up
        silently scored with a non-standard ROUGE implementation whose absolute
        values cannot be compared against published numbers - the baseline sweep
        did exactly that, because `nltk` was absent despite being pinned in
        requirements.txt.

        Set `require_metric_backend: evaluate` for a paper run and a missing
        dependency becomes a hard failure at the first evaluation, which costs
        minutes, instead of a footnote discovered after 24 GPU-hours.
        """
        required = str(self._get("require_metric_backend", "") or "").strip().lower()
        if required in {"evaluate", "hf", "huggingface"}:
            raise RuntimeError(
                f"require_metric_backend={required!r} but {reason}. "
                "Install the metric stack (`pip install evaluate rouge-score nltk`) "
                "or unset require_metric_backend to allow the built-in implementation."
            )
        (level or LOGGER.warning)(
            "Falling back to the built-in ROUGE-L / BLEU implementation: %s. "
            "Absolute values are not comparable with published `evaluate` numbers; "
            "the backend actually used is recorded as `generation_metric_backend` "
            "in the metrics file.",
            reason,
        )
        self._metric_backend = "builtin"

    # =========================================================================
    # Dry run
    # =========================================================================
    def _evaluate_dry_run(self, adapter_path: Optional[PathLike], label: str) -> Dict[str, Any]:
        """Deterministic placeholder metrics so plumbing can be tested offline."""
        start = time.perf_counter()
        loss = 2.0
        metrics = {
            "label": label or "dry_run",
            "adapter_path": str(adapter_path) if adapter_path else None,
            "loss": round(loss, 6),
            "perplexity": round(_safe_exp(loss), 6),
            "rouge_l": 0.0,
            "bleu": 0.0,
            "eval_latency_sec": round(time.perf_counter() - start, 6),
            "loss_latency_sec": 0.0,
            "generation_latency_sec": 0.0,
            "num_samples": 0,
            "num_samples_skipped": 0,
            "num_tokens": 0,
            "num_generation_samples": 0,
            "generation_metric_backend": None,
            "completion_only": self.completion_only,
            "peak_vram_mb": None,
            "dry_run": True,
        }
        LOGGER.info("[dry-run] Emitting placeholder evaluation metrics (loss=%.4f).", loss)
        self.history.append(metrics)
        return metrics

    # =========================================================================
    # Teardown
    # =========================================================================
    def unload(self) -> None:
        """Release the cached base model."""
        if self._base_model is not None:
            del self._base_model
            self._base_model = None
        free_cuda_memory()

    def cleanup(self) -> None:
        """Release the model, tokenizer and cached dataset."""
        self.unload()
        self._tokenizer = None
        self._dataset = None
        self._hf_rouge = None
        self._hf_bleu = None
        free_cuda_memory()
        LOGGER.debug("Evaluator cleaned up.")

    def __enter__(self) -> "Evaluator":
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        self.cleanup()
        return False


# =============================================================================
# Self-contained ROUGE-L / BLEU
# =============================================================================
_TOKEN_PATTERN = re.compile(r"[a-z0-9]+")


def simple_tokenize(text: str) -> List[str]:
    """Lowercase alphanumeric tokenisation, mirroring ``rouge_score``'s default."""
    return _TOKEN_PATTERN.findall(str(text).lower())


def lcs_length(a: Sequence[str], b: Sequence[str]) -> int:
    """Length of the longest common subsequence, O(len(a) * len(b)) time.

    Only two rows of the DP table are kept, so memory is O(min(len(a), len(b)))
    after the shorter sequence is placed on the inner axis.
    """
    if not a or not b:
        return 0
    if len(a) < len(b):
        a, b = b, a

    previous = [0] * (len(b) + 1)
    for token_a in a:
        current = [0] * (len(b) + 1)
        for j, token_b in enumerate(b, start=1):
            if token_a == token_b:
                current[j] = previous[j - 1] + 1
            else:
                current[j] = current[j - 1] if current[j - 1] >= previous[j] else previous[j]
        previous = current
    return previous[len(b)]


def rouge_l_fmeasure(predictions: Sequence[str], references: Sequence[str]) -> float:
    """Mean sentence-level ROUGE-L F1 over a corpus (0-1 scale).

    Matches ``rouge_score``'s ``rougeL`` aggregation: the F-measure is computed
    per pair and then averaged, rather than pooling counts across the corpus.
    """
    if not predictions:
        return 0.0

    scores: List[float] = []
    for prediction, reference in zip(predictions, references):
        pred_tokens = simple_tokenize(prediction)
        ref_tokens = simple_tokenize(reference)
        if not pred_tokens or not ref_tokens:
            scores.append(0.0)
            continue

        overlap = lcs_length(pred_tokens, ref_tokens)
        if overlap == 0:
            scores.append(0.0)
            continue

        precision = overlap / len(pred_tokens)
        recall = overlap / len(ref_tokens)
        scores.append(2.0 * precision * recall / (precision + recall))

    return sum(scores) / len(scores)


def corpus_bleu(
    predictions: Sequence[str], references: Sequence[str], max_n: int = 4
) -> float:
    """Smoothed corpus-level BLEU-``max_n`` (0-1 scale).

    Clipped n-gram counts are pooled over the corpus and combined with the
    standard brevity penalty. Add-one smoothing is applied to orders **n >= 2
    only**, following sacrebleu's ``exp`` smoothing: a single missing 4-gram
    match should not collapse the whole score to zero on the small 50-prompt
    subset this runs over, but a prediction sharing *no* unigram with its
    reference must still score 0 rather than an inflated floor value.
    """
    if not predictions:
        return 0.0

    clipped = [0] * max_n
    totals = [0] * max_n
    prediction_length = 0
    reference_length = 0

    for prediction, reference in zip(predictions, references):
        pred_tokens = simple_tokenize(prediction)
        ref_tokens = simple_tokenize(reference)
        prediction_length += len(pred_tokens)
        reference_length += len(ref_tokens)

        for order in range(1, max_n + 1):
            pred_ngrams = _ngram_counts(pred_tokens, order)
            ref_ngrams = _ngram_counts(ref_tokens, order)
            totals[order - 1] += max(0, len(pred_tokens) - order + 1)
            for ngram, count in pred_ngrams.items():
                clipped[order - 1] += min(count, ref_ngrams.get(ngram, 0))

    log_precision_sum = 0.0
    for order in range(max_n):
        numerator, denominator = clipped[order], totals[order]
        if denominator == 0:
            # The corpus is shorter than this n-gram order; BLEU-max_n is
            # undefined, which standard implementations report as 0.
            return 0.0
        if numerator == 0:
            if order == 0:
                # No unigram overlap at all: the score is genuinely zero.
                return 0.0
            # Add-one (Chen & Cherry smoothing 1) on unmatched higher orders.
            numerator, denominator = 1, denominator + 1
        log_precision_sum += math.log(numerator / denominator)

    geometric_mean = math.exp(log_precision_sum / max_n)

    if prediction_length == 0:
        return 0.0
    if prediction_length > reference_length:
        brevity_penalty = 1.0
    else:
        brevity_penalty = math.exp(1.0 - reference_length / prediction_length)

    return brevity_penalty * geometric_mean


def _ngram_counts(tokens: Sequence[str], order: int) -> Counter:
    if len(tokens) < order:
        return Counter()
    return Counter(tuple(tokens[i : i + order]) for i in range(len(tokens) - order + 1))


# =============================================================================
# Small helpers
# =============================================================================
def _safe_exp(value: float) -> float:
    """``exp`` that saturates instead of raising ``OverflowError``."""
    if value > _MAX_EXP_ARG:
        return float("inf")
    return float(math.exp(value))


def _fmt(value: Any) -> str:
    return f"{value:.4f}" if isinstance(value, (int, float)) else "n/a"


def _dtype_kwarg_name() -> str:
    """``dtype`` on transformers >= 4.56, ``torch_dtype`` before that."""
    try:
        import transformers

        parts = str(transformers.__version__).split(".")
        return "dtype" if (int(parts[0]), int(parts[1])) >= (4, 56) else "torch_dtype"
    except Exception:
        return "torch_dtype"
