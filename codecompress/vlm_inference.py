"""
VLM inference for image-based code understanding.

Supports two backends:
  - Local: Qwen2.5-VL via HuggingFace transformers (runs on GPU)
  - API:   OpenAI-compatible endpoints (OpenRouter, OpenAI, Google, etc.)

Usage:
    from codecompress.vlm_inference import LocalVLM, ApiVLM

    # Local inference (Qwen2.5-VL-7B)
    vlm = LocalVLM("Qwen/Qwen2.5-VL-7B-Instruct")
    completion = vlm.generate(images=[pil_image], text_prompt="Complete this code:")

    # API inference (OpenRouter)
    vlm = ApiVLM(model="qwen/qwen-2.5-vl-7b-instruct", api_key="...", base_url="https://openrouter.ai/api/v1")
    completion = vlm.generate(images=[pil_image], text_prompt="Complete this code:")

Reference implementation: CodeOCR/CodeOCR/client.py (read-only, not modified).
"""

from __future__ import annotations

import base64
import glob
import io
import os
import re
import time
from abc import ABC, abstractmethod

from loguru import logger
from PIL import Image


def _hf_snapshot_download(repo_id, revision=None, local_files_only=True):
    """Thin seam over huggingface_hub.snapshot_download (monkeypatched in tests)."""
    from huggingface_hub import snapshot_download

    return snapshot_download(repo_id, revision=revision, local_files_only=local_files_only)


_FULL_HEX_40 = re.compile(r"^[0-9a-f]{40}$")


def resolve_model_snapshot(model_name, revision):
    """Resolve ``model_name`` to a cached snapshot directory pinned at ``revision``.

    ``refs/main`` of a cached repo drifts (the Gemma repository's pointer moved
    to a WEIGHTLESS metadata-only snapshot), and HF_HUB_OFFLINE + an unpinned
    name is therefore NOT reproducibility. With ``revision`` set this resolves offline to the exact
    snapshot directory and fails loudly when (a) the revision is not a full
    40-hex commit id (short ids / branch names can move), (b) the cache does
    not hold that revision, (c) the snapshot carries no ``*.safetensors`` (the
    weightless class), or (d) the resolved directory is not the requested
    revision. ``revision`` None/"" returns ``model_name`` unchanged (unpinned).
    """
    if not revision:
        return model_name
    if not _FULL_HEX_40.match(str(revision)):
        raise ValueError(
            f"vlm_revision must be a full 40-hex HF commit id, got {revision!r} "
            "(short ids and branch names are not pins)"
        )
    path = str(_hf_snapshot_download(model_name, revision=revision, local_files_only=True))
    if os.path.basename(path.rstrip("/")) != revision:
        raise RuntimeError(
            f"resolved snapshot dir {path!r} does not end with revision {revision} "
            f"for {model_name}"
        )
    if not glob.glob(os.path.join(path, "*.safetensors")):
        raise RuntimeError(
            f"snapshot {model_name}@{revision} at {path} contains no *.safetensors "
            "(weightless metadata-only snapshot — refusing to load)"
        )
    return path

try:
    from vllm import LLM as _VLLM_LLM, SamplingParams as _VLLM_SamplingParams
    VLLM_AVAILABLE = True
except ImportError:
    VLLM_AVAILABLE = False
    _VLLM_LLM = None
    _VLLM_SamplingParams = None

# Run cuDNN conv3d probe once per process (see LocalVLM.__init__)
_CUDNN_CONV3D_CHECKED = False


class VLMBackend(ABC):
    """Abstract base class for VLM inference backends."""

    @abstractmethod
    def generate(
        self,
        images: list[Image.Image],
        text_prompt: str,
        max_tokens: int = 128,
        temperature: float = 0.0,
        system_prompt: str | None = None,
    ) -> str:
        """
        Generate a text completion given images and a text prompt.

        Args:
            images: List of PIL images (code rendered as images).
            text_prompt: Text instruction/context (e.g., current function to complete).
            max_tokens: Maximum tokens to generate.
            temperature: Sampling temperature (0 = greedy).
            system_prompt: optional system turn (fair-prompt arm); None keeps
                legacy behavior (see assemble_vlm_messages).

        Returns:
            Generated text completion.
        """
        ...

    @abstractmethod
    def generate_batch(
        self,
        batch: list[tuple[list[Image.Image], str]],
        max_tokens: int = 128,
        temperature: float = 0.0,
    ) -> list[str]:
        """
        Generate completions for a batch of (images, text_prompt) pairs.

        Default implementation processes sequentially. Subclasses can override
        for parallel API calls.
        """
        ...


def _images_to_base64(images: list[Image.Image]) -> list[str]:
    """Convert PIL images to base64-encoded PNG strings."""
    result = []
    for img in images:
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
        result.append(b64)
    return result


def assemble_vlm_messages(content, system_prompt, model_name):
    """Assemble the chat `messages` list for a VLM generate call.

    Shared by every backend so the image arm has a fair system-prompt channel
    (the text arm forwards ``benchmark_runner.system_prompt()``).

    Args:
        content: the user-turn content list (image items + the text item),
            already in the backend's native format.
        system_prompt: if not None, prepend a system turn carrying this text
            (the fair-prompt arm; "" is honored as an explicit empty system).
            If None, keep legacy behavior — an explicit empty system for
            Mistral-Small-3.1 (persona suppression) and no system otherwise.
        model_name: the model identifier (drives the Mistral legacy path).

    Returns:
        list[dict] — the messages list (system turn optional, then the user turn).
    """
    messages = []
    if system_prompt is not None:
        messages.append({"role": "system", "content": [{"type": "text", "text": system_prompt}]})
    elif "mistral-small-3.1" in (model_name or "").lower():
        messages.append({"role": "system", "content": [{"type": "text", "text": ""}]})
    messages.append({"role": "user", "content": content})
    return messages


# Complete <think>...</think> blocks anywhere in the output.
_THINK_BLOCK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)
# An UNTERMINATED <think> block: with a small max_tokens cap (128 in the image
# arm) a thinking model never reaches </think>, so the closed-block regex
# matches nothing and 128 tokens of chain-of-thought go straight to scoring.
# Everything
# from the opening tag onward is reasoning, never answer — drop it.
_THINK_OPEN_RE = re.compile(r"<think>.*\Z", re.DOTALL)
# GLM-4.1V wraps the final reply in <answer>...</answer> after the think
# block. Unwrap = drop the TAGS only (never the payload); handles a missing
# close tag from cap truncation. No other served model emits these tags.
_ANSWER_TAG_RE = re.compile(r"</?answer>")


def strip_think_blocks(text: str) -> str:
    """Remove chain-of-thought from a generation before it reaches scoring.

    Strips complete ``<think>...</think>`` blocks anywhere, then a trailing
    unterminated ``<think>...`` block (truncated thinking under a token cap),
    then UNWRAPS an ``<answer>...</answer>`` envelope (GLM-4.1V grammar:
    tags removed, payload kept; an unterminated ``<answer>`` under a
    cap keeps its content). Models that never emit these tags are untouched.
    Shared backstop for every backend; the primary fix is disabling thinking
    at the template level (probe_enable_thinking).
    """
    text = _THINK_BLOCK_RE.sub("", text)
    text = _THINK_OPEN_RE.sub("", text)
    return _ANSWER_TAG_RE.sub("", text)


def probe_enable_thinking(processor) -> dict:
    """Probe whether the model's chat template accepts ``enable_thinking``.

    Returns ``{"enable_thinking": False}`` when the template renders with the
    kwarg (hybrid-thinking models: Qwen3.5, MiniCPM-V 4.5, ...), else ``{}``.
    Probe-based, not hardcoded. Note this is a "doesn't hard-reject the
    kwarg" probe, not a capability probe: transformers silently forwards
    unknown kwargs into the Jinja context, so most renderable templates
    accept it — forwarding an ignored kwarg is a verified render no-op
    (Gemma-3/4, Qwen2.5/3-VL, Mistral all render byte-identically).

    Two-stage: first the processor's own apply_chat_template; if the PROCESSOR
    has no chat template (MiniCPM-V raises ValueError — its template lives on
    the tokenizer, which is also what vLLM renders with), fall back to
    ``processor.tokenizer``. The tokenizer probe uses STRING message content:
    MiniCPM-V's template calls ``message.content.startswith(...)``, so
    list-content raises inside Jinja and would false-negative the probe (vLLM
    itself flattens to string content before rendering — 'string' format).

    TypeError is ambiguous: kwarg rejection OR a template-internal error
    (mixed-type ``+``, bad slice). Disambiguate by re-rendering WITHOUT the
    kwarg: success -> genuine rejection -> ``{}``; failure -> this stage
    cannot render at all -> try the next stage (a masking early-return here
    would silently let <think> blocks leak into the output). Never
    raises (a probe quirk must not break engine init).
    """
    probe_processor = [{"role": "user", "content": [{"type": "text", "text": "x"}]}]
    probe_tokenizer = [{"role": "user", "content": "x"}]
    stages = (
        (processor, probe_processor),
        (getattr(processor, "tokenizer", None), probe_tokenizer),
    )
    for target, probe in stages:
        if target is None:
            continue
        try:
            target.apply_chat_template(
                probe, tokenize=False, add_generation_prompt=True,
                enable_thinking=False,
            )
            return {"enable_thinking": False}
        except TypeError:
            try:
                target.apply_chat_template(
                    probe, tokenize=False, add_generation_prompt=True,
                )
                return {}  # renders without the kwarg -> genuine rejection
            except Exception as e:  # noqa: BLE001 — stage can't render; try next
                logger.debug(f"enable_thinking probe stage skipped: {e}")
        except Exception as e:  # noqa: BLE001 — this stage can't render; try next
            logger.debug(f"enable_thinking probe stage skipped: {e}")
    return {}


# =============================================================================
# Local VLM (Qwen2.5-VL via HuggingFace transformers)
# =============================================================================

class LocalVLM(VLMBackend):
    """
    Local VLM inference using HuggingFace transformers.

    Loads the model onto GPU and runs inference locally. Suitable for
    Qwen2.5-VL-7B-Instruct and similar models.
    """

    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-VL-7B-Instruct",
        device_map: str = "auto",
        torch_dtype: str = "auto",
        gemma_max_soft_tokens: int | None = None,
        revision: str | None = None,
    ):
        import torch
        from transformers import AutoProcessor
        # transformers 5.x renamed AutoModelForVision2Seq → AutoModelForImageTextToText
        try:
            from transformers import AutoModelForImageTextToText as AutoVLM
        except ImportError:
            from transformers import AutoModelForVision2Seq as AutoVLM
        # Some VLMs (e.g. Kimi-VL) register their custom config only under
        # AutoModelForCausalLM, not under AutoModelForImageTextToText. Try
        # AutoVLM first; fall back to AutoModelForCausalLM on configuration
        # errors. AutoModelForCausalLM's config registry happens to include
        # the "Unrecognized class" failure mode via ValueError.
        from transformers import AutoModelForCausalLM

        # Compat shim: Kimi-VL's custom modeling_kimi_vl.py imports
        # `is_torch_fx_available` from transformers.utils.import_utils,
        # a symbol that was removed in transformers 5.x. Inject a stub
        # returning False so the import succeeds and Kimi's optional fx
        # optimization path is disabled (benign for inference).
        if "kimi" in model_name.lower():
            import transformers.utils.import_utils as _iu
            if not hasattr(_iu, "is_torch_fx_available"):
                _iu.is_torch_fx_available = lambda: False
                logger.info(
                    "Injected is_torch_fx_available stub for Kimi-VL "
                    "(transformers 5.x removed the symbol)."
                )

        # Workaround: cuDNN conv3d is broken on some torch/driver combos
        # (CUDNN_STATUS_NOT_INITIALIZED). Models with video/3D vision encoders
        # (e.g., Qwen3.5) need conv3d. Disabling cuDNN falls back to native
        # torch implementation which works fine, with minimal perf impact.
        # Only probe once per process (cached at module level).
        global _CUDNN_CONV3D_CHECKED
        if (torch.cuda.is_available() and torch.backends.cudnn.enabled
                and not _CUDNN_CONV3D_CHECKED):
            _CUDNN_CONV3D_CHECKED = True
            try:
                _x = torch.randn(1, 1, 2, 4, 4, device="cuda")
                torch.nn.functional.conv3d(_x, torch.randn(1, 1, 2, 2, 2, device="cuda"))
                del _x
            except RuntimeError:
                logger.warning("cuDNN conv3d broken — disabling cuDNN (torch native fallback)")
                torch.backends.cudnn.enabled = False

        logger.info(f"Loading local VLM: {model_name} (revision={revision})")
        self.model_name = model_name
        # Pinned snapshot path for every from_pretrained; model_name (the hub
        # id) stays the key for all name-keyed logic.
        self.revision = revision
        self.model_path = resolve_model_snapshot(model_name, revision)
        # Some VLMs ship custom processor/modeling code that HF can only
        # execute with trust_remote_code=True. Detect by substring. The
        # substrings are narrow enough to avoid false positives on
        # unrelated "kimi-*" text-only models or community forks with
        # "kimi" in the path. Add entries here as new models are added.
        _TRUST_REMOTE_CODE_SUBSTRINGS = (
            "moonshotai/kimi-vl",  # Kimi-VL family (Thinking, Instruct, variants)
        )
        needs_trust = any(s in model_name.lower() for s in _TRUST_REMOTE_CODE_SUBSTRINGS)
        if needs_trust:
            logger.info(
                f"Using trust_remote_code=True for {model_name} (custom model code)"
            )
        self.processor = AutoProcessor.from_pretrained(
            self.model_path, trust_remote_code=needs_trust,
        )
        # Gemma4 supports a max_soft_tokens knob (5 supported settings:
        # 70/140/280/560/1120 → 64/121/256/529/1089 actual <|image|>
        # tokens). Default is 280 → 256 tokens.
        # Apply only if caller explicitly set it AND we're loading a
        # Gemma model.
        if gemma_max_soft_tokens is not None:
            from codecompress.code_renderer import (
                GEMMA_SUPPORTED_MAX_SOFT_TOKENS,
                GEMMA_TOKENS_PER_IMAGE_BY_MAX_SOFT_TOKENS,
            )
            if "gemma" not in model_name.lower():
                logger.warning(
                    f"gemma_max_soft_tokens={gemma_max_soft_tokens} was set "
                    f"but model_name={model_name!r} is not a Gemma model; "
                    f"the flag will be ignored. Did you typo the model name?"
                )
            else:
                if gemma_max_soft_tokens not in GEMMA_SUPPORTED_MAX_SOFT_TOKENS:
                    raise ValueError(
                        f"gemma_max_soft_tokens={gemma_max_soft_tokens} is not "
                        f"supported by Gemma4. Allowed: "
                        f"{sorted(GEMMA_SUPPORTED_MAX_SOFT_TOKENS)}"
                    )
                self.processor.image_seq_length = gemma_max_soft_tokens
                if hasattr(self.processor, "image_processor"):
                    self.processor.image_processor.max_soft_tokens = gemma_max_soft_tokens
                    self.processor.image_processor.image_seq_length = gemma_max_soft_tokens
                logger.info(
                    f"Set Gemma max_soft_tokens={gemma_max_soft_tokens} on processor "
                    f"(actual <|image|> count will be "
                    f"{GEMMA_TOKENS_PER_IMAGE_BY_MAX_SOFT_TOKENS[gemma_max_soft_tokens]})"
                )
        self.gemma_max_soft_tokens = gemma_max_soft_tokens
        # Limit per-GPU memory to leave headroom for activations.
        # device_map="auto" can overload GPU 0, causing OOM on large models.
        # Cap at 20GiB per GPU (out of 24GiB) → 4GiB headroom for activations.
        load_kwargs = dict(
            torch_dtype=torch_dtype,
            device_map=device_map,
            trust_remote_code=needs_trust,
        )
        if device_map == "auto" and torch.cuda.is_available():
            n_gpus = torch.cuda.device_count()
            per_gpu_cap = "20GiB"
            load_kwargs["max_memory"] = {
                i: per_gpu_cap for i in range(n_gpus)
            }
            logger.info(f"Setting max_memory={per_gpu_cap} × {n_gpus} GPUs")
        try:
            self.model = AutoVLM.from_pretrained(self.model_path, **load_kwargs)
        except ValueError as e:
            if "Unrecognized configuration" in str(e):
                logger.info(
                    f"AutoModelForImageTextToText didn't recognize "
                    f"{model_name}'s config; retrying with AutoModelForCausalLM."
                )
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_path, **load_kwargs,
                )
            else:
                raise
        self.model.eval()
        logger.info(
            f"Local VLM {model_name} loaded as "
            f"{type(self.model).__name__}."
        )
        # Context-window accounting: capture the model's REAL window config
        # from the loaded config (not docs), and the
        # image-pad token id so generate() can measure visual tokens from the
        # actual input_ids. last_* are set by each generate() call.
        from codecompress import window_audit as _wa
        self.window_config = _wa.extract_window_config(self.model.config)
        self.image_token_id = self._resolve_image_token_id()
        # MiniCPM-V unk-run counting mode: its served image placeholder is
        # literal <unk>, which the standard resolver refuses. None for every
        # other model (counting unchanged).
        self.minicpmv_unk_id = _wa.resolve_minicpmv_unk_run_id(
            self.model.config, self.processor)
        self.last_input_tokens = None
        self.last_image_pad_tokens = None
        self.last_visual_token_runs = None
        self.last_generated_tokens = None
        self.last_latency_s = None

    def _resolve_image_token_id(self):
        """The image-pad token id, for counting visual tokens in input_ids.
        Delegates to the shared resolver in window_audit (single source of truth
        for the pad-string fallback list — see IMAGE_PAD_TOKEN_STRINGS)."""
        from codecompress import window_audit
        return window_audit.resolve_image_token_id(self.model.config, self.processor)

    def generate(
        self,
        images: list[Image.Image],
        text_prompt: str,
        max_tokens: int = 128,
        temperature: float = 0.0,
        system_prompt: str | None = None,
    ) -> str:
        import torch

        # Build message in Qwen-VL chat format
        content = []
        for img in images:
            content.append({"type": "image", "image": img})
        content.append({"type": "text", "text": text_prompt})

        # assemble_vlm_messages adds the fair system prompt when provided; with
        # system_prompt=None it preserves the legacy behavior — an explicit empty
        # system for Mistral-Small-3.1 (its chat template auto-injects a 600+ char
        # persona; the text arm uses an explicit empty system to suppress it, so we
        # mirror that for parity) and no system turn for any other model.
        messages = assemble_vlm_messages(content, system_prompt, self.model_name)

        # Process inputs
        # Disable thinking mode for models that support it (Qwen3.5, etc.)
        # to get direct code output instead of chain-of-thought reasoning.
        # Shared probe (probe_enable_thinking) — same logic as VllmVLM. Note:
        # tokenizer-template models (MiniCPM-V class) are vLLM-only on this
        # path: the probe falls back to the tokenizer, but the REAL render
        # below still uses the processor's template and raises there.
        chat_kwargs = probe_enable_thinking(self.processor)
        text_input = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
            **chat_kwargs,
        )
        inputs = self.processor(
            text=[text_input],
            images=images if images else None,
            padding=True,
            return_tensors="pt",
        ).to(self.model.device)

        # Cast floating-point inputs (e.g. pixel_values) to the model's
        # dtype. Some processors return fp32 tensors while the model is
        # loaded in bf16; the mismatch raises
        #   "Input type (torch.cuda.FloatTensor) and weight type
        #    (CUDABFloat16Type) should be the same"
        # on any model whose vision tower was explicitly bf16-loaded
        # (observed on Aria); safe no-op on models where the dtype already
        # matches.
        model_dtype = getattr(self.model, "dtype", None)
        if model_dtype is not None:
            for k, v in list(inputs.items()):
                if hasattr(v, "dtype") and v.dtype.is_floating_point:
                    inputs[k] = v.to(model_dtype)

        # Generate (timed, for the latency metric)
        from time import perf_counter as _perf_counter
        _t0 = _perf_counter()
        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=temperature if temperature > 0 else None,
                do_sample=temperature > 0,
            )
        self.last_latency_s = _perf_counter() - _t0

        # Decode only the generated tokens (skip the input)
        generated_ids = output_ids[0][inputs.input_ids.shape[1]:]
        completion = self.processor.decode(generated_ids, skip_special_tokens=True)
        # Strip thinking (Qwen3.5 <think>...</think>; also unterminated blocks)
        completion = strip_think_blocks(completion)

        # --- measure the ACTUAL tokens the model received ---
        # input_ids.shape[1] is ground truth (incl. text + image-pad + special
        # tokens); the image-pad count validates the analytic visual-token
        # formula at runtime (the QWEN_TOKENS_PER_TILE-class guard).
        _ids = inputs["input_ids"]
        self.last_input_tokens = int(_ids.shape[1])
        # Shared measurement path (window_audit.measure_visual_tokens): the
        # standard per-pad-id count, or MiniCPM-V's unk-run mode when active
        # (getattr keeps __new__-constructed test doubles working).
        from codecompress import window_audit as _wa
        self.last_image_pad_tokens, self.last_visual_token_runs = _wa.measure_visual_tokens(
            _ids[0].tolist(), image_token_id=self.image_token_id,
            minicpmv_unk_id=getattr(self, "minicpmv_unk_id", None))
        self.last_generated_tokens = int(generated_ids.shape[0])
        return completion.strip()

    def generate_batch(
        self,
        batch: list[tuple[list[Image.Image], str]],
        max_tokens: int = 128,
        temperature: float = 0.0,
    ) -> list[str]:
        # Sequential processing — VLM batch inference with variable-length
        # image inputs is complex; sequential is simpler and correct.
        # NOTE: each generate() overwrites self.last_input_tokens
        # / last_image_pad_tokens / last_generated_tokens / last_latency_s, so
        # after this returns they reflect ONLY the final item. The image
        # path reads them per-example via generate() (not generate_batch); don't
        # read last_* after a batch and expect per-item values.
        results = []
        for images, text_prompt in batch:
            try:
                result = self.generate(images, text_prompt, max_tokens, temperature)
                results.append(result)
            except Exception as e:
                logger.error(f"VLM generation error: {e}")
                results.append("ERROR_GENERATING")
        return results

    def cleanup(self):
        """Free GPU memory."""
        import gc
        import torch
        del self.model
        del self.processor
        gc.collect()
        torch.cuda.empty_cache()
        logger.info("Local VLM freed from GPU memory.")


# =============================================================================
# API VLM (OpenAI-compatible endpoints)
# =============================================================================

class ApiVLM(VLMBackend):
    """
    API-based VLM inference using OpenAI-compatible endpoints.

    Works with OpenRouter, OpenAI, Azure, and any provider that supports
    the OpenAI chat completions API with image_url content type.
    """

    def __init__(
        self,
        model: str,
        api_key: str | None = None,
        base_url: str = "https://openrouter.ai/api/v1",
        max_retries: int = 5,
        retry_delay: float = 2.0,
        image_detail: str = "high",
    ):
        import os
        self.model = model
        self.api_key = api_key or os.environ.get("OPENROUTER_API_KEY") or os.environ.get("OPENAI_API_KEY")
        self.base_url = base_url
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.image_detail = image_detail

        if not self.api_key:
            raise ValueError(
                "API key required. Set OPENROUTER_API_KEY or OPENAI_API_KEY env var, "
                "or pass api_key= to ApiVLM()."
            )

        # Lazy import — only needed for API calls
        try:
            from openai import OpenAI
            self.client = OpenAI(api_key=self.api_key, base_url=self.base_url)
        except ImportError:
            raise ImportError("openai package required for API VLM. Install with: pip install openai")

        logger.info(f"API VLM initialized: {model} via {base_url}")

    def generate(
        self,
        images: list[Image.Image],
        text_prompt: str,
        max_tokens: int = 128,
        temperature: float = 0.0,
        system_prompt: str | None = None,
    ) -> str:
        # Build message content with images + text
        content = []
        for b64 in _images_to_base64(images):
            content.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/png;base64,{b64}",
                    "detail": self.image_detail,
                },
            })
        content.append({"type": "text", "text": text_prompt})

        messages = assemble_vlm_messages(content, system_prompt, getattr(self, "model", ""))

        # Retry loop with exponential backoff
        for attempt in range(self.max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    max_tokens=max_tokens,
                    temperature=temperature,
                )
                return response.choices[0].message.content.strip()
            except Exception as e:
                if attempt < self.max_retries - 1:
                    delay = self.retry_delay * (2 ** attempt)
                    logger.warning(f"API call failed (attempt {attempt + 1}): {e}. Retrying in {delay}s...")
                    time.sleep(delay)
                else:
                    logger.error(f"API call failed after {self.max_retries} attempts: {e}")
                    raise

    def generate_batch(
        self,
        batch: list[tuple[list[Image.Image], str]],
        max_tokens: int = 128,
        temperature: float = 0.0,
    ) -> list[str]:
        # Sequential for now — could be parallelized with async/threading
        results = []
        for images, text_prompt in batch:
            try:
                result = self.generate(images, text_prompt, max_tokens, temperature)
                results.append(result)
            except Exception as e:
                logger.error(f"API VLM generation error: {e}")
                results.append("ERROR_GENERATING")
        return results

    def cleanup(self):
        """No-op for API backend."""
        pass


def _patch_vllm_minicpmv_list_prompts():
    """Fix the vLLM-nightly MiniCPM-V engine-init crash (list-vs-str prompts).

    On vLLM 0.22.1rc1 nightly, ``MiniCPMVProcessingInfo.get_hf_processor``
    unconditionally replaces the hub's batch-capable remote-code processor with
    vLLM's VENDORED single-sample ``MiniCPMVProcessor``
    (vllm/transformers_utils/processors/minicpmv.py), but
    ``MiniCPMVMultiModalProcessor._base_call_hf_processor`` still takes the
    "zipped" branch for model versions {(2,6),(4,0),(4,5),(4,6)} — calling the
    HF processor with ``text`` as a LIST. The vendored
    ``_convert_images_texts_to_inputs`` runs ``regex.findall(pattern, texts)``
    on the raw value -> ``TypeError: expected string or buffer``, wrapped into
    ``ValueError: Failed to apply MiniCPMVProcessor``. Engine init dies in
    encoder-budget profiling (``MultiModalBudget.__init__`` ->
    ``get_dummy_mm_inputs``) and every image-bearing request would hit the same
    path — so no engine kwarg can avoid it.

    This shim forces upstream's own per-item else-branch (verbatim semantics:
    the branch vLLM itself uses for older MiniCPM-V versions), which passes
    ``prompt`` as a str — the form the vendored processor supports. It patches
    the class in vllm's module (NOT site-packages files); the EngineCore child
    process inherits it via fork (vLLM's default worker start method — see the
    caller for the spawn guard). Idempotent. Tests:
    codecompress/tests/test_minicpmv_vllm_patch.py (characterization included:
    when the vendored processor learns list text upstream, remove this).
    """
    from collections import defaultdict

    import vllm.model_executor.models.minicpmv as _m

    if getattr(
        _m.MiniCPMVMultiModalProcessor.__dict__.get("_base_call_hf_processor"),
        "_cc_minicpmv_list_fix", False,
    ):
        return  # already patched

    def _base_call_hf_processor(
        self, prompts, mm_data, mm_kwargs, tok_kwargs, *, out_keys,
    ):
        # Upstream's non-zipped else-branch, applied unconditionally.
        inputs = defaultdict(list)
        for i, prompt in enumerate(prompts):
            inputs_one = super(_m.MiniCPMVMultiModalProcessor, self)._call_hf_processor(
                prompt=prompt,
                mm_data={k: v[i] for k, v in mm_data.items()},
                mm_kwargs=mm_kwargs,
                tok_kwargs=tok_kwargs,
            )
            for k, v in inputs_one.items():
                assert len(v) == 1, (k, len(v))
                inputs[k].append(v[0])
        return {k: inputs[k] for k in out_keys}

    _base_call_hf_processor._cc_minicpmv_list_fix = True
    _m.MiniCPMVMultiModalProcessor._base_call_hf_processor = _base_call_hf_processor
    logger.info(
        "Applied MiniCPM-V vLLM shim: per-item HF-processor calls "
        "(vendored MiniCPMVProcessor rejects list prompts on this nightly)."
    )


def _load_processor_or_tokenizer(model_name, trust_remote_code=True):
    """AutoProcessor, falling back to AutoTokenizer when the processor class
    itself fails to construct (e.g. transformers' native InternVLProcessor
    requires tokenizer.start_image_token, absent on the OpenGVLab repo).
    Text-path serving needs only a tokenizer: the image-token
    resolver and probe_enable_thinking both accept one."""
    from transformers import AutoProcessor, AutoTokenizer

    try:
        return AutoProcessor.from_pretrained(model_name, trust_remote_code=trust_remote_code)
    except Exception as e:  # noqa: BLE001 — any processor-construction failure
        logger.warning(
            f"AutoProcessor failed for {model_name} ({type(e).__name__}: {e}); "
            "falling back to AutoTokenizer (text-path serving only)."
        )
        return AutoTokenizer.from_pretrained(model_name, trust_remote_code=trust_remote_code)


def _last_subsequence_index(sequence, subsequence):
    """Return the final exact subsequence start, or None when absent."""
    width = len(subsequence)
    for start in range(len(sequence) - width, -1, -1):
        if sequence[start:start + width] == subsequence:
            return start
    return None


class VllmVLM(VLMBackend):
    """vLLM-backed multimodal VLM (flash + paged attention; the serving backend
    of the study).

    Mirrors LocalVLM's interface exactly — generate(images, text_prompt) plus the
    window-accounting attributes (window_config, last_input_tokens,
    last_image_pad_tokens, last_generated_tokens, last_latency_s) — so image.py
    and main._build_image_window_audit work unchanged regardless of backend.

    An equivalence check ran the same Gemma4 image examples through the
    transformers LocalVLM and through this class and asserted (a) identical
    image-pad token counts and (b) the same ES. Needs the vLLM environment.
    """

    def __init__(self, model_name, tensor_parallel_size=1, gpu_memory_utilization=0.9,
                 max_model_len=32768, gemma_max_soft_tokens=None, enforce_eager=False,
                 trust_remote_code=True, max_num_seqs=None, engine_recycle_every=None,
                 revision=None):
        if not VLLM_AVAILABLE:
            raise RuntimeError("VllmVLM requires vllm — use the .venv-vllm environment.")
        from transformers import AutoConfig
        from codecompress import window_audit as _wa

        self.model_name = model_name
        # Pinned snapshot path for processor/config/engine; model_name (the hub
        # id) stays the key for all name-keyed logic.
        self.revision = revision
        self.model_path = resolve_model_snapshot(model_name, revision)
        self.max_model_len = max_model_len
        if gemma_max_soft_tokens is not None:
            # The processor knob LocalVLM uses for the Gemma visual-token budget
            # isn't plumbed into vLLM's own image preprocessing: vLLM would use the
            # model DEFAULT regardless, so visual_tokens_formula (what the renderer
            # was told) would silently disagree with measured_visual_tokens (what
            # the model got) — and a sweep would collapse all settings to one
            # cell. Hard-fail rather than produce inconsistent token
            # accounting (validity > convenience). Sweep this knob on the
            # transformers backend (.venv).
            raise RuntimeError(
                "VllmVLM: gemma_max_soft_tokens is not supported under vLLM "
                f"(got {gemma_max_soft_tokens!r}). vLLM applies the model default, "
                "so the requested budget would not be honored and the token accounting "
                "would be internally inconsistent. Use the transformers backend to "
                "sweep gemma_max_soft_tokens."
            )
        self.processor = _load_processor_or_tokenizer(
            self.model_path, trust_remote_code=trust_remote_code)
        cfg = AutoConfig.from_pretrained(self.model_path, trust_remote_code=trust_remote_code)
        # image-pad token id (shared resolver) — for counting visual tokens in
        # the vLLM prompt_token_ids (the measured visual-token count).
        self.image_token_id = _wa.resolve_image_token_id(cfg, self.processor)
        # MiniCPM-V unk-run counting mode: its served image placeholder is
        # literal <unk> (the resolver above picks the
        # never-emitted Qwen <|image_pad|> from the shared vocab → count 0).
        # None for every other model (counting unchanged).
        self.minicpmv_unk_id = _wa.resolve_minicpmv_unk_run_id(cfg, self.processor)
        # window config: cap context_window at the OPERATIONAL max_model_len,
        # preserving the architectural value.
        wc = _wa.extract_window_config(cfg)
        if (isinstance(wc.get("context_window"), int)
                and max_model_len < wc["context_window"]):
            wc = {**wc, "architectural_window": wc["context_window"],
                  "operational_window": max_model_len, "context_window": max_model_len}
        self.window_config = wc

        # Mirror LocalVLM: suppress chain-of-thought on thinking VLMs so the
        # image arm returns code directly (and so both backends build the SAME
        # prompt). vLLM's chat() forwards chat_template_kwargs into the Jinja
        # template; we only pass enable_thinking when the template actually
        # accepts it (a no-op on Gemma4/Qwen3-VL/Mistral; REQUIRED on
        # MiniCPM-V 4.5, whose template lives on the tokenizer — the shared
        # probe falls back to it). The strip_think_blocks in generate() is the
        # backstop either way.
        self._chat_template_kwargs = probe_enable_thinking(self.processor)
        if self._chat_template_kwargs:
            logger.info(
                f"VllmVLM: chat template accepts enable_thinking — "
                f"forwarding {self._chat_template_kwargs} to llm.chat()."
            )
        else:
            logger.info("VllmVLM: enable_thinking not supported by the chat "
                        "template; relying on strip_think_blocks backstop.")

        logger.info(f"Loading vLLM VLM: {model_name} (revision={revision}, "
                    f"path={self.model_path}, TP={tensor_parallel_size}, "
                    f"max_model_len={max_model_len})")
        _llm_kwargs = dict(
            model=self.model_path, tensor_parallel_size=tensor_parallel_size,
            gpu_memory_utilization=gpu_memory_utilization, max_model_len=max_model_len,
            trust_remote_code=trust_remote_code, enforce_eager=enforce_eager,
            enable_prefix_caching=False,
        )
        if getattr(cfg, "model_type", None) == "minicpmv":
            # The shim below must be live in the EngineCore child too (it
            # constructs MultiModalBudget in GPUModelRunner.__init__). vLLM
            # forks the child by default, inheriting the patch — but it uses
            # spawn when VLLM_WORKER_MULTIPROC_METHOD=spawn or when CUDA is
            # already initialized here, and a spawned child re-imports vllm
            # unpatched. Fail loudly (pre-mutation) rather than crash cryptically.
            import os as _os

            import torch as _torch
            if _os.environ.get("VLLM_WORKER_MULTIPROC_METHOD") == "spawn":
                raise RuntimeError(
                    "VllmVLM(MiniCPM-V): VLLM_WORKER_MULTIPROC_METHOD=spawn is "
                    "set, so the EngineCore child will re-import vllm without "
                    "the MiniCPM-V processor shim and die at engine init. "
                    "Unset it (fork is required for this model)."
                )
            if _torch.cuda.is_initialized():
                raise RuntimeError(
                    "VllmVLM(MiniCPM-V): CUDA is already initialized in this "
                    "process, so vLLM will spawn (not fork) the EngineCore and "
                    "the MiniCPM-V processor shim will not propagate. Construct "
                    "VllmVLM before any CUDA work in this process."
                )
            # vLLM-nightly MiniCPM-V: engine init dies in encoder-budget
            # profiling (and every image request would too) because the model
            # code passes LIST prompts to the vendored single-sample processor.
            # Shim it before the engine is built (docstring on the patch fn).
            _patch_vllm_minicpmv_list_prompts()
            # We never send video; a 0 limit drops the video modality from
            # engine profiling (its dummy path is untested under the shim) and
            # trims profiling cost. Image limit stays at the vLLM default.
            _llm_kwargs["limit_mm_per_prompt"] = {"video": 0}
            # Encoder-cache memory profiling batches ~13 max-feature dummy
            # images (640 visual tokens / 10 tiles each) through the SigLIP
            # tower at once — a hard CUDA OOM on a 24 GB card already holding
            # the 16.2 GiB weights on a 24 GB card.
            # skip_mm_profiling is vLLM's own escape hatch for this (worker
            # gate in gpu_model_runner.profile_run). KV sizing then ignores
            # encoder activations. Exposure is small ONLY because generate()
            # submits one prompt per llm.chat() call (encoder batch = one
            # prompt's pages) — do NOT vectorize generate_batch for MiniCPM-V
            # without restoring profiling.
            _llm_kwargs["skip_mm_profiling"] = True
            if gpu_memory_utilization > 0.85:
                logger.warning(
                    f"VllmVLM(MiniCPM-V): gpu_memory_utilization="
                    f"{gpu_memory_utilization} with skip_mm_profiling=True — "
                    "encoder activations are NOT profiled; headroom for the "
                    "vision tower comes only from allocator slack. Watch for "
                    "mid-run CUDA OOM on many-page prompts."
                )
        # Hybrid linear-attention VLMs (e.g. a Qwen3.5-VL with GDN/Mamba layers)
        # need a lowered concurrency cap or engine init fails on Mamba cache blocks
        # (one block per concurrent sequence). Output-neutral. None = vLLM default.
        if max_num_seqs is not None:
            _llm_kwargs["max_num_seqs"] = max_num_seqs
        self.llm = _VLLM_LLM(**_llm_kwargs)
        # Engine recycling: some archs (InternVL3.5 on the pinned nightly)
        # wedge the EngineCore after ~415-430 requests (RPC sample_tokens
        # timeout) regardless of input. When set, the engine is torn down and
        # rebuilt every N generates (~2-3 min each). None = off.
        self._llm_kwargs_saved = dict(_llm_kwargs)
        self._recycle_every = engine_recycle_every
        self._gen_count = 0
        self._rebuilds = 0
        self.last_input_tokens = None
        self.last_image_pad_tokens = None
        self.last_visual_token_runs = None
        self.last_generated_tokens = None
        self.last_latency_s = None

    def _rebuild_engine(self):
        import gc

        import torch

        logger.warning(
            f"VllmVLM: recycling the engine after {self._gen_count} generates "
            f"(rebuild #{self._rebuilds + 1}; engine_recycle_every={self._recycle_every})"
        )
        del self.llm
        gc.collect()
        torch.cuda.empty_cache()
        self.llm = _VLLM_LLM(**self._llm_kwargs_saved)

    def _maybe_recycle_engine(self):
        # getattr defaults keep __new__-constructed test doubles working
        # (the established pattern in generate()).
        every = getattr(self, "_recycle_every", None)
        count = getattr(self, "_gen_count", 0)
        if every and count and count % every == 0:
            self._rebuild_engine()
            self._rebuilds = getattr(self, "_rebuilds", 0) + 1

    def generate(self, images, text_prompt, max_tokens=128, temperature=0.0, system_prompt=None):
        self._maybe_recycle_engine()
        self._gen_count = getattr(self, "_gen_count", 0) + 1
        # NOTE (truncation): unlike the text arm — which left-truncates over-length
        # prompts to the tail as TokensPrompt so vLLM never raises — this
        # multimodal path does NOT truncate: left-truncating an
        # interleaved image+text token stream would corrupt image-pad spans. An
        # over-max_model_len image prompt therefore raises inside vLLM and is
        # caught upstream (main.py) as ERROR_GENERATING. That can make the SCORED
        # set differ from the transformers backend on very long image cells; the
        # equivalence harness asserts equal scored-N (verify_vllm_image_equivalence
        # EVIDENCE check) so any such divergence fails loudly rather than silently.
        from time import perf_counter

        # Mirror LocalVLM's message construction exactly (Mistral gets an empty
        # system to suppress its persona; image items then the text).
        content = [{"type": "image_pil", "image_pil": img} for img in images]
        content.append({"type": "text", "text": text_prompt})
        messages = assemble_vlm_messages(content, system_prompt, self.model_name)

        sp = _VLLM_SamplingParams(temperature=temperature, top_p=1.0, max_tokens=max_tokens)
        # Forward enable_thinking=False where the template supports it (set in
        # __init__); getattr keeps __new__-constructed test doubles working.
        chat_kwargs = {}
        _ctk = getattr(self, "_chat_template_kwargs", None)
        if _ctk:
            chat_kwargs["chat_template_kwargs"] = _ctk
        _t0 = perf_counter()
        outs = self.llm.chat(messages, sp, use_tqdm=False, **chat_kwargs)
        self.last_latency_s = perf_counter() - _t0

        out = outs[0]
        ptids = list(out.prompt_token_ids)
        self.last_input_tokens = len(ptids)
        # Shared measurement path (window_audit.measure_visual_tokens): the
        # standard per-pad-id count, or MiniCPM-V's unk-run mode when active
        # (getattr keeps __new__-constructed test doubles working).
        from codecompress import window_audit as _wa
        self.last_image_pad_tokens, self.last_visual_token_runs = _wa.measure_visual_tokens(
            ptids, image_token_id=self.image_token_id,
            minicpmv_unk_id=getattr(self, "minicpmv_unk_id", None))
        comp = out.outputs[0]
        self.last_generated_tokens = len(comp.token_ids)
        # Expose vLLM's stop reason ("length" == answer-cap or window clip)
        # for the per-row window-fit record. getattr: test doubles.
        self.last_finish_reason = getattr(comp, "finish_reason", None)
        return strip_think_blocks(comp.text).strip()

    def score_choices(self, prompt_or_messages, choices: list[str]) -> list[float]:
        """Sum continuation-token prompt logprobs for each supplied choice."""
        self._maybe_recycle_engine()
        tokenizer = getattr(self.processor, "tokenizer", self.processor)
        scores = []
        for choice in choices:
            self._gen_count = getattr(self, "_gen_count", 0) + 1
            choice_ids = tokenizer.encode(choice, add_special_tokens=False)
            if not choice_ids:
                raise ValueError(f"Choice tokenized to an empty sequence: {choice!r}")
            messages = list(prompt_or_messages) + [
                {"role": "assistant", "content": choice}
            ]
            sp = _VLLM_SamplingParams(
                temperature=0.0, max_tokens=1, prompt_logprobs=1
            )
            chat_kwargs = {}
            template_kwargs = getattr(self, "_chat_template_kwargs", None)
            if template_kwargs:
                chat_kwargs["chat_template_kwargs"] = template_kwargs
            out = self.llm.chat(messages, sp, use_tqdm=False, **chat_kwargs)[0]
            prompt_ids = list(out.prompt_token_ids)
            start = _last_subsequence_index(prompt_ids, choice_ids)
            if start is None:
                raise RuntimeError(
                    f"Choice token sequence {choice_ids!r} not found in rendered prompt"
                )
            total = 0.0
            for offset, token_id in enumerate(choice_ids):
                entry = out.prompt_logprobs[start + offset]
                if entry is None or token_id not in entry:
                    raise RuntimeError(
                        f"vLLM omitted prompt logprob for choice token {token_id}"
                    )
                total += float(entry[token_id].logprob)
            scores.append(total)
        return scores

    def generate_batch(self, batch, max_tokens=128, temperature=0.0):
        # Sequential (mirrors LocalVLM); vLLM batches internally per call.
        return [self.generate(imgs, tp, max_tokens, temperature) for imgs, tp in batch]

    def cleanup(self):
        import gc
        if getattr(self, "llm", None) is not None:
            del self.llm
            self.llm = None
        gc.collect()
