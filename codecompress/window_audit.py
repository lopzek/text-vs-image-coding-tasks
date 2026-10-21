"""Context-window accounting — record, per example, whether the model's context
window was saturated and HOW the model handled it, uniformly across the text and
image arms, machine-readable for automated analysis.

Ground truth: the real config of each model was captured empirically (never trust
documentation for architecture constants).

Pure functions only — this module loads no model. The caller supplies:
  * a model config (dict or transformers PretrainedConfig), and
  * the measured token counts (from input_ids.shape[1] / processor output).

Two public functions:
  extract_window_config(config) -> dict   # the model's window properties
  classify(...)                 -> dict    # the per-example window_audit block
"""
from __future__ import annotations

from collections import Counter

# Attention regimes — how the model treats a long context. Derived from the
# EMPIRICAL per-layer pattern, never from the model name.
REGIME_FULL = "full"
REGIME_SLIDING_INTERLEAVED = "sliding_interleaved"  # mix of sliding + >=1 global layer (Gemma3/4, Qwen full_attention_interval)
REGIME_PURE_SLIDING = "pure_sliding"                # EVERY layer sliding (old Mistral-7B-v0.1 era)
REGIME_LINEAR_HYBRID = "linear_hybrid"              # linear/SSM + full layers (Qwen3.5)
REGIME_LINEAR = "linear"                            # all linear/SSM (no full layer)
REGIME_UNKNOWN = "unknown"

_LINEAR_TOKENS = ("linear", "mamba", "recurrent", "ssm")
_FULL_TOKENS = ("full", "global")
_SLIDING_TOKENS = ("sliding", "local")


def _get_raw(config, name):
    """Field accessor for a dict OR a transformers PretrainedConfig."""
    if isinstance(config, dict):
        return config.get(name)
    return getattr(config, name, None)


def _cfg_get(config, name):
    """Return (value, location) preferring top-level then ``text_config``.
    location ∈ {'top_level', 'text_config', None}. Returns the value even if it
    is ``False`` (so use_sliding_window=False is distinguishable from absent)."""
    if config is None:
        return None, None
    # top-level
    if isinstance(config, dict):
        present = name in config
    else:
        present = hasattr(config, name)
    if present:
        v = _get_raw(config, name)
        if v is not None:
            return v, "top_level"
    # nested text_config (multimodal models keep LM fields here)
    tc = _get_raw(config, "text_config")
    if tc is not None:
        if (isinstance(tc, dict) and name in tc) or (not isinstance(tc, dict) and hasattr(tc, name)):
            v = _get_raw(tc, name)
            if v is not None:
                return v, "text_config"
    return None, None


# Known image-pad token strings, ordered most- to least-specific. Single source
# of truth shared by vlm_inference.LocalVLM._resolve_image_token_id and
# the visual-token verification so the two can never drift (they did: 5 vs 7).
# Used only as a fallback when the config exposes no image_token_id/index.
IMAGE_PAD_TOKEN_STRINGS = (
    "<|image_pad|>",      # Qwen2.5/3-VL
    "<image_soft_token>",  # Gemma3/4
    "<|image|>",
    "[IMG]",               # Pixtral / Mistral
    "<|media_pad|>",
    "<img>",
    "<image>",
)


def resolve_image_token_id(config, processor=None):
    """The image-pad token id, for counting visual tokens in ``input_ids``.

    Prefer the model config (``image_token_id`` / ``image_token_index``,
    top-level OR nested under ``text_config`` — verified present for every
    current model), falling through on a ``None`` top-level value to the nested
    one. Only if the config exposes nothing do we try the known pad strings via
    the processor's tokenizer (skipping ``unk``). Returns ``int`` or ``None`` —
    NEVER guesses, so a missing id surfaces rather than miscounting.

    Pure: ``config`` is a dict or PretrainedConfig; ``processor`` exposes a
    ``.tokenizer`` (or is one) with ``convert_tokens_to_ids`` + ``unk_token_id``.
    """
    for attr in ("image_token_id", "image_token_index"):
        v = _get_raw(config, attr)
        if v is None:
            tc = _get_raw(config, "text_config")
            v = _get_raw(tc, attr) if tc is not None else None
        if isinstance(v, int):
            return v
    if processor is None:
        return None
    tok = getattr(processor, "tokenizer", processor)
    convert = getattr(tok, "convert_tokens_to_ids", None)
    if convert is None:
        return None
    unk = getattr(tok, "unk_token_id", None)
    for s in IMAGE_PAD_TOKEN_STRINGS:
        try:
            tid = convert(s)
        except Exception:  # noqa: BLE001 — a tokenizer that rejects the string
            continue
        if isinstance(tid, int) and tid >= 0 and tid != unk:
            return tid
    return None


# --- MiniCPM-V unk-run counting mode (not used by the study) ----------------
# MiniCPM-V (model_type "minicpmv") serves its image placeholder as literal
# ``<unk>`` (id 128244 on rev fd3209b), which resolve_image_token_id
# deliberately refuses to count (unk is a legitimate token elsewhere); its
# pad-string fallback instead resolves Qwen's ``<|image_pad|>`` (151655 —
# present in MiniCPM's Qwen-based vocab but never emitted by its processor),
# so measured_visual_tokens read 0 while the true count was e.g. 1792
# (4 padded pages x 448). Measured counting template: visual
# tokens appear as contiguous runs of exactly 64 <unk> ids per crop; abutting
# crops merge into multiples of 64.
MINICPMV_UNK_RUN_MIN = 64  # one crop = one 64-token run (measured, fd3209b)


def resolve_minicpmv_unk_run_id(config, processor=None):
    """The ``<unk>`` id for MiniCPM-V's unk-run visual-token counting mode.

    Returns the id ONLY when the config's ``model_type`` is ``minicpmv`` AND
    the tokenizer resolves the served placeholder (its unk token) to a real
    id; ``None`` otherwise — so the mode can never activate for another model
    (a text with >=64 natural consecutive unks must not be miscounted there)
    and every other model's counting stays byte-identical.
    """
    if _get_raw(config, "model_type") != "minicpmv":
        return None
    if processor is None:
        return None
    tok = getattr(processor, "tokenizer", processor)
    unk = getattr(tok, "unk_token_id", None)
    if isinstance(unk, int) and unk >= 0:
        return unk
    # transformers 5.x may load the stock TokenizersBackend without unk_token
    # set even though '<unk>' is in the added vocab (the _shim_tokenizer
    # lesson).
    convert = getattr(tok, "convert_tokens_to_ids", None)
    if convert is None:
        return None
    try:
        tid = convert("<unk>")
    except Exception:  # noqa: BLE001 — a tokenizer that rejects the string
        return None
    return tid if isinstance(tid, int) and tid >= 0 else None


def count_unk_run_tokens(token_ids, unk_id, min_run=MINICPMV_UNK_RUN_MIN):
    """``(count, run_lengths)`` of ids inside contiguous ``unk_id`` runs of
    length >= ``min_run``.

    Each qualifying run is counted at its FULL length — abutting 64-token
    crops merge into one multiple-of-64 run; a non-multiple length is NOT
    floored, it lands in ``run_lengths`` so the formula drift check can flag
    it. Runs shorter than ``min_run`` (scattered natural unks) are ignored.
    Pure — ``token_ids`` is any iterable of ints.
    """
    count, runs, cur = 0, [], 0
    for t in token_ids:
        if t == unk_id:
            cur += 1
            continue
        if cur >= min_run:
            count += cur
            runs.append(cur)
        cur = 0
    if cur >= min_run:
        count += cur
        runs.append(cur)
    return count, runs


def measure_visual_tokens(token_ids, *, image_token_id, minicpmv_unk_id=None):
    """``(measured_visual_tokens, run_lengths)`` from served token ids.

    Single measurement path shared by LocalVLM and VllmVLM. When
    ``minicpmv_unk_id`` is set (resolve_minicpmv_unk_run_id — minicpmv only),
    counts unk-runs and returns their lengths; otherwise the standard
    pad-id count, byte-identical to the historical behavior (``run_lengths``
    is None, and ``None`` count when no id resolved — never a guess).
    """
    if minicpmv_unk_id is not None:
        return count_unk_run_tokens(token_ids, minicpmv_unk_id)
    if image_token_id is None:
        return None, None
    return sum(1 for t in token_ids if t == image_token_id), None


def derive_attention_regime(layer_types, *, sliding_window_active,
                            full_attention_interval=None,
                            sliding_window_pattern=None) -> str:
    """Classify the long-context attention mechanism from the per-layer pattern.

    Per-layer ``layer_types`` is authoritative when present. Without it we fall
    back to scalar flags (best-effort, flagged by the caller via window_source)."""
    if isinstance(layer_types, (list, tuple)) and layer_types:
        kinds = {str(k).lower() for k in layer_types}
        has_linear = any(any(t in k for t in _LINEAR_TOKENS) for k in kinds)
        has_full = any(any(t in k for t in _FULL_TOKENS) or k == "attention" for k in kinds)
        has_sliding = any(any(t in k for t in _SLIDING_TOKENS) for k in kinds)
        if has_linear:
            return REGIME_LINEAR_HYBRID if has_full else REGIME_LINEAR
        if has_sliding and has_full:
            return REGIME_SLIDING_INTERLEAVED
        if has_sliding:
            return REGIME_PURE_SLIDING
        if has_full:
            return REGIME_FULL
        return REGIME_UNKNOWN
    # No per-layer info: infer from scalar flags.
    if sliding_window_active:
        # an interval / pattern means SOME layers are global → interleaved
        if full_attention_interval or sliding_window_pattern:
            return REGIME_SLIDING_INTERLEAVED
        return REGIME_PURE_SLIDING  # best guess; layer_types unavailable
    return REGIME_FULL


def _layer_types_summary(layer_types) -> str | None:
    if not isinstance(layer_types, (list, tuple)) or not layer_types:
        return None
    return ",".join(f"{n}x{t}" for t, n in Counter(str(x) for x in layer_types).items())


def extract_window_config(config) -> dict:
    """Read the model's context-window properties from its config.

    NEVER silently substitutes a default — if ``max_position_embeddings`` is
    absent, ``context_window`` is None and ``window_source`` is 'fallback', so a
    guess can never be mistaken for a measurement (the bug this module exists to
    prevent). ``context_window`` is the RAW max_position_embeddings — modern HF
    configs already store the post-rope-scaling window, so we do NOT multiply.
    """
    max_pos, where = _cfg_get(config, "max_position_embeddings")
    orig_max, _ = _cfg_get(config, "original_max_position_embeddings")
    sliding_window, _ = _cfg_get(config, "sliding_window")
    use_sliding_window, _ = _cfg_get(config, "use_sliding_window")
    full_attention_interval, _ = _cfg_get(config, "full_attention_interval")
    sliding_window_pattern, _ = _cfg_get(config, "sliding_window_pattern")
    layer_types, _ = _cfg_get(config, "layer_types")
    rope_scaling, _ = _cfg_get(config, "rope_scaling")
    rope_parameters, _ = _cfg_get(config, "rope_parameters")

    sliding_window_active = bool(sliding_window) and (use_sliding_window is not False)

    regime = derive_attention_regime(
        layer_types, sliding_window_active=sliding_window_active,
        full_attention_interval=full_attention_interval,
        sliding_window_pattern=sliding_window_pattern,
    )

    return {
        "context_window": max_pos,                         # RAW max_position_embeddings, no rope multiply
        "window_source": where or "fallback",
        "original_max_position_embeddings": orig_max,
        "sliding_window": sliding_window,
        "use_sliding_window": use_sliding_window,
        "sliding_window_active": sliding_window_active,
        "attention_regime": regime,
        "layer_types_summary": _layer_types_summary(layer_types),
        "rope_present": bool(rope_scaling or rope_parameters),
    }


def classify(*, pre_truncation_tokens, model_input_tokens, window_config,
             pipeline_truncated, measured) -> dict:
    """Build the per-example ``window_audit`` block.

    Args:
      pre_truncation_tokens: token count BEFORE any pipeline truncation (true pressure).
      model_input_tokens:    token count the model actually received (post-truncation).
      window_config:         output of extract_window_config().
      pipeline_truncated:    did our pipeline cut the input (any of the 3 sources)?
      measured:              is model_input_tokens an exact measurement (vs reconstructed)?

    fully_attended (the fair cross-model subset) is True only when the model
    received AND globally attended to ALL input: measured, window known, not
    truncated, not over-window-before-pipeline, and not pure-sliding-local.
    Interleaved sliding (Gemma) is NOT local (its global layers attend fully),
    so it stays fully_attended."""
    cw = window_config.get("context_window")
    regime = window_config.get("attention_regime")
    sw = window_config.get("sliding_window")
    sw_active = window_config.get("sliding_window_active")

    window_utilization = (pre_truncation_tokens / cw) if cw else None
    over_window_before_pipeline = (cw is not None) and (pre_truncation_tokens > cw)
    pure_sliding_window_local = bool(
        regime == REGIME_PURE_SLIDING and sw_active and sw is not None
        and model_input_tokens > sw
    )
    fully_attended = bool(
        measured and cw is not None and (not pipeline_truncated)
        and (not over_window_before_pipeline) and (not pure_sliding_window_local)
    )

    return {
        "model_input_tokens": model_input_tokens,
        "pre_truncation_tokens": pre_truncation_tokens,
        "measured": bool(measured),
        "context_window": cw,
        "window_source": window_config.get("window_source"),
        "original_max_position_embeddings": window_config.get("original_max_position_embeddings"),
        "sliding_window": sw,
        "sliding_window_active": sw_active,
        "attention_regime": regime,
        "window_utilization": window_utilization,
        "pipeline_truncated": bool(pipeline_truncated),
        "pipeline_dropped_tokens": max(0, pre_truncation_tokens - model_input_tokens),
        "over_window_before_pipeline": over_window_before_pipeline,
        "pure_sliding_window_local": pure_sliding_window_local,
        "fully_attended": fully_attended,
    }
