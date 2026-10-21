"""
Unified evaluation pipeline for context compression experiments.

Composes a benchmark (what to evaluate) with a pipeline (how to compress
and generate). Supports three pipeline modes:

Text pipeline stages:
  1. Load dataset via benchmark
  2. Initialize compression models
  3. Compress background context for each example → build text prompts
  4. Free compression models from GPU
  5. Initialize generation LLM (vLLM)
  6. Generate completions in batches
  7. Score via benchmark metrics and save results

Image pipeline stages:
  1. Load dataset via benchmark
  2. Initialize VLM + tokenizer
  3. For each example: render background as images → generate via VLM
  4. Score via benchmark metrics and save results

Hybrid pipeline stages:
  1. Load dataset via benchmark
  2. Initialize compression models (TextPipeline)
  3. Compress background context for all examples
  4. Free compression models from GPU
  5. Render compressed text as images (CPU-only)
  6. Initialize VLM (ImagePipeline)
  7. Generate completions via VLM
  8. Score via benchmark metrics and save results

The evaluate_completion() function is the CLI entry point (run_cell.py builds
its arguments for every cell of the study).
"""

import json
import os

import fire
from loguru import logger
from tqdm import tqdm

from dotenv import load_dotenv
load_dotenv()

from codecompress.benchmarks import BENCHMARKS
from codecompress.pipelines.text import TextPipeline, build_method_suffix, configure_cuda_visible_devices
from codecompress.pipelines.image import (
    FLAT_BASELINE,
    IMAGE_METHODS,
    TEXT_BASELINE,
    ImagePipeline,
    _flatten_code,
    _validate_min_char_px,
)


# =============================================================================
# Helpers
# =============================================================================

def save_json(data: dict, file_path: str):
    """Saves dictionary data to a JSON file, creating parent dirs as needed."""
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, 'w') as f:
        json.dump(data, f, indent=4)


def _build_scores_params_text(method, **params):
    """
    Build the 'parameters' sub-dict for the SCORES.json output (text pipeline).
    Only includes parameters relevant to the chosen method (others are None).
    """
    return {
        "dataset_path": params["dataset_path"],
        "dataset_split": params["dataset_split"],
        "filter_current_lines_max": params["filter_current_lines_max"],
        "filter_background_tokens_min": params["filter_background_tokens_min"],
        "embed_model_name": params["embed_model_name"] if method in ("rag", "function_rag") else None,
        "compression_model_name": params["compression_model_name"] if method in ("llmlingua", "longllmlingua", "code_compressor") else None,
        "max_new_tokens": params["max_new_tokens"],
        "batch_size": params["batch_size"],
        # RAG
        "rag_window_size": params["rag_window_size"] if method == "rag" else None,
        "rag_overlap": params["rag_overlap"] if method == "rag" else None,
        "rag_top_k": params["rag_top_k"] if method == "rag" else None,
        # Function RAG
        "function_rag_language": params["function_rag_language"] if method == "function_rag" else None,
        "function_rag_top_k": params["function_rag_top_k"] if method == "function_rag" else None,
        # LLMLingua / LongLLMLingua
        "lingua_target_token": params["lingua_target_token"] if method in ("llmlingua", "longllmlingua", "random_line", "random_token") else None,
        "longlingua_chunk_size": params["longlingua_chunk_size"] if method == "longllmlingua" else None,
        "longlingua_overlap": params["longlingua_overlap"] if method == "longllmlingua" else None,
        # CodeCompressor
        "code_compressor_target_token": params["code_compressor_target_token"] if method == "code_compressor" else None,
        "code_compressor_rank_only": (params["code_compressor_fine_ratio"] == 1.0) if method == "code_compressor" else None,
        "code_compressor_fine_ratio": params["code_compressor_fine_ratio"] if method == "code_compressor" else None,
        "importance_beta": params["importance_beta"] if method == "code_compressor" else None,
    }


def _resolve_benchmark_init_kwargs(BenchmarkClass, dataset_path, image_instruction_variant):
    """Build kwargs for instantiating a benchmark from CLI params.

    Passes dataset_path when the benchmark accepts it, and the
    image-instruction variant when set. Raises a clear error if the variant is
    requested for a benchmark that does not support it (key validation happens in
    the benchmark __init__).
    """
    import inspect

    init_params = inspect.signature(BenchmarkClass.__init__).parameters
    kwargs = {}
    if dataset_path and "dataset_path" in init_params:
        kwargs["dataset_path"] = dataset_path
    if image_instruction_variant is not None:
        if "image_instruction_variant" not in init_params:
            raise ValueError(
                "--image_instruction_variant is only supported by benchmarks that "
                f"accept it (e.g. lcc); {BenchmarkClass.__name__} does not."
            )
        kwargs["image_instruction_variant"] = image_instruction_variant
    return kwargs


def _build_scores_params_image(method, **params):
    """Build the 'parameters' sub-dict for SCORES.json (image pipeline)."""
    out = {
        "dataset_path": params["dataset_path"],
        "dataset_split": params["dataset_split"],
        "filter_current_lines_max": params["filter_current_lines_max"],
        "filter_background_tokens_min": params["filter_background_tokens_min"],
        "max_new_tokens": params["max_new_tokens"],
        "style": params.get("style"),
        "compression_ratio": params.get("compression_ratio"),
        "vlm_model": params.get("vlm_model"),
        "vlm_backend": params.get("vlm_backend"),
        "image_control": params.get("image_control", "matched"),
        "gemma_max_soft_tokens": params.get("gemma_max_soft_tokens"),
        # Which tokenizer did the token accounting (bg_text_tokens /
        # keep_last_text_tokens): the VLM's own; legacy_token_accounting
        # restores --model_name's.
        "token_accounting_tokenizer": params.get("token_accounting_tokenizer"),
    }
    # Image-instruction variant: record it when set so the per-cell summary
    # is complete. None-guarded so other runs stay byte-identical.
    if params.get("image_instruction_variant") is not None:
        out["image_instruction_variant"] = params.get("image_instruction_variant")
    # The pinned HF revision is part of the cell's provenance — record it
    # when set. None-guarded so unpinned cells stay byte-identical.
    if params.get("vlm_revision"):
        out["vlm_revision"] = params.get("vlm_revision")
    # Split study: K is the cell-distinguishing variable — record it in the
    # canonical per-cell summary. None-guarded so non-split runs stay identical.
    if params.get("num_image_splits") is not None:
        out["num_image_splits"] = params.get("num_image_splits")
        out["split_policy"] = params.get("split_policy", "lines")
        out["split_font_size"] = params.get("split_font_size")
    # The padded page geometry is part of the cell's operating point — record
    # it when set. None-guarded so existing runs stay identical.
    if params.get("pad_page_to") is not None:
        out["pad_page_to"] = list(params["pad_page_to"])
    # Encoder-OOM guard: the page cap changes which rows can score —
    # record it when set so blocked-row accounting is reproducible.
    if params.get("max_pages_per_request") is not None:
        out["max_pages_per_request"] = params.get("max_pages_per_request")
    # Page-fit policy: only serialize enabled knobs so existing cells keep
    # their byte-identical parameter schema.
    if params.get("keep_last_pages_fit_tokens") is not None:
        out["keep_last_pages_fit_tokens"] = params.get(
            "keep_last_pages_fit_tokens"
        )
    if params.get("record_page_metrics"):
        out["record_page_metrics"] = True
    if params.get("min_char_px") is not None:
        out["min_char_px"] = params.get("min_char_px")
    # Per-row window-fit policy knobs + the accounting-tokenizer pin.
    # None-guarded for byte-identity.
    if params.get("fit_window_tokens") is not None:
        out["fit_window_tokens"] = params.get("fit_window_tokens")
        out["fit_template_tokens"] = params.get("fit_template_tokens", 0)
        out["fit_margin_tokens"] = params.get("fit_margin_tokens", 0)
        out["fit_per_page_overhead"] = params.get("fit_per_page_overhead", 2)
    if params.get("token_accounting_revision"):
        out["token_accounting_revision"] = params.get("token_accounting_revision")
    # The text-side ablation flags are cell-distinguishing variables — record
    # them when set. False/None-guarded so every existing run's SCORES.json
    # stays byte-identical.
    for _ablation_flag in ("strip_indent", "shuffle_indent",
                           "strip_braces", "strip_semicolons"):
        if params.get(_ablation_flag):
            out[_ablation_flag] = True
    return out


def _build_image_method_suffix(method, compression_ratio, image_control="matched",
                               max_visual_tokens=None, num_image_splits=None):
    """Build a unique directory suffix for image method + compression ratio."""
    if max_visual_tokens is not None:
        suffix = f"method_{method}_vt{max_visual_tokens}"
    else:
        suffix = f"method_{method}_r{compression_ratio}"
    if image_control != "matched":
        suffix += f"_{image_control}"
    if num_image_splits is not None:
        # Write-isolate cells that differ only in K within one RUN_DIR.
        suffix += f"_split{num_image_splits}"
    return suffix


def _aggregate_window_audit(original_data, true_n=None):
    """Per-cell summary of the per-example window_audit blocks (parallel to the
    image_info aggregation). Returns None if no row carries one AND none errored, so
    SCORES.json is byte-identical on runs without the block. The delivered-in-full
    count (`n_fully_attended`/`frac_fully_attended`) is the headline;
    `visual_token_formula_mae` (image arm) flags formula drift.

    Coverage: an over-window IMAGE example raises inside vLLM → ERROR_GENERATING
    → has NO _window_audit block, so it is EXCLUDED from frac_fully_attended's denominator
    (which is the SURVIVORS). To keep that survivor-only nature transparent — not
    silently optimistic — we also surface n_cell_total (the cell's true N, from true_n),
    n_with_audit, n_errored (rows carrying _image_info.error), n_no_audit, and
    frac_audit_coverage. The runtime warning reads these so a heavily-over-window
    cell cannot report frac_fully_attended≈1.0 unnoticed."""
    blocks = [o["_window_audit"] for o in original_data
              if isinstance(o, dict) and o.get("_window_audit")]
    errored = sum(1 for o in original_data
                  if isinstance(o, dict) and isinstance(o.get("_image_info"), dict)
                  and o["_image_info"].get("error"))
    if not blocks and not errored:
        return None
    n = len(blocks)
    cell_total = true_n if (isinstance(true_n, int) and true_n > 0) else (n + errored)

    def _cnt(key):
        return sum(1 for b in blocks if b.get(key))

    def _nums(key):
        return [b[key] for b in blocks if isinstance(b.get(key), (int, float))]

    inputs, pre, utils = _nums("model_input_tokens"), _nums("pre_truncation_tokens"), _nums("window_utilization")
    maes = [abs(b["measured_visual_tokens"] - b["visual_tokens_formula"]) for b in blocks
            if isinstance(b.get("measured_visual_tokens"), (int, float))
            and isinstance(b.get("visual_tokens_formula"), (int, float))]
    n_fa = _cnt("fully_attended")
    first = blocks[0] if blocks else {}
    agg = {
        "context_window": first.get("context_window"),
        "sliding_window": first.get("sliding_window"),
        "sliding_window_active": first.get("sliding_window_active"),
        "attention_regime": first.get("attention_regime"),
        "window_source": first.get("window_source"),
        "n_examples": n,
        # Coverage: the cell's true N vs how many carry a block. n_errored
        # are rows that raised (no block) — over-window image raises hide here, so
        # frac_fully_attended (over n_with_audit) is survivor-only; frac_audit_coverage
        # exposes that. The warning reads these.
        "n_cell_total": cell_total,
        "n_with_audit": n,
        "n_errored": errored,
        "n_no_audit": cell_total - n,
        "frac_audit_coverage": (n / cell_total) if cell_total else None,
        "n_fully_attended": n_fa,
        "n_pipeline_truncated": _cnt("pipeline_truncated"),
        "n_over_window_before_pipeline": _cnt("over_window_before_pipeline"),
        "n_pure_sliding_local": _cnt("pure_sliding_window_local"),
        "n_unmeasured": sum(1 for b in blocks if not b.get("measured")),
        "frac_fully_attended": (n_fa / n) if n else None,
        "mean_input_tokens": (sum(inputs) / len(inputs)) if inputs else None,
        "max_pre_truncation_tokens": max(pre) if pre else None,
        "max_window_utilization": max(utils) if utils else None,
    }
    if maes:  # image arm only
        agg["visual_token_formula_mae"] = sum(maes) / len(maes)
    # Speed / cost axes: present on both arms once the text arm captures true
    # generation stats. None-safe — older blocks omit them.
    gen_toks, lats = _nums("generated_tokens"), _nums("latency_s")
    if gen_toks:
        agg["mean_generated_tokens"] = sum(gen_toks) / len(gen_toks)
    if lats:
        agg["mean_latency_s"] = sum(lats) / len(lats)
    return agg


def _window_audit_warning(agg):
    """Runtime over-window / truncation alarm for a finished cell.

    The pipelines do NOT raise on an over-context input — the text arm silently
    LEFT-TRUNCATES to fit and the image arm fails that example as ERROR_GENERATING.
    This surfaces the silent (text) case at runtime: returns a per-cell warning
    string when any example exceeded the context window BEFORE truncation, or was
    truncated by the pipeline, so context-window saturation is visible during the
    run — not only post-hoc in window_audit. Returns None when the cell is clean
    (over==0 and trunc==0) or carries no block (byte-identical behavior otherwise).
    """
    if not agg:
        return None
    n = agg.get("n_examples") or 0
    over = agg.get("n_over_window_before_pipeline") or 0
    trunc = agg.get("n_pipeline_truncated") or 0
    errored = agg.get("n_errored") or 0
    if over == 0 and trunc == 0 and errored == 0:
        return None
    total = agg.get("n_cell_total") or n
    cw = agg.get("context_window")
    mx = agg.get("max_pre_truncation_tokens")
    util = agg.get("max_window_utilization")
    parts = []
    if over:
        parts.append(f"{over}/{n} exceeded the context window ({cw} tok) BEFORE truncation")
    if trunc:
        parts.append(f"{trunc}/{n} were left-truncated to fit")
    if errored:
        # Over-window IMAGE examples raise → no block → invisible to
        # frac_fully_attended (survivor-only). Alarm so an all-errored cell can't
        # give a false all-clear.
        parts.append(f"{errored}/{total} ERRORED with NO window_audit (likely "
                     "over-window image raises — invisible to frac_fully_attended)")
    detail = f"max pre-truncation {mx} tok"
    if isinstance(util, (int, float)):
        detail += f", max utilization {util:.2f}x"
    return ("CONTEXT-WINDOW SATURATION: " + "; ".join(parts) + f" ({detail}). "
            "Affected examples saw a SHORTENED context (or none) — inspect window_audit "
            "(frac_fully_attended + frac_audit_coverage) before trusting this cell.")


def _build_text_window_audit(window_config, *, raw_pre_tokens, stat,
                             fallback_input_tokens, fallback_truncated,
                             operational_max_len=None):
    """Map one text-arm example to a window_audit block, preferring the TRUE
    per-prompt generation stats (post chat-template wrap + truncation, captured
    in LocalLLM._gen_stats) over the prep-time raw-prompt counts.

    Why the stats matter: under a chat template the model sees a WRAPPED prompt
    longer than the raw one, so the prep-time count undercounts model_input. The
    stat path also carries generated_tokens + latency for the speed/cost axes.
    Falls back to prep-time counts (exact only without a chat template) when no
    stat is available — never misattributes. Returns the block or None.

    Pure (no model, no I/O); window_config from extract_window_config().
    operational_max_len caps the EFFECTIVE window at what the run actually
    enforced (vLLM max_model_len / the transformers truncation budget) — without
    it, over_window/utilization would use the architectural 262144 while the run
    truncated at 32768. The architectural window is preserved on
    the block as `architectural_window` (with `operational_window` recorded too).
    """
    from codecompress import window_audit as _wa
    # Effective window = min(architectural, operational budget). Keeps the metrics
    # honest about saturation relative to what the run actually allowed.
    _arch_window, _capped = window_config.get("context_window"), False
    if (isinstance(operational_max_len, int) and operational_max_len > 0
            and isinstance(_arch_window, int) and operational_max_len < _arch_window):
        window_config = {**window_config, "context_window": operational_max_len}
        _capped = True

    def _annotate(block):
        if block is not None and _capped:
            block["architectural_window"] = _arch_window
            block["operational_window"] = operational_max_len
        return block

    if stat is not None and isinstance(stat.get("input_tokens"), int):
        model_input = stat["input_tokens"]
        truncated = bool(fallback_truncated) or bool(stat.get("safety_net_truncated"))
        # pre-truncation pressure = the largest available estimate of the prompt's
        # length BEFORE our truncation. Candidates: the prep-time raw count (from
        # the COMPRESSION tokenizer — wrong when it != the generation model), the
        # TRUE generation-tokenizer pre-truncation length (vLLM path, captured
        # pre-slice — the most accurate), and what the model actually saw. Taking
        # the max avoids an over_window/utilization undercount.
        _cands = [model_input]
        if isinstance(raw_pre_tokens, int):
            _cands.append(raw_pre_tokens)
        if isinstance(stat.get("pre_truncation_tokens_gen"), int):
            _cands.append(stat["pre_truncation_tokens_gen"])
        pre = max(_cands)
        block = _wa.classify(
            pre_truncation_tokens=pre, model_input_tokens=model_input,
            window_config=window_config, pipeline_truncated=truncated, measured=True)
        if isinstance(stat.get("generated_tokens"), int):
            block["generated_tokens"] = stat["generated_tokens"]
        if isinstance(stat.get("latency_s"), (int, float)):
            block["latency_s"] = float(stat["latency_s"])
        return _annotate(block)
    # Fallback: prep-time counts (raw prompt). Exact only without a chat template.
    if isinstance(fallback_input_tokens, int) and isinstance(raw_pre_tokens, int):
        return _annotate(_wa.classify(
            pre_truncation_tokens=raw_pre_tokens, model_input_tokens=fallback_input_tokens,
            window_config=window_config, pipeline_truncated=bool(fallback_truncated),
            measured=True))
    return None


def _text_window_config(pipeline, model_name, trust_remote_code, _autoconfig=None):
    """Backend-aware window config for the text arm.

    transformers backend: read the loaded model's config (pipeline._gen_model).
    vLLM backend: there is NO _gen_model, so load the config standalone via
    AutoConfig (cheap, no GPU) — without this the vLLM text window block would be
    silently skipped. Returns the extract_window_config() dict, or
    None if neither source is available.
    """
    from codecompress import window_audit as _wa
    _gm = getattr(pipeline, "_gen_model", None)
    if _gm is not None:
        return _wa.extract_window_config(_gm.config)
    if getattr(pipeline, "_gen_backend", None) == "vllm":
        if _autoconfig is None:
            from transformers import AutoConfig as _autoconfig
        cfg = _autoconfig.from_pretrained(model_name, trust_remote_code=trust_remote_code)
        return _wa.extract_window_config(cfg)
    return None


def parse_skip_example_ids(spec) -> set:
    """Comma-separated example indices to SKIP as ERROR_GENERATING without
    calling the engine (for a row that deterministically hangs an engine; the
    row is ERROR-blocked, the same treatment as window-guard blocks).

    Fire's CLI auto-coerces the value BEFORE it reaches here: "415" -> int
    415, "0,1,2" -> tuple (0, 1, 2). So accept int / tuple / list / str
    (comma-separated), not only str."""
    if spec is None or spec == "":
        return set()
    if isinstance(spec, (int,)):
        return {int(spec)}
    if isinstance(spec, (list, tuple, set)):
        return {int(x) for x in spec}
    return {int(x) for x in str(spec).split(",") if str(x).strip()}


def _build_image_window_audit(vlm, image_info, *, is_text_baseline,
                              operational_max_len=None):
    """window_audit block for one image-arm example, measured from the VLM's
    REAL input_ids (LocalVLM/VllmVLM.last_*, set by its last generate()). Shared
    by the single-pass and two-pass image paths so they cannot drift.

    Text-baseline (image_text) rows get a REAL block too. The VLM measures last_* on
    text-only generates as well — the builder runs right after generate(), so
    last_* is this example's, not a stale prior one. The text-baseline block
    carries measured_visual_tokens=0 and NO image fields (no
    visual_tokens_formula / pages_dropped), so the formula-MAE guard skips it.

    Returns None when no real local VLM just generated: ApiVLM/mocks expose no
    dict window_config or measured last_input_tokens. measured_visual_tokens
    (the image-pad count in input_ids) vs visual_tokens_formula is the cost-axis
    drift guard. Pure — no I/O; the caller still wraps it (instrumentation must
    never break a run).

    operational_max_len caps the EFFECTIVE window at what the run actually
    allows (vLLM's max_model_len), exactly as the text arm does — so
    fully_attended is comparable across backends regardless of the model's
    larger architectural window. When capped — here, or pre-capped inside
    VllmVLM's own window_config — the block records architectural_window +
    operational_window POST-classify, matching the text arm's _annotate
    classify() drops unknown window_config keys, so
    annotating the input dict alone would lose them.
    """
    wc = getattr(vlm, "window_config", None)
    last_input = getattr(vlm, "last_input_tokens", None)
    if not isinstance(wc, dict) or not isinstance(last_input, int):
        return None
    _arch = wc.get("context_window")
    _capped = False
    if (isinstance(operational_max_len, int) and operational_max_len > 0
            and isinstance(_arch, int) and operational_max_len < _arch):
        wc = {**wc, "context_window": operational_max_len}
        _capped = True

    def _annotate(block):
        if _capped:
            block["architectural_window"] = _arch
            block["operational_window"] = operational_max_len
        elif wc.get("architectural_window") is not None:
            # VllmVLM pre-caps its own window_config and records both values there.
            block["architectural_window"] = wc.get("architectural_window")
            block["operational_window"] = wc.get("operational_window")
        return block

    from codecompress import window_audit as _wa
    if is_text_baseline:
        # keep_last_text_tokens (the deliberate fair-text content budget) is NOT
        # flagged as pipeline truncation — parity with the text arm, whose
        # stage-3a' tail-truncate is unflagged too. An over-window prompt raises
        # inside the VLM → ERROR_GENERATING → no block, so pre == model_input.
        # Under the per-row window fit the text arm DOES tail-truncate its
        # background; flag it so fully_attended=False and
        # the pre-truncation size is recoverable.
        _ttrunc = bool(isinstance(image_info, dict) and image_info.get("fit_text_truncated"))
        _pre = last_input
        if _ttrunc:
            _pre = last_input + int(image_info.get("fit_text_pre_tokens", 0)) - int(
                image_info.get("fit_text_kept_tokens", 0))
        blk = _wa.classify(
            pre_truncation_tokens=_pre, model_input_tokens=last_input,
            window_config=wc, pipeline_truncated=_ttrunc, measured=True)
        blk["measured_visual_tokens"] = 0
        blk["generated_tokens"] = getattr(vlm, "last_generated_tokens", None)
        blk["latency_s"] = getattr(vlm, "last_latency_s", None)
        return _annotate(blk)
    # keep_last_pages drops earlier rendered pages BEFORE the model sees them —
    # a pre-render truncation of the visual context. Treat it as pipeline
    # truncation so the example is NOT counted fully_attended (the model never
    # saw the dropped pages). image_info carries total_pages_rendered + num_pages.
    pages_dropped = False
    if isinstance(image_info, dict):
        _total = image_info.get("total_pages_rendered")
        _kept = image_info.get("num_pages")
        if isinstance(_total, int) and isinstance(_kept, int) and _total > _kept:
            pages_dropped = True
        _dropped_front = image_info.get("pages_dropped_front")
        if isinstance(_dropped_front, int) and _dropped_front > 0:
            pages_dropped = True
    blk = _wa.classify(
        pre_truncation_tokens=last_input, model_input_tokens=last_input,
        window_config=wc, pipeline_truncated=pages_dropped, measured=True)
    blk["pages_dropped"] = pages_dropped
    blk["measured_visual_tokens"] = getattr(vlm, "last_image_pad_tokens", None)
    # MiniCPM-V unk-run mode only (window_audit.count_unk_run_tokens): the
    # per-run lengths behind measured_visual_tokens, for the drift check
    # (each crop = one 64-id <unk> run; abutting crops merge into multiples
    # of 64 — a non-multiple run exposes a counting anomaly). None on every
    # other model → key absent, block byte-identical.
    _runs = getattr(vlm, "last_visual_token_runs", None)
    if _runs is not None:
        blk["visual_token_runs"] = _runs
    blk["visual_tokens_formula"] = (image_info or {}).get("visual_tokens")
    blk["generated_tokens"] = getattr(vlm, "last_generated_tokens", None)
    blk["latency_s"] = getattr(vlm, "last_latency_s", None)
    return _annotate(blk)


def _image_extra_scores(original_data):
    """Aggregate image_info + latency stats for the image-pipeline SCORES.json.

    average_generation_latency_s keeps its historical semantics (mean over ALL
    latency-carrying rows). The first generate of a run pays one-off warmup
    (CUDA graphs / compile caches), inflating that mean 5-9% on short cells,
    so the aggregate ALSO emits
    average_generation_latency_s_excl_first — the mean excluding the FIRST row
    that carries a latency (None when <2 rows carry one; never a fake 0.0) —
    plus n_latency_rows. Returns None when no row has _image_info.
    """
    image_infos = [d["_image_info"] for d in original_data if "_image_info" in d]
    if not image_infos:
        return None
    avg_visual_tokens = sum(i["visual_tokens"] for i in image_infos) / len(image_infos)
    avg_actual_ratio = sum(i["actual_ratio"] for i in image_infos) / len(image_infos)
    # Latency axis: render_s (this arm; 0 for text baseline) +
    # generation latency_s (from window_audit) → end-to-end total. The analyzer
    # forms the within-run image/text total-latency ratio (the cost axis).
    avg_render_s = sum(i.get("render_s", 0.0) for i in image_infos) / len(image_infos)
    gen_lats = [d["_window_audit"]["latency_s"] for d in original_data
                if isinstance(d.get("_window_audit", {}).get("latency_s"), (int, float))]
    avg_gen_latency_s = (sum(gen_lats) / len(gen_lats)) if gen_lats else 0.0
    excl_first = gen_lats[1:]
    scores = {
        "average_visual_tokens": avg_visual_tokens,
        "average_actual_ratio": avg_actual_ratio,
        "average_render_s": avg_render_s,
        "average_generation_latency_s": avg_gen_latency_s,
        "average_generation_latency_s_excl_first": (
            sum(excl_first) / len(excl_first)) if excl_first else None,
        "n_latency_rows": len(gen_lats),
        "average_total_latency_s": avg_render_s + avg_gen_latency_s,
    }
    # Page-fit aggregates are independently conditional: rows from old
    # runs do not gain zero-valued keys, while partially instrumented rows sum
    # the values that are present.
    for source_key, aggregate_key in (
        ("pages_kept", "total_pages_kept"),
        ("pages_dropped_front", "total_pages_dropped_front"),
        ("budget_underfit", "n_budget_underfit"),
        ("char_px_below_threshold", "total_char_px_below_threshold"),
    ):
        if any(source_key in info for info in image_infos):
            scores[aggregate_key] = sum(
                info.get(source_key, 0) for info in image_infos
            )
    return scores


def _resolve_gpu_devices(cli_value, env_var):
    """Resolve a GPU-routing flag: the CLI value takes precedence over the env var.

    Explicit unset-checks (None/"") instead of `cli_value or env` — fire
    literal-evals `--generation_cuda_visible_devices 0` to int 0, which is falsy
    and was silently dropped in favor of the env var (GPU 0 was unselectable
    via CLI). Set values are canonicalized to the comma-joined
    string form the pipelines expect (int 3 → "3", tuple (0,1) → "0,1", str
    passthrough); bool is rejected (int subclass — would become "True").
    """
    if cli_value is None or cli_value == "":
        return os.environ.get(env_var, "")
    if isinstance(cli_value, bool):
        raise TypeError(f"bool is not a valid device spec for {env_var}: {cli_value!r}")
    if isinstance(cli_value, int):
        return str(cli_value)
    if isinstance(cli_value, (tuple, list)):
        if any(isinstance(d, bool) for d in cli_value):
            raise TypeError(f"bool is not a valid device spec for {env_var}: {cli_value!r}")
        return ",".join(str(d) for d in cli_value)
    if isinstance(cli_value, str):
        return cli_value
    raise TypeError(f"invalid device spec for {env_var}: {cli_value!r}")


def _parse_pad_page_to(value):
    """Parse the --pad_page_to CLI value into a (width, height) int tuple or None.

    Accepts the string form "896x1344" (or "896,1344" — but note fire
    literal-evals a bare `--pad_page_to 896,1344` into a tuple before we see
    it, so tuple/list forms are accepted too). None/"" → None (feature off).
    Anything else — wrong arity, junk text, a bare int, non-positive dims —
    raises ValueError (fail loud: a silently-dropped pad would break the
    padded page geometry without a trace).
    """
    if value is None or value == "":
        return None
    if isinstance(value, (tuple, list)):
        parts = list(value)
    elif isinstance(value, str):
        sep = "x" if "x" in value.lower() else ","
        parts = [p.strip() for p in value.lower().split(sep)]
    else:
        raise ValueError(
            f"pad_page_to must be 'WIDTHxHEIGHT' (e.g. '896x1344'), got {value!r}"
        )
    if len(parts) != 2:
        raise ValueError(
            f"pad_page_to must have exactly 2 dims 'WIDTHxHEIGHT', got {value!r}"
        )
    try:
        w, h = int(parts[0]), int(parts[1])
    except (TypeError, ValueError):
        raise ValueError(
            f"pad_page_to dims must be integers 'WIDTHxHEIGHT', got {value!r}"
        ) from None
    if w <= 0 or h <= 0:
        raise ValueError(f"pad_page_to dims must be positive, got {value!r}")
    return (w, h)


class EncoderFeasibilityBlocked(RuntimeError):
    """A row whose rendered page count exceeds the encoder-activation cap.

    Raised BEFORE the request reaches vLLM. With skip_mm_profiling the KV
    pool auto-sizes to fill gpu_util × VRAM on every rank, so encoder
    activations only get the (1−util) slack; one over-cap request OOMs the
    EngineCore FATALLY and takes every subsequent row of the cell with it.
    Flows through the pipeline's existing `except Exception` path →
    ERROR_GENERATING row with this message stored in image_info["error"] —
    the ENCODER-FEASIBILITY-BLOCKED label (distinct from window-blocking).
    """


def _enforce_page_cap(image_info, max_pages):
    """Raise EncoderFeasibilityBlocked if the rendered page count exceeds
    the smoke-proven cap. No-op when the cap is None or num_pages is absent
    (text-baseline rows carry num_pages=0 and can never trip it)."""
    if not max_pages:
        return
    num_pages = image_info.get("num_pages") or 0
    if num_pages > max_pages:
        raise EncoderFeasibilityBlocked(
            f"ENCODER-FEASIBILITY-BLOCKED: num_pages={num_pages} > "
            f"max_pages_per_request={max_pages} — row skipped pre-engine "
            f"(encoder-activation OOM guard)"
        )


class WindowFeasibilityBlocked(RuntimeError):
    """A row whose TEXT part alone (instruction + query + template + margin)
    leaves no page budget under --fit_window_tokens. Raised BEFORE
    render/engine with a labelled message so the row is an attributable
    infeasible row, never a vLLM raise. ``fit_info`` carries the per-row fit
    numbers so the blocked row keeps them."""

    def __init__(self, message, fit_info=None):
        super().__init__(message)
        self.fit_info = dict(fit_info or {})


def _row_fit_budget(pipeline, benchmark_runner, query_ctx, params):
    """Per-row page budget for the fit policy.

    budget = fit_window_tokens − tok(text prompt of THIS row, accounting
    tokenizer, no special tokens) − fit_template_tokens − fit_margin_tokens.
    The text prompt is previewed with an empty image list: every benchmark's
    build_image_prompt emits images as separate message items, so the text
    part is image-count independent (a prompt diff on the served prompts
    confirmed the served text equals this preview). Returns (budget, info) and raises
    WindowFeasibilityBlocked when budget <= 0.
    """
    tok = getattr(pipeline, "tokenizer", None)
    if tok is None:
        raise RuntimeError(
            "fit_window_tokens needs the image pipeline's accounting tokenizer "
            "(init_tokenizer) to be initialised before rendering"
        )
    preview = benchmark_runner.build_image_prompt([], query_ctx if query_ctx is not None else "")
    text_prompt = "\n".join(m["text"] for m in preview if m.get("type") == "text")
    return _fit_budget_from_text(tok, text_prompt, params)


def _row_fit_budget_text(pipeline, benchmark_runner, query_ctx, params):
    """Text-arm (image_text) twin of _row_fit_budget: the served text arm goes
    through the VLM path, which never truncates, so the tail-keep is realised
    here as a per-row CONTENT tail-keep of the background: budget =
    fit_window_tokens − tok(instruction + build_text_prompt("", query)) −
    template − margin. The row is then flagged pipeline_truncated
    (fully_attended=False) whenever the background exceeded the budget.
    Preview seam: measured on 25 real rows per task, the preview differs from
    served-minus-background by ≤2 tokens (lcc −1..+1, lca +1, longcodeqa
    −2..−1, repobench 0) — covered by the margin D."""
    tok = getattr(pipeline, "tokenizer", None)
    if tok is None:
        raise RuntimeError(
            "fit_window_tokens needs the image pipeline's accounting tokenizer "
            "(init_tokenizer) to be initialised before building the text prompt"
        )
    q = query_ctx if query_ctx is not None else ""
    preview = benchmark_runner.text_baseline_instruction() + benchmark_runner.build_text_prompt("", q)
    return _fit_budget_from_text(tok, preview, params)


def _fit_budget_from_text(tok, text_prompt, params):
    n_text = len(tok.encode(text_prompt, add_special_tokens=False))
    window = int(params["fit_window_tokens"])
    template = int(params.get("fit_template_tokens") or 0)
    margin = int(params.get("fit_margin_tokens") or 0)
    overhead = int(params.get("fit_per_page_overhead", 2) or 0)
    budget = window - n_text - template - margin
    info = {
        "fit_window_tokens": window,
        "fit_text_tokens": n_text,
        "fit_template_tokens": template,
        "fit_margin_tokens": margin,
        "fit_per_page_overhead": overhead,
        "fit_row_budget": budget,
    }
    if budget <= 0:
        raise WindowFeasibilityBlocked(
            f"WINDOW-FEASIBILITY-BLOCKED: text prompt alone ({n_text} tokens + "
            f"template {template} + margin {margin}) leaves no payload budget under "
            f"fit_window_tokens={window} — row skipped pre-engine",
            fit_info=info,
        )
    return budget, info


def _feasibility_block_fields(exc):
    """Machine-readable fields for pre-engine blocked rows:
    the per-row fit numbers survive on WINDOW-FEASIBILITY-BLOCKED rows and both
    block classes get an ``error_class``. Empty dict for any other exception."""
    if isinstance(exc, WindowFeasibilityBlocked):
        return {**exc.fit_info, "error_class": "WINDOW-FEASIBILITY-BLOCKED"}
    if isinstance(exc, EncoderFeasibilityBlocked):
        return {"error_class": "ENCODER-FEASIBILITY-BLOCKED"}
    return {}


def _enforce_fit_underfit(image_info, params):
    """Under the per-row window fit the renderer always keeps
    the LAST page even when that page alone exceeds the budget
    (budget_underfit). Such a row would reach vLLM and raise generically; turn
    it into a labelled pre-engine block with its fit numbers preserved."""
    if params.get("fit_window_tokens") is None or not isinstance(image_info, dict):
        return
    if not image_info.get("budget_underfit"):
        return
    kept = int(image_info.get("visual_tokens_kept", 0) or 0)
    pages = int(image_info.get("pages_kept", 0) or 0)
    overhead = int(params.get("fit_per_page_overhead", 2) or 0)
    budget = int(image_info.get("fit_row_budget", 0) or 0)
    if kept + overhead * pages > budget:
        raise WindowFeasibilityBlocked(
            f"WINDOW-FEASIBILITY-BLOCKED: last page alone ({kept} + {overhead}×{pages} tokens) "
            f"exceeds the row budget {budget} — row skipped pre-engine",
            fit_info={k: v for k, v in image_info.items() if k.startswith("fit_") or k.startswith("pages_") or k.startswith("visual_tokens_") or k == "budget_underfit"},
        )


def _resolve_fit_kwargs(pipeline, benchmark_runner, query_ctx, params):
    """Per-render fit kwargs: the per-row window budget when --fit_window_tokens
    is set, else the static keep_last_pages_fit_tokens.
    Returns (render_kwargs, image_info_additions)."""
    if params.get("fit_window_tokens") is not None:
        budget, info = _row_fit_budget(pipeline, benchmark_runner, query_ctx, params)
        return (
            {"keep_last_pages_fit_tokens": budget,
             "per_page_overhead": int(params.get("fit_per_page_overhead", 2) or 0)},
            info,
        )
    return {"keep_last_pages_fit_tokens": params.get("keep_last_pages_fit_tokens")}, {}


def _annotate_fit_window_audit(block, params, max_new_tokens, vlm):
    """Post-run instrument: under the per-row window fit,
    record whether the served prompt + answer cap overran the operational
    window (vLLM clips silently there) and the engine's finish_reason."""
    if params.get("fit_window_tokens") is None or not isinstance(block, dict):
        return block
    win = block.get("context_window")
    mi = block.get("model_input_tokens")
    if isinstance(win, int) and isinstance(mi, int):
        block["fit_window_overrun"] = (mi + int(max_new_tokens or 0)) > win
    fr = getattr(vlm, "last_finish_reason", None)
    if fr is not None:
        block["finish_reason"] = fr
    return block


def _validate_max_pages_per_request(value):
    """Validate the --max_pages_per_request CLI value → positive int or None.

    Fail loud on junk (a silently-dropped cap would let an over-cap row
    kill the EngineCore mid-cell without a trace)."""
    if value is None or value == "":
        return None
    try:
        cap = int(value)
    except (TypeError, ValueError):
        raise ValueError(
            f"max_pages_per_request must be a positive integer, got {value!r}"
        ) from None
    if cap <= 0:
        raise ValueError(
            f"max_pages_per_request must be a positive integer, got {value!r}"
        )
    return cap


def _generate_or_score_qa(pipeline, *, images, text_prompt, max_new_tokens,
                          system_prompt, qa_scoring, ground_truth):
    """Dispatch one image-path request; ``generate`` preserves the legacy call."""
    if qa_scoring != "likelihood":
        return pipeline.generate(
            images=images,
            text_prompt=text_prompt,
            max_new_tokens=max_new_tokens,
            system_prompt=system_prompt,
        ), None
    from codecompress.benchmarks.longcodeqa import (
        LIKELIHOOD_CHOICES,
        score_likelihood_choices,
    )

    choice_logprobs = pipeline.score_choices(
        images=images, text_prompt=text_prompt, choices=LIKELIHOOD_CHOICES,
        system_prompt=system_prompt,
    )
    result = score_likelihood_choices(choice_logprobs, ground_truth)
    return result["letter"], choice_logprobs


def _score_and_save(benchmark_runner, benchmark, method, model_name,
                    all_outputs, all_prompts, original_data,
                    result_dir, params, pipeline_type="text",
                    extra_scores_fields=None,
                    num_examples_requested=None, num_examples_loaded=None):
    """Score predictions and save results (shared by text and image pipelines).

    Args:
        benchmark_runner: The benchmark instance.
        benchmark: Benchmark name string (for metadata).
        method: Method name string.
        model_name: Model name string (for output paths).
        all_outputs: List of generated text strings.
        all_prompts: List of prompt strings (text) or description strings (image).
        original_data: List of original example dicts with cached fields.
        result_dir: Directory to write results.
        params: Dict of all parameters.
        pipeline_type: "text" or "image".
        extra_scores_fields: Optional dict of extra fields to include in SCORES.json.
    """
    # Handle length mismatch (can happen if a batch fails mid-way)
    if len(all_outputs) != len(original_data):
        logger.warning(
            f"Output/data length mismatch: {len(all_outputs)} outputs vs "
            f"{len(original_data)} examples. Truncating to shorter.")
        min_len = min(len(all_outputs), len(original_data))
        all_outputs = all_outputs[:min_len]
        original_data = original_data[:min_len]
        all_prompts = all_prompts[:min_len]

    # Build output paths
    if pipeline_type == "image":
        method_suffix = _build_image_method_suffix(
            method, params.get("compression_ratio", 1.0),
            params.get("image_control", "matched"),
            max_visual_tokens=params.get("max_visual_tokens"),
            num_image_splits=params.get("num_image_splits"))
    else:
        method_suffix = build_method_suffix(method, **params)
    method_result_dir = os.path.join(result_dir, method_suffix)
    os.makedirs(method_result_dir, exist_ok=True)
    model_slug = model_name.replace("/", "_slash_")
    model_output_path = os.path.join(method_result_dir, f"{model_slug}.jsonl")
    score_output_path = os.path.join(method_result_dir, f"{model_slug}-SCORES.json")

    # Score each example via benchmark and write per-example results
    total_scores = {name: 0.0 for name in benchmark_runner.metric_names()}
    valid_scores = 0
    logger.info(f"Scoring {len(all_outputs)} examples...")

    with open(model_output_path, "w") as f_out:
        for i, (output, orig, prompt) in enumerate(zip(all_outputs, original_data, all_prompts)):
            scores = benchmark_runner.score(output, orig)

            if output != "ERROR_GENERATING" and orig.get("_ground_truth") is not None:
                try:
                    # Build ALL metric contributions before mutating any
                    # running total: if scores[name] is
                    # missing for the 2nd metric, the old code had already
                    # incremented the 1st metric's total without bumping
                    # valid_scores, inflating that metric's average. Now a
                    # partial failure leaves totals + valid_scores untouched.
                    contrib = {name: scores[name]
                               for name in benchmark_runner.metric_names()}
                except Exception as e:
                    logger.error("Scoring error for example {}: {}", orig.get('id', i), str(e))
                else:
                    for name, value in contrib.items():
                        total_scores[name] += value
                    valid_scores += 1

            result_record = {
                "id": orig.get("id", i),
                "ground_truth": orig.get("_ground_truth"),
                "prompt": prompt,
                "output": output,
                **{k.lower(): v for k, v in scores.items()},
            }
            # Include image info if present
            if "_image_info" in orig:
                result_record["image_info"] = orig["_image_info"]
            # Record tokens-the-model-saw for the text side. Uses the model's
            # own tokenizer if reachable.
            if "_prompt_token_count" in orig:
                result_record["prompt_token_count"] = orig["_prompt_token_count"]
            # Context-window block (codecompress/window_audit.py): whether the
            # model's window was saturated and how it handled it. Built by the
            # pipeline (which has the model config + measured token counts);
            # persisted here verbatim, parallel to image_info. Absent on old
            # runs → row simply omits it (analyzer treats missing as 'unknown').
            if "_window_audit" in orig:
                result_record["window_audit"] = orig["_window_audit"]
            if params.get("qa_scoring") == "likelihood":
                result_record["scoring"] = "likelihood"
                result_record["choice_logprobs"] = orig["_choice_logprobs"]
            f_out.write(json.dumps(result_record) + "\n")

    logger.info(f"Per-example results saved to {model_output_path}")

    # Compute and save aggregate scores
    avg_scores = {name: (total_scores[name] / valid_scores if valid_scores > 0 else 0)
                  for name in benchmark_runner.metric_names()}

    if pipeline_type == "image":
        scores_params = _build_scores_params_image(method, **params)
    else:
        scores_params = _build_scores_params_text(method, **params)

    scores_json = {
        "model_name": model_name,
        "benchmark": benchmark,
        "method": method,
        "pipeline": pipeline_type,
        "num_examples_scored": valid_scores,
        "num_examples_total": len(original_data),
        **{f"average_{name.lower()}": avg_scores.get(name, 0) for name in benchmark_runner.metric_names()},
        "parameters": scores_params,
    }
    # Requested vs loaded so a loader-stage drop (loaded < requested) is
    # visible. num_examples_total == loaded once the pipeline placeholders
    # every row; requested is the manifest/--num_examples.
    if num_examples_loaded is not None:
        scores_json["num_examples_loaded"] = num_examples_loaded
    if num_examples_requested is not None:
        scores_json["num_examples_requested"] = num_examples_requested
    if extra_scores_fields:
        scores_json.update(extra_scores_fields)
    _wa_agg = _aggregate_window_audit(original_data, true_n=len(original_data))
    if _wa_agg is not None:
        scores_json["window_audit"] = _wa_agg
        _wa_warn = _window_audit_warning(_wa_agg)
        if _wa_warn:
            logger.warning(_wa_warn)

    metric_summary = ", ".join(
        f"Avg {name} = {avg_scores.get(name, 0):.2f}" for name in benchmark_runner.metric_names()
    )
    logger.info(f"Method {method}: {metric_summary} "
                f"({valid_scores}/{len(original_data)} scored)")
    save_json(scores_json, score_output_path)
    logger.info(f"Scores saved to {score_output_path}")


# =============================================================================
# Text pipeline flow (stages 2-6)
# =============================================================================

def _run_text_pipeline(pipeline, benchmark_runner, dataset, method, params,
                       model_name, compression_model_name, embed_model_name,
                       max_new_tokens, batch_size, vllm_max_model_len,
                       gpu_memory_utilization, tensor_parallel_size,
                       trust_remote_code, enforce_eager, use_chat_template,
                       compression_devices, generation_devices):
    """Run text compression + vLLM generation. Returns (all_outputs, all_prompts, original_data)."""
    import torch
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ---- Stage 2: Initialize compression models ----
    pipeline.init_compression(
        method, device, compression_devices,
        model_name=model_name,
        embed_model_name=embed_model_name,
        compression_model_name=compression_model_name,
    )

    # ---- Stage 3: Compress each example's background context ----
    logger.info(f"--- Compressing with method: {method} ---")
    all_prompts = []
    original_data = []

    for i, example in enumerate(tqdm(dataset, desc=f"Preparing prompts ({method})")):
        background_ctx, query_ctx = benchmark_runner.split_context(example)
        ground_truth = benchmark_runner.ground_truth(example)
        language = example.get("language", "python")

        try:
            # Stage 3a-pre: Apply code manipulation (indent / syntax ablation)
            if params.get("strip_indent"):
                from codecompress.code_renderer import strip_indent as _si
                background_ctx = _si(background_ctx)
            elif params.get("shuffle_indent"):
                from codecompress.code_renderer import shuffle_indent as _shi
                import hashlib
                seed = int(hashlib.sha256(background_ctx.encode()).hexdigest(), 16) % (2**31)
                background_ctx = _shi(background_ctx, seed=seed)
            if params.get("strip_braces"):
                from codecompress.code_renderer import strip_braces as _sb
                background_ctx = _sb(background_ctx)
            if params.get("strip_semicolons"):
                from codecompress.code_renderer import strip_semicolons as _ss
                background_ctx = _ss(background_ctx)

            # Stage 3a: Pre-truncate background if limit specified (CLI or benchmark)
            tokenizer = pipeline.resources.get("tokenizer")
            ctx_limit = params.get("max_context_tokens") or benchmark_runner.max_context_tokens()
            if ctx_limit and tokenizer and method == "full":
                ctx_tokens = tokenizer.encode(background_ctx)
                if len(ctx_tokens) > ctx_limit:
                    logger.warning(
                        f"Background context ({len(ctx_tokens)} tokens) exceeds "
                        f"benchmark limit ({ctx_limit}), head-truncating.")
                    background_ctx = tokenizer.decode(
                        ctx_tokens[:ctx_limit], skip_special_tokens=True)

            # Stage 3a'. Budget control: tail-truncate the background
            # to the last N text tokens (keeps most-recent context, which is
            # the most useful for completion on LCC). Applies to any method,
            # not just "full". Runs AFTER the optional head-truncate in 3a;
            # combining both with method="full" means first prefix then suffix
            # — avoid that pairing unless you intend the double-cut.
            keep_last_tt = params.get("keep_last_text_tokens")
            if keep_last_tt is not None:
                if tokenizer is None:
                    logger.warning(
                        f"keep_last_text_tokens={keep_last_tt} requested but no "
                        f"tokenizer is loaded for method={method}; silently skipping. "
                        f"Run metadata will be misleading — fix the pipeline setup."
                    )
                else:
                    from codecompress.pipelines.text import keep_last_text_tokens as _tail_trunc
                    background_ctx = _tail_trunc(background_ctx, keep_last_tt, tokenizer)

            # Stage 3b: Compress background
            compressed_bg = pipeline.compress(method, background_ctx, query_ctx, params)

            # Stage 3c: Assemble final prompt
            prompt = benchmark_runner.build_text_prompt(compressed_bg, query_ctx).strip()

            # Stage 3d: Truncate if prompt exceeds model's context window
            # window block (text arm): _wa_pre_tokens is the pre-truncation
            # length (true pressure); _wa_truncated flags our budget cut. Both
            # use add_special_tokens=True to match what the model receives.
            _wa_pre_tokens = None
            _wa_truncated = False
            if tokenizer:
                # clamp: max_new_tokens >= max_model_len would flip tokenized[-budget:]
                # into a HEAD-drop
                budget = max(1, vllm_max_model_len - max_new_tokens)
                tokenized = tokenizer.encode(prompt)
                _wa_pre_tokens = len(tokenized)
                if len(tokenized) > budget:
                    logger.warning(
                        f"Prompt ({len(tokenized)} tokens) exceeds budget ({budget}), "
                        f"keeping the last {budget} tokens.")
                    prompt = tokenizer.decode(tokenized[-budget:])
                    _wa_truncated = True

            all_prompts.append(prompt)
            # Record actual token count the model saw.
            prompt_token_count = None
            _wa_input_tokens = None
            if tokenizer is not None:
                try:
                    prompt_token_count = len(tokenizer.encode(prompt, add_special_tokens=False))
                    _wa_input_tokens = len(tokenizer.encode(prompt))  # final, with special tokens
                except Exception:
                    pass
            original_data.append({
                **example,
                "id": example.get("id", i),
                "_ground_truth": ground_truth,
                "_background_ctx": background_ctx,
                "_query_ctx": query_ctx,
                "_prompt_token_count": prompt_token_count,
                "_wa_pre_tokens": _wa_pre_tokens,
                "_wa_input_tokens": _wa_input_tokens,
                "_wa_truncated": _wa_truncated,
                "language": language,
            })
        except Exception as e:
            logger.warning(
                f"Error processing example {i} (ID: {example.get('id', 'N/A')}): {e}",
                exc_info=True,
            )
            # Keep the row as a prep-failure placeholder instead of
            # `continue` (which silently shrank num_examples_total with no
            # marker — asymmetric vs the image pipeline). A None prompt means
            # "no generation"; stage 6 emits ERROR_GENERATING for it, so the
            # row survives 1:1 with the dataset and the completion gate stays
            # honest.
            all_prompts.append(None)
            original_data.append({
                **example,
                "id": example.get("id", i),
                "_ground_truth": ground_truth,
                "_background_ctx": background_ctx,
                "_query_ctx": query_ctx,
                "_prompt_token_count": None,
                "language": language,
            })

    # ---- Stage 4: Free compression models ----
    pipeline.free_compression()

    # ---- Stage 5: Initialize generation LLM ----
    # all_prompts may hold None sentinels for prep-failed rows.
    real_idx = [j for j, p in enumerate(all_prompts) if p is not None]
    if not real_idx:
        logger.error(f"No valid prompts for method {method} "
                     f"({len(all_prompts)} prep-failed). Returning placeholders.")
        return (["ERROR_GENERATING"] * len(all_prompts),
                ["" for _ in all_prompts], original_data)

    configure_cuda_visible_devices(generation_devices, "generation")
    pipeline.init_generation(
        model_name=model_name,
        gpu_memory_utilization=gpu_memory_utilization,
        tensor_parallel_size=tensor_parallel_size,
        vllm_max_model_len=vllm_max_model_len,
        trust_remote_code=trust_remote_code,
        enforce_eager=enforce_eager,
        use_chat_template=use_chat_template,
        quantize=params.get("quantize"),
        max_num_seqs=params.get("vllm_max_num_seqs"),
        engine_recycle_every=params.get("engine_recycle_every"),
    )

    # ---- Extract the model's window config (text arm), now that it's loaded ----
    # INSTRUMENTATION — wrapped so it can never break a run. The per-example
    # window_audit is BUILT AFTER generation (below) so it can use the TRUE token
    # counts the model saw (captured per prompt in _gen_stats, both backends).
    # Backend-aware: transformers reads _gen_model.config; vLLM (no _gen_model)
    # loads the config via AutoConfig so the block is NOT silently skipped.
    _twc = None
    try:
        _twc = _text_window_config(pipeline, model_name, trust_remote_code)
    except Exception as _wae:  # noqa: BLE001 — instrumentation must not fail a run
        logger.warning(f"window_audit (text) config extraction skipped: {_wae}")

    # ---- Stage 6: Generate completions in batches ----
    # Generate only the real prompts, then re-align outputs to original_data
    # order with ERROR_GENERATING in the prep-failed (None) slots so all three
    # returned lists stay 1:1 with the loaded dataset.
    real_prompts = [all_prompts[j] for j in real_idx]
    gen_outputs = []
    logger.info(f"Generating completions for {len(real_prompts)} prompts...")
    for i in tqdm(range(0, len(real_prompts), batch_size), desc=f"Generating ({method})"):
        batch = real_prompts[i:i + batch_size]
        if not batch:
            continue
        try:
            gen_outputs.extend(pipeline.generate(
                batch, max_new_tokens=max_new_tokens,
                system_prompt=benchmark_runner.system_prompt(),
            ))
        except Exception as e:
            logger.error("Generation error at batch index {}: {}", i, str(e))
            gen_outputs.extend(["ERROR_GENERATING"] * len(batch))

    # ---- Build per-example window block (text arm) from TRUE generation stats ----
    # INSTRUMENTATION — wrapped so it can never break a run. Stats are 1:1 with
    # real_prompts (== real_idx order). Attach positionally ONLY when the counts
    # match exactly; a mid-run generation error would desync them, so in that
    # case every row falls back to prep-time counts (never misattribute).
    if _twc is not None:
        try:
            _stats = getattr(pipeline, "_gen_stats", None)
            _aligned = isinstance(_stats, list) and len(_stats) == len(real_idx)
            for _k, _j in enumerate(real_idx):
                _o = original_data[_j]
                _stat = _stats[_k] if _aligned else None
                _blk = _build_text_window_audit(
                    _twc, raw_pre_tokens=_o.get("_wa_pre_tokens"), stat=_stat,
                    fallback_input_tokens=_o.get("_wa_input_tokens"),
                    fallback_truncated=bool(_o.get("_wa_truncated")),
                    operational_max_len=vllm_max_model_len)
                if _blk is not None:
                    _o["_window_audit"] = _blk
        except Exception as _wae:  # noqa: BLE001 — instrumentation must not fail a run
            logger.warning(f"window_audit (text) build skipped: {_wae}")

    all_outputs = ["ERROR_GENERATING"] * len(all_prompts)
    for k, j in enumerate(real_idx):
        if k < len(gen_outputs):
            all_outputs[j] = gen_outputs[k]
    all_prompts = [p if p is not None else "" for p in all_prompts]

    return all_outputs, all_prompts, original_data


# =============================================================================
# Image pipeline flow (stages 2-3)
# =============================================================================

def _apply_image_control(all_rendered, image_control, vlm_model=""):
    """Apply image control condition to rendered examples.

    Modifies images in-place for mismatched/blank/noise conditions.
    For 'matched', this is a no-op.
    """
    if image_control == "matched":
        return

    from PIL import Image
    import numpy as np
    from codecompress.code_renderer import count_image_tokens

    N = len(all_rendered)

    # These fields describe the selected image set, not the receiving example.
    # A mismatched control must move them with the images as one provenance
    # block; otherwise the recipient would retain metrics for images it lost.
    policy_fields = (
        "total_pages_rendered",
        "pages_total",
        "pages_kept",
        "pages_dropped_front",
        "budget_underfit",
        "visual_tokens_total",
        "visual_tokens_total_estimator",
        "visual_tokens_kept",
        "visual_tokens_law",
        "page_char_px",
        "page_char_px_effective",
        "min_char_px_effective",
        "char_px_below_threshold",
        "gemma_max_soft_tokens_effective",
        "pages_rasterized",
    )
    page_policy_enabled = any(
        "visual_tokens_law" in item.get("image_info", {}) for item in all_rendered
    )

    if image_control == "mismatched":
        # Circular shift: example i gets images from example (i+50)%N
        shift = min(50, N // 2)
        if shift == 0:
            logger.warning("image_control=mismatched with N=1: shift=0, control is identical to matched")
        shifted = [all_rendered[(i + shift) % N]["images"] for i in range(N)]
        if not page_policy_enabled:
            # Exact legacy path: new knobs unset must not change existing
            # mismatched-control metadata semantics.
            for i in range(N):
                all_rendered[i]["images"] = shifted[i]
                vtok = count_image_tokens(shifted[i], model_name=vlm_model)
                all_rendered[i]["image_info"]["visual_tokens"] = vtok
                bg_tokens = all_rendered[i]["image_info"].get("bg_text_tokens", 0)
                all_rendered[i]["image_info"]["actual_ratio"] = (
                    round(bg_tokens / vtok, 2) if vtok > 0 else 0
                )
                all_rendered[i]["image_info"]["control"] = "mismatched"
            return
        shifted_visual_tokens = [
            all_rendered[(i + shift) % N]["image_info"].get("visual_tokens")
            for i in range(N)
        ]
        shifted_policy = [
            {
                key: all_rendered[(i + shift) % N]["image_info"][key]
                for key in policy_fields
                if key in all_rendered[(i + shift) % N]["image_info"]
            }
            for i in range(N)
        ]
        for i in range(N):
            all_rendered[i]["images"] = shifted[i]
            for key in policy_fields:
                all_rendered[i]["image_info"].pop(key, None)
            all_rendered[i]["image_info"].update(shifted_policy[i])
            all_rendered[i]["image_info"]["num_pages"] = len(shifted[i])
            all_rendered[i]["image_info"]["image_size"] = (
                f"{shifted[i][0].width}x{shifted[i][0].height}"
                if shifted[i]
                else "N/A"
            )
            vtok = shifted_visual_tokens[i]
            if not isinstance(vtok, int):
                vtok = count_image_tokens(shifted[i], model_name=vlm_model)
            all_rendered[i]["image_info"]["visual_tokens"] = vtok
            bg_tokens = all_rendered[i]["image_info"].get("bg_text_tokens", 0)
            all_rendered[i]["image_info"]["actual_ratio"] = round(bg_tokens / vtok, 2) if vtok > 0 else 0
            all_rendered[i]["image_info"]["control"] = "mismatched"

    elif image_control in ("blank", "noise"):
        rng = np.random.RandomState(42)
        for i in range(N):
            original_images = all_rendered[i]["images"]
            new_images = []
            for img in original_images:
                w, h = img.size
                if image_control == "blank":
                    new_images.append(Image.new("RGB", (w, h), (255, 255, 255)))
                else:  # noise
                    arr = rng.randint(0, 256, (h, w, 3), dtype=np.uint8)
                    new_images.append(Image.fromarray(arr))
            all_rendered[i]["images"] = new_images
            if not page_policy_enabled:
                vtok = count_image_tokens(new_images, model_name=vlm_model)
                all_rendered[i]["image_info"]["visual_tokens"] = vtok
                all_rendered[i]["image_info"]["control"] = image_control
                continue
            # The replacement pages preserve geometry/token cost, but blank or
            # random pixels contain no rendered characters. Retaining char-px
            # claims would attach stale legibility measurements to content that
            # no longer exists.
            for key in (
                "page_char_px",
                "page_char_px_effective",
                "min_char_px_effective",
                "char_px_below_threshold",
            ):
                all_rendered[i]["image_info"].pop(key, None)
            all_rendered[i]["image_info"]["control"] = image_control


def _text_baseline_prompt_and_tokens(benchmark_runner, background_ctx, query_ctx,
                                     tokenizer, keep_last_tt):
    """Fair-text (image_text) arm: build the text prompt + the DELIVERED background
    token count.

    Optionally tail-truncate the background to a delivered-input-token
    budget (`keep_last_text_tokens`) BEFORE building the prompt, so the reported token
    count is the delivered count (the budget x-axis), not the pre-truncation total.
    `keep_last_tt=None` reproduces the prior full-context behavior byte-for-byte.
    """
    if keep_last_tt is not None and not tokenizer:
        # Mirror the text pipeline's warning (main.py:600): a requested budget that
        # cannot be enforced must be loud, not silently dropped (run-metadata honesty).
        logger.warning(
            "keep_last_text_tokens=%s requested but no tokenizer loaded; "
            "background NOT truncated (delivered budget unenforced)", keep_last_tt)
    if keep_last_tt is not None and tokenizer and background_ctx:
        from codecompress.pipelines.text import keep_last_text_tokens as _tail_trunc
        background_ctx = _tail_trunc(background_ctx, keep_last_tt, tokenizer)
    bg_tokens = 0
    if tokenizer and background_ctx:
        bg_tokens = len(tokenizer.encode(background_ctx, add_special_tokens=False))
    # Benchmark-aware instruction: LCC supplies the completion prefix (its
    # build_text_prompt has none → reproduces byte-for-byte); benchmarks whose
    # build_text_prompt is already self-contained (LCA summarization) return ""
    # so the text baseline is NOT given a contradictory completion steer on top
    # of its task instruction (fair-prompt parity).
    instruction = benchmark_runner.text_baseline_instruction()
    text_prompt = instruction + benchmark_runner.build_text_prompt(background_ctx, query_ctx)
    return text_prompt, bg_tokens


def _run_image_pipeline(pipeline, benchmark_runner, dataset, method, params,
                        vlm_model, vlm_backend, vlm_api_key, vlm_api_base_url,
                        compression_ratio, max_new_tokens, model_name,
                        image_control="matched", max_visual_tokens=None,
                        gemma_max_soft_tokens=None, generation_devices="",
                        vlm_revision=None):
    """Run image rendering + VLM generation. Returns (all_outputs, all_prompts, original_data)."""
    # Restrict the local VLM to the requested generation GPUs BEFORE any vLLM/torch
    # init. The text pipeline does this (configure_cuda_visible_devices at generation);
    # the image pipeline previously did NOT, so VllmVLM inited on the default physical
    # GPUs (e.g. 0,1) and OOM'd on a shared box where those belong to another user.
    # Applies to both the single-pass and two-pass (image_control=matched) paths.
    configure_cuda_visible_devices(generation_devices, "generation")
    # Extract style from method name (e.g., "image_highlight" → "highlight")
    style = IMAGE_METHODS[method]
    is_text_baseline = (style == TEXT_BASELINE)
    is_flat_baseline = (style == FLAT_BASELINE)

    _image_only_flags = []
    if params.get("keep_last_pages_fit_tokens") is not None:
        _image_only_flags.append("keep_last_pages_fit_tokens")
    if params.get("record_page_metrics", False):
        _image_only_flags.append("record_page_metrics")
    if params.get("min_char_px") is not None:
        _image_only_flags.append("min_char_px")
    if is_text_baseline and _image_only_flags:
        raise ValueError(
            f"{', '.join(_image_only_flags)} are image-render policy flags "
            "and cannot be used with the image_text baseline"
        )

    # Fail-fast (at run start, before the model load / per-example loop):
    # strip_braces / strip_semicolons are consumed by the TEXT pipeline
    # (main.py Stage 3a-pre) and by the image_text baseline branch below —
    # pipeline.render has NO such params, so on a rendered image method they
    # would be silently unconsumed and every "ablated" arm would render the
    # unablated code byte-for-byte (a silent no-op ablation).
    # strip_indent / shuffle_indent DO reach the render
    # path (pipeline.render kwargs) and stay allowed on rendered arms.
    if not is_text_baseline:
        _unconsumed = [f for f in ("strip_braces", "strip_semicolons")
                       if params.get(f)]
        if _unconsumed:
            raise ValueError(
                f"{'/'.join(_unconsumed)} are not consumed by the image RENDER "
                f"path (method={method!r}): the rendered images would silently "
                f"contain the UNABLATED code. These flags apply only to "
                f"--pipeline text or the image_text baseline. For rendered-arm "
                f"ablations use strip_indent/shuffle_indent (which reach "
                f"pipeline.render)."
            )

    # Fail-fast: the text-baseline integrity fingerprint (bg_sha256) is
    # computed post-ablation but BEFORE the
    # keep_last_text_tokens tail-truncation (which happens downstream inside
    # _text_baseline_prompt_and_tokens). Combining ablation with tail-
    # truncation would let two arms carry DIFFERENT fingerprints while
    # delivering byte-IDENTICAL backgrounds (ablation-affected bytes all in
    # the truncated-away head) — a false PASS of an integrity check. Ablations
    # use content budgets (keep_first_content_*), never tail-truncation, so
    # refuse the pairing.
    _ablation_flags_set = [f for f in ("strip_indent", "shuffle_indent",
                                       "strip_braces", "strip_semicolons")
                           if params.get(f)]
    if (is_text_baseline and _ablation_flags_set
            and params.get("keep_last_text_tokens") is not None):
        raise ValueError(
            f"{'/'.join(_ablation_flags_set)} cannot be combined with "
            f"keep_last_text_tokens on the image_text baseline: the bg_sha256 "
            f"integrity fingerprint is computed before the tail-truncation, so "
            f"the anchor-vs-ablated byte-diff check could false-PASS on "
            f"identical delivered bytes. Use a content-matched head-keep "
            f"(keep_first_content_tokens/_lines) instead."
        )

    # Fail fast on unsupported split combinations BEFORE the expensive model load.
    _n_split = params.get("num_image_splits")
    if _n_split is not None:
        if is_text_baseline or is_flat_baseline:
            raise ValueError(
                "num_image_splits has no effect with the image_text/image_flat "
                "baselines (no rendering / a single flattened line). Drop one."
            )
        if image_control != "matched":
            raise ValueError(
                "num_image_splits is only supported with image_control='matched' "
                "(the two-pass mismatched-control path does not render splits). "
                f"Got image_control={image_control!r}."
            )
        if params.get("split_policy") == "blocks":
            # The renderer + pipeline support 'blocks', but the CLI does not yet
            # source per-snippet seam_offsets from the benchmark — wiring that is
            # a follow-up. Fail fast rather than crash per example.
            raise ValueError(
                "split_policy='blocks' is not yet wired through the CLI (it needs "
                "per-snippet seam_offsets from the benchmark). Use 'lines' for now."
            )
        if style == "ide_lint" and _n_split != 1:
            raise ValueError(
                "ide_lint is incompatible with num_image_splits>1 (lint line "
                "numbers refer to the whole body, not a chunk). Use "
                "num_image_splits=1 or a non-lint style."
            )
        if params.get("keep_last_pages") is not None:
            raise ValueError(
                "keep_last_pages truncates AFTER render_split, silently shrinking "
                "the split set (the model sees a fraction of the code) and "
                "corrupting the lines-per-image axis. Drop one of "
                "keep_last_pages / num_image_splits."
            )
        if params.get("keep_last_pages_fit_tokens") is not None:
            raise ValueError(
                "keep_last_pages_fit_tokens is incompatible with "
                "num_image_splits; fit operates on emergent rendered pages."
            )

    if (
        params.get("keep_last_pages") is not None
        and params.get("keep_last_pages_fit_tokens") is not None
    ):
        raise ValueError(
            "keep_last_pages and keep_last_pages_fit_tokens are mutually "
            "exclusive tail-retention policies"
        )
    _fit_window = params.get("fit_window_tokens")
    if _fit_window is not None:
        if (not isinstance(_fit_window, int) or isinstance(_fit_window, bool)
                or _fit_window <= 0):
            raise ValueError(
                f"fit_window_tokens must be a positive integer, got {_fit_window!r}"
            )
        # The companions are validated at run start too —
        # a negative template/margin silently inflates every budget, a negative
        # overhead would fail inside every render.
        for _k in ("fit_template_tokens", "fit_margin_tokens", "fit_per_page_overhead"):
            _v = params.get(_k, 0)
            if _v is None:
                continue
            if not isinstance(_v, int) or isinstance(_v, bool) or _v < 0:
                raise ValueError(f"{_k} must be a non-negative integer, got {_v!r}")
        for _other in ("keep_last_pages_fit_tokens", "keep_last_pages", "num_image_splits",
                       "keep_last_text_tokens"):
            if params.get(_other) is not None:
                raise ValueError(
                    f"fit_window_tokens and {_other} are mutually exclusive "
                    "(per-row window fit vs fixed page/text policies)"
                )
    _fit_budget = params.get("keep_last_pages_fit_tokens")
    if _fit_budget is not None and (
        not isinstance(_fit_budget, int)
        or isinstance(_fit_budget, bool)
        or _fit_budget <= 0
    ):
        raise ValueError(
            "keep_last_pages_fit_tokens must be a positive integer, got "
            f"{_fit_budget!r}"
        )
    _validate_min_char_px(params.get("min_char_px"))
    if params.get("record_page_metrics") and _n_split is None and not all(
        params.get(name) is not None
        for name in (
            "fixed_image_width",
            "fixed_image_height",
            "fixed_font_size",
        )
    ):
        raise ValueError(
            "record_page_metrics requires the complete fixed-layout triple "
            "before model initialization"
        )

    # ---- Stage 2: Initialize VLM + tokenizer ----
    # Token accounting (bg_text_tokens, keep_last_text_tokens, ratio budgets)
    # counts with the VLM'S OWN tokenizer — the model that consumes the text.
    # Historically it used --model_name's tokenizer (default Qwen2.5-Coder) even
    # when the VLM was Gemma → ~21% undercount.
    # Covers BOTH the single-pass and two-pass paths (init happens before the
    # two-pass dispatch below).
    if params.get("legacy_token_accounting"):
        accounting_tokenizer = model_name
        logger.warning(
            f"LEGACY TOKEN ACCOUNTING: counting text tokens with --model_name's "
            f"tokenizer ('{model_name}') while the VLM is '{vlm_model}'. Token "
            f"budgets/ratios are in the WRONG unit unless the tokenizers match — "
            f"use only to reproduce historical runs.")
    elif vlm_backend == "api":
        # API VLMs (gpt-4o etc.) are not HF repos — their tokenizer is
        # unobtainable, so accounting stays in --model_name units.
        accounting_tokenizer = model_name
        logger.warning(
            f"API VLM '{vlm_model}': its tokenizer is unobtainable, so token "
            f"accounting counts with --model_name's tokenizer ('{model_name}') — "
            f"budgets/ratios are NOT in the API model's own token units.")
    else:
        accounting_tokenizer = vlm_model
        if model_name != vlm_model:
            logger.warning(
                f"Token accounting uses the VLM's tokenizer ('{vlm_model}'), NOT "
                f"--model_name's ('{model_name}'). Pre-fix runs counted with "
                f"--model_name; --legacy_token_accounting reproduces that.")
    # The accounting tokenizer follows the engine's
    # pin whenever it IS the VLM's tokenizer (never for --model_name legacy
    # accounting or the API backend, whose tokenizer is a different repo).
    _acct_rev = (vlm_revision or None) if accounting_tokenizer == vlm_model else None
    # kwarg only when pinned: unpinned pipelines/test doubles keep their exact
    # call shape.
    pipeline.init_tokenizer(accounting_tokenizer, **({"revision": _acct_rev} if _acct_rev else {}))
    params["token_accounting_revision"] = _acct_rev
    pipeline.init_generation(
        vlm_backend=vlm_backend,
        vlm_model=vlm_model,
        vlm_revision=vlm_revision or None,
        vlm_api_key=vlm_api_key,
        vlm_api_base_url=vlm_api_base_url,
        gemma_max_soft_tokens=gemma_max_soft_tokens,
        tensor_parallel_size=params.get("tensor_parallel_size", 1),
        gpu_memory_utilization=params.get("gpu_memory_utilization", 0.9),
        vllm_max_model_len=params.get("vllm_max_model_len", 32768),
        max_num_seqs=params.get("vllm_max_num_seqs"),
        engine_recycle_every=params.get("engine_recycle_every"),
        enforce_eager=params.get("enforce_eager", False),
        trust_remote_code=params.get("trust_remote_code", True),
    )

    # ---- Stage 3: Render + generate for each example ----
    if is_text_baseline:
        desc = "Image (text baseline)"
    elif is_flat_baseline:
        desc = "Image (flat baseline)"
    else:
        desc = f"Image ({style} {compression_ratio}x)"
    logger.info(f"--- Image pipeline: style={style}, ratio={compression_ratio}x, control={image_control} ---")

    # For non-matched controls on image methods, use two-pass:
    #   Pass 1: render all examples → Pass 2: apply control → generate
    # For matched or text_baseline, use single-pass (no extra memory).
    use_two_pass = (image_control != "matched" and not is_text_baseline)

    if use_two_pass:
        return _run_image_pipeline_two_pass(
            pipeline, benchmark_runner, dataset, method, params,
            style, is_flat_baseline, compression_ratio, max_new_tokens,
            image_control, desc,
            max_visual_tokens=max_visual_tokens,
        )

    # ---- Single-pass: render and generate per example (original flow) ----
    all_outputs = []
    all_prompts = []
    original_data = []

    import hashlib
    from time import perf_counter

    from codecompress.code_renderer import (
        shuffle_indent as _ab_shuffle_indent,
        strip_braces as _ab_strip_braces,
        strip_indent as _ab_strip_indent,
        strip_semicolons as _ab_strip_semicolons,
    )
    from codecompress.pipelines.content_select import resolve_content_budget

    _skip_ids = parse_skip_example_ids(params.get("skip_example_ids"))
    for i, example in enumerate(tqdm(dataset, desc=desc)):
        if i in _skip_ids:
            logger.warning(f"Example {i}: SKIPPED via --skip_example_ids (poison row; ERROR-blocked)")
            all_outputs.append("ERROR_GENERATING")
            all_prompts.append("")
            original_data.append({
                **example,
                "id": example.get("id", i),
                "error": "SKIPPED via skip_example_ids (deterministic engine hang)",
                "image_info": {"mode": "skipped", "num_pages": 0, "visual_tokens": 0,
                                "bg_text_tokens": 0, "actual_ratio": 0.0, "render_s": 0.0},
            })
            continue
        background_ctx, query_ctx = benchmark_runner.split_context(example)
        # Content-matched budget: head-keep the first N tokens ONCE
        # (keep_first_content_tokens) so the text-baseline arm and the image render
        # below both receive the IDENTICAL code string — the budget axis is "how much
        # code", not "which slice". None (the study) → background unchanged.
        background_ctx = resolve_content_budget(
            background_ctx, params, getattr(pipeline, "tokenizer", None)
        )
        ground_truth = benchmark_runner.ground_truth(example)
        language = example.get("language", "python")
        image_info = None

        try:
            if is_text_baseline:
                # Text baseline: pass raw code as text to VLM, no images.
                # Syntax/indent ablation for the fair-text arm.
                # Mirrors the text pipeline's Stage 3a-pre precedence:
                # strip_indent ELIF shuffle_indent (mutually
                # exclusive, strip wins); braces + semicolons independent.
                # Applied POST-budget (after resolve_content_budget above) so the
                # anchor and ablated arms head-keep the IDENTICAL slice first —
                # all four transforms are line-count-preserving, so the head-keep
                # commutes on the lines axis. NOTE the shuffle seed: content-hash-
                # seeded on the POST-BUDGET string, whereas the text pipeline
                # seeds on the PRE-truncation string — the same example
                # therefore shuffles DIFFERENTLY across the two paths. Within
                # this path the seed is deterministic (same content → same
                # shuffle).
                _bg_pre_ablation = background_ctx
                if params.get("strip_indent"):
                    background_ctx = _ab_strip_indent(background_ctx)
                elif params.get("shuffle_indent"):
                    _seed = int(hashlib.sha256(
                        background_ctx.encode()).hexdigest(), 16) % (2**31)
                    background_ctx = _ab_shuffle_indent(background_ctx, seed=_seed)
                if params.get("strip_braces"):
                    background_ctx = _ab_strip_braces(background_ctx)
                if params.get("strip_semicolons"):
                    background_ctx = _ab_strip_semicolons(background_ctx)
                # Optional delivered-input-token budget on the background
                # (keep_last_text_tokens) gives the fair-text arm a controllable budget
                # axis; None reproduces the prior full-context behavior byte-for-byte.
                _keep_last_tt = params.get("keep_last_text_tokens")
                _text_fit_info = {}
                _row_fit_fields = {}
                if params.get("fit_window_tokens") is not None:
                    # Symmetric per-row window fit for
                    # the text arm — content tail-keep to the row's budget, flagged
                    # as pipeline truncation below (fully_attended=False).
                    _keep_last_tt, _text_fit_info = _row_fit_budget_text(
                        pipeline, benchmark_runner, query_ctx, params
                    )
                    _bg_pre_tokens = (
                        len(pipeline.tokenizer.encode(background_ctx, add_special_tokens=False))
                        if background_ctx else 0
                    )
                text_prompt, bg_text_tokens = _text_baseline_prompt_and_tokens(
                    benchmark_runner, background_ctx, query_ctx,
                    pipeline.tokenizer, _keep_last_tt,
                )
                if _text_fit_info:
                    _text_fit_info = {
                        **_text_fit_info,
                        "fit_text_pre_tokens": _bg_pre_tokens,
                        "fit_text_kept_tokens": bg_text_tokens,
                        "fit_text_truncated": _bg_pre_tokens > bg_text_tokens,
                    }
                    _row_fit_fields = dict(_text_fit_info)
                completion, choice_logprobs = _generate_or_score_qa(
                    pipeline, images=[], text_prompt=text_prompt,
                    max_new_tokens=max_new_tokens,
                    system_prompt=benchmark_runner.system_prompt(),
                    qa_scoring=params.get("qa_scoring", "generate"),
                    ground_truth=ground_truth,
                )
                image_info = {
                    "num_pages": 0,
                    "image_size": "N/A",
                    "bg_text_tokens": bg_text_tokens,
                    "visual_tokens": 0,
                    "actual_ratio": 1.0,
                    "render_s": 0.0,  # text arm has no render step (latency axis)
                    "mode": "text_baseline",
                    # Per-row integrity fingerprint: sha256 of the post-budget,
                    # post-ablation background (recorded as _background_ctx).
                    # NOTE the basis: keep_last_text_tokens tail-truncation
                    # happens DOWNSTREAM inside _text_baseline_prompt_and_tokens
                    # and is NOT part of the fingerprint — on runs without it
                    # this IS the delivered background.
                    "bg_sha256": hashlib.sha256(
                        background_ctx.encode()).hexdigest(),
                    **_text_fit_info,
                }
                if _ablation_flags_set:
                    # Proof the ablation branch RAN and whether it changed
                    # bytes. An ablation can be a content-level no-op
                    # (strip_indent on unindented markdown); without this field
                    # an integrity check could not distinguish "no affected
                    # content" from "ablation not applied".
                    # Guarded on flags so anchor rows / prior runs are unchanged.
                    image_info["bg_ablation_changed"] = (
                        background_ctx != _bg_pre_ablation)
            elif is_flat_baseline:
                flat_code = _flatten_code(background_ctx)
                _fit_kw, _fit_info = _resolve_fit_kwargs(
                    pipeline, benchmark_runner, query_ctx, params)
                _t_render = perf_counter()
                images, image_info = pipeline.render(
                    flat_code,
                    style="plain",
                    language="text",
                    compression_ratio=compression_ratio,
                    max_visual_tokens=max_visual_tokens,
                    fixed_image_width=params.get("fixed_image_width"),
                    fixed_image_height=params.get("fixed_image_height"),
                    fixed_font_size=params.get("fixed_font_size"),
                    strip_indent=params.get("strip_indent", False),
                    shuffle_indent=params.get("shuffle_indent", False),
                    downsample_resolution=params.get("downsample_resolution"),
                    keep_last_pages=params.get("keep_last_pages"),
                    **_fit_kw,
                    record_page_metrics=params.get("record_page_metrics", False),
                    min_char_px=params.get("min_char_px"),
                    pad_page_to=params.get("pad_page_to"),
                )
                image_info["render_s"] = perf_counter() - _t_render
                image_info.update(_fit_info)
                _enforce_fit_underfit(image_info, params)
                image_info["mode"] = "flat_baseline"
            else:
                _fit_kw, _fit_info = _resolve_fit_kwargs(
                    pipeline, benchmark_runner, query_ctx, params)
                _t_render = perf_counter()
                images, image_info = pipeline.render(
                    background_ctx,
                    style=style,
                    language=language,
                    compression_ratio=compression_ratio,
                    max_visual_tokens=max_visual_tokens,
                    fixed_image_width=params.get("fixed_image_width"),
                    fixed_image_height=params.get("fixed_image_height"),
                    fixed_font_size=params.get("fixed_font_size"),
                    strip_indent=params.get("strip_indent", False),
                    shuffle_indent=params.get("shuffle_indent", False),
                    downsample_resolution=params.get("downsample_resolution"),
                    keep_last_pages=params.get("keep_last_pages"),
                    **_fit_kw,
                    record_page_metrics=params.get("record_page_metrics", False),
                    min_char_px=params.get("min_char_px"),
                    pad_page_to=params.get("pad_page_to"),
                    num_image_splits=params.get("num_image_splits"),
                    split_policy=params.get("split_policy", "lines"),
                    split_font_size=params.get("split_font_size"),
                )
                # Latency axis: time the code→image render so end-to-end
                # = render_s + generation latency_s (the latter already in window_audit).
                image_info["render_s"] = perf_counter() - _t_render
                image_info.update(_fit_info)
                _enforce_fit_underfit(image_info, params)

            if not is_text_baseline:
                # Encoder-activation OOM guard: skip over-cap rows
                # BEFORE the engine — one such request kills the EngineCore
                # and every row after it. Raises into the except path below.
                _enforce_page_cap(image_info, params.get("max_pages_per_request"))
                image_prompt = benchmark_runner.build_image_prompt(images, query_ctx)
                text_parts = [m["text"] for m in image_prompt if m.get("type") == "text"]
                text_prompt = "\n".join(text_parts)
                prompt_images = [m["image"] for m in image_prompt if m.get("type") == "image"]
                completion, choice_logprobs = _generate_or_score_qa(
                    pipeline, images=prompt_images, text_prompt=text_prompt,
                    max_new_tokens=max_new_tokens,
                    system_prompt=benchmark_runner.system_prompt(),
                    qa_scoring=params.get("qa_scoring", "generate"),
                    ground_truth=ground_truth,
                )

            all_outputs.append(completion)
            all_prompts.append(text_prompt)
            _orig = {
                **example,
                "id": example.get("id", i),
                "_ground_truth": ground_truth,
                "_background_ctx": background_ctx,
                "_query_ctx": query_ctx,
                "_image_info": image_info,
                "language": language,
            }
            if params.get("qa_scoring") == "likelihood":
                _orig["_choice_logprobs"] = choice_logprobs
            # window block (image arm): measured from the VLM's REAL input_ids
            # for the cell just generated. INSTRUMENTATION — must never break a
            # run, so the shared builder is wrapped here.
            try:
                _wab = _build_image_window_audit(
                    getattr(pipeline, "vlm", None), image_info,
                    is_text_baseline=is_text_baseline,
                    operational_max_len=params.get("vllm_max_model_len", 32768))
                if _wab is not None:
                    _annotate_fit_window_audit(_wab, params, max_new_tokens,
                                               getattr(pipeline, "vlm", None))
                    _orig["_window_audit"] = _wab
            except Exception as _wae:  # noqa: BLE001 — instrumentation must not fail a run
                logger.warning(f"window_audit (image) skipped for example {i}: {_wae}")
            original_data.append(_orig)

        except Exception as e:
            logger.warning(
                f"Error processing example {i} (ID: {example.get('id', 'N/A')}): {e}",
                exc_info=True,
            )
            # Match text-pipeline behavior (line ~422): record an
            # ERROR_GENERATING placeholder so the example survives in
            # the jsonl with a recognizable failure marker. Earlier the
            # exception path used `continue`, which silently dropped the
            # example and produced misleading lower N in scored output.
            # background_ctx, query_ctx, ground_truth, language are set
            # before the try, so they're always defined in scope here.
            all_outputs.append("ERROR_GENERATING")
            all_prompts.append("")
            # Complete-shape image_info so downstream consumers (the
            # summary at main.py:1221, generate_report.py, length-strata
            # analyzer) don't KeyError. The "error" key marks the row as
            # a failure; all numeric fields default to 0.
            preserve_fit_info = (
                (
                    params.get("keep_last_pages_fit_tokens") is not None
                    or params.get("fit_window_tokens") is not None
                    or params.get("record_page_metrics", False)
                )
                and isinstance(locals().get("image_info"), dict)
                and "pages_total" in image_info
            )
            failed_image_info = (
                {**image_info, "error": str(e)[:200]}
                if preserve_fit_info
                else {
                    "num_pages": 0,
                    "image_size": "N/A",
                    "bg_text_tokens": 0,
                    "visual_tokens": 0,
                    "actual_ratio": 0,
                    "error": str(e)[:200],
                }
            )
            _blocked_extra = _feasibility_block_fields(e)
            if _blocked_extra:
                failed_image_info = {**failed_image_info, **_blocked_extra}
            # Text-arm engine-side failures keep their per-row fit numbers
            # (computed before generate) for attribution.
            _rff = locals().get("_row_fit_fields")
            if isinstance(_rff, dict) and _rff and not preserve_fit_info:
                failed_image_info = {**_rff, **failed_image_info}
            original_data.append({
                **example,
                "id": example.get("id", i),
                "_ground_truth": ground_truth,
                "_background_ctx": background_ctx,
                "_query_ctx": query_ctx,
                "_image_info": failed_image_info,
                "language": language,
            })

    return all_outputs, all_prompts, original_data


def _run_image_pipeline_two_pass(pipeline, benchmark_runner, dataset, method, params,
                                  style, is_flat_baseline, compression_ratio,
                                  max_new_tokens, image_control, desc,
                                  max_visual_tokens=None):
    """Two-pass image pipeline for image control experiments.

    Pass 1: Render all examples and store images.
    Apply control: Swap/replace images according to image_control condition.
    Pass 2: Generate completions using (possibly modified) images.
    """
    # ---- Pass 1: Render all examples ----
    all_rendered = []
    render_failed = []  # render failures, placeholdered after pass 2
    for i, example in enumerate(tqdm(dataset, desc=f"{desc} [render]")):
        background_ctx, query_ctx = benchmark_runner.split_context(example)
        ground_truth = benchmark_runner.ground_truth(example)
        language = example.get("language", "python")

        try:
            from time import perf_counter
            if is_flat_baseline:
                flat_code = _flatten_code(background_ctx)
                _fit_kw, _fit_info = _resolve_fit_kwargs(
                    pipeline, benchmark_runner, query_ctx, params)
                _t_render = perf_counter()
                images, image_info = pipeline.render(
                    flat_code, style="plain", language="text",
                    compression_ratio=compression_ratio,
                    max_visual_tokens=max_visual_tokens,
                    fixed_image_width=params.get("fixed_image_width"),
                    fixed_image_height=params.get("fixed_image_height"),
                    fixed_font_size=params.get("fixed_font_size"),
                    strip_indent=params.get("strip_indent", False),
                    shuffle_indent=params.get("shuffle_indent", False),
                    downsample_resolution=params.get("downsample_resolution"),
                    keep_last_pages=params.get("keep_last_pages"),
                    **_fit_kw,
                    record_page_metrics=params.get("record_page_metrics", False),
                    min_char_px=params.get("min_char_px"),
                    pad_page_to=params.get("pad_page_to"),
                )
                image_info["render_s"] = perf_counter() - _t_render  # latency axis (path-independent)
                image_info.update(_fit_info)
                _enforce_fit_underfit(image_info, params)
                image_info["mode"] = "flat_baseline"
            else:
                _fit_kw, _fit_info = _resolve_fit_kwargs(
                    pipeline, benchmark_runner, query_ctx, params)
                _t_render = perf_counter()
                images, image_info = pipeline.render(
                    background_ctx, style=style, language=language,
                    compression_ratio=compression_ratio,
                    max_visual_tokens=max_visual_tokens,
                    fixed_image_width=params.get("fixed_image_width"),
                    fixed_image_height=params.get("fixed_image_height"),
                    fixed_font_size=params.get("fixed_font_size"),
                    strip_indent=params.get("strip_indent", False),
                    shuffle_indent=params.get("shuffle_indent", False),
                    downsample_resolution=params.get("downsample_resolution"),
                    keep_last_pages=params.get("keep_last_pages"),
                    **_fit_kw,
                    record_page_metrics=params.get("record_page_metrics", False),
                    min_char_px=params.get("min_char_px"),
                    pad_page_to=params.get("pad_page_to"),
                )
                image_info["render_s"] = perf_counter() - _t_render  # latency axis (path-independent)
                image_info.update(_fit_info)
                _enforce_fit_underfit(image_info, params)

            all_rendered.append({
                "index": i,
                "images": images,
                "image_info": image_info,
                "background_ctx": background_ctx,
                "query_ctx": query_ctx,
                "ground_truth": ground_truth,
                "language": language,
                "example": example,
            })
        except Exception as e:
            logger.warning(
                f"Error rendering example {i} (ID: {example.get('id', 'N/A')}): {e}",
                exc_info=True,
            )
            # Keep render failures for an ERROR_GENERATING placeholder
            # after pass 2 instead of `continue`-dropping (which shrank
            # num_examples_total with no marker). Tracked separately so the
            # failed entry doesn't disturb _apply_image_control / pass 2.
            render_failed.append({
                "index": i, "example": example, "ground_truth": ground_truth,
                "query_ctx": query_ctx, "background_ctx": background_ctx,
                "language": language, "error": str(e)[:200],
                "block_fields": _feasibility_block_fields(e),
            })
            continue

    # ---- Apply image control ----
    logger.info(f"Applying image control: {image_control} (N={len(all_rendered)})")
    _apply_image_control(all_rendered, image_control,
                         vlm_model=params.get("vlm_model", ""))

    # ---- Pass 2: Generate completions ----
    all_outputs = []
    all_prompts = []
    original_data = []

    for item in tqdm(all_rendered, desc=f"{desc} [generate]"):
        try:
            images = item["images"]
            query_ctx = item["query_ctx"]
            image_info = item["image_info"]
            example = item["example"]

            # Enforce the cap after image-control substitution so the check
            # describes the images actually delivered to the model.
            if (params.get("keep_last_pages_fit_tokens") is not None
                    or params.get("fit_window_tokens") is not None):
                _enforce_page_cap(
                    image_info,
                    params.get("max_pages_per_request"),
                )

            image_prompt = benchmark_runner.build_image_prompt(images, query_ctx)
            text_parts = [m["text"] for m in image_prompt if m.get("type") == "text"]
            text_prompt = "\n".join(text_parts)
            prompt_images = [m["image"] for m in image_prompt if m.get("type") == "image"]

            completion = pipeline.generate(
                images=prompt_images,
                text_prompt=text_prompt,
                max_new_tokens=max_new_tokens,
                # Fair-prompt parity (path-independent with single-pass): forward the
                # benchmark's system prompt to the image arm here too.
                system_prompt=benchmark_runner.system_prompt(),
            )

            all_outputs.append(completion)
            all_prompts.append(text_prompt)
            _orig = {
                **example,
                "id": example.get("id", item["index"]),
                "_ground_truth": item["ground_truth"],
                "_background_ctx": item["background_ctx"],
                "_query_ctx": query_ctx,
                "_image_info": image_info,
                "language": item["language"],
            }
            # window block (image arm, two-pass control runs). Two-pass is
            # image-only — no text baseline. INSTRUMENTATION — wrapped via the
            # SAME shared builder as the single-pass path so they cannot drift.
            try:
                _wab = _build_image_window_audit(
                    getattr(pipeline, "vlm", None), image_info, is_text_baseline=False,
                    operational_max_len=params.get("vllm_max_model_len", 32768))
                if _wab is not None:
                    _annotate_fit_window_audit(_wab, params, max_new_tokens,
                                               getattr(pipeline, "vlm", None))
                    _orig["_window_audit"] = _wab
            except Exception as _wae:  # noqa: BLE001 — instrumentation must not fail a run
                logger.warning(
                    f"window_audit (image two-pass) skipped for example "
                    f"{item['index']}: {_wae}")
            original_data.append(_orig)

        except Exception as e:
            logger.opt(exception=True).warning(
                "Error generating example {} (ID: {}): {}",
                item['index'], item['example'].get('id', 'N/A'), str(e),
            )
            # Record an ERROR_GENERATING placeholder instead of silently
            # dropping the row (two-pass used to `continue`, shrinking
            # num_examples_total with no marker).
            # All fields are read from `item` (set in pass 1), so they're
            # always in scope here regardless of where the exception fired.
            all_outputs.append("ERROR_GENERATING")
            all_prompts.append("")
            preserve_fit_info = (
                (
                    params.get("keep_last_pages_fit_tokens") is not None
                    or params.get("fit_window_tokens") is not None
                    or params.get("record_page_metrics", False)
                )
                and "pages_total" in item["image_info"]
            )
            failed_image_info = (
                {**item["image_info"], "error": str(e)[:200]}
                if preserve_fit_info
                else {
                    "num_pages": 0,
                    "image_size": "N/A",
                    "bg_text_tokens": 0,
                    "visual_tokens": 0,
                    "actual_ratio": 0,
                    "error": str(e)[:200],
                }
            )
            _blocked_extra = _feasibility_block_fields(e)
            if _blocked_extra:
                failed_image_info = {**failed_image_info, **_blocked_extra}
            # Text-arm engine-side failures keep their per-row fit numbers
            # (computed before generate) for attribution.
            _rff = locals().get("_row_fit_fields")
            if isinstance(_rff, dict) and _rff and not preserve_fit_info:
                failed_image_info = {**_rff, **failed_image_info}
            original_data.append({
                **item["example"],
                "id": item["example"].get("id", item["index"]),
                "_ground_truth": item["ground_truth"],
                "_background_ctx": item["background_ctx"],
                "_query_ctx": item["query_ctx"],
                "_image_info": failed_image_info,
                "language": item["language"],
            })
            continue

    # ---- placeholders for examples that failed to render (pass 1) ----
    # NOTE: these are appended at the END (not at their original dataset index),
    # so the returned lists are NOT in dataset order when renders fail. This is
    # intentional and safe: _score_and_save zips the three lists positionally and
    # keys each row by its own id/_ground_truth, and no analyzer indexes the jsonl
    # positionally against the dataset — order-independence is relied upon here.
    for rf in render_failed:
        all_outputs.append("ERROR_GENERATING")
        all_prompts.append("")
        original_data.append({
            **rf["example"],
            "id": rf["example"].get("id", rf["index"]),
            "_ground_truth": rf["ground_truth"],
            "_background_ctx": rf["background_ctx"],
            "_query_ctx": rf["query_ctx"],
            "_image_info": {"num_pages": 0, "image_size": "N/A", "bg_text_tokens": 0,
                            "visual_tokens": 0, "actual_ratio": 0, "error": rf["error"],
                            **rf.get("block_fields", {})},
            "language": rf["language"],
        })

    return all_outputs, all_prompts, original_data


# =============================================================================
# Hybrid pipeline flow (text compression → image rendering → VLM generation)
# =============================================================================


def evaluate_completion(
    benchmark: str = "lcc",
    pipeline: str = "text",
    model_name: str = "Qwen/Qwen2.5-Coder-7B-Instruct",
    method: str = "full",
    result_dir: str = "results/completion_baselines",
    embed_model_name: str = "microsoft/unixcoder-base",
    compression_model_name: str = "Qwen/Qwen2.5-Coder-7B-Instruct",
    dataset_path: str = "",
    dataset_split: str = "test",
    num_examples: int = 200,
    max_new_tokens: int | None = None,
    batch_size: int = 16,
    # RAG params
    rag_window_size: int = 80,
    rag_overlap: int = 40,
    rag_top_k: int = 3,
    # Function RAG params
    function_rag_language: str = "python",
    function_rag_top_k: int = 3,
    # LLMLingua params
    lingua_target_token: int = 500,
    lingua_instruction: str = "Complete the following code function given the context.",
    # LongLLMLingua params
    longlingua_chunk_size: int = 80,
    longlingua_overlap: int = 40,
    # CodeCompressor params
    code_compressor_target_token: int = 500,
    code_compressor_fine_ratio: float = 1.0,
    importance_beta: float = 0.0,
    # vLLM params (text pipeline)
    vllm_max_model_len: int = 32768,
    vllm_max_num_seqs: int | None = None,
    engine_recycle_every: int | None = None,
    tensor_parallel_size: int = 1,
    trust_remote_code: bool = True,
    gpu_memory_utilization: float = 0.9,
    enforce_eager: bool = False,
    # Chat template: wrap prompts for instruct models (required for RepoQA)
    use_chat_template: bool = False,
    # Max background context tokens (truncate head if exceeded). Prevents OOM on large models.
    max_context_tokens: int | None = None,
    # Quantization for text generation model (8bit or 4bit via bitsandbytes)
    quantize: str | None = None,
    # Dataset filter params
    filter_current_lines_max: int = 50,
    filter_background_tokens_min: int = 5000,
    filter_background_tokens_max: int | None = None,
    filter_valid_python: bool = False,
    filter_max_pages: int | None = None,
    filter_page_width: int = 1120,
    filter_page_height: int = 1120,
    filter_font_size: int = 20,
    filter_consistent_pages: bool = False,
    # Manifest-driven example selection (line-stratified instance lists).
    # Path to a scripts/sample_lcc_by_line_count.py manifest, or "" to use
    # the default line/token filter path.
    example_ids: str = "",
    # GPU routing: split compression and generation across different devices
    compression_cuda_visible_devices: str = "",
    generation_cuda_visible_devices: str = "",
    # Image pipeline params
    vlm_backend: str = "local",
    vlm_model: str = "Qwen/Qwen2.5-VL-7B-Instruct",
    # Full 40-hex HF commit id pinning the VLM snapshot ("" = unpinned
    # resolution via refs/main).
    vlm_revision: str = "",
    vlm_api_key: str = "",
    vlm_api_base_url: str = "https://openrouter.ai/api/v1",
    qa_scoring: str = "generate",
    compression_ratio: float = 1.0,
    max_visual_tokens: int | None = None,
    # Fixed layout: bypass layout optimizer, render all pages at these dimensions
    fixed_image_width: int | None = None,
    fixed_image_height: int | None = None,
    fixed_font_size: int | None = None,
    # Indentation ablation: remove leading whitespace before rendering
    strip_indent: bool = False,
    shuffle_indent: bool = False,
    # Syntax ablation (Java/C#): remove block delimiters or statement terminators
    strip_braces: bool = False,
    strip_semicolons: bool = False,
    # Resolution downsample: render at full res then shrink
    downsample_resolution: int | None = None,
    # Budget control: tail-truncate to the last N rendered pages
    # (image pipeline). Applies post-render, pre-VLM. visual_tokens in jsonl
    # reflects the truncated page set.
    keep_last_pages: int | None = None,
    # Page-fit policy: fit a retained page suffix to the real processor
    # token law and optionally record post-resize legibility metrics.
    keep_last_pages_fit_tokens: int | None = None,
    record_page_metrics: bool = False,
    min_char_px: float | None = None,
    # Per-row window fit. When set, the image
    # arm's page budget is fit_window_tokens − tok(this row's text prompt) −
    # fit_template_tokens − fit_margin_tokens, with fit_per_page_overhead
    # framing tokens charged per retained page inside the fit. None = off.
    fit_window_tokens: int | None = None,
    fit_template_tokens: int = 0,
    fit_margin_tokens: int = 0,
    fit_per_page_overhead: int = 2,
    # Pad every rendered page (incl. the auto-cropped last page) to exactly
    # WxH with the style background, applied AFTER auto-crop — freezes the
    # page geometry for area-driven encoders. CLI form: --pad_page_to 896x1344.
    # None = zero behavior change (existing paths byte-identical).
    pad_page_to: str | None = None,
    # Encoder-activation OOM guard: rows whose rendered page count
    # exceeds this cap are skipped PRE-ENGINE as ENCODER-FEASIBILITY-BLOCKED
    # (ERROR_GENERATING row, reason in image_info["error"]) instead of OOMing
    # the EngineCore fatally and losing the rest of the cell. Applies to the
    # standard (single-pass) image pipeline path. None = no cap.
    max_pages_per_request: int | None = None,
    # Split rendering (one big image vs K small images). When num_image_splits
    # is set, the image arm renders the body as exactly K single-canvas images
    # (K=1 = one big image) instead of the layout-optimised pages. split_policy:
    # "lines" (balanced contiguous) or "blocks" (cut at caller-supplied seams).
    # split_font_size fixes the font for iso-legibility; max_visual_tokens (if
    # set) becomes the iso-budget. None = unchanged (existing render path).
    num_image_splits: int | None = None,
    split_policy: str = "lines",
    split_font_size: int | None = None,
    # Budget control: tail-truncate background to the last N text
    # tokens before compression (text pipeline). Opposite of the existing
    # max_context_tokens which head-truncates.
    keep_last_text_tokens: int | None = None,
    # Content-matched budget: head-keep the first N code tokens, fed
    # IDENTICALLY to both the text and image arms (vs keep_last_text_tokens, which
    # tail-truncates the text arm only). Mutually exclusive with keep_last_text_tokens.
    keep_first_content_tokens: int | None = None,
    # LoC content budget: head-keep the first N code LINES, fed IDENTICALLY to
    # both arms. The model-agnostic unit (token counts are
    # encoder-specific). Mutually exclusive with keep_first_content_tokens and
    # keep_last_text_tokens.
    keep_first_content_lines: int | None = None,
    skip_example_ids: str | None = None,
    # Image control experiment: swap images to test if VLM reads code content
    image_control: str = "matched",
    # Gemma4 max_soft_tokens knob (5 supported settings:
    # 70, 140, 280, 560, 1120 → 64, 121, 256, 529, 1089 actual visual
    # tokens). Default None preserves prior behavior (Gemma uses its
    # processor default of 280 → 256). Applies only to Gemma models.
    gemma_max_soft_tokens: int | None = None,
    # Image-instruction variant (LCC image arm). None preserves the
    # instruction of the study (v0); "v1"/"v2"/"v3" select rewordings.
    image_instruction_variant: str | None = None,
    # Escape hatch: count image-arm text tokens with --model_name's tokenizer
    # (wrong unit when it != vlm_model) instead of the VLM's. ONLY for
    # reproducing historical runs; warns loudly.
    legacy_token_accounting: bool = False,
    # Hybrid pipeline params
    compress_method: str = "code_compressor",
    compress_budget: int = 1024,
    image_compression_ratio: float = 1.0,
):
    """
    Evaluate with a specified benchmark and context compression method.

    This is the main entry point. It composes a benchmark (lcc, longcodeqa, etc.)
    with a pipeline (text or image) to run the evaluation.

    Args:
        pipeline: "text" for text compression + vLLM, "image" for rendering + VLM.
        vlm_backend: VLM backend for image pipeline ("local" or "api").
        vlm_model: VLM model name for image pipeline.
        vlm_api_key: API key for VLM API backend.
        vlm_api_base_url: Base URL for VLM API backend.
        compression_ratio: Image compression ratio (1x, 2x, 4x, 8x).
    """

    if qa_scoring not in ("generate", "likelihood"):
        raise ValueError("qa_scoring must be 'generate' or 'likelihood'")
    if qa_scoring == "likelihood" and (benchmark != "longcodeqa" or pipeline != "image"):
        raise ValueError(
            "qa_scoring='likelihood' is supported only for LongCodeQA on the image pipeline"
        )
    if qa_scoring == "likelihood" and vlm_backend != "local":
        raise ValueError("qa_scoring='likelihood' requires the local vLLM backend")

    # Guard (fail-fast — before any benchmark/model construction):
    # keep_first_content_tokens (the content-matched budget → resolve_content_budget,
    # the seam guaranteeing both arms see the identical retained code C_b) is consumed ONLY by
    # the single-pass image pipeline. On --pipeline text (and the two-pass image path) it is
    # silently DROPPED → the text arm would get FULL un-budgeted code while the image arm gets
    # the head-kept C_b (a content advantage at every budget). Fail loud rather than launch a
    # content-mismatched run: --pipeline image (image_text vs image_*) is the only
    # content-budget-safe path.
    # The two content-matched head-keeps are mutually exclusive — fail fast at the top
    # (before any model/dataset load), not deep in the per-example resolve_content_budget loop.
    if keep_first_content_tokens is not None and keep_first_content_lines is not None:
        raise ValueError(
            "keep_first_content_tokens and keep_first_content_lines are both content-matched "
            "head-keeps — set at most one."
        )
    _content_budget = (
        keep_first_content_tokens if keep_first_content_tokens is not None else keep_first_content_lines
    )
    if _content_budget is not None and pipeline != "image":
        raise ValueError(
            f"A content-matched budget (keep_first_content_tokens={keep_first_content_tokens} / "
            f"keep_first_content_lines={keep_first_content_lines}) is honored ONLY by "
            f"--pipeline image (single-pass); it is silently dropped on --pipeline "
            f"{pipeline}, corrupting the content-matched budget. Use --pipeline image, "
            f"or wire resolve_content_budget into _run_text_pipeline first."
        )
    # The single-pass image path applies the budget (resolve_content_budget); the TWO-PASS
    # path (image_control != "matched") does NOT — it would render full un-budgeted code
    # while the analysis assumes a fixed content budget (a content-mismatched
    # comparison). Fail loud rather than silently drop it.
    if _content_budget is not None and pipeline == "image" and image_control != "matched":
        raise ValueError(
            f"A content-matched budget is silently DROPPED on the two-pass image path "
            f"(image_control={image_control!r} != 'matched'), which renders full un-budgeted "
            f"code → content-mismatched frontier. Use image_control='matched' with a content "
            f"budget, or wire resolve_content_budget into _run_image_pipeline_two_pass first."
        )

    # -- Setup benchmark --
    if benchmark not in BENCHMARKS:
        raise ValueError(f"Unknown benchmark '{benchmark}'. Available: {list(BENCHMARKS.keys())}")
    BenchmarkClass = BENCHMARKS[benchmark]

    benchmark_runner = BenchmarkClass(
        **_resolve_benchmark_init_kwargs(
            BenchmarkClass, dataset_path, image_instruction_variant
        )
    )

    # Use benchmark's recommended max_new_tokens if not explicitly provided via CLI.
    gen_config = benchmark_runner.generation_config()
    if max_new_tokens is None:
        max_new_tokens = gen_config.get("max_new_tokens", 128)
        logger.info(f"Using benchmark default max_new_tokens={max_new_tokens}")

    # -- Setup pipeline --
    if pipeline == "text":
        pipe = TextPipeline()
    elif pipeline == "image":
        pipe = ImagePipeline()
    else:
        raise ValueError(f"Unknown pipeline '{pipeline}'. Available: text, image")

    # Validate method against pipeline
    if method not in pipe.available_methods():
        raise ValueError(f"Unknown method '{method}' for {pipeline} pipeline. "
                         f"Available: {pipe.available_methods()}")

    # Validate image_control
    VALID_IMAGE_CONTROLS = ("matched", "mismatched", "blank", "noise")
    if pipeline == "image" and image_control not in VALID_IMAGE_CONTROLS:
        raise ValueError(f"Unknown image_control '{image_control}'. "
                         f"Available: {VALID_IMAGE_CONTROLS}")

    logger.info(f"Pipeline: {pipeline}, Benchmark: {benchmark}, Method: {method}")

    # CLI args take precedence over env vars for GPU routing. Explicit
    # None/""-checks + string canonicalization (`X or env` silently
    # dropped `--generation_cuda_visible_devices 0` — fire parses it as int 0).
    compression_devices = _resolve_gpu_devices(
        compression_cuda_visible_devices, "CC_COMPRESS_DEVICES")
    generation_devices = _resolve_gpu_devices(
        generation_cuda_visible_devices, "CC_GENERATE_DEVICES")

    # Collect all params into a dict
    params = {
        "model_name": model_name,
        "embed_model_name": embed_model_name,
        "compression_model_name": compression_model_name,
        "dataset_path": dataset_path,
        "dataset_split": dataset_split,
        "max_new_tokens": max_new_tokens,
        "batch_size": batch_size,
        "vllm_max_model_len": vllm_max_model_len,
        "vllm_max_num_seqs": vllm_max_num_seqs,
        "engine_recycle_every": engine_recycle_every,
        "tensor_parallel_size": tensor_parallel_size,
        "gpu_memory_utilization": gpu_memory_utilization,
        "enforce_eager": enforce_eager,
        # The prompting condition must be reconstructable from the artifacts
        # alone.
        "use_chat_template": use_chat_template,
        "trust_remote_code": trust_remote_code,
        "rag_window_size": rag_window_size,
        "rag_overlap": rag_overlap,
        "rag_top_k": rag_top_k,
        "function_rag_language": function_rag_language,
        "function_rag_top_k": function_rag_top_k,
        "lingua_target_token": lingua_target_token,
        "lingua_instruction": lingua_instruction,
        "longlingua_chunk_size": longlingua_chunk_size,
        "longlingua_overlap": longlingua_overlap,
        "code_compressor_target_token": code_compressor_target_token,
        "code_compressor_fine_ratio": code_compressor_fine_ratio,
        "importance_beta": importance_beta,
        "filter_current_lines_max": filter_current_lines_max,
        "filter_background_tokens_min": filter_background_tokens_min,
        "filter_background_tokens_max": filter_background_tokens_max,
        # Image-specific params
        "style": IMAGE_METHODS.get(method),
        "compression_ratio": compression_ratio,
        "max_visual_tokens": max_visual_tokens,
        "fixed_image_width": fixed_image_width,
        "fixed_image_height": fixed_image_height,
        "fixed_font_size": fixed_font_size,
        "strip_indent": strip_indent,
        "shuffle_indent": shuffle_indent,
        "strip_braces": strip_braces,
        "strip_semicolons": strip_semicolons,
        "max_context_tokens": max_context_tokens,
        "quantize": quantize,
        "downsample_resolution": downsample_resolution,
        "keep_last_pages": keep_last_pages,
        "keep_last_pages_fit_tokens": keep_last_pages_fit_tokens,
        "record_page_metrics": record_page_metrics,
        "min_char_px": min_char_px,
        # Parsed to a (W, H) tuple here (fail-fast on a malformed value,
        # before any model/dataset load).
        "pad_page_to": _parse_pad_page_to(pad_page_to),
        "max_pages_per_request": _validate_max_pages_per_request(max_pages_per_request),
        "num_image_splits": num_image_splits,
        "split_policy": split_policy,
        "split_font_size": split_font_size,
        "keep_last_text_tokens": keep_last_text_tokens,
        "keep_first_content_tokens": keep_first_content_tokens,
        "keep_first_content_lines": keep_first_content_lines,
        "skip_example_ids": skip_example_ids,
        "vlm_model": vlm_model,
        "vlm_revision": vlm_revision or None,
        "fit_window_tokens": fit_window_tokens,
        "fit_template_tokens": fit_template_tokens,
        "fit_margin_tokens": fit_margin_tokens,
        "fit_per_page_overhead": fit_per_page_overhead,
        "vlm_backend": vlm_backend,
        "image_control": image_control,
        "qa_scoring": qa_scoring,
        "gemma_max_soft_tokens": gemma_max_soft_tokens,
        "image_instruction_variant": image_instruction_variant,
        # Which tokenizer does the image arm's token accounting
        # (recorded in SCORES for honesty; None on the text pipeline).
        "legacy_token_accounting": legacy_token_accounting,
        "token_accounting_tokenizer": (
            (model_name if legacy_token_accounting or vlm_backend == "api"
             else vlm_model)
            if pipeline == "image" else None),
        # Hybrid-specific params
        "compress_method": compress_method,
        "compress_budget": compress_budget,
        "image_compression_ratio": image_compression_ratio,
    }

    # ---- Stage 1: Load dataset via benchmark ----
    dataset, _ = benchmark_runner.load({
        "dataset_path": dataset_path,
        "dataset_split": dataset_split,
        "num_examples": num_examples,
        "filter_current_lines_max": filter_current_lines_max,
        "filter_background_tokens_min": filter_background_tokens_min,
        "filter_background_tokens_max": filter_background_tokens_max,
        "filter_valid_python": filter_valid_python,
        "filter_max_pages": filter_max_pages,
        "filter_page_width": filter_page_width,
        "filter_page_height": filter_page_height,
        "filter_font_size": filter_font_size,
        "filter_consistent_pages": filter_consistent_pages,
        # "" (Fire default) -> None so the benchmark takes the default
        # filter path rather than treating it as an empty-manifest selection.
        "example_ids": example_ids or None,
        "model_name": model_name,
    })
    logger.info(f"Loaded {len(dataset)} examples from {dataset_path} ({dataset_split} split)")

    # Requested vs loaded counts: a loader-stage drop (length/page filter, or
    # fewer manifest hits than the manifest declares) leaves loaded <
    # requested. Recording both lets a check measure completion against what
    # was ASKED for, so scored/total cannot read 100% on a silently-shrunk
    # dataset.
    num_examples_loaded = len(dataset)
    num_examples_requested = num_examples_loaded
    if example_ids:
        try:
            _mani = json.load(open(example_ids))
            num_examples_requested = (len(_mani.get("example_indices", _mani))
                                      if isinstance(_mani, dict) else len(_mani))
        except (FileNotFoundError, json.JSONDecodeError, OSError, TypeError):
            num_examples_requested = num_examples_loaded
    elif num_examples:
        num_examples_requested = num_examples
    if num_examples_loaded < num_examples_requested:
        logger.warning(
            f"LOADER DROP: requested {num_examples_requested} but loaded "
            f"{num_examples_loaded} ({num_examples_requested - num_examples_loaded} "
            f"row(s) lost at load); the completion gate uses the requested N.")

    # ---- Stages 2-6: Pipeline-specific processing ----
    if pipeline == "text":
        all_outputs, all_prompts, original_data = _run_text_pipeline(
            pipe, benchmark_runner, dataset, method, params,
            model_name=model_name,
            compression_model_name=compression_model_name,
            embed_model_name=embed_model_name,
            max_new_tokens=max_new_tokens,
            batch_size=batch_size,
            vllm_max_model_len=vllm_max_model_len,
            gpu_memory_utilization=gpu_memory_utilization,
            tensor_parallel_size=tensor_parallel_size,
            trust_remote_code=trust_remote_code,
            enforce_eager=enforce_eager,
            use_chat_template=use_chat_template,
            compression_devices=compression_devices,
            generation_devices=generation_devices,
        )
    elif pipeline == "image":
        all_outputs, all_prompts, original_data = _run_image_pipeline(
            pipe, benchmark_runner, dataset, method, params,
            vlm_model=vlm_model,
            vlm_revision=vlm_revision or None,
            vlm_backend=vlm_backend,
            vlm_api_key=vlm_api_key,
            vlm_api_base_url=vlm_api_base_url,
            compression_ratio=compression_ratio,
            max_new_tokens=max_new_tokens,
            model_name=model_name,
            image_control=image_control,
            max_visual_tokens=max_visual_tokens,
            gemma_max_soft_tokens=gemma_max_soft_tokens,
            generation_devices=generation_devices,
        )

    if not all_outputs:
        logger.error(f"No outputs produced for method {method}. Nothing to score.")
        pipe.cleanup()
        return

    # ---- Stage 7: Score and save results ----
    # For the image pipeline, compute average image stats for SCORES.json
    # (incl. the warmup-aware latency aggregate — see _image_extra_scores).
    extra_scores = _image_extra_scores(original_data) if pipeline == "image" else None

    # Use vlm_model for output paths when running the image pipeline
    output_model_name = vlm_model if pipeline == "image" else model_name

    _score_and_save(
        benchmark_runner, benchmark, method, output_model_name,
        all_outputs, all_prompts, original_data,
        result_dir, params, pipeline_type=pipeline,
        extra_scores_fields=extra_scores,
        num_examples_requested=num_examples_requested,
        num_examples_loaded=num_examples_loaded,
    )

    # Clean up
    pipe.cleanup()
    logger.info("Evaluation complete.")


def main():
    """CLI entry point."""
    fire.Fire(evaluate_completion)


if __name__ == "__main__":
    main()
