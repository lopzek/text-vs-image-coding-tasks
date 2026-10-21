"""Image compression pipeline.

Renders background code as images and generates completions via a
Vision-Language Model (VLM). Supports local models (Qwen2.5-VL) and
API backends (OpenAI-compatible).

This module provides the rendering, VLM initialization, and generation
logic as a reusable pipeline class that can be composed with any benchmark
via main.py's --pipeline image flag.
"""

import gc
import math
from numbers import Real

import torch
from codecompress.code_renderer import (
    DEFAULT_FONT_SIZE,
    DEFAULT_LINE_HEIGHT,
    char_px_effective,
    char_px_post_resize,
    count_image_tokens,
    generate_lint_errors,
    lint_tool_version,
    page_visual_tokens,
    render_and_compress,
    render_code_tail_fit,
    render_split,
)
from loguru import logger
from transformers import AutoTokenizer
from codecompress.vlm_inference import (
    VLLM_AVAILABLE,
    ApiVLM,
    LocalVLM,
    VllmVLM,
    assemble_vlm_messages,
)


# Valid rendering styles
VALID_STYLES = ["plain", "highlight", "bold", "ide", "ide_lint", "dark", "dark_highlight"]

# Special baselines (not rendering styles)
TEXT_BASELINE = "text"    # raw text to VLM, no image
FLAT_BASELINE = "flat"    # code flattened into one line, rendered as image

# Available image methods: one per style + baselines
IMAGE_METHODS = {f"image_{s}": s for s in VALID_STYLES}
IMAGE_METHODS["image_text"] = TEXT_BASELINE
IMAGE_METHODS["image_flat"] = FLAT_BASELINE


def _validate_min_char_px(min_char_px):
    """Reject invalid legibility thresholds before rendering or model work."""
    if min_char_px is not None and (
        isinstance(min_char_px, bool)
        or not isinstance(min_char_px, Real)
        or not math.isfinite(min_char_px)
        or min_char_px <= 0
    ):
        raise ValueError(
            "min_char_px must be a positive finite real number, got "
            f"{min_char_px!r}"
        )


def _flatten_code(code: str) -> str:
    """Flatten code into a single line with visible escape characters.

    Escapes existing backslashes first so that literal \\n in source code
    (e.g., in regexes or string escapes) is distinguishable from real newlines.
    """
    return (
        code
        .replace("\\", "\\\\")
        .replace("\t", "\\t")
        .replace("\r\n", "\\n")
        .replace("\r", "\\n")
        .replace("\n", "\\n")
    )


class ImagePipeline:
    """Image rendering → VLM generation pipeline.

    Renders code as images using code_renderer.py, sends to a VLM via
    vlm_inference.py, and returns generated text. Can be composed with
    any benchmark.
    """

    def __init__(self):
        self.vlm = None
        self.tokenizer = None
        self.resources = {}  # mirrors TextPipeline; holds {"tokenizer": ...} for main.py

    # -- Rendering --

    def available_methods(self):
        """Return list of available image method names."""
        return list(IMAGE_METHODS.keys())

    def init_tokenizer(self, tokenizer_name="Qwen/Qwen2.5-Coder-7B-Instruct",
                       revision=None):
        """Initialize a text tokenizer for compression ratio calculation.

        We need to know how many text tokens the background has to compute
        the target visual token budget at a given compression ratio.

        revision: optional full 40-hex HF commit id; the tokenizer then loads
        from the pinned cached snapshot (same resolver as the engines) instead
        of refs/main. None = unpinned.
        """
        # Some model repos (e.g. moonshotai/Kimi-VL*) ship a custom
        # tokenizer class that only loads with trust_remote_code=True.
        # Keep this list in sync with LocalVLM._TRUST_REMOTE_CODE_SUBSTRINGS.
        # The substring check runs on the HUB ID, never on a snapshot path.
        _TRUST_REMOTE_CODE_SUBSTRINGS = ("moonshotai/kimi-vl",)
        needs_trust = any(s in tokenizer_name.lower()
                          for s in _TRUST_REMOTE_CODE_SUBSTRINGS)
        from codecompress.vlm_inference import resolve_model_snapshot

        load_from = resolve_model_snapshot(tokenizer_name, revision)
        self.tokenizer = AutoTokenizer.from_pretrained(
            load_from, trust_remote_code=needs_trust,
        )
        self.tokenizer_revision = revision or None
        self.resources["tokenizer"] = self.tokenizer
        logger.info(f"Text tokenizer {tokenizer_name} (revision={revision}) "
                    "initialized (for ratio calculation).")

    def render(self, background_ctx, style="highlight", language="python",
               compression_ratio=1.0, max_visual_tokens=None,
               fixed_image_width=None, fixed_image_height=None,
               fixed_font_size=None, strip_indent=False,
               shuffle_indent=False, downsample_resolution=None,
               keep_last_pages=None, keep_last_pages_fit_tokens=None,
               record_page_metrics=False, min_char_px=None, pad_page_to=None,
               num_image_splits=None, split_policy="lines",
               split_font_size=None, seam_offsets=None, per_page_overhead=0):
        """Render background code as image(s) at the target compression ratio.

        Args:
            background_ctx: The background code to render.
            style: Rendering style (plain, highlight, bold, ide, ide_lint).
            language: Programming language for syntax highlighting.
            compression_ratio: Target compression ratio (1x, 2x, 4x, 8x).
            max_visual_tokens: Fixed visual token budget. Overrides ratio-based calc.
            fixed_image_width: Fixed page width (bypasses layout optimizer).
            fixed_image_height: Fixed page height (bypasses layout optimizer).
            fixed_font_size: Fixed font size (bypasses layout optimizer).
            strip_indent: If True, remove leading whitespace before rendering.
            keep_last_pages: If set, pass only the last N rendered pages to
                the VLM (tail-truncate). visual_tokens in image_info reflects
                the truncated page set.
            keep_last_pages_fit_tokens: If set, retain the newest contiguous
                pages whose real processor-token cost fits this budget. The
                last page is always kept and marked underfit when necessary.
            record_page_metrics: Emit retained-page physical post-processor
                and information-bottleneck char-px measurements. Intended for
                the fixed-layout iso-legibility path.
            min_char_px: Optional strict lower threshold used only for
                recording how many retained pages' effective char-px fall
                below it.
            per_page_overhead: Framing tokens charged per retained page inside
                the keep_last_pages_fit_tokens decision (2 on both studied
                encoders). 0 = no framing charged.
            pad_page_to: If set to (W, H), pad every rendered page (incl. the
                auto-cropped last page) to exactly WxH with the style
                background, top-left anchored — applied after auto-crop
                (freezes the page geometry for area-driven encoders).
                Ignored (with a warning) on the split-rendering path.
                Default None = unchanged behavior.

        Returns:
            (images, image_info) — list of PIL Images and a dict of stats.
        """
        if style not in VALID_STYLES:
            raise ValueError(f"Unknown style '{style}'. Available: {VALID_STYLES}")
        if keep_last_pages is not None and keep_last_pages < 1:
            raise ValueError(
                f"keep_last_pages must be >= 1, got {keep_last_pages}"
            )
        if keep_last_pages is not None and keep_last_pages_fit_tokens is not None:
            raise ValueError(
                "keep_last_pages and keep_last_pages_fit_tokens are mutually "
                "exclusive tail-retention policies"
            )
        if keep_last_pages_fit_tokens is not None and (
            not isinstance(keep_last_pages_fit_tokens, int)
            or isinstance(keep_last_pages_fit_tokens, bool)
            or keep_last_pages_fit_tokens <= 0
        ):
            raise ValueError(
                "keep_last_pages_fit_tokens must be a positive integer, got "
                f"{keep_last_pages_fit_tokens!r}"
            )
        if keep_last_pages_fit_tokens is not None and num_image_splits is not None:
            raise ValueError(
                "keep_last_pages_fit_tokens is incompatible with "
                "num_image_splits; fit operates on emergent rendered pages"
            )
        if not isinstance(record_page_metrics, bool):
            raise TypeError(
                f"record_page_metrics must be bool, got {record_page_metrics!r}"
            )
        _validate_min_char_px(min_char_px)
        if (
            record_page_metrics
            and num_image_splits is None
            and not all(
                value is not None
                for value in (
                    fixed_image_width,
                    fixed_image_height,
                    fixed_font_size,
                )
            )
        ):
            raise ValueError(
                "record_page_metrics requires the complete fixed-layout "
                "triple (fixed_image_width, fixed_image_height, "
                "fixed_font_size), unless num_image_splits is used; an "
                "optimized layout has no fixed source character size"
            )

        model_name = getattr(self, "vlm_model_name", "")
        # Parse this once, then share the exact value across the page fit,
        # char-px, split accounting, and provenance.
        gemma_max_soft_tokens_effective = (
            getattr(self, "gemma_max_soft_tokens", None) or 280
        )
        policy_enabled = keep_last_pages_fit_tokens is not None or record_page_metrics
        page_policy_enabled = policy_enabled or min_char_px is not None
        page_raster_scale_y = None

        # Count text tokens in background for ratio calculation
        bg_text_tokens = 0
        if self.tokenizer and background_ctx:
            bg_text_tokens = len(self.tokenizer.encode(background_ctx, add_special_tokens=False))

        # Generate lint errors for ide_lint style
        lint_errors = None
        if style == "ide_lint" and background_ctx:
            lint_errors = generate_lint_errors(background_ctx)

        # Render background code as image(s).
        if num_image_splits is not None:
            # Split-rendering path: one big image (K=1) vs K small images, as a
            # controlled variable. max_visual_tokens (when set) becomes the
            # iso-budget; otherwise the font is fixed and the cost floats
            # (iso-legibility). Cost/latency are recorded per example downstream
            # via the window_audit block (measured pad-tokens + latency_s).
            _ignored = [n for n, v in (
                ("fixed_font_size", fixed_font_size),
                ("fixed_image_width", fixed_image_width),
                ("fixed_image_height", fixed_image_height),
                ("downsample_resolution", downsample_resolution),
                ("pad_page_to", pad_page_to),
            ) if v is not None]
            if _ignored:
                logger.warning(
                    f"render(): {_ignored} are ignored on the split path — use "
                    f"split_font_size and max_visual_tokens (iso-budget) instead."
                )
            split_result = render_split(
                background_ctx,
                n_images=num_image_splits,
                style=style,
                language=language,
                split_policy=split_policy,
                seam_offsets=seam_offsets,
                font_size=(split_font_size if split_font_size is not None
                           else DEFAULT_FONT_SIZE),
                total_token_budget=max_visual_tokens,
                model_name=model_name,
                gemma_max_soft_tokens=gemma_max_soft_tokens_effective,
                lint_errors=lint_errors,
                strip_indent=strip_indent,
                shuffle_indent=shuffle_indent,
                return_raster_scales=record_page_metrics,
            )
            if record_page_metrics:
                images, page_raster_scale_y = split_result
            else:
                images = split_result
            policy_info = None
        elif (
            policy_enabled
            and fixed_image_width is not None
            and fixed_image_height is not None
            and fixed_font_size is not None
        ):
            images, policy_info = render_code_tail_fit(
                background_ctx,
                token_budget=(
                    keep_last_pages_fit_tokens
                    if keep_last_pages_fit_tokens is not None
                    else 2**63 - 1
                ),
                model_name=model_name,
                style=style,
                language=language,
                width=fixed_image_width,
                height=fixed_image_height,
                font_size=fixed_font_size,
                gemma_max_soft_tokens=gemma_max_soft_tokens_effective,
                lint_errors=lint_errors,
                strip_indent=strip_indent,
                shuffle_indent=shuffle_indent,
                downsample_resolution=downsample_resolution,
                pad_page_to=pad_page_to,
                per_page_overhead=per_page_overhead,
            )
            page_raster_scale_y = policy_info.pop("_page_raster_scale_y")
        else:
            images = render_and_compress(
                code=background_ctx,
                style=style,
                language=language,
                text_token_count=bg_text_tokens,
                compression_ratio=compression_ratio,
                lint_errors=lint_errors,
                max_visual_tokens=max_visual_tokens,
                fixed_image_width=fixed_image_width,
                fixed_image_height=fixed_image_height,
                fixed_font_size=fixed_font_size,
                strip_indent=strip_indent,
                shuffle_indent=shuffle_indent,
                downsample_resolution=downsample_resolution,
                pad_page_to=pad_page_to,
            )
            policy_info = None

            # The large-object optimization is defined for fixed-layout
            # iso-legibility. Keep the new policy mechanically correct for
            # other layouts by selecting from the already-rendered pages.
            if keep_last_pages_fit_tokens is not None:
                all_images = images
                all_costs = [
                    page_visual_tokens(
                        image.width,
                        image.height,
                        model_name,
                        gemma_max_soft_tokens=gemma_max_soft_tokens_effective,
                    )
                    for image in all_images
                ]
                kept_start = len(all_images) - 1
                # kept_cost = page tokens; the fit decision adds per-page framing.
                kept_cost = all_costs[-1]
                underfit = kept_cost + per_page_overhead > keep_last_pages_fit_tokens
                if not underfit:
                    for index in range(len(all_images) - 2, -1, -1):
                        _pages = len(all_images) - index
                        if (kept_cost + all_costs[index] + _pages * per_page_overhead
                                > keep_last_pages_fit_tokens):
                            break
                        kept_start = index
                        kept_cost += all_costs[index]
                images = all_images[kept_start:]
                policy_info = {
                    "pages_total": len(all_images),
                    "pages_kept": len(images),
                    "pages_dropped_front": kept_start,
                    "budget_underfit": underfit,
                    "visual_tokens_total": sum(all_costs),
                    "visual_tokens_total_estimator": "measured",
                    "visual_tokens_kept": kept_cost,
                    "pages_rasterized": len(all_images),
                }
                if per_page_overhead:
                    policy_info["per_page_overhead"] = per_page_overhead

        if page_raster_scale_y is None:
            if num_image_splits is not None or downsample_resolution is None:
                page_raster_scale_y = [1.0] * len(images)
            else:
                # Optimized-layout fallback: fixed-layout policy rows take the
                # exact renderer hand-off above.  For legacy non-fixed layouts,
                # preserve the renderer's documented nominal scale.
                page_raster_scale_y = [
                    downsample_resolution / 1120
                ] * len(images)

        images_before_count_keep = images
        total_pages_rendered = (
            policy_info["pages_total"]
            if policy_info is not None
            else len(images_before_count_keep)
        )
        if keep_last_pages is not None and keep_last_pages < len(images):
            images = images[-keep_last_pages:]
            page_raster_scale_y = page_raster_scale_y[-keep_last_pages:]

        if policy_enabled and policy_info is None:
            all_exact_costs = [
                page_visual_tokens(
                    image.width,
                    image.height,
                    model_name,
                    gemma_max_soft_tokens=gemma_max_soft_tokens_effective,
                )
                for image in images_before_count_keep
            ]
            kept_exact_costs = all_exact_costs[
                len(images_before_count_keep) - len(images):
            ]
            policy_info = {
                "pages_total": len(images_before_count_keep),
                "pages_kept": len(images),
                "pages_dropped_front": (
                    len(images_before_count_keep) - len(images)
                ),
                "budget_underfit": False,
                "visual_tokens_total": sum(all_exact_costs),
                "visual_tokens_total_estimator": "measured",
                "visual_tokens_kept": sum(kept_exact_costs),
                "pages_rasterized": len(images_before_count_keep),
            }
        elif (
            policy_enabled
            and policy_info is not None
            and len(images) < len(images_before_count_keep)
        ):
            # record_page_metrics may legally accompany the older fixed-count
            # tail policy. Keep the full-render provenance and make the new
            # explicit truncation fields describe that retained suffix.
            kept_exact_costs = [
                page_visual_tokens(
                    image.width,
                    image.height,
                    model_name,
                    gemma_max_soft_tokens=gemma_max_soft_tokens_effective,
                )
                for image in images
            ]
            policy_info["pages_kept"] = len(images)
            policy_info["pages_dropped_front"] = (
                policy_info["pages_total"] - len(images)
            )
            policy_info["visual_tokens_kept"] = sum(kept_exact_costs)

        # This existing field intentionally stays on its historical accounting
        # law on every path.  Fit decisions and the exact totals live only in
        # visual_tokens_kept / visual_tokens_total.
        visual_tokens = count_image_tokens(
            images,
            model_name=model_name,
            gemma_max_soft_tokens=gemma_max_soft_tokens_effective,
        )

        image_info = {
            "num_pages": len(images),
            "image_size": f"{images[0].width}x{images[0].height}" if images else "N/A",
            "bg_text_tokens": bg_text_tokens,
            "visual_tokens": visual_tokens,
            "actual_ratio": round(bg_text_tokens / visual_tokens, 2) if visual_tokens > 0 else 0,
        }
        if page_policy_enabled:
            image_info["visual_tokens_law"] = "legacy_count_image_tokens"
        if style == "ide_lint":
            # Per-example lint stats so a run-level check can catch silent
            # degeneration to ide (generate_lint_errors returns [] on ANY
            # failure, e.g. ruff missing).
            # None-guarded (empty background -> lint_errors is None -> 0);
            # non-ide_lint runs get NO new keys (byte-identical).
            _lint = lint_errors or []
            image_info["lint_error_count"] = len(_lint)
            image_info["lint_severity_counts"] = {
                sev: sum(1 for _, s, _ in _lint if s == sev) for sev in ("error", "warning")
            }
            # Provenance: WHICH linter produced the marks (cached lookup;
            # None = no ruff resolved, i.e. the arm degenerated to ide).
            image_info["lint_tool_version"] = lint_tool_version()
        if keep_last_pages is not None:
            # Record pre-truncation page count so downstream analysis knows
            # how much context was dropped.
            image_info["total_pages_rendered"] = total_pages_rendered
            image_info["keep_last_pages"] = keep_last_pages

        if policy_enabled:
            image_info.update(policy_info)
            # Keep the old detector key alive: main.py and historical analysis
            # both use it, while pages_dropped_front is the new explicit key.
            image_info["total_pages_rendered"] = policy_info["pages_total"]
            if "gemma" in model_name.lower():
                image_info["gemma_max_soft_tokens_effective"] = (
                    gemma_max_soft_tokens_effective
                )

        if record_page_metrics:
            effective_font_size = (
                split_font_size
                if num_image_splits is not None and split_font_size is not None
                else (
                    DEFAULT_FONT_SIZE
                    if num_image_splits is not None
                    else (
                        fixed_font_size
                        if fixed_font_size is not None
                        else DEFAULT_FONT_SIZE
                    )
                )
            )
            page_char_px = [
                raster_scale_y * char_px_post_resize(
                    effective_font_size,
                    image.width,
                    image.height,
                    model_name,
                    gemma_max_soft_tokens=gemma_max_soft_tokens_effective,
                )
                for image, raster_scale_y in zip(
                    images,
                    page_raster_scale_y,
                    strict=True,
                )
            ]
            page_char_px_effective = [
                min(
                    float(int(effective_font_size * DEFAULT_LINE_HEIGHT)),
                    raster_scale_y
                    * char_px_effective(
                        effective_font_size,
                        image.width,
                        image.height,
                        model_name,
                        gemma_max_soft_tokens=gemma_max_soft_tokens_effective,
                    ),
                )
                for image, raster_scale_y in zip(
                    images,
                    page_raster_scale_y,
                    strict=True,
                )
            ]
            image_info["page_char_px"] = page_char_px
            image_info["page_char_px_effective"] = page_char_px_effective
            image_info["min_char_px_effective"] = min(page_char_px_effective)
            if min_char_px is not None:
                image_info["char_px_below_threshold"] = sum(
                    value < min_char_px for value in page_char_px_effective
                )

        if num_image_splits is not None:
            # Record the split design so each cell's (n, policy, font) and the
            # per-image sizes are recoverable (per-image visibility matters for
            # the blocks policy, where chunks can be unequal).
            image_info["num_image_splits"] = num_image_splits        # requested K
            image_info["actual_num_images"] = len(images)            # produced (≤ K if capped)
            image_info["bg_line_count"] = len(background_ctx.splitlines())
            image_info["split_policy"] = split_policy
            image_info["split_font_size"] = (
                split_font_size if split_font_size is not None else DEFAULT_FONT_SIZE
            )
            image_info["image_sizes"] = [f"{im.width}x{im.height}" for im in images]
            # Per-image token cost — explicit for heterogeneous 'blocks' chunks
            # and for resolution-invariant encoders where WxH alone is not enough.
            image_info["image_token_sizes"] = [
                count_image_tokens(
                    [im], model_name=model_name,
                    gemma_max_soft_tokens=gemma_max_soft_tokens_effective,
                )
                for im in images
            ]

        return images, image_info

    # -- Generation --

    def init_generation(self, vlm_backend="local", vlm_model="Qwen/Qwen2.5-VL-7B-Instruct",
                        vlm_api_key=None, vlm_api_base_url="https://openrouter.ai/api/v1",
                        gemma_max_soft_tokens=None, tensor_parallel_size=1,
                        gpu_memory_utilization=0.9, vllm_max_model_len=32768,
                        enforce_eager=False, trust_remote_code=True, max_num_seqs=None,
                        engine_recycle_every=None,
                        vlm_revision=None):
        """Initialize the VLM backend.

        Args:
            vlm_backend: "local" for an on-box engine, "api" for OpenAI-compatible.
                For "local" the engine is chosen the SAME way the text arm chooses
                (text.py keys on VLLM_AVAILABLE): vLLM when importable (i.e. running
                under .venv-vllm), else transformers LocalVLM. The interpreter
                therefore selects the engine — no separate flag.
            vlm_model: Model name or path.
            vlm_revision: Optional full 40-hex HF commit id. Resolved offline
                to the cached snapshot and fails loudly on a weightless
                snapshot; None = unpinned.
            vlm_api_key: API key (for api backend).
            vlm_api_base_url: API base URL (for api backend).
            gemma_max_soft_tokens: For Gemma4 only — selects the
                visual-token budget per image. Supported: 70, 140,
                280 (default), 560, 1120.
                Ignored for non-Gemma models. NOT supported under vLLM: VllmVLM
                HARD-FAILS if this is set (vLLM applies the model default, so the
                requested budget would not be honored). Sweep it on transformers.
            tensor_parallel_size, gpu_memory_utilization, vllm_max_model_len,
            enforce_eager, trust_remote_code: vLLM engine params; used only by the
                VllmVLM branch (ignored by LocalVLM/ApiVLM).
        """
        self.vlm_model_name = vlm_model
        self.vlm_revision = vlm_revision
        self.gemma_max_soft_tokens = gemma_max_soft_tokens
        # revision kwarg only when pinned — unpinned constructor call shapes stay
        # unchanged.
        _rev_kw = {"revision": vlm_revision} if vlm_revision else {}
        if vlm_backend == "local":
            engine = "vLLM" if VLLM_AVAILABLE else "transformers"
            logger.info(f"Initializing VLM: {vlm_model} (backend: local/{engine})")
            if VLLM_AVAILABLE:
                self.vlm = VllmVLM(
                    model_name=vlm_model,
                    tensor_parallel_size=tensor_parallel_size,
                    gpu_memory_utilization=gpu_memory_utilization,
                    max_model_len=vllm_max_model_len,
                    gemma_max_soft_tokens=gemma_max_soft_tokens,
                    enforce_eager=enforce_eager,
                    trust_remote_code=trust_remote_code,
                    max_num_seqs=max_num_seqs,
                    engine_recycle_every=engine_recycle_every,
                    **_rev_kw,
                )
            else:
                self.vlm = LocalVLM(
                    model_name=vlm_model,
                    gemma_max_soft_tokens=gemma_max_soft_tokens,
                    **_rev_kw,
                )
        elif vlm_backend == "api":
            logger.info(f"Initializing VLM: {vlm_model} (backend: api)")
            self.vlm = ApiVLM(
                model=vlm_model,
                api_key=vlm_api_key or None,
                base_url=vlm_api_base_url,
            )
        else:
            raise ValueError(f"Unknown vlm_backend '{vlm_backend}'. Use 'local' or 'api'.")
        logger.info("VLM initialized.")

    def generate(self, images, text_prompt, max_new_tokens=128, temperature=0.0,
                 system_prompt=None):
        """Generate a completion from images + text prompt.

        Args:
            images: List of PIL Image objects.
            text_prompt: Text portion of the prompt.
            max_new_tokens: Maximum tokens to generate.
            temperature: Sampling temperature (0 = greedy).
            system_prompt: optional fair system prompt forwarded to the VLM
                backend (None keeps legacy behavior — see assemble_vlm_messages).

        Returns:
            str — generated text.
        """
        if self.vlm is None:
            raise RuntimeError("VLM not initialized. Call init_generation() first.")
        return self.vlm.generate(
            images=images,
            text_prompt=text_prompt,
            max_tokens=max_new_tokens,
            temperature=temperature,
            system_prompt=system_prompt,
        )

    def score_choices(self, images, text_prompt, choices, system_prompt=None):
        """Score MCQ continuations through the same assembled vLLM chat path."""
        if self.vlm is None:
            raise RuntimeError("VLM not initialized. Call init_generation() first.")
        content = [{"type": "image_pil", "image_pil": img} for img in images]
        content.append({"type": "text", "text": text_prompt})
        messages = assemble_vlm_messages(content, system_prompt, self.vlm_model_name)
        return self.vlm.score_choices(messages, choices)

    # -- Cleanup --

    def free_generation(self):
        """Delete the VLM and free GPU memory."""
        if self.vlm is not None:
            self.vlm.cleanup()
            self.vlm = None
        gc.collect()
        torch.cuda.empty_cache()
        logger.info("VLM freed from GPU memory.")

    def cleanup(self):
        """Free all resources."""
        self.free_generation()
        self.tokenizer = None
        self.resources = {}
