#!/usr/bin/env python3
"""Page-legibility calibration harness (the character-height calibration of the study).

The three subcommands deliberately separate the CPU-only, deterministic page
instrument from the later GPU calibration:

* ``gen`` selects real code, extracts one unambiguous ground-truth page,
  renders the resolution ladder, records both encoders' post-resize geometry,
  and proves that the ladder reaches a RapidOCR-negative region.
* ``transcribe`` runs either the CPU negative control or one local VLM.
* ``analyze`` applies the pre-registered discrete 95%-of-native knee rule.

No network access is permitted: the models and datasets come from the local
Hugging Face cache.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import re
import statistics
import sys
import warnings
from collections import defaultdict
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


def force_offline_environment():
    """Force every Hugging Face layer onto local caches only."""
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_DATASETS_OFFLINE"] = "1"


# Apply before any Hugging Face module is imported lazily below.  Assignment
# (not setdefault) prevents a caller's pre-existing ``*=0`` from weakening the
# offline-only contract.
force_offline_environment()

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

QWEN_MODEL = "Qwen/Qwen3.8-27B"
QWEN_REVISION = "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0"
GEMMA_MODEL = "google/gemma-4-26B-A4B-it"
# The Gemma repository's refs/main later moved to a weightless metadata-only
# snapshot (4d7ae498); pin the snapshot with the weights explicitly.
GEMMA_REVISION = "20da991ab4afab98e8f910c4a2e8f4fbefc404ad"
ENCODER_MODELS = (QWEN_MODEL, GEMMA_MODEL)
_ENCODER_REVISIONS = {QWEN_MODEL: QWEN_REVISION, GEMMA_MODEL: GEMMA_REVISION}
GEMMA_SUPPORTED_MAX_SOFT_TOKENS = (70, 140, 280, 560, 1120)
DEFAULT_GEMMA_MAX_SOFT_TOKENS = 280
TASKS = ("lcc", "repobench", "lca", "longcodeqa")
TRANSCRIPTION_PROMPT = (
    "Transcribe the code in this image exactly. Output only the code."
)
SCHEMA_VERSION = 1
DEFAULT_WIDTH = 1120
DEFAULT_HEIGHT = 1120
DEFAULT_FONT_SIZE = 20
PLATEAU_ACCURACY_FLOOR = 0.60
FENCE_POLICY = "first-fenced-block"
DEFAULT_CENSUS_DIR = REPO_ROOT / "census"   # output of scripts/loc_census.py; without it the pages are drawn from the datasets directly

_FONT_PATHS = (
    "DejaVuSansMono",
    "DejaVuSansMono.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
)


def _encoder_revision(model_name):
    return _ENCODER_REVISIONS.get(model_name)


def resolve_local_model_path(model_name, snapshot_download_fn=None):
    """Resolve a pinned encoder revision to its local snapshot directory.

    Both studied encoders are pinned; any other model name passes through
    unresolved.
    """
    revision = _encoder_revision(model_name)
    if revision is None:
        return model_name
    if snapshot_download_fn is None:
        from huggingface_hub import snapshot_download

        snapshot_download_fn = snapshot_download
    return str(
        snapshot_download_fn(
            repo_id=model_name,
            revision=revision,
            local_files_only=True,
        )
    )


# ---------------------------------------------------------------------------
# Stable JSON and character scoring (same rule as ocr_resolution_probe.py).
# ---------------------------------------------------------------------------


def _json_line(value):
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ) + "\n"


def write_manifest(rows, path):
    """Write rows with stable key order and no run-time provenance."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(_json_line(row))


def _write_json(value, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(
            json.dumps(
                value,
                ensure_ascii=False,
                sort_keys=True,
                indent=2,
            )
        )
        handle.write("\n")


def _read_jsonl(path):
    rows = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(
                    "invalid JSONL at {}:{}: {}".format(path, line_number, exc)
                ) from exc
    return rows


def char_edit_distance(s1, s2):
    """Character-level Levenshtein distance."""
    if len(s1) < len(s2):
        s1, s2 = s2, s1
    if not s2:
        return len(s1)
    previous = list(range(len(s2) + 1))
    for i, c1 in enumerate(s1):
        current = [i + 1]
        for j, c2 in enumerate(s2):
            insertion = current[j] + 1
            deletion = previous[j + 1] + 1
            substitution = previous[j] + (c1 != c2)
            current.append(min(insertion, deletion, substitution))
        previous = current
    return previous[-1]


def char_accuracy(predicted, ground_truth):
    """Return bounded ``1 - edit_distance / max_length`` accuracy."""
    if not ground_truth:
        return 0.0
    distance = char_edit_distance(predicted, ground_truth)
    return max(
        0.0,
        1.0 - distance / max(len(predicted), len(ground_truth)),
    )


def strip_markdown_fences(output):
    """Keep the first fenced block of a transcription (diagnostic only)."""
    if "```" not in output:
        return output
    match = re.search(
        r"```[A-Za-z0-9_+-]*[ \t]*\n(.*?)(?:\n[ \t]*```|$)",
        output,
        re.DOTALL,
    )
    if match:
        return match.group(1)
    return "\n".join(
        line
        for line in output.splitlines()
        if not line.strip().startswith("```")
    )


def _transcription_diagnostics(prediction, ground_truth, backend):
    fence_applicable = backend == "vlm"
    fence_present = fence_applicable and "```" in prediction
    stripped = (
        strip_markdown_fences(prediction)
        if fence_applicable
        else prediction
    )
    return {
        "markdown_fence_present": fence_present,
        "transcription_fence_stripped": stripped,
        "char_accuracy_fence_stripped": char_accuracy(
            stripped, ground_truth
        ),
    }


# ---------------------------------------------------------------------------
# Deterministic four-task sampling and single-page excerpt selection.
# ---------------------------------------------------------------------------


def _stable_row_key(row):
    return (
        str(row.get("task", "")),
        str(row.get("lang", "")),
        str(row.get("example_id", "")),
    )


def _task_quotas(pages):
    if pages <= 0:
        raise ValueError("pages must be positive")
    base, remainder = divmod(pages, len(TASKS))
    return {
        task: base + (1 if index < remainder else 0)
        for index, task in enumerate(TASKS)
    }


def seeded_start_index(line_count, seed):
    if line_count <= 0:
        raise ValueError("line_count must be positive")
    return random.Random(seed).randrange(line_count)


def _load_monospace_font(font_size):
    for path in _FONT_PATHS:
        try:
            return ImageFont.truetype(path, font_size)
        except (IOError, OSError):
            continue
    return ImageFont.load_default()


def line_fits_page(line, width, font_size, margin_ratio=0.01):
    """Use Pillow's public text measurement API to reject wrapped lines."""
    if width <= 0 or font_size <= 0:
        raise ValueError("width and font_size must be positive")
    font = _load_monospace_font(font_size)
    canvas = Image.new("RGB", (1, 1), "white")
    draw = ImageDraw.Draw(canvas)
    usable_width = width - 2 * int(width * margin_ratio)
    return float(draw.textlength(line, font=font)) <= float(usable_width)


def _stable_seed(*parts):
    payload = "\x1f".join(str(part) for part in parts).encode(
        "utf-8", "surrogatepass"
    )
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def select_single_page_excerpt(
    background,
    seed,
    width,
    height,
    font_size,
    style="plain",
    language="python",
    render_fn=None,
):
    """Return the Lead-approved longest full single-page source prefix.

    A seeded line supplies the first candidate start.  Any line that exceeds
    the public Pillow width check invalidates the whole current window, so the
    next candidate starts one line later.  A result is accepted only when the
    next line would render a second page; reaching the source tail while still
    on one page is rejected as a sparse tail page.
    """
    if render_fn is None:
        from codecompress.code_renderer import render_code

        render_fn = render_code

    lines = background.splitlines()
    if not lines:
        raise ValueError("background contains no source lines")

    initial = seeded_start_index(len(lines), seed)
    starts = list(range(initial, len(lines))) + list(range(0, initial))
    for start in starts:
        last_good = None
        for end in range(start + 1, len(lines) + 1):
            line = lines[end - 1]
            if not line_fits_page(line, width=width, font_size=font_size):
                # The entire window is disqualified; do not keep a prefix that
                # happens to end immediately before an overwide line.
                last_good = None
                break
            candidate = "\n".join(lines[start:end])
            images = render_fn(
                candidate,
                style=style,
                language=language,
                width=width,
                height=height,
                font_size=font_size,
            )
            if len(images) == 1:
                last_good = candidate
                continue
            if len(images) > 1:
                if last_good is not None:
                    return last_good
                break
            raise RuntimeError("render_code returned an empty page list")
        # A one-page prefix that merely hit EOF is intentionally not accepted:
        # it is not proven maximal at the page boundary and may be a short tail.
    raise ValueError(
        "no full single-page excerpt without wrapped lines was found"
    )


# ---------------------------------------------------------------------------
# Raster ladder and true-processor post-resize character geometry.
# ---------------------------------------------------------------------------


def resize_for_ladder(page, ratio):
    if not (0.0 < ratio <= 1.0):
        raise ValueError("ladder ratios must be in (0, 1]")
    width, height = page.size
    target = (
        max(1, int(round(width * ratio))),
        max(1, int(round(height * ratio))),
    )
    return page.resize(target, Image.Resampling.LANCZOS)


def char_px_effective(rendered_line_px, post_line_px):
    """Return the information-bottleneck character-height metric."""
    values = (float(rendered_line_px), float(post_line_px))
    if any(not math.isfinite(value) or value < 0.0 for value in values):
        raise ValueError("character pixel heights must be finite and non-negative")
    return min(values)


class ProcessorCharPxMetric:
    """Measure vertical post-processor character pixels without model weights.

    Without the renderer's shared helper, this fallback invokes each real local
    AutoProcessor and derives the actual output height from its emitted grid.
    The result is cached by ``(model, width, height)``.  This is measurement,
    not a documentation-derived model constant.
    """

    def __init__(
        self, gemma_max_soft_tokens=DEFAULT_GEMMA_MAX_SOFT_TOKENS
    ):
        if gemma_max_soft_tokens not in GEMMA_SUPPORTED_MAX_SOFT_TOKENS:
            raise ValueError(
                "unsupported gemma_max_soft_tokens={}; allowed: {}".format(
                    gemma_max_soft_tokens,
                    GEMMA_SUPPORTED_MAX_SOFT_TOKENS,
                )
            )
        self.gemma_max_soft_tokens = int(gemma_max_soft_tokens)
        self._processors = {}
        self._height_cache = {}
        self._shared_default_line_height = None
        try:
            from codecompress import code_renderer
        except ImportError:
            self._shared = None
        else:
            self._shared_default_line_height = float(
                code_renderer.DEFAULT_LINE_HEIGHT
            )
            self._shared = getattr(
                code_renderer, "char_px_post_resize", None
            )

    @property
    def source(self):
        if self._shared is not None:
            return "code_renderer.char_px_post_resize"
        return "inline_true_processor_geometry_TODO_unify_WO009"

    def _load_processor(self, model_name):
        if model_name not in self._processors:
            from transformers import AutoProcessor

            kwargs = {"local_files_only": True}
            if model_name == QWEN_MODEL:
                kwargs["revision"] = QWEN_REVISION
            processor = AutoProcessor.from_pretrained(
                model_name, **kwargs
            )
            if model_name == GEMMA_MODEL:
                # Mirror LocalVLM's existing, processor-backed mst plumbing.
                # This changes the real emitted patch grid; it is not a
                # documentation-derived approximation.
                processor.image_seq_length = self.gemma_max_soft_tokens
                image_processor = processor.image_processor
                for attribute in ("max_soft_tokens", "image_seq_length"):
                    if hasattr(image_processor, attribute):
                        setattr(
                            image_processor,
                            attribute,
                            self.gemma_max_soft_tokens,
                        )
            self._processors[model_name] = processor
        return self._processors[model_name]

    def processor_height(self, model_name, width, height):
        key = (
            model_name,
            int(width),
            int(height),
            self.gemma_max_soft_tokens,
        )
        if key in self._height_cache:
            return self._height_cache[key]
        if width <= 0 or height <= 0:
            raise ValueError("page dimensions must be positive")

        processor = self._load_processor(model_name)
        blank = Image.new("RGB", (int(width), int(height)), "white")
        batch = processor.image_processor(images=[blank], return_tensors="pt")
        image_processor = processor.image_processor
        patch_size = int(getattr(image_processor, "patch_size", 16))

        if model_name == QWEN_MODEL:
            grid = batch["image_grid_thw"][0]
            output_height = int(grid[1].item()) * patch_size
        elif model_name == GEMMA_MODEL:
            positions = batch["image_position_ids"][0]
            valid_y = positions[:, 1]
            valid_y = valid_y[valid_y >= 0]
            if int(valid_y.numel()) == 0:
                raise RuntimeError("Gemma processor emitted no valid patch rows")
            output_height = (int(valid_y.max().item()) + 1) * patch_size
        else:
            raise ValueError("unsupported encoder model: {}".format(model_name))

        self._height_cache[key] = output_height
        return output_height

    def _inline_vertical_scale(self, model_name, width, height):
        return self.processor_height(model_name, width, height) / float(height)

    def _shared_vertical_scale(self, model_name, width, height):
        if self._shared is None:
            raise RuntimeError("shared post-resize helper is unavailable")
        reference_font_size = DEFAULT_FONT_SIZE
        reference_line_px = int(
            reference_font_size * self._shared_default_line_height
        )
        if reference_line_px <= 0:
            raise RuntimeError("shared renderer line pitch must be positive")
        post_line_px = float(
            self._shared(
                reference_font_size,
                width,
                height,
                model_name,
                gemma_max_soft_tokens=self.gemma_max_soft_tokens,
            )
        )
        return post_line_px / float(reference_line_px)

    def __call__(self, effective_line_px, width, height, model_name):
        if self._shared is not None:
            vertical_scale = self._shared_vertical_scale(
                model_name, width, height
            )
        else:
            vertical_scale = self._inline_vertical_scale(
                model_name, width, height
            )
        return float(effective_line_px) * vertical_scale

    def probe(self):
        """Record mandatory upscale/extreme-aspect true-processor evidence."""
        sizes = ((112, 112), (112, 4480), (2240, 600))
        rows = []
        for model_name in ENCODER_MODELS:
            for width, height in sizes:
                output_height = self.processor_height(
                    model_name, width, height
                )
                inline_scale = output_height / float(height)
                shared_scale = None
                helper_inline_agree = None
                if self._shared is not None:
                    shared_scale = self._shared_vertical_scale(
                        model_name, width, height
                    )
                    helper_inline_agree = math.isclose(
                        shared_scale,
                        inline_scale,
                        rel_tol=1e-9,
                        abs_tol=1e-9,
                    )
                    if not helper_inline_agree:
                        raise RuntimeError(
                            "post-resize geometry drift: model={} revision={} "
                            "mst={} input={}x{} helper_scale={} "
                            "processor_scale={}".format(
                                model_name,
                                _encoder_revision(model_name),
                                self.gemma_max_soft_tokens,
                                width,
                                height,
                                shared_scale,
                                inline_scale,
                            )
                        )
                rows.append(
                    {
                        "model": model_name,
                        "model_revision": _encoder_revision(model_name),
                        "gemma_max_soft_tokens": self.gemma_max_soft_tokens,
                        "input_size": [width, height],
                        "processor_output_height": output_height,
                        "vertical_scale": inline_scale,
                        "shared_vertical_scale": shared_scale,
                        "helper_inline_agree": helper_inline_agree,
                    }
                )
        return rows


def build_page_variants(
    excerpt,
    styles,
    ladder,
    width,
    height,
    font_size,
    language="python",
    encoders=ENCODER_MODELS,
    render_fn=None,
    metric_fn=None,
):
    """Render one exact-GT page across styles and native-relative ratios."""
    from codecompress.code_renderer import DEFAULT_LINE_HEIGHT

    if render_fn is None:
        from codecompress.code_renderer import render_code

        render_fn = render_code
    if metric_fn is None:
        metric_fn = ProcessorCharPxMetric()

    records = []
    for style in styles:
        pages = render_fn(
            excerpt,
            style=style,
            language=language,
            width=width,
            height=height,
            font_size=font_size,
        )
        if len(pages) != 1:
            raise RuntimeError(
                "approved excerpt rendered as {} pages for style {}".format(
                    len(pages), style
                )
            )
        native = pages[0]
        native_width, native_height = native.size
        for ratio in ladder:
            image = native.copy() if ratio == 1.0 else resize_for_ladder(native, ratio)
            image_width, image_height = image.size
            actual_vertical_ratio = image_height / float(native_height)
            native_line_px = int(font_size * DEFAULT_LINE_HEIGHT)
            # ``char_px_post`` follows the actual integer raster geometry.
            # The information input, however, is requested
            # r * h_rendered; integer dimension rounding must not move a rung.
            actual_raster_line_px = native_line_px * actual_vertical_ratio
            rendered_line_px = native_line_px * float(ratio)
            post_by_encoder = {}
            effective_by_encoder = {}
            for model_name in encoders:
                post_line_px = float(
                    metric_fn(
                        actual_raster_line_px,
                        image_width,
                        image_height,
                        model_name,
                    )
                )
                post_by_encoder[model_name] = post_line_px
                effective_by_encoder[model_name] = char_px_effective(
                    rendered_line_px, post_line_px
                )
            records.append(
                {
                    "ground_truth": excerpt,
                    "style": style,
                    "ladder_ratio": float(ratio),
                    "native_size": [native_width, native_height],
                    "image_size": [image_width, image_height],
                    "font_size": int(font_size),
                    "char_px_rendered": rendered_line_px,
                    "char_px_post_by_encoder": post_by_encoder,
                    "char_px_effective_by_encoder": effective_by_encoder,
                    "_image": image,
                }
            )
    return records


# ---------------------------------------------------------------------------
# Frozen knee analysis and CPU negative-control reachability.
# ---------------------------------------------------------------------------


def _percentile(sorted_values, probability):
    if not sorted_values:
        raise ValueError("cannot take percentile of an empty sample")
    index = int(probability * len(sorted_values))
    index = min(max(index, 0), len(sorted_values) - 1)
    return float(sorted_values[index])


def _bootstrap_ci(values, iterations, seed):
    if iterations <= 0:
        raise ValueError("bootstrap_iterations must be positive")
    rng = random.Random(seed)
    means = []
    for _ in range(iterations):
        sample = [rng.choice(values) for _ in range(len(values))]
        means.append(statistics.mean(sample))
    means.sort()
    return [
        _percentile(means, 0.025),
        _percentile(means, 0.975),
    ]


def _distinct_float_count(values):
    distinct = []
    for value in sorted(float(item) for item in values):
        if not distinct or not math.isclose(
            value, distinct[-1], rel_tol=1e-12, abs_tol=1e-12
        ):
            distinct.append(value)
    return len(distinct)


def analyze_records(records, bootstrap_iterations=2000, seed=42):
    grouped = defaultdict(list)
    for row in records:
        grouped[(str(row["encoder"]), str(row["style"]))].append(row)

    groups = []
    for (encoder, style), rows in sorted(grouped.items()):
        by_ratio = defaultdict(list)
        for row in rows:
            by_ratio[float(row["ladder_ratio"])].append(row)

        points = []
        for ratio, rung_rows in by_ratio.items():
            accuracies = [float(row["char_accuracy"]) for row in rung_rows]
            stripped_accuracies = [
                float(
                    row.get(
                        "char_accuracy_fence_stripped",
                        row["char_accuracy"],
                    )
                )
                for row in rung_rows
            ]
            fence_applicable = any(
                row.get("backend") == "vlm" for row in rung_rows
            )
            fenced_n = (
                sum(
                    bool(row.get("markdown_fence_present", False))
                    for row in rung_rows
                )
                if fence_applicable
                else 0
            )
            post_values = [
                float(row["char_px_post"]) for row in rung_rows
            ]
            effective_values = [
                float(row["char_px_effective"]) for row in rung_rows
            ]
            mean_accuracy = statistics.mean(accuracies)
            ci_seed = _stable_seed(seed, encoder, style, ratio)
            stripped_ci_seed = _stable_seed(
                seed, encoder, style, ratio, "fence-stripped"
            )
            points.append(
                {
                    "ladder_ratio": ratio,
                    "n": len(rung_rows),
                    "mean_accuracy": mean_accuracy,
                    "mean_char_px_post": statistics.mean(post_values),
                    "mean_char_px_effective": statistics.mean(
                        effective_values
                    ),
                    "bootstrap_ci_95": _bootstrap_ci(
                        accuracies, bootstrap_iterations, ci_seed
                    ),
                    "mean_accuracy_fence_stripped": statistics.mean(
                        stripped_accuracies
                    ),
                    "bootstrap_ci_95_fence_stripped": _bootstrap_ci(
                        stripped_accuracies,
                        bootstrap_iterations,
                        stripped_ci_seed,
                    ),
                    "fenced_n": fenced_n,
                    "fence_rate": fenced_n / float(len(rung_rows)),
                }
            )
        points.sort(
            key=lambda point: (
                point["mean_char_px_effective"], point["ladder_ratio"]
            )
        )

        native_points = [
            point
            for point in points
            if math.isclose(point["ladder_ratio"], 1.0, abs_tol=1e-12)
        ]
        if len(native_points) != 1:
            raise ValueError(
                "group ({}, {}) needs exactly one r=1.0 rung".format(
                    encoder, style
                )
            )
        plateau = native_points[0]["mean_accuracy"]
        native_attainable = native_points[0]["mean_char_px_effective"]
        target = 0.95 * plateau
        plateau_floor_passed = plateau >= PLATEAU_ACCURACY_FLOOR
        distinct_axis_values = _distinct_float_count(
            point["mean_char_px_effective"] for point in points
        )
        axis_degenerate = distinct_axis_values < 3
        knee = None
        monotonicity_violation = None
        if not axis_degenerate:
            monotonicity_violation = any(
                current["mean_accuracy"] + 1e-12
                < previous["mean_accuracy"]
                for previous, current in zip(points, points[1:])
            )
            if plateau_floor_passed:
                qualifying = [
                    point
                    for point in points
                    if point["mean_accuracy"] >= target
                ]
                if not qualifying:
                    raise RuntimeError(
                        "native plateau did not qualify against itself"
                    )
                knee = qualifying[0]["mean_char_px_effective"]
        if not plateau_floor_passed:
            group_status = "CALIBRATION-FAIL"
        elif axis_degenerate:
            group_status = "AXIS-DEGENERATE"
        else:
            group_status = "OK"
        threshold_unclipped = None if knee is None else knee * 1.25
        threshold_clipped = (
            None
            if threshold_unclipped is None
            else (
                threshold_unclipped > native_attainable
                and not math.isclose(
                    threshold_unclipped,
                    native_attainable,
                    rel_tol=1e-12,
                    abs_tol=1e-12,
                )
            )
        )
        threshold_clipped_value = (
            None
            if threshold_unclipped is None
            else min(threshold_unclipped, native_attainable)
        )
        group_fence_applicable = any(
            row.get("backend") == "vlm" for row in rows
        )
        group_fenced_n = (
            sum(
                bool(row.get("markdown_fence_present", False))
                for row in rows
            )
            if group_fence_applicable
            else 0
        )
        max_mean = max(point["mean_accuracy"] for point in points)
        groups.append(
            {
                "encoder": encoder,
                "style": style,
                "analysis_status": group_status,
                "axis_metric": "char_px_effective",
                "axis_distinct_values": distinct_axis_values,
                "axis_degenerate": axis_degenerate,
                "plateau_ladder_ratio": 1.0,
                "plateau_accuracy": plateau,
                "plateau_floor_accuracy": PLATEAU_ACCURACY_FLOOR,
                "plateau_floor_passed": plateau_floor_passed,
                "knee_target_accuracy": target,
                "knee_char_px_effective": knee,
                "threshold_margin_multiplier": 1.25,
                "threshold_proposal_char_px_effective": threshold_unclipped,
                "threshold_unclipped_char_px_effective": threshold_unclipped,
                "native_attainable_char_px_effective": native_attainable,
                "threshold_clipped_char_px_effective": threshold_clipped_value,
                "threshold_clipped": threshold_clipped,
                "monotonicity_violation": monotonicity_violation,
                "native_not_maximum": plateau + 1e-12 < max_mean,
                "fence_diagnostics": {
                    "applicable": group_fence_applicable,
                    "n": len(rows),
                    "fenced_n": group_fenced_n,
                    "fence_rate": group_fenced_n / float(len(rows)),
                    "policy": FENCE_POLICY,
                },
                "points": points,
            }
        )

    if any(
        group["analysis_status"] == "CALIBRATION-FAIL"
        for group in groups
    ):
        overall_status = "CALIBRATION-FAIL"
    elif any(group["axis_degenerate"] for group in groups):
        overall_status = "AXIS-DEGENERATE"
    else:
        overall_status = "OK"
    return {
        "schema_version": SCHEMA_VERSION,
        "analysis_status": overall_status,
        "analysis_policy": {
            "plateau": "mean accuracy at native ladder_ratio=1.0",
            "plateau_floor": (
                "CALIBRATION-FAIL when native mean accuracy < 0.60"
            ),
            "axis": "char_px_effective=min(r*h_rendered,char_px_post)",
            "knee": (
                "first ascending mean char_px_effective rung "
                ">= 0.95 * plateau"
            ),
            "axis_guard": "AXIS-DEGENERATE when distinct effective x < 3",
            "threshold": "raw knee*1.25 with native-attainable clip diagnostic",
            "fence_diagnostic": (
                "raw accuracy remains primary; first-block stripped "
                "accuracy is sensitivity-only"
            ),
            "empty_ground_truth_accuracy": 0.0,
            "interpolation": False,
            "bootstrap_iterations": int(bootstrap_iterations),
            "bootstrap_seed": int(seed),
        },
        "groups": groups,
    }


def ensure_negative_control(
    ladder,
    evaluate_lowest,
    threshold=0.50,
    extension_factor=0.50,
    max_extensions=8,
):
    """Extend the smallest ratio until mean OCR accuracy is strictly <50%."""
    if not ladder:
        raise ValueError("ladder must not be empty")
    if not (0.0 < extension_factor < 1.0):
        raise ValueError("extension_factor must be in (0, 1)")
    effective = sorted({float(value) for value in ladder}, reverse=True)
    auto_extended = False
    extensions = 0
    while True:
        lowest = min(effective)
        accuracy = float(evaluate_lowest(lowest))
        if accuracy < threshold:
            return {
                "ladder": effective,
                "auto_extended": auto_extended,
                "extensions": extensions,
                "passed": True,
                "threshold": float(threshold),
                "lowest_ratio": lowest,
                "lowest_accuracy": accuracy,
            }
        if extensions >= max_extensions:
            return {
                "ladder": effective,
                "auto_extended": auto_extended,
                "extensions": extensions,
                "passed": False,
                "threshold": float(threshold),
                "lowest_ratio": lowest,
                "lowest_accuracy": accuracy,
            }
        new_ratio = lowest * extension_factor
        warnings.warn(
            "negative control accuracy {:.4f} at ratio {} did not pass; "
            "extend ladder to {}".format(accuracy, lowest, new_ratio),
            RuntimeWarning,
        )
        effective.append(new_ratio)
        effective.sort(reverse=True)
        auto_extended = True
        extensions += 1


# ---------------------------------------------------------------------------
# Offline real-background adapters.
# ---------------------------------------------------------------------------


class BackgroundResolver:
    def __init__(self):
        self._datasets = {}

    def _load_dataset(self, task, lang=None):
        key = (task, lang or "")
        if key in self._datasets:
            return self._datasets[key]
        import datasets

        if task == "lcc":
            language = lang or "python"
            dataset = datasets.load_dataset(
                "microsoft/LCC_{}".format(language), split="test"
            )
        elif task == "repobench":
            dataset = datasets.load_dataset(
                "tianyang/repobench_python_v1.1",
                split="cross_file_first",
            )
        elif task == "lca":
            dataset = datasets.load_dataset(
                "JetBrains-Research/lca-module-summarization",
                split="test",
            )
        elif task == "longcodeqa":
            from huggingface_hub import hf_hub_download

            local_zip = hf_hub_download(
                "Steefano/LCB",
                "LongCodeQA.zip",
                repo_type="dataset",
                local_files_only=True,
            )
            dataset = datasets.load_dataset(
                "json",
                data_files="zip://LQA/*.json::{}".format(local_zip),
                split="train",
            )
        else:
            raise ValueError("unknown task: {}".format(task))
        self._datasets[key] = dataset
        return dataset

    def candidate_rows(self, task):
        rows = []
        if task == "lcc":
            for language in ("python", "java", "csharp"):
                dataset = self._load_dataset(task, language)
                rows.extend(
                    {
                        "task": task,
                        "lang": language,
                        "example_id": index,
                    }
                    for index in range(len(dataset))
                )
            return rows
        dataset = self._load_dataset(task)
        return [
            {"task": task, "example_id": index}
            for index in range(len(dataset))
        ]

    def background(self, row):
        task = row["task"]
        language = row.get("lang")
        dataset = self._load_dataset(task, language)
        try:
            example_id = int(row["example_id"])
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "example_id must be a raw split index for task {}: {!r}".format(
                    task, row.get("example_id")
                )
            ) from exc
        example = dataset[example_id]
        if task == "lcc":
            background = example["context"]
            language = language or "python"
        elif task == "repobench":
            from codecompress.benchmarks.repobench import (
                normalize_blanklines,
                repobench_cross_file_prompt,
            )

            background = normalize_blanklines(
                repobench_cross_file_prompt(example, "python")
            )
            language = "python"
        elif task == "lca":
            background = example.get("relevant_code_context", "") or ""
            language = "python"
        elif task == "longcodeqa":
            background = example["repo_text"]
            language = "python"
        else:
            raise ValueError("unknown task: {}".format(task))
        return str(background), str(language)


def _load_census_candidates(census_dir):
    census_dir = Path(census_dir)
    paths = {
        task: census_dir / "census_rows_{}.jsonl".format(task)
        for task in TASKS
    }
    if not all(path.exists() for path in paths.values()):
        return None
    rows = []
    for task in TASKS:
        for row in _read_jsonl(paths[task]):
            normalized = dict(row)
            normalized["task"] = task
            rows.append(normalized)
    return rows


def _page_sources(pages, seed, width, height, font_size, census_dir):
    resolver = BackgroundResolver()
    census_rows = _load_census_candidates(census_dir)
    quotas = _task_quotas(pages)
    source_mode = (
        "census_rows"
        if census_rows is not None
        else "temporary_loader_replay"
    )
    if census_rows is None:
        grouped = {
            task: (
                resolver.candidate_rows(task) if quotas[task] > 0 else []
            )
            for task in TASKS
        }
    else:
        grouped = {task: [] for task in TASKS}
        for row in census_rows:
            if row.get("task") in grouped:
                grouped[row["task"]].append(row)

    selected = []
    for task in TASKS:
        target = quotas[task]
        if target == 0:
            continue
        candidates = sorted(grouped[task], key=_stable_row_key)
        random.Random("{}:{}:page-source".format(seed, task)).shuffle(
            candidates
        )
        accepted = 0
        failures = []
        for row in candidates:
            try:
                background, language = resolver.background(row)
                page_seed = _stable_seed(
                    seed,
                    task,
                    row.get("lang", ""),
                    row.get("example_id"),
                )
                excerpt = select_single_page_excerpt(
                    background,
                    seed=page_seed,
                    width=width,
                    height=height,
                    font_size=font_size,
                    style="plain",
                    language=language,
                )
            except (ValueError, IndexError, KeyError) as exc:
                if len(failures) < 5:
                    failures.append(str(exc))
                continue
            selected.append(
                {
                    "task": task,
                    "example_id": row.get("example_id"),
                    "lang": row.get("lang", language),
                    "language": language,
                    "excerpt": excerpt,
                    "excerpt_sha256": hashlib.sha256(
                        excerpt.encode("utf-8", "surrogatepass")
                    ).hexdigest(),
                    "start_seed": page_seed,
                    "source_mode": source_mode,
                }
            )
            accepted += 1
            if accepted >= target:
                break
        if accepted < target:
            raise RuntimeError(
                "task {} produced only {}/{} valid pages; sample failures: {}".format(
                    task, accepted, target, failures
                )
            )

    # Interleave task groups in their accepted order.
    by_task = {task: [] for task in TASKS}
    for row in selected:
        by_task[row["task"]].append(row)
    interleaved = []
    for index in range(max(quotas.values())):
        for task in TASKS:
            if index < len(by_task[task]):
                interleaved.append(by_task[task][index])
    return interleaved, source_mode


# ---------------------------------------------------------------------------
# Generation, transcription, and CLI orchestration.
# ---------------------------------------------------------------------------


def _ratio_slug(ratio):
    return format(float(ratio), ".12g").replace(".", "p")


def _encoder_label(model_name):
    if model_name == QWEN_MODEL:
        return "qwen"
    if model_name == GEMMA_MODEL:
        return "gemma"
    raise ValueError(
        "model must be one of the studied encoders: {}, {}".format(
            QWEN_MODEL, GEMMA_MODEL
        )
    )


def _rapidocr_text(image, engine):
    import numpy as np

    result, _elapsed = engine(np.asarray(image))
    if result is None:
        return ""
    return "\n".join(str(item[1]) for item in result)


def _make_rapidocr():
    try:
        from rapidocr_onnxruntime import RapidOCR
    except ImportError as exc:
        raise RuntimeError(
            "RapidOCR is required for gen reachability and the CPU backend; "
            "install the dev group (rapidocr-onnxruntime==1.4.4)."
        ) from exc
    return RapidOCR()


def _render_and_save_ratio(
    sources,
    styles,
    ratio,
    out_dir,
    width,
    height,
    font_size,
    metric,
):
    rows = []
    images_dir = Path(out_dir) / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    for page_index, source in enumerate(sources):
        variants = build_page_variants(
            source["excerpt"],
            styles=styles,
            ladder=(ratio,),
            width=width,
            height=height,
            font_size=font_size,
            language=source["language"],
            metric_fn=metric,
        )
        page_id = "p{:04d}".format(page_index)
        for variant in variants:
            image = variant.pop("_image")
            filename = "{}__{}__r{}.png".format(
                page_id, variant["style"], _ratio_slug(ratio)
            )
            relative_path = Path("images") / filename
            image.save(Path(out_dir) / relative_path, format="PNG")
            row = dict(variant)
            row.update(
                {
                    "schema_version": SCHEMA_VERSION,
                    "page_id": page_id,
                    "task": source["task"],
                    "example_id": source["example_id"],
                    "language": source["language"],
                    "excerpt_sha256": source["excerpt_sha256"],
                    "start_seed": source["start_seed"],
                    "source_mode": source["source_mode"],
                    "encoder_revisions": {
                        model_name: _encoder_revision(model_name)
                        for model_name in ENCODER_MODELS
                    },
                    "gemma_max_soft_tokens_effective": (
                        metric.gemma_max_soft_tokens
                    ),
                    "image_path": relative_path.as_posix(),
                }
            )
            rows.append(row)
    return rows


def generate_pages(args):
    styles = tuple(
        style.strip() for style in args.styles.split(",") if style.strip()
    )
    if not styles:
        raise ValueError("styles must not be empty")
    allowed_styles = {"plain", "dark_highlight"}
    if not set(styles).issubset(allowed_styles):
        raise ValueError(
            "styles are limited to plain,dark_highlight; got {}".format(
                styles
            )
        )
    ladder = [
        float(value) for value in args.ladder.split(",") if value.strip()
    ]
    if not ladder or any(value <= 0.0 or value > 1.0 for value in ladder):
        raise ValueError("ladder ratios must be in (0, 1]")
    if not any(math.isclose(value, 1.0, abs_tol=1e-12) for value in ladder):
        raise ValueError("ladder must include native ratio 1.0")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    sources, source_mode = _page_sources(
        args.pages,
        args.seed,
        args.width,
        args.height,
        args.font_size,
        args.census_dir,
    )
    metric = ProcessorCharPxMetric(args.gemma_max_soft_tokens)
    processor_probes = metric.probe()
    rows_by_ratio = {}

    def ensure_ratio(ratio):
        key = float(ratio)
        if key not in rows_by_ratio:
            rows_by_ratio[key] = _render_and_save_ratio(
                sources,
                styles,
                key,
                out_dir,
                args.width,
                args.height,
                args.font_size,
                metric,
            )
        return rows_by_ratio[key]

    for ratio in sorted(set(ladder), reverse=True):
        ensure_ratio(ratio)

    ocr = _make_rapidocr()

    def lowest_accuracy(ratio):
        rows = ensure_ratio(ratio)
        accuracies = []
        for row in rows:
            with Image.open(out_dir / row["image_path"]) as image:
                prediction = _rapidocr_text(image.convert("RGB"), ocr)
            accuracies.append(char_accuracy(prediction, row["ground_truth"]))
        return statistics.mean(accuracies)

    negative = ensure_negative_control(
        ladder,
        lowest_accuracy,
        threshold=0.50,
        extension_factor=0.50,
        max_extensions=args.max_negative_extensions,
    )
    if not negative["passed"]:
        raise RuntimeError(
            "negative control remained readable at ratio {}: accuracy={:.6f}".format(
                negative["lowest_ratio"], negative["lowest_accuracy"]
            )
        )

    all_rows = []
    for ratio in negative["ladder"]:
        all_rows.extend(ensure_ratio(ratio))
    all_rows.sort(
        key=lambda row: (
            row["page_id"],
            row["style"],
            -float(row["ladder_ratio"]),
        )
    )
    write_manifest(all_rows, out_dir / "manifest.jsonl")
    source_rows = [
        {key: value for key, value in row.items() if key != "excerpt"}
        for row in sources
    ]
    write_manifest(source_rows, out_dir / "page_sources.jsonl")
    report = {
        "schema_version": SCHEMA_VERSION,
        "pages_requested": int(args.pages),
        "pages_generated": len(sources),
        "styles": list(styles),
        "initial_ladder": sorted(set(ladder), reverse=True),
        "effective_ladder": negative["ladder"],
        "geometry": {
            "requested_width": int(args.width),
            "requested_height": int(args.height),
            "font_size": int(args.font_size),
        },
        "source_mode": source_mode,
        "post_resize_metric_source": metric.source,
        "encoder_revisions": {
            model_name: _encoder_revision(model_name)
            for model_name in ENCODER_MODELS
        },
        "gemma_max_soft_tokens_effective": metric.gemma_max_soft_tokens,
        "processor_geometry_probes": processor_probes,
        "negative_control": negative,
        "manifest_rows": len(all_rows),
    }
    _write_json(report, out_dir / "gen_report.json")
    print(
        "calibration gen: pages={} rows={} source={} lowest_ratio={} "
        "lowest_accuracy={:.6f} negative_control=PASS".format(
            len(sources),
            len(all_rows),
            source_mode,
            negative["lowest_ratio"],
            negative["lowest_accuracy"],
        )
    )
    return report


def _make_vlm(
    model_name,
    engine_name,
    tensor_parallel_size,
    gemma_max_soft_tokens=DEFAULT_GEMMA_MAX_SOFT_TOKENS,
    enforce_eager=False,
    max_num_seqs=None,
):
    if max_num_seqs is not None and max_num_seqs <= 0:
        raise ValueError("max_num_seqs must be a positive integer")
    from codecompress.vlm_inference import LocalVLM, VLLM_AVAILABLE, VllmVLM

    resolved_model_path = resolve_local_model_path(model_name)
    if (
        model_name == GEMMA_MODEL
        and gemma_max_soft_tokens != DEFAULT_GEMMA_MAX_SOFT_TOKENS
    ):
        if engine_name == "vllm":
            raise RuntimeError(
                "Gemma non-default max_soft_tokens is unsupported by vLLM; "
                "use --vlm-engine transformers"
            )
        if engine_name == "auto":
            engine_name = "transformers"
    if engine_name == "vllm" or (
        engine_name == "auto" and VLLM_AVAILABLE
    ):
        vlm = VllmVLM(
            resolved_model_path,
            tensor_parallel_size=tensor_parallel_size,
            max_model_len=32768,
            enforce_eager=bool(enforce_eager),
            max_num_seqs=max_num_seqs,
        )
        vlm.calibration_engine = "vllm"
        return vlm
    kwargs = {}
    if model_name == GEMMA_MODEL:
        kwargs["gemma_max_soft_tokens"] = gemma_max_soft_tokens
    vlm = LocalVLM(resolved_model_path, **kwargs)
    vlm.calibration_engine = "transformers"
    return vlm


def single_gemma_mst(rows):
    values = {
        int(
            row.get(
                "gemma_max_soft_tokens_effective",
                DEFAULT_GEMMA_MAX_SOFT_TOKENS,
            )
        )
        for row in rows
    }
    if len(values) > 1:
        raise ValueError(
            "input mixes gemma_max_soft_tokens values: {}".format(
                sorted(values)
            )
        )
    value = next(iter(values)) if values else DEFAULT_GEMMA_MAX_SOFT_TOKENS
    if value not in GEMMA_SUPPORTED_MAX_SOFT_TOKENS:
        raise ValueError(
            "unsupported gemma_max_soft_tokens={}".format(value)
        )
    return value


def _validate_analysis_provenance(rows):
    grouped = defaultdict(list)
    for row in rows:
        grouped[str(row.get("encoder_model"))].append(row)
    for group_key, group_rows in grouped.items():
        for field in (
            "backend",
            "model",
            "model_revision",
            "encoder_model",
            "encoder_revision",
            "vlm_engine",
            "vllm_enforce_eager",
            "vllm_max_num_seqs",
            "tensor_parallel_size",
            "vllm_max_model_len",
        ):
            values = {
                json.dumps(row.get(field), sort_keys=True)
                for row in group_rows
            }
            if len(values) > 1:
                raise ValueError(
                    "analysis group {} mixes {} values: {}".format(
                        group_key, field, sorted(values)
                    )
                )


def _analysis_input_paths(args):
    values = getattr(args, "inputs", None)
    if values is None:
        legacy = getattr(args, "input", None)
        values = [legacy] if legacy is not None else []
    if isinstance(values, (str, Path)):
        values = [values]
    paths = [Path(value) for value in values]
    if not paths:
        raise ValueError("analyze requires at least one --in file")
    return paths


def _sorted_present_values(rows, field):
    return sorted(
        {
            str(row[field])
            for row in rows
            if field in row and row[field] is not None
        }
    )


def _analysis_file_provenance(path, rows):
    return {
        "path": str(path),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "n_rows": len(rows),
        "backends": _sorted_present_values(rows, "backend"),
        "models": _sorted_present_values(rows, "model"),
        "encoder_models": _sorted_present_values(rows, "encoder_model"),
        "encoder_revisions": _sorted_present_values(
            rows, "encoder_revision"
        ),
        "model_revisions": _sorted_present_values(rows, "model_revision"),
        "vlm_engines": _sorted_present_values(rows, "vlm_engine"),
        "vllm_enforce_eager": _sorted_present_values(
            rows, "vllm_enforce_eager"
        ),
        "vllm_max_num_seqs": _sorted_present_values(
            rows, "vllm_max_num_seqs"
        ),
        "tensor_parallel_sizes": _sorted_present_values(
            rows, "tensor_parallel_size"
        ),
        "vllm_max_model_lens": _sorted_present_values(
            rows, "vllm_max_model_len"
        ),
        "source_manifest_sha256s": _sorted_present_values(
            rows, "source_manifest_sha256"
        ),
        "gemma_max_soft_tokens_effective": single_gemma_mst(rows),
    }


def _validate_analysis_rows(rows):
    required = (
        "schema_version",
        "backend",
        "model",
        "encoder",
        "encoder_model",
        "encoder_revision",
        "gemma_max_soft_tokens_effective",
        "source_manifest_sha256",
        "page_id",
        "task",
        "style",
        "ladder_ratio",
        "char_px_post",
        "char_px_effective",
        "image_path",
        "ground_truth",
        "transcription",
    )
    seen = set()
    for index, row in enumerate(rows):
        missing = [field for field in required if field not in row]
        if missing:
            raise ValueError(
                "analysis row {} missing required fields: {}".format(
                    index, ", ".join(missing)
                )
            )
        if row["schema_version"] != SCHEMA_VERSION:
            raise ValueError(
                "analysis row {} has unsupported schema_version={!r}".format(
                    index, row["schema_version"]
                )
            )
        if row["backend"] not in ("rapidocr", "vlm"):
            raise ValueError(
                "analysis row {} has unsupported backend={!r}".format(
                    index, row["backend"]
                )
            )
        if not isinstance(row["ground_truth"], str):
            raise ValueError(
                "analysis row {} ground_truth must be a string".format(index)
            )
        if not isinstance(row["transcription"], str):
            raise ValueError(
                "analysis row {} transcription must be a string".format(index)
            )
        manifest_sha = row["source_manifest_sha256"]
        if (
            not isinstance(manifest_sha, str)
            or len(manifest_sha) != 64
            or any(character not in "0123456789abcdef" for character in manifest_sha)
        ):
            raise ValueError(
                "analysis row {} source_manifest_sha256 must be lowercase SHA-256".format(
                    index
                )
            )
        if row["encoder"] != _encoder_label(row["encoder_model"]):
            raise ValueError(
                "analysis row {} encoder label does not match encoder_model".format(
                    index
                )
            )
        if row["backend"] == "vlm":
            vlm_required = (
                "model_revision",
                "prompt",
                "vlm_engine",
                "vllm_enforce_eager",
                "vllm_max_num_seqs",
                "tensor_parallel_size",
                "vllm_max_model_len",
            )
            missing_vlm = [
                field for field in vlm_required if field not in row
            ]
            if missing_vlm:
                raise ValueError(
                    "analysis VLM row {} missing serving provenance: {}".format(
                        index, ", ".join(missing_vlm)
                    )
                )
            if row["model"] != row["encoder_model"]:
                raise ValueError(
                    "analysis VLM row {} model must equal encoder_model".format(
                        index
                    )
                )
            if row["model_revision"] != row["encoder_revision"]:
                raise ValueError(
                    "analysis VLM row {} model_revision must equal "
                    "encoder_revision".format(index)
                )
            if row["prompt"] != TRANSCRIPTION_PROMPT:
                raise ValueError(
                    "analysis VLM row {} prompt differs from the transcription prompt".format(
                        index
                    )
                )
            if row["vlm_engine"] not in ("vllm", "transformers"):
                raise ValueError(
                    "analysis VLM row {} has unsupported vlm_engine".format(index)
                )
            if row["vlm_engine"] == "vllm":
                if not isinstance(row["vllm_enforce_eager"], bool):
                    raise ValueError(
                        "analysis VLM row {} vllm_enforce_eager must be bool".format(
                            index
                        )
                    )
                for field in ("tensor_parallel_size", "vllm_max_model_len"):
                    value = row[field]
                    if (
                        not isinstance(value, int)
                        or isinstance(value, bool)
                        or value <= 0
                    ):
                        raise ValueError(
                            "analysis VLM row {} {} must be positive".format(
                                index, field
                            )
                        )
                max_num_seqs = row["vllm_max_num_seqs"]
                if max_num_seqs is not None and (
                    not isinstance(max_num_seqs, int)
                    or isinstance(max_num_seqs, bool)
                    or max_num_seqs <= 0
                ):
                    raise ValueError(
                        "analysis VLM row {} vllm_max_num_seqs must be null "
                        "or positive".format(index)
                    )
        logical_key = (
            str(row["encoder_model"]),
            str(row["page_id"]),
            str(row["style"]),
            float(row["ladder_ratio"]),
        )
        if logical_key in seen:
            raise ValueError(
                "duplicate logical transcription row: {}".format(
                    logical_key
                )
            )
        seen.add(logical_key)


def _validate_analysis_alignment(rows):
    manifest_hashes = {
        str(row["source_manifest_sha256"]) for row in rows
    }
    if len(manifest_hashes) != 1:
        raise ValueError(
            "analysis inputs mix source manifest hashes: {}".format(
                sorted(manifest_hashes)
            )
        )

    cells_by_encoder = defaultdict(set)
    for row in rows:
        cells_by_encoder[str(row["encoder_model"])].add(
            (
                str(row["page_id"]),
                str(row["task"]),
                str(row["style"]),
                float(row["ladder_ratio"]),
                str(row["image_path"]),
                row["ground_truth"],
            )
        )
    if len(cells_by_encoder) > 1:
        reference_model = sorted(cells_by_encoder)[0]
        reference = cells_by_encoder[reference_model]
        for model_name in sorted(cells_by_encoder)[1:]:
            if cells_by_encoder[model_name] != reference:
                raise ValueError(
                    "analysis encoder cell set differs: {} versus {}".format(
                        reference_model, model_name
                    )
                )


def transcribe_pages(args):
    in_dir = Path(args.in_dir)
    manifest_path = in_dir / "manifest.jsonl"
    source_manifest_sha256 = hashlib.sha256(
        manifest_path.read_bytes()
    ).hexdigest()
    rows = _read_jsonl(manifest_path)
    gemma_max_soft_tokens = single_gemma_mst(rows)
    output = []

    if args.backend == "rapidocr":
        engine = _make_rapidocr()
        for row in rows:
            with Image.open(in_dir / row["image_path"]) as image:
                prediction = _rapidocr_text(image.convert("RGB"), engine)
            diagnostics = _transcription_diagnostics(
                prediction, row["ground_truth"], "rapidocr"
            )
            for model_name in ENCODER_MODELS:
                output_row = {
                        "schema_version": SCHEMA_VERSION,
                        "backend": "rapidocr",
                        "model": "rapidocr-onnxruntime",
                        "encoder": _encoder_label(model_name),
                        "encoder_model": model_name,
                        "encoder_revision": _encoder_revision(model_name),
                        "source_manifest_sha256": source_manifest_sha256,
                        "gemma_max_soft_tokens_effective": (
                            gemma_max_soft_tokens
                        ),
                        "page_id": row["page_id"],
                        "task": row["task"],
                        "style": row["style"],
                        "ladder_ratio": row["ladder_ratio"],
                        "char_px_post": row[
                            "char_px_post_by_encoder"
                        ][model_name],
                        "char_px_effective": row[
                            "char_px_effective_by_encoder"
                        ][model_name],
                        "image_path": row["image_path"],
                        "ground_truth": row["ground_truth"],
                        "transcription": prediction,
                        "char_accuracy": char_accuracy(
                            prediction, row["ground_truth"]
                        ),
                    }
                output_row.update(diagnostics)
                output.append(output_row)
    else:
        if not args.model:
            raise ValueError("--model is required for --backend vlm")
        encoder = _encoder_label(args.model)
        vlm = _make_vlm(
            args.model,
            args.vlm_engine,
            args.tensor_parallel_size,
            gemma_max_soft_tokens,
            enforce_eager=args.enforce_eager,
            max_num_seqs=args.max_num_seqs,
        )
        effective_engine = getattr(
            vlm, "calibration_engine", args.vlm_engine
        )
        try:
            for row in rows:
                with Image.open(in_dir / row["image_path"]) as image:
                    prediction = vlm.generate(
                        [image.convert("RGB")],
                        TRANSCRIPTION_PROMPT,
                        max_tokens=args.max_tokens,
                        temperature=0.0,
                    )
                diagnostics = _transcription_diagnostics(
                    prediction, row["ground_truth"], "vlm"
                )
                is_vllm = effective_engine == "vllm"
                output_row = {
                        "schema_version": SCHEMA_VERSION,
                        "backend": "vlm",
                        "vlm_engine": effective_engine,
                        "model": args.model,
                        "model_revision": _encoder_revision(args.model),
                        "encoder": encoder,
                        "encoder_model": args.model,
                        "encoder_revision": _encoder_revision(args.model),
                        "source_manifest_sha256": source_manifest_sha256,
                        "vllm_enforce_eager": (
                            bool(args.enforce_eager) if is_vllm else None
                        ),
                        "vllm_max_num_seqs": (
                            args.max_num_seqs if is_vllm else None
                        ),
                        "tensor_parallel_size": (
                            int(args.tensor_parallel_size) if is_vllm else None
                        ),
                        "vllm_max_model_len": 32768 if is_vllm else None,
                        "gemma_max_soft_tokens_effective": (
                            gemma_max_soft_tokens
                        ),
                        "page_id": row["page_id"],
                        "task": row["task"],
                        "style": row["style"],
                        "ladder_ratio": row["ladder_ratio"],
                        "char_px_post": row[
                            "char_px_post_by_encoder"
                        ][args.model],
                        "char_px_effective": row[
                            "char_px_effective_by_encoder"
                        ][args.model],
                        "image_path": row["image_path"],
                        "ground_truth": row["ground_truth"],
                        "transcription": prediction,
                        "char_accuracy": char_accuracy(
                            prediction, row["ground_truth"]
                        ),
                        "prompt": TRANSCRIPTION_PROMPT,
                    }
                output_row.update(diagnostics)
                output.append(output_row)
        finally:
            cleanup = getattr(vlm, "cleanup", None)
            if cleanup is not None:
                cleanup()

    output.sort(
        key=lambda row: (
            row["encoder"],
            row["page_id"],
            row["style"],
            -float(row["ladder_ratio"]),
        )
    )
    write_manifest(output, args.out)
    print(
        "calibration transcribe: backend={} rows={} out={}".format(
            args.backend, len(output), args.out
        )
    )
    return output


def analyze_file(args):
    input_paths = _analysis_input_paths(args)
    rows = []
    input_files = []
    for input_path in input_paths:
        file_rows = _read_jsonl(input_path)
        if not file_rows:
            raise ValueError(
                "analysis input file is empty: {}".format(input_path)
            )
        rows.extend(file_rows)
        input_files.append(
            _analysis_file_provenance(input_path, file_rows)
        )
    gemma_max_soft_tokens = single_gemma_mst(rows)
    _validate_analysis_rows(rows)
    _validate_analysis_alignment(rows)
    _validate_analysis_provenance(rows)
    for row in rows:
        if (
            row.get("encoder_model") == QWEN_MODEL
            and row.get("encoder_revision") != QWEN_REVISION
        ):
            raise ValueError(
                "Qwen input row is not pinned to revision {}".format(
                    QWEN_REVISION
                )
            )
    normalized = []
    for row in rows:
        item = dict(row)
        transcription = str(item["transcription"])
        ground_truth = str(item["ground_truth"])
        item["char_accuracy"] = char_accuracy(
            transcription,
            ground_truth,
        )
        item.update(
            _transcription_diagnostics(
                transcription, ground_truth, str(item["backend"])
            )
        )
        normalized.append(item)
    report = analyze_records(
        normalized,
        bootstrap_iterations=args.bootstrap_iterations,
        seed=args.seed,
    )
    report["input_rows"] = len(normalized)
    report["input_files"] = input_files
    if len(input_files) == 1:
        report["input_jsonl_sha256"] = input_files[0]["sha256"]
    report["gemma_max_soft_tokens_effective"] = gemma_max_soft_tokens
    report["input_provenance"] = {
        "backends": sorted(
            {str(row.get("backend")) for row in rows}
        ),
        "models": sorted({str(row.get("model")) for row in rows}),
        "encoder_models": sorted(
            {str(row.get("encoder_model")) for row in rows}
        ),
        "encoder_revisions": sorted(
            {
                str(row.get("encoder_revision"))
                for row in rows
                if row.get("encoder_revision") is not None
            }
        ),
        "model_revisions": sorted(
            {
                str(row.get("model_revision"))
                for row in rows
                if row.get("model_revision") is not None
            }
        ),
        "vlm_engines": sorted(
            {
                str(row.get("vlm_engine"))
                for row in rows
                if row.get("vlm_engine") is not None
            }
        ),
        "vllm_enforce_eager": sorted(
            {
                bool(row["vllm_enforce_eager"])
                for row in rows
                if row.get("vllm_enforce_eager") is not None
            }
        ),
        "vllm_max_num_seqs": sorted(
            {
                int(row["vllm_max_num_seqs"])
                for row in rows
                if row.get("vllm_max_num_seqs") is not None
            }
        ),
        "tensor_parallel_sizes": sorted(
            {
                int(row["tensor_parallel_size"])
                for row in rows
                if row.get("tensor_parallel_size") is not None
            }
        ),
        "vllm_max_model_lens": sorted(
            {
                int(row["vllm_max_model_len"])
                for row in rows
                if row.get("vllm_max_model_len") is not None
            }
        ),
        "source_manifest_sha256s": sorted(
            {str(row["source_manifest_sha256"]) for row in rows}
        ),
    }
    _write_json(report, args.out)
    print(
        "calibration analyze: rows={} groups={} out={}".format(
            len(normalized), len(report["groups"]), args.out
        )
    )
    return report


def _positive_int(value):
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    generate = subparsers.add_parser("gen", help="generate deterministic pages")
    generate.add_argument("--pages", type=int, default=50)
    generate.add_argument("--seed", type=int, default=42)
    generate.add_argument("--styles", default="plain,dark_highlight")
    generate.add_argument("--ladder", default="1.0,0.75,0.5,0.35,0.25")
    generate.add_argument("--out-dir", required=True)
    generate.add_argument("--width", type=int, default=DEFAULT_WIDTH)
    generate.add_argument("--height", type=int, default=DEFAULT_HEIGHT)
    generate.add_argument("--font-size", type=int, default=DEFAULT_FONT_SIZE)
    generate.add_argument(
        "--gemma-max-soft-tokens",
        type=int,
        choices=GEMMA_SUPPORTED_MAX_SOFT_TOKENS,
        default=DEFAULT_GEMMA_MAX_SOFT_TOKENS,
        help="Gemma processor grid setting recorded with every page",
    )
    generate.add_argument(
        "--census-dir", default=str(DEFAULT_CENSUS_DIR)
    )
    generate.add_argument("--max-negative-extensions", type=int, default=8)
    generate.set_defaults(handler=generate_pages)

    transcribe = subparsers.add_parser(
        "transcribe", help="transcribe generated pages"
    )
    transcribe.add_argument(
        "--backend", choices=("rapidocr", "vlm"), required=True
    )
    transcribe.add_argument("--model")
    transcribe.add_argument("--in-dir", required=True)
    transcribe.add_argument("--out", required=True)
    transcribe.add_argument(
        "--vlm-engine",
        choices=("auto", "vllm", "transformers"),
        default="auto",
    )
    transcribe.add_argument(
        "--tensor-parallel-size", type=_positive_int, default=2
    )
    transcribe.add_argument("--max-tokens", type=int, default=4096)
    transcribe.add_argument("--enforce-eager", action="store_true")
    transcribe.add_argument("--max-num-seqs", type=_positive_int)
    transcribe.set_defaults(handler=transcribe_pages)

    analyze = subparsers.add_parser(
        "analyze", help="compute the knee thresholds"
    )
    analyze.add_argument(
        "--in", dest="inputs", action="append", required=True
    )
    analyze.add_argument("--out", required=True)
    analyze.add_argument("--bootstrap-iterations", type=int, default=2000)
    analyze.add_argument("--seed", type=int, default=42)
    analyze.set_defaults(handler=analyze_file)
    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    result = args.handler(args)
    if (
        args.command == "analyze"
        and any(
            group.get("axis_degenerate", False)
            for group in result.get("groups", [])
        )
    ):
        affected = [
            "{}/{}".format(group["encoder"], group["style"])
            for group in result["groups"]
            if group.get("axis_degenerate", False)
        ]
        print(
            "calibration analyze: AXIS-DEGENERATE; refusing knee for {}".format(
                ", ".join(affected)
            ),
            file=sys.stderr,
        )
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
