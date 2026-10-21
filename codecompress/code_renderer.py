"""
Render source code as images for VLM-based code understanding.

Supports multiple rendering styles:
  - plain:     Black monospace text on white background
  - highlight: Syntax highlighting (VS Code "Default Light" theme via Pygments)
  - bold:      Plain with bold font weight
  - ide:       Syntax highlight + line numbers + indentation guides (novel)
  - ide_lint:  IDE-style + linting annotations (novel, requires lint_errors list)

Compression is achieved by downsampling images to reduce visual token count.
Token counting follows the Qwen-VL tiling scheme (112x112 tiles, 16 tokens/tile).

Reference implementation: CodeOCR/CodeOCR/core/ (read-only, not modified).

Usage:
    from codecompress.code_renderer import render_code, compress_images, count_image_tokens

    images = render_code("def foo():\n    return 42", style="highlight", language="python")
    compressed = compress_images(images, text_token_count=500, compression_ratio=4)
    tokens = count_image_tokens(compressed)
"""

from __future__ import annotations

import functools
import math
from typing import Literal

from PIL import Image, ImageDraw, ImageFont

# Try to import Pygments for syntax highlighting
try:
    import pygments
    from pygments import lex
    from pygments.lexers import get_lexer_by_name, guess_lexer
    from pygments.token import (
        Comment, Error, Keyword, Name, Number,
        Operator, Punctuation, String, Token,
    )
    PYGMENTS_AVAILABLE = True
except ImportError:
    PYGMENTS_AVAILABLE = False


# =============================================================================
# Constants (matching CodeOCR paper specs where applicable)
# =============================================================================

# Default rendering parameters
DEFAULT_FONT_SIZE = 40          # px, matching CodeOCR
DEFAULT_LINE_HEIGHT = 1.0       # multiplier on font size
DEFAULT_DPI = 300
DEFAULT_MARGIN_RATIO = 0.01     # 1% of image width

# Image dimensions — 2240x2240 is CodeOCR's base resolution,
# divisible by common VLM patch sizes (14, 16, 112)
DEFAULT_IMAGE_WIDTH = 2240
DEFAULT_IMAGE_HEIGHT = 2240

# Monospace character width approximation
CHAR_WIDTH_RATIO = 0.6          # char_width ≈ font_size * 0.6

# Qwen2.5-VL token counting: patch_size=14, spatial_merge_size=2.
# Each 112x112 tile = 8x8 = 64 pre-merge patches → merged 2x2 → 16 actual LLM tokens.
# Verified empirically against the real processor.
QWEN_TILE_SIZE = 112
QWEN_TOKENS_PER_TILE = 16

# Qwen3-VL token counting: patch_size=16, spatial_merge_size=2.
# No tile boundary — processor resizes to nearest 32px multiple, min 256×256.
# tokens = (round(w/32) * round(h/32)), minimum 8×8 = 64 tokens.
# Verified empirically against the real processor.
QWEN3_MERGE_BLOCK = 32
QWEN3_MIN_SIDE = 8  # minimum tokens per side

# Qwen3.6/3.8 processor smart-resize law (verified against the real
# AutoProcessor).  Keep these separate from count_image_tokens()'s historical
# per-axis clamp: that function is a compatibility surface, while
# page_visual_tokens below is the exact law used by the page-fit decision.
QWEN35_RESIZE_FACTOR = 32
QWEN35_MIN_PIXELS = 65_536       # 256 x 256
QWEN35_MAX_PIXELS = 16_777_216  # 4096 x 4096

# Gemma 3 / Gemma 4 token counting: SigLIP vision encoder with internal
# resize and CONFIGURABLE pooling. Empirically verified:
# Gemma4 processor accepts a `max_soft_tokens` parameter selecting
# from 5 supported pooling settings, each producing a fixed
# square-grid count of <|image|> tokens regardless of input
# resolution:
#
#   max_soft_tokens=70   →  64 tokens per image  (8×8 grid)
#   max_soft_tokens=140  → 121 tokens per image (11×11 grid)
#   max_soft_tokens=280  → 256 tokens per image (16×16 grid; default)
#   max_soft_tokens=560  → 529 tokens per image (23×23 grid)
#   max_soft_tokens=1120 → 1089 tokens per image (33×33 grid)
#
# All 5 settings produce a token count independent of input resolution;
# only the patch-grid density changes. Default 280 → 256 tokens
# matches the original "GEMMA_TOKENS_PER_IMAGE = 256" assumption.
GEMMA_SUPPORTED_MAX_SOFT_TOKENS = (70, 140, 280, 560, 1120)
GEMMA_TOKENS_PER_IMAGE_BY_MAX_SOFT_TOKENS = {
    70: 64,
    140: 121,
    280: 256,
    560: 529,
    1120: 1089,
}
# Backward-compat alias preserving the original constant for any
# caller that imported it directly. Equals the default-mapping value.
GEMMA_TOKENS_PER_IMAGE = GEMMA_TOKENS_PER_IMAGE_BY_MAX_SOFT_TOKENS[280]


def gemma_tokens_per_image(max_soft_tokens: int = 280) -> int:
    """Return actual <|image|> token count for a given max_soft_tokens setting.

    Empirically verified against the google/gemma-4-26B-A4B-it processor.
    Raises ValueError for
    unsupported settings.
    """
    if max_soft_tokens not in GEMMA_TOKENS_PER_IMAGE_BY_MAX_SOFT_TOKENS:
        raise ValueError(
            f"max_soft_tokens={max_soft_tokens} unsupported for Gemma4. "
            f"Allowed: {sorted(GEMMA_SUPPORTED_MAX_SOFT_TOKENS)}"
        )
    return GEMMA_TOKENS_PER_IMAGE_BY_MAX_SOFT_TOKENS[max_soft_tokens]


def _validate_page_geometry(width: int, height: int) -> None:
    """Validate dimensions consumed by the measured processor laws."""
    if any(
        not isinstance(value, int) or isinstance(value, bool) or value <= 0
        for value in (width, height)
    ):
        raise ValueError(
            f"page dimensions must be positive integers, got {width}x{height}"
        )


def _qwen35_resized_size(width: int, height: int) -> tuple[int, int]:
    """Return the real Qwen3.6 processor output size as ``(width, height)``.

    This is the joint area-constrained smart-resize law verified by the live
    processor.  It intentionally does
    not reuse ``count_image_tokens``: that compatibility helper clamps each
    axis independently and is wrong for non-square pages near the area floor.
    """
    _validate_page_geometry(width, height)
    factor = QWEN35_RESIZE_FACTOR
    resized_h = round(height / factor) * factor or factor
    resized_w = round(width / factor) * factor or factor
    area = resized_h * resized_w
    if area > QWEN35_MAX_PIXELS:
        beta = math.sqrt((height * width) / QWEN35_MAX_PIXELS)
        resized_h = math.floor(height / beta / factor) * factor
        resized_w = math.floor(width / beta / factor) * factor
    elif area < QWEN35_MIN_PIXELS:
        beta = math.sqrt(QWEN35_MIN_PIXELS / (height * width))
        resized_h = math.ceil(height * beta / factor) * factor
        resized_w = math.ceil(width * beta / factor) * factor
    return resized_w, resized_h


def _gemma4_resized_size(
    width: int,
    height: int,
    gemma_max_soft_tokens: int = 280,
) -> tuple[int, int]:
    """Return Gemma4's measured patch-grid size as ``(width, height)``.

    Mirrors ``Gemma4ImageProcessor.get_aspect_ratio_preserving_size`` for
    patch_size=16 and pooling_kernel_size=3, including its one-axis-zero
    handling for extreme aspect ratios.  The five supported soft-token values
    are the processor's existing public configurations.
    """
    _validate_page_geometry(width, height)
    # Reuse the established validator/supported-value source of truth.
    gemma_tokens_per_image(gemma_max_soft_tokens)
    patch_size = 16
    pooling_kernel_size = 3
    side_multiple = patch_size * pooling_kernel_size
    max_patches = gemma_max_soft_tokens * pooling_kernel_size**2
    target_pixels = max_patches * patch_size**2
    scale = math.sqrt(target_pixels / (height * width))
    resized_h = math.floor((scale * height) / side_multiple) * side_multiple
    resized_w = math.floor((scale * width) / side_multiple) * side_multiple

    if resized_h == 0 and resized_w == 0:
        raise ValueError(
            "Gemma4 processor law would resize both page dimensions to zero "
            f"for {width}x{height}"
        )

    max_side_length = (max_patches // pooling_kernel_size**2) * side_multiple
    if resized_h == 0:
        resized_h = side_multiple
        resized_w = min(
            math.floor(width / height) * side_multiple,
            max_side_length,
        )
    elif resized_w == 0:
        resized_w = side_multiple
        resized_h = min(
            math.floor(height / width) * side_multiple,
            max_side_length,
        )

    if resized_h * resized_w > target_pixels:
        raise ValueError(
            f"Gemma4 processor law resized {width}x{height} to "
            f"{resized_w}x{resized_h}, exceeding its patch budget"
        )
    return resized_w, resized_h


def page_visual_tokens(
    width: int,
    height: int,
    model_name: str,
    gemma_max_soft_tokens: int = 280,
) -> int:
    """Return the real processor visual-token cost for one rendered page.

    Supports the two studied encoders.  The Gemma soft-token argument is
    ignored for non-Gemma models so callers can forward one effective value
    without branching.  ``count_image_tokens`` remains unchanged for backward
    compatibility and must not be used for the page fit.
    """
    name = str(model_name).lower()
    if "gemma-4" in name:
        resized_w, resized_h = _gemma4_resized_size(
            width, height, gemma_max_soft_tokens
        )
        patch_count = (resized_w // 16) * (resized_h // 16)
        return patch_count // (3**2)
    if "gemma" in name:
        raise ValueError(
            "page_visual_tokens supports Gemma-4 geometry only; "
            f"got a different Gemma model_name={model_name!r}"
        )
    if "qwen3" in name:
        resized_w, resized_h = _qwen35_resized_size(width, height)
        return (
            (resized_w // QWEN35_RESIZE_FACTOR)
            * (resized_h // QWEN35_RESIZE_FACTOR)
        )
    raise ValueError(
        "page_visual_tokens supports Gemma4 and Qwen3 publication encoders; "
        f"got model_name={model_name!r}"
    )


def char_px_post_resize(
    font_size: int,
    width: int,
    height: int,
    model_name: str,
    gemma_max_soft_tokens: int = 280,
) -> float:
    """Return post-processor vertical pixels per rendered character line.

    The rendering-side character-height unit is the renderer's line pitch,
    ``int(font_size * DEFAULT_LINE_HEIGHT)``.  Processor scaling is explicitly
    vertical because Gemma's resize may be anisotropic.
    """
    if not isinstance(font_size, int) or isinstance(font_size, bool) or font_size <= 0:
        raise ValueError(f"font_size must be a positive integer, got {font_size!r}")
    name = str(model_name).lower()
    if "gemma-4" in name:
        _, resized_h = _gemma4_resized_size(
            width, height, gemma_max_soft_tokens
        )
    elif "gemma" in name:
        raise ValueError(
            "char_px_post_resize supports Gemma-4 geometry only; "
            f"got a different Gemma model_name={model_name!r}"
        )
    elif "qwen3" in name:
        _, resized_h = _qwen35_resized_size(width, height)
    else:
        raise ValueError(
            "char_px_post_resize supports Gemma4 and Qwen3 publication "
            f"encoders; got model_name={model_name!r}"
        )
    line_px = int(font_size * DEFAULT_LINE_HEIGHT)
    return line_px * (resized_h / height)


def char_px_effective(
    font_size: int,
    width: int,
    height: int,
    model_name: str,
    gemma_max_soft_tokens: int = 280,
) -> float:
    """Return information-bottleneck vertical pixels per character line.

    Processor upsampling cannot restore detail that was absent from the
    rendered page.  The effective height is therefore the smaller of the
    renderer line pitch and its physical height after processor resizing.
    """
    line_px = int(font_size * DEFAULT_LINE_HEIGHT)
    return min(
        float(line_px),
        char_px_post_resize(
            font_size,
            width,
            height,
            model_name,
            gemma_max_soft_tokens=gemma_max_soft_tokens,
        ),
    )

# Pixtral encoder (Mistral Small 3.1+) token counting:
#   tokens(W, H) = min(MAX_SIDE, ceil(W'/28)) * min(MAX_SIDE, ceil(H'/28))
# where (W', H') is (W, H) rescaled so longest_side ≤ PIXTRAL_LONGEST_SIDE_PX
# (preserving aspect ratio). The internal cap is PIXTRAL_MAX_PATCHES_PER_SIDE
# patches per side. Verified empirically against
# mistralai/Mistral-Small-3.1-24B-Instruct-2503 processor; image pad
# token is `[IMG]` (id 10). At our standard 1120×1120 canvas (below the
# rescale threshold), tokens = 40×40 = 1600.
PIXTRAL_MERGE_BLOCK = 28
PIXTRAL_MAX_PATCHES_PER_SIDE = 55
PIXTRAL_LONGEST_SIDE_PX = PIXTRAL_MAX_PATCHES_PER_SIDE * PIXTRAL_MERGE_BLOCK  # 1540

# Font paths (priority order)
FONT_PATHS = [
    "DejaVuSansMono",
    "DejaVuSansMono.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
]
BOLD_FONT_PATHS = [
    "DejaVuSansMono-Bold",
    "DejaVuSansMono-Bold.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf",
]

# VS Code "Default Light" theme colors (matching CodeOCR)
THEME_COLORS_LIGHT = {
    Token:              "#000000",  # default black
    Keyword:            "#0000FF",  # blue
    Keyword.Constant:   "#0000FF",
    Keyword.Namespace:  "#AF00DB",  # magenta (import/from)
    Name.Function:      "#795E26",  # brown
    Name.Class:         "#267F99",  # teal
    Name.Decorator:     "#795E26",  # brown
    Name.Builtin:       "#267F99",  # teal
    String:             "#A31515",  # red
    String.Doc:         "#A31515",
    Comment:            "#008000",  # green
    Comment.Single:     "#008000",
    Comment.Multiline:  "#008000",
    Number:             "#098658",  # dark green
    Number.Integer:     "#098658",
    Number.Float:       "#098658",
    Operator:           "#000000",  # black
    Punctuation:        "#000000",
    Name:               "#000000",
    Error:              "#FF0000",  # red (for linting)
}

# VS Code "Dark+" theme colors
THEME_COLORS_DARK = {
    Token:              "#D4D4D4",  # default light gray
    Keyword:            "#569CD6",  # blue
    Keyword.Constant:   "#569CD6",
    Keyword.Namespace:  "#C586C0",  # magenta (import/from)
    Name.Function:      "#DCDCAA",  # yellow
    Name.Class:         "#4EC9B0",  # teal
    Name.Decorator:     "#DCDCAA",  # yellow
    Name.Builtin:       "#4EC9B0",  # teal
    String:             "#CE9178",  # orange
    String.Doc:         "#CE9178",
    Comment:            "#6A9955",  # green
    Comment.Single:     "#6A9955",
    Comment.Multiline:  "#6A9955",
    Number:             "#B5CEA8",  # light green
    Number.Integer:     "#B5CEA8",
    Number.Float:       "#B5CEA8",
    Operator:           "#D4D4D4",  # light gray
    Punctuation:        "#D4D4D4",
    Name:               "#9CDCFE",  # light blue
    Error:              "#F44747",  # red
}

# IDE-style colors (light theme)
IDE_LINE_NUMBER_COLOR = "#858585"       # gray
IDE_LINE_NUMBER_BG = "#F5F5F5"          # light gray gutter
IDE_INDENTATION_GUIDE_COLOR = "#D3D3D3" # light gray
IDE_LINT_ERROR_COLOR = "#FF0000"        # red underline
IDE_LINT_WARNING_COLOR = "#FFA500"      # orange underline

# IDE-style colors (dark theme)
IDE_LINE_NUMBER_COLOR_DARK = "#858585"       # gray (same in dark)
IDE_LINE_NUMBER_BG_DARK = "#1E1E1E"          # dark gutter (same as bg)
IDE_INDENTATION_GUIDE_COLOR_DARK = "#404040" # dark gray

RenderStyle = Literal["plain", "highlight", "bold", "ide", "ide_lint", "dark", "dark_highlight"]
STYLES: list[str] = ["plain", "highlight", "bold", "ide", "ide_lint", "dark", "dark_highlight"]


# =============================================================================
# Font loading
# =============================================================================

def _load_font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    """Load a monospace font at the given size. Falls back to default if not found."""
    paths = BOLD_FONT_PATHS if bold else FONT_PATHS
    for path in paths:
        try:
            return ImageFont.truetype(path, size)
        except (IOError, OSError):
            continue
    # Fallback: try the non-bold version for bold, or Pillow default
    if bold:
        for path in FONT_PATHS:
            try:
                return ImageFont.truetype(path, size)
            except (IOError, OSError):
                continue
    return ImageFont.load_default()


# =============================================================================
# Token counting
# =============================================================================

# Lazily-loaded module holding the MiniCPM-V 4.5 measured token law
# (scripts/verify_minicpm_visual_tokens.py, a measurement script that is not
# part of this package: the MiniCPM branch is not used by the study). It is
# loaded by file path relative to the repo root. Cached after first load.
_MINICPM_LAW_MODULE = None


def _load_minicpm_token_law():
    """Load the verified MiniCPM-V 4.5 per-image token law (pure helpers).

    The law (64·(1 + grid_cols·grid_rows), area-driven LLaVA-UHD slicing,
    640-token cap at max_slice_nums=9) was MEASURED against the real
    openbmb/MiniCPM-V-4_5 processor (rev fd3209b) and lives in
    scripts/verify_minicpm_visual_tokens.py (not part of this package) —
    imported here rather than duplicated (never hand-copy an architecture
    law).
    """
    global _MINICPM_LAW_MODULE
    if _MINICPM_LAW_MODULE is None:
        import importlib.util
        from pathlib import Path

        path = (Path(__file__).resolve().parent.parent
                / "scripts" / "verify_minicpm_visual_tokens.py")
        if not path.exists():
            raise FileNotFoundError(
                f"MiniCPM token law reference not found at {path} — "
                f"count_image_tokens' minicpm branch imports its measured "
                f"helpers from that script, which is not part of this package."
            )
        spec = importlib.util.spec_from_file_location(
            "verify_minicpm_visual_tokens", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        _MINICPM_LAW_MODULE = module
    return _MINICPM_LAW_MODULE


def count_image_tokens(
    images: list[Image.Image],
    model_name: str = "",
    *,
    gemma_max_soft_tokens: int = 280,
) -> int:
    """
    Count visual tokens for a list of images.

    Supports six schemes:
      - Qwen2.5-VL (default): tile-based, ceil(W/112) × ceil(H/112) × 16
      - Qwen3-VL / Qwen3.5-VL: block-based, max(8, round(W/32)) × max(8, round(H/32))
      - Gemma 3 / Gemma 4: configurable resolution-invariant count via
        max_soft_tokens setting (5 supported values; default 280 → 256
        tokens per image; see GEMMA_TOKENS_PER_IMAGE_BY_MAX_SOFT_TOKENS).
      - Aria (rhymes-ai): fixed 256 tokens per image (resolution-invariant)
      - Kimi-VL (Moonshot AI, Instruct): min(1024, ceil(W/28) * ceil(H/28)) per image
      - Pixtral (Mistral Small 3.1+): rescale longest side to ≤1540, then
        ceil(W'/28) × ceil(H'/28) capped at 55² = 3025 per image

    Model scheme is chosen by substring in model_name:
      - "gemma"    → Gemma formula (configurable, default 256/image)
      - "minicpm"  → MiniCPM-V 4.5 formula: the FULL per-image LLaVA-UHD
                     slicing law 64·(1 + cols·rows), msn=9 cap 640 — imported
                     from scripts/verify_minicpm_visual_tokens.py (not part of
                     this package; never a flat 448·len(pages))
      - "aria"     → Aria formula (256/image)
      - "kimi"     → Kimi-VL formula (block-28, 1024 Instruct cap; Thinking-2506: 4094, unsupported)
      - "mistral"  → Pixtral formula (block-28 with 1540-px longest-side rescale)
      - "qwen3"    → Qwen3 formula
      - else       → Qwen2.5 tile formula

    The `gemma_max_soft_tokens` keyword argument only affects the Gemma
    branch; it is silently ignored on other model families. Default 280 is
    the processor default.
    """
    name = model_name.lower()
    if "gemma" in name:
        return gemma_tokens_per_image(gemma_max_soft_tokens) * len(images)
    if "minicpm" in name:
        minicpm_visual_tokens = _load_minicpm_token_law().minicpm_visual_tokens
        return sum(minicpm_visual_tokens(img.size) for img in images)
    if "mistral" in name:
        total = 0
        for img in images:
            w, h = img.size
            longest = max(w, h)
            if longest > PIXTRAL_LONGEST_SIDE_PX:
                scale = PIXTRAL_LONGEST_SIDE_PX / longest
                w = int(round(w * scale))
                h = int(round(h * scale))
            blocks_w = min(PIXTRAL_MAX_PATCHES_PER_SIDE, math.ceil(w / PIXTRAL_MERGE_BLOCK))
            blocks_h = min(PIXTRAL_MAX_PATCHES_PER_SIDE, math.ceil(h / PIXTRAL_MERGE_BLOCK))
            total += blocks_w * blocks_h
        return total
    use_qwen3 = "qwen3" in name
    total = 0
    for img in images:
        w, h = img.size
        if use_qwen3:
            rw = max(QWEN3_MIN_SIDE, round(w / QWEN3_MERGE_BLOCK))
            rh = max(QWEN3_MIN_SIDE, round(h / QWEN3_MERGE_BLOCK))
            total += rw * rh
            continue
        tiles_w = math.ceil(w / QWEN_TILE_SIZE)
        tiles_h = math.ceil(h / QWEN_TILE_SIZE)
        total += tiles_w * tiles_h * QWEN_TOKENS_PER_TILE
    return total


def find_resolution_for_tokens(target_tokens: int) -> tuple[int, int]:
    """
    Find a square resolution that produces approximately target_tokens visual tokens.

    Returns (width, height) in pixels. Resolution is snapped to multiples of 112
    for clean tiling (no partial tiles wasted).
    """
    if target_tokens <= 0:
        return (14, 14)  # minimum

    # target_tokens = tiles * QWEN_TOKENS_PER_TILE, so tiles = target_tokens / 16
    # For square: tiles = tiles_per_side^2, so tiles_per_side = sqrt(tiles)
    tiles = target_tokens / QWEN_TOKENS_PER_TILE
    tiles_per_side = max(1, round(math.sqrt(tiles)))
    size = tiles_per_side * QWEN_TILE_SIZE
    return (size, size)


# =============================================================================
# Syntax highlighting
# =============================================================================

def _get_token_color(token_type, theme: dict | None = None) -> str:
    """Look up color for a Pygments token type, walking up the token hierarchy."""
    if theme is None:
        theme = THEME_COLORS_LIGHT
    while token_type:
        if token_type in theme:
            return theme[token_type]
        token_type = token_type.parent
    return theme[Token]


def _tokenize_code(
    code: str, language: str = "python", theme: dict | None = None,
) -> list[tuple[str, str]]:
    """
    Tokenize code with Pygments and return list of (color_hex, text) pairs.

    Falls back to plain black if Pygments is unavailable.
    """
    fallback = "#D4D4D4" if theme is THEME_COLORS_DARK else "#000000"
    if not PYGMENTS_AVAILABLE:
        return [(fallback, code)]

    # stripnl=False: Pygments strips leading and trailing newlines by default.
    # With the leading ones gone, the token stream starts one or more lines
    # late, _build_line_color_map shifts every colour segment up by that many
    # lines, and a highlighted row then draws the NEXT line's text cut to the
    # current line's length. The token text must reproduce the source text.
    try:
        lexer = get_lexer_by_name(language, stripnl=False)
    except pygments.util.ClassNotFound:
        lexer = guess_lexer(code, stripnl=False)

    result = []
    for token_type, value in lex(code, lexer):
        color = _get_token_color(token_type, theme)
        result.append((color, value))
    return result


# =============================================================================
# Code preprocessing & linting
# =============================================================================

# The ruff version the environments pin. Used to pin the uv-tool fallback
# below so an env without a PATH ruff can't silently lint with a different
# (latest) ruff.
_RUFF_FALLBACK_PIN = "0.15.15"


@functools.lru_cache(maxsize=1)
def lint_tool_version() -> str | None:
    """Resolved lint tool + version string (e.g. "ruff 0.15.15"), or None.

    Uses the SAME resolution order as generate_lint_errors (PATH ruff first,
    then the pinned uv-tool fallback), so it reports the binary the lint marks
    actually came from. Recorded per ide_lint example in image_info
    (run-level provenance for the render arm's linter). Cached —
    the resolution cannot change mid-run.
    """
    import subprocess

    for cmd in [
        ["ruff", "--version"],
        ["uv", "tool", "run", f"ruff@{_RUFF_FALLBACK_PIN}", "--version"],
    ]:
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        except (FileNotFoundError, subprocess.TimeoutExpired):
            continue
        if result.returncode == 0 and result.stdout.strip():
            return result.stdout.strip()
    return None


def generate_lint_errors(code: str) -> list[tuple[int, str, str]]:
    """Run ruff on code and return lint errors as (line_no, severity, message) tuples.

    Writes code to a temporary file, runs ruff with default rules (E+F),
    and parses JSON output. Returns empty list if ruff is not installed or fails.
    """
    import json
    import subprocess
    import tempfile
    import os

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".py", delete=False
    ) as f:
        f.write(code)
        fname = f.name

    try:
        # Try ruff directly (PATH — the launcher prepends the environment's
        # bin, so the pinned binary wins), then fall back to uv tool run with the
        # SAME pin (an unpinned fallback would lint with whatever is latest).
        for cmd in [
            ["ruff", "check", "--output-format=json", "--no-fix", fname],
            ["uv", "tool", "run", f"ruff@{_RUFF_FALLBACK_PIN}", "check",
             "--output-format=json", "--no-fix", fname],
        ]:
            try:
                result = subprocess.run(
                    cmd, capture_output=True, text=True, timeout=30,
                )
                break
            except FileNotFoundError:
                continue
        else:
            return []

        if not result.stdout.strip():
            return []
        diagnostics = json.loads(result.stdout)
        errors = []
        for d in diagnostics:
            line = d.get("location", {}).get("row", 0)
            # ruff >=0.5 reports syntax errors as code "invalid-syntax"
            # (older ruff emitted JSON null — guard with `or ""`).
            rule = d.get("code") or ""
            msg = d.get("message", "")
            severity = (
                "error"
                if rule == "invalid-syntax" or rule.startswith("E") or rule.startswith("F")
                else "warning"
            )
            errors.append((line, severity, f"{rule}: {msg}"))
        return errors
    except (subprocess.TimeoutExpired, json.JSONDecodeError):
        return []
    finally:
        os.unlink(fname)


def strip_indent(code: str) -> str:
    """Remove all leading whitespace from each line, preserving line breaks.

    This is used as an ablation to test whether VLMs can leverage indentation
    structure in code images. For Python, indentation encodes the syntax tree,
    so removing it destroys structural information while preserving content.
    """
    if not code:
        return code
    return "\n".join(line.lstrip() for line in code.split("\n"))


def shuffle_indent(code: str, seed: int = 0) -> str:
    """Randomly reassign indentation levels, preserving line content.

    Each non-empty line gets a random indent level (0, 4, 8, 12, or 16 spaces).
    Empty lines are preserved. The seed ensures reproducibility per example.

    This is used as an ablation to test whether VLMs leverage the *correctness*
    of indentation structure. Unlike strip_indent (which removes all indent),
    shuffle_indent preserves similar visual density but destroys structural meaning.
    """
    if not code:
        return code
    import random
    rng = random.Random(seed)
    levels = [0, 4, 8, 12, 16]
    lines = code.split("\n")
    result = []
    for line in lines:
        stripped = line.lstrip()
        if not stripped:
            result.append(line)  # preserve empty lines as-is
        else:
            new_indent = rng.choice(levels)
            result.append(" " * new_indent + stripped)
    return "\n".join(result)


def strip_braces(code: str) -> str:
    """Remove all `{` and `}` characters from code.

    For Java/C#, braces delimit code blocks (same role as indentation in Python).
    Removing them destroys block structure while preserving statement content.
    Used in cross-language syntax ablation experiments.
    """
    if not code:
        return code
    return code.replace("{", "").replace("}", "")


def strip_semicolons(code: str) -> str:
    """Remove all `;` characters from code.

    For Java/C#, semicolons terminate statements. Removing them destroys
    statement boundaries while preserving content. Used in cross-language
    syntax ablation experiments.
    """
    if not code:
        return code
    return code.replace(";", "")


# Private aliases so render_code can call these when its bool params
# shadow the function names.
_strip_indent = strip_indent
_shuffle_indent = shuffle_indent


# =============================================================================
# Core rendering
# =============================================================================

def _get_style_config(style: RenderStyle) -> dict:
    """Return rendering flags for a given style."""
    configs = {
        "plain":          {"use_syntax": False, "use_bold": False, "show_line_numbers": False, "show_indent_guides": False},
        "highlight":      {"use_syntax": True,  "use_bold": False, "show_line_numbers": False, "show_indent_guides": False},
        "bold":           {"use_syntax": False, "use_bold": True,  "show_line_numbers": False, "show_indent_guides": False},
        "ide":            {"use_syntax": True,  "use_bold": False, "show_line_numbers": True,  "show_indent_guides": True},
        "ide_lint":       {"use_syntax": True,  "use_bold": False, "show_line_numbers": True,  "show_indent_guides": True},
        "dark":           {"use_syntax": False, "use_bold": False, "show_line_numbers": False, "show_indent_guides": False,
                           "bg_color": "#1E1E1E", "text_color": "#D4D4D4", "theme": "dark"},
        "dark_highlight": {"use_syntax": True,  "use_bold": False, "show_line_numbers": False, "show_indent_guides": False,
                           "bg_color": "#1E1E1E", "text_color": "#D4D4D4", "theme": "dark"},
    }
    return configs[style]


def render_code(
    code: str,
    style: RenderStyle = "plain",
    language: str = "python",
    width: int = DEFAULT_IMAGE_WIDTH,
    height: int = DEFAULT_IMAGE_HEIGHT,
    font_size: int = DEFAULT_FONT_SIZE,
    lint_errors: list[tuple[int, str, str]] | None = None,
    strip_indent: bool = False,
    shuffle_indent: bool = False,
    downsample_resolution: int | None = None,
    pad_page_to: tuple[int, int] | None = None,
) -> list[Image.Image]:
    """
    Render source code as one or more images.

    Args:
        code: Source code string to render.
        style: Rendering style — "plain", "highlight", "bold", "ide", or "ide_lint".
        language: Programming language for syntax highlighting.
        width: Image width in pixels.
        height: Image height in pixels.
        font_size: Font size in pixels.
        lint_errors: For "ide_lint" style only. List of (line_number, severity, message)
                     where severity is "error" or "warning" and line_number is 1-based.
        strip_indent: If True, remove all leading whitespace from each line before
                      rendering. Used as an ablation to test indentation's effect.
        downsample_resolution: If set, downsample all pages after rendering.
                              E.g., 448 means scale from 1120 base to ~448 px.
        pad_page_to: If set to (W, H), pad EVERY output page — including the
                     auto-cropped last page — to exactly WxH with the style
                     background color, content top-left anchored. Applied AFTER
                     _auto_crop (and after downsample_resolution, so the output
                     dims are always exact). Freezes the page geometry for
                     area-driven encoders. Raises ValueError if any page is
                     larger than (W, H). Default None = zero behavior change.

    Returns:
        List of PIL Image objects (one per page).
    """
    pad_page_to = _validate_pad_page_to(pad_page_to)
    if strip_indent:
        code = _strip_indent(code)
    elif shuffle_indent:
        import hashlib
        seed = int(hashlib.sha256(code.encode()).hexdigest(), 16) % (2**31)
        code = _shuffle_indent(code, seed=seed)

    style_flags = _get_style_config(style)
    use_bold = style_flags["use_bold"]
    use_syntax = style_flags["use_syntax"]
    show_line_numbers = style_flags["show_line_numbers"]
    show_indent_guides = style_flags["show_indent_guides"]

    # Load font
    font = _load_font(font_size, bold=use_bold)
    # Measure actual character width from the font instead of approximating
    bbox = font.getbbox("M")
    char_width = bbox[2] - bbox[0] if bbox else font_size * CHAR_WIDTH_RATIO
    line_px = int(font_size * DEFAULT_LINE_HEIGHT)

    # Calculate margins and text area
    margin = int(width * DEFAULT_MARGIN_RATIO)

    # Line number gutter width (if showing)
    lines = code.splitlines()
    if not lines:
        lines = [""]
    total_lines = len(lines)

    gutter_width = 0
    if show_line_numbers:
        # Width of the widest line number + padding
        max_digits = len(str(total_lines))
        gutter_width = int((max_digits + 2) * char_width)  # +2 for padding

    text_area_x = margin + gutter_width
    text_area_width = width - text_area_x - margin
    chars_per_line = max(1, int(text_area_width / char_width))

    # Calculate how many lines fit per page
    text_area_height = height - 2 * margin
    lines_per_page = max(1, int(text_area_height / line_px))

    # Wrap long lines and track original line numbers + char offset
    # Each entry: (original_line_number, text, char_offset)
    # char_offset is the starting character position within the original line
    wrapped_lines: list[tuple[int, str, int]] = []
    for i, line in enumerate(lines):
        if len(line) <= chars_per_line:
            wrapped_lines.append((i + 1, line, 0))
        else:
            for j in range(0, len(line), chars_per_line):
                chunk = line[j:j + chars_per_line]
                # Only first chunk gets the line number
                wrapped_lines.append((i + 1 if j == 0 else -1, chunk, j))

    # Split into pages
    pages_data = []
    for page_start in range(0, len(wrapped_lines), lines_per_page):
        pages_data.append(wrapped_lines[page_start:page_start + lines_per_page])

    if not pages_data:
        pages_data = [[(1, "")]]

    # Determine theme colors
    is_dark = style_flags.get("theme") == "dark"
    theme = THEME_COLORS_DARK if is_dark else THEME_COLORS_LIGHT
    bg_color = style_flags.get("bg_color", "#FFFFFF")
    text_color = style_flags.get("text_color", "#000000")
    ln_color = IDE_LINE_NUMBER_COLOR_DARK if is_dark else IDE_LINE_NUMBER_COLOR
    ln_bg = IDE_LINE_NUMBER_BG_DARK if is_dark else IDE_LINE_NUMBER_BG
    guide_color = IDE_INDENTATION_GUIDE_COLOR_DARK if is_dark else IDE_INDENTATION_GUIDE_COLOR

    # Tokenize for syntax highlighting (once for the whole code)
    if use_syntax:
        colored_tokens = _tokenize_code(code, language, theme=theme)
    else:
        colored_tokens = None

    # Build per-line color maps if using syntax highlighting
    line_colors = None
    if colored_tokens:
        line_colors = _build_line_color_map(code, colored_tokens)

    # Build lint error lookup
    lint_lookup: dict[int, list[tuple[str, str]]] = {}
    if style == "ide_lint" and lint_errors:
        for line_no, severity, msg in lint_errors:
            lint_lookup.setdefault(line_no, []).append((severity, msg))

    # Render each page
    images = []
    for page_lines in pages_data:
        img = _render_page(
            page_lines=page_lines,
            width=width,
            height=height,
            font=font,
            font_size=font_size,
            char_width=char_width,
            line_px=line_px,
            margin=margin,
            gutter_width=gutter_width,
            text_area_x=text_area_x,
            bg_color=bg_color,
            text_color=text_color,
            show_line_numbers=show_line_numbers,
            show_indent_guides=show_indent_guides,
            line_colors=line_colors,
            use_syntax=use_syntax,
            lint_lookup=lint_lookup,
            line_number_color=ln_color,
            line_number_bg=ln_bg,
            indent_guide_color=guide_color,
        )
        img = _auto_crop(img, bg_color=bg_color)
        images.append(img)

    if downsample_resolution is not None:
        images = downsample_images(images, downsample_resolution)

    if pad_page_to is not None:
        images = [_pad_page(img, pad_page_to, bg_color) for img in images]

    return images


def render_code_tail_fit(
    code: str,
    *,
    token_budget: int,
    model_name: str,
    style: RenderStyle = "plain",
    language: str = "python",
    width: int = DEFAULT_IMAGE_WIDTH,
    height: int = DEFAULT_IMAGE_HEIGHT,
    font_size: int = DEFAULT_FONT_SIZE,
    gemma_max_soft_tokens: int = 280,
    lint_errors: list[tuple[int, str, str]] | None = None,
    strip_indent: bool = False,
    shuffle_indent: bool = False,
    downsample_resolution: int | None = None,
    pad_page_to: tuple[int, int] | None = None,
    per_page_overhead: int = 0,
) -> tuple[list[Image.Image], dict]:
    """Render only the newest pages that fit a real processor-token budget.

    ``per_page_overhead``: framing tokens charged per retained image inside
    the fit decision (2 on both studied encoders — Qwen vision_start/end,
    Gemma boi/eoi) so that ``token_budget`` is the page budget the model's
    prompt actually has to absorb. The reported ``visual_tokens_*`` stay pure
    page tokens; 0 = no framing charged.

    This is the fixed-layout path of the study.  It performs the same pagination,
    global syntax-tokenization, page rasterization, auto-crop, downsample and
    padding operations as :func:`render_code`, but walks page data from the
    tail.  Once the measured kept-page cost reaches the budget, all older page
    data remains layout-only and is never rasterized.

    The final page is always kept.  When even that page exceeds ``token_budget``
    the returned metadata marks ``budget_underfit=True``.  Dropped-page cost is
    estimated from layout-derived, tile-snapped post-crop geometry, while
    every kept page uses its measured post-crop dimensions.
    """
    if (
        not isinstance(token_budget, int)
        or isinstance(token_budget, bool)
        or token_budget <= 0
    ):
        raise ValueError(
            f"token_budget must be a positive integer, got {token_budget!r}"
        )
    if (
        not isinstance(per_page_overhead, int)
        or isinstance(per_page_overhead, bool)
        or per_page_overhead < 0
    ):
        raise ValueError(
            f"per_page_overhead must be a non-negative integer, got {per_page_overhead!r}"
        )
    pad_page_to = _validate_pad_page_to(pad_page_to)
    if strip_indent:
        code = _strip_indent(code)
    elif shuffle_indent:
        import hashlib

        seed = int(hashlib.sha256(code.encode()).hexdigest(), 16) % (2**31)
        code = _shuffle_indent(code, seed=seed)

    style_flags = _get_style_config(style)
    use_bold = style_flags["use_bold"]
    use_syntax = style_flags["use_syntax"]
    show_line_numbers = style_flags["show_line_numbers"]
    show_indent_guides = style_flags["show_indent_guides"]

    font = _load_font(font_size, bold=use_bold)
    bbox = font.getbbox("M")
    char_width = bbox[2] - bbox[0] if bbox else font_size * CHAR_WIDTH_RATIO
    line_px = int(font_size * DEFAULT_LINE_HEIGHT)
    margin = int(width * DEFAULT_MARGIN_RATIO)

    lines = code.splitlines()
    if not lines:
        lines = [""]
    gutter_width = 0
    if show_line_numbers:
        max_digits = len(str(len(lines)))
        gutter_width = int((max_digits + 2) * char_width)
    text_area_x = margin + gutter_width
    text_area_width = width - text_area_x - margin
    chars_per_line = max(1, int(text_area_width / char_width))
    text_area_height = height - 2 * margin
    lines_per_page = max(1, int(text_area_height / line_px))

    wrapped_lines: list[tuple[int, str, int]] = []
    for i, line in enumerate(lines):
        if len(line) <= chars_per_line:
            wrapped_lines.append((i + 1, line, 0))
        else:
            for j in range(0, len(line), chars_per_line):
                chunk = line[j : j + chars_per_line]
                wrapped_lines.append((i + 1 if j == 0 else -1, chunk, j))
    pages_data = [
        wrapped_lines[start : start + lines_per_page]
        for start in range(0, len(wrapped_lines), lines_per_page)
    ]
    if not pages_data:
        pages_data = [[(1, "")]]

    is_dark = style_flags.get("theme") == "dark"
    theme = THEME_COLORS_DARK if is_dark else THEME_COLORS_LIGHT
    bg_color = style_flags.get("bg_color", "#FFFFFF")
    text_color = style_flags.get("text_color", "#000000")
    ln_color = IDE_LINE_NUMBER_COLOR_DARK if is_dark else IDE_LINE_NUMBER_COLOR
    ln_bg = IDE_LINE_NUMBER_BG_DARK if is_dark else IDE_LINE_NUMBER_BG
    guide_color = (
        IDE_INDENTATION_GUIDE_COLOR_DARK
        if is_dark
        else IDE_INDENTATION_GUIDE_COLOR
    )
    colored_tokens = _tokenize_code(code, language, theme=theme) if use_syntax else None
    line_colors = _build_line_color_map(code, colored_tokens) if colored_tokens else None
    lint_lookup: dict[int, list[tuple[str, str]]] = {}
    if style == "ide_lint" and lint_errors:
        for line_no, severity, msg in lint_errors:
            lint_lookup.setdefault(line_no, []).append((severity, msg))

    @functools.lru_cache(maxsize=4096)
    def _font_ink_bbox(
        text: str,
        fill_color: str,
        start_frac: float = 0.0,
    ) -> tuple[int, int, int, int] | None:
        """Return painted glyph bounds, excluding advance-only whitespace.

        ``font.getbbox`` includes advance width and therefore crosses crop
        tiles for punctuation, narrow glyphs, and trailing spaces that paint
        no pixel.  The FreeType mask is still a small per-run layout artifact,
        not a page canvas, and exposes the actual ink bounds needed to choose K
        without rasterizing a discarded page.

        ``start_frac`` is the fractional part of the run's x position (syntax
        segments start at the font's real advance, a multiple of 1/64 px): the
        run is drawn through the same ``ImageDraw.text`` call as ``_render_page``
        at that sub-pixel start, and the bounds are relative to the integer
        part of the position.
        """
        font_bbox = font.getbbox(text)
        if font_bbox is None:
            return None
        box_width = font_bbox[2] - font_bbox[0]
        box_height = font_bbox[3] - font_bbox[1]
        if box_width <= 0 or box_height <= 0:
            return None
        if start_frac:
            # Sub-pixel start: draw on a padded scratch canvas at a positive
            # origin (int() and modf() agree for positive coordinates) and
            # report the ink bounds relative to that integer origin.
            pad = 8
            run_image = Image.new(
                "RGB",
                (pad + max(font_bbox[2], 0) + pad + 2, pad + max(font_bbox[3], 0) + pad),
                bg_color,
            )
            ImageDraw.Draw(run_image).text(
                (pad + start_frac, pad), text, fill=fill_color, font=font
            )
            bg_gray_frac = int(
                0.299 * int(bg_color[1:3], 16)
                + 0.587 * int(bg_color[3:5], 16)
                + 0.114 * int(bg_color[5:7], 16)
            )
            frac_bbox = (
                run_image.convert("L")
                .point([255 if abs(v - bg_gray_frac) > 5 else 0 for v in range(256)])
                .getbbox()
            )
            if frac_bbox is None:
                return None
            return (
                frac_bbox[0] - pad,
                frac_bbox[1] - pad,
                frac_bbox[2] - pad,
                frac_bbox[3] - pad,
            )
        run_image = Image.new("RGB", (box_width, box_height), bg_color)
        ImageDraw.Draw(run_image).text(
            (-font_bbox[0], -font_bbox[1]),
            text,
            fill=fill_color,
            font=font,
        )
        bg_r = int(bg_color[1:3], 16)
        bg_g = int(bg_color[3:5], 16)
        bg_b = int(bg_color[5:7], 16)
        bg_gray = int(0.299 * bg_r + 0.587 * bg_g + 0.114 * bg_b)
        threshold_table = [
            255 if abs(value - bg_gray) > 5 else 0 for value in range(256)
        ]
        mask_bbox = run_image.convert("L").point(threshold_table).getbbox()
        if mask_bbox is None:
            return None
        return (
            font_bbox[0] + mask_bbox[0],
            font_bbox[1] + mask_bbox[1],
            font_bbox[0] + mask_bbox[2],
            font_bbox[1] + mask_bbox[3],
        )

    @functools.lru_cache(maxsize=4096)
    def _wavy_ink_extent(text_len: int) -> tuple[int, int]:
        """Return underline right/bottom extents relative to its start/base."""
        amplitude = max(1, font_size // 10)
        wavelength = max(4, font_size // 4)
        line_width = max(1, font_size // 20)
        x_start = text_area_x
        x_end = text_area_x + int(text_len * char_width)
        pad = amplitude + line_width + 2
        baseline = pad + amplitude
        canvas = Image.new(
            "L",
            (max(1, x_end - x_start) + 2 * pad, 4 * pad + 1),
            0,
        )
        points = [
            (
                pad + x - x_start,
                baseline
                + int(amplitude * math.sin(2 * math.pi * x / wavelength)),
            )
            for x in range(x_start, x_end)
        ]
        if len(points) >= 2:
            ImageDraw.Draw(canvas).line(points, fill=255, width=line_width)
        ink_bbox = canvas.getbbox()
        if ink_bbox is None:
            return 0, 0
        return ink_bbox[2] - pad, ink_bbox[3] - baseline

    def _estimated_final_size(
        page_lines: list[tuple[int, str, int]],
        *,
        exact_ink: bool = False,
    ) -> tuple[int, int]:
        """Predict the tile-snapped post-crop size without a page canvas.

        ``_auto_crop`` only needs the rightmost/bottommost content pixel and
        then snaps both axes to 112px. The layout already contains every
        position needed to recover those tile bounds, so dropped pages can be
        costed in O(lines) without rasterization. Fit-K itself still uses
        measured survivor pages.
        """
        right_exclusive = 0
        bottom_exclusive = 0
        has_content = False

        if show_line_numbers and gutter_width > 0:
            has_content = True
            right_exclusive = margin + gutter_width
            bottom_exclusive = margin + len(page_lines) * line_px + 1

        last_orig_line_no = 0
        for row_index, (orig_line_no, text, char_offset) in enumerate(page_lines):
            y = margin + row_index * line_px
            if orig_line_no > 0:
                last_orig_line_no = orig_line_no
            if y + line_px > height - margin:
                break
            if show_line_numbers and orig_line_no > 0:
                label = str(orig_line_no)
                label_x = margin + gutter_width - int((len(label) + 1) * char_width)
                label_bbox = (
                    _font_ink_bbox(label, ln_color)
                    if exact_ink
                    else font.getbbox(label)
                )
                if label_bbox:
                    has_content = True
                    right_exclusive = max(
                        right_exclusive, label_x + label_bbox[2]
                    )
                    bottom_exclusive = max(
                        bottom_exclusive, y + label_bbox[3]
                    )
            if show_indent_guides and text:
                indent_level = len(text) - len(text.lstrip(" "))
                if indent_level >= 4:
                    has_content = True
                    guide_x = text_area_x + int((indent_level // 4) * 4 * char_width)
                    right_exclusive = max(right_exclusive, guide_x + 1)
                    bottom_exclusive = max(bottom_exclusive, y + line_px + 1)
            if text:
                # Mirror _render_page's syntax-segment positioning.  Measuring
                # the whole source line is not equivalent for proportional
                # glyph metrics, and a tile-boundary error here could change K.
                text_runs: list[tuple[float, str, str]] = []
                if use_syntax and line_colors is not None:
                    line_idx = last_orig_line_no - 1 if last_orig_line_no > 0 else -1
                    segments = line_colors.get(line_idx, [])
                    if segments:
                        sliced = _slice_color_segments(
                            segments,
                            char_offset,
                            char_offset + len(text),
                        )
                        # Same positions as _render_page: each colour segment
                        # starts at the font's real advance of the text drawn
                        # so far on this line (not an integer char_width grid).
                        drawn = ""
                        for color, segment_text in sliced:
                            text_runs.append(
                                (text_area_x + font.getlength(drawn), segment_text, color)
                            )
                            drawn += segment_text
                    else:
                        text_runs.append((text_area_x, text, text_color))
                else:
                    text_runs.append((text_area_x, text, text_color))

                for run_x, text_run, run_color in text_runs:
                    # ImageDraw.text splits a float x into int(x) and a
                    # sub-pixel start; mirror that split here.
                    text_x = int(run_x)
                    start_frac = math.modf(run_x)[0]
                    if exact_ink:
                        text_bbox = (
                            _font_ink_bbox(text_run, run_color, start_frac)
                            if start_frac
                            else _font_ink_bbox(text_run, run_color)
                        )
                    else:
                        # Cheap dropped-page total estimator. Only U+0020 is
                        # known to be advance-only for the shipped font; tabs
                        # and Unicode whitespace may paint fallback glyphs.
                        visible_text_run = text_run.rstrip(" ")
                        text_bbox = (
                            font.getbbox(visible_text_run)
                            if visible_text_run
                            else None
                        )
                    if text_bbox:
                        has_content = True
                        right_exclusive = max(
                            right_exclusive, text_x + text_bbox[2]
                        )
                        bottom_exclusive = max(
                            bottom_exclusive, y + text_bbox[3]
                        )
            if orig_line_no in lint_lookup:
                text_len = len(text) if text else 1
                has_content = True
                if exact_ink:
                    underline_right, underline_bottom = _wavy_ink_extent(text_len)
                    right_exclusive = max(
                        right_exclusive,
                        text_area_x + underline_right,
                    )
                    bottom_exclusive = max(
                        bottom_exclusive,
                        y + line_px - 2 + underline_bottom,
                    )
                else:
                    amplitude = max(1, font_size // 10)
                    right_exclusive = max(
                        right_exclusive,
                        text_area_x + int(text_len * char_width),
                    )
                    bottom_exclusive = max(
                        bottom_exclusive, y + line_px - 1 + amplitude,
                    )

        if not has_content:
            crop_w, crop_h = width, height
        else:
            # Mirror _auto_crop's ordering exactly: clamp the pixel bound,
            # then snap it. For a non-tile canvas (e.g. width=223), Pillow may
            # therefore crop to 224 and pad one axis when the other axis really
            # shrinks. If neither axis shrinks, _auto_crop returns the original.
            crop_w = min(width, right_exclusive + 4)
            crop_h = min(height, bottom_exclusive + 4)
            crop_w = math.ceil(crop_w / QWEN_TILE_SIZE) * QWEN_TILE_SIZE
            crop_h = math.ceil(crop_h / QWEN_TILE_SIZE) * QWEN_TILE_SIZE
            if crop_w >= width and crop_h >= height:
                crop_w, crop_h = width, height

        if downsample_resolution is not None:
            scale = downsample_resolution / 1120
            crop_w = max(
                QWEN_TILE_SIZE,
                round(crop_w * scale / QWEN_TILE_SIZE) * QWEN_TILE_SIZE,
            )
            crop_h = max(
                QWEN_TILE_SIZE,
                round(crop_h * scale / QWEN_TILE_SIZE) * QWEN_TILE_SIZE,
            )
        if pad_page_to is not None:
            crop_w, crop_h = pad_page_to
        return crop_w, crop_h

    def _rasterize(
        page_lines: list[tuple[int, str, int]],
    ) -> tuple[Image.Image, float]:
        image = _render_page(
            page_lines=page_lines,
            width=width,
            height=height,
            font=font,
            font_size=font_size,
            char_width=char_width,
            line_px=line_px,
            margin=margin,
            gutter_width=gutter_width,
            text_area_x=text_area_x,
            bg_color=bg_color,
            text_color=text_color,
            show_line_numbers=show_line_numbers,
            show_indent_guides=show_indent_guides,
            line_colors=line_colors,
            use_syntax=use_syntax,
            lint_lookup=lint_lookup,
            line_number_color=ln_color,
            line_number_bg=ln_bg,
            indent_guide_color=guide_color,
        )
        image = _auto_crop(image, bg_color=bg_color)
        pre_downsample_height = image.height
        if downsample_resolution is not None:
            image = downsample_images([image], downsample_resolution)[0]
        raster_scale_y = image.height / pre_downsample_height
        if pad_page_to is not None:
            image = _pad_page(image, pad_page_to, bg_color)
        return image, raster_scale_y

    pages_total = len(pages_data)

    def _layout_page_cost(page_index: int, *, exact_ink: bool) -> int:
        page_width, page_height = _estimated_final_size(
            pages_data[page_index],
            exact_ink=exact_ink,
        )
        return page_visual_tokens(
            page_width,
            page_height,
            model_name,
            gemma_max_soft_tokens=gemma_max_soft_tokens,
        )

    # Select K from exact per-run ink masks, without allocating a page canvas.
    # Only the tail and its first rejected boundary need this more expensive
    # geometry. All earlier dropped pages use the cheap bbox total estimator.
    kept_start = pages_total - 1
    exact_kept_costs = [_layout_page_cost(kept_start, exact_ink=True)]
    # estimated_kept_tokens tracks page tokens + per-page framing overhead for
    # the fit decision only; reported visual_tokens_* exclude the overhead.
    estimated_kept_tokens = exact_kept_costs[0] + per_page_overhead
    budget_underfit = estimated_kept_tokens > token_budget
    if not budget_underfit:
        for page_index in range(pages_total - 2, -1, -1):
            page_cost = _layout_page_cost(page_index, exact_ink=True)
            candidate_total = estimated_kept_tokens + page_cost + per_page_overhead
            if candidate_total > token_budget:
                break
            kept_start = page_index
            estimated_kept_tokens = candidate_total
            exact_kept_costs.insert(0, page_cost)

    rasterized_pages = [_rasterize(page) for page in pages_data[kept_start:]]
    images = [image for image, _scale_y in rasterized_pages]
    page_raster_scale_y = [scale_y for _image, scale_y in rasterized_pages]
    measured_costs = [
        page_visual_tokens(
            image.width,
            image.height,
            model_name,
            gemma_max_soft_tokens=gemma_max_soft_tokens,
        )
        for image in images
    ]
    visual_tokens_kept = sum(measured_costs)
    if measured_costs != exact_kept_costs:
        raise RuntimeError(
            "layout-derived crop geometry drifted from rendered survivor "
            "geometry; refusing an unverified fit-K decision"
        )

    pages_rasterized = len(images)
    pages_kept = len(images)
    pages_dropped = pages_total - pages_kept
    if pages_dropped:
        estimated_dropped_tokens = sum(
            _layout_page_cost(page_index, exact_ink=False)
            for page_index in range(pages_dropped)
        )
        visual_tokens_total = visual_tokens_kept + estimated_dropped_tokens
        estimator = "bbox_dropped+inkmask_kept"
    else:
        visual_tokens_total = visual_tokens_kept
        estimator = "measured"

    info = {
        "pages_total": pages_total,
        "pages_kept": pages_kept,
        "pages_dropped_front": pages_dropped,
        "budget_underfit": budget_underfit,
        "visual_tokens_total": visual_tokens_total,
        "visual_tokens_total_estimator": estimator,
        "visual_tokens_kept": visual_tokens_kept,
        "pages_rasterized": pages_rasterized,
    }
    if per_page_overhead:
        info["per_page_overhead"] = per_page_overhead
    return images, {
        **info,
        # Private pipeline hand-off: rendered characters are scaled before the
        # processor when downsample_resolution is active.  image.py consumes
        # this and never serializes the private key into image_info.
        "_page_raster_scale_y": page_raster_scale_y,
    }


def _build_line_color_map(
    code: str, colored_tokens: list[tuple[str, str]]
) -> dict[int, list[tuple[str, str]]]:
    """
    Build a per-line map of (color, text) segments from Pygments tokens.

    Returns: {line_number_0based: [(color, text), ...]}
    """
    line_map: dict[int, list[tuple[str, str]]] = {}
    current_line = 0

    for color, text in colored_tokens:
        # A single token can span multiple lines (e.g., multi-line strings)
        parts = text.split("\n")
        for i, part in enumerate(parts):
            if i > 0:
                current_line += 1
            if part:  # skip empty splits
                line_map.setdefault(current_line, []).append((color, part))

    return line_map


def _slice_color_segments(
    segments: list[tuple[str, str]], start: int, end: int
) -> list[tuple[str, str]]:
    """
    Slice color segments to a character range [start, end).

    Given segments like [("#0000FF", "def "), ("#000000", "foo():")]
    and range [4, 10], returns [("#000000", "foo():")].
    """
    result = []
    pos = 0
    for color, text in segments:
        seg_start = pos
        seg_end = pos + len(text)
        # Overlap with [start, end)?
        overlap_start = max(seg_start, start)
        overlap_end = min(seg_end, end)
        if overlap_start < overlap_end:
            sliced_text = text[overlap_start - seg_start : overlap_end - seg_start]
            result.append((color, sliced_text))
        pos = seg_end
        if pos >= end:
            break
    return result


def _render_page(
    page_lines: list[tuple[int, str, int]],
    width: int,
    height: int,
    font: ImageFont.FreeTypeFont,
    font_size: int,
    char_width: float,
    line_px: int,
    margin: int,
    gutter_width: int,
    text_area_x: int,
    bg_color: str,
    text_color: str,
    show_line_numbers: bool,
    show_indent_guides: bool,
    line_colors: dict[int, list[tuple[str, str]]] | None,
    use_syntax: bool,
    lint_lookup: dict[int, list[tuple[str, str]]],
    line_number_color: str = IDE_LINE_NUMBER_COLOR,
    line_number_bg: str = IDE_LINE_NUMBER_BG,
    indent_guide_color: str = IDE_INDENTATION_GUIDE_COLOR,
) -> Image.Image:
    """Render a single page of code."""
    img = Image.new("RGB", (width, height), bg_color)
    draw = ImageDraw.Draw(img)

    # Draw gutter background — only to the height of actual content,
    # not the full page. This allows auto-crop to trim empty bottom space.
    if show_line_numbers and gutter_width > 0:
        content_height = margin + len(page_lines) * line_px
        draw.rectangle(
            [0, 0, margin + gutter_width - 1, content_height],
            fill=line_number_bg,
        )

    last_orig_line_no = 0  # tracks last real line number for continuation lines
    for row_idx, (orig_line_no, text, char_offset) in enumerate(page_lines):
        y = margin + row_idx * line_px

        # Track the original line number (continuation lines use -1)
        if orig_line_no > 0:
            last_orig_line_no = orig_line_no

        # Skip if below the page
        if y + line_px > height - margin:
            break

        # --- Line numbers ---
        if show_line_numbers and orig_line_no > 0:
            line_num_str = str(orig_line_no)
            # Right-align in gutter
            num_x = margin + gutter_width - int((len(line_num_str) + 1) * char_width)
            draw.text((num_x, y), line_num_str, fill=line_number_color, font=font)

        # --- Indentation guides ---
        if show_indent_guides and text:
            indent_level = len(text) - len(text.lstrip(" "))
            tab_width = 4  # spaces per indent level
            for guide_col in range(tab_width, indent_level + 1, tab_width):
                guide_x = text_area_x + int(guide_col * char_width)
                draw.line(
                    [(guide_x, y), (guide_x, y + line_px)],
                    fill=indent_guide_color,
                    width=1,
                )

        # --- Code text ---
        if use_syntax and line_colors is not None:
            # Render with per-character colors from syntax highlighting.
            # We need to slice the color segments to match this wrapped chunk's
            # character range [char_offset, char_offset + len(text)).
            # For continuation lines (orig_line_no == -1), use the last real
            # line number to look up the correct color segments.
            line_idx = last_orig_line_no - 1 if last_orig_line_no > 0 else -1
            segments = line_colors.get(line_idx, [])
            if segments:
                # Slice segments to the character range of this chunk
                sliced = _slice_color_segments(segments, char_offset, char_offset + len(text))
                # Position every colour segment at the font's real advance of
                # the text already drawn on this line, so glyphs sit exactly
                # where a single whole-line draw puts them. (An integer
                # char_width grid — the ink width of "M", 10 px at 16 px font
                # vs a 9.64 px advance — drifted +0.36 px per preceding
                # character and opened visible gaps at token boundaries.)
                drawn = ""
                for color, seg_text in sliced:
                    draw.text(
                        (text_area_x + font.getlength(drawn), y),
                        seg_text,
                        fill=color,
                        font=font,
                    )
                    drawn += seg_text
            else:
                draw.text((text_area_x, y), text, fill=text_color, font=font)
        else:
            draw.text((text_area_x, y), text, fill=text_color, font=font)

        # --- Lint annotations ---
        if orig_line_no in lint_lookup:
            for severity, _msg in lint_lookup[orig_line_no]:
                underline_color = (
                    IDE_LINT_ERROR_COLOR if severity == "error"
                    else IDE_LINT_WARNING_COLOR
                )
                # Draw wavy underline across the text.
                # Scale amplitude relative to font size so it doesn't overflow
                # into adjacent lines at small sizes (e.g., font_size=8).
                amp = max(1, font_size // 10)
                wl = max(4, font_size // 4)
                w = max(1, font_size // 20)
                text_len = len(text) if text else 1
                _draw_wavy_underline(
                    draw,
                    x_start=text_area_x,
                    x_end=text_area_x + int(text_len * char_width),
                    y=y + line_px - 2,
                    color=underline_color,
                    amplitude=amp,
                    wavelength=wl,
                    width=w,
                )

    return img


def _auto_crop(img: Image.Image, bg_color: str = "#FFFFFF", margin: int = 4) -> Image.Image:
    """
    Crop whitespace from the right and bottom edges of a rendered code image.

    Scans for the rightmost and bottommost non-background pixels, then crops
    the image to that bounding box plus a small margin. The left and top edges
    are never cropped (line numbers and first line must stay at position 0).

    This saves visual tokens by removing empty space that the VLM would
    otherwise process. Following Glyph's approach (auto-crop-width and
    auto-crop-last-page).

    Args:
        img: The rendered page image.
        bg_color: Background color hex string (e.g., "#FFFFFF").
        margin: Pixels of padding to keep around content.

    Returns:
        Cropped image (or original if no content found).
    """
    import numpy as np

    gray = np.array(img.convert("L"))

    # Determine background gray value from the bg_color
    # Parse hex color to grayscale
    r = int(bg_color[1:3], 16)
    g = int(bg_color[3:5], 16)
    b = int(bg_color[5:7], 16)
    bg_gray = int(0.299 * r + 0.587 * g + 0.114 * b)

    # Find pixels that differ from background
    tolerance = 5
    mask = np.abs(gray.astype(int) - bg_gray) > tolerance

    if not mask.any():
        return img  # All background — return as-is

    # Find content bounds
    rows = np.where(mask.any(axis=1))[0]
    cols = np.where(mask.any(axis=0))[0]

    if rows.size == 0 or cols.size == 0:
        return img

    # Crop right and bottom, keep left and top
    right = min(img.width, cols[-1] + 1 + margin)
    bottom = min(img.height, rows[-1] + 1 + margin)

    # Snap to multiples of QWEN_TILE_SIZE for clean tiling (no partial tile waste)
    right = math.ceil(right / QWEN_TILE_SIZE) * QWEN_TILE_SIZE
    bottom = math.ceil(bottom / QWEN_TILE_SIZE) * QWEN_TILE_SIZE

    # Don't crop if it would barely change anything
    if right >= img.width and bottom >= img.height:
        return img

    return img.crop((0, 0, right, bottom))


def _validate_pad_page_to(pad_page_to) -> tuple[int, int] | None:
    """Validate a pad_page_to value; returns a (width, height) int tuple or None.

    Accepts a 2-item tuple/list of positive ints (fire literal-evals CLI values
    into tuples/lists). Anything else — wrong arity, floats, strings, bools,
    non-positive dims — raises ValueError. The CLI string form ("896x1344") is
    parsed upstream in main.py, never here.
    """
    if pad_page_to is None:
        return None
    if (
        not isinstance(pad_page_to, (tuple, list))
        or len(pad_page_to) != 2
        or not all(
            isinstance(v, int) and not isinstance(v, bool) and v > 0
            for v in pad_page_to
        )
    ):
        raise ValueError(
            f"pad_page_to must be a (width, height) pair of positive ints, "
            f"got {pad_page_to!r}"
        )
    return (pad_page_to[0], pad_page_to[1])


def _pad_page(
    img: Image.Image, size: tuple[int, int], bg_color: str = "#FFFFFF"
) -> Image.Image:
    """Pad a page to exactly `size` with `bg_color`, content top-left anchored.

    Applied AFTER _auto_crop: auto-crop trims right/bottom
    whitespace, so pasting the cropped page at (0, 0) restores the exact page
    geometry without moving any content pixel. Raises ValueError if the page
    is larger than the target in either dimension — padding must never crop
    or shrink content.
    """
    target_w, target_h = size
    if img.width > target_w or img.height > target_h:
        raise ValueError(
            f"pad_page_to={size} is smaller than a rendered page "
            f"({img.width}x{img.height}) — padding never crops content; "
            f"use render dims <= the pad target."
        )
    if (img.width, img.height) == (target_w, target_h):
        return img
    canvas = Image.new("RGB", (target_w, target_h), bg_color)
    canvas.paste(img, (0, 0))
    return canvas


def _draw_wavy_underline(
    draw: ImageDraw.ImageDraw,
    x_start: int,
    x_end: int,
    y: int,
    color: str,
    amplitude: int = 4,
    wavelength: int = 10,
    width: int = 2,
) -> None:
    """Draw a wavy underline (like IDE error/warning squiggles)."""
    points = []
    for x in range(x_start, x_end):
        offset = int(amplitude * math.sin(2 * math.pi * x / wavelength))
        points.append((x, y + offset))
    if len(points) >= 2:
        draw.line(points, fill=color, width=width)


# =============================================================================
# Layout optimization — find (resolution, font_size) to hit a token budget
# =============================================================================

# Minimum font size to keep code readable before downsampling
MIN_FONT_SIZE = 8
MAX_FONT_SIZE = 60

def _estimate_pages(
    code: str,
    width: int,
    height: int,
    font_size: int,
    show_line_numbers: bool = False,
) -> int:
    """Estimate how many pages the code will need at the given layout, without rendering."""
    font = _load_font(font_size)
    bbox = font.getbbox("M")
    char_width = bbox[2] - bbox[0] if bbox else font_size * CHAR_WIDTH_RATIO
    line_px = int(font_size * DEFAULT_LINE_HEIGHT)
    margin = int(width * DEFAULT_MARGIN_RATIO)

    lines = code.splitlines()
    if not lines:
        return 1

    # Gutter width
    gutter_width = 0
    if show_line_numbers:
        max_digits = len(str(len(lines)))
        gutter_width = int((max_digits + 2) * char_width)

    text_area_width = width - margin - gutter_width - margin
    chars_per_line = max(1, int(text_area_width / char_width))

    text_area_height = height - 2 * margin
    lines_per_page = max(1, int(text_area_height / line_px))

    # Count wrapped lines
    total_wrapped = 0
    for line in lines:
        if len(line) <= chars_per_line:
            total_wrapped += 1
        else:
            total_wrapped += math.ceil(len(line) / chars_per_line)

    return max(1, math.ceil(total_wrapped / lines_per_page))


def _optimize_layout(
    code: str,
    target_tokens: int,
    show_line_numbers: bool = False,
) -> tuple[int, int, int]:
    """
    Find the best (width, height, font_size) to render code within a token budget.

    Strategy: try square resolutions from large to small, with font sizes from
    large to small. Pick the combination that:
      1. Fits within the token budget
      2. Maximizes readability (larger font preferred)
      3. Minimizes wasted space (fewer pages preferred)

    Returns:
        (width, height, font_size)
    """
    if target_tokens <= 0:
        return (14, 14, MIN_FONT_SIZE)

    # Generate candidate resolutions: multiples of 112 (tile-aligned)
    candidate_resolutions = []
    for tiles_per_side in range(1, 41):  # 112 to 4480
        size = tiles_per_side * QWEN_TILE_SIZE
        candidate_resolutions.append(size)

    # Two-step strategy:
    #   Step 1: Find the best layout that fits within the token budget.
    #   Step 2: If nothing fits (code too long), find the layout that
    #           minimizes total tokens while keeping pages reasonable.
    #
    # Key insight: many small pages (32 x 112px) is far worse than one
    # larger page, even if the larger page exceeds the budget more in
    # absolute tokens. VLMs handle 1-4 images much better than 30+.

    # Max pages for the optimizer path. Up to 17 pages were verified to work
    # with the fixed layout. This limit only applies to the optimizer; fixed
    # layout bypasses it.
    MAX_OVERFLOW_PAGES = 8

    # Step 1: find best within-budget layout
    best = None
    best_score = -1

    for res in candidate_resolutions:
        tiles = math.ceil(res / QWEN_TILE_SIZE) ** 2
        tokens_per_page = tiles * QWEN_TOKENS_PER_TILE
        max_pages = min(MAX_OVERFLOW_PAGES, max(1, target_tokens // tokens_per_page))

        for fs in range(MAX_FONT_SIZE, MIN_FONT_SIZE - 1, -2):
            num_pages = _estimate_pages(code, res, res, fs, show_line_numbers)
            total_tokens = num_pages * tokens_per_page

            if num_pages <= max_pages and total_tokens <= target_tokens:
                utilization = total_tokens / target_tokens
                score = fs * 10 + utilization * 200 - num_pages * 5
                if score > best_score:
                    best_score = score
                    best = (res, res, fs)
                break  # largest font at this res — move to next res

    if best is not None:
        return best

    # Step 2: nothing fits within budget. The code is too long for the
    # target token count. Find the layout with fewest total tokens while
    # keeping page count reasonable.
    best_overflow = None
    best_overflow_tokens = float("inf")

    for res in candidate_resolutions:
        tiles = math.ceil(res / QWEN_TILE_SIZE) ** 2
        tokens_per_page = tiles * QWEN_TOKENS_PER_TILE

        # At each resolution, use smallest font to minimize pages
        fs = MIN_FONT_SIZE
        num_pages = _estimate_pages(code, res, res, fs, show_line_numbers)

        if num_pages > MAX_OVERFLOW_PAGES:
            continue  # too many pages — skip

        total_tokens = num_pages * tokens_per_page
        if total_tokens < best_overflow_tokens:
            best_overflow_tokens = total_tokens
            best_overflow = (res, res, fs)

    if best_overflow is not None:
        return best_overflow

    # Nothing fits in 8 pages — allow more pages but pick minimum tokens
    best_any = None
    best_any_tokens = float("inf")
    for res in candidate_resolutions:
        tiles = math.ceil(res / QWEN_TILE_SIZE) ** 2
        tokens_per_page = tiles * QWEN_TOKENS_PER_TILE
        fs = MIN_FONT_SIZE
        num_pages = _estimate_pages(code, res, res, fs, show_line_numbers)
        total_tokens = num_pages * tokens_per_page
        if total_tokens < best_any_tokens:
            best_any_tokens = total_tokens
            best_any = (res, res, fs)

    return best_any or (QWEN_TILE_SIZE, QWEN_TILE_SIZE, MIN_FONT_SIZE)


# =============================================================================
# Downsample (proportional resize for fixed-layout experiments)
# =============================================================================

def downsample_images(
    images: list[Image.Image],
    target_resolution: int,
) -> list[Image.Image]:
    """Downsample rendered pages proportionally, snapping to the Qwen tile grid.

    Each page is scaled so that its larger dimension maps to approximately
    ``target_resolution`` pixels (relative to a 1120 px base), then both
    dimensions are snapped to multiples of ``QWEN_TILE_SIZE`` (112 px).

    Args:
        images: List of rendered page images (possibly non-square from auto-crop).
        target_resolution: Target for the base dimension (1120 → target_resolution).
            E.g. 560 means "half resolution", 336 means "~1/3".

    Returns:
        List of downsampled PIL Images, each dimension on the tile grid.
    """
    base = 1120
    scale = target_resolution / base
    result = []
    for img in images:
        w, h = img.size
        new_w = max(QWEN_TILE_SIZE, round(w * scale / QWEN_TILE_SIZE) * QWEN_TILE_SIZE)
        new_h = max(QWEN_TILE_SIZE, round(h * scale / QWEN_TILE_SIZE) * QWEN_TILE_SIZE)
        if new_w == w and new_h == h:
            result.append(img)
        else:
            result.append(img.resize((new_w, new_h), Image.LANCZOS))
    return result


# =============================================================================
# Compression (resolution downsampling for already-rendered images)
# =============================================================================

def compress_images(
    images: list[Image.Image],
    text_token_count: int,
    compression_ratio: float = 1.0,
) -> list[Image.Image]:
    """
    Compress already-rendered images by downsampling.

    Prefer using render_and_compress() instead, which renders at the optimal
    resolution from the start. This function is a fallback for when you already
    have rendered images and need to resize them.

    Strategy:
      1. Try resizing each page independently to fit (budget / num_pages) tokens.
      2. If that exceeds the budget (too many pages at the per-page floor of 1 tile),
         concatenate pages vertically into fewer combined images and resize to fit.

    Args:
        images: List of rendered code images.
        text_token_count: Number of text tokens in the original code.
        compression_ratio: Compression ratio (1, 2, 4, 8, etc.).

    Returns:
        List of resized images.
    """
    if compression_ratio <= 0:
        raise ValueError("compression_ratio must be positive")

    target_tokens = int(text_token_count / compression_ratio)

    if not images:
        return images

    # Minimum tokens per image is 1 tile (QWEN_TOKENS_PER_TILE).
    # If num_pages * min_per_page > target, we need to reduce page count
    # by concatenating pages vertically before resizing.
    min_per_page = QWEN_TOKENS_PER_TILE  # 16 tokens (one 112x112 tile)
    max_pages = max(1, target_tokens // min_per_page)

    if len(images) > max_pages:
        # Concatenate pages into max_pages combined images
        images = _concat_pages(images, max_pages)

    # Distribute token budget across (possibly combined) pages
    per_image_tokens = max(1, target_tokens // len(images))
    target_res = find_resolution_for_tokens(per_image_tokens)

    compressed = []
    for img in images:
        # Only downscale, never upscale
        new_w = min(img.width, target_res[0])
        new_h = min(img.height, target_res[1])
        if (new_w, new_h) != img.size:
            resized = img.resize((new_w, new_h), Image.LANCZOS)
        else:
            resized = img.copy()
        compressed.append(resized)

    return compressed


def _concat_pages(images: list[Image.Image], max_pages: int) -> list[Image.Image]:
    """Concatenate images vertically into at most max_pages combined images."""
    if max_pages >= len(images):
        return images

    # Distribute images as evenly as possible across max_pages groups
    n = len(images)
    groups = []
    base_size = n // max_pages
    remainder = n % max_pages
    idx = 0
    for i in range(max_pages):
        count = base_size + (1 if i < remainder else 0)
        groups.append(images[idx:idx + count])
        idx += count

    combined = []
    for group in groups:
        if len(group) == 1:
            combined.append(group[0])
            continue
        # Concatenate vertically: use widest image width, sum heights
        max_w = max(img.width for img in group)
        total_h = sum(img.height for img in group)
        canvas = Image.new("RGB", (max_w, total_h), (255, 255, 255))
        y = 0
        for img in group:
            canvas.paste(img, (0, y))
            y += img.height
        combined.append(canvas)

    return combined


# =============================================================================
# High-level API
# =============================================================================

def render_and_compress(
    code: str,
    style: RenderStyle = "plain",
    language: str = "python",
    text_token_count: int | None = None,
    compression_ratio: float = 1.0,
    lint_errors: list[tuple[int, str, str]] | None = None,
    max_visual_tokens: int | None = None,
    fixed_image_width: int | None = None,
    fixed_image_height: int | None = None,
    fixed_font_size: int | None = None,
    strip_indent: bool = False,
    shuffle_indent: bool = False,
    downsample_resolution: int | None = None,
    pad_page_to: tuple[int, int] | None = None,
) -> list[Image.Image]:
    """
    Render code as images, then compress to a target token budget.

    This is the main entry point for the image compression pipeline.

    The approach (following CodeOCR):
      1. First, render at a resolution optimized for the code length —
         large enough to be readable, small enough to minimize pages.
      2. Then, downsample all pages to hit the target visual token count.

    At compression_ratio=1, visual tokens ≈ text tokens (same cost).
    At compression_ratio=4, visual tokens ≈ text_tokens / 4.

    **Fixed layout mode**: When fixed_image_width, fixed_image_height, and
    fixed_font_size are all set, the layout optimizer is bypassed. Code is
    rendered at exactly those dimensions and font size, producing as many
    pages as needed. This ensures every example has identical page dimensions,
    which is important for controlled style comparison experiments.

    Args:
        code: Source code string.
        style: Rendering style.
        language: Programming language.
        text_token_count: Number of text tokens in the code (for compression ratio calc).
                         If None, renders at default full resolution (no compression).
        compression_ratio: 1.0 = same tokens as text, 4.0 = 1/4 the tokens.
        lint_errors: For "ide_lint" style. List of (line_no, severity, message).
        max_visual_tokens: Fixed visual token budget. When set, overrides the
                          text_token_count / compression_ratio calculation.
        fixed_image_width: Fixed page width in pixels. Bypasses layout optimizer.
        fixed_image_height: Fixed page height in pixels. Bypasses layout optimizer.
        fixed_font_size: Fixed font size in pixels. Bypasses layout optimizer.
        pad_page_to: If set to (W, H), pad every rendered page (incl. the
                     auto-cropped last page) to exactly WxH with the style
                     background, top-left anchored. Forwarded to
                     render_code on every branch. Note: on the over-budget
                     compress_images path (step 2) pages are resized AFTER
                     padding, so exact dims are only guaranteed when the render
                     fits the budget — use fixed layout mode (no budget) for
                     a fixed page geometry. Default None =
                     unchanged behavior.

    Returns:
        List of PIL Images (one per page).
    """
    # Fixed layout mode: bypass optimizer entirely
    if fixed_image_width is not None and fixed_image_height is not None and fixed_font_size is not None:
        return render_code(
            code, style=style, language=language,
            width=fixed_image_width, height=fixed_image_height,
            font_size=fixed_font_size, lint_errors=lint_errors,
            strip_indent=strip_indent, shuffle_indent=shuffle_indent,
            downsample_resolution=downsample_resolution,
            pad_page_to=pad_page_to,
        )

    if max_visual_tokens is not None:
        target_tokens = max_visual_tokens
    elif text_token_count is None or compression_ratio <= 0:
        # No compression target — render at default full resolution
        return render_code(
            code, style=style, language=language, lint_errors=lint_errors,
            strip_indent=strip_indent, shuffle_indent=shuffle_indent,
            downsample_resolution=downsample_resolution,
            pad_page_to=pad_page_to,
        )
    else:
        target_tokens = int(text_token_count / compression_ratio)

    style_flags = _get_style_config(style)
    show_line_numbers = style_flags["show_line_numbers"]

    # Step 1: Find the best layout that fits within the token budget.
    w, h, fs = _optimize_layout(code, target_tokens, show_line_numbers)

    images = render_code(
        code, style=style, language=language,
        width=w, height=h, font_size=fs,
        lint_errors=lint_errors, strip_indent=strip_indent,
        shuffle_indent=shuffle_indent,
        downsample_resolution=downsample_resolution,
        pad_page_to=pad_page_to,
    )

    # Step 2: If the optimizer couldn't meet the budget (code too long for any
    # layout), downsample the rendered images to hit the target token count.
    actual_tokens = count_image_tokens(images)
    if actual_tokens > target_tokens:
        # compress_images expects (text_token_count, ratio) → target = text/ratio.
        # When max_visual_tokens is set, pass target_tokens directly with ratio=1.0.
        if max_visual_tokens is not None:
            images = compress_images(images, target_tokens, 1.0)
        else:
            images = compress_images(images, text_token_count, compression_ratio)

    return images


def images_to_base64(images: list[Image.Image]) -> list[str]:
    """Convert PIL images to base64-encoded PNG strings (for API calls)."""
    import base64
    import io

    result = []
    for img in images:
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
        result.append(b64)
    return result


# =============================================================================
# Split rendering — one big image vs K small images (a controlled variable)
# =============================================================================
#
# render_code already paginates a long body into multiple pages, but the page
# count is an *emergent* consequence of (font, dimensions, length). For the
# "one big image vs K small images" study the number of images must be a CLEAN,
# CONTROLLED variable, and each image a single legible canvas (its chunk, not
# re-paginated). split_code() does the validity-critical partition; render_split()
# renders each chunk to exactly one image with two fairness modes:
#   - iso-legibility (total_token_budget=None): fixed font; the visual-token cost
#     floats and is measured downstream (count_image_tokens / real pad-tokens).
#   - iso-budget     (total_token_budget=B):    render at fixed font, then
#     uniformly downsample so the WHOLE set fits B visual tokens for the model.
#     No-op for resolution-invariant encoders (Gemma/Aria) — their cost is
#     256·n, controlled by n, not resolution.


def _even_groups(units: list, n: int) -> list[list]:
    """Partition `units` into n contiguous groups, balanced by unit count.

    Mirrors _concat_pages' even distribution (base = U//n; the first U%n groups
    get one extra). Caller guarantees 1 <= n <= len(units), so each group is
    non-empty.
    """
    base, rem = divmod(len(units), n)
    groups, idx = [], 0
    for i in range(n):
        cnt = base + (1 if i < rem else 0)
        groups.append(units[idx:idx + cnt])
        idx += cnt
    return groups


def _blocks_from_seams(lines: list[str], seam_offsets: list[int]) -> list[list[str]]:
    """Group `lines` into contiguous blocks beginning at the given seam offsets.

    seam_offsets are 0-based line indices where a natural block starts (e.g. the
    first line of each cross-file snippet). 0 is always an implicit start;
    out-of-range / duplicate offsets are ignored. Concatenating the blocks
    reproduces `lines` exactly.
    """
    n = len(lines)
    starts = sorted({0} | {s for s in seam_offsets if 0 < s < n})
    blocks = []
    for i, s in enumerate(starts):
        e = starts[i + 1] if i + 1 < len(starts) else n
        blocks.append(lines[s:e])
    return blocks


def split_code(
    code: str,
    n_images: int,
    *,
    policy: str = "lines",
    seam_offsets: list[int] | None = None,
) -> list[str]:
    """Split `code` into at most `n_images` contiguous chunks.

    Validity gate: ``"".join(split_code(code, n)) == code`` for every n — chunks
    never lose, duplicate, or reorder content.

    policy:
      - "lines"  (default): split by line into n balanced contiguous groups.
      - "blocks": cut ONLY at caller-supplied natural seams (`seam_offsets`); the
        seam-blocks are grouped into n contiguous groups balanced by BLOCK COUNT
        (not line count — so unequal blocks yield unequal-sized chunks; the
        caller should record per-chunk size), and a cut never falls inside a
        block. Requires seam_offsets.

        seam_offsets are 0-BASED indices into ``code.splitlines(keepends=True)``
        — the line index where each block STARTS — NOT 1-based editor line
        numbers. A 1-based mistake silently mis-cuts every boundary by one line
        while still passing the join==code gate, so callers should compute them
        from the same splitlines() the renderer uses. 0 is always an implicit
        start; out-of-range / duplicate offsets are ignored. Example: a 9-line
        body whose files span lines 0-3, 4-6, 7-8 → seam_offsets=[0, 4, 7].

    n is capped at the number of available units (lines, or seam-blocks), so a
    body with fewer lines than n_images yields fewer chunks — the caller records
    the actual count.
    """
    if n_images < 1:
        raise ValueError(f"n_images must be >= 1, got {n_images}")
    lines = code.splitlines(keepends=True)
    if not lines:
        return [code]  # empty / contentless → a single (empty) chunk

    if policy == "lines":
        units: list[list[str]] = [[ln] for ln in lines]
    elif policy == "blocks":
        if seam_offsets is None:
            raise ValueError("policy='blocks' requires seam_offsets")
        units = _blocks_from_seams(lines, seam_offsets)
    else:
        raise ValueError(f"unknown split policy {policy!r} (use 'lines' or 'blocks')")

    n = max(1, min(n_images, len(units)))
    groups = _even_groups(units, n)
    return ["".join(ln for unit in group for ln in unit) for group in groups]


def _content_height(code: str, width: int, font_size: int,
                    show_line_numbers: bool, use_bold: bool = False) -> int:
    """Pixel height that fits ALL of `code` on one page at (width, font_size).

    Mirrors _estimate_pages' wrapped-line accounting plus a one-line safety pad.
    render_code auto-crops, so any surplus is trimmed — this only guarantees a
    single page. `use_bold` matches render_code's font so the wrap estimate is
    accurate for bold styles (a wider glyph wraps sooner).
    """
    font = _load_font(font_size, bold=use_bold)
    bbox = font.getbbox("M")
    char_width = max(1.0, (bbox[2] - bbox[0]) if bbox else font_size * CHAR_WIDTH_RATIO)
    line_px = max(1, int(font_size * DEFAULT_LINE_HEIGHT))
    margin = int(width * DEFAULT_MARGIN_RATIO)
    lines = code.splitlines() or [""]
    gutter = 0
    if show_line_numbers:
        gutter = int((len(str(len(lines))) + 2) * char_width)
    text_w = max(1, width - margin - gutter - margin)
    cpl = max(1, int(text_w / char_width))
    wrapped = sum(1 if len(ln) <= cpl else math.ceil(len(ln) / cpl) for ln in lines)
    return max(1, wrapped) * line_px + 2 * margin + line_px


# Cap a single canvas below PIL's decompression-bomb WARNING limit
# (Image.MAX_IMAGE_PIXELS default = 89,478,485; hard error at 2×). The n=1
# "one big image" arm of very long code would otherwise exceed this and crash on
# the base64 round-trip every VLM call performs. We cap *before* allocating, by
# proportionally shrinking the canvas (font included) so the whole body stays in
# ONE image — the natural "one big image of long code is illegible" behaviour;
# the illegibility/cost floats and is recorded downstream.
MAX_CANVAS_PIXELS = 85_000_000


def _render_one_canvas(
    chunk: str, *, style: RenderStyle, language: str, width: int, font_size: int,
    lint_errors=None, strip_indent: bool = False, shuffle_indent: bool = False,
    return_raster_scale: bool = False,
) -> Image.Image | tuple[Image.Image, float]:
    """Render a code chunk as exactly ONE image (single canvas, never paginated).

    If the single canvas would exceed MAX_CANVAS_PIXELS, width and font are
    proportionally shrunk first, so the whole chunk still fits one image without a
    giant allocation or a PIL decompression-bomb crash when the image is reloaded.
    """
    requested_line_px = max(1, int(font_size * DEFAULT_LINE_HEIGHT))
    flags = _get_style_config(style)
    show_ln, use_bold = flags["show_line_numbers"], flags["use_bold"]
    height = _content_height(chunk, width, font_size, show_ln, use_bold)
    if width * height > MAX_CANVAS_PIXELS:
        shrink = (MAX_CANVAS_PIXELS / (width * height)) ** 0.5
        width = max(QWEN_TILE_SIZE, int(width * shrink))
        font_size = max(MIN_FONT_SIZE, int(font_size * shrink))
        height = _content_height(chunk, width, font_size, show_ln, use_bold)
    pages = render_code(
        chunk, style=style, language=language, width=width, height=height,
        font_size=font_size, lint_errors=lint_errors,
        strip_indent=strip_indent, shuffle_indent=shuffle_indent,
    )
    img = pages[0] if len(pages) == 1 else _concat_pages(pages, 1)[0]
    raster_scale_y = (
        max(1, int(font_size * DEFAULT_LINE_HEIGHT)) / requested_line_px
    )
    # Belt-and-suspenders: never hand back an over-cap image (PIL bomb on reload).
    if img.width * img.height > MAX_CANVAS_PIXELS:
        f = (MAX_CANVAS_PIXELS / (img.width * img.height)) ** 0.5
        old_height = img.height
        img = img.resize(
            (max(1, int(img.width * f)), max(1, int(img.height * f))), Image.LANCZOS
        )
        raster_scale_y *= img.height / old_height
    if return_raster_scale:
        return img, raster_scale_y
    return img


def _scale_images(images: list[Image.Image], factor: float) -> list[Image.Image]:
    """Uniformly resize every image by `factor`, snapped to the Qwen tile grid."""
    out = []
    for img in images:
        w, h = img.size
        nw = max(QWEN_TILE_SIZE, round(w * factor / QWEN_TILE_SIZE) * QWEN_TILE_SIZE)
        nh = max(QWEN_TILE_SIZE, round(h * factor / QWEN_TILE_SIZE) * QWEN_TILE_SIZE)
        out.append(img if (nw, nh) == (w, h) else img.resize((nw, nh), Image.LANCZOS))
    return out


def _is_resolution_invariant(model_name: str) -> bool:
    """True for encoders whose visual-token count depends only on image *count*,
    not resolution (Gemma, Aria) — detected by name, the same way
    count_image_tokens branches. Probing by scale is unreliable: Pixtral's 1540px
    longest-side clamp and the per-axis tile floor both fake a 'scaling changes
    nothing' result on images that are NOT resolution-invariant.
    """
    name = model_name.lower()
    return "gemma" in name


# Tile-quantized downscale candidates (large→small) for the iso-budget search.
# An exhaustive scan, NOT a binary search: some encoders are non-monotone in
# scale (Pixtral's 1540px clamp + 28-px block ceiling make the token count wobble
# as the image shrinks), so a monotone assumption can miss the best feasible scale.
_BUDGET_SCALES = [s / 100 for s in (95, 90, 80, 70, 60, 50, 40, 30, 20, 12, 6, 2)]


def _downsample_to_budget(
    images: list[Image.Image], budget: int, model_name: str,
    gemma_max_soft_tokens: int = 280,
) -> list[Image.Image]:
    """Uniformly downsample `images` so the whole set fits `budget` visual tokens.

    Never returns over-budget silently. Raises ValueError when the budget is
    unreachable:
      - budget <= 0;
      - a resolution-invariant encoder (Gemma/Aria) whose fixed cost (e.g. 256·n)
        already exceeds the budget — reduce n_images or raise the budget;
      - an area/dynamic encoder whose per-image tile floor (n × min-tile) exceeds
        the budget.
    For area/dynamic encoders the largest tile-quantized scale whose total fits
    the budget is chosen by an exhaustive scan (robust to non-monotone encoders).
    """
    if not images:
        return images
    if budget <= 0:
        raise ValueError(f"total_token_budget must be > 0, got {budget}")

    def _toks(imgs):
        return count_image_tokens(imgs, model_name=model_name,
                                  gemma_max_soft_tokens=gemma_max_soft_tokens)

    base = _toks(images)
    if base <= budget:
        return images  # already within budget (incl. a feasible fixed-count set)
    if _is_resolution_invariant(model_name):
        raise ValueError(
            f"budget {budget} < fixed cost {base} for resolution-invariant encoder "
            f"{model_name!r} with {len(images)} image(s); its cost is set by image "
            f"count, not resolution — reduce n_images or raise the budget."
        )
    best, best_toks = None, -1
    for scale in _BUDGET_SCALES:
        scaled = _scale_images(images, scale)
        t = _toks(scaled)
        if t <= budget and t > best_toks:
            best, best_toks = scaled, t
    if best is not None:
        return best
    floor = _toks(_scale_images(images, _BUDGET_SCALES[-1]))
    raise ValueError(
        f"budget {budget} is below the achievable floor {floor} for model "
        f"{model_name!r} with {len(images)} image(s); raise the budget or reduce "
        f"n_images."
    )


def render_split(
    code: str,
    n_images: int = 1,
    *,
    style: RenderStyle = "plain",
    language: str = "python",
    split_policy: str = "lines",
    seam_offsets: list[int] | None = None,
    font_size: int = DEFAULT_FONT_SIZE,
    width: int = DEFAULT_IMAGE_WIDTH,
    total_token_budget: int | None = None,
    model_name: str = "",
    gemma_max_soft_tokens: int = 280,
    lint_errors: list[tuple[int, str, str]] | None = None,
    strip_indent: bool = False,
    shuffle_indent: bool = False,
    return_raster_scales: bool = False,
) -> list[Image.Image] | tuple[list[Image.Image], list[float]]:
    """Render `code` as (up to) `n_images` single-canvas images.

    n_images=1 is the "one big image" arm; n_images=K is the "K small images"
    arm. Each chunk becomes exactly one image (never re-paginated), so the image
    count is a clean controlled variable.

    Fairness modes:
      - iso-legibility (total_token_budget=None): every image rendered at the
        same `font_size`; the visual-token cost floats and is measured downstream.
        A single canvas that would exceed MAX_CANVAS_PIXELS (very long code at
        n=1) is proportionally shrunk to stay within PIL limits — so the n=1 arm
        of very long code is capped, which is the real-world "one big legible
        image of thousands of lines is infeasible" behaviour. Detect such cells
        downstream from the returned image size.
      - iso-budget (total_token_budget=B): images rendered at `font_size`, then
        uniformly downsampled so the whole set fits B visual tokens for
        `model_name`. Raises ValueError if B is below the achievable floor —
        for resolution-invariant encoders (Gemma/Aria) when the fixed cost
        (256·n) exceeds B, and for area encoders when n × min-tile exceeds B.
        Never returns over budget.

    Returns at most n_images images (capped at the available line/block count).
    """
    chunks = split_code(code, n_images, policy=split_policy, seam_offsets=seam_offsets)
    if lint_errors is not None and len(chunks) > 1:
        raise ValueError(
            "lint_errors with >1 chunk is not supported — lint line numbers refer "
            "to the whole body, not a chunk. Use n_images=1 or a non-lint style."
        )
    rendered = [
        _render_one_canvas(
            chunk, style=style, language=language, width=width, font_size=font_size,
            lint_errors=lint_errors, strip_indent=strip_indent,
            shuffle_indent=shuffle_indent,
            return_raster_scale=return_raster_scales,
        )
        for chunk in chunks
    ]
    if return_raster_scales:
        images = [image for image, _scale_y in rendered]
        raster_scales = [scale_y for _image, scale_y in rendered]
    else:
        images = rendered
        raster_scales = None
    if total_token_budget is not None:
        source_images = images
        images = _downsample_to_budget(
            source_images, total_token_budget, model_name, gemma_max_soft_tokens,
        )
        if raster_scales is not None:
            raster_scales = [
                scale_y * (image.height / source.height)
                for scale_y, source, image in zip(
                    raster_scales,
                    source_images,
                    images,
                    strict=True,
                )
            ]
    if return_raster_scales:
        return images, raster_scales
    return images
