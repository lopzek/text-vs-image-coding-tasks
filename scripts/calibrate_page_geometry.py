#!/usr/bin/env python3
"""Tabulate the candidate page geometries and select one per encoder.

For every candidate fixed triple (width × height × font_size) and both
studied encoders this CPU-only script records (a) the native effective
character height ``min(rendered line px, post-resize line px)`` — the axis
the legibility knee/threshold lives on — and (b) the visual-token cost per
source line measured with the study's own tail-fit renderer on the first
``--rows`` python rows (by example_id) of the lcc 300–500 bin.  It also records
each candidate's ``chars_per_line`` and the fraction of calibration source lines
that would hard-wrap (``wrap_frac``; wrapping splits identifiers and the
indentation column, and the calibration never uses wrapped text).  The
selection rule is then mechanical once each encoder's threshold is known —
``select_geometry``: admit candidates with native effective ≥ threshold AND
wrap_frac ≤ anchor wrap_frac + 0.01 (anchor = 1120×1120/f20, the calibration
page canvas); order by (tokens/line ascending, font descending, area
ascending); take the first.  The calibration row set is recorded by example_id
list + content sha256 in the report.

Run BEFORE the VLM transcription so the candidate table is on disk before any
knee is known.  No network, no GPU.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

QWEN_MODEL = "Qwen/Qwen3.8-27B"
GEMMA_MODEL = "google/gemma-4-26B-A4B-it"
ENCODERS = {"qwen": QWEN_MODEL, "gemma": GEMMA_MODEL}
WIDTHS = (640, 768, 896, 1120)
HEIGHTS = (560, 672, 784, 896, 1120, 1568, 2240)
FONTS = (16, 20, 24)
MST = 280
BIN = (300, 500)

_ROWS: list[str] = []


def candidates():
    return [(w, h, f) for w in WIDTHS for h in HEIGHTS for f in FONTS]


ANCHOR = (1120, 1120, 20)  # the calibration page canvas
WRAP_SLACK = 0.01


def chars_per_line(width, font_size, style="plain", n_lines=1000):
    """Mirror render_code_tail_fit's wrap width for the given page geometry."""
    from codecompress.code_renderer import (
        CHAR_WIDTH_RATIO,
        DEFAULT_MARGIN_RATIO,
        _get_style_config,
        _load_font,
    )

    flags = _get_style_config(style)
    font = _load_font(font_size, bold=flags["use_bold"])
    bbox = font.getbbox("M")
    char_width = bbox[2] - bbox[0] if bbox else font_size * CHAR_WIDTH_RATIO
    margin = int(width * DEFAULT_MARGIN_RATIO)
    gutter = int((len(str(n_lines)) + 2) * char_width) if flags["show_line_numbers"] else 0
    return max(1, int((width - margin - gutter - margin) / char_width))


def wrap_fraction(rows, cpl):
    n = wrapped = 0
    for text in rows:
        for line in text.splitlines():
            n += 1
            wrapped += len(line) > cpl
    return wrapped / max(n, 1)


def _q1(x):
    """Quantise to 1 dp with ROUND_HALF_UP via Decimal (python's round() is
    binary-repr sensitive at .x5). The threshold is itself quoted at 1 dp with
    the same rule."""
    from decimal import ROUND_HALF_UP, Decimal

    return float(Decimal(repr(float(x))).quantize(Decimal("0.1"), rounding=ROUND_HALF_UP))


def select_geometry(candidates, encoder, threshold, anchor=ANCHOR, slack=WRAP_SLACK):
    """The selection rule. ``candidates`` = the report's list; returns the
    chosen candidate dict or None when nothing is admissible."""
    by_key = {(c["width"], c["height"], c["font_size"]): c for c in candidates}
    anchor_wrap = by_key[anchor]["wrap_frac"]
    # Precision rule: compare at the table's precision — both the candidate's
    # p05 and the threshold are rounded to 1 dp before
    # the comparison — so a threshold quoted from the table admits the row it
    # names.
    thr = _q1(threshold)
    admissible = [
        c
        for c in candidates
        if _q1(c["encoders"][encoder]["effective_cropped_p05"]) >= thr
        and c["wrap_frac"] <= anchor_wrap + slack
    ]
    if not admissible:
        return None
    admissible.sort(
        key=lambda c: (
            round(c["encoders"][encoder]["tokens_per_source_line"], 2),
            -c["font_size"],
            c["width"] * c["height"],
        )
    )
    return admissible[0]


def native_effective(width, height, font_size, model_name):
    from codecompress.code_renderer import DEFAULT_LINE_HEIGHT, char_px_post_resize

    rendered = float(int(font_size * DEFAULT_LINE_HEIGHT))
    post = float(
        char_px_post_resize(font_size, width, height, model_name, gemma_max_soft_tokens=MST)
    )
    return {"rendered_line_px": rendered, "post_line_px": post, "effective": min(rendered, post)}


def _init(rows):
    global _ROWS
    _ROWS = rows


def _work(cand):
    from codecompress.code_renderer import render_code_tail_fit

    w, h, f = cand
    cpl = chars_per_line(w, f)
    out = {
        "width": w,
        "height": h,
        "font_size": f,
        "chars_per_line": cpl,
        "wrap_frac": wrap_fraction(_ROWS, cpl),
        "encoders": {},
    }
    from codecompress.code_renderer import DEFAULT_LINE_HEIGHT, char_px_post_resize

    rendered_px = float(int(f * DEFAULT_LINE_HEIGHT))
    for enc, model in ENCODERS.items():
        tot_tok = tot_lines = tot_pages = 0
        page_eff = []
        for text in _ROWS:
            imgs, info = render_code_tail_fit(
                text,
                token_budget=2**40,
                model_name=model,
                style="plain",
                language="python",
                width=w,
                height=h,
                font_size=f,
                gemma_max_soft_tokens=MST,
            )
            tot_tok += info["visual_tokens_total"]
            tot_pages += info["pages_total"]
            tot_lines += max(1, len(text.splitlines()))
            # Legibility axis on the pages the study actually serves: every
            # page is auto-cropped, and Gemma's resize is aspect-dependent, so
            # the nominal canvas is only the full-width worst case. The knee is
            # measured on cropped pages too.
            for img in imgs:
                post = float(
                    char_px_post_resize(f, img.width, img.height, model, gemma_max_soft_tokens=MST)
                )
                page_eff.append(min(rendered_px, post))
        page_eff.sort()
        n = len(page_eff)
        out["encoders"][enc] = {
            **native_effective(w, h, f, model),
            "effective_cropped_min": page_eff[0],
            "effective_cropped_p05": page_eff[max(0, int(0.05 * n) - 1)] if n else None,
            "effective_cropped_median": page_eff[n // 2] if n else None,
            "n_pages": n,
            "tokens_per_source_line": tot_tok / tot_lines,
            "pages_per_1000_lines": 1000.0 * tot_pages / tot_lines,
            "n_rows": len(_ROWS),
        }
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--rows", type=int, default=50)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--out", required=True)
    ap.add_argument(
        "--select",
        default=None,
        help="JSON {encoder: threshold} (1 dp): apply the selection rule to the table at "
        "--out and write the selected geometry JSON to --geometry-out",
    )
    ap.add_argument("--geometry-out", default=None)
    args = ap.parse_args(argv)
    if args.select:
        import json as _json

        table = _json.load(open(args.out))["candidates"]
        thresholds = _json.load(open(args.select))
        geometry = {}
        for enc, thr in thresholds.items():
            c = select_geometry(table, enc, thr)
            if c is None:
                print(f"{enc}: CALIBRATION-FAIL — no admissible candidate at threshold {thr}")
                return 1
            geometry[enc] = {
                "width": c["width"],
                "height": c["height"],
                "font_size": c["font_size"],
                "min_char_px": _q1(thr),
                "threshold_1dp": _q1(thr),
                "effective_cropped_p05": c["encoders"][enc]["effective_cropped_p05"],
                "tokens_per_source_line": c["encoders"][enc]["tokens_per_source_line"],
                "wrap_frac": c["wrap_frac"],
            }
            print(f"{enc}: threshold {thr} → {c['width']}x{c['height']} f{c['font_size']}")
        if args.geometry_out:
            Path(args.geometry_out).write_text(_json.dumps(geometry, indent=2) + "\n")
        return 0
    from scripts.loc_census import load_task_rows

    pool_rows = [r for r in load_task_rows("lcc", lcc_lang="python") if BIN[0] <= r.lines < BIN[1]]
    pool_rows.sort(key=lambda r: int(r.example_id))
    from codecompress.benchmarks.lcc import split_context_ast

    # Render exactly what the study renders: the background half of the
    # AST split (the query tail travels as text).
    rows = [split_context_ast(r.text)[0] for r in pool_rows[: args.rows]]
    ids = [r.example_id for r in pool_rows[: args.rows]]
    import hashlib

    rows_sha = hashlib.sha256("\x00".join(rows).encode()).hexdigest()
    ctx = multiprocessing.get_context("fork")
    with ctx.Pool(args.workers, initializer=_init, initargs=(rows,)) as pool:
        table = list(pool.imap(_work, candidates()))
    report = {
        "schema_version": 1,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "rule": "admit effective_cropped_p05 (5th percentile of "
        "min(rendered, post-resize) line px over the calibration rows' AUTO-CROPPED "
        "pages) >= threshold(encoder) AND wrap_frac <= anchor(1120x1120/f20) "
        "wrap_frac + 0.01; order by (tokens_per_source_line asc, font desc, area "
        "asc); take first. 'effective' (nominal canvas) is kept for reference only.",
        "anchor": {
            "width": ANCHOR[0],
            "height": ANCHOR[1],
            "font_size": ANCHOR[2],
            "note": "the calibration page canvas",
        },
        "wrap_slack": WRAP_SLACK,
        "candidate_grid": {"widths": WIDTHS, "heights": HEIGHTS, "fonts": FONTS, "mst": MST},
        "calibration_rows": {
            "task": "lcc",
            "lang": "python",
            "bin": BIN,
            "n": len(rows),
            "selection": "census pool rows with BIN[0] <= lines < BIN[1], sorted by int(example_id), first --rows",
            "example_ids": ids,
            "content_sha256": rows_sha,
        },
        "encoders": ENCODERS,
        "candidates": table,
    }
    Path(args.out).write_text(json.dumps(report, indent=2) + "\n")
    for c in table:
        q, g = c["encoders"]["qwen"], c["encoders"]["gemma"]
        print(
            f"{c['width']:>5}x{c['height']:<5} f{c['font_size']:<3} cpl {c['chars_per_line']:>3} "
            f"wrap {100 * c['wrap_frac']:4.1f}% | qwen p05 {q['effective_cropped_p05']:5.1f} "
            f"tok/line {q['tokens_per_source_line']:6.2f} | gemma p05 {g['effective_cropped_p05']:5.1f} "
            f"(nom {g['effective']:4.1f}) tok/line {g['tokens_per_source_line']:5.2f}"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
