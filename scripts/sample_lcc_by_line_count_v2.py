#!/usr/bin/env python3
"""Sample task-well-posed LCC examples by raw-context line count.

This is the successor to ``sample_lcc_by_line_count.py``.  It keeps
the v1 line-count, tier, and deterministic-sampling contracts, but scans the
entire requested split and excludes rows whose current function/class is
longer than 50 lines.  It deliberately applies no tokenizer-dependent
background-length filter.

The default output directory is distinct from v1 so invoking this script with
no arguments cannot overwrite earlier manifests.
"""

# ruff: noqa: E402 -- repo-root bootstrap must precede worktree-local imports

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

try:
    from datasets import load_dataset
except ImportError:
    sys.stderr.write(
        "error: the `datasets` package is required.\n       install it with: pip install datasets\n"
    )
    sys.exit(1)

from codecompress.benchmarks.lcc import (
    compute_selected_sha256,
    split_context_ast,
)
from scripts.sample_lcc_by_line_count import (
    DEFAULT_TIERS,
    LANGUAGES,
    assign_tier,
    build_manifest,
    line_count,
    parse_tiers,
    stratified_sample,
    tier_seed,
)


DEFAULT_OUT_DIR = "data/lengthstrat_lcc_v2"
CURRENT_FUNCTION_LINES_MAX = 50
FILTERS_APPLIED = {
    "current_function_lines_max": CURRENT_FUNCTION_LINES_MAX,
    "background_tokens_min": None,
    "scan_scope": "full_split",
}


def _context(row: dict) -> str:
    """Return a row's raw context using the same coercion as the v1 sampler."""
    text = row.get("context", "")
    if isinstance(text, str):
        return text
    return "" if text is None else str(text)


def row_passes_filter(row: dict, language: str) -> bool:
    """Whether the AST-split current function/class has at most 50 lines."""
    _, current_function = split_context_ast(_context(row), language)
    return len(current_function.splitlines()) <= CURRENT_FUNCTION_LINES_MAX


def sample_language(
    lang: str,
    split: str,
    tiers: list[tuple[str, int, int]],
    n: int,
    seed: int,
) -> list[dict]:
    """Scan a complete LCC split, filter within raw-line-count tiers, sample."""
    dataset_name = f"microsoft/LCC_{lang}"
    ds = load_dataset(dataset_name, split=split)
    n_rows_total = len(ds)

    try:
        import datasets as _datasets

        datasets_version = str(_datasets.__version__)
    except Exception:  # pragma: no cover - version introspection is best-effort
        datasets_version = "unknown"

    pre_filter_pools: dict[str, list[int]] = {label: [] for label, _, _ in tiers}
    eligible_pools: dict[str, list[int]] = {label: [] for label, _, _ in tiers}
    contexts: dict[int, str] = {}

    # Deliberately traverse every row.  In particular, there is no legacy
    # loader-style 10*n prescan window here.
    for i in range(n_rows_total):
        row = ds[i]
        context = _context(row)
        label = assign_tier(line_count(context), tiers)
        if label is None:
            continue
        contexts[i] = context
        pre_filter_pools[label].append(i)
        if row_passes_filter(row, language=lang):
            eligible_pools[label].append(i)

    manifests = []
    for label, lo, hi in tiers:
        pre_filter_total = len(pre_filter_pools[label])
        pool = eligible_pools[label]
        total_in_stratum = len(pool)
        filter_removed = pre_filter_total - total_in_stratum
        filter_removed_fraction = filter_removed / pre_filter_total if pre_filter_total else 0.0
        selected = stratified_sample(pool, n, tier_seed(seed, label))
        selected_sha256 = compute_selected_sha256(contexts[idx] for idx in selected)

        manifest = build_manifest(
            lang,
            split,
            label,
            lo,
            hi,
            seed,
            n,
            selected,
            total_in_stratum,
            dataset=dataset_name,
            n_rows_total=n_rows_total,
            datasets_version=datasets_version,
            selected_content_sha256=selected_sha256,
        )
        manifest.update(
            {
                "pre_filter_total": pre_filter_total,
                "filter_removed": filter_removed,
                "filter_removed_fraction": filter_removed_fraction,
                "filters_applied": dict(FILTERS_APPLIED),
            }
        )
        manifests.append(manifest)
        print(
            f"  {lang:<7} {label:<12} {len(selected):>4}/"
            f"{total_in_stratum:<6} sampled; filtered "
            f"{filter_removed}/{pre_filter_total} "
            f"({filter_removed_fraction:.1%})"
        )

    return manifests


def main() -> None:
    parser = argparse.ArgumentParser(
        description=("Sample task-well-posed LCC examples stratified by raw source line count."),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--lang",
        choices=(*LANGUAGES, "all"),
        default="all",
        help="Language subset(s) to sample.",
    )
    parser.add_argument("--split", default="test", help="Dataset split.")
    parser.add_argument("--n", type=int, default=200, help="Examples to sample per tier.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for sampling.")
    parser.add_argument(
        "--tiers",
        default=DEFAULT_TIERS,
        help="Comma-separated half-open line-count tiers 'lo-hi,lo-hi,...'.",
    )
    parser.add_argument(
        "--out-dir",
        default=DEFAULT_OUT_DIR,
        help="Directory to write per-(language,tier) manifest JSON files.",
    )
    args = parser.parse_args()

    try:
        tiers = parse_tiers(args.tiers)
    except ValueError as exc:
        parser.error(f"--tiers: {exc}")

    langs = LANGUAGES if args.lang == "all" else (args.lang,)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Tiers (half-open, line count): {[tier[0] for tier in tiers]}")
    print(f"Sampling n={args.n} per tier, seed={args.seed}, split={args.split}\n")

    written = []
    for lang in langs:
        print(f"{lang}:")
        for manifest in sample_language(lang, args.split, tiers, args.n, args.seed):
            path = out_dir / f"{lang}_{manifest['tier']}.json"
            path.write_text(json.dumps(manifest, indent=2) + "\n")
            written.append(path)
        print()

    print(f"Wrote {len(written)} manifest(s) to {out_dir}/")


if __name__ == "__main__":
    main()
