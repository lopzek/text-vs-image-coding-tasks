#!/usr/bin/env python3
"""
Sample RepoBench examples stratified by RENDERED-BACKGROUND LINE COUNT.

Mirrors
scripts/sample_lcc_by_line_count.py: stratifies the
``tianyang/repobench_python_v1.1`` ``cross_file_first`` split into
line-count tiers and draws a fixed-seed sample per tier, writing one JSON
manifest per tier. Manifests pin exactly which examples each phase uses.

WHY LINE COUNT (not tokens): identical reasoning to the LCC sampler —
tokens are encoder-specific, lines are not, so lines are the only fair
length axis for a text-vs-image modality contrast.

NOT SELF-CONTAINED (deliberate, unlike the LCC sampler): the length axis
and the manifest hash are computed by importing the benchmark's own
``repobench_cross_file_prompt`` / ``normalize_blanklines`` /
``repobench_audit_contents``. This guarantees the line count we stratify
on is byte-identical to the background string the pipeline renders/feeds,
and the manifest's ``selected_content_sha256`` is byte-identical to what
``RepoBenchBenchmark.load`` recomputes — sampler↔benchmark lockstep. A
self-contained copy would risk silent drift between the two.

    uv run python scripts/sample_repobench_by_line_count.py
    uv run python scripts/sample_repobench_by_line_count.py \
        --tiers 200-300,300-500,500-1000,1000-2000 \
        --out-dir data/lengthstrat_repobench

`line_count` semantics match scripts/sample_lcc_by_line_count.py exactly:
count("\\n") plus one for the final line unless the text ends in a
newline. Tiers are half-open [lo, hi).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

try:
    from datasets import load_dataset
except ImportError:
    sys.stderr.write(
        "error: the `datasets` package is required.\n"
        "       install it with: pip install datasets\n"
    )
    sys.exit(1)

# Lockstep with the pipeline: same background builder + content-hash protocol
# the benchmark uses. Requires running from the repo (codecompress importable).
from codecompress.benchmarks.lcc import compute_selected_sha256
from codecompress.benchmarks.repobench import (
    normalize_blanklines,
    repobench_audit_contents,
    repobench_cross_file_prompt,
)


DEFAULT_DATASET = "tianyang/repobench_python_v1.1"
DEFAULT_SPLIT = "cross_file_first"
DEFAULT_TIERS = "200-300,300-500,500-1000,1000-2000"
LINE_COUNT_METRIC = "count('\\n') + (0 if endswith('\\n') else 1)"


def line_count(text: str) -> int:
    """Count of lines; a trailing newline does not add an empty final line."""
    if not text:
        return 0
    n = text.count("\n")
    return n + (0 if text.endswith("\n") else 1)


def background_text(row: dict) -> str:
    """The exact rendered/fed cross-file background for *row*.

    Identical to what RepoBenchBenchmark.load stores as
    ``background_context`` (single source of truth)."""
    return normalize_blanklines(repobench_cross_file_prompt(row, "python"))


def background_line_count(row: dict) -> int:
    return line_count(background_text(row))


# --- tier helpers (mirror the LCC sampler — half-open [lo, hi)) -------------

def parse_tiers(spec: str) -> list[tuple[str, int, int]]:
    tiers: list[tuple[str, int, int]] = []
    for tok in spec.split(","):
        tok = tok.strip()
        if not tok:
            raise ValueError(f"empty tier token in {spec!r}")
        parts = tok.split("-")
        if len(parts) != 2:
            raise ValueError(f"invalid tier {tok!r}: expected 'lo-hi'")
        try:
            lo = int(parts[0].strip())
            hi = int(parts[1].strip())
        except ValueError as exc:
            raise ValueError(
                f"invalid tier {tok!r}: bounds must be integers"
            ) from exc
        if lo >= hi:
            raise ValueError(f"invalid tier {tok!r}: lo must be < hi")
        tiers.append((f"{lo}-{hi}", lo, hi))

    for a, b in zip(tiers, tiers[1:]):
        if a[1] > b[1] or a[2] > b[1]:
            raise ValueError(
                "tiers must be sorted ascending and non-overlapping; "
                f"{a[0]} conflicts with {b[0]}"
            )
    return tiers


def assign_tier(n_lines: int, tiers: list[tuple[str, int, int]]) -> str | None:
    for label, lo, hi in tiers:
        if lo <= n_lines < hi:
            return label
    return None


def tier_seed(base_seed: int, tier_label: str) -> str:
    """Tier-specific seed string so each tier draws an independent RNG
    stream (avoids cross-tier rank correlation). See the LCC sampler."""
    return f"{base_seed}:{tier_label}"


def stratified_sample(pool: list[int], n: int, seed) -> list[int]:
    """Deterministically sample up to *n* from *pool*; output sorted.

    Prefix-stable across n (shuffle-once-then-slice) so pub N=500 nests
    dev N=200 at the same seed — see the LCC sampler's docstring for
    the full rationale. NOTE: behavior change vs the previous
    ``random.Random(seed).sample`` implementation; manifests committed
    under the old implementation have different selected_content_sha256
    values and must be regenerated."""
    import random

    if n <= 0:
        return []
    rng = random.Random(seed)
    pool_copy = list(pool)
    rng.shuffle(pool_copy)
    take = min(n, len(pool_copy))
    return sorted(pool_copy[:take])


def selected_content_sha256(dataset, indices: list[int]) -> str:
    """Manifest provenance hash — MUST equal what RepoBenchBenchmark.load
    recomputes (compute_selected_sha256 ∘ repobench_audit_contents)."""
    return compute_selected_sha256(
        repobench_audit_contents(dataset.select(indices))
    )


def build_manifest(
    lang: str,
    split: str,
    tier_label: str,
    lo: int,
    hi: int,
    seed: int,
    n_requested: int,
    selected_indices: list[int],
    total_in_stratum: int,
    dataset_name: str,
    n_rows_total: int,
    datasets_version: str,
    sha256: str,
) -> dict:
    return {
        "language": lang,
        "split": split,
        "tier": tier_label,
        "line_lo": lo,
        "line_hi": hi,
        "seed": seed,
        "n_requested": n_requested,
        "n_selected": len(selected_indices),
        "total_in_stratum": total_in_stratum,
        "line_count_metric": LINE_COUNT_METRIC,
        "line_count_basis": "rendered cross-file background "
                            "(repobench_cross_file_prompt → normalize_blanklines)",
        "dataset": dataset_name,
        "n_rows_total": n_rows_total,
        "datasets_version": datasets_version,
        "selected_content_sha256": sha256,
        "example_indices": list(selected_indices),
    }


def sample_language(
    lang: str,
    split: str,
    tiers: list[tuple[str, int, int]],
    n: int,
    seed: int,
) -> list[dict]:
    """Load RepoBench, bucket by background line count, sample each tier."""
    dataset_name = DEFAULT_DATASET  # python v1.1
    ds = load_dataset(dataset_name, split=split)
    n_rows_total = len(ds)

    try:
        import datasets as _datasets

        datasets_version = str(_datasets.__version__)
    except Exception:  # pragma: no cover - best effort
        datasets_version = "unknown"

    pools: dict[str, list[int]] = {label: [] for label, _, _ in tiers}
    for i in range(n_rows_total):
        label = assign_tier(background_line_count(ds[i]), tiers)
        if label is not None:
            pools[label].append(i)

    manifests = []
    for label, lo, hi in tiers:
        pool = pools[label]
        selected = stratified_sample(pool, n, tier_seed(seed, label))
        sha = selected_content_sha256(ds, selected) if selected else \
            compute_selected_sha256(iter(()))
        manifests.append(
            build_manifest(
                lang, split, label, lo, hi, seed, n, selected, len(pool),
                dataset_name=dataset_name,
                n_rows_total=n_rows_total,
                datasets_version=datasets_version,
                sha256=sha,
            )
        )
        print(f"  {lang:<7} {label:<12} {len(selected):>4}/{len(pool):<6} sampled")
    return manifests


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Sample RepoBench examples stratified by rendered "
                    "cross-file background line count.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--lang", choices=("python",), default="python",
                        help="Language subset (RepoBench v1.1 = python here).")
    parser.add_argument("--split", default=DEFAULT_SPLIT,
                        help="RepoBench split.")
    parser.add_argument("--n", type=int, default=200,
                        help="Examples to sample per tier.")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for sampling.")
    parser.add_argument("--tiers", default=DEFAULT_TIERS,
                        help="Comma-separated half-open line tiers.")
    parser.add_argument("--out-dir", default="data/lengthstrat_repobench",
                        help="Directory for per-tier manifest JSON files.")
    args = parser.parse_args()

    try:
        tiers = parse_tiers(args.tiers)
    except ValueError as exc:
        parser.error(f"--tiers: {exc}")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Tiers (half-open, line count): {[t[0] for t in tiers]}")
    print(f"Sampling n={args.n} per tier, seed={args.seed}, "
          f"split={args.split}\n")

    written = []
    print(f"{args.lang}:")
    for m in sample_language(args.lang, args.split, tiers, args.n, args.seed):
        path = out_dir / f"{args.lang}_{m['tier']}.json"
        path.write_text(json.dumps(m, indent=2) + "\n")
        written.append(path)
    print()
    print(f"Wrote {len(written)} manifest(s) to {out_dir}/")


if __name__ == "__main__":
    main()
