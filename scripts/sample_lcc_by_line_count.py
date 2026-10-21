#!/usr/bin/env python3
"""
Sample LCC examples stratified by SOURCE LINE COUNT.

Stratifies the LCC test split into line-count tiers and draws a fixed-seed
random sample per (language, tier), writing one JSON manifest per cell. The
manifests pin exactly which examples each experiment uses, so results stay
reproducible even if microsoft/LCC or this script changes.

WHY LINE COUNT (not tokens): tokens are encoder-specific. Qwen2.5-Coder,
Qwen3.5, Gemma4 and the various image encoders (Qwen block-32, Pixtral
block-28, Gemma SigLIP fixed-256) all yield different token counts for
the SAME code. Stratifying by tokens would mean each model effectively
sees different tier boundaries — not a fair length axis. Lines of code
are identical across every model and modality, so they are the only fair
stratification variable. Token cost is still reported downstream as a
secondary descriptive axis, but never as the stratifier.

Self-contained: depends only on `datasets` plus stdlib. No tokenizer, no
codecompress imports — can be copied out of the repo and run as-is.

    uv run python scripts/sample_lcc_by_line_count.py
    uv run python scripts/sample_lcc_by_line_count.py --lang python --n 200
    uv run python scripts/sample_lcc_by_line_count.py \
        --tiers 200-300,300-500,500-1000,1000-2000 --out-dir data/lengthstrat

`line_count` semantics are identical to scripts/probe_lcc_dataset.py:
count("\\n") plus one for the final line unless the text ends in a
newline. Tiers are half-open [lo, hi): a 300-line file with tiers
200-300,300-500 lands in 300-500, not 200-300.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
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


LANGUAGES = ("python", "java", "csharp")
DEFAULT_TIERS = "200-300,300-500,500-1000,1000-2000"
LINE_COUNT_METRIC = "count('\\n') + (0 if endswith('\\n') else 1)"


def line_count(text: str) -> int:
    """Count of lines; a trailing newline does not add an empty final line."""
    if not text:
        return 0
    n = text.count("\n")
    return n + (0 if text.endswith("\n") else 1)


def parse_tiers(spec: str) -> list[tuple[str, int, int]]:
    """Parse "lo-hi,lo-hi,..." into [(label, lo, hi), ...].

    Tiers must be sorted ascending and non-overlapping (adjacent is fine:
    200-300 then 300-500 is valid because bounds are half-open [lo, hi)).
    Raises ValueError on any malformed / unsorted / overlapping input.
    """
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
        # b must start at or after a's exclusive upper bound, and lower
        # bounds must be non-decreasing — otherwise assignment is ambiguous.
        if a[1] > b[1] or a[2] > b[1]:
            raise ValueError(
                "tiers must be sorted ascending and non-overlapping; "
                f"{a[0]} conflicts with {b[0]}"
            )
    return tiers


def assign_tier(n_lines: int, tiers: list[tuple[str, int, int]]) -> str | None:
    """Return the tier label whose half-open [lo, hi) contains n_lines, else None."""
    for label, lo, hi in tiers:
        if lo <= n_lines < hi:
            return label
    return None


def tier_seed(base_seed: int, tier_label: str) -> str:
    """Derive a deterministic, tier-specific seed string.

    Using the same `base_seed` for every tier would make the RNG draw the
    same ordinal positions in each tier's pool (cross-tier rank
    correlation). Folding the tier label into the seed gives each tier an
    independent RNG stream while staying fully reproducible. `random.Random`
    accepts a str seed directly.
    """
    return f"{base_seed}:{tier_label}"


def stratified_sample(pool: list[int], n: int, seed) -> list[int]:
    """Deterministically sample up to `n` items from `pool` — PREFIX-STABLE.

    n<=0 yields []. If the pool has <= n items, returns all of them. Output
    is always sorted (arrow datasets read faster in index order). `seed` may
    be an int or a str (see tier_seed) — random.Random accepts both.

    **Prefix-stability across n**: the implementation is shuffle-once-then-slice,
    so calling with seed=42 at n=200 yields a strict subset of the result at
    seed=42 n=500 (same shuffled order, longer prefix). ``random.Random.sample``
    is NOT prefix-stable, so do not switch back to it.
    """
    if n <= 0:
        return []
    rng = random.Random(seed)
    pool_copy = list(pool)
    rng.shuffle(pool_copy)
    take = min(n, len(pool_copy))
    return sorted(pool_copy[:take])


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
    dataset: str,
    n_rows_total: int,
    datasets_version: str,
    selected_content_sha256: str,
) -> dict:
    """Assemble the JSON-serializable manifest for one (language, tier) cell.

    The provenance fields (dataset / n_rows_total / datasets_version /
    selected_content_sha256) let a future check verify the pinned indices
    still reference the same code, even if microsoft/LCC is re-uploaded or
    the `datasets` library changes Arrow row order.
    """
    selected = list(selected_indices)
    return {
        "language": lang,
        "split": split,
        "tier": tier_label,
        "line_lo": lo,
        "line_hi": hi,
        "seed": seed,
        "n_requested": n_requested,
        "n_selected": len(selected),
        "total_in_stratum": total_in_stratum,
        "line_count_metric": LINE_COUNT_METRIC,
        "dataset": dataset,
        "n_rows_total": n_rows_total,
        "datasets_version": datasets_version,
        "selected_content_sha256": selected_content_sha256,
        "example_indices": selected,
    }


def sample_language(
    lang: str,
    split: str,
    tiers: list[tuple[str, int, int]],
    n: int,
    seed: int,
) -> list[dict]:
    """Load one LCC language, bucket by line count, sample each tier."""
    dataset_name = f"microsoft/LCC_{lang}"
    ds = load_dataset(dataset_name, split=split)
    n_rows_total = len(ds)

    try:
        import datasets as _datasets

        datasets_version = str(_datasets.__version__)
    except Exception:  # pragma: no cover - version introspection is best-effort
        datasets_version = "unknown"

    def _ctx(i: int) -> str:
        text = ds[i].get("context", "")
        if not isinstance(text, str):
            text = "" if text is None else str(text)
        return text

    pools: dict[str, list[int]] = {label: [] for label, _, _ in tiers}
    for i in range(n_rows_total):
        label = assign_tier(line_count(_ctx(i)), tiers)
        if label is not None:
            pools[label].append(i)

    manifests = []
    for label, lo, hi in tiers:
        pool = pools[label]
        selected = stratified_sample(pool, n, tier_seed(seed, label))
        # Hash the actual selected source so a later check can detect if the
        # pinned indices ever stop pointing at the same code.
        h = hashlib.sha256()
        for idx in selected:
            h.update(_ctx(idx).encode("utf-8", "surrogatepass"))
            h.update(b"\x00")
        manifests.append(
            build_manifest(
                lang, split, label, lo, hi, seed, n, selected, len(pool),
                dataset=dataset_name,
                n_rows_total=n_rows_total,
                datasets_version=datasets_version,
                selected_content_sha256=h.hexdigest(),
            )
        )
        print(f"  {lang:<7} {label:<12} {len(selected):>4}/{len(pool):<6} sampled")
    return manifests


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Sample LCC examples stratified by source line count.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--lang",
        choices=(*LANGUAGES, "all"),
        default="all",
        help="Language subset(s) to sample.",
    )
    parser.add_argument("--split", default="test", help="Dataset split.")
    parser.add_argument(
        "--n", type=int, default=200, help="Examples to sample per tier."
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for sampling."
    )
    parser.add_argument(
        "--tiers",
        default=DEFAULT_TIERS,
        help="Comma-separated half-open line-count tiers 'lo-hi,lo-hi,...'.",
    )
    parser.add_argument(
        "--out-dir",
        default="data/lengthstrat",
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

    print(f"Tiers (half-open, line count): {[t[0] for t in tiers]}")
    print(f"Sampling n={args.n} per tier, seed={args.seed}, split={args.split}\n")

    written = []
    for lang in langs:
        print(f"{lang}:")
        for m in sample_language(lang, args.split, tiers, args.n, args.seed):
            path = out_dir / f"{lang}_{m['tier']}.json"
            path.write_text(json.dumps(m, indent=2) + "\n")
            written.append(path)
        print()

    print(f"Wrote {len(written)} manifest(s) to {out_dir}/")


if __name__ == "__main__":
    main()
