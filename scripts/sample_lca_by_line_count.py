#!/usr/bin/env python3
"""Sample LCA module-summarization rows by natural background line count.

The line axis is the exact background returned by
``LCASummarizationBenchmark.split_context`` and uses ``str.splitlines()``,
matching the LoC census.  Each manifest pins full-split row indices and
an aggregate hash over ``background, target_text`` for every selected row.
"""

# ruff: noqa: E402 -- repo-root bootstrap must precede worktree-local imports

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    # A shared editable venv may point at the main worktree. Prefer the code
    # beside this script so branch/worktree execution cannot import stale
    # benchmark modules.
    sys.path.insert(0, str(REPO_ROOT))

try:
    from datasets import load_dataset
except ImportError:
    sys.stderr.write("error: the `datasets` package is required.\n")
    sys.exit(1)

from codecompress.benchmarks.lca_summarization import (
    LCASummarizationBenchmark,
    lca_audit_contents,
)
from codecompress.benchmarks.lcc import compute_selected_sha256
from scripts.sample_lcc_by_line_count import (
    assign_tier,
    line_count,  # noqa: F401 -- intentionally re-exported v1 contract
    parse_tiers,
    stratified_sample,
    tier_seed,
)


DEFAULT_DATASET = "JetBrains-Research/lca-module-summarization"
DEFAULT_SPLIT = "test"
DEFAULT_TIERS = "0-1000,1000-2000,2000-5000,5000-999999"
DEFAULT_OUT_DIR = "data/lengthstrat_lca"
LINE_COUNT_METRIC = "len(split_context(row)[0].splitlines())"


def background_text(row: dict) -> str:
    """Return the exact LCA background consumed by the benchmark."""
    background, _query = LCASummarizationBenchmark().split_context(row)
    return background


def background_line_count(row: dict) -> int:
    """Count LCA background lines with the census's ``splitlines`` metric."""
    return len(background_text(row).splitlines())


def selected_content_sha256(dataset, indices: list[int]) -> str:
    """Hash selected LCA rows with the loader's shared content-hash protocol."""
    return compute_selected_sha256(lca_audit_contents(dataset.select(indices)))


def _datasets_version() -> str:
    try:
        import datasets

        return str(datasets.__version__)
    except Exception:  # pragma: no cover - provenance introspection only
        return "unknown"


def _build_manifest(
    *,
    dataset_name: str,
    split: str,
    tier_label: str,
    lo: int,
    hi: int,
    seed: int,
    n_requested: int,
    selected_indices: list[int],
    total_in_stratum: int,
    n_rows_total: int,
    selected_sha256: str,
) -> dict:
    return {
        "task": "lca_summarization",
        "dataset": dataset_name,
        "split": split,
        "tier": tier_label,
        "line_lo": lo,
        "line_hi": hi,
        "seed": seed,
        "n_requested": n_requested,
        "n_selected": len(selected_indices),
        "total_in_stratum": total_in_stratum,
        "line_count_metric": LINE_COUNT_METRIC,
        "line_count_basis": "LCASummarizationBenchmark.split_context background",
        "n_rows_total": n_rows_total,
        "datasets_version": _datasets_version(),
        "selected_content_sha256": selected_sha256,
        "example_indices": list(selected_indices),
    }


def sample_dataset(
    dataset_name: str,
    split: str,
    tiers: list[tuple[str, int, int]],
    n: int,
    seed: int,
) -> list[dict]:
    """Load the full LCA split, bucket it, and sample every requested tier."""
    dataset = load_dataset(dataset_name, split=split)
    pools: dict[str, list[int]] = {label: [] for label, _, _ in tiers}
    for index in range(len(dataset)):
        label = assign_tier(background_line_count(dataset[index]), tiers)
        if label is not None:
            pools[label].append(index)

    manifests = []
    for label, lo, hi in tiers:
        pool = pools[label]
        selected = stratified_sample(pool, n, tier_seed(seed, label))
        manifests.append(
            _build_manifest(
                dataset_name=dataset_name,
                split=split,
                tier_label=label,
                lo=lo,
                hi=hi,
                seed=seed,
                n_requested=n,
                selected_indices=selected,
                total_in_stratum=len(pool),
                n_rows_total=len(dataset),
                selected_sha256=selected_content_sha256(dataset, selected),
            )
        )
        print(f"  lca {label:<16} {len(selected):>4}/{len(pool):<4} sampled")
    return manifests


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Sample LCA rows by natural background line count.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--dataset-path", default=DEFAULT_DATASET)
    parser.add_argument("--split", default=DEFAULT_SPLIT)
    parser.add_argument("--tiers", default=DEFAULT_TIERS)
    parser.add_argument("--n", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    args = parser.parse_args()

    try:
        tiers = parse_tiers(args.tiers)
    except ValueError as exc:
        parser.error(f"--tiers: {exc}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifests = sample_dataset(args.dataset_path, args.split, tiers, args.n, args.seed)
    for manifest in manifests:
        path = out_dir / f"lca_{manifest['tier']}.json"
        path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {len(manifests)} manifest(s) to {out_dir}/")


if __name__ == "__main__":
    main()
