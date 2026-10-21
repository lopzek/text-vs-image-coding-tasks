#!/usr/bin/env python3
"""Sample LongCodeQA rows stratified by repository line count.

The natural-length axis is ``len(repo_text.splitlines())``, matching the
LoC census and the exact background returned by
``LongCodeQABenchmark.split_context``. Each manifest pins the selected row
indices, an aggregate repo/question/correct-letter content hash, and an
aligned per-row ``repo_sha256`` list used for repository-clustered analysis.

The generic tier and RNG helpers are imported from the LCC sampler. This
keeps half-open tier assignment, tier-specific RNG streams, and prefix-stable
sampling identical across the sampler family.
"""

# ruff: noqa: E402 -- repo-root bootstrap must precede worktree-local imports

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import datasets
from huggingface_hub import hf_hub_download

from codecompress.benchmarks.lcc import compute_selected_sha256
from codecompress.benchmarks.longcodeqa import (
    load_longcodeqa_split,
    longcodeqa_audit_contents,
)
from scripts.sample_lcc_by_line_count import (
    assign_tier,
    line_count,  # noqa: F401 -- intentionally re-exported v1 contract
    parse_tiers,
    stratified_sample,
    tier_seed,
)


DEFAULT_DATASET = "Steefano/LCB"
DEFAULT_SPLIT = "train"
DEFAULT_TIERS = "0-5000,5000-10000,10000-20000,20000-999999"
DEFAULT_OUT_DIR = "data/lengthstrat_longcodeqa"
LINE_COUNT_METRIC = "len(repo_text.splitlines())"


def background_line_count(row: dict) -> int:
    """Count natural lines in the exact LongCodeQA repository background."""
    return len(row["repo_text"].splitlines())


def repo_sha256(repo_text: str) -> str:
    """Return the repository-cluster digest."""
    return hashlib.sha256(repo_text.encode("utf-8", "surrogatepass")).hexdigest()


def selected_content_sha256(dataset, indices: list[int]) -> str:
    """Hash selected repo/question/letter triples in manifest order."""
    return compute_selected_sha256(longcodeqa_audit_contents(dataset.select(indices)))


def build_manifest(
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
    selected_content_hash: str,
    repo_hashes: list[str],
) -> dict:
    """Assemble one JSON-serializable LongCodeQA tier manifest."""
    selected = list(selected_indices)
    hashes = list(repo_hashes)
    if len(hashes) != len(selected):
        raise ValueError("repo_sha256 must be aligned one-to-one with example_indices")
    return {
        "task": "longcodeqa",
        "split": split,
        "tier": tier_label,
        "line_lo": lo,
        "line_hi": hi,
        "seed": seed,
        "n_requested": n_requested,
        "n_selected": len(selected),
        "total_in_stratum": total_in_stratum,
        "line_count_metric": LINE_COUNT_METRIC,
        "line_count_basis": "repo_text (LongCodeQABenchmark background)",
        "dataset": dataset_name,
        "n_rows_total": n_rows_total,
        "datasets_version": datasets_version,
        "selected_content_sha256": selected_content_hash,
        "repo_sha256": hashes,
        "example_indices": selected,
    }


def sample_split(
    dataset_name: str,
    split: str,
    tiers: list[tuple[str, int, int]],
    n: int,
    seed: int,
) -> list[dict]:
    """Load the full LongCodeQA split and sample every requested tier."""
    ds = load_longcodeqa_split(
        dataset_name,
        split,
        hub_download=hf_hub_download,
        dataset_loader=datasets.load_dataset,
    )
    n_rows_total = len(ds)
    datasets_version = str(getattr(datasets, "__version__", "unknown"))

    pools: dict[str, list[int]] = {label: [] for label, _, _ in tiers}
    for index in range(n_rows_total):
        label = assign_tier(background_line_count(ds[index]), tiers)
        if label is not None:
            pools[label].append(index)

    manifests = []
    for label, lo, hi in tiers:
        pool = pools[label]
        selected = stratified_sample(pool, n, tier_seed(seed, label))
        content_hash = selected_content_sha256(ds, selected)
        repo_hashes = [repo_sha256(ds[index]["repo_text"]) for index in selected]
        manifests.append(
            build_manifest(
                split=split,
                tier_label=label,
                lo=lo,
                hi=hi,
                seed=seed,
                n_requested=n,
                selected_indices=selected,
                total_in_stratum=len(pool),
                dataset_name=dataset_name,
                n_rows_total=n_rows_total,
                datasets_version=datasets_version,
                selected_content_hash=content_hash,
                repo_hashes=repo_hashes,
            )
        )
        print(f"  {label:<14} {len(selected):>4}/{len(pool):<4} sampled")
    return manifests


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Sample LongCodeQA rows by natural repository line count.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--dataset-path", default=DEFAULT_DATASET)
    parser.add_argument("--split", default=DEFAULT_SPLIT)
    parser.add_argument("--n", type=int, default=200, help="Examples to sample per tier.")
    parser.add_argument("--seed", type=int, default=42, help="Base random seed.")
    parser.add_argument(
        "--tiers", default=DEFAULT_TIERS, help="Comma-separated half-open line tiers."
    )
    parser.add_argument(
        "--out-dir", default=DEFAULT_OUT_DIR, help="Directory for per-tier manifest JSON files."
    )
    args = parser.parse_args()

    try:
        tiers = parse_tiers(args.tiers)
    except ValueError as exc:
        parser.error(f"--tiers: {exc}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Tiers (half-open, line count): {[tier[0] for tier in tiers]}")
    print(f"Sampling n={args.n} per tier, seed={args.seed}, split={args.split}\n")

    manifests = sample_split(
        args.dataset_path,
        args.split,
        tiers,
        args.n,
        args.seed,
    )
    written = []
    for manifest in manifests:
        path = out_dir / f"longcodeqa_{manifest['tier']}.json"
        path.write_text(json.dumps(manifest, indent=2) + "\n")
        written.append(path)
    print(f"\nWrote {len(written)} manifest(s) to {out_dir}/")


if __name__ == "__main__":
    main()
