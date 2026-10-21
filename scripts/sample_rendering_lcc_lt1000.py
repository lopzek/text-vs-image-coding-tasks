#!/usr/bin/env python3
"""Rendering-ablation instance set: 500 LCC items below 1,000 LoC, drawn from the main grid's sample.

The pool is the union of the three main-grid LCC manifests under 1,000 lines
(``manifests/main/lcc/python_{200-300,300-500,500-1000}.json``, 1,500 dataset
row indices). 500 of them are drawn WITHOUT replacement with
``numpy.random.default_rng(42).choice(sorted_pool, 500, replace=False)`` and
written in ascending dataset-index order to
``manifests/rendering/python_lt1000_n500.json`` in the main-grid manifest format
(every key, same order and meaning; ``selected_content_sha256`` is the
same byte protocol over the selected contexts, so the run-time hash check in
``codecompress.benchmarks.lcc`` applies). A ``derived_from`` block records, per
position, the dataset index, the natural length in lines and the source
main-grid (bin, position) -- the join key to the main grid's text runs, whose
per-row ``id`` is the position inside the bin manifest.

Usage (research environment, HF cache offline):
    HF_HUB_OFFLINE=1 python scripts/sample_rendering_lcc_lt1000.py [--check]
``--check`` re-reads an existing manifest and asserts it equals a fresh draw.
"""

# ruff: noqa: E402 -- repo-root bootstrap must precede worktree-local imports
from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.sample_lcc_by_line_count import line_count

SEED = 42
N_DEFAULT = 500
SOURCE_BINS = ("200-300", "300-500", "500-1000")
SOURCE_DIR = REPO_ROOT / "manifests/main/lcc"
OUT_PATH = REPO_ROOT / "manifests/rendering/python_lt1000_n500.json"
SAMPLER_DESC = (
    "numpy.random.default_rng(42).choice(sorted pool, 500, replace=False), ids written ascending"
)


def load_source_manifests(src_dir: Path = SOURCE_DIR) -> dict[str, dict]:
    return {b: json.loads((src_dir / f"python_{b}.json").read_text()) for b in SOURCE_BINS}


def build_pool(manifests: dict[str, dict]) -> list[int]:
    """Sorted union of the source manifests' dataset indices; an id in two manifests is an error."""
    seen: dict[int, str] = {}
    for b, m in manifests.items():
        for i in m["example_indices"]:
            if i in seen:
                raise ValueError(f"dataset index {i} appears in both {seen[i]} and {b}")
            seen[i] = b
    return sorted(seen)


def source_map(manifests: dict[str, dict]) -> dict[int, tuple[str, int]]:
    """dataset index -> (source bin, position inside that bin's manifest = main-grid row id)."""
    out: dict[int, tuple[str, int]] = {}
    for b, m in manifests.items():
        for pos, i in enumerate(m["example_indices"]):
            out[i] = (b, pos)
    return out


def sample_ids(pool: list[int], n: int = N_DEFAULT, seed: int = SEED) -> list[int]:
    """n distinct ids from the sorted pool, numpy default_rng(seed), without replacement, ascending."""
    if n > len(pool):
        raise ValueError(f"cannot draw {n} from a pool of {len(pool)}")
    rng = np.random.default_rng(seed)
    draw = rng.choice(np.asarray(sorted(pool), dtype=np.int64), size=n, replace=False)
    return sorted(int(i) for i in draw)


def build_manifest(
    ids: list[int],
    manifests: dict[str, dict],
    lines: dict[int, int],
    selected_sha256: str,
    n_rows_total: int,
    datasets_version: str,
) -> dict:
    """The main-grid manifest keys (same order) for the derived sample, plus ``derived_from``."""
    ref = manifests[SOURCE_BINS[0]]
    pool = build_pool(manifests)
    src = source_map(manifests)
    man = {
        "language": ref["language"],
        "split": ref["split"],
        "tier": "lt1000",
        "line_lo": min(m["line_lo"] for m in manifests.values()),
        "line_hi": max(m["line_hi"] for m in manifests.values()),
        "seed": SEED,
        "n_requested": len(ids),
        "n_selected": len(ids),
        "total_in_stratum": len(pool),
        "line_count_metric": ref["line_count_metric"],
        "dataset": ref["dataset"],
        "n_rows_total": n_rows_total,
        "datasets_version": datasets_version,
        "selected_content_sha256": selected_sha256,
        "example_indices": list(ids),
        "pre_filter_total": len(pool),
        "filter_removed": 0,
        "filter_removed_fraction": 0.0,
        "filters_applied": dict(ref["filters_applied"]),
    }
    man["derived_from"] = {
        "phase": "rendering_ablation",
        "sampler": SAMPLER_DESC,
        "source_manifests": [f"python_{b}.json" for b in SOURCE_BINS],
        "pool_size": len(pool),
        "sources": [
            {
                "position": pos,
                "dataset_index": i,
                "lines": lines[i],
                "source_bin": src[i][0],
                "source_position": src[i][1],
            }
            for pos, i in enumerate(ids)
        ],
    }
    return man


def loc_summary(ids: list[int], lines: dict[int, int], src: dict[int, tuple[str, int]]) -> dict:
    vals = [lines[i] for i in ids]
    per_bin = {b: 0 for b in SOURCE_BINS}
    for i in ids:
        per_bin[src[i][0]] += 1
    return {
        "n": len(ids),
        "min": min(vals),
        "median": float(statistics.median(vals)),
        "max": max(vals),
        "mean": float(statistics.fmean(vals)),
        "per_source_bin": per_bin,
    }


def _dataset_lines_and_sha(ids: list[int], dataset_name: str, split: str):
    from datasets import load_dataset
    import datasets as _datasets

    from codecompress.benchmarks.lcc import compute_selected_sha256

    ds = load_dataset(dataset_name, split=split)
    contexts = [ds[i]["context"] for i in ids]
    lines = {i: line_count(c) for i, c in zip(ids, contexts)}
    return lines, compute_selected_sha256(contexts), len(ds), str(_datasets.__version__)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--n", type=int, default=N_DEFAULT)
    ap.add_argument("--out", type=Path, default=OUT_PATH)
    ap.add_argument(
        "--check", action="store_true", help="assert the existing manifest equals a fresh draw"
    )
    a = ap.parse_args(argv)
    manifests = load_source_manifests()
    pool = build_pool(manifests)
    ids = sample_ids(pool, n=a.n, seed=SEED)
    ref = manifests[SOURCE_BINS[0]]
    lines, sha, n_rows_total, dsv = _dataset_lines_and_sha(ids, ref["dataset"], ref["split"])
    man = build_manifest(ids, manifests, lines, sha, n_rows_total, dsv)
    # every sampled item must sit inside its source bin's half-open [lo, hi) line range
    for i in ids:
        b = source_map(manifests)[i][0]
        lo, hi = manifests[b]["line_lo"], manifests[b]["line_hi"]
        if not (lo <= lines[i] < hi):
            raise AssertionError(f"id {i}: {lines[i]} lines outside its source bin {b}")
    summary = loc_summary(ids, lines, source_map(manifests))
    if a.check:
        existing = json.loads(a.out.read_text())
        if existing != man:
            print("CHECK FAILED: existing manifest differs from a fresh draw", file=sys.stderr)
            return 1
        print(f"CHECK OK: {a.out} equals a fresh draw; sha256={sha}")
    else:
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(json.dumps(man, indent=2) + "\n")
        print(f"wrote {a.out}")
    print(json.dumps(summary, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
