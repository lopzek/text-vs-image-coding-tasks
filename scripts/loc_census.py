#!/usr/bin/env python3
"""Generate the deterministic natural-length (lines of code) census of the four datasets.

Line statistics always cover the complete selected dataset split.  ``--limit``
only limits the token pass (per language for LCC); row files and token
statistics are marked partial exactly when the selected token pool is smaller
than its complete line pool.

LCC statistics are reported only under ``by_language.<lang>``; no pooled
three-language statistics are emitted at the top level.  LCC row identity is
the composite ``(lang, example_id)`` because ``example_id`` intentionally
restarts at zero in every language split.

The four data paths deliberately mirror the benchmark inputs:

* LCC: raw ``context`` and the LCC sampler's literal-LF line metric;
* RepoBench: normalized rendered cross-file prompt and the same line metric;
* LCA: raw, unfiltered test split passed through ``split_context()``;
* LongCodeQA: the locally cached zip passed through ``split_context()``.

No network fallback exists.  Each multiprocessing worker loads its own pinned
tokenizer because Hugging Face fast tokenizers are not shared between workers.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing
import os
import socket
import statistics
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

# Offline policy must be established before importing datasets/transformers or
# the LCC sampler (which itself imports datasets).
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

try:  # imported as ``scripts.loc_census`` (module use)
    from scripts.sample_lcc_by_line_count import line_count as lcc_line_count
except ModuleNotFoundError:  # executed as ``python scripts/loc_census.py``
    from sample_lcc_by_line_count import line_count as lcc_line_count


REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUT_DIR = REPO_ROOT / "census"

TOKENIZER_NAME = "Qwen/Qwen3.6-27B"
TOKENIZER_REVISION = "6a9e13bd6fc8f0983b9b99948120bc37f49c13e9"

LCC_DATASET_TEMPLATE = "microsoft/LCC_{lang}"
REPOBENCH_DATASET = "tianyang/repobench_python_v1.1"
LCA_DATASET = "JetBrains-Research/lca-module-summarization"
LONGCODEQA_DATASET = "Steefano/LCB"

LCC_LANGUAGES = ("python", "java", "csharp")
TASK_ORDER = ("lcc", "repobench", "lca", "longcodeqa")

DATASET_IDS = {
    "repobench": REPOBENCH_DATASET,
    "lca": LCA_DATASET,
    "longcodeqa": LONGCODEQA_DATASET,
}
DATASET_SPLITS = {
    "lcc": "test",
    "repobench": "cross_file_first",
    "lca": "test",
    "longcodeqa": "train",
}


@dataclass(frozen=True)
class TaskSpec:
    bin_edges: tuple[int, ...]
    usable_window: int
    line_count_metric: str


TASK_SPECS = {
    "lcc": TaskSpec(
        (0, 200, 300, 500, 1000, 2000, 5000),
        32768 - 128,
        "raw context; scripts/sample_lcc_by_line_count.py::line_count "
        "(count('\\n') + (0 if endswith('\\n') else 1))",
    ),
    "repobench": TaskSpec(
        (0, 50, 100, 200, 300, 500, 1000, 2000, 5000),
        32768 - 128,
        "normalize_blanklines(repobench_cross_file_prompt(row, 'python')); "
        "scripts/sample_lcc_by_line_count.py::line_count",
    ),
    "lca": TaskSpec(
        (0, 200, 500, 1000, 2000, 5000, 10000),
        32768 - 128,
        "LCASummarizationBenchmark.split_context(row)[0]; str.splitlines()",
    ),
    "longcodeqa": TaskSpec(
        (0, 1000, 2000, 5000, 10000, 20000, 50000),
        32768 - 2048,
        "LongCodeQABenchmark.split_context(row)[0]; str.splitlines()",
    ),
}

PERCENTILES = (5, 10, 25, 50, 75, 90, 95, 99)


@dataclass(frozen=True)
class SourceRow:
    """One stable dataset row before tokenization.

    ``example_id`` is the row key except for LCC, whose row key is the
    composite ``(lang, example_id)``.
    """

    task: str
    example_id: int | str
    text: str
    lines: int
    lang: str | None = None
    repo_sha256: str | None = None


def count_task_lines(task: str, text: str) -> int:
    """Return the task's natural-line metric."""
    if task in {"lcc", "repobench"}:
        return lcc_line_count(text)
    if task in {"lca", "longcodeqa"}:
        return len(text.splitlines())
    raise ValueError(f"unknown task: {task}")


def linear_percentile(values: Sequence[int | float], percentile: float) -> float:
    """NumPy ``percentile(method='linear')`` without a NumPy dependency."""
    if not values:
        raise ValueError("percentile needs at least one value")
    if not 0 <= percentile <= 100:
        raise ValueError("percentile must be in [0, 100]")
    ordered = sorted(values)
    position = (len(ordered) - 1) * percentile / 100.0
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return float(ordered[lower] + (ordered[upper] - ordered[lower]) * fraction)


def line_percentiles(values: Sequence[int]) -> dict[str, int | float]:
    if not values:
        raise ValueError("line census cannot be empty")
    result: dict[str, int | float] = {
        f"p{percentile}": linear_percentile(values, percentile) for percentile in PERCENTILES
    }
    result["max"] = max(values)
    return result


def line_histogram(values: Sequence[int], edges: Sequence[int]) -> list[dict[str, int | None]]:
    """Count half-open bins ``[lo, hi)`` plus an open-ended final bin."""
    if not edges or tuple(edges) != tuple(sorted(set(edges))):
        raise ValueError("histogram edges must be non-empty, unique, and sorted")
    counts = [0] * len(edges)
    for value in values:
        if value < edges[0]:
            raise ValueError(f"line count {value} is below first edge {edges[0]}")
        for index, lo in enumerate(edges):
            hi = edges[index + 1] if index + 1 < len(edges) else None
            if value >= lo and (hi is None or value < hi):
                counts[index] += 1
                break
    return [
        {
            "lo": lo,
            "hi": edges[index + 1] if index + 1 < len(edges) else None,
            "count": counts[index],
        }
        for index, lo in enumerate(edges)
    ]


def is_over_window(tokens: int, usable_window: int) -> bool:
    return tokens > usable_window


def repo_sha256(repo_text: str) -> str:
    return hashlib.sha256(repo_text.encode("utf-8", "surrogatepass")).hexdigest()


def pool_content_sha256(rows: Sequence[SourceRow]) -> str:
    """Hash the ordered full input pool (same byte protocol as the manifests)."""
    digest = hashlib.sha256()
    for row in sorted(rows, key=lambda item: item.example_id):
        digest.update(row.text.encode("utf-8", "surrogatepass"))
        digest.update(b"\x00")
    return digest.hexdigest()


def _dataset_semantics(
    task: str, full_rows: Sequence[SourceRow]
) -> tuple[str | dict[str, str], str, str | dict[str, str]]:
    """Return deterministic dataset id/split/content provenance."""
    if task != "lcc":
        return (
            DATASET_IDS[task],
            DATASET_SPLITS[task],
            pool_content_sha256(full_rows),
        )

    languages = [
        language for language in LCC_LANGUAGES if any(row.lang == language for row in full_rows)
    ]
    dataset_ids = {language: LCC_DATASET_TEMPLATE.format(lang=language) for language in languages}
    fingerprints = {
        language: pool_content_sha256([row for row in full_rows if row.lang == language])
        for language in languages
    }
    return dataset_ids, DATASET_SPLITS[task], fingerprints


def select_token_rows(task: str, rows: Sequence[SourceRow], limit: int | None) -> list[SourceRow]:
    """Select the stable token pool while leaving the line pool untouched."""
    if limit is None:
        return list(rows)
    if limit <= 0:
        raise ValueError("limit must be positive")
    if task != "lcc":
        return list(rows[:limit])

    # The rule is N per selected language, preserving the language order and
    # the original row order inside each split.
    selected: list[SourceRow] = []
    for language in LCC_LANGUAGES:
        language_rows = [row for row in rows if row.lang == language]
        selected.extend(language_rows[:limit])
    return selected


def _token_stats(
    rows: Sequence[SourceRow],
    token_counts: Sequence[int],
    *,
    partial: bool,
    usable_window: int,
) -> dict[str, bool | int | float | None]:
    if len(rows) != len(token_counts):
        raise ValueError("token row/count length mismatch")
    ratios = [
        tokens / row.lines for row, tokens in zip(rows, token_counts, strict=True) if row.lines > 0
    ]
    return {
        "partial": partial,
        "median_tokens_per_line": statistics.median(ratios) if ratios else None,
        "over_window_count": sum(is_over_window(tokens, usable_window) for tokens in token_counts),
    }


def _output_row(row: SourceRow, tokens: int, usable_window: int) -> dict[str, Any]:
    """Serialize one row; LCC consumers key on ``(lang, example_id)``."""
    output: dict[str, Any] = {
        "task": row.task,
        "example_id": row.example_id,
        "lines": row.lines,
        "tokens": tokens,
        "over_window": is_over_window(tokens, usable_window),
    }
    if row.task == "lcc":
        if row.lang is None:
            raise ValueError("LCC rows require lang")
        output["lang"] = row.lang
    if row.task == "longcodeqa":
        if row.repo_sha256 is None:
            raise ValueError("LongCodeQA rows require repo_sha256")
        output["repo_sha256"] = row.repo_sha256
    return output


def build_task_outputs(
    task: str,
    full_rows: Sequence[SourceRow],
    token_rows: Sequence[SourceRow],
    token_counts: Sequence[int],
    *,
    limit: int | None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Build deterministic summary and per-tokenized-row artifacts."""
    if task not in TASK_SPECS:
        raise ValueError(f"unknown task: {task}")
    if not full_rows:
        raise ValueError(f"{task}: dataset split is empty")
    if len(token_rows) != len(token_counts):
        raise ValueError("token row/count length mismatch")
    if any(row.task != task for row in full_rows) or any(row.task != task for row in token_rows):
        raise ValueError("source row task mismatch")

    spec = TASK_SPECS[task]
    line_values = [row.lines for row in full_rows]
    dataset_id, dataset_split, content_fingerprint = _dataset_semantics(task, full_rows)
    summary: dict[str, Any] = {
        "task": task,
        "dataset_id": dataset_id,
        "dataset_split": dataset_split,
        "pool_content_sha256": content_fingerprint,
        "n_total": len(full_rows),
        "n_tokenized": len(token_rows),
        "limit": limit,
        "line_count_metric": spec.line_count_metric,
        "line_percentiles": line_percentiles(line_values),
        "line_histogram": line_histogram(line_values, spec.bin_edges),
        "tokenizer": {
            "name": TOKENIZER_NAME,
            "revision": TOKENIZER_REVISION,
        },
        "usable_window": spec.usable_window,
        "token_stats": _token_stats(
            token_rows,
            token_counts,
            partial=len(token_rows) < len(full_rows),
            usable_window=spec.usable_window,
        ),
    }

    if task == "lcc":
        token_pairs = list(zip(token_rows, token_counts, strict=True))
        by_language: dict[str, Any] = {}
        for language in LCC_LANGUAGES:
            language_full = [row for row in full_rows if row.lang == language]
            if not language_full:
                continue
            language_pairs = [(row, tokens) for row, tokens in token_pairs if row.lang == language]
            language_token_rows = [row for row, _ in language_pairs]
            language_counts = [tokens for _, tokens in language_pairs]
            language_lines = [row.lines for row in language_full]
            by_language[language] = {
                "dataset_id": LCC_DATASET_TEMPLATE.format(lang=language),
                "dataset_split": DATASET_SPLITS[task],
                "pool_content_sha256": pool_content_sha256(language_full),
                "n_total": len(language_full),
                "n_tokenized": len(language_token_rows),
                "line_percentiles": line_percentiles(language_lines),
                "line_histogram": line_histogram(language_lines, spec.bin_edges),
                "token_stats": _token_stats(
                    language_token_rows,
                    language_counts,
                    partial=len(language_token_rows) < len(language_full),
                    usable_window=spec.usable_window,
                ),
            }
        summary["by_language"] = by_language
        # Three-language pooled statistics are not meaningful.  Shared
        # semantic fields stay at the top level; all statistics live
        # exclusively under by_language.<lang>.
        for field in (
            "n_total",
            "n_tokenized",
            "line_percentiles",
            "line_histogram",
            "token_stats",
        ):
            del summary[field]

    output_rows = [
        _output_row(row, tokens, spec.usable_window)
        for row, tokens in zip(token_rows, token_counts, strict=True)
    ]
    return summary, output_rows


def _json_text(value: Any) -> str:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
            allow_nan=False,
        )
        + "\n"
    )


def write_task_artifacts(
    out_dir: Path | str,
    task: str,
    summary: dict[str, Any],
    rows: Sequence[dict[str, Any]],
) -> dict[str, Path]:
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    summary_path = out_path / f"census_{task}.json"
    rows_path = out_path / f"census_rows_{task}.jsonl"
    summary_path.write_text(_json_text(summary), encoding="utf-8")
    rows_path.write_text(
        "".join(
            json.dumps(
                row,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            )
            + "\n"
            for row in rows
        ),
        encoding="utf-8",
    )
    return {"summary": summary_path, "rows": rows_path}


def update_meta(out_dir: Path | str, task_entries: dict[str, dict[str, Any]]) -> Path:
    """Incrementally update the allowed-to-vary provenance sidecar."""
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    meta_path = out_path / "census_meta.json"
    if meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    else:
        meta = {"schema_version": 1, "tasks": {}}
    if not isinstance(meta.get("tasks"), dict):
        raise ValueError(f"{meta_path}: tasks must be an object")
    meta["schema_version"] = 1
    meta["tasks"].update(task_entries)
    # FIX1 migrates legacy singleton strings as well as normalizing new
    # entries, so incremental updates cannot leave a mixed str/list schema.
    for task, task_entry in meta["tasks"].items():
        if not isinstance(task_entry, dict):
            raise ValueError(f"{meta_path}: task {task!r} metadata must be an object")
        revisions = task_entry.get("dataset_cache_revisions")
        if revisions is None:
            continue
        if not isinstance(revisions, dict):
            raise ValueError(
                f"{meta_path}: task {task!r} dataset_cache_revisions must be an object"
            )
        normalized: dict[str, list[str]] = {}
        for repo_id, commits in revisions.items():
            if not isinstance(repo_id, str):
                raise ValueError(f"{meta_path}: dataset revision key must be a string")
            if isinstance(commits, str):
                commit_list = [commits]
            elif isinstance(commits, list) and all(isinstance(commit, str) for commit in commits):
                commit_list = commits
            else:
                raise ValueError(
                    f"{meta_path}: dataset revisions for {repo_id!r} must be string(s)"
                )
            normalized[repo_id] = sorted(set(commit_list))
        task_entry["dataset_cache_revisions"] = dict(sorted(normalized.items()))
    meta_path.write_text(_json_text(meta), encoding="utf-8")
    return meta_path


# ---------------------------------------------------------------------------
# Data adapters.  Thin wrappers make the offline paths independently testable.


def _load_dataset(*args: Any, **kwargs: Any):
    import datasets

    return datasets.load_dataset(*args, **kwargs)


def _hf_hub_download(*args: Any, **kwargs: Any):
    from huggingface_hub import hf_hub_download

    return hf_hub_download(*args, **kwargs)


def _make_lca_benchmark():
    from codecompress.benchmarks.lca_summarization import LCASummarizationBenchmark

    return LCASummarizationBenchmark()


def _make_longcodeqa_benchmark():
    from codecompress.benchmarks.longcodeqa import LongCodeQABenchmark

    return LongCodeQABenchmark()


def load_lcc_rows(lcc_lang: str = "all") -> list[SourceRow]:
    languages = LCC_LANGUAGES if lcc_lang == "all" else (lcc_lang,)
    rows: list[SourceRow] = []
    for language in languages:
        dataset = _load_dataset(LCC_DATASET_TEMPLATE.format(lang=language), split="test")
        for example_id, example in enumerate(dataset):
            text = example.get("context", "") or ""
            rows.append(
                SourceRow(
                    task="lcc",
                    example_id=example_id,
                    text=text,
                    lines=count_task_lines("lcc", text),
                    lang=language,
                )
            )
    return rows


def load_repobench_rows() -> list[SourceRow]:
    from codecompress.benchmarks.repobench import (
        normalize_blanklines,
        repobench_cross_file_prompt,
    )

    dataset = _load_dataset(REPOBENCH_DATASET, split="cross_file_first")
    rows: list[SourceRow] = []
    for example_id, example in enumerate(dataset):
        text = normalize_blanklines(repobench_cross_file_prompt(example, "python"))
        rows.append(
            SourceRow(
                task="repobench",
                example_id=example_id,
                text=text,
                lines=count_task_lines("repobench", text),
            )
        )
    return rows


def load_lca_rows() -> list[SourceRow]:
    # Intentionally bypass LCASummarizationBenchmark.load(): it hard-filters at
    # 32768 Qwen2.5 tokens and would silently remove about 31% of the raw pool.
    dataset = _load_dataset(LCA_DATASET, split="test")
    benchmark = _make_lca_benchmark()
    rows: list[SourceRow] = []
    for example_id, example in enumerate(dataset):
        text, _query = benchmark.split_context(example)
        text = text or ""
        rows.append(
            SourceRow(
                task="lca",
                example_id=example_id,
                text=text,
                lines=count_task_lines("lca", text),
            )
        )
    return rows


def load_longcodeqa_rows() -> list[SourceRow]:
    local_zip = _hf_hub_download(
        LONGCODEQA_DATASET,
        "LongCodeQA.zip",
        repo_type="dataset",
        local_files_only=True,
    )
    dataset = _load_dataset(
        "json",
        data_files=f"zip://LQA/*.json::{local_zip}",
        split="train",
    )
    benchmark = _make_longcodeqa_benchmark()
    rows: list[SourceRow] = []
    for example_id, example in enumerate(dataset):
        text, _query = benchmark.split_context(example)
        text = text or ""
        rows.append(
            SourceRow(
                task="longcodeqa",
                example_id=example_id,
                text=text,
                lines=count_task_lines("longcodeqa", text),
                repo_sha256=repo_sha256(text),
            )
        )
    return rows


def load_task_rows(task: str, lcc_lang: str = "all") -> list[SourceRow]:
    if task == "lcc":
        return load_lcc_rows(lcc_lang)
    if task == "repobench":
        return load_repobench_rows()
    if task == "lca":
        return load_lca_rows()
    if task == "longcodeqa":
        return load_longcodeqa_rows()
    raise ValueError(f"unknown task: {task}")


# ---------------------------------------------------------------------------
# Token pass.


def load_tokenizer(auto_tokenizer_cls=None):
    if auto_tokenizer_cls is None:
        from transformers import AutoTokenizer

        auto_tokenizer_cls = AutoTokenizer
    return auto_tokenizer_cls.from_pretrained(
        TOKENIZER_NAME,
        revision=TOKENIZER_REVISION,
        local_files_only=True,
    )


def token_count(tokenizer: Any, text: str) -> int:
    return len(tokenizer.encode(text, add_special_tokens=False))


_WORKER_TOKENIZER: Any | None = None


def _init_token_worker() -> None:
    global _WORKER_TOKENIZER
    _WORKER_TOKENIZER = load_tokenizer()


def _worker_token_count(text: str) -> int:
    if _WORKER_TOKENIZER is None:  # pragma: no cover - process invariant
        raise RuntimeError("token worker was not initialized")
    return token_count(_WORKER_TOKENIZER, text)


def count_tokens(texts: Sequence[str], workers: int) -> list[int]:
    if workers <= 0:
        raise ValueError("workers must be positive")
    if not texts:
        return []

    # QA has 443 questions but only 98 unique repositories.  Deduplication is
    # exact and preserves per-row output while avoiding repeated million-token
    # work on identical repo_text values.
    unique_texts = list(dict.fromkeys(texts))
    if workers == 1:
        tokenizer = load_tokenizer()
        unique_counts = [token_count(tokenizer, text) for text in unique_texts]
    else:
        process_count = min(workers, len(unique_texts))
        chunksize = max(1, len(unique_texts) // (process_count * 4))
        # Python 3.14 defaults to ``forkserver`` on POSIX.  That context needs
        # a listener socket, which is unavailable in the executor sandbox and
        # fails before the tokenizer initializer runs.  ``spawn`` avoids both
        # that socket and forking a potentially multi-threaded datasets parent;
        # every child still loads an independent tokenizer in the initializer.
        context = multiprocessing.get_context("spawn")
        with context.Pool(
            processes=process_count,
            initializer=_init_token_worker,
        ) as pool:
            unique_counts = list(pool.imap(_worker_token_count, unique_texts, chunksize=chunksize))
    by_text = dict(zip(unique_texts, unique_counts, strict=True))
    return [by_text[text] for text in texts]


# ---------------------------------------------------------------------------
# CLI and volatile provenance.


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate the deterministic natural-length census.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--task", choices=(*TASK_ORDER, "all"), required=True)
    parser.add_argument("--lcc-lang", choices=(*LCC_LANGUAGES, "all"), default="all")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument(
        "--limit",
        type=_positive_int,
        default=None,
        help="Limit only the token pass (per language for LCC); line stats stay full.",
    )
    parser.add_argument(
        "--workers",
        type=_positive_int,
        default=min(32, os.cpu_count() or 1),
        help="Tokenizer worker processes; each independently loads the tokenizer.",
    )
    return parser.parse_args(argv)


def best_effort_dataset_cache_revisions(
    scan_cache_fn=None,
) -> dict[str, list[str]]:
    """Return locally visible dataset snapshot commits without ever failing."""
    dataset_ids = {
        *DATASET_IDS.values(),
        *(LCC_DATASET_TEMPLATE.format(lang=lang) for lang in LCC_LANGUAGES),
    }
    try:
        if scan_cache_fn is None:
            from huggingface_hub import scan_cache_dir

            scan_cache_fn = scan_cache_dir
        cache_info = scan_cache_fn()
        result: dict[str, list[str]] = {}
        for repo in cache_info.repos:
            if repo.repo_type != "dataset" or repo.repo_id not in dataset_ids:
                continue
            commits = sorted({revision.commit_hash for revision in repo.revisions})
            if commits:
                result[repo.repo_id] = commits
        return dict(sorted(result.items()))
    except Exception:
        # This is explicitly best-effort volatile provenance.  Missing or
        # unreadable cache metadata must not block an otherwise local census.
        return {}


def runtime_meta(workers: int, command: Sequence[str]) -> dict[str, Any]:
    try:
        git_commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=REPO_ROOT,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except OSError:
        git_commit = "unknown"
    except subprocess.CalledProcessError:
        git_commit = "unknown"
    meta = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "git_commit": git_commit,
        "host": socket.gethostname(),
        "workers": workers,
        "command": list(command),
    }
    dataset_revisions = best_effort_dataset_cache_revisions()
    if dataset_revisions:
        meta["dataset_cache_revisions"] = dataset_revisions
    return meta


def _command_for_meta(argv: Sequence[str] | None) -> list[str]:
    if argv is None:
        return list(sys.argv)
    return ["scripts/loc_census.py", *argv]


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    tasks = TASK_ORDER if args.task == "all" else (args.task,)
    command = _command_for_meta(argv)
    meta_entry = runtime_meta(args.workers, command)

    for task in tasks:
        print(f"[{task}] loading full line census", flush=True)
        full_rows = load_task_rows(task, lcc_lang=args.lcc_lang)
        token_rows = select_token_rows(task, full_rows, args.limit)
        print(
            f"[{task}] tokenizing {len(token_rows)}/{len(full_rows)} rows "
            f"with {args.workers} worker(s)",
            flush=True,
        )
        token_counts = count_tokens([row.text for row in token_rows], workers=args.workers)
        summary, output_rows = build_task_outputs(
            task,
            full_rows,
            token_rows,
            token_counts,
            limit=args.limit,
        )
        paths = write_task_artifacts(args.out_dir, task, summary, output_rows)
        # Update provenance immediately after this task's two data artifacts.
        # If a later task in ``--task all`` fails, completed earlier artifacts
        # must not be left paired with stale or missing metadata.
        update_meta(args.out_dir, {task: dict(meta_entry)})
        print(
            f"[{task}] wrote {paths['summary']} and {paths['rows']}",
            flush=True,
        )

    print(f"wrote {Path(args.out_dir) / 'census_meta.json'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
