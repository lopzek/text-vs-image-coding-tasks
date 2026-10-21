#!/usr/bin/env python3
"""Per-instance scores of one finished cell, as the paper uses them.

Reads the JSONL that ``codecompress.main`` wrote for the cell and writes ``scores.jsonl`` next to
it with, per instance: ``score`` (the paper's metric on 0-100), ``delivered_in_full`` (the whole
prompt fitted the context window in this modality), the served input tokens, and the input
tokens before any window fitting. It prints the cell summary.

The score is the benchmark's own metric. On the two completion tasks a markdown code fence
around the model's output is removed before the edit similarity is computed (0.2% of the
outputs carried one); summarization (BERTScore F1) and question answering (accuracy) use the
stored score. An instance that errored during generation carries ``score: null``.

Usage: python score_cell.py runs/main/lcc_gem_300-500_ip [more run directories ...]
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

TASK_OF_BENCHMARK = {"lcc": "lcc", "repobench": "rb", "lca_summarization": "lca", "longcodeqa": "qa"}
METRIC_FIELD = {"lcc": "es", "rb": "es", "lca": "bertscore_f1", "qa": "accuracy"}


def strip_fences(output: str) -> str:
    """The content of the first markdown code fence if the output carries one (prose outside the
    fence was never part of the code); an unclosed fence keeps everything after its opening;
    fence markers without a parseable block are dropped line-wise; fence-free output is unchanged."""
    if "```" not in output:
        return output
    m = re.search(r"```[A-Za-z0-9_+-]*[ \t]*\n(.*?)(?:\n[ \t]*```|$)", output, re.DOTALL)
    if m:
        return m.group(1)
    return "\n".join(line for line in output.splitlines() if not line.strip().startswith("```"))


def score_of(task: str, row: dict) -> float | None:
    out = row.get("output") or ""
    if out.startswith("ERROR"):
        return None
    if task == "lcc":
        from codecompress.benchmarks.lcc import compute_ES
        return float(compute_ES(row["ground_truth"], strip_fences(out)))
    if task == "rb":
        from codecompress.benchmarks.repobench import compute_repobench_ES, get_first_line_not_comment
        first = get_first_line_not_comment(strip_fences(out), language="python")
        return float(compute_repobench_ES(first, row["ground_truth"]))
    v = row.get(METRIC_FIELD[task])
    return None if v is None else float(v)


def load_run(run_dir: Path) -> tuple[str, list[dict]]:
    scores = sorted(run_dir.glob("method_*/*-SCORES.json"))
    if not scores:
        raise SystemExit(f"{run_dir}: no method_*/<model>-SCORES.json (has the cell finished?)")
    task = TASK_OF_BENCHMARK[json.loads(scores[0].read_text())["benchmark"]]
    jsonl = [p for p in sorted(run_dir.glob("method_*/*.jsonl")) if not p.name.endswith("SCORES.json")]
    rows = [json.loads(ln) for ln in jsonl[0].read_text().splitlines() if ln.strip()]
    return task, rows


def score_run(run_dir: Path) -> dict:
    task, rows = load_run(run_dir)
    recs = []
    for r in rows:
        wa = r.get("window_audit") or {}
        recs.append({
            "id": r["id"],
            "score": score_of(task, r),
            "delivered_in_full": bool(wa.get("fully_attended")),
            "input_tokens": wa.get("model_input_tokens"),
            "input_tokens_before_fit": wa.get("pre_truncation_tokens"),
        })
    (run_dir / "scores.jsonl").write_text("".join(json.dumps(x) + "\n" for x in recs))
    scored = [x["score"] for x in recs if x["score"] is not None]
    summary = {
        "run": run_dir.name, "task": task, "n": len(recs), "n_scored": len(scored),
        "delivered_in_full": sum(x["delivered_in_full"] for x in recs),
        "mean_score": sum(scored) / len(scored) if scored else None,
        "mean_input_tokens": sum(x["input_tokens"] for x in recs if x["input_tokens"] is not None)
        / max(1, sum(1 for x in recs if x["input_tokens"] is not None)),
    }
    print(json.dumps(summary))
    return summary


if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise SystemExit(__doc__)
    for d in sys.argv[1:]:
        score_run(Path(d))
