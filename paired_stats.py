#!/usr/bin/env python3
"""The paper's per-group numbers from the scored cells (run score_cell.py on every cell first).

Main grid (RQ1-RQ3). For every task and model the instances of the four sampling groups are
re-grouped on the six LoC groups shared by all tasks (<500, 500-1K, 1K-2K, 2K-4K, 4K-8K, >=8K
lines; ``manifests/loc/<task>.jsonl`` holds each instance's LoC) and, per group:
  - feasibility: the share of instances delivered in full by each modality;
  - cost: the mean served input tokens of each modality over all instances, and a two-sided
    Wilcoxon signed-rank test on the per-instance token difference (Holm-adjusted over all
    groups with more than one instance);
  - effectiveness, on the instances delivered in full in both modalities ("pairs"): the mean of
    each modality, the mean paired difference image - text with its 95% percentile-bootstrap
    interval (100,000 resamples, seed 42), a two-sided Wilcoxon signed-rank test on the
    non-zero differences with Holm adjustment within the task family (the two completion tasks
    form one family; summarization and question answering their own), and Cliff's delta.
    Groups with fewer than 30 pairs are flagged ``robust: false`` (reported descriptively); a
    group with no pair is "not testable".

Rendering ablation (RQ4). Per model: the mean score of each rendering with its 95% bootstrap
interval (100,000 resamples, as in the paper), the paired contrast of each variant against the plain rendering (Wilcoxon, Holm over
the four contrasts), and the gap to the text input of the same instances taken from the main
grid.

Usage: python paired_stats.py [--runs runs] [--out results.json] [--iters 100000]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parent
MODELS = ("gem", "qwen38")
TASK_GROUPS = {  # the sampling groups, in order
    "lcc": ("200-300", "300-500", "500-1000", "1000+"),
    "rb": ("0-500", "500-1000", "1000-2000", "2000+"),
    "lca": ("0-1000", "1000-2000", "2000-5000", "5000+"),
    "qa": ("0-3500", "3500-10000", "10000-20000", "20000+"),
}
FAMILY = {"lcc": "completion", "rb": "completion", "lca": "summarization", "qa": "qa"}
SHARED_GROUPS = ("<500", "500-1K", "1K-2K", "2K-4K", "4K-8K", ">=8K")
RENDERING_ARMS = ("ip", "ihf", "id", "idhf", "ib")
MIN_ROBUST_PAIRS = 30
SEED = 42


def shared_group(lines: int) -> str:
    for hi, lab in ((500, "<500"), (1000, "500-1K"), (2000, "1K-2K"), (4000, "2K-4K"), (8000, "4K-8K")):
        if lines < hi:
            return lab
    return ">=8K"


def boot_means(vals, iters: int, seed: int = SEED) -> np.ndarray:
    """Percentile-bootstrap distribution of the mean (drawn in chunks of 10,000; a fresh generator per call)."""
    a = np.asarray(vals, dtype=float)
    rng = np.random.default_rng(seed)
    out, pos = np.empty(iters), 0
    while pos < iters:
        c = min(10_000, iters - pos)
        out[pos:pos + c] = a[rng.integers(0, len(a), size=(c, len(a)))].mean(axis=1)
        pos += c
    return out


def ci95(boot: np.ndarray) -> list[float]:
    return [float(x) for x in np.quantile(boot, [0.025, 0.975])]


def holm(pvals: list) -> list:
    """Holm step-down adjustment over the non-None entries (None stays None)."""
    idx = [i for i, p in enumerate(pvals) if p is not None]
    ps = [pvals[i] for i in idx]
    order = sorted(range(len(ps)), key=lambda k: ps[k])
    adj, running = [0.0] * len(ps), 0.0
    for rank, k in enumerate(order):
        running = max(running, min(1.0, (len(ps) - rank) * ps[k]))
        adj[k] = running
    out = [None] * len(pvals)
    for j, i in enumerate(idx):
        out[i] = adj[j]
    return out


def wilcoxon_p(diffs) -> float | None:
    nz = np.asarray(diffs, dtype=float)
    nz = nz[nz != 0]
    return float(stats.wilcoxon(nz, alternative="two-sided").pvalue) if nz.size else None


def cliffs_delta(x, y) -> float:
    x, y = np.asarray(x, dtype=float)[:, None], np.asarray(y, dtype=float)[None, :]
    return float(((x > y).sum() - (x < y).sum()) / (x.size * y.size))


def load_scores(path: Path) -> dict:
    if not path.exists():
        raise SystemExit(f"{path} is missing: run score_cell.py on that cell first")
    return {r["id"]: r for r in (json.loads(ln) for ln in path.read_text().splitlines() if ln.strip())}


def load_loc(task: str) -> dict:
    p = ROOT / "manifests" / "loc" / f"{task}.jsonl"
    return {(r["group"], r["id"]): r["lines"] for r in (json.loads(ln) for ln in p.read_text().splitlines() if ln.strip())}


def items_of(runs: Path, task: str, model: str) -> list[dict]:
    """One record per instance of a task x model: both modalities' scores, flags and tokens."""
    loc, items = load_loc(task), []
    for g in TASK_GROUPS[task]:
        it = load_scores(runs / "main" / f"{task}_{model}_{g}_it" / "scores.jsonl")
        ip = load_scores(runs / "main" / f"{task}_{model}_{g}_ip" / "scores.jsonl")
        for i, t in it.items():
            p = ip.get(i) or {}
            items.append({"group": g, "id": i, "lines": loc[(g, i)],
                          "txt": t["score"], "img": p.get("score"),
                          "fa_it": t["delivered_in_full"], "fa_ip": bool(p.get("delivered_in_full")),
                          "tok_it": t["input_tokens"], "tok_ip": p.get("input_tokens")})
    return items


def cell(task: str, model: str, group: str, rows: list[dict], iters: int) -> dict:
    n = len(rows)
    rec = {"task": task, "model": model, "group": group, "family": FAMILY[task], "n_items": n,
           "share_delivered_text": sum(r["fa_it"] for r in rows) / n,
           "share_delivered_image": sum(r["fa_ip"] for r in rows) / n,
           "lines_min": min(r["lines"] for r in rows), "lines_max": max(r["lines"] for r in rows)}
    for side in ("it", "ip"):
        toks = [r[f"tok_{side}"] for r in rows if r[f"tok_{side}"] is not None]
        rec[f"mean_tokens_{'text' if side == 'it' else 'image'}"] = float(np.mean(toks)) if toks else None
    both = [r for r in rows if r["tok_it"] is not None and r["tok_ip"] is not None]
    rec["p_tokens"] = wilcoxon_p([r["tok_ip"] - r["tok_it"] for r in both]) if len(both) > 1 else None
    pairs = [r for r in rows if r["fa_it"] and r["fa_ip"] and r["img"] is not None and r["txt"] is not None]
    rec["n_pairs"] = len(pairs)
    if not pairs:
        rec["testable"] = False
        return rec
    img, txt = np.array([r["img"] for r in pairs]), np.array([r["txt"] for r in pairs])
    d = img - txt
    rec.update({"testable": True, "robust": len(pairs) >= MIN_ROBUST_PAIRS,
                "mean_text": float(txt.mean()), "mean_image": float(img.mean()), "mean_diff": float(d.mean()),
                "ci95_diff": ci95(boot_means(d, iters)), "p_wilcoxon": wilcoxon_p(d), "cliffs_delta": cliffs_delta(img, txt)})
    return rec


def main_grid(runs: Path, iters: int) -> list[dict]:
    cells = []
    for task in TASK_GROUPS:
        for model in MODELS:
            groups: dict = {}
            for r in items_of(runs, task, model):
                groups.setdefault(shared_group(r["lines"]), []).append(r)
            for g in SHARED_GROUPS:
                if g in groups:
                    cells.append(cell(task, model, g, groups[g], iters))
    for fam in set(FAMILY.values()):   # effectiveness: Holm within the task family
        members = [c for c in cells if c["family"] == fam]
        for c, p in zip(members, holm([c.get("p_wilcoxon") for c in members])):
            c["p_holm"] = p
            c["m_family"] = sum(m.get("p_wilcoxon") is not None for m in members)
    for c, p in zip(cells, holm([c["p_tokens"] for c in cells])):   # cost: Holm over all groups
        c["p_tokens_holm"] = p
    return cells


def rendering(runs: Path, iters: int) -> list[dict]:
    manifest = json.loads((ROOT / "manifests" / "rendering" / "python_lt1000_n500.json").read_text())
    sources = {s["position"]: s for s in manifest["derived_from"]["sources"]}
    out = []
    for model in MODELS:
        arms = {a: load_scores(runs / "rendering" / f"{model}_{a}" / "scores.jsonl") for a in RENDERING_ARMS}
        ids = sorted(i for i in arms["ip"] if all(arms[a].get(i, {}).get("score") is not None for a in RENDERING_ARMS))
        text_runs = {g: load_scores(runs / "main" / f"lcc_{model}_{g}_it" / "scores.jsonl") for g in ("200-300", "300-500", "500-1000")}
        text = np.array([text_runs[sources[i]["source_bin"]][sources[i]["source_position"]]["score"] for i in ids])
        plain = np.array([arms["ip"][i]["score"] for i in ids])
        rec = {"model": model, "n": len(ids), "text_input_mean": float(text.mean()), "arms": {}}
        contrasts = []
        for a in RENDERING_ARMS:
            v = np.array([arms[a][i]["score"] for i in ids])
            entry = {"mean": float(v.mean()), "ci95": ci95(boot_means(v, iters)), "gap_to_text_input": float((v - text).mean())}
            if a != "ip":
                entry.update({"diff_vs_plain": float((v - plain).mean()), "p_wilcoxon": wilcoxon_p(v - plain)})
                contrasts.append(entry)
            rec["arms"][a] = entry
        for e, p in zip(contrasts, holm([e["p_wilcoxon"] for e in contrasts])):
            e["p_holm"] = p
        out.append(rec)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", default=str(ROOT / "runs"))
    ap.add_argument("--out", default=str(ROOT / "results.json"))
    ap.add_argument("--iters", type=int, default=100_000)
    ap.add_argument("--skip-rendering", action="store_true")
    a = ap.parse_args()
    runs = Path(a.runs)
    res = {"iters": a.iters, "main_grid": main_grid(runs, a.iters)}
    if not a.skip_rendering:
        res["rendering"] = rendering(runs, a.iters)
    Path(a.out).write_text(json.dumps(res, indent=1) + "\n")
    print(f"{'task':<4} {'model':<7} {'group':<7} {'n':>5} {'pairs':>5} {'del.txt':>7} {'del.img':>7} {'txt':>7} {'img':>7} {'diff':>7} {'p_holm':>7} {'delta':>7} {'tok ratio':>9}")
    for c in res["main_grid"]:
        def f(k, w=7, nd=1, c=c):
            return f"{c[k]:{w}.{nd}f}" if isinstance(c.get(k), (int, float)) else f"{'-':>{w}}"
        ratio = (c["mean_tokens_image"] / c["mean_tokens_text"]) if c.get("mean_tokens_image") and c.get("mean_tokens_text") else None
        print(f"{c['task']:<4} {c['model']:<7} {c['group']:<7} {c['n_items']:>5} {c['n_pairs']:>5} "
              f"{c['share_delivered_text']:>7.2f} {c['share_delivered_image']:>7.2f} {f('mean_text')} {f('mean_image')} "
              f"{f('mean_diff')} {f('p_holm', 7, 3)} {f('cliffs_delta', 7, 3)} {ratio if ratio is None else f'{ratio:9.2f}'}")
    print("wrote", a.out)


if __name__ == "__main__":
    main()
