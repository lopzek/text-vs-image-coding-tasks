#!/usr/bin/env python3
"""Run one cell of the study with the exact serving, page and window-fit settings of the paper.

A cell is one task x model x LoC group x input modality (the main grid, 64 cells), or one
rendering variant of the single-file completion sample (the rendering ablation, 10 cells).
The script assembles the ``python -m codecompress.main`` command for that cell, prints it, and
runs it (``--dry-run`` prints only). Outputs land in ``runs/main/<task>_<model>_<group>_<arm>/``
or ``runs/rendering/<model>_<arm>/``: one JSONL with one record per instance (model output,
score, served tokens, delivered-in-full flag) and a SCORES.json summary.

Examples
    python run_cell.py --task lcc --model gem --group 300-500 --arm ip
    python run_cell.py --task lcc --model gem --group 300-500 --arm it
    python run_cell.py --rendering --model qwen38 --arm idhf
    python run_cell.py --list            # every cell of the paper
    python run_cell.py --all main        # the 64 main cells, one after the other
"""
from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent

# The two models, their pinned Hugging Face revisions, the vLLM serving settings and the page
# geometry of the image input (page width x height in px, font size in px, and the smallest
# character height the calibration admits after the encoder's own resizing).
MODELS = {
    "gem": {
        "hf": "google/gemma-4-26B-A4B-it",
        "revision": "20da991ab4afab98e8f910c4a2e8f4fbefc404ad",
        "gpu_memory_utilization": 0.85,
        "vllm_max_num_seqs": 8,
        "page": (1120, 2240, 16),
        "min_char_px": 6.3,
    },
    "qwen38": {
        "hf": "Qwen/Qwen3.8-27B",
        "revision": "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0",
        "gpu_memory_utilization": 0.90,
        "vllm_max_num_seqs": 16,
        "page": (896, 784, 16),
        "min_char_px": 12.5,
    },
}

# The four tasks: benchmark id in the code base, dataset split, generation cap, the input limit of
# the whole prompt (served tokens), and the instance manifest of each sampling group.
TASKS = {
    "lcc": {
        "benchmark": "lcc", "split": "test", "max_new_tokens": 128, "window": 32640,
        "groups": {"200-300": "python_200-300.json", "300-500": "python_300-500.json",
                   "500-1000": "python_500-1000.json", "1000+": "python_1000-999999.json"},
    },
    "rb": {
        "benchmark": "repobench", "split": "cross_file_first", "max_new_tokens": 128, "window": 32640,
        "groups": {"0-500": "python_0-500.json", "500-1000": "python_500-1000.json",
                   "1000-2000": "python_1000-2000.json", "2000+": "python_2000-999999.json"},
    },
    "lca": {
        "benchmark": "lca_summarization", "split": "test", "max_new_tokens": 128, "window": 32640,
        "groups": {"0-1000": "lca_0-1000.json", "1000-2000": "lca_1000-2000.json",
                   "2000-5000": "lca_2000-5000.json", "5000+": "lca_5000-999999.json"},
    },
    "qa": {
        "benchmark": "longcodeqa", "split": "test", "max_new_tokens": 2048, "window": 30720,
        "groups": {"0-3500": "longcodeqa_0-3500.json", "3500-10000": "longcodeqa_3500-10000.json",
                   "10000-20000": "longcodeqa_10000-20000.json", "20000+": "longcodeqa_20000-999999.json"},
    },
}
MANIFEST_DIRS = {"lcc": "lcc", "rb": "repobench", "lca": "lca", "qa": "longcodeqa"}

# Window-fit constants per task and model, measured before the study on a ten-instance pilot:
# c = the tokens the chat template and the instruction framing add to a prompt, D = a margin for
# tokenizer drift. An instance is delivered in full when its whole prompt fits
# window - c - D (with 2 framing tokens per retained page on the image side).
FIT = {
    ("lcc", "gem"): (13, 9), ("lcc", "qwen38"): (12, 9),
    ("rb", "gem"): (12, 8), ("rb", "qwen38"): (11, 8),
    ("lca", "gem"): (26, 9), ("lca", "qwen38"): (24, 9),
    ("qa", "gem"): (12, 10), ("qa", "qwen38"): (11, 9),
}

# Input modalities and rendering variants: arm code -> method of codecompress.main.
ARMS = {
    "it": "image_text",             # text input (the code as plain text in the same chat path)
    "ip": "image_plain",            # image input, plain rendering (black on white, no highlighting)
    "ihf": "image_highlight",       # syntax highlighting, light background (VS Code Default Light)
    "id": "image_dark",             # dark background, no highlighting
    "idhf": "image_dark_highlight", # syntax highlighting, dark background (VS Code Dark+)
    "ib": "image_bold",             # bold font
}
MAIN_ARMS = ("it", "ip")
RENDERING_ARMS = ("ip", "ihf", "id", "idhf", "ib")
RENDERING_MANIFEST = ROOT / "manifests" / "rendering" / "python_lt1000_n500.json"


def manifest_n(path: Path) -> int:
    return int(json.loads(path.read_text())["n_selected"])


def build(task: str, model: str, group: str | None, arm: str, gpus: str, rendering: bool) -> tuple[list[str], Path]:
    """The main.py argument list of one cell and its output directory."""
    m, t = MODELS[model], TASKS[task]
    if rendering:
        if task != "lcc" or arm not in RENDERING_ARMS:
            raise SystemExit("the rendering ablation covers single-file completion (--task lcc) with arms " + ", ".join(RENDERING_ARMS))
        manifest, out = RENDERING_MANIFEST, ROOT / "runs" / "rendering" / f"{model}_{arm}"
    else:
        if arm not in MAIN_ARMS:
            raise SystemExit("the main grid has two arms: it (text input) and ip (image input)")
        if group not in t["groups"]:
            raise SystemExit(f"groups of {task}: " + ", ".join(t["groups"]))
        manifest = ROOT / "manifests" / "main" / MANIFEST_DIRS[task] / t["groups"][group]
        out = ROOT / "runs" / "main" / f"{task}_{model}_{group}_{arm}"
    c, d = FIT[(task, model)]
    n_gpus = len([g for g in gpus.split(",") if g.strip()])
    args = [
        "--benchmark", t["benchmark"], "--pipeline", "image", "--method", ARMS[arm],
        "--result_dir", str(out), "--dataset_split", t["split"],
        "--example_ids", str(manifest), "--num_examples", str(manifest_n(manifest)),
        "--filter_background_tokens_min", "0", "--filter_current_lines_max", "50",
        "--max_new_tokens", str(t["max_new_tokens"]), "--batch_size", "16",
        "--model_name", m["hf"], "--vlm_backend", "local", "--vlm_model", m["hf"], "--vlm_revision", m["revision"],
        "--generation_cuda_visible_devices", gpus, "--tensor_parallel_size", str(n_gpus),
        "--gpu_memory_utilization", str(m["gpu_memory_utilization"]),
        "--vllm_max_model_len", "32768", "--vllm_max_num_seqs", str(m["vllm_max_num_seqs"]), "--enforce_eager",
        "--compression_ratio", "1.0", "--image_control", "matched",
    ]
    if arm != "it":   # the image input: fixed page geometry, legibility floor, per-page metrics
        w, h, fs = m["page"]
        args += ["--fixed_image_width", str(w), "--fixed_image_height", str(h), "--fixed_font_size", str(fs),
                 "--min_char_px", str(m["min_char_px"]), "--record_page_metrics"]
    args += ["--fit_window_tokens", str(t["window"]), "--fit_template_tokens", str(c),
             "--fit_margin_tokens", str(d), "--fit_per_page_overhead", "2"]
    return args, out


def all_cells() -> list[dict]:
    cells = []
    for task, t in TASKS.items():
        for model in MODELS:
            for group in t["groups"]:
                for arm in MAIN_ARMS:
                    cells.append({"task": task, "model": model, "group": group, "arm": arm, "rendering": False})
    for model in MODELS:
        for arm in RENDERING_ARMS:
            cells.append({"task": "lcc", "model": model, "group": None, "arm": arm, "rendering": True})
    return cells


def run(cell: dict, gpus: str, dry_run: bool) -> int:
    args, out = build(cell["task"], cell["model"], cell["group"], cell["arm"], gpus, cell["rendering"])
    cmd = [sys.executable, "-m", "codecompress.main", *args]
    print("$ " + " ".join(shlex.quote(a) for a in cmd), flush=True)
    if dry_run:
        return 0
    # vLLM's workers JIT-compile kernels with the `ninja` binary of the serving environment, so the
    # interpreter's bin directory must be on PATH (the study's launcher exported it the same way).
    env = dict(os.environ, PYTHONPATH=str(ROOT), VLLM_NO_USAGE_STATS="1",
               PATH=str(Path(sys.executable).parent) + os.pathsep + os.environ.get("PATH", ""),
               PYTORCH_ALLOC_CONF=os.environ.get("PYTORCH_ALLOC_CONF", "expandable_segments:True"))
    out.mkdir(parents=True, exist_ok=True)
    return subprocess.run(cmd, cwd=ROOT, env=env).returncode


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--task", choices=TASKS)
    ap.add_argument("--model", choices=MODELS)
    ap.add_argument("--group", help="sampling group of the task (see --list)")
    ap.add_argument("--arm", choices=ARMS)
    ap.add_argument("--rendering", action="store_true", help="a rendering-ablation cell (task lcc, 500 instances below 1,000 LoC)")
    ap.add_argument("--gpus", default="0,1,2,3", help="CUDA devices; tensor parallelism = their count (the study: four 24 GB GPUs)")
    ap.add_argument("--dry-run", action="store_true", help="print the command without running it")
    ap.add_argument("--list", action="store_true", help="list every cell of the paper")
    ap.add_argument("--all", choices=("main", "rendering"), help="run every cell of the main grid or of the rendering ablation")
    ap.add_argument("--json", action="store_true", help="with --dry-run: print the argument list as JSON instead")
    a = ap.parse_args()
    if a.list:
        for c in all_cells():
            print(("rendering  " if c["rendering"] else "main       ") + " ".join(f"{k}={v}" for k, v in c.items() if k != "rendering" and v))
        return
    if a.all:
        cells = [c for c in all_cells() if c["rendering"] == (a.all == "rendering")]
    else:
        if a.rendering:   # the rendering ablation is always single-file completion, one sample
            if a.task not in (None, "lcc") or a.group:
                ap.error("--rendering takes --model and --arm only (task lcc, one instance list)")
            a.task, a.group = "lcc", None
        if not (a.task and a.model and a.arm) or (not a.rendering and not a.group):
            ap.error("give --task --model --group --arm (or --rendering --model --arm), or --list / --all")
        cells = [{"task": a.task, "model": a.model, "group": a.group, "arm": a.arm, "rendering": a.rendering}]
    if a.json:
        for c in cells:
            args, out = build(c["task"], c["model"], c["group"], c["arm"], a.gpus, c["rendering"])
            print(json.dumps({"cell": c, "result_dir": str(out), "args": args}))
        return
    failed = [c for c in cells if run(c, a.gpus, a.dry_run) != 0]
    if failed:
        raise SystemExit(f"{len(failed)} cell(s) failed: " + "; ".join(json.dumps(c) for c in failed))


if __name__ == "__main__":
    main()
