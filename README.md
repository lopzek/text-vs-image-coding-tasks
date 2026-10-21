# Text or image? Replication package

Code to re-run the experiments of the paper *Not All Pixels Cost the Same: Vision Encoder Type
Governs the Text-vs-Image Trade-off in Coding Tasks*: delivering source code to a vision-language
model (VLM) as rendered page images versus as plain text, within the same model, on four coding
tasks, with every instance delivered whole at its natural length.

The package is deliberately minimal. It contains the experiment code at the exact revision that
produced the paper's runs, the instance lists, a launcher that reproduces every cell's command, the
scoring and the statistics of the paper, the calibration tools, and the environment pins. It does
not contain the model outputs (several hundred MB), the figure and table generators, or tests.

## What the study runs

| | Gemma-4 26B-A4B (`gem`) | Qwen3.8-27B (`qwen38`) |
| --- | --- | --- |
| Hugging Face id | `google/gemma-4-26B-A4B-it` | `Qwen/Qwen3.8-27B` |
| Revision used | `20da991ab4afab98e8f910c4a2e8f4fbefc404ad` | `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0` |
| Vision encoder | fixed-length (about 256 tokens per image) | variable-length (one token per 32x32 px) |
| Page geometry of the image input | 1120x2240 px, 16 px font | 896x784 px, 16 px font |
| vLLM serving | tensor parallel 4, eager, 8 sequences, 0.85 memory | tensor parallel 4, eager, 16 sequences, 0.90 memory |

| Task (`code`) | Benchmark | Instances | Sampling groups (LoC) | Metric | Output cap |
| --- | --- | --- | --- | --- | --- |
| single-file completion (`lcc`) | `microsoft/LCC_python`, test | 1,867 | 200-300, 300-500, 500-1000, 1000+ | edit similarity | 128 tokens |
| cross-file completion (`rb`) | `tianyang/repobench_python_v1.1`, cross_file_first | 2,000 | 0-500, 500-1000, 1000-2000, 2000+ | edit similarity | 128 tokens |
| module summarization (`lca`) | `JetBrains-Research/lca-module-summarization`, test | 216 | 0-1000, 1000-2000, 2000-5000, 5000+ | BERTScore F1 | 128 tokens |
| long-context QA (`qa`) | `Steefano/LCB` (LongCodeQA) | 443 | 0-3500, 3500-10000, 10000-20000, 20000+ | accuracy | 2,048 tokens |

Every cell is one task x model x sampling group x input modality: `it` = the code as plain text,
`ip` = the code as page images (plain rendering). That is 4 x 2 x 4 x 2 = 64 cells. Both modalities
go through the same chat path of the same model with the same instruction; only the words that point
the model to the code differ (`codecompress/benchmarks/*.py` hold the prompts). The context window is
32,768 tokens; an instance whose whole prompt does not fit is flagged *not delivered in full* in that
modality and enters the feasibility result only. The paper's six shared LoC groups (<500, 500-1K,
1-2K, 2-4K, 4-8K, 8K+) are a re-grouping of the same instances at analysis time (`manifests/loc/`).

The rendering ablation (RQ4) re-renders 500 single-file completion instances below 1,000 LoC under
four variants: `ihf` syntax highlighting on white (VS Code Default Light), `id` dark background, `idhf`
highlighting on dark (VS Code Dark+), `ib` bold font; `ip` is the plain rendering. That is 2 x 5 = 10
more cells. The text input of the same instances comes from the main grid.

## Layout

| Path | Content |
| --- | --- |
| `codecompress/` | The experiment code: `main.py` (entry point), `pipelines/image.py` (prompt assembly, page rendering, window fit, generation), `code_renderer.py` (code to page images, the five renderings, visual-token laws), `vlm_inference.py` (vLLM serving), `window_audit.py` (served-token accounting and the delivered-in-full flag), `benchmarks/` (loaders, prompts and metrics of the four tasks). |
| `manifests/main/<task>/*.json` | The instances of every sampling group: dataset row indices, the sampling seed, and a SHA-256 over the selected contents that the loader verifies. |
| `manifests/rendering/python_lt1000_n500.json` | The 500 instances of the rendering ablation and their join to the main grid. |
| `manifests/loc/<task>.jsonl` | Lines of code of every instance (the re-grouping key of the analysis). |
| `run_cell.py` | Builds and runs the `codecompress.main` command of one cell with the paper's settings. |
| `score_cell.py` | Per-instance scores and delivered-in-full flags of a finished cell. |
| `paired_stats.py` | The per-group numbers of the paper (feasibility, cost, effectiveness, rendering). |
| `scripts/` | The samplers that produced the manifests, the LoC census, and the three calibration tools. |
| `requirements-vllm.lock.txt`, `setup_vllm_env.sh` | The exact serving environment (Python 3.12, vLLM nightly, CUDA 12.9). |
| `requirements-tools.txt` | The CPU tools environment (Python 3.14). |


## Environments

Hardware used: one machine with four NVIDIA RTX 3090 (24 GB each), driver 575+ (CUDA 12.9). Every
model run uses all four GPUs (tensor parallelism 4).

**Serving environment** (all GPU runs). Needs [uv](https://docs.astral.sh/uv/) and network access:

```bash
VLLM_COMMIT_FULL=71df063c494c111ab60f6a33c54aafe7b9ae1d02 bash setup_vllm_env.sh --from-lock
```

This creates `.venv-vllm` (Python 3.12) from `requirements-vllm.lock.txt`: vLLM
`0.22.1rc1.dev116+g71df063c4` (a nightly build served from `wheels.vllm.ai/<commit>/cu129`),
torch 2.11.0+cu129, transformers 5.10.1. The script refuses to finish unless the installed set
matches the lock exactly. The lock differs from the study's own environment in one package:
`hf-xet`, the Hub download transport of `huggingface-hub`, was the development build `1.5.1.dev1`
there, which has since been removed from PyPI; the lock pins the released `1.5.1` instead. It plays
no part in rendering, serving or scoring, and every run is offline. The code is used through `PYTHONPATH` (the launcher sets it); nothing is
pip-installed from this package.

**Tools environment** (sampling, census, calibration pages, OCR control, scoring, statistics).
Built with `uv` as well; `pip` refuses one of the pins on Python 3.14 (`rapidocr-onnxruntime==1.4.4`
declares `Requires-Python <3.13` although its pure-Python wheel runs there):

```bash
uv venv .venv-tools --python 3.14 && uv pip install --python .venv-tools/bin/python -r requirements-tools.txt
```

**Font.** The pages are rendered with DejaVu Sans Mono (Debian package `fonts-dejavu-core`). Install it
before rendering: the renderer falls back to Pillow's default bitmap font if it is missing, which
would silently change every page.

**Assets.** Download once, then run offline (`export HF_HUB_OFFLINE=1`) so that the pinned
snapshots are the ones used:

```bash
.venv-vllm/bin/hf download google/gemma-4-26B-A4B-it --revision 20da991ab4afab98e8f910c4a2e8f4fbefc404ad
.venv-vllm/bin/hf download Qwen/Qwen3.8-27B --revision 1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0
.venv-vllm/bin/hf download Qwen/Qwen2.5-Coder-7B-Instruct --include 'tokenizer*' --include '*.json' --include 'merges.txt'   # tokenizer the dataset loaders open (not used for any reported number)
.venv-vllm/bin/hf download roberta-large                                                                         # BERTScore (summarization only)
.venv-vllm/bin/hf download Qwen/Qwen3.6-27B --revision 6a9e13bd6fc8f0983b9b99948120bc37f49c13e9 --include 'tokenizer*' --include '*.json' --include 'merges.txt'   # tokenizer of the LoC census only (scripts/loc_census.py)
```

The two models and the census tokenizer are resolved by commit id in the code, so they are pinned
with `--revision`. `Qwen/Qwen2.5-Coder-7B-Instruct` (opened by the dataset loaders; every token count
in the paper comes from the served model's own tokenizer) and `roberta-large` (opened by `bert-score`)
are loaded by repository name, i.e. from the cache's `main` pointer, which only a download without
`--revision` creates. The study used `c03e6d358207e414f1eca0bb1891e29f1db0e242` and
`722cf37b1afa9454edce342e7895e588b6ff1d59`, the current `main` of both repositories; check with
`cat "$HF_HOME/hub/models--roberta-large/refs/main"` (default `HF_HOME` is `~/.cache/huggingface`).

Datasets are loaded with `datasets.load_dataset` by id; the revisions the study used are
`microsoft/LCC_python` `3e6eb60f415ac8f77c0a9e3168072c3159419eaf`, `tianyang/repobench_python_v1.1`
`8a7cf0c8942cc1aa066bf261839650ac55a2ff79`, `JetBrains-Research/lca-module-summarization`
`ac7ce1539f66c369817f524c1e19194f0873d706`, `Steefano/LCB` `989d5eff750d65a72c522e47f8f745ef2e22906b`.
A changed dataset cannot go unnoticed: every manifest carries a SHA-256 over the selected instance
contents and the loader stops if it does not match.

## Running the cells

```bash
source .venv-vllm/bin/activate      # or call .venv-vllm/bin/python explicitly
python run_cell.py --list                                                # the 74 cells of the paper
python run_cell.py --task lcc --model gem --group 300-500 --arm ip        # one image cell
python run_cell.py --task lcc --model gem --group 300-500 --arm it        # its text counterpart
python run_cell.py --rendering --model qwen38 --arm idhf                  # one rendering variant
python run_cell.py --all main                                             # the 64 main cells, sequentially
python run_cell.py --all rendering                                        # the 10 rendering cells
```

`--dry-run` prints the exact `python -m codecompress.main ...` command without running it, so every
setting (model revision, serving flags, page geometry, window-fit constants, instance list, output cap)
can be read off. `--gpus` selects the devices; tensor parallelism is their count.

A cell writes `runs/main/<task>_<model>_<group>_<arm>/` (or `runs/rendering/<model>_<arm>/`) with
`method_*/<model>.jsonl`, one record per instance (prompt, model output, score, page record
`image_info`, served tokens and the delivered-in-full flag in `window_audit`), and a `SCORES.json`
summary. On the study's hardware the 64 main cells took about 54 GPU-hours in total (the longest, the
cross-file 2000+ image cell on Qwen3.8, 4.3 h) and the 10 rendering cells about 6 h. Decoding is
greedy, so a rerun on the same hardware and software reproduces the outputs almost verbatim (a rerun
of one 500-instance cell matched 99.4% of the outputs exactly, mean score within 0.02); other GPUs
or kernels can shift single outputs. A check of this package on the study's machine reran the summarization
1000-2000 LoC group on both models and both input modalities (4 cells, 144 instances): every model
output, score, served-token count and delivered-in-full flag matched the study's own records exactly.

## Scoring and the paper's numbers

```bash
.venv-tools/bin/python score_cell.py runs/main/* runs/rendering/*     # writes scores.jsonl into every cell
.venv-tools/bin/python paired_stats.py                                 # prints the per-group table, writes results.json
```

`score_cell.py` applies the benchmark's own metric (edit similarity after removing a markdown code
fence from the output, 0.2% of the completion outputs carried one; BERTScore F1; accuracy) and keeps
the delivered-in-full flag and the served input tokens of every instance. `paired_stats.py`
re-groups the instances on the six shared LoC groups and computes, per task, model and group: the
share of instances each modality delivered in full (RQ1); the mean served tokens of each modality and
a Wilcoxon signed-rank test on the per-instance token difference, Holm-adjusted over all groups
(RQ3); and, on the instances delivered in full in both modalities, the mean of each modality, the mean
paired difference image minus text with a 95% percentile-bootstrap interval (100,000 resamples, seed
42), a two-sided Wilcoxon signed-rank test with Holm adjustment within the task family, and Cliff's
delta (RQ2). Groups with fewer than 30 pairs are flagged as not robust; a group without a pair is not
testable. For the rendering ablation it reports each rendering's mean with its interval, the paired
contrast against the plain rendering (Wilcoxon, Holm over the four contrasts), and the gap to the text
input of the same instances (RQ4). Applied to the study's own outputs, these two scripts reproduce
every number of the paper's figures.

## Regenerating the instance lists

The manifests are shipped, so this step is optional. They are reproduced exactly by (tools
environment, from the package root with `PYTHONPATH=.`):

```bash
python scripts/sample_lcc_by_line_count_v2.py --lang python --tiers 200-300,300-500,500-1000,1000-999999 --n 500 --seed 42 --out-dir manifests/main/lcc
python scripts/sample_repobench_by_line_count.py --tiers 0-500,500-1000,1000-2000,2000-999999 --n 500 --seed 42 --out-dir manifests/main/repobench
python scripts/sample_lca_by_line_count.py --tiers 0-1000,1000-2000,2000-5000,5000-999999 --n 500 --seed 42 --out-dir manifests/main/lca
python scripts/sample_longcodeqa_by_line_count.py --tiers 0-3500,3500-10000,10000-20000,20000-999999 --n 500 --seed 42 --out-dir manifests/main/longcodeqa
python scripts/sample_rendering_lcc_lt1000.py --check     # the rendering sample: 500 of the 1,500 instances below 1,000 LoC, numpy seed 42
```

Lines of code are physical lines of the delivered code, including blank and comment lines
(`line_count` in `scripts/sample_lcc_by_line_count.py`). Up to 500 instances per group are drawn with
`random.Random(f"{seed}:{tier}")`; the two small benchmarks are used in full.
`scripts/loc_census.py --task all --out-dir census` reproduces the full LoC census of the four
datasets that the groups were chosen from.

## Calibration of the page geometry

The character height of the page images was set before the study by a transcription test
(`scripts/calibrate_legibility.py`): 50 real code pages are rendered at a ladder of decreasing character
heights, each model transcribes them, and the height at which accuracy reaches 95% of its plateau is
the model's threshold; the page geometry keeps every rendered line at 1.25x that height. A traditional
OCR engine (RapidOCR) is the control that the ladder spans the legible range, and an
obfuscated-identifier control (`scripts/calibrate_obfuscation_probe.py`) checks whether a model reads
or recites.

```bash
# pages (tools environment; without a census directory the pages are drawn from the datasets)
python scripts/calibrate_legibility.py gen --pages 50 --seed 42 --styles plain,dark_highlight --ladder 1.0,0.75,0.5,0.35,0.25 --out-dir calib/pages
python scripts/calibrate_legibility.py transcribe --backend rapidocr --in-dir calib/pages --out calib/ocr.jsonl
# transcription by the models (serving environment)
python scripts/calibrate_legibility.py transcribe --backend vlm --vlm-engine vllm --model google/gemma-4-26B-A4B-it --in-dir calib/pages --out calib/gemma.jsonl --tensor-parallel-size 4 --enforce-eager --max-num-seqs 8
python scripts/calibrate_legibility.py transcribe --backend vlm --vlm-engine vllm --model Qwen/Qwen3.8-27B --in-dir calib/pages --out calib/qwen.jsonl --tensor-parallel-size 4 --enforce-eager --max-num-seqs 16
python scripts/calibrate_legibility.py analyze --in calib/gemma.jsonl --in calib/qwen.jsonl --in calib/ocr.jsonl --out calib/report.json
# Gemma-4 stays legible down to the end of that ladder; its threshold comes from an extended one
python scripts/calibrate_legibility.py gen --pages 50 --seed 42 --styles plain --ladder 1.0,0.75,0.5,0.35,0.25,0.175,0.125,0.1 --out-dir calib/pages_ext
# obfuscated-identifier control (every third page, identifiers renamed to same-length pseudo-words)
python scripts/calibrate_obfuscation_probe.py gen --in-dir calib/pages --out-dir calib/obf --every 3
python scripts/calibrate_obfuscation_probe.py transcribe --model Qwen/Qwen3.8-27B --in-dir calib/obf --out calib/obf_qwen.jsonl --max-num-seqs 16
python scripts/calibrate_obfuscation_probe.py analyze --obfuscated calib/obf_qwen.jsonl --original calib/qwen.jsonl --out calib/obf_report.json
# the page geometries that keep every line above the threshold height
python scripts/calibrate_page_geometry.py --out calib/geometry_table.json --select '{"gemma": 6.25, "qwen": 12.5}' --geometry-out calib/geometry.json
```

The calibration scripts load the models by id; run them offline with the pinned snapshots cached.

## Provenance

The code is the revision that produced the paper's runs. The only change to it during the study was a
fix of the page renderer for the two highlighted renderings (glyph positions in highlighted text),
after which those two rendering cells were rerun; plain, dark and bold pages are byte-identical before
and after the fix, and the main grid does not use highlighting. For this package the code was reduced
to the modules on the execution path of these runs; comments that pointed at private paths were
reworded, and nothing else was changed.

## License and third-party material

The code in this repository is released under the MIT License (see `LICENSE`). Two parts of the
benchmark code are ports of their benchmarks' official implementations and keep their origin:

- `codecompress/benchmarks/repobench.py`: `get_first_line_not_comment` and the edit-similarity
  scoring are ported verbatim from the RepoBench repository (Liu et al., "RepoBench: Benchmarking
  Repository-Level Code Auto-Completion Systems", ICLR 2024; https://github.com/Leolty/repobench),
  which is distributed under the Creative Commons Attribution 4.0 license.
- `codecompress/benchmarks/longcodeqa.py`: the answer-parsing rule `parse_final_answer` and the
  question prompt follow the official LongCodeBench evaluation (Rando et al., "LongCodeBench",
  2025; https://github.com/Zteefano/long-code-bench) and were verified against it.

The four datasets (`microsoft/LCC_python`, `tianyang/repobench_python_v1.1`,
`JetBrains-Research/lca-module-summarization`, `Steefano/LCB`) and the two models are used under
their own terms from the Hugging Face Hub; this repository redistributes none of their content, only
row indices, content hashes and lines-of-code counts of the instances.
