"""RepoBench benchmark — repository-level cross-file next-line completion.

Dataset: tianyang/repobench_python_v1.1 (split ``cross_file_first``)
Task:    Given cross-file repo context + the in-file prefix, predict the
         next line of code. A *different task* from LCC (single-file
         next-line).
Metrics: Edit Similarity (ES) + Exact Match (EM) — RepoBench-OFFICIAL
         scoring, ported faithfully from the upstream repo:
           upstream RepoBench run.py                (get_first_line_not_comment)
           upstream RepoBench evaluation/metrics.py (fuzzywuzzy ES,
                                                             split()-eq EM)
Reference: Liu et al., "RepoBench", ICLR 2024.

Context split (imposed for our text/image modality contrast, mapping
cleanly onto RepoBench's own cross-file/in-file prompt split in
``data/utils.construct_prompt``):

  background = cross-file prompt  (repo header + each context snippet,
               each prefixed ``# Path: ...``)  — the long, length-varying
               part that the image arm renders and the text arm prepends.
  query      = in-file prompt     (``# Path: <file>`` + import_statement +
               cropped_code.rstrip())  — kept as TEXT in the image arm for
               character-level precision (same rationale as LCC).
  gt         = next_line (raw, single line, as upstream uses it).

Without compression, ``build_text_prompt(bg, query)`` reproduces upstream
``construct_prompt`` EXACTLY (no tokenizer ⇒ no truncation).
"""

from __future__ import annotations

import re

import datasets
from fuzzywuzzy import fuzz
from loguru import logger

from codecompress.benchmarks.base import BaseBenchmark

# Manifest machinery is task-agnostic — reuse the LCC implementation
# verbatim so every task shares one code path.
from codecompress.benchmarks.lcc import (
    compute_selected_sha256,
    manifest_expected_sha256,
    resolve_example_ids,
    select_by_ids,
)


# =============================================================================
# Prompt construction — faithful to upstream data/utils.construct_prompt
# =============================================================================

def _comment_symbol(language: str) -> str:
    return "#" if language == "python" else "//"


def normalize_blanklines(text: str) -> str:
    """Collapse runs of 4+ newlines to exactly 2 (upstream's final
    ``re.sub(r'\\n{4,}', '\\n\\n', prompt)``). Idempotent."""
    return re.sub(r"\n{4,}", "\n\n", text)


def repobench_cross_file_prompt(example: dict, language: str = "python") -> str:
    """The cross-file half of the upstream prompt (RAW, pre-normalization).

    Exactly upstream's ``cross_file_prompt``:
        # Repo Name: <repo>\\n
        # Path: <p1>\\n<snippet1>\\n\\n
        # Path: <p2>\\n<snippet2>\\n\\n ...
    """
    cs = _comment_symbol(language)
    out = f"{cs} Repo Name: {example['repo_name']}\n"
    for snippet in example["context"]:
        out += f"{cs} Path: {snippet['path']}\n{snippet['snippet']}" + "\n\n"
    return out


def repobench_in_file_prompt(example: dict, language: str = "python") -> str:
    """The in-file half of the upstream prompt.

    Exactly upstream's ``in_file_prompt``:
        # Path: <file_path>\\n<import_statement>\\n<cropped_code.rstrip()>\\n
    """
    cs = _comment_symbol(language)
    return (
        f"{cs} Path: {example['file_path']}\n"
        f"{example['import_statement']}\n"
        f"{example['cropped_code'].rstrip()}\n"
    )


# =============================================================================
# Prediction post-processing — ported verbatim from upstream run.py
# =============================================================================

def get_first_line_not_comment(code: str, language: str = "python") -> str:
    """First line of *code* that is not blank and not a comment.

    Verbatim port of the upstream RepoBench run.py — including its
    quirks: the returned line is NOT whitespace-stripped, and an
    unterminated multi-line comment falls through to ``lines[0]``.
    """
    assert language in ["python", "java"], \
        "language must be one of [python, java]"

    code = code.lstrip("\n")
    lines = code.split("\n")
    in_multiline_comment = False

    if language == "python":
        for line in lines:
            if not line.strip():
                continue
            if not in_multiline_comment and (
                line.strip().startswith('"""') or line.strip().startswith("'''")
            ):
                in_multiline_comment = True
                continue
            if in_multiline_comment and (
                line.strip().endswith('"""') or line.strip().endswith("'''")
            ):
                in_multiline_comment = False
                continue
            if in_multiline_comment:
                continue
            if line.strip().startswith("#"):
                continue
            return line

    elif language == "java":
        for line in lines:
            if not line.strip():
                continue
            if not in_multiline_comment and line.strip().startswith("/*"):
                in_multiline_comment = True
                continue
            if in_multiline_comment and line.strip().endswith("*/"):
                in_multiline_comment = False
                continue
            if in_multiline_comment:
                continue
            if line.strip().startswith("//"):
                continue
            return line

    return lines[0]


# =============================================================================
# RepoBench-official metrics (upstream evaluation/metrics.py)
# =============================================================================

def compute_repobench_EM(prediction_first_line: str, ground_truth: str) -> int:
    """Upstream ``exact_match_score`` per-pair rule: whitespace-insensitive
    token equality. Returns 0 or 100."""
    return 100 if prediction_first_line.split() == ground_truth.split() else 0


def compute_repobench_ES(prediction_first_line: str, ground_truth: str) -> float:
    """Upstream ``edit_similarity_score`` per-pair rule: fuzzywuzzy
    ``fuzz.ratio`` (0–100, char-level / whitespace-SENSITIVE).

    python-Levenshtein is a hard dependency (pyproject) so fuzz.ratio is
    deterministic and matches the upstream RepoBench environment, which
    pins ``fuzzywuzzy`` + ``python-Levenshtein``.
    """
    return float(fuzz.ratio(prediction_first_line, ground_truth))


# =============================================================================
# Hashed content — single source of truth for manifest hashing
# =============================================================================

def repobench_audit_contents(rows):
    """Yield the ordered content the manifest hash pins, two items/row:

      1. the normalized cross-file background (what the image arm renders)
      2. ``next_line`` (the target)

    The sampler (scripts/sample_repobench_by_line_count.py) and
    ``RepoBenchBenchmark.load`` BOTH feed this into
    ``lcc.compute_selected_sha256`` so a manifest can never silently
    desync from the code it pins. Language is fixed to ``python`` (the
    study language) so the hash protocol is stable.
    """
    for r in rows:
        yield normalize_blanklines(repobench_cross_file_prompt(r, "python"))
        yield r["next_line"]


# =============================================================================
# Benchmark class
# =============================================================================

class RepoBenchBenchmark(BaseBenchmark):
    """RepoBench repository-level cross-file next-line completion."""

    def __init__(self, dataset_path: str = "tianyang/repobench_python_v1.1",
                 language: str = "python"):
        self.dataset_path = dataset_path
        self.language = language

    # ---- loading ---------------------------------------------------------

    def load(self, args):
        if isinstance(args, dict):
            get = args.get
        else:
            get = lambda k, d=None: getattr(args, k, d)  # noqa: E731

        dataset_path = get("dataset_path", self.dataset_path) or self.dataset_path
        split = get("dataset_split", "cross_file_first") or "cross_file_first"
        num_examples = get("num_examples", 500)
        example_ids_spec = get("example_ids", None)
        example_ids = resolve_example_ids(example_ids_spec)
        expected_sha256 = manifest_expected_sha256(example_ids_spec)

        logger.info(
            f"Loading RepoBench from {dataset_path} (split={split}); "
            f"manifest={'yes' if example_ids is not None else 'no'}"
        )
        dataset = datasets.load_dataset(dataset_path, split=split)

        if example_ids is not None:
            if len(example_ids) == 0:
                raise ValueError(
                    "example_ids resolved to an empty selection — refusing "
                    "to run an empty cell (check the manifest)"
                )
            dataset = select_by_ids(dataset, example_ids)
            if expected_sha256 is not None:
                actual = compute_selected_sha256(
                    repobench_audit_contents(dataset)
                )
                if actual != expected_sha256:
                    raise ValueError(
                        "manifest content hash mismatch: the pinned indices "
                        "no longer point at the same RepoBench rows "
                        "(dataset re-upload or Arrow row reorder?). "
                        f"expected sha256={expected_sha256}, got {actual}. "
                        "Regenerate with "
                        "scripts/sample_repobench_by_line_count.py."
                    )
            id_list = list(example_ids)
        else:
            take = min(num_examples, len(dataset))
            dataset = dataset.select(range(take))
            id_list = list(range(take))

        lang = self.language

        def add_fields(example, idx):
            example["background_context"] = normalize_blanklines(
                repobench_cross_file_prompt(example, lang)
            )
            example["current_function_context"] = repobench_in_file_prompt(
                example, lang
            )
            example["gt"] = example["next_line"]
            example["language"] = lang
            example["id"] = id_list[idx]
            return example

        processed = dataset.map(add_fields, with_indices=True)
        logger.info(f"RepoBench ready: {len(processed)} rows")
        # No tokenizer needed (no line/token filter — manifest is
        # authoritative, mirroring lcc's example_ids path). main.py
        # unpacks `dataset, _`.
        return processed, None

    # ---- context split ---------------------------------------------------

    def split_context(self, example):
        bg = example.get("background_context", "")
        query = example.get("current_function_context", "")
        return bg, query

    def text_baseline_instruction(self):
        """RepoBench's ``build_text_prompt`` is bare (normalized bg+query), so the
        image-pipeline TEXT BASELINE arm (``image_text``) must supply the completion
        instruction here to stay symmetric with ``build_image_prompt`` — otherwise the
        text baseline is under-instructed. Text-framed counterpart of
        ``build_image_prompt``'s directives.
        """
        return (
            "The following is cross-file source code from a repository, provided as "
            "context, followed by the in-file code prefix from the file being edited. "
            "Output ONLY the next line of code that continues it. Do not explain. "
            "Do not add comments. Do not use markdown. Do not wrap in ```python. "
            "Output raw code only.\n\n"
        )

    def build_text_prompt(self, compressed_background, query):
        """Reproduce upstream construct_prompt (sans tokenizer truncation):
        ``re.sub(r'\\n{4,}', '\\n\\n', cross_file_prompt + in_file_prompt)``.

        We re-normalize the FULL concatenation ``background + query`` here.
        That equals upstream ``normalize(cross_raw + in_raw)`` because of a
        load-bearing invariant: ``query`` is the in-file prompt, which
        ALWAYS begins with the comment symbol (``#``/``//``), so ``query[0]``
        is never ``\\n``. Hence the inner pre-normalization of ``background``
        can neither merge nor split a newline run across the bg/query seam
        (``N(N(a)+b) == N(a+b)`` is NOT a general identity — it holds only
        because ``b`` does not start with ``\\n``). Pinned by
        test_build_text_prompt_equals_upstream_* incl. the 4+‑newline case."""
        if compressed_background:
            return normalize_blanklines(compressed_background + query)
        return query

    def build_image_prompt(self, images, query):
        # Two-arm byte parity: the text arm normalizes bg+query JOINTLY in
        # build_text_prompt; the render content is already normalized at
        # load(), so the image arm must normalize its query text too, else
        # 4+-newline runs inside the in-file prompt make the arms byte-differ.
        # Separate normalization == the text arm's joint one because the query
        # starts with the comment symbol, never '\n'.
        query = normalize_blanklines(query)
        messages = []
        for img in images:
            messages.append({"type": "image", "image": img})
        messages.append({
            "type": "text",
            "text": (
                "The image(s) above show cross-file source code from a "
                "repository, provided as context. Below is the in-file "
                "code prefix from the file being edited. Output ONLY the "
                "next line of code that continues it. "
                "Do not explain. Do not add comments. Do not use markdown. "
                "Do not wrap in ```python. Output raw code only.\n\n" + query
            ),
        })
        return messages

    def generation_config(self):
        """Greedy decoding (deterministic), matching LCC.

        Upstream RepoBench samples (do_sample=True); we use greedy for
        reproducibility — the standard choice across our study and a
        non-issue for a paired within-example modality contrast."""
        return {"max_new_tokens": 128, "temperature": 0}

    def score(self, prediction, example):
        gt = example.get("gt")
        if prediction == "ERROR_GENERATING" or gt is None:
            return {"ES": 0.0, "EM": 0.0}
        # Defense-in-depth: the pipeline always hands us a decoded str (or
        # the ERROR_GENERATING sentinel handled above), but never let a
        # non-str prediction raise inside scoring — score it as empty.
        if not isinstance(prediction, str):
            prediction = ""
        lang = example.get("language", "python")
        first_line = get_first_line_not_comment(prediction, language=lang)
        return {
            "ES": compute_repobench_ES(first_line, gt),
            "EM": float(compute_repobench_EM(first_line, gt)),
        }

    def metric_names(self):
        return ["ES", "EM"]

    def ground_truth(self, example):
        return example["gt"]
