"""LongCodeQA benchmark (Extreme-Length Code Comprehension).

Dataset: Steefano/LCB (HuggingFace, LongCodeQA subset)
Task: Multiple-choice QA about entire Python repositories (32K–1M tokens).
Metrics: Accuracy (correct answer letter match)
Reference: Rando et al., "LongCodeBench", COLM 2025

The context is an entire multi-file Python repository. The "query" is a
multiple-choice question derived from a real GitHub issue. The model must
answer with a single letter (A, B, C, or D).

Both the prompt and the scoring are verified byte-for-byte against the official
LongCodeBench artifacts: `build_text_prompt` equals the official `Steefano/LCB`
`prompt` field on all 443 rows, and `parse_final_answer` equals the upstream
`LongCodeQAEvaluator._parse_final_answer` (executed from the upstream code).
"""

import re
import warnings

import datasets
from loguru import logger
from transformers import AutoTokenizer

from codecompress.benchmarks.base import BaseBenchmark
from codecompress.benchmarks.lcc import (
    compute_selected_sha256,
    manifest_expected_sha256,
    resolve_example_ids,
    select_by_ids,
)


# Upstream instruction (``prompt_goal``), verbatim from LongCodeBench
# ``long_code_bench/data/codeqa.py`` and confirmed
# byte-identical to the ``prompt_goal`` field of every ``Steefano/LCB`` row. The embedded
# newlines are part of the string (do not reflow).
PROMPT_GOAL = (
    "You are going to be provided the content of a\n"
    "repository and a question about it. Provide the answer to the question by\n"
    "stating ONLY the letter associated to the question."
)

# Image-arm instruction: PROMPT_GOAL with the ONLY modality-necessary substitution
# ("repository" → "repository as image(s)"); the constraint sentence is word-identical
# to the text arm's.
IMAGE_PROMPT_GOAL = (
    "You are going to be provided the content of a\n"
    "repository as image(s) and a question about it. Provide the answer to the question by\n"
    "stating ONLY the letter associated to the question."
)


# =============================================================================
# Dataset loading + manifest content hash
# =============================================================================

def load_longcodeqa_split(
    dataset_path: str,
    split: str = "train",
    *,
    hub_download=None,
    dataset_loader=None,
):
    """Load the LongCodeQA zip through one local-first implementation.

    Both the benchmark loader and the line-count sampler use this function,
    which keeps their archive member selection and resulting row order in
    lockstep. Dependency injection is intentionally narrow: it lets the
    sampler's offline tests replace the Hub resolver and datasets loader
    without maintaining a second copy of the zip-loading protocol.
    """
    if hub_download is None:
        from huggingface_hub import hf_hub_download

        hub_download = hf_hub_download
    if dataset_loader is None:
        dataset_loader = datasets.load_dataset

    try:
        local_zip = hub_download(
            dataset_path,
            "LongCodeQA.zip",
            repo_type="dataset",
            local_files_only=True,
        )
    except Exception:
        # Preserve the historical cache-miss behaviour. With
        # HF_HUB_OFFLINE=1 the Hub client itself remains offline and fails
        # promptly; otherwise this is the one cache-fill attempt.
        local_zip = hub_download(
            dataset_path,
            "LongCodeQA.zip",
            repo_type="dataset",
        )
    return dataset_loader(
        "json",
        data_files=f"zip://LQA/*.json::{local_zip}",
        split=split,
    )


def longcodeqa_audit_contents(rows):
    """Yield manifest-pinned content, three ordered items per row.

    The sampler and :meth:`LongCodeQABenchmark.load` both feed this
    exact iterator into ``lcc.compute_selected_sha256``. Keeping the hash
    protocol here makes repo/question/answer ordering impossible to drift
    between manifest creation and verification.
    """
    for row in rows:
        yield row["repo_text"]
        yield row["question"]
        yield row["correct_letter"]


# =============================================================================
# LongCodeQA-specific metric
# =============================================================================

LIKELIHOOD_CHOICES = [" A", " B", " C", " D"]


def score_likelihood_choices(choice_logprobs: list[float], correct_letter: str) -> dict:
    """Choose the first maximum likelihood answer and score it on the 0/100 scale."""
    if len(choice_logprobs) != len(LIKELIHOOD_CHOICES):
        raise ValueError(
            f"Expected {len(LIKELIHOOD_CHOICES)} choice scores, got {len(choice_logprobs)}"
        )
    best = max(choice_logprobs)
    tied = [i for i, value in enumerate(choice_logprobs) if value == best]
    if len(tied) > 1:
        message = (
            "Likelihood choice tie at indices "
            f"{tied}; choosing the lowest index deterministically."
        )
        logger.warning(message)
        warnings.warn(message, RuntimeWarning, stacklevel=2)
    letter = LIKELIHOOD_CHOICES[tied[0]].strip()
    return {
        "letter": letter,
        "accuracy": 100.0 if letter == correct_letter.upper() else 0.0,
    }

def parse_final_answer(response: str):
    """Extract the chosen A/B/C/D letter — a FAITHFUL port of upstream's
    ``LongCodeQAEvaluator._parse_final_answer`` (LongCodeBench, Rando et al. 2025;
    upstream ``long_code_bench/inference/codeqa_eval.py``).

    Two-tier, strict (verified against the executed upstream):
      1. ``Final Answer:\\s*([ABCD])`` anywhere (case-insensitive), else
      2. a single ``A/B/C/D`` at the very END of the stripped response
         (``\\b([ABCD])\\s*$``), else ``None``.

    NOTE this is intentionally strict — ``"B)"`` → None, ``"(D)"`` → None,
    ``"the answer is A. Note D is plausible"`` → None. The previous
    "last standalone A-D letter anywhere" heuristic DIVERGED from upstream and
    mis-scored these (do not reintroduce the heuristic).

    Returns:
        The uppercase letter, or ``None`` if unparseable (counted incorrect upstream).
    """
    match = re.search(r"Final Answer:\s*([ABCD])", response, re.IGNORECASE)
    if match:
        return match.group(1).upper()
    match = re.search(r"\b([ABCD])\s*$", response.strip(), re.IGNORECASE)
    if match:
        return match.group(1).upper()
    return None


def compute_accuracy(prediction: str, correct_letter: str) -> float:
    """Score one LongCodeQA answer against the gold letter, upstream-faithfully.

    Parses via :func:`parse_final_answer` (upstream's exact rule); an unparseable
    response is counted INCORRECT (upstream: ``None`` never matches the gold).

    Returns:
        float: 100.0 if the parsed letter equals ``correct_letter``, else 0.0.
        (Upstream reports accuracy in [0, 1]; we use the project's 0/100 scale, so
        the per-example correctness is identical and the mean differs only by ×100.)
    """
    letter = parse_final_answer(prediction)
    if letter is None:
        return 0.0
    return 100.0 if letter == correct_letter.upper() else 0.0


# =============================================================================
# Benchmark class
# =============================================================================

class LongCodeQABenchmark(BaseBenchmark):
    """LongCodeQA extreme-length code comprehension benchmark.

    Context truncation note (TODO — decision pending):
        LongCodeQA repos range from 32K to 1M tokens, but vLLM's default context
        window is 32K. Currently max_context_tokens() returns None, so for the
        ``full`` method main.py's tail-truncation kicks in: it keeps the LAST
        ~32K tokens of the assembled prompt (question + end of repo). This means
        the model loses the repo's beginning (imports, class definitions, module
        structure).

        Option A: Add max_context_tokens() returning e.g. 32000 to HEAD-truncate
        the repo before prompt assembly. The model would see repo start + question,
        losing the repo tail instead. This is likely better for code comprehension.

        Option B: Keep current tail-truncation. The ``full`` method naturally
        performs worse on extreme-length repos, which is part of what the benchmark
        measures (compression methods' advantage over naive truncation).

        Neither option affects compression methods — they reduce the repo to
        target_token size before prompt assembly, so truncation rarely triggers.
    """

    def __init__(self, dataset_path="Steefano/LCB"):
        self.dataset_path = dataset_path

    def load(self, args):
        """Load LongCodeQA dataset from HuggingFace.

        The dataset contains both LongCodeQA and LongSWE-Bench. We load
        only the LongCodeQA subset via data_files filter.

        Args:
            args: dict or namespace with keys:
                dataset_path (str): HuggingFace dataset path
                num_examples (int): Max examples (default all 443)
                filter_background_tokens_min (int): Min repo tokens (default None).
                    Maps to the CLI's --filter_background_tokens_min. The entire
                    repo is the "background" for LongCodeQA.

        Returns:
            (dataset_list, tokenizer)
        """
        if isinstance(args, dict):
            get = args.get
        else:
            get = lambda k, d=None: getattr(args, k, d)  # noqa: E731

        dataset_path = get("dataset_path", self.dataset_path) or self.dataset_path
        num_examples = get("num_examples", None)
        filter_background_tokens_min = get("filter_background_tokens_min", None)
        example_ids_spec = get("example_ids", None)
        example_ids = resolve_example_ids(example_ids_spec)
        expected_sha256 = manifest_expected_sha256(example_ids_spec)

        # A path denotes a content-pinned manifest, not merely a convenient list
        # of indices. Unlike older LCC manifests, these manifests
        # must always pin content and therefore never silently skip the hash
        # check when the field is missing.
        if isinstance(example_ids_spec, str) and expected_sha256 is None:
            raise ValueError(
                f"manifest {example_ids_spec!r} has no "
                "'selected_content_sha256' key"
            )

        logger.info(f"Loading LongCodeQA from {dataset_path}...")
        # Load only LongCodeQA subset (not LongSWE-Bench)
        # LOCAL-FIRST loading: a streaming load_dataset(repo_id, data_files=...)
        # re-reads the zip from the Hub over HTTP on every run, and a flaky
        # network surfaces as BadZipFile / stalls mid-run. Resolve the archive
        # to the local hub cache (download only if absent), then hand the json
        # builder chained fsspec URLs into the LOCAL file — zero network on the
        # hot path. Row content and order are unchanged (fsspec glob order is
        # sorted, identical to the hub-side xglob order).
        ds = load_longcodeqa_split(dataset_path)

        tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-Coder-7B-Instruct")
        logger.info("Tokenizer initialized.")

        if example_ids is not None:
            if len(example_ids) == 0:
                raise ValueError(
                    "example_ids resolved to an empty selection — refusing "
                    "to run an empty cell (check the manifest)"
                )
            selected = select_by_ids(ds, example_ids)
            if expected_sha256 is not None:
                actual_sha256 = compute_selected_sha256(
                    longcodeqa_audit_contents(selected)
                )
                if actual_sha256 != expected_sha256:
                    raise ValueError(
                        "manifest content hash mismatch: the pinned indices "
                        "no longer point at the same LongCodeQA rows "
                        "(dataset re-upload or archive row reorder?). "
                        f"expected sha256={expected_sha256}, "
                        f"got {actual_sha256}. Regenerate with "
                        "scripts/sample_longcodeqa_by_line_count.py."
                    )
            examples = [dict(ex) for ex in selected]
            logger.info(
                f"LongCodeQA example_ids selection: kept all {len(examples)} "
                "rows (num/token filters skipped)"
            )
            return examples, tokenizer

        # Collect and optionally filter
        examples = []
        for ex in ds:
            # Token-range filtering if specified
            if filter_background_tokens_min:
                n_tokens = len(tokenizer.encode(ex["repo_text"], add_special_tokens=False))
                if n_tokens < filter_background_tokens_min:
                    continue
            examples.append(dict(ex))
            if num_examples is not None and len(examples) >= num_examples:
                break

        logger.info(f"LongCodeQA: {len(examples)} examples loaded")
        return examples, tokenizer

    def split_context(self, example):
        """Split into (repo_code, question).

        Background = entire repository code.
        Query = the multiple-choice question.
        """
        return example["repo_text"], example["question"]

    def build_text_prompt(self, compressed_background, query):
        """Reproduce the canonical LongCodeQA model input (the dataset ``prompt`` field).

        Verified byte-for-byte on ``Steefano/LCB`` (the authors' official dataset):

            prompt == PROMPT_GOAL + "\\nRepository: " + repo_text
                                  + "\\nQuestion: "   + question + "\\n"

        where ``compressed_background`` = ``repo_text`` (the compressible arm content)
        and ``query`` = ``question``. At full budget this equals the dataset ``prompt``
        exactly; under compression the same wrapper holds so both arms see the identical
        instruction/question and differ only in the (matched) code content.
        """
        return f"{PROMPT_GOAL}\nRepository: {compressed_background}\nQuestion: {query}\n"

    def build_image_prompt(self, images, query):
        """Build multimodal prompt: repo images + question.

        Instruction parity with the text arm: the text part
        mirrors ``build_text_prompt`` — IMAGE_PROMPT_GOAL (PROMPT_GOAL with only the
        modality substitution) + the same ``\\nQuestion: {query}\\n`` wrapper; the
        ``Repository: {content}`` segment is replaced by the images themselves.
        """
        messages = []
        for img in images:
            messages.append({"type": "image", "image": img})
        messages.append({
            "type": "text",
            "text": f"{IMAGE_PROMPT_GOAL}\nQuestion: {query}\n",
        })
        return messages

    def generation_config(self):
        """Generation config for the MCQ.

        DIVERGENCE FROM UPSTREAM: upstream leaves
        ``max_output_length=None`` (uncapped) and relies on its end-of-string parser
        to pick the terminal letter after any free-form reasoning. Our prompt requests
        "ONLY the letter", so a compliant model needs very few tokens; we cap at 32 (a
        small margin over a bare letter / "The answer is X") to bound generation. A
        VERBOSE, non-compliant model reasoning past 32 tokens would be truncated with no
        terminal letter → ``parse_final_answer`` returns None → scored incorrect — i.e.
        this cap can UNDER-score a rambling model relative to upstream's uncapped run.
        The launcher sets the operative value (2,048 in the study).
        """
        return {
            "max_new_tokens": 32,
            "temperature": 0,
        }

    def score(self, prediction, example):
        """Score by checking if the predicted letter matches the correct answer.

        Returns:
            {"Accuracy": 0.0 or 100.0}
        """
        correct = example["correct_letter"]
        if prediction == "ERROR_GENERATING" or correct is None or not prediction.strip():
            return {"Accuracy": 0.0}

        return {"Accuracy": compute_accuracy(prediction, correct)}

    def metric_names(self):
        """LongCodeQA reports Accuracy."""
        return ["Accuracy"]

    def ground_truth(self, example):
        """Extract ground truth answer letter."""
        return example["correct_letter"]
