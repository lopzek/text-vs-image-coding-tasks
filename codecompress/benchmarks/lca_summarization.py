"""LCA Module Summarization benchmark.

Long Code Arena Module Summarization (``JetBrains-Research/lca-module-summarization``):
summarize a Python module's ``relevant_code_context`` into NL documentation about an
``intent`` for a target ``docfile_name``, scored against the gold ``target_text``.

The prompt wording follows CodeOCR's ``downstream/tasks/code_summarization.py``
(Shi et al.), made symmetric between the two input modalities:

- **identical system prompt both arms** — the neutral ``"You are a helpful assistant."``,
  NOT CodeOCR's image-specialized ``"...that generates documentation from code images."``;
- **identical user instruction both arms** — only the code-delivery CHANNEL differs
  (inline text vs "in the images"), kept grammatically parallel;
- the **anti-hallucination clause is a CONTROLLED FACTOR** given to BOTH arms or NEITHER
  (CodeOCR gave it to the image arm only). Primary contrast a priori = clause OFF.

Metric: BERTScore-F1 against the reference documentation (``score``).
"""

from loguru import logger

from codecompress.benchmarks.base import BaseBenchmark
from codecompress.benchmarks.lcc import (
    compute_selected_sha256,
    manifest_expected_sha256,
    resolve_example_ids,
    select_by_ids,
)


# --- Prompt strings, symmetric between the two modalities ---------------------

#: Neutral system prompt, identical for the text and image arms.
SYSTEM_PROMPT = "You are a helpful assistant."

#: The anti-hallucination steer CodeOCR attached to the IMAGE arm only. Here it is
#: a controlled factor: given to BOTH arms (clause ON) or NEITHER (clause OFF).
ANTI_HALLUCINATION_CLAUSE = (
    " Attention: If you are unsure or the code is unclear, do not make things up."
    " Only describe what can be clearly summarized."
)

#: User-instruction template. ``{query}`` = "documentation for {docfile} about {intent}";
#: ``{code_channel}`` = inline code (text arm) or the image pointer (image arm);
#: ``{clause}`` = the clause when the factor is on, else "". The ONLY thing that may
#: differ between the two arms is ``{code_channel}`` — the symmetry invariant (tested).
_USER_TEMPLATE = (
    "I have code collected from one or more files joined into one string. "
    "Using the code generate {query}.\n\n"
    "{code_channel}\n\n"
    "As answer return {query}. Do not return the instruction how to make "
    "documentation, return only documentation itself.{clause}"
)

#: Code-delivery channels — the single permitted point of divergence between arms.
#: Kept grammatically parallel ("My code is …") to avoid CodeOCR's "code"/"codes" drift.
_TEXT_CHANNEL = "My code:\n\n{background}"
_IMAGE_CHANNEL = "My code is in the images."


class LCASummarizationBenchmark(BaseBenchmark):
    """Composable LCA-LMS summarization benchmark with a fair, symmetric prompt.

    Args:
        dataset_path: HF dataset id (default the LCA module-summarization set).
        anti_hallucination_clause: the controlled factor — when True the clause is
            appended to BOTH arms' instruction; when False (default, the primary
            contrast) to NEITHER.
        max_context_toks: upper bound on ``relevant_code_context`` tokens used by
            ``load()``'s filter (CodeOCR-style single upper bound).
    """

    def __init__(
        self,
        dataset_path="JetBrains-Research/lca-module-summarization",
        anti_hallucination_clause=False,
        max_context_toks=32768,
    ):
        self.dataset_path = dataset_path
        self.anti_hallucination_clause = bool(anti_hallucination_clause)
        self._max_context_toks = max_context_toks
        self._bert = None  # lazy BERTScorer, populated in build 4

    # -- prompt helpers --------------------------------------------------------

    def _clause(self):
        return ANTI_HALLUCINATION_CLAUSE if self.anti_hallucination_clause else ""

    @staticmethod
    def _query_phrase(example):
        """The human-readable target, packed into ``query`` (ABC builders only
        receive ``(background, query)``)."""
        docfile = example.get("docfile_name", example.get("docfile", ""))
        intent = example.get("intent", "")
        return f"documentation for {docfile} about {intent}"

    # -- BaseBenchmark interface ----------------------------------------------

    def load(self, args):
        import datasets
        from transformers import AutoTokenizer

        get = args.get if isinstance(args, dict) else (lambda k, d=None: getattr(args, k, d))
        dataset_path = get("dataset_path", self.dataset_path) or self.dataset_path
        split = get("dataset_split", "test")
        num_examples = get("num_examples", 139)
        max_toks = get("filter_background_tokens_max", self._max_context_toks) or self._max_context_toks
        example_ids_spec = get("example_ids", None)
        example_ids = resolve_example_ids(example_ids_spec)
        expected_sha256 = manifest_expected_sha256(example_ids_spec)

        tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-Coder-7B-Instruct")
        ds = datasets.load_dataset(dataset_path, split=split)

        if example_ids is not None:
            if len(example_ids) == 0:
                raise ValueError(
                    "example_ids resolved to an empty selection — refusing "
                    "to run an empty LCA cell (check the manifest)"
                )
            if isinstance(example_ids_spec, str) and expected_sha256 is None:
                raise ValueError(
                    "LCA manifest has no 'selected_content_sha256'; "
                    "regenerate it with scripts/sample_lca_by_line_count.py"
                )
            ds = select_by_ids(ds, example_ids)
            if expected_sha256 is not None:
                actual_sha256 = compute_selected_sha256(lca_audit_contents(ds))
                if actual_sha256 != expected_sha256:
                    raise ValueError(
                        "manifest content hash mismatch: the pinned indices no "
                        "longer point at the same LCA rows (dataset re-upload or "
                        "Arrow row reorder?). "
                        f"expected sha256={expected_sha256}, got {actual_sha256}. "
                        "Regenerate with scripts/sample_lca_by_line_count.py."
                    )
            logger.info(
                f"LCA example_ids manifest: kept all {len(ds)} selected rows "
                "(token filter and num_examples truncation skipped)"
            )
            return ds, tokenizer

        def _fits(ex):
            code = ex.get("relevant_code_context", "") or ""
            n = len(tokenizer(code, add_special_tokens=False)["input_ids"])
            return n <= max_toks

        before_filter = len(ds)
        ds = ds.filter(_fits)
        logger.info(
            f"LCA token filter: kept {len(ds)}/{before_filter} rows "
            f"(dropped {before_filter - len(ds)}, max_tokens={max_toks})"
        )
        if num_examples and num_examples < len(ds):
            ds = ds.select(range(num_examples))
        return ds, tokenizer

    def split_context(self, example):
        """background = the module code; query = the target-documentation phrase."""
        bg = example.get("relevant_code_context", example.get("background", "")) or ""
        return bg, self._query_phrase(example)

    def build_text_prompt(self, compressed_background, query):
        return _USER_TEMPLATE.format(
            query=query,
            code_channel=_TEXT_CHANNEL.format(background=compressed_background),
            clause=self._clause(),
        )

    def build_image_prompt(self, images, query):
        text = _USER_TEMPLATE.format(
            query=query, code_channel=_IMAGE_CHANNEL, clause=self._clause()
        )
        messages = [{"type": "image", "image": img} for img in images]
        messages.append({"type": "text", "text": text})
        return messages

    def system_prompt(self):
        # Same neutral system prompt for both arms — forwarded to the text arm by
        # the pipeline and (since build 1/6) to the image arm too.
        return SYSTEM_PROMPT

    def generation_config(self):
        # A module summary is a paragraph, not a line (LCC=128) or a letter (QA=16).
        return {"max_new_tokens": 512, "temperature": 0}

    def ground_truth(self, example):
        return example.get("target_text", example.get("gt"))

    def metric_names(self):
        # ONLY the in-pipeline metric score() returns — main.py KeyErrors per example
        # if metric_names lists a key score() omits.
        return ["BERTScore_F1"]

    def max_context_tokens(self):
        # The budget axis is a content-matched selection (build 3), NOT head-
        # truncation here — return None so main.py does not pre-truncate.
        return None

    def _get_bert_scorer(self):
        """Lazy-load a single BERTScorer (roberta-large) — pinned for determinism."""
        if self._bert is None:
            try:
                from bert_score import BERTScorer
            except ImportError as e:  # pragma: no cover - env guard
                raise ImportError(
                    "LCASummarizationBenchmark.score needs the 'bert-score' package "
                    "(declared in pyproject; run `uv sync`)."
                ) from e
            self._bert = BERTScorer(
                model_type="roberta-large", lang="en",
                rescale_with_baseline=False, idf=False,
            )
        return self._bert

    def score(self, prediction, example):
        """Primary metric: BERTScore-F1 (0-100) of the prediction vs the gold summary.

        Deterministic (fixed model, no idf, no baseline rescale). A failed generation
        or a missing gold scores 0. Tests may inject a fake scorer via ``self._bert``.
        """
        gold = self.ground_truth(example)
        if prediction == "ERROR_GENERATING" or not gold:
            return {"BERTScore_F1": 0.0}
        # Empty/whitespace-only prediction: bert-score 0.3.13 on transformers 5.x
        # crashes (AttributeError) on empty sentences — score 0 BEFORE any model load.
        if not prediction or not prediction.strip():
            return {"BERTScore_F1": 0.0}
        _P, _R, F = self._get_bert_scorer().score([prediction], [gold])
        return {"BERTScore_F1": float(F[0]) * 100.0}


def lca_audit_contents(rows):
    """Yield the two manifest-hashed strings for each LCA row, in order.

    Both the sampler and loader use this generator so the provenance hash
    cannot drift from the benchmark's own context/ground-truth semantics.
    Each yielded string is separated with NUL by
    :func:`codecompress.benchmarks.lcc.compute_selected_sha256`.
    """
    benchmark = LCASummarizationBenchmark()
    for row in rows:
        background, _query = benchmark.split_context(row)
        yield background
        yield benchmark.ground_truth(row) or ""
