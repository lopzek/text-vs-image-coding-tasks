"""Abstract base class for benchmarks.

Each benchmark provides dataset-specific logic for:
- Loading and filtering data
- Splitting context into (background, query)
- Building prompts for text and image pipelines
- Generation configuration
- Scoring predictions against ground truth
"""

from abc import ABC, abstractmethod


class BaseBenchmark(ABC):
    """Interface that all benchmarks must implement."""

    @abstractmethod
    def load(self, args):
        """Load and filter the dataset.

        Args:
            args: Namespace or dict with dataset_path, num_examples,
                filter params, etc.

        Returns:
            (dataset, tokenizer) — the filtered dataset and the tokenizer
            used for token counting.
        """

    @abstractmethod
    def split_context(self, example):
        """Split a dataset example into background context and query.

        Args:
            example: A single dataset row (dict).

        Returns:
            (background: str, query: str) — background is the long part
            to be compressed; query is the short part (e.g., function
            signature, question).
        """

    @abstractmethod
    def build_text_prompt(self, compressed_background, query):
        """Build a text prompt from compressed background and query.

        Args:
            compressed_background: The compressed background context string.
            query: The query/current function string.

        Returns:
            str — the full prompt to send to the text LLM.
        """

    @abstractmethod
    def build_image_prompt(self, images, query):
        """Build a multimodal prompt from rendered images and query.

        Args:
            images: List of PIL Image objects (rendered background).
            query: The query/current function string.

        Returns:
            list[dict] — message list in the format expected by the VLM
            backend (e.g., OpenAI chat format with image_url entries).
        """

    @abstractmethod
    def generation_config(self):
        """Return generation parameters for this benchmark's task.

        Returns:
            dict with keys like max_new_tokens, temperature, etc.
        """

    @abstractmethod
    def score(self, prediction, example):
        """Score a single prediction against ground truth.

        Args:
            prediction: The model's generated output string.
            example: The original dataset row (dict) containing ground truth.

        Returns:
            dict of metric name → value (e.g., {"ES": 56.3, "EM": 100}).
        """

    def system_prompt(self):
        """Return an optional system prompt for chat-template generation.

        Benchmarks that require instruction-following (e.g., summarization)
        should override this to return a system message string.

        Returns:
            str or None — system prompt, or None to omit.
        """
        return None

    def text_baseline_instruction(self):
        """Return the instruction prefix for the image-pipeline TEXT BASELINE arm
        (``--pipeline image --method image_text``: the SAME VLM fed raw code text,
        the cleanest within-model contrast to the image arm).

        This exists because some benchmarks (LCC completion) carry NO instruction in
        ``build_text_prompt`` (it is just ``background + query``) — the instruction
        that the image arm gets via ``build_image_prompt`` must be supplied separately
        so the text baseline is symmetric. Benchmarks whose ``build_text_prompt`` is
        already self-contained (e.g. LCA summarization's shared ``_USER_TEMPLATE``)
        return ``""`` (the default): prepending a completion prefix on top of a
        summarization instruction would be a pro-text/anti-image asymmetry. Keep it
        parallel to ``build_image_prompt``'s instruction.

        Returns:
            str — instruction prefix (including any trailing separator), or "".
        """
        return ""

    def max_context_tokens(self):
        """Return optional token limit for background context pre-truncation.

        When set, main.py truncates the background context to this many tokens
        BEFORE building the prompt, keeping the head (first N tokens). This
        preserves prompt framing instructions that wrap the context.

        Returns:
            int or None — token limit, or None to skip pre-truncation.
        """
        return None

    @abstractmethod
    def metric_names(self):
        """Return the list of metric names this benchmark produces.

        Returns:
            list[str] — e.g., ["ES", "EM"] for LCC.
        """
