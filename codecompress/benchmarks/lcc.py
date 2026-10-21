"""LCC (Long Code Completion) benchmark.

Dataset: microsoft/LCC_python (or LCC_java, LCC_csharp)
Task: Code completion — given background code + function signature, generate the function body.
Metrics: Edit Similarity (ES), Exact Match (EM)
Reference: Guo et al., "LongCoder", ICML 2023

Context splitting: the code file is split at the last top-level def/class boundary.
Background = everything before; current function = the last def/class to EOF.
The model completes the current function.
"""

import hashlib
import json
import re

import datasets
import editdistance
from loguru import logger
from transformers import AutoTokenizer

from codecompress.benchmarks.base import BaseBenchmark


# =============================================================================
# example_ids selection (manifest-driven instance lists)
# =============================================================================

def resolve_example_ids(spec):
    """Resolve an ``example_ids`` spec to a list[int], or None.

    Accepts:
      * None                       -> None (default: no manifest filter)
      * list/tuple of ints         -> validated list (order preserved)
      * str path to a manifest JSON -> the manifest's ``example_indices``

    The manifest is the one written by
    ``scripts/sample_lcc_by_line_count.py``. Order is preserved (not
    re-sorted) so callers can control evaluation order if they want.
    """
    if spec is None:
        return None

    if isinstance(spec, str):
        ids = _read_manifest(spec)["example_indices"]
    elif isinstance(spec, (list, tuple)):
        ids = list(spec)
    else:
        raise ValueError(
            f"example_ids must be None, a list, or a manifest path; "
            f"got {type(spec).__name__}"
        )

    out = []
    seen = set()
    for i in ids:
        if isinstance(i, bool) or not isinstance(i, int):
            raise ValueError(f"example id {i!r} is not an int")
        if i < 0:
            raise ValueError(f"example id {i} must be non-negative (>= 0)")
        if i in seen:
            raise ValueError(
                f"duplicate example id {i} — a manifest must not list the "
                f"same example twice (sampler bug?)"
            )
        seen.add(i)
        out.append(i)
    return out


def _read_manifest(path):
    """Load + validate a sampler manifest JSON. Raises clear errors.

    A truncated/partial manifest write is a realistic failure, so a bad
    JSON body becomes a ValueError (not a raw JSONDecodeError). A missing
    file still raises FileNotFoundError (intentional — that's a hard
    misconfiguration, not data corruption).
    """
    with open(path) as f:  # FileNotFoundError propagates intentionally
        body = f.read()
    try:
        manifest = json.loads(body)
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"manifest {path!r} is not valid JSON (truncated/corrupt?): {exc}"
        ) from exc
    if not isinstance(manifest, dict) or "example_indices" not in manifest:
        raise ValueError(f"manifest {path!r} has no 'example_indices' key")
    return manifest


def compute_selected_sha256(contexts):
    """sha256 over an ordered iterable of context strings.

    MUST match the byte protocol in scripts/sample_lcc_by_line_count.py
    (utf-8 / surrogatepass, NUL separator after each item) so manifest
    provenance hashes can be verified. The golden test
    test_matches_documented_protocol_golden pins this — do not change one
    side without the other.
    """
    h = hashlib.sha256()
    for c in contexts:
        h.update(c.encode("utf-8", "surrogatepass"))
        h.update(b"\x00")
    return h.hexdigest()


def manifest_expected_sha256(spec):
    """Return the content hash recorded in a manifest path, else None.

    None for a bare id list / None spec, or a (legacy) manifest that has
    no selected_content_sha256 field — verification is then skipped.
    """
    if not isinstance(spec, str):
        return None
    return _read_manifest(spec).get("selected_content_sha256")


def select_by_ids(dataset, ids):
    """Return a dataset containing only ``ids``, in the given order.

    Raises ValueError (not the opaque datasets IndexError) if any id is
    outside ``[0, len(dataset))`` so misconfigured manifests fail loudly.
    """
    n = len(dataset)
    for i in ids:
        if i < 0 or i >= n:
            bound = f"0..{n - 1}" if n > 0 else "(dataset is empty)"
            raise ValueError(
                f"example id {i} is out of range {bound} "
                f"(dataset has {n} rows)"
            )
    return dataset.select(ids)


# =============================================================================
# LCC-specific metrics
# =============================================================================

def _normalize_lines(text):
    """Strip blank lines, comment-only lines, and indentation.

    Used for both target and prediction to ensure symmetric normalization.
    Removes ``#``-comment-only lines because models may emit extra comments
    that are not in the ground truth (upstream LongCodeZip convention).

    Note: only strips Python-style ``#`` comments. Java/C# ``//`` comments
    are NOT stripped. This matches LongCodeZip upstream behavior and is
    acceptable because LCC ground truths contain no comment-only lines in
    any language (0/10000 for Python, Java, and C#).

    Note: upstream LongCodeZip applies comment stripping only to predictions,
    not targets. We apply it to both for symmetry — otherwise identical
    strings containing comment-only lines produce EM=0 and ES<100. This does
    not affect scores on real LCC data (no ground truths contain comment-only
    lines in Python, Java, or C# splits).
    """
    return [line.strip() for line in text.splitlines()
            if line.strip() and not line.strip().startswith("#")]


# Keep old names as aliases so existing callers don't break
_normalize_target_lines = _normalize_lines
_normalize_prediction_lines = _normalize_lines


def compute_ES(target, prediction):
    """Compute edit similarity score. Returns 0.0 when both inputs are empty."""
    target_lines = _normalize_lines(target)
    target_str = '\n'.join(target_lines)
    prediction_lines = _normalize_lines(prediction)[:len(target_lines)]
    prediction_str = '\n'.join(prediction_lines)

    max_len = max(len(target_str), len(prediction_str))
    if max_len == 0:
        return 0.0

    return (1 - (editdistance.eval(target_str, prediction_str) / max_len)) * 100


def compute_EM(target, prediction):
    """Compute exact match score.

    Returns 0 when target is empty (nothing to match against).
    """
    target_lines = _normalize_lines(target)
    if not target_lines:
        return 0
    prediction_lines = _normalize_lines(prediction)[:len(target_lines)]

    if len(target_lines) != len(prediction_lines):
        return 0
    return (int(target_lines == prediction_lines)) * 100


# =============================================================================
# LCC-specific context splitting
# =============================================================================

def find_last_func_or_class_start(code_string, language="python"):
    """
    Finds the starting line of the last top-level function or class definition
    using line-based heuristics, robust to syntax errors.
    Accounts for decorators (Python) and annotations (Java/C#).
    Returns the 1-based line number or None if not found.
    """
    lines = code_string.splitlines()
    if not lines:
        return None
    last_def_line_index = -1

    # Language-specific patterns for function/class definitions
    if language in ("java", "csharp"):
        # Java/C#: method, constructor, or class definition.
        # Strategy: a line is a definition if it has `identifier(` pattern
        # (method/constructor call) or is a class/interface/struct/enum.
        # We scan backwards for the last such line.
        _class_kw = re.compile(r'^(?:class|interface|struct|enum)\s+')
        _modifier = re.compile(
            r'^(?:public|private|protected|static|final|abstract|override|virtual|'
            r'async|synchronized|volatile|readonly|partial|sealed|internal)\s+'
        )
        for i in range(len(lines) - 1, -1, -1):
            stripped_line = lines[i].lstrip()
            # Class/interface/struct/enum definitions
            if _class_kw.match(stripped_line):
                last_def_line_index = i
                break
            # Lines starting with a modifier keyword
            if _modifier.match(stripped_line):
                if _class_kw.search(stripped_line):
                    last_def_line_index = i
                    break
                # Must have parens (method/constructor) with no `=` before `(`
                # to distinguish `public void foo()` from `public int x = bar()`
                if '(' in stripped_line:
                    paren_idx = stripped_line.index('(')
                    eq_idx = stripped_line.find('=')
                    if eq_idx == -1 or eq_idx > paren_idx:
                        last_def_line_index = i
                        break
                continue  # field declaration — skip
            # Lines without modifiers: match `type identifier(` or `identifier(`
            # This catches package-private methods: void foo(), int bar(), Foo()
            if '(' in stripped_line and ')' in stripped_line:
                paren_idx = stripped_line.index('(')
                eq_idx = stripped_line.find('=')
                # Exclude: `=` before `(` (field init: `x = foo(5)`)
                if eq_idx != -1 and eq_idx < paren_idx:
                    continue
                # Exclude: `.` immediately before method name before `(`
                # e.g., `obj.method(` or `System.out.println(` — method calls
                # but NOT `java.util.List<String> getItems(` — qualified return type
                # Heuristic: check char before the identifier that precedes `(`
                pre_paren = stripped_line[:paren_idx].rstrip()
                if pre_paren and re.search(r'\.\w+$', pre_paren):
                    # Looks like `something.identifier(` — likely a method call
                    # But exclude if there's a space before the dot-chain
                    # `List<String> getItems(` — space before getItems, no dot
                    # `obj.method(` — no space before .method
                    last_space = pre_paren.rfind(' ')
                    last_dot = pre_paren.rfind('.')
                    if last_dot > last_space:
                        # dot is in the "identifier" part → method call
                        continue
                # Exclude: control flow keywords
                if re.match(r'^(?:if|else|for|while|switch|catch|return|throw|new|try|super|this)\b', stripped_line):
                    continue
                last_def_line_index = i
                break
    else:
        # Python: def/async def/class
        for i in range(len(lines) - 1, -1, -1):
            stripped_line = lines[i].lstrip()
            if re.match(r'^(def|async\s+def|class)\s+', stripped_line):
                last_def_line_index = i
                break

    if last_def_line_index != -1:
        # Found a potential start, now check for decorators/annotations above it
        start_line_index = last_def_line_index
        # Python: @decorator, Java/C#: @Annotation
        decorator_prefix = '@'
        # Comments to skip: Python=#, Java/C#=// or * (javadoc)
        comment_prefixes = ('#',) if language == "python" else ('//', '*', '/*')
        for i in range(last_def_line_index - 1, -1, -1):
            stripped_line = lines[i].lstrip()
            if stripped_line.startswith(decorator_prefix):
                start_line_index = i
            elif stripped_line == '' or any(stripped_line.startswith(c) for c in comment_prefixes):
                continue
            else:
                break
        return start_line_index + 1  # Return 1-based line number
    else:
        return None  # No function or class definition found


def split_context_ast(code_string, language="python"):
    """
    Splits the code context into background and current function/class context.

    Uses line-based heuristics (not actual AST parsing) to find the last
    function/class definition. Supports Python, Java, and C#.
    """
    lines = code_string.splitlines()
    split_line_1_based = find_last_func_or_class_start(code_string, language=language)

    if split_line_1_based is not None and split_line_1_based > 0:
        # split_line_1_based is the start of the function/class
        # Background is lines *before* that line
        background_lines = lines[:split_line_1_based - 1]
        current_func_lines = lines[split_line_1_based - 1:]
        return '\n'.join(background_lines), '\n'.join(current_func_lines)
    else:
        # If no function/class found or parse error, treat all as current
        return "", code_string


# =============================================================================
# LCC-specific filtering
# =============================================================================

def filter_dataset(examples, tokenizer, filter_current_lines_max=50,
                   filter_background_tokens_min=5000,
                   filter_background_tokens_max=None,
                   num_examples=None,
                   filter_valid_python=False,
                   filter_max_pages=None, filter_page_width=1120,
                   filter_page_height=1120, filter_font_size=20,
                   filter_consistent_pages=False):
    """Filter pre-processed examples by current-function length and background token count.

    Args:
        examples: Iterable of dicts with 'background_context' and
            'current_function_context' keys.
        tokenizer: Any object with an ``encode(text, add_special_tokens=...)`` method.
        filter_current_lines_max: Keep examples with at most this many lines
            in the current function context.
        filter_background_tokens_min: Keep examples with at least this many
            tokens in the background context.
        filter_background_tokens_max: If set, reject examples whose background
            token count is >= this value (half-open upper bound). The kept
            range is [filter_background_tokens_min, filter_background_tokens_max),
            so adjacent bands like [4000, 6000) and [6000, 10000) are
            non-overlapping. None = no upper bound.
        num_examples: If set, cap the returned list at this size.
        filter_valid_python: If True, reject examples whose background_context
            does not parse as valid Python 3 (via ``ast.parse``).
        filter_max_pages: If set, reject examples whose background_context
            would need more than this many pages when rendered at
            (filter_page_width, filter_page_height, filter_font_size).
        filter_page_width: Page width in pixels for page-count estimation.
        filter_page_height: Page height in pixels for page-count estimation.
        filter_font_size: Font size in pixels for page-count estimation.
        filter_consistent_pages: If True, reject examples where page count
            differs between styles with and without line numbers (i.e.,
            plain vs ide). Ensures all styles produce identical page counts.

    Returns:
        List of example dicts that passed all filters, capped at *num_examples*.
    """
    filtered = []
    for example in examples:
        curr_ctx = example['current_function_context']
        bg_ctx = example['background_context']

        curr_line_count = len(curr_ctx.splitlines())

        bg_token_count = 0
        if bg_ctx and bg_ctx.strip():
            bg_token_count = len(tokenizer.encode(bg_ctx, add_special_tokens=False))

        if curr_line_count > filter_current_lines_max or bg_token_count < filter_background_tokens_min:
            continue
        # Half-open upper bound: [min, max). An example with bg_tokens
        # exactly equal to `max` is REJECTED, so adjacent bands like
        # [4000, 6000) and [6000, 10000) are non-overlapping.
        if filter_background_tokens_max is not None and bg_token_count >= filter_background_tokens_max:
            continue

        if filter_valid_python:
            if not _is_valid_python(bg_ctx):
                continue

        if filter_max_pages is not None or filter_consistent_pages:
            from codecompress.code_renderer import _estimate_pages
            pages_no_ln = _estimate_pages(
                bg_ctx, filter_page_width, filter_page_height, filter_font_size,
                show_line_numbers=False,
            )
            pages_ln = _estimate_pages(
                bg_ctx, filter_page_width, filter_page_height, filter_font_size,
                show_line_numbers=True,
            )
            if filter_max_pages is not None and pages_ln > filter_max_pages:
                continue
            if filter_consistent_pages and pages_no_ln != pages_ln:
                continue

        filtered.append(example)

    if num_examples is not None and len(filtered) > num_examples:
        filtered = filtered[:num_examples]

    return filtered


def _is_valid_python(code):
    """Check if code parses as valid Python 3."""
    import ast
    if not code or not code.strip():
        return False
    try:
        ast.parse(code)
        return True
    except SyntaxError:
        return False


# =============================================================================
# Benchmark class
# =============================================================================

class LCCBenchmark(BaseBenchmark):
    """LCC code completion benchmark."""

    # Image-instruction variants. The image arm's text part is the only thing
    # that changes between variants (same images, same query, same decoding).
    # v0 is the instruction of the study; omitting --image_instruction_variant
    # selects it.
    IMAGE_INSTRUCTION_VARIANTS = {
        "v0": (  # baseline (unchanged)
            "The image(s) above show Python source code that provides context. "
            "Below is an incomplete function from the same file. "
            "Output ONLY the next line of code that continues this function. "
            "Do not explain. Do not add comments. Do not use markdown. "
            "Do not wrap in ```python. Output raw code only.\n\n"
        ),
        "v1": (  # verbose-role / engine framing
            "You are a code-completion engine. The image(s) above show Python "
            "source code that provides context. Below is an incomplete function "
            "from the same file. Emit exactly the single next line of code that "
            "continues this function — nothing else. Do not explain. Do not add "
            "comments. Do not use markdown. Do not wrap in ```python. "
            "Output raw code only.\n\n"
        ),
        "v2": (  # explicit-continuation + indentation cue (image can't see exact indent)
            "The image(s) above show Python source code that provides context. "
            "Read the code in the image carefully. The incomplete function below "
            "is from the same file. Output ONLY the next line of code that "
            "continues the function, preserving the exact indentation the next "
            "line requires. Do not explain. Do not add comments. Do not use "
            "markdown. Do not wrap in ```python. Output raw code only.\n\n"
        ),
        "v3": (  # terse-minimal
            "The image(s) above show Python source code. Continue the incomplete "
            "function below. Output only the next line of raw code.\n\n"
        ),
    }

    def __init__(
        self,
        dataset_path="microsoft/LCC_python",
        language="python",
        image_instruction_variant="v0",
    ):
        self.dataset_path = dataset_path
        # Auto-detect language from dataset path if not explicitly set
        if language == "python" and dataset_path != "microsoft/LCC_python":
            path_lower = dataset_path.lower()
            if "_java" in path_lower and "javascript" not in path_lower:
                language = "java"
            elif "_csharp" in path_lower or "csharp" in path_lower:
                language = "csharp"
        self.language = language
        if image_instruction_variant not in self.IMAGE_INSTRUCTION_VARIANTS:
            raise ValueError(
                f"Unknown image_instruction_variant '{image_instruction_variant}'. "
                f"Available: {sorted(self.IMAGE_INSTRUCTION_VARIANTS)}"
            )
        self.image_instruction_variant = image_instruction_variant

    def load(self, args):
        """Load LCC dataset, split contexts, and filter.

        Args:
            args: dict or namespace with keys:
                dataset_path (str): HuggingFace dataset path
                dataset_split (str): Dataset split (default "test")
                num_examples (int): Number of examples after filtering
                filter_current_lines_max (int): Max lines in current function
                filter_background_tokens_min (int): Min tokens in background

        Returns:
            (dataset, tokenizer)
        """
        # Support both dict and namespace
        if isinstance(args, dict):
            get = args.get
        else:
            get = lambda k, d=None: getattr(args, k, d)

        dataset_path = get("dataset_path", self.dataset_path) or self.dataset_path
        split = get("dataset_split", "test")
        num_examples = get("num_examples", 500)
        filter_current_lines_max = get("filter_current_lines_max", 50)
        filter_background_tokens_min = get("filter_background_tokens_min", 5000)
        filter_background_tokens_max = get("filter_background_tokens_max", None)
        filter_valid_python = get("filter_valid_python", False)
        filter_max_pages = get("filter_max_pages", None)
        filter_page_width = get("filter_page_width", 1120)
        filter_page_height = get("filter_page_height", 1120)
        filter_font_size = get("filter_font_size", 20)
        filter_consistent_pages = get("filter_consistent_pages", False)
        example_ids_spec = get("example_ids", None)
        example_ids = resolve_example_ids(example_ids_spec)
        expected_sha256 = manifest_expected_sha256(example_ids_spec)

        logger.info(f"Loading {num_examples} examples from {dataset_path} ({split} split)...")
        dataset = datasets.load_dataset(dataset_path, split=split)
        if example_ids is not None:
            # Manifest-driven selection: the manifest already
            # encodes the (line-count) selection criterion, so index the FULL
            # split and skip the line/token filter entirely below.
            if len(example_ids) == 0:
                raise ValueError(
                    "example_ids resolved to an empty selection — refusing to "
                    "run an empty cell (every length tier should have >=200 "
                    "rows; check the manifest)"
                )
            dataset = select_by_ids(dataset, example_ids)
            if expected_sha256 is not None:
                actual = compute_selected_sha256(
                    [dataset[i]["context"] for i in range(len(dataset))]
                )
                if actual != expected_sha256:
                    raise ValueError(
                        "manifest content hash mismatch: the pinned indices no "
                        "longer point at the same code (dataset re-upload or "
                        "Arrow row reorder?). "
                        f"expected sha256={expected_sha256}, got {actual}. "
                        "Regenerate the manifest with "
                        "scripts/sample_lcc_by_line_count.py."
                    )
        else:
            # Load extra examples to have enough after filtering
            dataset = dataset.select(range(min(num_examples * 10, len(dataset))))
        original_size = len(dataset)

        # Initialize tokenizer for filtering
        tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-Coder-7B-Instruct")
        logger.info("Tokenizer Qwen/Qwen2.5-Coder-7B-Instruct initialized.")

        # Split each example's context into background + current function
        logger.info("Splitting context into background and current function...")

        lang = self.language
        def add_split_context(example):
            background, current_func = split_context_ast(example["context"], language=lang)
            example["background_context"] = background
            example["current_function_context"] = current_func
            return example

        processed_dataset = dataset.map(add_split_context, num_proc=4)

        if example_ids is not None:
            # Manifest is authoritative: keep every selected row, in manifest
            # order, without the line/token filter (which would otherwise
            # drop rows the length-stratified design intends to evaluate).
            selected_dataset = processed_dataset
            logger.info(
                f"example_ids manifest filter: kept all {len(selected_dataset)} "
                f"selected rows (line/token filter skipped)"
            )
            return selected_dataset, tokenizer

        # Filter by current function length and background token count
        bg_range_desc = f">= {filter_background_tokens_min}"
        if filter_background_tokens_max is not None:
            bg_range_desc = (
                f"in [{filter_background_tokens_min}, "
                f"{filter_background_tokens_max})"
            )
        logger.info(
            f"Filtering: current func lines <= {filter_current_lines_max}, "
            f"background tokens {bg_range_desc}"
        )
        filtered_list = filter_dataset(
            processed_dataset,
            tokenizer,
            filter_current_lines_max=filter_current_lines_max,
            filter_background_tokens_min=filter_background_tokens_min,
            filter_background_tokens_max=filter_background_tokens_max,
            num_examples=num_examples,
            filter_valid_python=filter_valid_python,
            filter_max_pages=filter_max_pages,
            filter_page_width=filter_page_width,
            filter_page_height=filter_page_height,
            filter_font_size=filter_font_size,
            filter_consistent_pages=filter_consistent_pages,
        )

        selected_dataset = datasets.Dataset.from_list(filtered_list)
        logger.info(
            f"Filtering complete. Original: {original_size}, "
            f"Filtered: {len(selected_dataset)}"
        )

        return selected_dataset, tokenizer

    def split_context(self, example):
        """Split an LCC example into (background, current_function).

        The split is already done during load() — we just read the fields.
        Handles both the load()-produced field names (background_context,
        current_function_context) and the fixture field names (background, current).
        """
        bg = example.get("background_context", example.get("background", ""))
        query = example.get("current_function_context", example.get("current", ""))
        return bg, query

    def text_baseline_instruction(self):
        """LCC's ``build_text_prompt`` carries no instruction (it is just
        ``background + query``), so the image-pipeline TEXT BASELINE arm
        (``image_text``) must supply the completion instruction here to stay
        symmetric with the image arm's ``IMAGE_INSTRUCTION_VARIANTS`` text.

        Non-completion benchmarks (LCA summarization) opt out through the base
        default ``""``.
        """
        return (
            "Below is Python source code that provides context, "
            "followed by an incomplete function from the same file. "
            "Output ONLY the next line of code that continues the "
            "function. Do not explain. Do not add comments. Do not "
            "use markdown. Do not wrap in ```python. "
            "Output raw code only.\n\n"
        )

    def build_text_prompt(self, compressed_background, query):
        """Build prompt: compressed background + separator + current function.

        For the 'full' method, this is the entire file with truncation.
        For compression methods, compressed_background is already shortened.
        The query is the current function signature/prefix.
        """
        if compressed_background:
            return compressed_background + "\n\n" + query
        return query

    def build_image_prompt(self, images, query, instruction=None):
        """Build multimodal prompt: images of background + text of current function.

        The background code is rendered as image(s). The current function
        signature stays as text because it's short and needs character-level
        precision (the model must continue from the exact characters).

        instruction: optional explicit instruction text (must include any trailing
        separator before the query). When None, uses the configured
        image_instruction_variant (default "v0" = the original instruction).
        """
        if instruction is None:
            instruction = self.IMAGE_INSTRUCTION_VARIANTS[self.image_instruction_variant]
        messages = []
        # Add images
        for img in images:
            messages.append({"type": "image", "image": img})
        # Add instruction + current function as text
        messages.append({"type": "text", "text": instruction + query})
        return messages

    def generation_config(self):
        """LCC uses greedy decoding with 128 max new tokens."""
        return {
            "max_new_tokens": 128,
            "temperature": 0,
        }

    def score(self, prediction, example):
        """Score using Edit Similarity and Exact Match.

        Args:
            prediction: Model's generated text.
            example: Dict with 'gt' key containing ground truth.

        Returns:
            {"ES": float, "EM": float}
        """
        gt = example["gt"]
        if prediction == "ERROR_GENERATING" or gt is None:
            return {"ES": 0.0, "EM": 0.0}
        return {
            "ES": compute_ES(gt, prediction),
            "EM": compute_EM(gt, prediction),
        }

    def metric_names(self):
        """LCC reports Edit Similarity and Exact Match."""
        return ["ES", "EM"]

    def ground_truth(self, example):
        """Extract ground truth from an LCC example."""
        return example["gt"]
