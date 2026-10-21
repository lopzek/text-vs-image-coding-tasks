"""Content-matched budget selection (not used by the study: every instance is delivered whole).

The budget axis is "how much code", held IDENTICAL across the text and image arms.
Select the retained code ONCE per (example, budget), then feed the same string
``C_b`` to both arms — text inline, image rendered at full legibility. Budget = |C_b|
tokens.

The selector is deterministic head-keep (the first N tokens): no model, no query,
so both arms provably see the same content.
"""


def select_content(background, budget_tokens, tokenizer):
    """Head-keep the first ``budget_tokens`` tokens of ``background``.

    Args:
        background: the full code-context string.
        budget_tokens: token budget. ``None`` keeps everything; ``0`` keeps nothing.
        tokenizer: an HF-style tokenizer — ``tokenizer(text, add_special_tokens=False)``
            returns a dict with ``input_ids``, and ``tokenizer.decode(ids)`` round-trips.

    Returns:
        str — the retained prefix, identical for both arms. If ``background`` already
        fits the budget it is returned verbatim (byte-identical, not re-decoded).
    """
    if budget_tokens is None:
        return background
    ids = tokenizer(background, add_special_tokens=False)["input_ids"]
    if len(ids) <= budget_tokens:
        return background
    return tokenizer.decode(ids[:budget_tokens])


def select_content_lines(background, budget_lines):
    """Head-keep the first ``budget_lines`` LINES of ``background``.

    The LoC unit is the model-agnostic budget: token counts are
    encoder-specific, so a token budget is not comparable across models, but a
    line-of-code budget is model-agnostic. Line-counting needs no tokenizer.

    Args:
        background: the full code-context string.
        budget_lines: line budget. ``None`` keeps everything; ``0`` keeps nothing.

    Returns:
        str — the first ``budget_lines`` lines, identical for both arms. If
        ``background`` already fits the budget it is returned verbatim (byte-identical,
        not re-joined).
    """
    if budget_lines is None:
        return background
    lines = background.split("\n")
    if len(lines) <= budget_lines:
        return background
    return "\n".join(lines[:budget_lines])


def resolve_content_budget(background_ctx, params, tokenizer):
    """Apply the content-matched budget once so BOTH arms get identical code.

    Reads a content-matched head-keep from ``params`` and applies it to
    ``background_ctx``; the caller feeds the result to the text arm AND the image
    render, so the budget axis is "how much code", not "which slice". This is the
    single seam where the budget is applied — calling
    it once before the text/image branch is what guarantees parity. Two content-matched
    units are supported and mutually exclusive:

    - ``keep_first_content_tokens`` — head-keep by tokens (needs ``tokenizer``).
    - ``keep_first_content_lines`` — head-keep by LINES of code (model-agnostic
      unit; no tokenizer required).

    Either is mutually exclusive with ``keep_last_text_tokens`` (tail-truncate, text
    arm only).

    Args:
        background_ctx: the full code-context string from ``split_context``.
        params: the run params dict (reads ``keep_first_content_tokens`` /
            ``keep_first_content_lines`` and, for the mutual-exclusion guard,
            ``keep_last_text_tokens``).
        tokenizer: HF-style tokenizer (only used by the token unit).

    Returns:
        str — ``background_ctx`` unchanged when no content budget is set (returned as
        the same object), else the head-kept ``C_b``.
    """
    keep_first_tokens = params.get("keep_first_content_tokens")
    keep_first_lines = params.get("keep_first_content_lines")
    if keep_first_tokens is None and keep_first_lines is None:
        return background_ctx
    if keep_first_tokens is not None and keep_first_lines is not None:
        raise ValueError(
            "keep_first_content_tokens and keep_first_content_lines are both "
            "content-matched head-keeps — set at most one."
        )
    if params.get("keep_last_text_tokens") is not None:
        raise ValueError(
            "content-matched head-keep (both arms) and keep_last_text_tokens "
            "(tail-truncate, text arm only) are mutually exclusive — set exactly one."
        )
    if keep_first_lines is not None:
        return select_content_lines(background_ctx, keep_first_lines)
    if tokenizer is None:
        raise ValueError("keep_first_content_tokens set but no tokenizer available to count tokens.")
    return select_content(background_ctx, keep_first_tokens, tokenizer)
