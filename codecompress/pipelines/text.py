"""Helpers of the text side that the image pipeline and main.py share.

In this study the text input is served through the same chat path as the image input
(``--pipeline image --method image_text``, see pipelines/image.py), so the standalone text
pipeline of the original code base is not part of the package. This module keeps only the
three helpers that main.py and the image pipeline import from it, verbatim.
"""
from __future__ import annotations

import os

from loguru import logger


def keep_last_text_tokens(text: str, n: int | None, tokenizer) -> str:
    """Tail-truncate `text` to its last `n` tokens.

    Args:
        text: Raw text to truncate.
        n: If None, return `text` unchanged. If a positive int, keep the
           last n tokens. Must be >= 1.
        tokenizer: Any object with .encode(text, add_special_tokens=False) →
                   List[...] and .decode(tokens, skip_special_tokens=True) → str.

    Returns:
        The truncated text. If n >= tokenized length, returns the re-decoded
        original (which may differ from the input by whitespace but preserves
        content).
    """
    if n is None:
        return text
    if n < 1:
        raise ValueError(f"keep_last_text_tokens n must be >= 1, got {n}")
    tokens = tokenizer.encode(text, add_special_tokens=False)
    if len(tokens) <= n:
        return text
    return tokenizer.decode(tokens[-n:], skip_special_tokens=True)


def build_method_suffix(method, **params):
    """
    Build a unique directory suffix encoding the method and its key parameters.
    This ensures different configurations write to different result directories.
    """
    suffix = f"method_{method}"

    if method in ("random_line", "random_token"):
        suffix += f"_t{params['lingua_target_token']}"
    elif method == "rag":
        suffix += f"_w{params['rag_window_size']}_o{params['rag_overlap']}_k{params['rag_top_k']}"
    elif method == "function_rag":
        suffix += f"_lang{params['function_rag_language']}_k{params['function_rag_top_k']}"
    elif method == "llmlingua":
        suffix += f"_t{params['lingua_target_token']}"
    elif method == "longllmlingua":
        suffix += f"_t{params['lingua_target_token']}_cs{params['longlingua_chunk_size']}_o{params['longlingua_overlap']}"
    elif method == "code_compressor":
        rank_only = (params.get("code_compressor_fine_ratio", 1.0) == 1.0)
        detail = "_rankonly" if rank_only else f"fr{params['code_compressor_fine_ratio']}"
        if params.get("importance_beta", 0) > 0:
            detail += f"_b{params['importance_beta']}"
        suffix += f"_t{params['code_compressor_target_token']}{detail}"

    return suffix


def configure_cuda_visible_devices(
    devices: str | int | tuple | list | None, stage_name: str
) -> None:
    """Set CUDA_VISIBLE_DEVICES for a pipeline stage. No-op if devices is None/"".

    fire.Fire literal-evals CLI values, so "3" arrives as int 3 and "1,3" as a
    tuple — canonicalize everything to the comma-joined string os.environ needs.
    int 0 is a valid device (GPU 0), NOT a falsy no-op. bool is rejected because
    it is an int subclass and would silently become "True"/"False".
    """
    if devices is None or devices == "":
        logger.info(f"[{stage_name}] CUDA_VISIBLE_DEVICES unchanged (using current process setting).")
        return
    if isinstance(devices, bool):
        raise TypeError(f"[{stage_name}] bool is not a valid device spec: {devices!r}")
    if isinstance(devices, int):
        devices = str(devices)
    elif isinstance(devices, (tuple, list)):
        for d in devices:
            if isinstance(d, bool) or not isinstance(d, (int, str)):
                raise TypeError(f"[{stage_name}] invalid device entry {d!r} in {devices!r}")
        devices = ",".join(str(d) for d in devices)
    elif isinstance(devices, str):
        devices = devices.replace(" ", "")
    else:
        raise TypeError(f"[{stage_name}] invalid device spec type: {devices!r}")
    prev = os.environ.get("CUDA_VISIBLE_DEVICES", None)
    os.environ["CUDA_VISIBLE_DEVICES"] = devices
    logger.info(f"[{stage_name}] CUDA_VISIBLE_DEVICES set to '{devices}' (previous: {prev}).")


class TextPipeline:
    """Placeholder for the registry: the study's text input runs through the image pipeline."""

    def __init__(self, *args, **kwargs):
        raise NotImplementedError(
            "The text input of this study is served with --pipeline image --method image_text; "
            "the standalone text pipeline is not part of this package."
        )
