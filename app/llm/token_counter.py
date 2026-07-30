"""Token counting utility using tiktoken with DeepSeek-compatible fallback.

DeepSeek models (deepseek-v4-flash etc.) are not in tiktoken's model registry.
We fall back to `o200k_base` encoding (GPT-4o tokenizer), which is based on the
same BPE algorithm and gives a close approximation (±5-10%) for DeepSeek's
tokenizer. Writing a fully accurate token counter would require either:

1. Downloading DeepSeek's specific tokenizer model (heavy dependency)
2. Reimplementing BPE from scratch (thousands of lines)

Neither is worthwhile since the count is only used for a compression-threshold
heuristic — an approximation is sufficient."""


import logging

import tiktoken

logger = logging.getLogger(__name__)

# Fallback when model name is not recognized by tiktoken
_FALLBACK_ENCODING = "o200k_base"


def _get_encoding(model: str) -> tiktoken.Encoding:
    try:
        return tiktoken.encoding_for_model(model)
    except KeyError:
        logger.debug("Unknown model %s, using fallback encoding %s", model, _FALLBACK_ENCODING)
        return tiktoken.get_encoding(_FALLBACK_ENCODING)


def count_tokens(messages: list[dict], model: str) -> int:
    """Count total tokens for a list of role/content message dicts."""
    if not messages:
        return 0

    enc = _get_encoding(model)
    total = 0
    for msg in messages:
        role = msg.get("role", "")
        content = msg.get("content", "") or ""
        total += len(enc.encode(role))
        total += len(enc.encode(content))
        # Each message adds ~4 tokens of overhead (formatting)
        total += 4
    return total
