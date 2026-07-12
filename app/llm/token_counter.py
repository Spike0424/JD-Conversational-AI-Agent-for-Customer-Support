"""Token counting utility using tiktoken."""

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
