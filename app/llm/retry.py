"""Shared retry utilities for LLM API calls.

Used by both `agent_runtime.py` (primary LLM + fallback) and `input_builder.py`
(agens summary LLM) to avoid duplicating backoff / exception-classification logic.
"""

from __future__ import annotations

import asyncio
import logging
import random
from typing import Any

import httpx
import openai

logger = logging.getLogger(__name__)

# Network-level exceptions eligible for tier-3 retry (timeout / connection reset).
NETWORK_EXC: tuple[type[Exception], ...] = (
    openai.APITimeoutError,
    openai.APIConnectionError,
    httpx.TimeoutException,
    httpx.ConnectError,
)


def backoff_delay(attempt: int) -> float:
    """Exponential backoff with jitter: 0.5s, 1s, 2s, 4s, 8s (capped) + 0-0.5s jitter."""
    base = min(0.5 * (2 ** attempt), 8.0)
    return base + random.uniform(0, 0.5)


async def ainvoke_with_network_retry(
    llm: Any, messages: list, max_retries: int, log_prefix: str = "",
) -> Any:
    """Call llm.ainvoke with tier-3 network retry (timeout / connection reset)."""
    for attempt in range(max_retries + 1):
        try:
            return await llm.ainvoke(messages)
        except NETWORK_EXC as exc:
            if attempt < max_retries:
                delay = backoff_delay(attempt)
                logger.warning(
                    "%s network error attempt=%s/%s delay=%.1fs %s: %s",
                    log_prefix, attempt + 1, max_retries, delay, type(exc).__name__, exc,
                )
                await asyncio.sleep(delay)
                continue
            raise

