"""Full-lifecycle request state machine with ANSI bold phase logging.

Tracks every state transition from request received to response sent,
including RAG retrieval/ranking/generation phases.
"""

from __future__ import annotations

import logging
import time
from enum import Enum

logger = logging.getLogger(__name__)



class RequestState(str, Enum):
    # ── Orchestrator ──
    RECEIVED = "received"
    CONTEXT_PARSED = "context_parsed"
    SILENT_REPLY = "silent_reply"
    MEDIA_HANDOFF = "media_handoff"
    WELCOME = "welcome"
    GOODS_CARD_PROMPT = "goods_card_prompt"
    SCENE_CLASSIFIED = "scene_classified"
    RESPONSE_SENT = "response_sent"

    # ── Agent ──
    LLM_GENERATING = "llm_generating"
    LLM_DONE = "llm_done"
    ANSWER_NORMALIZED = "answer_normalized"
    L1_RETRY = "l1_retry"
    L2_FALLBACK = "l2_fallback"
    L3_DEFAULT = "l3_default"

    # ── RAG retrieval ──
    RAG_QUERY_CLEANED = "rag_query_cleaned"
    RAG_FETCHING = "rag_fetching"
    RAG_RANKING = "rag_ranking"
    RAG_RESULT_FORMATTED = "rag_result_formatted"
    RAG_NO_HIT = "rag_no_hit"

    # ── Tool calls ──
    TOOL_SEARCH = "tool_search"
    TOOL_PRODUCT_CARD = "tool_product_card"
    TOOL_DONE = "tool_done"


class RequestTracker:
    """Tracks state transitions for a single request, logging each with ANSI bold phase."""

    def __init__(self, session_id: str, trace_id: str) -> None:
        self.session_id = session_id
        self.trace_id = trace_id
        self._t0 = time.perf_counter()

    def transition(self, new_state: RequestState, **context: object) -> None:
        now = time.perf_counter()

        ctx_str = ""
        if context:
            ctx_str = " " + " ".join(f"{k}={v}" for k, v in context.items())

        elapsed = f" elapsed={int((now - self._t0) * 1000)}ms" if new_state == RequestState.RESPONSE_SENT else ""

        msg = f"▶ [{self.trace_id[:8]}] {new_state.value}{ctx_str}{elapsed}"
        logger.info(msg)