"""Direct LLM agent with single-round tool calling (no ReAct loop).

Fully async: uses ainvoke / astream and asyncio.sleep so the event loop
isn't blocked during retries or LLM calls.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import AsyncIterator
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from api.core.request_state import RequestState, RequestTracker

import openai
from concurrent.futures import ThreadPoolExecutor, TimeoutError as _FuturesTimeout

import jieba
from langchain_core.messages import ToolMessage, convert_to_messages
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI

from api.core.embedding import get_embeddings
from api.core.scene_classifier import _jd_pool_getconn
from api.core.token_counter import count_tokens

from api.core.config import get_settings
from api.services.rag_logger import RAGLogger
from api.core.input_builder import InputBuilder
from api.core.retry import NETWORK_EXC as _NETWORK_EXC, ainvoke_with_network_retry, backoff_delay as _backoff_delay
from api.models.session_store import SessionStore
from api.services.knowledge import SearchKnowledge, get_search_knowledge, set_search_knowledge
from api.services.product_card import pop_cards_for_session
from util.agent_tool import get_registered_tools

logger = logging.getLogger(__name__)

_L1_RETRY_NUDGE = "上一次回答为空。请直接给出简明中文回答，不要只思考。"

_CLIENT_ERROR_MESSAGE = "抱歉，服务器暂时出现了点问题，请稍后再试。"


class ClientError(Exception):
    """4xx LLM error - should not retry, return friendly message to customer."""


# Lazy-loaded to avoid circular import (app.orchestrator.__init__ -> service -> agent_runtime)
_RequestState = None


def _RS():
    """Cached RequestState enum import."""
    global _RequestState
    if _RequestState is None:
        from api.core.request_state import RequestState
        _RequestState = RequestState
    return _RequestState


def _append_retry_nudge(messages: list[dict]) -> list[dict]:
    """Append L1 retry nudge to messages (mutates in place).

    args:
        messages: current message list (mutated in place).

    returns: the same messages list (for chaining).
    """
    messages.append({"role": "system", "content": _L1_RETRY_NUDGE})
    return messages


def _thinking_extra_body(settings) -> dict[str, Any] | None:
    """Build the LLM request body for thinking-mode control.

    args:
        settings: app settings (chat_thinking_mode, openai_base_url).

    returns: extra_body dict to disable/enable thinking, or None.
    """
    mode = (settings.chat_thinking_mode or "auto").strip().lower()
    base = (settings.openai_base_url or "").lower()
    if mode == "auto" and "deepseek.com" in base:
        mode = "disabled"
    if mode == "disabled":
        return {"thinking": {"type": "disabled"}}
    if mode == "enabled":
        return {"thinking": {"type": "enabled"}}
    return None


class ReActQAAgent:
    """Direct LLM agent with optional single-round tool calling."""

    def __init__(self) -> None:
        self._settings = get_settings()
        self._session_store = SessionStore(max_history_turns=self._settings.max_history_turns)

        self._llm = ChatOpenAI(
            api_key=self._settings.openai_api_key,
            base_url=self._settings.openai_base_url,
            model=self._settings.model_name,
            temperature=self._settings.temperature,
            request_timeout=self._settings.chat_timeout_seconds,
            max_tokens=self._settings.chat_max_tokens,
            stop=["\n---\n"],
            extra_body=_thinking_extra_body(self._settings),
        )

        set_search_knowledge(SearchKnowledge())

        self._agens_llm: ChatOpenAI | None = None
        if self._settings.agens_api_key and self._settings.agens_base_url:
            self._agens_llm = ChatOpenAI(
                api_key=self._settings.agens_api_key,
                base_url=self._settings.agens_base_url,
                model=self._settings.agens_model_name,
                temperature=self._settings.agens_temperature,
                request_timeout=self._settings.chat_timeout_seconds,
            )

        self._fallback_llm: ChatOpenAI | None = None
        if self._settings.fallback_api_key and self._settings.fallback_base_url and self._settings.fallback_model_name:
            self._fallback_llm = ChatOpenAI(
                api_key=self._settings.fallback_api_key,
                base_url=self._settings.fallback_base_url,
                model=self._settings.fallback_model_name,
                temperature=self._settings.temperature,
                request_timeout=self._settings.chat_timeout_seconds,
            )
            logger.info("Fallback LLM configured: %s", self._settings.fallback_model_name)

        self._input_builder = InputBuilder(
            session_store=self._session_store,
            settings=self._settings,
            agens_llm=self._agens_llm,
        )

        self._langchain_tools = self._build_langchain_tools()
        self._llm_with_tools = (
            self._llm.bind_tools(self._langchain_tools, parallel_tool_calls=False)
            if self._langchain_tools else self._llm
        )
        self._tools_by_name = {e.name: e for e in get_registered_tools()}

    # ── 3-tier error handling helpers ────────────────────────────────

    async def _call_llm_ainvoke(
        self, llm: ChatOpenAI, messages: list,
    ) -> Any:
        """Call llm.ainvoke with 3-tier error handling.

        args:
            llm: ChatOpenAI instance (primary or fallback).
            messages: prompt messages in OpenAI format.

        returns: LLM response object.
        """
        supplier_retries = self._settings.supplier_retries
        for attempt in range(supplier_retries + 1):
            try:
                return await ainvoke_with_network_retry(
                    llm, messages, self._settings.network_retries, "LLM",
                )
            except openai.APIStatusError as exc:
                status = getattr(exc, "status_code", 0) or 0
                if 400 <= status < 500 and status != 429:
                    logger.warning("LLM 4xx status=%s - not retrying", status)
                    raise ClientError(f"LLM client error {status}") from exc
                if attempt < supplier_retries:
                    delay = _backoff_delay(attempt)
                    logger.warning(
                        "LLM 5xx status=%s attempt=%s/%s delay=%.1fs",
                        status, attempt + 1, supplier_retries, delay,
                    )
                    await asyncio.sleep(delay)
                    continue
                if self._fallback_llm and llm is not self._fallback_llm:
                    logger.warning("Switching to fallback model after 5xx exhaustion")
                    return await ainvoke_with_network_retry(
                        self._fallback_llm, messages, self._settings.network_retries, "LLM fallback",
                    )
                raise
        raise RuntimeError("_call_llm_ainvoke exhausted without returning")

    async def _call_llm_astream(
        self, llm: ChatOpenAI, messages: list,
    ) -> tuple[Any, Any]:
        """Call llm.astream_events with 3-tier error handling.

        args:
            llm: ChatOpenAI instance (primary or fallback).
            messages: prompt messages in OpenAI format.

        returns: (async_gen, first_event) — caller iterates gen for rest.
        """
        supplier_retries = self._settings.supplier_retries
        network_retries = self._settings.network_retries
        current_llm = llm
        network_attempt = 0
        supplier_attempt = 0
        while True:
            try:
                gen = current_llm.astream_events(messages, version="v2")
                first_event = await gen.__anext__()
                return gen, first_event
            except openai.APIStatusError as exc:
                status = getattr(exc, "status_code", 0) or 0
                if 400 <= status < 500 and status != 429:
                    logger.warning("LLM stream 4xx status=%s - not retrying", status)
                    raise ClientError(f"LLM client error {status}") from exc
                if supplier_attempt < supplier_retries:
                    delay = _backoff_delay(supplier_attempt)
                    logger.warning(
                        "LLM stream 5xx status=%s attempt=%s/%s delay=%.1fs",
                        status, supplier_attempt + 1, supplier_retries, delay,
                    )
                    await asyncio.sleep(delay)
                    supplier_attempt += 1
                    network_attempt = 0
                    continue
                if self._fallback_llm and current_llm is not self._fallback_llm:
                    logger.warning("Switching to fallback model after 5xx exhaustion (stream)")
                    current_llm = self._fallback_llm
                    supplier_attempt = 0
                    network_attempt = 0
                    continue
                raise
            except _NETWORK_EXC as exc:
                if network_attempt < network_retries:
                    delay = _backoff_delay(network_attempt)
                    logger.warning(
                        "Network error (stream) attempt=%s/%s delay=%.1fs %s: %s",
                        network_attempt + 1, network_retries, delay, type(exc).__name__, exc,
                    )
                    await asyncio.sleep(delay)
                    network_attempt += 1
                    continue
                # Network exhausted on primary -> try fallback once (consistent with 5xx path)
                if self._fallback_llm and current_llm is not self._fallback_llm:
                    logger.warning("Switching to fallback model after network exhaustion (stream)")
                    current_llm = self._fallback_llm
                    supplier_attempt = 0
                    network_attempt = 0
                    continue
                raise

    # ── Tool wiring ───────────────────────────────────────────────────

    def _build_langchain_tools(self) -> list:
        """Wrap registered @agent_tool entries as LangChain tool objects.

        returns: list of LangChain tool callables.
        """
        result: list = []
        for entry in get_registered_tools():
            if entry.param_model is not None:
                wrapped = tool(
                    entry.name,
                    description=entry.description,
                    args_schema=entry.param_model,
                )(entry.func)
            else:
                wrapped = tool(entry.name, description=entry.description)(entry.func)
            result.append(wrapped)
        return result

    def _execute_tool_call(self, tool_call: Any, tracker: RequestTracker | None) -> str:
        """Execute a single tool call synchronously (caller wraps in to_thread).

        args:
            tool_call: LangChain tool_call object/dict with name + args.
            tracker: request state tracker (for transition logging).

        returns: tool output string, or error message on failure.
        """
        name = getattr(tool_call, "name", "") or (tool_call.get("name", "") if isinstance(tool_call, dict) else "")
        args = getattr(tool_call, "args", {}) or (tool_call.get("args", {}) if isinstance(tool_call, dict) else {})
        RS = _RS()
        if tracker:
            state = RS.TOOL_SEARCH if name == "search_knowledge" else RS.TOOL_PRODUCT_CARD if name == "send_product_card" else RS.TOOL_DONE
            tracker.transition(state, tool=name)
        entry = self._tools_by_name.get(name)
        if entry is None:
            return f"工具 {name} 不存在"
        try:
            if tracker and name == "search_knowledge":
                _sk = get_search_knowledge()
                if _sk:
                    _sk.set_tracker(tracker)
            result = str(entry.func(**args))
            logger.info("【Tool 结果】tool=%s result_len=%d result=%.500s", name, len(result), result)
            if tracker:
                tracker.transition(RS.TOOL_DONE, tool=name)
            return result
        except Exception as exc:
            logger.exception("Tool %s failed", name)
            return f"工具 {name} 执行失败：{exc}"

    # ── Response parsing ──────────────────────────────────────────────

    @staticmethod
    def _extract_text(content: Any) -> str:
        if isinstance(content, list):
            return " ".join(
                str(item.get("text", "") if isinstance(item, dict) else item)
                for item in content
            ).strip()
        return str(content).strip()

    # ── Product card buffer ───────────────────────────────────────────

    def pop_product_cards(self, session_id: str) -> list[dict]:
        return pop_cards_for_session(session_id)

    def recent_user_messages(self, session_id: str, n: int = 3) -> list[str]:
        return self._session_store.recent_user_messages(session_id, n=n)

    # ── L2 fallback ───────────────────────────────────────────────────

    def _fallback_to_knowledge_base(self, question: str, scene: str) -> str:
        """L2 fallback: search KB and return formatted result.

        args:
            question: user's question to search.
            scene: current scene.

        returns: KB result prefixed with intro, or "" if no result.
        """
        try:
            sk = get_search_knowledge()
            if sk is None:
                return ""
            result = sk.search(question)
            if not result:
                return ""
            if any(marker in result for marker in ("No ", "未找到", "无相关", "Cannot search")):
                return ""
            return f"以下是根据知识库检索到的相关信息：\n\n{result}"
        except Exception:
            logger.exception("L2 knowledge base fallback failed")
            return ""

    # ── Pre-warm ───────────────────────────────────────────────────────

    def prewarm(self) -> None:
        """Warm everything the first chat request would otherwise touch cold.

        Each step is bounded by an explicit timeout so a slow upstream never
        stalls backend startup — if any step exceeds its budget we log and
        move on; the lazy initializer for that step will run on first real
        request.
        """
        logger.info("Pre-warming agent …")

        # ── 1. Embedding model ─────────────────────────────
        # Local model load can take a few seconds; cap at 30s.
        self._run_with_timeout(
            lambda: get_embeddings(),
            seconds=30,
            label="embedding model",
        )
        # ── 2. DB connection pool ──────────────────────────
        self._run_with_timeout(
            lambda: _jd_pool_getconn(),
            seconds=5,
            label="DB pool",
        )
        # ── 3. Token counter (tiktoken) ─────────────────────
        self._run_with_timeout(
            lambda: count_tokens(
                [{"role": "user", "content": "hi"}],
                self._settings.model_name,
            ),
            seconds=10,
            label="token counter",
        )
        # ── 4. jieba dict ───────────────────────────────────
        # Pure-CPU, no IO. 5s is generous.
        self._run_with_timeout(
            lambda: list(jieba.cut("预热", cut_all=False)),
            seconds=5,
            label="jieba",
        )
        # ── 5. LLM ping (TCP + auth warmup) ────────────────
        self._run_with_timeout(
            lambda: self._llm.invoke(
                [{"role": "user", "content": "hi"}],
                config={"max_tokens": 1, "temperature": 0, "timeout": 5},
            ),
            seconds=10,
            label="LLM ping",
        )
        logger.info("Agent pre-warm complete")

    @staticmethod
    def _run_with_timeout(fn, seconds: int, label: str) -> None:
        """Run ``fn`` in a worker thread; hard-timeout after ``seconds``.

        On timeout the orphan future is left running (Python can't kill
        threads) but the lifespan is no longer blocked. Errors are logged
        but never propagate, so one failed step doesn't abort the rest of
        prewarm.
        """
        ex = ThreadPoolExecutor(max_workers=1)
        try:
            ex.submit(fn).result(timeout=seconds)
        except _FuturesTimeout:
            logger.warning("%s prewarm timed out (>%ds)", label, seconds)
        except Exception as exc:
            logger.warning("%s prewarm failed: %s", label, exc)
        finally:
            # DO NOT wait for orphan thread. With wait=True, an in-flight
            # tiktoken HTTPS download or slow embedding load would block
            # startup long past the timeout. Let the daemon thread finish
            # in the background.
            ex.shutdown(wait=False)

    # ── Core LLM call with single-round tool ──────────────────────────

    @staticmethod
    def _log_llm_messages(lc_messages: list, label: str) -> None:
        """Log the LLM message list (for debug).

        args:
            lc_messages: list of LangChain message objects.
            label: tag for the log line.
        """
        parts = []
        for i, msg in enumerate(lc_messages):
            role = msg.__class__.__name__.replace("Message", "")
            content = getattr(msg, "content", "")
            tool_calls = getattr(msg, "tool_calls", None)
            text = str(content)[:300] if content else ""
            line = f"  [{i}] {role}: {text}"
            if tool_calls:
                line += f" tool_calls={len(tool_calls)}"
            parts.append(line)
        logger.info("【LLM Messages】%s (%d msgs)\n%s", label, len(lc_messages), "\n".join(parts))

    async def _ainvoke_with_tools(
        self,
        messages: list[dict],
        allow_tools: bool,
        tracker: RequestTracker | None,
    ) -> str:
        """Non-streaming LLM call with optional tool execution.

        args:
            messages: prompt messages in OpenAI format.
            allow_tools: whether LLM may call tools (False on L1 retry).
            tracker: request state tracker.

        returns: final assistant text (after tools if any).
        """
        lc_messages = convert_to_messages(messages)
        llm = self._llm_with_tools if allow_tools else self._llm
        response = await self._call_llm_ainvoke(llm, lc_messages)

        if not allow_tools:
            self._log_llm_messages(lc_messages, "ainvoke_no_tools")
            return self._extract_text(getattr(response, "content", ""))

        tool_calls = getattr(response, "tool_calls", None)
        if not tool_calls:
            return self._extract_text(getattr(response, "content", ""))

        lc_messages.append(response)
        tool_outputs = await asyncio.gather(
            *(asyncio.to_thread(self._execute_tool_call, tc, tracker) for tc in tool_calls)
        )
        for tc, output in zip(tool_calls, tool_outputs):
            tc_id = getattr(tc, "id", "") or (tc.get("id", "") if isinstance(tc, dict) else "")
            lc_messages.append(ToolMessage(content=output, tool_call_id=tc_id or "unknown"))

        self._log_llm_messages(lc_messages, "ainvoke_final")
        final = await self._call_llm_ainvoke(self._llm, lc_messages)
        return self._extract_text(getattr(final, "content", ""))

    async def _stream_with_tools(
        self,
        messages: list[dict],
        tracker: RequestTracker | None,
    ) -> AsyncIterator[str]:
        """Streaming LLM call with optional tool execution.

        args:
            messages: prompt messages in OpenAI format.
            tracker: request state tracker.

        yields: text deltas from the LLM response.
        """
        lc_messages = convert_to_messages(messages)

        # Phase 1: stream LLM with tools. Buffer text + collect tool_calls so we
        # can either discard the "thinking" text (when a tool is called) or
        # emit it verbatim (when no tool is needed). The frontend shows a
        # "thinking..." placeholder while we wait.
        collected_tool_calls: list = []
        final_response: Any = None
        phase1_buffer: list[str] = []

        gen, first_event = await self._call_llm_astream(self._llm_with_tools, lc_messages)
        try:
            for event in [first_event] + [e async for e in gen]:
                kind = event.get("event", "")
                if kind == "on_chat_model_stream":
                    chunk = event.get("data", {}).get("chunk")
                    if chunk is None:
                        continue
                    text = self._extract_text(getattr(chunk, "content", ""))
                    if text:
                        phase1_buffer.append(text)
                elif kind == "on_chat_model_end":
                    final_response = event.get("data", {}).get("output")
                    if final_response:
                        collected_tool_calls = getattr(final_response, "tool_calls", None) or []
        except Exception as exc:
            logger.exception("Mid-stream error (cannot retry): %s", exc)
            yield _CLIENT_ERROR_MESSAGE
            return

        if not collected_tool_calls:
            for t in phase1_buffer:
                yield t
            return

        # Phase 2: tool calls. Phase 1 buffer is discarded; the answer comes
        # from the post-tool LLM call below.
        lc_messages.append(final_response)
        tool_outputs = await asyncio.gather(
            *(asyncio.to_thread(self._execute_tool_call, tc, tracker) for tc in collected_tool_calls)
        )
        for tc, output in zip(collected_tool_calls, tool_outputs):
            tc_id = getattr(tc, "id", "") or (tc.get("id", "") if isinstance(tc, dict) else "")
            lc_messages.append(ToolMessage(content=output, tool_call_id=tc_id or "unknown"))

        self._log_llm_messages(lc_messages, "astream_final")
        gen2, first_event2 = await self._call_llm_astream(self._llm, lc_messages)
        try:
            for event in [first_event2] + [e async for e in gen2]:
                kind = event.get("event", "")
                if kind == "on_chat_model_stream":
                    chunk = event.get("data", {}).get("chunk")
                    if chunk is None:
                        continue
                    text = self._extract_text(getattr(chunk, "content", ""))
                    if text:
                        yield text
        except Exception as exc:
            logger.exception("Mid-stream error in phase 2 (cannot retry): %s", exc)
            yield _CLIENT_ERROR_MESSAGE

    def _resolve_fallback_answer(
        self,
        question: str,
        scene: str,
        last_err: Exception | None,
        tracker: RequestTracker | None,
        is_stream: bool,
    ) -> str:
        """Resolve L2 -> L3 fallback (KB search -> default reply).

        args:
            question: user's question.
            scene: current scene.
            last_err: last exception from LLM call (for logging).
            tracker: request state tracker.
            is_stream: True for streaming path (affects log msg).

        returns: L2 KB result, or L3 default reply if KB empty.
        """
        RS = _RS()
        answer = self._fallback_to_knowledge_base(question, scene)
        if tracker and answer:
            tracker.transition(RS.L2_FALLBACK)

        if answer:
            logger.warning("L2 %sknowledge-base fallback used", "streaming " if is_stream else "")
            return answer

        if tracker:
            tracker.transition(RS.L3_DEFAULT)
        logger.error(
            "All %sfallbacks exhausted last_err=%s",
            "streaming " if is_stream else "", last_err,
        )
        return "抱歉，我暂时无法回答，请换个问题或稍后再试。"

    # ── ask ───────────────────────────────────────────────────────────

    def _reject_if_leaked(self, answer: str, session_id: str, attempt: int, is_stream: bool) -> str:
        """Return "" if answer contains prompt-injection leak indicators.

        args:
            answer: LLM response text.
            session_id: for log correlation.
            attempt: retry attempt number (for log).
            is_stream: True for streaming path (affects log msg).

        returns: "" if rejected, else original answer.
        """
        if not InputBuilder.detect_response_leak(answer):
            return answer
        logger.warning(
            "LLM %s response contained prompt-injection leak indicators session_id=%s attempt=%s; rejecting",
            "streaming" if is_stream else "", session_id, attempt,
        )
        return ""

    async def ask(
        self,
        session_id: str,
        question: str,
        scene: str = "",
        dependencies: dict | None = None,
        tracker: RequestTracker | None = None,
        user_id: str | None = None,
        goods_name: str | None = None,
    ) -> str:
        """Non-streaming chat turn with L1 retry and L2/L3 fallback.

        args:
            session_id: conversation ID.
            question: user's message.
            scene: detected scene (presale/insale/aftersale).
            dependencies: context (shop_id, goods_id, order_sn, ...).
            tracker: request state tracker.

        returns: final assistant text (or L3 default reply on failure).
        """
        logger.info(
            "══════════════════ Q&A ── session=%s scene=%s ── question=%s ══════════════════",
            session_id, scene, question,
        )
        started = time.perf_counter()
        max_retries = self._settings.chat_retries
        last_err: Exception | None = None
        answer = ""

        messages = await self._input_builder.build_input_messages(
            session_id=session_id, question=question,
            scene=scene, dependencies=dependencies,
        )

        for attempt in range(max_retries + 1):
            try:
                logger.info(
                    "Agent request session_id=%s scene=%s attempt=%s/%s question=%s",
                    session_id, scene, attempt + 1, max_retries + 1, question,
                )

                # L1 retry nudge: on retry, append a system message asking for
                # direct answer. Helps when the first attempt returned empty
                # due to over-cautious safety filter or thinking-only output.
                if attempt > 0:
                    _append_retry_nudge(messages)

                if tracker:
                    tracker.transition(_RS().LLM_GENERATING)
                # Persist user message FIRST so DB row ordering matches
                # the conversation: user, then assistant. Without this,
                # normalize_answer below would land assistant at id=N and
                # persist_user at id=N+1 — on refresh the history would
                # render AI reply ABOVE the user question.
                self._input_builder.persist_user(
                    session_id, question, user_id=user_id, goods_name=goods_name,
                )
                raw_answer = await self._ainvoke_with_tools(messages, allow_tools=(attempt == 0), tracker=tracker)
                answer = self._input_builder.normalize_answer(raw_answer, session_id, user_id=user_id)
                answer = self._reject_if_leaked(answer, session_id, attempt + 1, is_stream=False)

                if not answer:
                    if attempt < max_retries:
                        if tracker:
                            tracker.transition(_RS().L1_RETRY, attempt=attempt + 1)
                        delay = _backoff_delay(attempt)
                        logger.warning(
                            "Empty answer attempt=%s/%s delay=%.1fs session_id=%s - retrying",
                            attempt + 1, max_retries + 1, delay, session_id,
                        )
                        await asyncio.sleep(delay)
                        continue
                    break

                elapsed_ms = int((time.perf_counter() - started) * 1000)
                logger.info(
                    "【LLM 输出】session=%s elapsed=%dms answer=%s",
                    session_id, elapsed_ms, answer,
                )
                return answer
            except ClientError as exc:
                logger.warning("ClientError, returning friendly message: %s", exc)
                return _CLIENT_ERROR_MESSAGE
            except Exception as exc:
                last_err = exc
                if attempt < max_retries:
                    delay = _backoff_delay(attempt)
                    logger.warning(
                        "Agent retry %s/%s delay=%.1fs session_id=%s error=%s",
                        attempt + 1, max_retries, delay, session_id, exc,
                    )
                    await asyncio.sleep(delay)

        # ── Fallback path ──
        # User message already written at the top of the loop above; we only
        # need to produce the fallback text.
        return self._resolve_fallback_answer(
            question=question, scene=scene, last_err=last_err, tracker=tracker, is_stream=False,
        )

    # ── ask_stream ────────────────────────────────────────────────────

    async def ask_stream(
        self,
        session_id: str,
        question: str,
        scene: str = "",
        dependencies: dict | None = None,
        tracker: RequestTracker | None = None,
        user_id: str | None = None,
        goods_name: str | None = None,
    ) -> AsyncIterator[str]:
        """Streaming chat turn with L1 retry and L2/L3 fallback.

        args:
            session_id: conversation ID.
            question: user's message.
            scene: detected scene (presale/insale/aftersale).
            dependencies: context (shop_id, goods_id, order_sn, ...).
            tracker: request state tracker.

        yields: assistant text deltas (or L3 default reply on failure).
        """
        logger.info(
            "══════════════════ Q&A (stream) ── session=%s scene=%s ── question=%s ══════════════════",
            session_id, scene, question,
        )
        started = time.perf_counter()
        max_retries = self._settings.chat_retries
        last_err: Exception | None = None
        answer = ""

        messages = await self._input_builder.build_input_messages(
            session_id=session_id, question=question,
            scene=scene, dependencies=dependencies,
        )

        for attempt in range(max_retries + 1):
            try:
                logger.info(
                    "Streaming request session_id=%s scene=%s attempt=%s/%s question=%s",
                    session_id, scene, attempt + 1, max_retries + 1, question,
                )

                if attempt > 0:
                    _append_retry_nudge(messages)

                answer_parts: list[str] = []
                if tracker:
                    tracker.transition(_RS().LLM_GENERATING)
                # Persist user message FIRST so row ordering matches
                # user, then assistant.
                self._input_builder.persist_user(
                    session_id, question, user_id=user_id, goods_name=goods_name,
                )
                async for chunk in self._stream_with_tools(messages, tracker):
                    answer_parts.append(chunk)
                    yield chunk

                answer = "".join(answer_parts).strip()
                answer = self._input_builder.normalize_answer(answer, session_id, user_id=user_id)
                answer = self._reject_if_leaked(answer, session_id, attempt + 1, is_stream=True)

                if not answer:
                    if attempt < max_retries:
                        if tracker:
                            tracker.transition(_RS().L1_RETRY, attempt=attempt + 1)
                        delay = _backoff_delay(attempt)
                        logger.warning(
                            "Empty streaming answer attempt=%s/%s delay=%.1fs session_id=%s - retrying",
                            attempt + 1, max_retries + 1, delay, session_id,
                        )
                        await asyncio.sleep(delay)
                        continue
                    break

                elapsed_ms = int((time.perf_counter() - started) * 1000)
                logger.info(
                    "【LLM 输出】session=%s elapsed=%dms answer=%s",
                    session_id, elapsed_ms, answer,
                )
                return
            except ClientError as exc:
                logger.warning("ClientError (stream), yielding friendly message: %s", exc)
                yield _CLIENT_ERROR_MESSAGE
                return
            except Exception as exc:
                last_err = exc
                if attempt < max_retries:
                    delay = _backoff_delay(attempt)
                    logger.warning(
                        "Streaming retry %s/%s delay=%.1fs session_id=%s error=%s",
                        attempt + 1, max_retries, delay, session_id, exc,
                    )
                    await asyncio.sleep(delay)

        # ── Fallback path ──
        # User message already written at the top of the loop above.
        yield self._resolve_fallback_answer(
            question=question, scene=scene, last_err=last_err, tracker=tracker, is_stream=True,
        )