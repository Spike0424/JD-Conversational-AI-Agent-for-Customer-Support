import logging
import re
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from langchain.agents import create_agent
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI

from app.config import Settings, get_settings
from app.evaluation.rag_logger import RAGLogger
from app.llm.token_counter import count_tokens
from app.rag import RAGIndex
from app.session_store import SessionStore
from app.tools.search_knowledge import SearchKnowledge, set_search_knowledge
from util.agent_tool import get_registered_tools

logger = logging.getLogger(__name__)


def _thinking_extra_body(settings: Settings) -> dict[str, Any] | None:
    mode = (settings.chat_thinking_mode or "auto").strip().lower()
    base = (settings.openai_base_url or "").lower()
    if mode == "auto" and "deepseek.com" in base:
        mode = "disabled"
    if mode == "disabled":
        return {"thinking": {"type": "disabled"}}
    if mode == "enabled":
        return {"thinking": {"type": "enabled"}}
    return None


E_COMMERCE_PLATFORMS = "京东"

REACT_SYSTEM_PROMPT = f"""You are an intelligent question-answering assistant for a Chinese e-commerce customer service scenario.
Use a ReAct style workflow internally:
1) Reason about the question.
2) Call search_knowledge to retrieve relevant information from the knowledge base about {E_COMMERCE_PLATFORMS} products, policies, shipping, and after-sales procedures.
3) Observe tool outputs.
4) Answer the question based on the retrieved information.

Never expose chain-of-thought in your final response.
Return only a concise and helpful final answer in Chinese unless user asks otherwise."""


AGENS_SYSTEM_PROMPT = "你是一个对话摘要助手，简明总结对话要点。"


class ReActQAAgent:
    def __init__(self) -> None:
        self._settings = get_settings()
        self._session_store = SessionStore(max_history_turns=self._settings.max_history_turns)

        self._llm = ChatOpenAI(
            api_key=self._settings.openai_api_key,
            base_url=self._settings.openai_base_url,
            model=self._settings.model_name,
            temperature=self._settings.temperature,
            request_timeout=self._settings.chat_timeout_seconds,
            extra_body=_thinking_extra_body(self._settings),
        )
        self._graph: Any = None

        set_search_knowledge(SearchKnowledge())

        # Agens LLM for history compression (independent config)
        self._agens_llm: ChatOpenAI | None = None
        if self._settings.agens_api_key and self._settings.agens_base_url:
            self._agens_llm = ChatOpenAI(
                api_key=self._settings.agens_api_key,
                base_url=self._settings.agens_base_url,
                model=self._settings.agens_model_name,
                temperature=self._settings.agens_temperature,
                request_timeout=self._settings.chat_timeout_seconds,
            )

    def _create_graph(self):
        langchain_tools: list = []
        for entry in get_registered_tools():
            if entry.param_model is not None:
                wrapped = tool(
                    entry.name,
                    description=entry.description,
                    args_schema=entry.param_model,
                )(entry.func)
            else:
                wrapped = tool(entry.name, description=entry.description)(entry.func)
            langchain_tools.append(wrapped)
        if not langchain_tools:
            raise ValueError("No tools registered.  Ensure search_knowledge is imported and decorated.")

        return create_agent(
            model=self._llm,
            tools=langchain_tools,
            debug=False,
        )

    def _get_graph(self):
        if self._graph is None:
            self._graph = self._create_graph()
        return self._graph

    @staticmethod
    def _extract_text(content: Any) -> str:
        if isinstance(content, list):
            return " ".join(str(item.get("text", "") if isinstance(item, dict) else item) for item in content).strip()
        return str(content).strip()

    @staticmethod
    def _extract_answer(result: dict[str, Any]) -> str:
        messages = result.get("messages", [])
        if not messages:
            return ""
        content = getattr(messages[-1], "content", "")
        return ReActQAAgent._extract_text(content)

    @staticmethod
    def _normalize_text(text: str) -> str:
        return re.sub(r"\s+", " ", text.strip())

    def _normalize_answer(self, raw: str, session_id: str) -> str:
        """Filter internal terms, dedup against last assistant message, persist to history.

        L4: empty answer is NOT persisted to history (avoids polluting future context).
        """
        filtered = raw
        for word in self._settings.output_filter_words:
            filtered = filtered.replace(word, "")

        # L4: empty after filtering → return empty, do not persist
        if not filtered.strip():
            return ""

        normalized = self._normalize_text(filtered)
        last = self._session_store.load_last_assistant(session_id)
        if last is not None and self._normalize_text(last) == normalized:
            logger.info("Dedup hit session_id=%s — skipping history write", session_id)
            return filtered

        self._session_store.append(
            session_id=session_id, role="assistant", content=filtered,
        )
        return filtered

    def _fallback_to_knowledge_base(self, question: str, scene: str) -> str:
        """L2: Try to answer directly from SearchKnowledge when LLM produces empty."""
        try:
            from app.tools.search_knowledge import get_search_knowledge

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

    _SCENE_PROMPT_FILE: dict[str, str] = {
        "presale": "presales.md",
        "insale": "sales.md",
        "aftersale": "aftersales.md",
        "mixed": "mixed.md",
    }

    def _load_scene_prompt(self, scene: str) -> str:
        """Load scene-specific system prompt from prompt/{file}, fallback to default."""
        if not scene or not scene.strip():
            return REACT_SYSTEM_PROMPT

        filename = self._SCENE_PROMPT_FILE.get(scene.strip())
        if not filename:
            return REACT_SYSTEM_PROMPT

        prompt_path = Path(self._settings.prompt_dir) / filename
        try:
            return prompt_path.read_text(encoding="utf-8").strip()
        except (FileNotFoundError, OSError):
            logger.warning("Scene prompt not found: %s, using default", prompt_path)
            return REACT_SYSTEM_PROMPT

    def _build_input_messages(
        self, session_id: str, question: str, scene: str = "",
    ) -> list[dict[str, str]]:
        system_prompt = self._load_scene_prompt(scene)
        history = self._session_store.load(session_id=session_id)
        messages = [
            {"role": "system", "content": system_prompt},
            *history,
            {"role": "user", "content": question},
        ]
        return self._compress_history(messages, session_id)

    def _compress_history(
        self, messages: list[dict[str, str]], session_id: str
    ) -> list[dict[str, str]]:
        """Compress old messages into a summary when token count exceeds threshold."""
        if not self._agens_llm:
            return messages

        token_count = count_tokens(messages, self._settings.model_name)
        threshold = int(self._settings.history_window_tokens * self._settings.history_compress_threshold)

        if token_count <= threshold:
            return messages

        logger.info(
            "Compressing history session_id=%s tokens=%s threshold=%s",
            session_id, token_count, threshold,
        )

        keep_count = self._settings.history_keep_recent_turns * 2
        if keep_count >= len(messages):
            return messages

        old_messages = messages[:-keep_count]
        keep_messages = messages[-keep_count:]

        # Build compression prompt
        old_text = "\n".join(
            f"{m['role']}: {m['content']}" for m in old_messages
        )
        compress_prompt = f"请总结以下对话历史，保留关键信息：\n\n{old_text}"

        try:
            summary_msg = self._agens_llm.invoke([
                {"role": "system", "content": AGENS_SYSTEM_PROMPT},
                {"role": "user", "content": compress_prompt},
            ])
            summary = (
                getattr(summary_msg, "content", "") if hasattr(summary_msg, "content") else str(summary_msg)
            )
        except Exception:
            logger.exception("History compression failed session_id=%s", session_id)
            return messages

        if not summary.strip():
            return messages

        # Persist summary as system message
        self._session_store.append(
            session_id=session_id,
            role="system",
            content=f"[历史摘要] {summary.strip()}",
        )

        logger.info(
            "History compressed session_id=%s old_messages=%s summary_len=%s",
            session_id, len(old_messages), len(summary),
        )

        return [{"role": "system", "content": summary.strip()}, *keep_messages]

    def _persist_user(self, session_id: str, question: str) -> None:
        self._session_store.append(session_id=session_id, role="user", content=question)

    def ask(self, session_id: str, question: str, scene: str = "") -> str:
        started = time.perf_counter()
        max_retries = self._settings.chat_retries
        last_err: Exception | None = None
        answer = ""

        for attempt in range(max_retries + 1):
            try:
                logger.info(
                    "Agent request session_id=%s scene=%s attempt=%s/%s question=%s",
                    session_id, scene, attempt + 1, max_retries + 1, question,
                )
                result = self._get_graph().invoke(
                    {"messages": self._build_input_messages(session_id=session_id, question=question, scene=scene)},
                    config={"recursion_limit": self._settings.agent_recursion_limit},
                )
                raw_answer = self._extract_answer(result)
                answer = self._normalize_answer(raw_answer, session_id)

                # L1: silent retry on empty answer
                if not answer:
                    if attempt < max_retries:
                        delay = 0.5 * (attempt + 1)
                        logger.warning(
                            "Empty answer attempt=%s/%s delay=%.1fs session_id=%s — retrying",
                            attempt + 1, max_retries + 1, delay, session_id,
                        )
                        time.sleep(delay)
                        continue
                    # All retries exhausted → break out to fallback path
                    break

                self._persist_user(session_id, question)
                elapsed_ms = int((time.perf_counter() - started) * 1000)
                logger.info("Agent response session_id=%s elapsed_ms=%s", session_id, elapsed_ms)
                RAGLogger.log_trace(
                    session_id,
                    phase="generation",
                    user_query=question,
                    scene=scene,
                    llm_answer=answer,
                    elapsed_ms=elapsed_ms,
                )
                return answer
            except Exception as exc:
                last_err = exc
                if attempt < max_retries:
                    delay = 0.5 * (attempt + 1)
                    logger.warning(
                        "Agent retry %s/%s delay=%.1fs session_id=%s error=%s",
                        attempt + 1, max_retries, delay, session_id, exc,
                    )
                    time.sleep(delay)

        # ── Fallback path: L1 retries exhausted OR last_err ──
        elapsed_ms = int((time.perf_counter() - started) * 1000)
        self._persist_user(session_id, question)

        # L2: knowledge base direct fallback
        answer = self._fallback_to_knowledge_base(question, scene)

        # L3: default reply if L2 also fails
        if not answer:
            answer = "抱歉，我暂时无法回答，请换个问题或稍后再试。"
            logger.error(
                "All fallbacks exhausted session_id=%s last_err=%s",
                session_id, last_err,
            )
        else:
            logger.warning(
                "L2 knowledge-base fallback used session_id=%s",
                session_id,
            )

        RAGLogger.log_trace(
            session_id,
            phase="generation",
            user_query=question,
            scene=scene,
            llm_answer=answer,
            elapsed_ms=elapsed_ms,
            error=str(last_err) if last_err else "empty_answer",
        )
        return answer

    def _get_rag_index(self) -> RAGIndex:
        # Lazy-init only for admin ingest, not used in chat path
        if not hasattr(self, "_rag_idx"):
            self._rag_idx = RAGIndex(self._settings)
        return self._rag_idx

    def ingest_document(self, source: str, content: str) -> int:
        return self._get_rag_index().add_document(source=source, content=content)

    def ingest_pdf_bytes(self, source: str, filename: str, pdf_bytes: bytes) -> int:
        return self._get_rag_index().add_pdf_bytes(source=source, filename=filename, pdf_bytes=pdf_bytes)

    def ingest_pdf_path(self, pdf_path: str, source: str | None = None) -> int:
        return self._get_rag_index().add_pdf_path(pdf_path=pdf_path, source=source)

    def ask_stream(self, session_id: str, question: str, scene: str = "") -> Iterator[str]:
        started = time.perf_counter()
        max_retries = self._settings.chat_retries
        last_err: Exception | None = None
        answer = ""

        for attempt in range(max_retries + 1):
            try:
                logger.info(
                    "Streaming request session_id=%s scene=%s attempt=%s/%s question=%s",
                    session_id, scene, attempt + 1, max_retries + 1, question,
                )

                answer_parts: list[str] = []
                graph = self._get_graph()
                messages = self._build_input_messages(session_id=session_id, question=question, scene=scene)
                stream = graph.stream(
                    {"messages": messages},
                    stream_mode="messages",
                    config={"recursion_limit": self._settings.agent_recursion_limit},
                )

                for event in stream:
                    chunk = event[0] if isinstance(event, tuple) else event
                    chunk_type = chunk.__class__.__name__
                    if "AIMessageChunk" not in chunk_type:
                        continue

                    text = self._extract_text(getattr(chunk, "content", ""))
                    if not text:
                        continue

                    answer_parts.append(text)
                    yield text

                answer = "".join(answer_parts).strip()
                if not answer:
                    result = graph.invoke(
                        {"messages": self._build_input_messages(session_id=session_id, question=question, scene=scene)},
                    )
                    answer = self._extract_answer(result)
                    if answer:
                        yield answer

                answer = self._normalize_answer(answer, session_id)

                # L1: silent retry on empty answer
                if not answer:
                    if attempt < max_retries:
                        delay = 0.5 * (attempt + 1)
                        logger.warning(
                            "Empty streaming answer attempt=%s/%s delay=%.1fs session_id=%s — retrying",
                            attempt + 1, max_retries + 1, delay, session_id,
                        )
                        time.sleep(delay)
                        continue
                    break

                self._persist_user(session_id, question)
                elapsed_ms = int((time.perf_counter() - started) * 1000)
                logger.info("Streaming response session_id=%s elapsed_ms=%s", session_id, elapsed_ms)
                RAGLogger.log_trace(
                    session_id,
                    phase="generation",
                    user_query=question,
                    scene=scene,
                    llm_answer=answer,
                    elapsed_ms=elapsed_ms,
                )
                return
            except Exception as exc:
                last_err = exc
                if attempt < max_retries:
                    delay = 0.5 * (attempt + 1)
                    logger.warning(
                        "Streaming retry %s/%s delay=%.1fs session_id=%s error=%s",
                        attempt + 1, max_retries, delay, session_id, exc,
                    )
                    time.sleep(delay)

        # ── Fallback path: L1 retries exhausted OR last_err ──
        elapsed_ms = int((time.perf_counter() - started) * 1000)
        self._persist_user(session_id, question)

        # L2: knowledge base direct fallback
        answer = self._fallback_to_knowledge_base(question, scene)

        # L3: default reply if L2 also fails
        if not answer:
            answer = "抱歉，我暂时无法回答，请换个问题或稍后再试。"
            logger.error(
                "All streaming fallbacks exhausted session_id=%s last_err=%s",
                session_id, last_err,
            )
        else:
            logger.warning(
                "L2 streaming knowledge-base fallback used session_id=%s",
                session_id,
            )

        yield answer

        RAGLogger.log_trace(
            session_id,
            phase="generation",
            user_query=question,
            scene=scene,
            llm_answer=answer,
            elapsed_ms=elapsed_ms,
            error=str(last_err) if last_err else "empty_answer",
        )
        return
