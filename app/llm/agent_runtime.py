import logging
import time
from collections.abc import Iterator
from typing import Any

from langchain.agents import create_agent
from langchain_openai import ChatOpenAI

from app.config import Settings, get_settings
from app.rag import RAGIndex
from app.session_store import SessionStore
from app.tools import (
    get_business_tools,
    get_dev_tools,
)

logger = logging.getLogger(__name__)


def _thinking_extra_body(settings: Settings) -> dict[str, Any] | None:
    """Build provider-specific thinking-mode parameters."""
    mode = (settings.chat_thinking_mode or "auto").strip().lower()
    base = (settings.openai_base_url or "").lower()
    if mode == "auto" and "deepseek.com" in base:
        mode = "disabled"
    if mode == "disabled":
        return {"thinking": {"type": "disabled"}}
    if mode == "enabled":
        return {"thinking": {"type": "enabled"}}
    return None


REACT_SYSTEM_PROMPT = """You are an intelligent question-answering assistant.
Use a ReAct style workflow internally:
1) Reason about the question.
2) If needed, call tools.
3) Observe tool outputs.
4) Continue reasoning until you can answer.

Never expose chain-of-thought in your final response.
Return only a concise and helpful final answer in Chinese unless user asks otherwise.
"""


class ReActQAAgent:
    def __init__(self) -> None:
        self._settings = get_settings()
        self._session_store = SessionStore(self._settings)
        self._rag_index = RAGIndex(self._settings)

        llm = ChatOpenAI(
            api_key=self._settings.openai_api_key,
            base_url=self._settings.openai_base_url,
            model=self._settings.model_name,
            temperature=self._settings.temperature,
            extra_body=_thinking_extra_body(self._settings),
        )
        tools = []
        if self._settings.enable_business_tools:
            tools.extend(get_business_tools(self._settings, self._settings.docs_root, self._rag_index))
        if self._settings.enable_dev_tools:
            tools.extend(get_dev_tools())
        if not tools:
            raise ValueError("No tools enabled. Set ENABLE_BUSINESS_TOOLS or ENABLE_DEV_TOOLS to true.")
        self._graph = create_agent(
            model=llm,
            tools=tools,
            system_prompt=REACT_SYSTEM_PROMPT,
            debug=False,
        )

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

    def _build_input_messages(self, session_id: str, question: str) -> list[dict[str, str]]:
        history = self._session_store.load(session_id=session_id)
        return [*history, {"role": "user", "content": question}]

    def _persist_exchange(self, session_id: str, question: str, answer: str) -> None:
        self._session_store.append(session_id=session_id, role="user", content=question)
        self._session_store.append(session_id=session_id, role="assistant", content=answer)

    def ask(self, session_id: str, question: str) -> str:
        started = time.perf_counter()
        logger.info("Agent request session_id=%s question=%s", session_id, question)
        result = self._graph.invoke(
            {"messages": self._build_input_messages(session_id=session_id, question=question)},
        )
        answer = self._extract_answer(result)
        self._persist_exchange(session_id=session_id, question=question, answer=answer)
        elapsed_ms = int((time.perf_counter() - started) * 1000)
        logger.info("Agent response session_id=%s elapsed_ms=%s", session_id, elapsed_ms)
        return answer

    def ask_stream(self, session_id: str, question: str) -> Iterator[str]:
        started = time.perf_counter()
        logger.info("Streaming request session_id=%s question=%s", session_id, question)

        answer_parts: list[str] = []
        stream = self._graph.stream(
            {"messages": self._build_input_messages(session_id=session_id, question=question)},
            stream_mode="messages",
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
            result = self._graph.invoke(
                {"messages": self._build_input_messages(session_id=session_id, question=question)},
            )
            answer = self._extract_answer(result)
            if answer:
                yield answer

        self._persist_exchange(session_id=session_id, question=question, answer=answer)
        elapsed_ms = int((time.perf_counter() - started) * 1000)
        logger.info("Streaming response session_id=%s elapsed_ms=%s", session_id, elapsed_ms)
