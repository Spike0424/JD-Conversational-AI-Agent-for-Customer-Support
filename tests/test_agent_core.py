"""Core tests: ask(), ask_stream(), InputBuilder message construction + compression."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from api.core.config import get_settings
from api.core.agent_runtime import ReActQAAgent
from api.core.input_builder import InputBuilder
from api.models.session_store import SessionStore
from tests.conftest import run_async


@pytest.fixture(autouse=True)
def _clear_env_for_agent(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
    monkeypatch.setenv("CHAT_RETRIES", "0")
    get_settings.cache_clear()


# ── ask() ──────────────────────────────────────────────────────────────


class FakeContentResponse:
    content: str

    def __init__(self, content: str) -> None:
        self.content = content


class FakeToolResponse:
    content: str = ""
    tool_calls: list[MagicMock]

    def __init__(self, tool_calls: list[MagicMock]) -> None:
        self.tool_calls = tool_calls


def _make_agent() -> ReActQAAgent:
    """Build an agent with a mocked LLM."""
    agent = ReActQAAgent.__new__(ReActQAAgent)
    agent._settings = get_settings()
    agent._session_store = SessionStore(max_history_turns=6)
    agent._llm = MagicMock()
    agent._agens_llm = None
    agent._llm_with_tools = agent._llm  # no tools bound in tests
    agent._langchain_tools = []
    agent._tools_by_name = {}
    agent._input_builder = InputBuilder(
        session_store=agent._session_store,
        settings=agent._settings,
        agens_llm=None,
    )
    return agent


def test_ask_returns_normalized_answer() -> None:
    agent = _make_agent()
    agent._llm.ainvoke = AsyncMock(return_value=MagicMock(content=" 你好！  "))
    answer = run_async(agent.ask(session_id="s1", question="hi"))
    assert answer == "你好！"


def test_ask_empty_answer_retries_then_default() -> None:
    agent = _make_agent()
    # Always return empty -> L2 fallback empty -> L3 default
    agent._llm.ainvoke = AsyncMock(return_value=MagicMock(content=""))
    # Disable L2 by making get_search_knowledge return None
    answer = run_async(agent.ask(session_id="s1", question="hi"))
    assert "抱歉" in answer or "暂时无法回答" in answer


def test_ask_with_single_retry() -> None:
    agent = _make_agent()
    agent._settings.chat_retries = 1
    # First call empty, second call returns answer
    agent._llm.ainvoke = AsyncMock(
        side_effect=[
            MagicMock(content=""),
            MagicMock(content="第二次回答"),
        ]
    )
    answer = run_async(agent.ask(session_id="s1", question="hi"))
    assert answer == "第二次回答"


def test_ask_exception_retries_then_fallback() -> None:
    agent = _make_agent()
    agent._settings.chat_retries = 1
    agent._llm.ainvoke = AsyncMock(
        side_effect=[
            RuntimeError("LLM timeout"),
            RuntimeError("LLM timeout again"),
        ]
    )
    answer = run_async(agent.ask(session_id="s1", question="hi"))
    assert "抱歉" in answer or "暂时无法回答" in answer


def test_ask_pops_product_cards_after_answer() -> None:
    agent = _make_agent()
    agent._llm.ainvoke = AsyncMock(return_value=MagicMock(content="答案"))
    run_async(agent.ask(session_id="s1", question="hi"))
    # Should not crash even without cards
    cards = agent.pop_product_cards("s1")
    assert cards == []


# ── ask_stream() ──────────────────────────────────────────────────────


def test_ask_stream_yields_answer_as_chunks() -> None:
    agent = _make_agent()
    # astream_events yields text tokens, no tool_calls at end
    async def _fake_astream_events(messages, version=None, **kw):
        yield {"event": "on_chat_model_stream", "data": {"chunk": MagicMock(content="流式")}}
        yield {"event": "on_chat_model_stream", "data": {"chunk": MagicMock(content="回答")}}
        yield {"event": "on_chat_model_end", "data": {"output": MagicMock(tool_calls=None)}}
    agent._llm_with_tools.astream_events = _fake_astream_events
    chunks = run_async(_collect_stream(agent, "s1", "hello"))
    assert "".join(chunks) == "流式回答"


def test_ask_stream_empty_answer_fallback() -> None:
    agent = _make_agent()
    async def _fake_astream_events(messages, version=None, **kw):
        yield {"event": "on_chat_model_end", "data": {"output": MagicMock(content="", tool_calls=None)}}
    agent._llm_with_tools.astream_events = _fake_astream_events
    chunks = run_async(_collect_stream(agent, "s1", "hello"))
    assert "抱歉" in "".join(chunks) or "暂时无法回答" in "".join(chunks)


async def _collect_stream(agent: ReActQAAgent, sid: str, q: str) -> list[str]:
    parts: list[str] = []
    async for chunk in agent.ask_stream(session_id=sid, question=q):
        parts.append(chunk)
    return parts


# ── InputBuilder.build_input_messages ────────────────────────────────


def _make_builder(agens_llm=None) -> InputBuilder:
    settings = get_settings()
    store = SessionStore(max_history_turns=6)
    return InputBuilder(session_store=store, settings=settings, agens_llm=agens_llm)


def test_build_input_messages_includes_scene_prompt() -> None:
    b = _make_builder()
    msgs = run_async(b.build_input_messages(session_id="s1", question="hi", scene="presale"))
    system = msgs[0]
    assert system["role"] == "system"
    assert "京东配送" in system["content"] or "京东" in system["content"]
    assert msgs[-1]["role"] == "user"
    assert msgs[-1]["content"] == "hi"


def test_build_input_messages_appends_session_info() -> None:
    b = _make_builder()
    deps = {"shop_id": "JD-001", "user_id": "u1", "goods_id": 2001}
    msgs = run_async(b.build_input_messages(session_id="s1", question="q", dependencies=deps))
    content = msgs[0]["content"]
    assert "【当前会话信息】" in content
    assert "<input>JD-001</input>" in content
    assert "<input>2001</input>" in content


def test_build_input_messages_no_session_info_when_deps_empty() -> None:
    b = _make_builder()
    msgs = run_async(b.build_input_messages(session_id="s1", question="q", dependencies={}))
    # The actual session-info block (appended by format_session_info) should not be present
    assert "\n\n【当前会话信息】\n" not in msgs[0]["content"]


def test_build_input_falls_back_to_default_prompt_for_unknown_scene() -> None:
    b = _make_builder()
    msgs = run_async(b.build_input_messages(session_id="s1", question="hi", scene="nonexistent"))
    content = msgs[0]["content"]
    assert "京东" in content


# ── InputBuilder._compress_history ────────────────────────────────────


def test_no_compress_when_agens_is_none() -> None:
    b = _make_builder(agens_llm=None)
    msgs = [
        {"role": "system", "content": "prompt"},
        {"role": "user", "content": "x" * 10000},
    ]
    # agens_llm is None -> should return unchanged
    result = run_async(b._compress_history(msgs, "s1"))
    assert result == msgs


def test_no_compress_under_threshold() -> None:
    mock_agens = MagicMock()
    mock_agens.ainvoke.return_value = MagicMock(content="summary")
    b = _make_builder(agens_llm=mock_agens)
    msgs = [{"role": "user", "content": "hello"}]
    # Token count will be tiny -> no compression
    result = run_async(b._compress_history(msgs, "s2"))
    assert result == msgs
    mock_agens.ainvoke.assert_not_called()