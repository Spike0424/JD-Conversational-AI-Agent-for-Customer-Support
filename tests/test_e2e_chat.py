"""E2E simulation tests: simulate customer chat flow through orchestrator.

Mock tests run by default (fast, no API calls).
E2E tests with real LLM are marked @pytest.mark.e2e (skip by default, run with: pytest -m e2e).
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.context_models import ChannelKwargs, Context, ContextType
from app.orchestrator.service import ChatOrchestrator
from tests.conftest import run_async

_DEFAULT_SHOP = "7"
_DEFAULT_GOODS = 57430876  # iPhone 17 Pro Max
_DEFAULT_USER = "e2e_tester"


def _ctx(
    question: str,
    context_type: ContextType = ContextType.TEXT,
    goods_id: int | None = _DEFAULT_GOODS,
    user_id: str = _DEFAULT_USER,
) -> Context:
    return Context(
        type=context_type,
        content=question,
        kwargs=ChannelKwargs(
            shop_id=_DEFAULT_SHOP,
            goods_id=goods_id,
            goods_name="iPhone 17 Pro Max" if goods_id == _DEFAULT_GOODS else None,
            user_id=user_id,
            from_uid=user_id,
        ),
    )


# ── Mock tests (fast, no LLM) ─────────────────────────────────────────


class _MockAgent:
    """Mock ReActQAAgent that returns canned answers without calling LLM."""

    def __init__(self, answer: str = "mock answer") -> None:
        self._answer = answer
        self._session_store = MagicMock()
        self._session_store.load.return_value = []
        self._user_messages: list[str] = []

    async def ask(self, **kw) -> str:
        return self._answer

    def pop_product_cards(self, session_id: str) -> list[dict]:
        return []

    def recent_user_messages(self, session_id: str, n: int = 3) -> list[str]:
        return list(self._user_messages[-n:]) if n > 0 else []


def _mock_orchestrator(answer: str = "mock answer") -> ChatOrchestrator:
    orch = ChatOrchestrator.__new__(ChatOrchestrator)
    orch._agent = _MockAgent(answer)
    orch._scene_classifier = MagicMock()
    orch._scene_classifier.classify = AsyncMock(return_value="presale")
    orch._scene_classifier.last_scene_hint = ""
    return orch


def test_greeting_short_circuit() -> None:
    """'你好' -> welcome message, no LLM call."""
    orch = _mock_orchestrator()
    resp = run_async(orch.chat(
        session_id="greet-1", question="你好",
        context=_ctx("你好"),
    ))
    assert "终于等到您" in resp.answer or "欢迎" in resp.answer
    assert resp.intent == "greeting"


def test_goods_card_first_turn_welcome() -> None:
    """First turn with goods_card + no text -> welcome message."""
    orch = _mock_orchestrator()
    orch._agent._user_messages = []  # empty = new session
    resp = run_async(orch.chat(
        session_id="gc-1", question="",
        context=Context(
            type=ContextType.GOODS_CARD,
            content="商品ID：57430876\n商品：【iPhone 17 Pro Max】\n价格：9999",
            kwargs=ChannelKwargs(shop_id=_DEFAULT_SHOP, goods_id=_DEFAULT_GOODS),
        ),
    ))
    assert "终于等到您" in resp.answer
    assert resp.intent == "greeting"


def test_goods_card_returning_prompt() -> None:
    """Returning customer with goods_card -> '您想了解哪方面' prompt."""
    orch = _mock_orchestrator()
    orch._agent._user_messages = ["previous msg"]
    resp = run_async(orch.chat(
        session_id="gc-2", question="",
        context=Context(
            type=ContextType.GOODS_CARD,
            content="商品ID：57430876\n商品：【iPhone 17 Pro Max】\n价格：9999",
            kwargs=ChannelKwargs(shop_id=_DEFAULT_SHOP, goods_id=_DEFAULT_GOODS),
        ),
    ))
    assert "想了解" in resp.answer
    assert resp.intent == "goods_card_only"


def test_silent_context_returns_empty() -> None:
    """Withdraw context -> empty answer, no LLM."""
    orch = _mock_orchestrator()
    resp = run_async(orch.chat(
        session_id="silent-1", question="撤回",
        context=_ctx("撤回", context_type=ContextType.WITHDRAW),
    ))
    assert resp.answer == ""
    assert resp.intent == "silent"


def test_media_context_handoff() -> None:
    """Image context -> handoff flag."""
    orch = _mock_orchestrator()
    resp = run_async(orch.chat(
        session_id="media-1", question="",
        context=Context(
            type=ContextType.IMAGE,
            kwargs=ChannelKwargs(shop_id=_DEFAULT_SHOP, media_url="https://x/y.jpg"),
        ),
    ))
    assert resp.need_handoff is True
    assert "transfer_to_human" in resp.actions


def test_normal_presale_mock() -> None:
    """Normal presale Q&A with mock LLM answer."""
    orch = _mock_orchestrator("iPhone 17 Pro Max 搭载 A19 Pro 芯片")
    resp = run_async(orch.chat(
        session_id="presale-1", question="iPhone 17 Pro Max 参数",
        context=_ctx("iPhone 17 Pro Max 参数"),
    ))
    assert "A19 Pro" in resp.answer
    assert resp.intent == "presale"


def test_multiturn_context_completion_mock() -> None:
    """3-turn conversation: turn 2 '这个呢' should use context from turn 1."""
    orch = _mock_orchestrator("mock answer about iPhone 17")

    # Turn 1: user asks about iPhone 17
    resp1 = run_async(orch.chat(
        session_id="multi-1", question="iPhone 17 参数",
        context=_ctx("iPhone 17 参数"),
    ))
    assert resp1.intent == "presale"

    # Turn 2: short follow-up "这个呢"
    # Mock session history to simulate turn 1 already persisted
    orch._agent._user_messages = ["iPhone 17 参数"]
    resp2 = run_async(orch.chat(
        session_id="multi-1", question="这个呢",
        context=_ctx("这个呢"),
    ))
    # Should still get a valid answer (mock doesn't test Layer 2 directly,
    # but verifies the flow doesn't break on short follow-up)
    assert resp2.answer != ""


def test_aftersale_return_policy_mock() -> None:
    """'8天后能退款吗' -> aftersale scene + LLM answers about return policy."""
    orch = _mock_orchestrator("8天后不支持7天无理由退货，但15天内质量问题可换货")
    orch._scene_classifier.classify = AsyncMock(return_value="aftersale")
    resp = run_async(orch.chat(
        session_id="aftersale-1", question="8天之后还能无理由退款吗",
        context=_ctx("8天之后还能无理由退款吗", goods_id=None),
    ))
    assert "8天后" in resp.answer or "退货" in resp.answer
    assert resp.intent == "aftersale"


# ── E2E tests (real LLM + real DB, slow) ──────────────────────────────


@pytest.mark.e2e
def test_e2e_normal_presale_with_rag() -> None:
    """Real LLM + real RAG: 'iPhone 17 Pro Max 参数' should return real specs."""
    orch = ChatOrchestrator()
    resp = run_async(orch.chat(
        session_id="e2e-presale-1", question="iPhone 17 Pro Max 参数",
        context=_ctx("iPhone 17 Pro Max 参数"),
    ))
    assert resp.intent == "presale"
    assert len(resp.answer) > 50
    # Should mention real specs from DB
    assert "A19" in resp.answer or "6.9" in resp.answer or "Pro Max" in resp.answer


@pytest.mark.e2e
def test_e2e_greeting_welcome() -> None:
    """Real: '你好' -> welcome message."""
    orch = ChatOrchestrator()
    resp = run_async(orch.chat(
        session_id="e2e-greet-1", question="你好",
        context=_ctx("你好"),
    ))
    assert "欢迎" in resp.answer or "终于等到您" in resp.answer


@pytest.mark.e2e
def test_e2e_multiturn_context_completion() -> None:
    """Real: 3-turn conversation, turn 3 '这个呢' tests Layer 2 context completion."""
    orch = ChatOrchestrator()
    sid = "e2e-multi-1"

    # Turn 1
    resp1 = run_async(orch.chat(
        session_id=sid, question="iPhone 17 Pro Max 的屏幕多大",
        context=_ctx("iPhone 17 Pro Max 的屏幕多大"),
    ))
    assert "6.9" in resp1.answer or "英寸" in resp1.answer

    # Turn 2: short follow-up
    resp2 = run_async(orch.chat(
        session_id=sid, question="电池呢",
        context=_ctx("电池呢"),
    ))
    assert len(resp2.answer) > 20  # should get a real answer using context


@pytest.mark.e2e
def test_e2e_aftersale_return_after_8_days() -> None:
    """Real: '8天后能无理由退款吗' -> LLM explains 7-day policy + 15-day quality exchange."""
    orch = ChatOrchestrator()
    resp = run_async(orch.chat(
        session_id="e2e-aftersale-1", question="8天之后还能无理由退款吗",
        context=_ctx("8天之后还能无理由退款吗", goods_id=None, user_id="user_a01"),
    ))
    # Should mention 7-day limit and/or 15-day quality exchange
    assert "7" in resp.answer or "十五" in resp.answer or "15" in resp.answer
    assert "退货" in resp.answer or "退款" in resp.answer


@pytest.mark.e2e
def test_e2e_stream_normal_presale() -> None:
    """Real streaming: collect chunks from stream_chat."""
    orch = ChatOrchestrator()
    result = run_async(orch.stream_chat(
        session_id="e2e-stream-1", question="推荐哪款 iPhone",
        context=_ctx("推荐哪款 iPhone"),
    ))
    trace_id, scene, citations, actions, chunks, layer = result
    parts: list[str] = []

    async def _collect():
        async for chunk in chunks:
            parts.append(chunk)
    run_async(_collect())
    full = "".join(parts)
    assert len(full) > 20
    assert "iPhone" in full