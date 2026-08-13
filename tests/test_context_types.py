"""Tests for the platform context models (ContextType / ChannelType / Context)."""

from api.models.context import (
    ChannelKwargs,
    ChannelType,
    Context,
    ContextType,
)
from api.models.schemas import ChatRequest, ChatResponse


# ── ContextType flags ─────────────────────────────────────────────────


def test_silent_types() -> None:
    for t in (
        ContextType.WITHDRAW,
        ContextType.AUTH,
        ContextType.MALL_SYSTEM_MSG,
        ContextType.SYSTEM_BIZ,
        ContextType.SYSTEM_STATUS,
    ):
        assert t.is_silent, f"{t} should be silent"


def test_non_silent_types() -> None:
    for t in (ContextType.TEXT, ContextType.GOODS_CARD, ContextType.ORDER_INFO):
        assert not t.is_silent


def test_human_required_types() -> None:
    for t in (ContextType.IMAGE, ContextType.VIDEO):
        assert t.requires_human


# ── ChannelType enum ─────────────────────────────────────────────────


def test_channel_type_values() -> None:
    assert ChannelType.JINGDONG.value == "jingdong"
    assert ChannelType.PINDUODUO.value == "pinduoduo"


# ── Context defaults ──────────────────────────────────────────────────


def test_context_default_text() -> None:
    ctx = Context()
    assert ctx.type is ContextType.TEXT
    assert ctx.kwargs.shop_id is None


def test_context_with_kwargs() -> None:
    ctx = Context(
        type=ContextType.GOODS_CARD,
        channel_type=ChannelType.JINGDONG,
        kwargs=ChannelKwargs(shop_id="JD-MOCK-001", goods_id=2001, goods_name="iPhone 17"),
    )
    assert ctx.type is ContextType.GOODS_CARD
    assert ctx.channel_type is ChannelType.JINGDONG
    assert ctx.kwargs.shop_id == "JD-MOCK-001"
    assert ctx.kwargs.goods_id == 2001
    assert ctx.kwargs.goods_name == "iPhone 17"


# ── ChatRequest accepts context (backward compat) ────────────────────


def test_chat_request_without_context() -> None:
    req = ChatRequest(session_id="x", question="hi")
    assert req.context is None


def test_chat_request_with_context() -> None:
    req = ChatRequest(
        session_id="x",
        question="在吗？",
        context=Context(
            type=ContextType.TEXT,
            kwargs=ChannelKwargs(shop_id="JD-MOCK-001", goods_id=2001),
        ),
    )
    assert req.context is not None
    assert req.context.kwargs.shop_id == "JD-MOCK-001"


# ── ChatResponse default context_type ─────────────────────────────────


def test_chat_response_default_context_type() -> None:
    resp = ChatResponse(session_id="x", answer="ok")
    assert resp.context_type == "text"
    assert resp.actions == []
    assert resp.need_handoff is False