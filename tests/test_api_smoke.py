"""API smoke tests: health, chat context routing, backward compatibility."""

from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi.testclient import TestClient


def test_health() -> None:
    from main import app

    r = TestClient(app).get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}


def test_chat_get_returns_usage_when_no_query_params(api_client: TestClient) -> None:
    r = api_client.get("/chat")
    assert r.status_code == 200
    body = r.json()
    assert "message" in body


# ── Chat with Context payload ─────────────────────────────────────────


def test_chat_with_silent_context_returns_empty(api_client: TestClient, orch_stub: MagicMock) -> None:
    orch_stub.chat = AsyncMock(
        return_value=MagicMock(
            session_id="s1", answer="", trace_id="t1", intent="silent",
            context_type="withdraw", citations=[], actions=[], need_handoff=False,
            metadata=None,
        )
    )
    r = api_client.post(
        "/chat",
        json={"session_id": "s1", "question": "撤回", "context": {"type": "withdraw"}},
    )
    assert r.status_code == 200
    assert r.json()["answer"] == ""
    assert r.json()["context_type"] == "withdraw"


def test_chat_with_goods_card_only_returns_prompt(api_client: TestClient, orch_stub: MagicMock) -> None:
    orch_stub.chat = AsyncMock(
        return_value=MagicMock(
            session_id="s1", answer="亲，您想了解这款商品的哪方面呢？", trace_id="t1",
            intent="goods_card_only", context_type="goods_card", citations=[],
            actions=[], need_handoff=False, metadata=None,
        )
    )
    r = api_client.post(
        "/chat",
        json={
            "session_id": "s1", "question": "",
            "context": {"type": "goods_card", "kwargs": {"goods_id": 2001, "goods_name": "iPhone 17"}},
        },
    )
    assert r.status_code == 200
    assert "想了解" in r.json()["answer"]


def test_chat_with_image_returns_handoff_flag(api_client: TestClient, orch_stub: MagicMock) -> None:
    orch_stub.chat = AsyncMock(
        return_value=MagicMock(
            session_id="s1", answer="", trace_id="t1", intent="media_handoff",
            context_type="image", citations=[], actions=["transfer_to_human"],
            need_handoff=True, metadata=None,
        )
    )
    r = api_client.post(
        "/chat",
        json={
            "session_id": "s1", "question": "",
            "context": {"type": "image", "kwargs": {"media_url": "https://x/y.jpg"}},
        },
    )
    body = r.json()
    assert body["need_handoff"] is True
    assert "transfer_to_human" in body["actions"]


def test_chat_without_context_still_works(api_client: TestClient, orch_stub: MagicMock) -> None:
    orch_stub.chat = AsyncMock(
        return_value=MagicMock(
            session_id="s1", answer="hi", trace_id="t1", intent="presale",
            context_type="text", citations=[], actions=[], need_handoff=False,
            metadata=None,
        )
    )
    r = api_client.post("/chat", json={"session_id": "s1", "question": "你好"})
    assert r.status_code == 200