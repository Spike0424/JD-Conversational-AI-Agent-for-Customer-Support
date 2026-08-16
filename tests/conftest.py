"""Pytest 配置：默认注入假环境变量，避免依赖本机 .env；API 测试用 dependency_overrides 避免拉起真实 Agent。"""

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from api.models.schemas import ChatResponse


def run_async(coro):
    """Run an async coroutine from a sync test (shared shim)."""
    return asyncio.run(coro)


@pytest.fixture(autouse=True)
def _default_test_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
    monkeypatch.setenv("RAG_ENABLED", "false")
    from api.core.config import get_settings

    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


@pytest.fixture
def orch_stub() -> MagicMock:
    o = MagicMock()
    o.chat = AsyncMock(return_value=ChatResponse(session_id="s", answer="stub-ok", citations=[]))
    o.stream_chat.return_value = ("t1", "general_qa", [], [], iter(()))
    o.handoff.return_value = MagicMock()
    return o


@pytest.fixture
def auth_token() -> str:
    """Issue a test JWT for the default test user."""
    from api.core.auth import issue_token
    return issue_token("test@example.com")[0]


@pytest.fixture
def api_client(orch_stub: MagicMock, auth_token: str):
    from fastapi.testclient import TestClient

    from api.controllers.deps import get_orchestrator
    from api.main import app

    app.dependency_overrides[get_orchestrator] = lambda: orch_stub
    client = TestClient(app)
    client.headers["Authorization"] = f"Bearer {auth_token}"
    yield client
    app.dependency_overrides.clear()
