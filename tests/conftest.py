"""Pytest 配置：默认注入假环境变量，避免依赖本机 .env；API 测试用 dependency_overrides 避免拉起真实 Agent。"""

from unittest.mock import MagicMock

import pytest

from app.schemas import ChatResponse, DocumentIngestResponse


@pytest.fixture(autouse=True)
def _default_test_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
    monkeypatch.setenv("RAG_ENABLED", "false")
    from app.config import get_settings

    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


@pytest.fixture
def orch_stub() -> MagicMock:
    o = MagicMock()
    o.chat.return_value = ChatResponse(session_id="s", answer="stub-ok", citations=[])
    o.stream_chat.return_value = ("t1", "general_qa", [], [], iter(()))
    o.ingest_document.return_value = DocumentIngestResponse(source="doc.txt", chunks_added=1)
    o.ingest_pdf_bytes.side_effect = lambda source, filename, pdf_bytes: DocumentIngestResponse(
        source=source,
        chunks_added=2,
    )
    o.ingest_pdf_path.side_effect = lambda pdf_path, source=None: DocumentIngestResponse(
        source=source or pdf_path,
        chunks_added=2,
    )
    o.handoff.return_value = MagicMock()
    return o


@pytest.fixture
def api_client(orch_stub: MagicMock):
    from fastapi.testclient import TestClient

    from app.api.deps import get_orchestrator
    from main import app

    app.dependency_overrides[get_orchestrator] = lambda: orch_stub
    yield TestClient(app)
    app.dependency_overrides.clear()
