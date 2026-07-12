from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient
from langchain_core.documents import Document


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


def test_ingest_document_endpoint(api_client: TestClient, orch_stub: MagicMock) -> None:
    r = api_client.post(
        "/rag/documents",
        json={"source": "doc.txt", "content": "需要进入向量库的文档内容"},
    )
    assert r.status_code == 200
    assert r.json()["chunks_added"] == 1
    orch_stub.ingest_document.assert_called_once_with(source="doc.txt", content="需要进入向量库的文档内容")


def test_ingest_pdf_upload_endpoint(api_client: TestClient, orch_stub: MagicMock) -> None:
    r = api_client.post(
        "/rag/documents/pdf",
        files={"file": ("manual.pdf", b"%PDF-1.4 fake", "application/pdf")},
        data={"source": "售后手册.pdf"},
    )
    assert r.status_code == 200
    assert r.json()["source"] == "售后手册.pdf"
    orch_stub.ingest_pdf_bytes.assert_called_once()
    call_kwargs = orch_stub.ingest_pdf_bytes.call_args.kwargs
    assert call_kwargs["source"] == "售后手册.pdf"
    assert call_kwargs["filename"] == "manual.pdf"
    assert isinstance(call_kwargs["pdf_bytes"], bytes)


def test_ingest_pdf_path_endpoint(api_client: TestClient, orch_stub: MagicMock) -> None:
    r = api_client.post(
        "/rag/documents/pdf/path",
        json={"pdf_path": "/tmp/returns-policy.pdf", "source": "退货协议"},
    )
    assert r.status_code == 200
    assert r.json()["source"] == "退货协议"
    orch_stub.ingest_pdf_path.assert_called_once_with(pdf_path="/tmp/returns-policy.pdf", source="退货协议")


def test_rag_init_respects_disabled_flag(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RAG_ENABLED", "false")
    from app.config import get_settings
    from app.rag import RAGIndex
    get_settings.cache_clear()
    idx = RAGIndex(get_settings())
    assert idx._embeddings is None
    assert idx.retrieve("hello") == []


def test_rag_embeds_only_when_document_is_ingested(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RAG_ENABLED", "true")
    monkeypatch.setenv("EMBEDDING_PROVIDER", "local")
    from app.config import get_settings
    from app.rag import RAGIndex
    fake_emb = MagicMock()
    fake_emb.embed_documents.return_value = [[0.1, 0.2, 0.3]]
    get_settings.cache_clear()
    with (
        patch("app.rag._build_embeddings", return_value=fake_emb),
        patch("app.rag.RAGIndex._reload_cache"),
        patch("app.rag.RAGIndex._add_chunks", return_value=1),
    ):
        idx = RAGIndex(get_settings())
        chunks = idx.add_document(source="x.txt", content="doc content")
    assert chunks == 1


def test_rag_splitter_respects_overlap(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RAG_ENABLED", "true")
    monkeypatch.setenv("EMBEDDING_PROVIDER", "local")
    monkeypatch.setenv("RAG_CHUNK_SIZE", "12")
    monkeypatch.setenv("RAG_CHUNK_OVERLAP", "4")
    from app.config import get_settings
    from app.rag import RAGIndex
    get_settings.cache_clear()
    with patch("app.rag._build_embeddings", return_value=MagicMock()), patch("app.rag.RAGIndex._reload_cache"):
        idx = RAGIndex(get_settings())
        docs = idx._chunk_document(source="x.txt", text="第一段内容。第二段内容。第三段内容。")
    assert len(docs) >= 2
    assert any("第二段内容" in doc.page_content for doc in docs)


def test_rag_pdf_ingest_stores_page_and_filename_metadata(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RAG_ENABLED", "true")
    monkeypatch.setenv("EMBEDDING_PROVIDER", "local")
    from app.config import get_settings
    from app.rag import RAGIndex
    get_settings.cache_clear()
    with patch("app.rag._build_embeddings", return_value=MagicMock()), patch("app.rag.RAGIndex._reload_cache"):
        idx = RAGIndex(get_settings())
        docs = idx._chunk_document(
            source="退货协议", text="第一页内容",
            metadata={"file_name": "returns-policy.pdf", "page_number": 1},
        )
    assert len(docs) >= 1
    assert docs[0].metadata.get("file_name") == "returns-policy.pdf"
    assert docs[0].metadata.get("page_number") == 1


def test_rag_retrieve_applies_threshold_and_dedup(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RAG_ENABLED", "true")
    monkeypatch.setenv("EMBEDDING_PROVIDER", "local")
    monkeypatch.setenv("RAG_SCORE_THRESHOLD", "0.3")
    monkeypatch.setenv("RAG_DEDUPLICATE_RESULTS", "true")
    from app.config import get_settings
    from app.rag import RAGIndex
    import numpy as np
    fake_emb = MagicMock()
    fake_emb.embed_query.return_value = [1.0, 0.0, 0.0]
    get_settings.cache_clear()
    with patch("app.rag._build_embeddings", return_value=fake_emb), patch("app.rag.RAGIndex._reload_cache"):
        idx = RAGIndex(get_settings())
        idx._embedding_array = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0]], dtype=np.float32)
        idx._chunks_meta = [
            {"source": "A", "snippet": "s1", "file_name": "", "page_number": 0},
            {"source": "B", "snippet": "s2", "file_name": "", "page_number": 0},
            {"source": "A", "snippet": "s1", "file_name": "", "page_number": 0},
        ]
        results = idx.retrieve("test")
    sources = [r["source"] for r in results]
    assert len(results) == 1
    assert sources == ["A"]
