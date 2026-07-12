"""PostgreSQL-backed RAG index with in-memory vector similarity search."""

import logging
import time
import uuid
from io import BytesIO
from pathlib import Path
from typing import Any

import numpy as np
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import FastEmbedEmbeddings

from app.config import Settings
from app.db import create_session, init_db
from app.evaluation.rag_logger import RAGLogger
from app.rag_models import RagChunk

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Embedding helpers
# ---------------------------------------------------------------------------


def _build_embeddings(settings: Settings) -> Embeddings | None:
    """Build embedding model based on config. Currently only supports local (FastEmbed)."""
    provider = settings.embedding_provider.strip().lower()
    if provider in {"local", "bge", "m3e"}:
        import os
        cache_dir = os.getenv("FASTEMBED_CACHE_DIR", None)
        return FastEmbedEmbeddings(model_name=settings.local_embedding_model_name, cache_dir=cache_dir)
    logger.warning("EMBEDDING_PROVIDER=%s not supported for local RAG; use 'local'.", provider)
    return None


# ---------------------------------------------------------------------------
# RAGIndex — PostgreSQL + in-memory numpy vectors
# ---------------------------------------------------------------------------


class RAGIndex:
    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._top_k = settings.rag_top_k
        self._enabled = settings.rag_enabled
        self._score_threshold = settings.rag_score_threshold
        self._deduplicate_results = settings.rag_deduplicate_results

        self._embeddings: Embeddings | None = None
        self._embedding_array: np.ndarray | None = None
        self._chunks_meta: list[dict[str, Any]] = []

        init_db()
        if self._enabled:
            self._embeddings = _build_embeddings(settings)
            self._reload_cache()

    def warmup(self) -> None:
        if self._embeddings is None:
            self._embeddings = _build_embeddings(self._settings)

    # ------------------------------------------------------------------
    # In-memory cache
    # ------------------------------------------------------------------

    def _reload_cache(self) -> None:
        """Load all chunk embeddings from PostgreSQL into memory."""
        session = create_session()
        try:
            rows = session.query(RagChunk).filter(RagChunk.embedding_bytes.isnot(None)).all()
            if not rows:
                self._embedding_array = None
                self._chunks_meta = []
                return
            vectors = []
            metas = []
            for r in rows:
                try:
                    vec = np.frombuffer(r.embedding_bytes, dtype=np.float32)
                    vectors.append(vec)
                except Exception:
                    continue
                metas.append({
                    "source": r.source,
                    "snippet": r.snippet,
                    "file_name": r.file_name or "",
                    "page_number": r.page_number or 0,
                    "chunk_index": r.chunk_index,
                })
            self._embedding_array = np.stack(vectors) if vectors else None
            self._chunks_meta = metas
            logger.info("RAG cache loaded: %d chunks", len(metas))
        finally:
            session.close()

    # ------------------------------------------------------------------
    # Public API — same interface as before
    # ------------------------------------------------------------------

    def retrieve(self, user_query: str, trace_id: str = "") -> list[dict[str, Any]]:
        tid = trace_id or uuid.uuid4().hex[:12]
        if not self._enabled:
            return []
        if self._embeddings is None or self._embedding_array is None or len(self._embedding_array) == 0:
            return []

        started = time.perf_counter()
        RAGLogger.log_embedding(
            tid,
            model_name=getattr(self._embeddings, "model_name", "unknown"),
            dim=self._embedding_array.shape[1] if self._embedding_array.ndim == 2 else None,
            query_preview=user_query[:50],
        )

        query_vec = np.array(self._embeddings.embed_query(user_query), dtype=np.float32)
        scores = np.linalg.norm(self._embedding_array - query_vec, axis=1)  # L2 distance

        top_k = min(self._top_k, len(scores))
        if top_k == 0:
            return []
        top_indices = np.argpartition(scores, top_k - 1)[:top_k]
        top_indices = top_indices[np.argsort(scores[top_indices])]

        results: list[dict[str, Any]] = []
        seen: set[tuple[str, str, int, str]] = set()

        for idx in top_indices:
            score = float(scores[idx])
            if self._score_threshold >= 0 and score > self._score_threshold:
                continue
            meta = self._chunks_meta[idx]
            if self._deduplicate_results:
                dedup_key = (
                    meta["source"],
                    meta["file_name"],
                    meta["page_number"],
                    " ".join(meta["snippet"].split()),
                )
                if dedup_key in seen:
                    continue
                seen.add(dedup_key)
            results.append({
                "source": meta["source"],
                "score": score,
                "snippet": meta["snippet"],
                "file_name": meta["file_name"],
                "page_number": meta["page_number"],
            })

        elapsed_ms = (time.perf_counter() - started) * 1000
        RAGLogger.log_retrieval(tid, len(results), elapsed_ms, results)
        RAGLogger.log_rerank(tid, results)
        return results

    def query(self, user_query: str, trace_id: str = "") -> str:
        if not self._enabled:
            return "RAG is disabled."
        matches = self.retrieve(user_query, trace_id=trace_id)
        if not matches:
            return "No relevant context found."
        lines = []
        for item in matches:
            file_name = str(item.get("file_name", "")).strip()
            page_number = int(item.get("page_number", 0) or 0)
            meta_parts = [str(item["source"])]
            if file_name:
                meta_parts.append(file_name)
            if page_number > 0:
                meta_parts.append(f"第{page_number}页")
            lines.append(f"[{' | '.join(meta_parts)}] score={float(item['score']):.4f} | {item['snippet']}")
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Document ingestion
    # ------------------------------------------------------------------

    def add_document(self, source: str, content: str) -> int:
        text = content.strip()
        if not text:
            raise ValueError("Document content is empty.")
        documents = self._chunk_document(source=source, text=text)
        return self._add_chunks(source, documents)

    def add_pdf_bytes(self, source: str, filename: str, pdf_bytes: bytes) -> int:
        pages = self._extract_pdf_pages_from_bytes(pdf_bytes)
        if not pages:
            raise ValueError("Unable to extract text from PDF.")
        label = source.strip() or filename.strip() or "uploaded.pdf"
        documents: list[Document] = []
        for page_number, page_text in pages:
            documents.extend(
                self._chunk_document(source=label, text=page_text,
                                     metadata={"file_name": filename, "page_number": page_number})
            )
        return self._add_chunks(label, documents)

    def add_pdf_path(self, pdf_path: str, source: str | None = None) -> int:
        pages = self._extract_pdf_pages_from_path(pdf_path)
        if not pages:
            raise ValueError("Unable to extract text from PDF.")
        label = (source or "").strip() or Path(pdf_path).name
        file_name = Path(pdf_path).name
        documents: list[Document] = []
        for page_number, page_text in pages:
            documents.extend(
                self._chunk_document(source=label, text=page_text,
                                     metadata={"file_name": file_name, "page_number": page_number})
            )
        return self._add_chunks(label, documents)

    # ------------------------------------------------------------------
    # Internal methods
    # ------------------------------------------------------------------

    def _chunk_document(self, source: str, text: str, metadata: dict[str, Any] | None = None) -> list[Document]:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=max(1, self._settings.rag_chunk_size),
            chunk_overlap=max(0, self._settings.rag_chunk_overlap),
            separators=["\n\n", "\n", "。", "！", "？", "；", "，", " ", ""],
            keep_separator=True,
        )
        chunks = splitter.split_text(text)
        documents: list[Document] = []
        for idx, chunk in enumerate(chunks):
            value = chunk.strip()
            if not value:
                continue
            merged = {"source": source, "chunk_index": idx}
            if metadata:
                merged.update(metadata)
            documents.append(Document(page_content=value, metadata=merged))
        return documents

    def _add_chunks(self, source: str, documents: list[Document]) -> int:
        if not documents:
            raise ValueError("Document content is empty.")
        if self._embeddings is None:
            raise ValueError("Embedding model is not configured.")

        texts = [d.page_content for d in documents]
        vectors = self._embeddings.embed_documents(texts)

        session = create_session()
        try:
            for doc, vec in zip(documents, vectors):
                chunk = RagChunk(
                    source=str(doc.metadata.get("source", source)),
                    snippet=doc.page_content,
                    file_name=str(doc.metadata.get("file_name", "")),
                    page_number=int(doc.metadata.get("page_number", 0) or 0),
                    chunk_index=int(doc.metadata.get("chunk_index", 0)),
                    embedding_bytes=np.array(vec, dtype=np.float32).tobytes(),
                )
                session.add(chunk)
            session.commit()
        finally:
            session.close()

        self._reload_cache()
        return len(documents)

    @staticmethod
    def _extract_pdf_pages_from_bytes(pdf_bytes: bytes) -> list[tuple[int, str]]:
        if not pdf_bytes:
            raise ValueError("PDF file is empty.")
        try:
            from pypdf import PdfReader
        except ImportError:
            raise ValueError("Missing pypdf dependency.") from None
        reader = PdfReader(BytesIO(pdf_bytes))
        pages = []
        for idx, page in enumerate(reader.pages, start=1):
            text = (page.extract_text() or "").strip()
            if text:
                pages.append((idx, text))
        return pages

    @staticmethod
    def _extract_pdf_pages_from_path(pdf_path: str) -> list[tuple[int, str]]:
        path = Path(pdf_path).expanduser().resolve()
        if not path.exists() or not path.is_file():
            raise ValueError(f"PDF path does not exist: {pdf_path}")
        try:
            from pypdf import PdfReader
        except ImportError:
            raise ValueError("Missing pypdf dependency.") from None
        reader = PdfReader(str(path))
        pages = []
        for idx, page in enumerate(reader.pages, start=1):
            text = (page.extract_text() or "").strip()
            if text:
                pages.append((idx, text))
        return pages
