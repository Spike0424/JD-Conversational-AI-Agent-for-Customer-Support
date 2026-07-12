"""In-memory numpy vector search over Q&A pairs."""

import logging
from typing import Any

import numpy as np
from langchain_core.embeddings import Embeddings
from langchain_community.embeddings import FastEmbedEmbeddings

from app.config import get_settings
from app.db import create_session as create_qa_session
from app.qa_models import QAPair

logger = logging.getLogger(__name__)


class QAIndex:
    """Lightweight QA pair vector index: embed query → numpy L2 → top-k matches."""

    def __init__(self) -> None:
        settings = get_settings()
        self._top_k = settings.rag_top_k
        self._embeddings: Embeddings | None = FastEmbedEmbeddings(
            model_name=settings.local_embedding_model_name
        )
        self._qa_array: np.ndarray | None = None
        self._qa_rows: list[dict[str, Any]] = []
        self._reload()

    def _reload(self) -> None:
        session = create_qa_session()
        try:
            rows = session.query(QAPair).filter(QAPair.embedding_bytes.isnot(None)).all()
            if not rows:
                self._qa_array = None
                self._qa_rows = []
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
                    "id": r.id,
                    "question": r.question,
                    "answer": r.answer,
                    "source": r.source,
                    "source_doc": r.source_doc,
                    "category": r.category,
                })
            self._qa_array = np.stack(vectors) if vectors else None
            self._qa_rows = metas
            logger.info("QAIndex loaded: %d pairs", len(metas))
        finally:
            session.close()

    def search(self, query: str, top_k: int | None = None) -> list[dict[str, Any]]:
        k = top_k or self._top_k
        if self._qa_array is None or len(self._qa_array) == 0:
            return []

        query_vec = np.array(self._embeddings.embed_query(query), dtype=np.float32)
        scores = np.linalg.norm(self._qa_array - query_vec, axis=1)

        top_n = min(k, len(scores))
        top_indices = np.argpartition(scores, top_n - 1)[:top_n]
        top_indices = top_indices[np.argsort(scores[top_indices])]

        results = []
        for idx in top_indices:
            meta = self._qa_rows[idx]
            meta["score"] = float(scores[idx])
            results.append(meta)
        return results

    def count(self) -> int:
        return len(self._qa_rows)


# Singleton
_qa_index: QAIndex | None = None


def get_qa_index() -> QAIndex:
    global _qa_index
    if _qa_index is None:
        _qa_index = QAIndex()
    return _qa_index
