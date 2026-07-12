"""Aftersale two-layer retrieval.

Layer 1: exact alias match on aftersale_knowledge → answer
Layer 2: pgvector cosine-distance search on aftersale_chunks → chunk_content
"""

import logging
from typing import Any

from sqlalchemy import text

from app.db import create_session
from app.retrieval.embedding import get_embeddings

logger = logging.getLogger(__name__)


class AftersaleRetriever:
    """Layer 2: pgvector search on aftersale_chunks for a given set of knowledge IDs."""

    def __init__(self, knowledge_ids: list[int], top_k: int = 4) -> None:
        self._knowledge_ids = knowledge_ids
        self._top_k = top_k

    def search(self, query: str) -> list[dict[str, Any]]:
        if not self._knowledge_ids:
            return []

        emb = get_embeddings()
        query_vec = emb.embed_query(query)

        session = create_session()
        try:
            session.execute(text(f"SET hnsw.ef_search = {self._top_k * 4}"))
            placeholders = ",".join(
                [f":id{i}" for i in range(len(self._knowledge_ids))]
            )
            params = {
                f"id{i}": v for i, v in enumerate(self._knowledge_ids)
            }
            params["vec"] = query_vec
            params["top_k"] = self._top_k

            result = session.execute(
                text(
                    f"SELECT id, knowledge_id, chunk_content, "
                    f"       embedding <=> :vec AS distance "
                    f"FROM aftersale_chunks "
                    f"WHERE knowledge_id IN ({placeholders}) AND embedding IS NOT NULL "
                    f"ORDER BY embedding <=> :vec LIMIT :top_k"
                ),
                params,
            )
            rows = result.fetchall()
        finally:
            session.close()

        results = []
        for r in rows:
            results.append({
                "source": f"aftersale_chunk#{r[0]}",
                "score": float(r[3]),
                "snippet": (r[2] or "")[:300],
                "chunk_content": r[2],
                "knowledge_id": r[1],
            })
        return results

    def search_formatted(self, query: str) -> str:
        results = self.search(query)
        if not results:
            return "No matching aftersale rule found."

        lines = []
        for i, r in enumerate(results, start=1):
            lines.append(f"[{i}] score={r['score']:.4f}\n{r['chunk_content']}")
        return "\n\n".join(lines)


def match_aftersale_exact(
    query: str, shop_id: int, goods_id: int | None = None
) -> str | None:
    """Layer 1: exact keyword match on aftersale_knowledge.aliases."""
    session = create_session()
    try:
        sql = (
            "SELECT id, aliases, answer FROM aftersale_knowledge "
            "WHERE shop_id = :shop AND enabled = true "
        )
        params: dict = {"shop": shop_id}
        if goods_id is not None:
            sql += "AND (goods_id = :goods OR goods_id IS NULL) "
            params["goods"] = goods_id

        result = session.execute(text(sql), params)
        rows = result.fetchall()

        query_lower = query.lower()
        for r in rows:
            aliases = (r[1] or "").lower()
            if any(alias.strip() in query_lower for alias in aliases.split(";")):
                logger.info("Aftersale Layer1 match: knowledge_id=%s", r[0])
                return str(r[2])

        return None
    finally:
        session.close()


def get_knowledge_ids_for_shop(
    shop_id: int, goods_id: int | None = None
) -> list[int]:
    """Get all enabled aftersale_knowledge IDs for a shop."""
    session = create_session()
    try:
        sql = (
            "SELECT id FROM aftersale_knowledge "
            "WHERE shop_id = :shop AND enabled = true"
        )
        params: dict = {"shop": shop_id}
        if goods_id is not None:
            sql += " AND (goods_id = :goods OR goods_id IS NULL)"
            params["goods"] = goods_id

        result = session.execute(text(sql), params)
        return [r[0] for r in result.fetchall()]
    finally:
        session.close()
