"""Filtered vector retrieval from scene_knowledge_embeddings via pgvector HNSW.

Uses PostgreSQL pgvector extension with HNSW index for cosine-distance search.
No in-memory vector loading — all search happens in the database.
"""

import hashlib
import logging
from datetime import datetime, timezone
from typing import Any

from sqlalchemy import text

from app.config import get_settings
from app.db import create_session
from app.retrieval.embedding import get_embeddings

logger = logging.getLogger(__name__)


class ShopSceneRetriever:
    """pgvector cosine-distance search over scene_knowledge_embeddings."""

    def __init__(
        self,
        shop_id: int,
        scene: str,
        goods_id: int | None = None,
        top_k: int = 4,
    ) -> None:
        self._shop_id = shop_id
        self._scene = scene
        self._goods_id = goods_id
        self._top_k = top_k

    def search(self, query: str) -> list[dict[str, Any]]:
        emb = get_embeddings()
        query_vec = emb.embed_query(query)

        session = create_session()
        try:
            session.execute(text(f"SET hnsw.ef_search = {self._top_k * 4}"))
            sql = (
                "SELECT id, knowledge_table, knowledge_id, shop_id, goods_id, "
                "       embedding_text, embedding_model, scene, "
                "       embedding <=> :vec AS distance "
                "FROM scene_knowledge_embeddings "
                "WHERE shop_id = :shop_id AND scene = :scene "
            )
            params: dict = {
                "shop_id": self._shop_id,
                "scene": self._scene,
                "vec": query_vec,
            }
            if self._goods_id is not None:
                sql += "AND (goods_id = :goods_id OR goods_id IS NULL) "
                params["goods_id"] = self._goods_id

            sql += "ORDER BY embedding <=> :vec LIMIT :top_k"
            params["top_k"] = self._top_k

            result = session.execute(text(sql), params)
            rows = result.fetchall()
        finally:
            session.close()

        results = []
        for r in rows:
            results.append({
                "source": f"{r[1]}#{r[2]}",
                "score": float(r[8]),
                "snippet": (r[5] or "")[:200],
                "shop_id": r[3],
                "goods_id": r[4],
                "scene": r[7],
                "knowledge_table": r[1],
                "knowledge_id": r[2],
            })
        return results

    def search_formatted(self, query: str) -> str:
        results = self.search(query)
        if not results:
            return "No matching knowledge found for this shop and scene."

        lines = []
        for i, r in enumerate(results, start=1):
            goods = f" goods={r['goods_id']}" if r.get("goods_id") else ""
            lines.append(
                f"[{i}] source={r['source']}{goods} score={r['score']:.4f}\n{r['snippet']}"
            )
        return "\n\n".join(lines)


# ── Sync: knowledge tables → scene_knowledge_embeddings ────────────


def sync_knowledge_to_embeddings(shop_id: int | None = None) -> dict[str, int]:
    """Read presale/insale/aftersale_knowledge rows and embed into scene_knowledge_embeddings.

    Args:
        shop_id: If given, only sync this shop.  If None, sync all.

    Returns: {"inserted": N, "skipped": N, "errors": N}
    """
    settings = get_settings()
    embeddings_instance = get_embeddings()
    session = create_session()
    now = datetime.now(timezone.utc)

    stats = {"inserted": 0, "skipped": 0, "errors": 0}

    try:
        for table_name, scene in [
            ("presale_knowledge", "presale"),
            ("insale_knowledge", "insale"),
            ("aftersale_knowledge", "aftersale"),
        ]:
            sql = (
                f"SELECT id, shop_id, goods_id, aliases, answer FROM {table_name} "
                "WHERE enabled = true"
            )
            params: dict = {}
            if shop_id is not None:
                sql += " AND shop_id = :shop_id"
                params["shop_id"] = shop_id

            result = session.execute(text(sql), params)
            rows = result.fetchall()

            if not rows:
                continue

            existing_hashes: set[tuple[str, int, str]] = set()
            existing = session.execute(
                text(
                    "SELECT knowledge_table, knowledge_id, content_hash "
                    "FROM scene_knowledge_embeddings "
                    "WHERE scene = :scene AND knowledge_table = :table"
                ),
                {"scene": scene, "table": table_name},
            ).fetchall()
            existing_hashes = {(r[0], r[1], r[2]) for r in existing}

            # Build candidate list, filtering out already-synced rows
            pending: list[dict] = []
            for r in rows:
                knowledge_id, row_shop_id = r[0], r[1]
                row_goods_id = r[2]
                aliases = r[3] or ""
                answer = r[4] or ""
                embedding_text = f"{aliases}\n{answer}"[:4096]
                content_hash = hashlib.sha256(embedding_text.encode()).hexdigest()

                if (table_name, knowledge_id, content_hash) in existing_hashes:
                    stats["skipped"] += 1
                    continue

                pending.append({
                    "scene": scene,
                    "table": table_name,
                    "kid": knowledge_id,
                    "shop": row_shop_id,
                    "goods": row_goods_id,
                    "etext": embedding_text,
                    "hash": content_hash,
                })

            if not pending:
                continue

            # Batch embed all pending texts
            texts = [p["etext"] for p in pending]
            try:
                vecs = embeddings_instance.embed_documents(texts)
            except Exception:
                logger.exception("Batch embedding failed for %s", table_name)
                stats["errors"] += len(pending)
                continue

            # Build bulk INSERT
            bulk_values: list[dict] = []
            for p, vec in zip(pending, vecs):
                bulk_values.append({
                    **p,
                    "emb": vec,
                    "model": settings.local_embedding_model_name,
                    "dim": len(vec),
                })
                stats["inserted"] += 1

            placeholders = ", ".join([
                f"(:scene_{i}, :table_{i}, :kid_{i}, :shop_{i}, :goods_{i}, "
                f":etext_{i}, :emb_{i}, :model_{i}, :dim_{i}, :hash_{i}, "
                f":created_{i}, :updated_{i})"
                for i in range(len(bulk_values))
            ])
            flat_params: dict = {}
            for i, v in enumerate(bulk_values):
                flat_params.update({
                    f"scene_{i}": v["scene"], f"table_{i}": v["table"],
                    f"kid_{i}": v["kid"], f"shop_{i}": v["shop"],
                    f"goods_{i}": v["goods"], f"etext_{i}": v["etext"],
                    f"emb_{i}": v["emb"], f"model_{i}": v["model"],
                    f"dim_{i}": v["dim"], f"hash_{i}": v["hash"],
                    f"created_{i}": now, f"updated_{i}": now,
                })
            session.execute(
                text(
                    "INSERT INTO scene_knowledge_embeddings "
                    "(scene, knowledge_table, knowledge_id, shop_id, goods_id, "
                    "embedding_text, embedding, embedding_model, embedding_dim, content_hash, "
                    "created_at, updated_at) "
                    f"VALUES {placeholders}"
                ),
                flat_params,
            )

        session.commit()
    finally:
        session.close()

    logger.info("sync_knowledge_to_embeddings done: %s", stats)
    return stats
