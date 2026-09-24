"""One-time migration: bytea → pgvector + HNSW indexes.

Usage:
    uv run python -m script.migrate_pgvector
"""

import logging
import sys

from api.models.db import get_engine

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")


def migrate() -> None:
    engine = get_engine()

    with engine.connect() as conn:
        # ── 1. Ensure pgvector extension ──
        from sqlalchemy import text

        conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
        conn.commit()
        logger.info("pgvector extension ensured")

        # ── 2. Clear legacy rows: bytea payloads are unreadable by pgvector
        #      and would violate NOT NULL during the type change. ──
        for table in ("scene_knowledge_embeddings", "aftersale_chunks"):
            res = conn.execute(text(f"DELETE FROM {table}"))
            logger.info("Cleared %d legacy rows from %s", res.rowcount, table)
        conn.commit()

        # ── 3. Alter column types ──
        conn.execute(
            text(
                "ALTER TABLE scene_knowledge_embeddings "
                "ALTER COLUMN embedding TYPE vector(512) USING NULL"
            )
        )
        conn.execute(
            text(
                "ALTER TABLE aftersale_chunks "
                "ALTER COLUMN embedding TYPE vector(512) USING NULL"
            )
        )
        conn.commit()
        logger.info("Column types changed to vector(512) (bge-small-zh-v1.5)")

        # ── 3. Drop old indexes ──
        for idx_name, _table in [
            ("ix_ske_embedding_hnsw", "scene_knowledge_embeddings"),
            ("ix_ac_embedding_hnsw", "aftersale_chunks"),
        ]:
            conn.execute(text(f"DROP INDEX IF EXISTS {idx_name}"))
        conn.commit()

        # ── 5. Create HNSW indexes ──
        conn.execute(
            text(
                "CREATE INDEX IF NOT EXISTS ix_ske_embedding_hnsw "
                "ON scene_knowledge_embeddings "
                "USING hnsw (embedding vector_cosine_ops) "
                "WITH (m = 16, ef_construction = 64)"
            )
        )
        conn.execute(
            text(
                "CREATE INDEX IF NOT EXISTS ix_ac_embedding_hnsw "
                "ON aftersale_chunks "
                "USING hnsw (embedding vector_cosine_ops) "
                "WITH (m = 16, ef_construction = 64)"
            )
        )
        conn.commit()
        logger.info("HNSW indexes created")

    logger.info("Migration complete. Run util/import_docs.py to (re)build knowledge embeddings.")


if __name__ == "__main__":
    migrate()
