"""Shared database engine singleton — one connection pool for all tables."""

from sqlmodel import Session, SQLModel, create_engine

from api.core.config import get_settings

_engine = None


def get_engine():
    global _engine
    if _engine is None:
        settings = get_settings()
        _engine = create_engine(settings.database_url, echo=False)

        from sqlalchemy import text

        with _engine.connect() as conn:
            conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
            conn.commit()

        # Register pgvector adapter globally so all pooled connections handle
        # Python list → vector conversion in raw text() queries with <=>
        with _engine.connect() as conn:
            raw_conn = conn.connection
            from pgvector.psycopg2.register import register_vector
            register_vector(raw_conn, globally=True)

    return _engine


def create_session() -> Session:
    return Session(get_engine())


def init_db():
    """Create all SQLModel tables."""
    SQLModel.metadata.create_all(get_engine())


def apply_migrations() -> None:
    """Idempotent ALTER TABLE for columns added after the initial schema.

    SQLModel.metadata.create_all only creates missing tables, not missing
    columns. Run hand-written ALTERs here so cold starts on an existing
    database pick up new fields without a separate migration tool.
    Called once from main.py lifespan, NOT from init_db.
    """
    from sqlalchemy import text

    with get_engine().connect() as conn:
        conn.execute(text(
            "ALTER TABLE agent_messages ADD COLUMN IF NOT EXISTS user_id VARCHAR(64)"
        ))
        conn.execute(text(
            "ALTER TABLE agent_messages ADD COLUMN IF NOT EXISTS goods_name VARCHAR(255)"
        ))
        conn.execute(text(
            "CREATE INDEX IF NOT EXISTS ix_agent_messages_user_id ON agent_messages (user_id)"
        ))
        conn.execute(text(
            "CREATE TABLE IF NOT EXISTS users ("
            "  id SERIAL PRIMARY KEY,"
            "  email VARCHAR(255) NOT NULL,"
            "  password_hash VARCHAR(128) NOT NULL,"
            "  created_at TIMESTAMP NOT NULL DEFAULT NOW()"
            ")"
        ))
        conn.execute(text(
            "CREATE UNIQUE INDEX IF NOT EXISTS ix_users_email ON users (email)"
        ))
        conn.commit()
