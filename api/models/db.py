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
