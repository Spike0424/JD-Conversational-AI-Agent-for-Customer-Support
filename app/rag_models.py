"""RAG chunk storage — uses shared db.py engine."""

from datetime import datetime, timezone
from typing import Optional

from sqlmodel import Field, SQLModel

from app.db import create_session, init_db


class RagChunk(SQLModel, table=True):
    __tablename__ = "rag_chunks"
    __table_args__ = {"extend_existing": True}

    id: Optional[int] = Field(default=None, primary_key=True)
    source: str = Field(index=True, max_length=512)
    snippet: str = Field(max_length=4096)
    file_name: str = Field(default="", max_length=512)
    page_number: int = Field(default=0)
    chunk_index: int = Field(default=0)
    embedding_bytes: Optional[bytes] = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
