"""Q&A pair storage — uses shared db.py engine."""

from datetime import datetime, timezone
from typing import Optional

from sqlmodel import Field, SQLModel

from app.db import create_session, init_db


class QAPair(SQLModel, table=True):
    __tablename__ = "qa_pairs"
    __table_args__ = {"extend_existing": True}

    id: Optional[int] = Field(default=None, primary_key=True)
    question: str = Field(max_length=2048)
    answer: str = Field(max_length=8192)
    source: str = Field(default="", max_length=4096)
    source_doc: str = Field(default="", max_length=512, index=True)
    category: str = Field(default="after_sale", max_length=64)
    embedding_bytes: Optional[bytes] = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
