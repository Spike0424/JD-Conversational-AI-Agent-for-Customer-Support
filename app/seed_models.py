"""Seed scripts table + in-memory vector index for Layer 2 semantic routing."""

from datetime import datetime, timezone
from typing import Any, Optional

import numpy as np
from pydantic import BaseModel
from sqlmodel import Field, SQLModel, UniqueConstraint

from app.db import create_session, init_db


class SeedScript(SQLModel, table=True):
    __tablename__ = "seed_scripts"
    __table_args__ = (UniqueConstraint("platform", "scenario", "trigger_text", name="uq_seed_trigger"), {"extend_existing": True})

    id: Optional[int] = Field(default=None, primary_key=True)
    platform: str = Field(default="", max_length=32, index=True)
    scenario: str = Field(default="aftersale", max_length=32, index=True)
    trigger_text: str = Field(max_length=1024)
    response_template: str = Field(max_length=4096)
    embedding_bytes: Optional[bytes] = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


# Typed route results
class RegexRouteResult(BaseModel):
    action: str
    label: str
    matched: str


class SemanticRouteResult(BaseModel):
    platform: str
    scenario: str
    trigger_text: str
    response_template: str
    score: float


# ---------------------------------------------------------------------------
# In-memory semantic index
# ---------------------------------------------------------------------------

_seed_array: np.ndarray | None = None
_seed_metas: list[dict[str, Any]] = []


def load_seed_index() -> None:
    global _seed_array, _seed_metas
    session = create_session()
    try:
        rows = session.exec(
            select(SeedScript).where(SeedScript.embedding_bytes.isnot(None))  # noqa: F821
        ).all()
        if not rows:
            _seed_array = None
            _seed_metas = []
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
                "platform": r.platform,
                "scenario": r.scenario,
                "trigger_text": r.trigger_text,
                "response_template": r.response_template,
            })
        _seed_array = np.stack(vectors) if vectors else None
        _seed_metas = metas
    finally:
        session.close()


def get_seed_count() -> int:
    return len(_seed_metas)


def get_seed_index(scenario: str | None = None) -> tuple[np.ndarray | None, list[dict[str, Any]]]:
    if not _seed_metas:
        load_seed_index()
    if scenario and _seed_array is not None and len(_seed_metas) > 0:
        indices = [i for i, m in enumerate(_seed_metas) if m["scenario"] == scenario]
        if not indices:
            return None, []
        filtered_array = _seed_array[indices]
        filtered_metas = [_seed_metas[i] for i in indices]
        return filtered_array, filtered_metas
    return _seed_array, _seed_metas


# Fix import needed by load_seed_index
from sqlmodel import select  # noqa: E402
