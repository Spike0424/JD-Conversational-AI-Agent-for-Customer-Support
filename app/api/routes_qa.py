"""QA knowledge expansion API routes."""

import sqlalchemy as db
from fastapi import APIRouter, Query
from pydantic import BaseModel, Field

from app.qa_generator import generate_qa_pairs
from app.qa_index import get_qa_index
from app.db import create_session as create_qa_session
from app.qa_models import QAPair

router = APIRouter(tags=["qa"])


class GenerateRequest(BaseModel):
    source_filter: str = Field(default="", description="只处理 source 包含此字符串的文档，空=全部")
    questions_per_doc: int = Field(default=5, ge=1, le=20, description="每个文档生成几个问答")


class SearchResponse(BaseModel):
    query: str
    results: list[dict]
    total_indexed: int


class StatsResponse(BaseModel):
    total_qa_pairs: int
    sources: list[dict]


@router.post("/qa/generate")
def generate(request: GenerateRequest):
    result = generate_qa_pairs(
        source_filter=request.source_filter,
        questions_per_doc=request.questions_per_doc,
    )
    get_qa_index()._reload()
    return result


@router.get("/qa/search", response_model=SearchResponse)
def search(q: str = Query(..., description="搜索查询"), top_k: int = Query(default=5, ge=1, le=20)):
    idx = get_qa_index()
    results = idx.search(q, top_k=top_k)
    return SearchResponse(query=q, results=results, total_indexed=idx.count())


@router.get("/qa/stats", response_model=StatsResponse)
def stats():
    session = create_qa_session()
    try:
        total = session.query(QAPair).count()
        rows = (
            session.query(QAPair.source_doc, db.func.count())
            .group_by(QAPair.source_doc)
            .order_by(db.func.count().desc())
            .all()
        )
        sources = [{"source_doc": r[0], "count": r[1]} for r in rows]
        return StatsResponse(total_qa_pairs=total, sources=sources)
    finally:
        session.close()
