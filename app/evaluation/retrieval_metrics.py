from __future__ import annotations

from typing import Any

from app.evaluation.schemas import RetrievedDoc, RetrievalEvalResult


def _is_relevant(source: str, patterns: list[str]) -> bool:
    if not patterns:
        return False
    source_lower = source.lower()
    return any(p.lower() in source_lower for p in patterns)


def normalize_docs(
    retrieved_docs: list[dict[str, Any]], k: int
) -> list[RetrievedDoc]:
    results: list[RetrievedDoc] = []
    for rank, doc in enumerate(retrieved_docs[:k], start=1):
        results.append(
            RetrievedDoc(
                source=str(doc.get("source", "")),
                score=float(doc.get("score", 0.0)),
                snippet=str(doc.get("snippet", "")),
                file_name=str(doc.get("file_name", "")),
                page_number=int(doc.get("page_number", 0) or 0),
                rank=rank,
            )
        )
    return results


def compute_retrieval_metrics(
    query_id: str,
    retrieved_docs: list[dict[str, Any]],
    relevant_patterns: list[str],
    k: int,
) -> RetrievalEvalResult:
    normalized = normalize_docs(retrieved_docs, k)
    total_relevant = len(relevant_patterns)

    if not normalized or total_relevant == 0:
        return RetrievalEvalResult(
            query_id=query_id,
            recall_at_k=0.0,
            hit_rate_at_k=0,
            mrr=0.0,
            precision_at_k=0.0,
            retrieved_count=len(normalized),
            relevant_retrieved_count=0,
            total_relevant=total_relevant,
        )

    relevant_retrieved = 0
    first_relevant_rank: int | None = None

    for doc in normalized:
        if _is_relevant(doc.source, relevant_patterns):
            relevant_retrieved += 1
            if first_relevant_rank is None:
                first_relevant_rank = doc.rank

    recall_at_k = relevant_retrieved / max(total_relevant, 1)
    hit_rate_at_k = 1 if relevant_retrieved > 0 else 0
    mrr = (1.0 / first_relevant_rank) if first_relevant_rank is not None else 0.0
    precision_at_k = relevant_retrieved / k

    return RetrievalEvalResult(
        query_id=query_id,
        recall_at_k=recall_at_k,
        hit_rate_at_k=hit_rate_at_k,
        mrr=mrr,
        precision_at_k=precision_at_k,
        retrieved_count=len(normalized),
        relevant_retrieved_count=relevant_retrieved,
        total_relevant=total_relevant,
    )


def aggregate_retrieval(results: list[RetrievalEvalResult]) -> dict[str, float]:
    if not results:
        return {
            "avg_recall_at_k": 0.0,
            "avg_hit_rate_at_k": 0.0,
            "avg_mrr": 0.0,
            "avg_precision_at_k": 0.0,
        }

    n = len(results)
    return {
        "avg_recall_at_k": sum(r.recall_at_k for r in results) / n,
        "avg_hit_rate_at_k": sum(r.hit_rate_at_k for r in results) / n,
        "avg_mrr": sum(r.mrr for r in results) / n,
        "avg_precision_at_k": sum(r.precision_at_k for r in results) / n,
    }
