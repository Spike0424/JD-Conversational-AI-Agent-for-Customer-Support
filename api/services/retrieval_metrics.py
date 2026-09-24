from __future__ import annotations

from typing import Any

from api.models.eval_schemas import RetrievedDoc, RetrievalEvalResult


def _normalize_patterns(patterns: list[str]) -> tuple[str, ...]:
    """Normalize and deduplicate relevance labels before matching."""
    return tuple(dict.fromkeys(
        pattern.strip().lower()
        for pattern in patterns
        if pattern and pattern.strip()
    ))


def _is_relevant(source: str, patterns: list[str]) -> bool:
    normalized_patterns = _normalize_patterns(patterns)
    if not source or not normalized_patterns:
        return False
    source_lower = source.lower()
    return any(pattern in source_lower for pattern in normalized_patterns)


def normalize_docs(
    retrieved_docs: list[dict[str, Any]], k: int
) -> list[RetrievedDoc]:
    """Deduplicate retrieval rows by database primary key before Top-K slicing."""
    results: list[RetrievedDoc] = []
    seen_ids: set[int | str] = set()
    for doc in retrieved_docs:
        document_id = doc.get("id", doc.get("document_id"))
        if document_id is not None:
            if document_id in seen_ids:
                continue
            seen_ids.add(document_id)
        results.append(
            RetrievedDoc(
                document_id=document_id,
                source=str(doc.get("source", "")),
                score=float(doc.get("score", 0.0)),
                snippet=str(doc.get("snippet", "")),
                file_name=str(doc.get("file_name", "")),
                page_number=int(doc.get("page_number", 0) or 0),
                rank=len(results) + 1,
            )
        )
        if len(results) >= k:
            break
    return results


def compute_retrieval_metrics(
    query_id: str,
    retrieved_docs: list[dict[str, Any]],
    relevant_patterns: list[str],
    k: int,
    shop_id: int | None = None,
    scene: str | None = None,
    total_relevant_override: int | None = None,
) -> RetrievalEvalResult:
    normalized = normalize_docs(retrieved_docs, k)

    # Production callers pass the database truth count via the override.
    # The pattern-count fallback is retained only for pure metric unit tests.
    if total_relevant_override is None:
        total_relevant = len(relevant_patterns)
    elif total_relevant_override < 0:
        total_relevant = 0
    else:
        total_relevant = total_relevant_override

    if not normalized or total_relevant == 0:
        return RetrievalEvalResult(
            query_id=query_id,
            recall_at_k=0.0,
            hit_rate_at_k=0,
            mrr=0.0,
            precision_at_k=0.0,
            f1_score=0.0,
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
    # F1 = harmonic mean of Precision and Recall. When both are 0, F1=0.
    f1_score = (
        2 * precision_at_k * recall_at_k / (precision_at_k + recall_at_k)
        if (precision_at_k + recall_at_k) > 0
        else 0.0
    )

    return RetrievalEvalResult(
        query_id=query_id,
        recall_at_k=recall_at_k,
        hit_rate_at_k=hit_rate_at_k,
        mrr=mrr,
        precision_at_k=precision_at_k,
        f1_score=f1_score,
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
            "avg_f1_score": 0.0,
        }

    n = len(results)
    return {
        "avg_recall_at_k": sum(r.recall_at_k for r in results) / n,
        "avg_hit_rate_at_k": sum(r.hit_rate_at_k for r in results) / n,
        "avg_mrr": sum(r.mrr for r in results) / n,
        "avg_precision_at_k": sum(r.precision_at_k for r in results) / n,
        "avg_f1_score": sum(r.f1_score for r in results) / n,
    }
