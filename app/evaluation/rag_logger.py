"""RAG-dedicated structured logging: trace-level JSON records per request."""

import json
import logging
from typing import Any

logger = logging.getLogger("rag")

_RAG_PREFIX = "[RAG]"


def _safe_preview(text: str | None, max_len: int = 200) -> str:
    if not text:
        return ""
    return text.replace("\n", " ")[:max_len]


class RAGLogger:
    """Structured RAG pipeline logging, keyed by request_id."""

    @staticmethod
    def log_trace(request_id: str, **fields: Any) -> None:
        """Write a single structured JSON log line for this request phase.

        Expected fields (all optional — only populated ones are written):
            request_id, phase, user_query, scene, shop_id, goods_id,
            filters, retrieved_chunks, top_k, rerank_result,
            llm_answer, gold_answer, retrieval_hit, answer_correct,
            elapsed_ms, error
        """
        record: dict[str, Any] = {"request_id": request_id}
        for key in (
            "phase", "user_query", "scene", "shop_id", "goods_id",
            "filters", "top_k", "retrieval_hit", "answer_correct",
            "elapsed_ms", "error",
        ):
            if key in fields:
                record[key] = fields[key]

        # Truncate text fields for log readability
        if "retrieved_chunks" in fields:
            record["retrieved_chunks"] = [
                {
                    "source": c.get("source", ""),
                    "score": round(c.get("score", 0), 4),
                    "snippet": _safe_preview(c.get("snippet", ""), 120),
                }
                for c in fields["retrieved_chunks"][:10]
            ]

        if "rerank_result" in fields:
            record["rerank_result"] = [
                {
                    "rank": r.get("rank", i + 1),
                    "source": r.get("source", ""),
                    "match_type": r.get("match_type", ""),
                    "final_score": r.get("final_score", 0),
                    "alias_score": r.get("alias_score", 0),
                    "keyword_score": r.get("keyword_score", 0),
                    "vector_similarity": r.get("vector_similarity", 0),
                }
                for i, r in enumerate(fields["rerank_result"][:10])
            ]

        if "llm_answer" in fields:
            record["llm_answer"] = _safe_preview(fields["llm_answer"], 500)
        if "gold_answer" in fields:
            record["gold_answer"] = _safe_preview(fields["gold_answer"], 500)

        logger.info("%s %s", _RAG_PREFIX, json.dumps(record, ensure_ascii=False))

    # ── legacy phase methods (kept for backward compatibility) ──────

    @staticmethod
    def log_query(trace_id: str, question: str) -> None:
        RAGLogger.log_trace(
            trace_id, phase="query", user_query=_safe_preview(question, 100),
        )

    @staticmethod
    def log_embedding(trace_id: str, model_name: str, dim: int | None, query_preview: str) -> None:
        logger.info(
            "%s trace=%s phase=embed model=%s dim=%s query_preview=%s",
            _RAG_PREFIX, trace_id, model_name, dim or "?", query_preview[:50],
        )

    @staticmethod
    def log_retrieval(
        trace_id: str,
        retrieved_count: int,
        elapsed_ms: float,
        chunks: list[dict[str, Any]],
    ) -> None:
        logger.info(
            "%s trace=%s phase=retrieval count=%d elapsed_ms=%.1f",
            _RAG_PREFIX, trace_id, retrieved_count, elapsed_ms,
        )
        for i, c in enumerate(chunks, start=1):
            snippet = str(c.get("snippet", ""))[:80].replace("\n", " ")
            logger.info(
                "%s trace=%s phase=retrieval chunk=%d source=%s file=%s page=%s score=%.4f snippet=%s",
                _RAG_PREFIX, trace_id, i,
                c.get("source", "?"), c.get("file_name", ""), c.get("page_number", 0),
                c.get("score", 0.0), snippet,
            )

    @staticmethod
    def log_rerank(trace_id: str, ranked: list[dict[str, Any]]) -> None:
        logger.info("%s trace=%s phase=rerank count=%d", _RAG_PREFIX, trace_id, len(ranked))
        for i, r in enumerate(ranked, start=1):
            logger.info(
                "%s trace=%s phase=rerank rank=%d source=%s score=%.4f",
                _RAG_PREFIX, trace_id, i, r.get("source", "?"), r.get("score", 0.0),
            )

    @staticmethod
    def log_generation(trace_id: str, answer: str, elapsed_ms: float) -> None:
        RAGLogger.log_trace(
            trace_id, phase="generation",
            llm_answer=answer, elapsed_ms=elapsed_ms,
        )

    @staticmethod
    def log_eval(trace_id: str, metrics: dict[str, Any]) -> None:
        parts = []
        for k, v in metrics.items():
            if v is None:
                parts.append(f"{k}=N/A")
            elif isinstance(v, float):
                parts.append(f"{k}={v:.1f}")
            else:
                parts.append(f"{k}={v}")
        logger.info(
            "%s trace=%s phase=eval %s",
            _RAG_PREFIX, trace_id, " ".join(parts),
        )
