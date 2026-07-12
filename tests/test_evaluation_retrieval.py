from __future__ import annotations

from app.evaluation.retrieval_metrics import (
    _is_relevant,
    aggregate_retrieval,
    compute_retrieval_metrics,
    normalize_docs,
)
from app.evaluation.schemas import RetrievalEvalResult


def make_doc(source: str, score: float = 0.1, snippet: str = "") -> dict:
    return {"source": source, "score": score, "snippet": snippet, "file_name": "", "page_number": 0}


class TestIsRelevant:
    def test_exact_match(self):
        assert _is_relevant("退货协议", ["退货协议"]) is True

    def test_case_insensitive(self):
        assert _is_relevant("Return Policy", ["return policy"]) is True
        assert _is_relevant("return policy", ["Return Policy"]) is True

    def test_substring_match(self):
        assert _is_relevant("product_manual_v2.pdf", ["product_manual"]) is True

    def test_no_match(self):
        assert _is_relevant("irrelevant_doc", ["target"]) is False

    def test_empty_patterns(self):
        assert _is_relevant("anything", []) is False

    def test_one_of_many_patterns(self):
        assert _is_relevant("doc_c", ["doc_a", "doc_b", "doc_c"]) is True

    def test_empty_source(self):
        assert _is_relevant("", ["target"]) is False


class TestNormalizeDocs:
    def test_truncates_to_k(self):
        docs = [make_doc(f"src_{i}") for i in range(10)]
        result = normalize_docs(docs, k=3)
        assert len(result) == 3
        assert result[0].rank == 1
        assert result[2].rank == 3

    def test_fewer_than_k(self):
        docs = [make_doc("only_one")]
        result = normalize_docs(docs, k=5)
        assert len(result) == 1

    def test_empty_list(self):
        result = normalize_docs([], k=5)
        assert result == []

    def test_rank_is_1_based(self):
        docs = [make_doc("a"), make_doc("b")]
        result = normalize_docs(docs, k=5)
        assert result[0].rank == 1
        assert result[1].rank == 2


class TestComputeRetrievalMetrics:
    def test_perfect_retrieval(self):
        docs = [
            make_doc("target_doc", score=0.1),
            make_doc("other_doc", score=0.2),
        ]
        result = compute_retrieval_metrics("q1", docs, ["target_doc"], k=3)
        assert result.query_id == "q1"
        assert result.recall_at_k == 1.0
        assert result.hit_rate_at_k == 1
        assert result.mrr == 1.0
        assert result.precision_at_k == 1 / 3
        assert result.retrieved_count == 2
        assert result.relevant_retrieved_count == 1
        assert result.total_relevant == 1

    def test_perfect_retrieval_all_relevant(self):
        docs = [
            make_doc("target_a", score=0.1),
            make_doc("target_b", score=0.2),
        ]
        result = compute_retrieval_metrics("q1", docs, ["target_a", "target_b"], k=2)
        assert result.recall_at_k == 1.0
        assert result.hit_rate_at_k == 1
        assert result.precision_at_k == 1.0

    def test_partial_retrieval(self):
        docs = [
            make_doc("target_a", score=0.1),
            make_doc("irrelevant", score=0.2),
        ]
        result = compute_retrieval_metrics("q2", docs, ["target_a", "target_b"], k=2)
        assert result.recall_at_k == 1 / 2
        assert result.hit_rate_at_k == 1
        assert result.precision_at_k == 1 / 2

    def test_no_relevant_found(self):
        docs = [
            make_doc("irrelevant_a", score=0.1),
            make_doc("irrelevant_b", score=0.2),
        ]
        result = compute_retrieval_metrics("q3", docs, ["target"], k=3)
        assert result.recall_at_k == 0.0
        assert result.hit_rate_at_k == 0
        assert result.mrr == 0.0
        assert result.precision_at_k == 0.0
        assert result.relevant_retrieved_count == 0

    def test_empty_retrieved(self):
        result = compute_retrieval_metrics("q4", [], ["target"], k=3)
        assert result.recall_at_k == 0.0
        assert result.hit_rate_at_k == 0
        assert result.mrr == 0.0
        assert result.precision_at_k == 0.0
        assert result.retrieved_count == 0

    def test_empty_relevant_patterns(self):
        docs = [make_doc("any_doc")]
        result = compute_retrieval_metrics("q5", docs, [], k=3)
        assert result.recall_at_k == 0.0
        assert result.total_relevant == 0

    def test_mrr_second_rank(self):
        docs = [
            make_doc("irrelevant", score=0.1),
            make_doc("target", score=0.2),
            make_doc("other", score=0.3),
        ]
        result = compute_retrieval_metrics("q6", docs, ["target"], k=3)
        assert result.hit_rate_at_k == 1
        assert result.mrr == 1.0 / 2

    def test_mrr_third_rank(self):
        docs = [
            make_doc("irrelevant_a", score=0.1),
            make_doc("irrelevant_b", score=0.2),
            make_doc("target", score=0.3),
        ]
        result = compute_retrieval_metrics("q7", docs, ["target"], k=3)
        assert result.mrr == 1.0 / 3

    def test_truncation_respects_k(self):
        docs = [
            make_doc("irrelevant_a"),
            make_doc("irrelevant_b"),
            make_doc("irrelevant_c"),
            make_doc("target"),
        ]
        result = compute_retrieval_metrics("q8", docs, ["target"], k=3)
        assert result.hit_rate_at_k == 0
        assert result.mrr == 0.0

    def test_substring_matching_in_metrics(self):
        docs = [make_doc("product_manual_chapter_3")]
        result = compute_retrieval_metrics("q9", docs, ["product_manual"], k=3)
        assert result.hit_rate_at_k == 1


class TestAggregateRetrieval:
    def test_aggregates_multiple_results(self):
        results = [
            RetrievalEvalResult(
                query_id="q1",
                recall_at_k=1.0,
                hit_rate_at_k=1,
                mrr=1.0,
                precision_at_k=0.5,
                retrieved_count=2,
                relevant_retrieved_count=1,
                total_relevant=1,
            ),
            RetrievalEvalResult(
                query_id="q2",
                recall_at_k=0.0,
                hit_rate_at_k=0,
                mrr=0.0,
                precision_at_k=0.0,
                retrieved_count=0,
                relevant_retrieved_count=0,
                total_relevant=2,
            ),
        ]
        agg = aggregate_retrieval(results)
        assert agg["avg_recall_at_k"] == 0.5
        assert agg["avg_hit_rate_at_k"] == 0.5
        assert agg["avg_mrr"] == 0.5
        assert agg["avg_precision_at_k"] == 0.25

    def test_empty_results(self):
        agg = aggregate_retrieval([])
        assert agg["avg_recall_at_k"] == 0.0
        assert agg["avg_hit_rate_at_k"] == 0.0
        assert agg["avg_mrr"] == 0.0
        assert agg["avg_precision_at_k"] == 0.0
