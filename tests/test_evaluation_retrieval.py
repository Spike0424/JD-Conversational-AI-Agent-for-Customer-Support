from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

from api.services.retrieval_metrics import (
    _is_relevant,
    aggregate_retrieval,
    compute_retrieval_metrics,
    normalize_docs,
)
from api.models.eval_schemas import RetrievalEvalResult
from tests.conftest import run_async


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

    def test_deduplicates_same_database_id_before_top_k(self):
        docs = [
            make_doc("duplicate", score=0.9) | {"id": 42},
            make_doc("duplicate", score=0.8) | {"id": 42},
            make_doc("other", score=0.7) | {"id": 43},
        ]
        result = normalize_docs(docs, k=2)
        assert [doc.document_id for doc in result] == [42, 43]
        assert [doc.rank for doc in result] == [1, 2]

    def test_keeps_rows_without_id(self):
        docs = [make_doc("a"), make_doc("a")]
        result = normalize_docs(docs, k=2)
        assert len(result) == 2

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
                f1_score=2 * 0.5 * 1.0 / 1.5,
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
                f1_score=0.0,
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
        assert agg["avg_f1_score"] == (2 * 0.5 * 1.0 / 1.5) / 2

    def test_empty_results(self):
        agg = aggregate_retrieval([])
        assert agg["avg_recall_at_k"] == 0.0
        assert agg["avg_hit_rate_at_k"] == 0.0
        assert agg["avg_mrr"] == 0.0
        assert agg["avg_precision_at_k"] == 0.0
        assert agg["avg_f1_score"] == 0.0


class TestTotalRelevantFromDB:
    """total_relevant is fully driven by the DB lookup helper. When the helper
    result is passed via total_relevant_override, no cap is applied (the value
    is the actual count of rows that match the patterns in the KB). When the
    helper is not called (None), it falls back to len(patterns). When the
    helper returns -1 (DB error), total_relevant is forced to 0 and Recall=0."""

    def test_override_used_when_positive(self):
        # 2 of 5 DB-ground-truth relevant docs retrieved → recall = 2/5 = 0.4
        docs = [make_doc("target_a"), make_doc("target_b")]
        result = compute_retrieval_metrics(
            "q_db", docs, ["target"], k=2,
            total_relevant_override=5,
        )
        assert result.total_relevant == 5
        assert result.recall_at_k == 0.4

    def test_override_none_falls_back_to_patterns_length(self):
        # No helper called → use len(patterns) as denominator
        result = compute_retrieval_metrics(
            "q_fb", [make_doc("target")], ["a", "b", "c"], k=2,
        )
        assert result.total_relevant == 3

    def test_override_zero_means_zero_ground_truth(self):
        # DB says no relevant docs → recall = 0 even though we retrieved 2
        docs = [make_doc("target_a"), make_doc("target_b")]
        result = compute_retrieval_metrics(
            "q_zero", docs, ["target"], k=2,
            total_relevant_override=0,
        )
        assert result.total_relevant == 0
        assert result.recall_at_k == 0.0
        assert result.hit_rate_at_k == 0  # hit_rate requires total_relevant > 0

    def test_override_minus_one_db_error(self):
        # DB failure signal → total_relevant=0, Recall=0
        docs = [make_doc("target_a"), make_doc("target_b")]
        result = compute_retrieval_metrics(
            "q_db_err", docs, ["target"], k=2,
            total_relevant_override=-1,
        )
        assert result.total_relevant == 0
        assert result.recall_at_k == 0.0

    def test_override_no_cap(self):
        # DB returns 13 (e.g. patterns=["退货"] matches 13 rows) → no cap to 2.
        # Recall = 2/13 ≈ 0.154, NOT >1.0 nor artificially clamped.
        docs = [make_doc("退货"), make_doc("退货again")]
        result = compute_retrieval_metrics(
            "q_nocap", docs, ["退货"], k=2,
            total_relevant_override=13,
        )
        assert result.total_relevant == 13
        assert abs(result.recall_at_k - 2/13) < 1e-6

    def test_total_relevant_is_database_denominator(self):
        # The single total_relevant field now represents the DB truth count.
        from api.models.eval_schemas import RetrievalEvalResult
        r = RetrievalEvalResult(
            query_id="q", recall_at_k=0.0, hit_rate_at_k=0,
            mrr=0.0, precision_at_k=0.0, f1_score=0.0,
            retrieved_count=0, relevant_retrieved_count=0,
            total_relevant=5,
        )
        assert r.total_relevant == 5


class TestF1Score:
    """F1 = 2 · P · R / (P + R). F1=0 when both P and R are 0."""

    def test_perfect_score(self):
        # P=1.0, R=1.0 → F1=1.0
        docs = [make_doc("target_a"), make_doc("target_b")]
        result = compute_retrieval_metrics(
            "q1", docs, ["target_a", "target_b"], k=2,
        )
        assert result.precision_at_k == 1.0
        assert result.recall_at_k == 1.0
        assert result.f1_score == 1.0

    def test_imbalanced_p_half_r_full(self):
        # P=0.5, R=1.0 → F1 = 2*0.5*1/1.5 ≈ 0.667
        docs = [make_doc("target"), make_doc("irrelevant")]
        result = compute_retrieval_metrics(
            "q2", docs, ["target"], k=2,
        )
        assert result.precision_at_k == 0.5
        assert result.recall_at_k == 1.0  # capped total_relevant=1
        assert abs(result.f1_score - (2 * 0.5 * 1.0 / 1.5)) < 1e-6

    def test_imbalanced_p_full_r_half(self):
        # P=1.0, R=0.5 → F1 = 2*1.0*0.5/1.5 ≈ 0.667
        docs = [make_doc("target")]
        result = compute_retrieval_metrics(
            "q3", docs, ["target", "other"], k=1,
        )
        assert result.precision_at_k == 1.0
        assert result.recall_at_k == 0.5  # 1/2, capped at 2
        assert abs(result.f1_score - (2 * 1.0 * 0.5 / 1.5)) < 1e-6

    def test_zero_when_no_relevant(self):
        # Both P and R are 0 → F1=0 (no division by zero)
        result = compute_retrieval_metrics(
            "q_zero", [make_doc("x"), make_doc("y")], ["target"], k=2,
        )
        assert result.precision_at_k == 0.0
        assert result.recall_at_k == 0.0
        assert result.f1_score == 0.0

    def test_f1_in_result_schema(self):
        # Schema field exists with default 0.0
        result = compute_retrieval_metrics("q", [], [], k=1)
        assert hasattr(result, "f1_score")
        assert result.f1_score == 0.0


class TestDatabaseGroundTruthMatching:
    """Database-grounded relevance uses knowledge rows, not pattern count."""

    def test_empty_patterns_never_match(self):
        assert _is_relevant("anything", ["", "  "]) is False

    def test_duplicate_database_id_counts_once_for_metrics(self):
        docs = [
            make_doc("退货", score=0.9) | {"id": 7},
            make_doc("退货", score=0.8) | {"id": 7},
            make_doc("退款", score=0.7) | {"id": 8},
        ]
        result = compute_retrieval_metrics(
            "q_dedup", docs, ["退货", "退款"], k=3,
            total_relevant_override=2,
        )
        assert result.retrieved_count == 2
        assert result.relevant_retrieved_count == 2
        assert result.recall_at_k == 1.0

    def test_count_relevant_documents_deduplicates_patterns_per_row(self, monkeypatch):
        from api.services.knowledge import SearchKnowledge

        search = SearchKnowledge()
        search.set_context(shop_id=7, scene="aftersale")
        candidates = [
            {"id": 1, "aliases": "退货;运费险"},
            {"id": 2, "aliases": "退货"},
            {"id": 3, "aliases": "退款"},
        ]
        monkeypatch.setattr(search, "_fetch_candidates", lambda **_: candidates)

        assert search.count_relevant_documents(["退货", "退货", " 运费险 "]) == 2

    def test_count_relevant_documents_deduplicates_duplicate_database_rows(self, monkeypatch):
        from api.services.knowledge import SearchKnowledge

        search = SearchKnowledge()
        search.set_context(shop_id=7, scene="aftersale")
        candidates = [
            {"id": 1, "aliases": "退货"},
            {"id": 1, "aliases": "退货"},
            {"id": 2, "aliases": "退款"},
        ]
        monkeypatch.setattr(search, "_fetch_candidates", lambda **_: candidates)

        assert search.count_relevant_documents(["退货", "退款"]) == 2

    def test_search_structured_deduplicates_duplicate_database_rows(self, monkeypatch):
        from api.services.knowledge import SearchKnowledge

        search = SearchKnowledge()
        search.set_context(shop_id=7, scene="aftersale")
        monkeypatch.setattr(search, "_fetch_candidates", lambda **_: [{"id": 1}])
        monkeypatch.setattr(
            search,
            "_rank",
            lambda *_: [
                {"id": 7, "aliases": "duplicate", "answer": "first", "final_score": 80.0},
                {"id": 7, "aliases": "duplicate", "answer": "second", "final_score": 70.0},
            ],
        )

        result = search.search_structured("query")

        assert len(result) == 1
        assert result[0]["id"] == 7

    def test_search_structured_applies_final_score_threshold(self, monkeypatch):
        from api.services.knowledge import SearchKnowledge

        search = SearchKnowledge()
        search.set_context(shop_id=7, scene="aftersale")
        monkeypatch.setattr(search, "_fetch_candidates", lambda **_: [{"id": 1}])
        monkeypatch.setattr(
            search,
            "_rank",
            lambda *_: [
                {"id": 1, "aliases": "weak", "answer": "weak", "final_score": 49.99},
                {"id": 2, "aliases": "boundary", "answer": "boundary", "final_score": 50.0},
            ],
        )

        result = search.search_structured("query")

        assert [doc["source"] for doc in result] == ["boundary"]


class TestRawQueryRecording:
    """The report records the RAW user question verbatim; tokenization detail
    lives in search_terms — no separate cleaned field."""

    def test_eval_runner_records_raw_question(self, monkeypatch):
        from api.models.eval_schemas import EvalQuery
        from api.services import eval_runner as eval_runner_module
        from api.services.eval_runner import EvalRunner

        sk = MagicMock()
        sk.count_relevant_documents.return_value = 3
        sk.search_structured.return_value = [
            {"id": 1, "source": "退货", "score": 100.0, "snippet": "s"}
        ]
        sk.last_search_terms = ["退货", "政策"]
        monkeypatch.setattr(eval_runner_module, "get_search_knowledge", lambda: sk)

        runner = EvalRunner()
        runner._scene_classifier = MagicMock()
        runner._scene_classifier.classify = AsyncMock(return_value="aftersale")
        runner._agent = MagicMock()
        runner._agent.ask = AsyncMock(return_value="答案")

        query = EvalQuery(query_id="q1", question="怎么退货",
                          relevant_source_patterns=["退货"])
        report = run_async(runner.run([query], dataset_name="t", k=2,
                                      run_generation_eval=False, shop_id=1))

        d = report.retrieval_details[0]
        assert d.query == "怎么退货"          # verbatim raw question
        assert d.search_terms == ["退货", "政策"]
