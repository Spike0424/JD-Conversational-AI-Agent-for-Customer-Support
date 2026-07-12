from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from app.evaluation.generation_metrics import (
    _clamp_score,
    _parse_judge_response,
    aggregate_generation,
    build_judge_llm,
    evaluate_all_generation_metrics,
    evaluate_answer_relevance,
    evaluate_context_usage,
    evaluate_faithfulness,
    evaluate_noise_sensitivity,
)
from app.evaluation.retrieval_metrics import normalize_docs
from app.evaluation.schemas import GenerationEvalResult, RetrievedDoc


def _make_retrieved_docs(sources: list[str]) -> list[RetrievedDoc]:
    return normalize_docs(
        [{"source": s, "score": 0.1, "snippet": f"text of {s}", "file_name": "", "page_number": 0}
         for s in sources],
        k=10,
    )


class TestClampScore:
    def test_within_range(self):
        assert _clamp_score(3) == 3
        assert _clamp_score(1) == 1
        assert _clamp_score(5) == 5

    def test_below_range(self):
        assert _clamp_score(0) == 1
        assert _clamp_score(-5) == 1

    def test_above_range(self):
        assert _clamp_score(6) == 5
        assert _clamp_score(10) == 5


class TestParseJudgeResponse:
    def test_valid_json(self):
        raw = '{"score": 4, "justification": "Mostly faithful with minor issues."}'
        score, justification = _parse_judge_response(raw)
        assert score == 4
        assert "Mostly faithful" in justification

    def test_json_with_extra_text(self):
        raw = 'Here is my evaluation: {"score": 5, "justification": "Perfect."} Thanks!'
        score, justification = _parse_judge_response(raw)
        assert score == 5
        assert "Perfect" in justification

    def test_score_only_json(self):
        raw = '{"score": 2}'
        score, justification = _parse_judge_response(raw)
        assert score == 2
        assert justification == ""

    def test_regex_fallback(self):
        raw = 'The score is 3. Justification: somewhat relevant.'
        score, justification = _parse_judge_response(raw)
        assert score == 3
        assert "The score is 3" in justification

    def test_total_failure(self):
        raw = "I cannot evaluate this."
        score, justification = _parse_judge_response(raw)
        assert score == 3
        assert "unparseable" in justification

    def test_score_clamped_in_parsing(self):
        raw = '{"score": 999, "justification": "out of range"}'
        score, _ = _parse_judge_response(raw)
        assert score == 5

    def test_nested_json_resists_extraction(self):
        raw = '{"score": 4, "justification": "good"}'
        score, _ = _parse_judge_response(raw)
        assert score == 4


class TestBuildJudgeLLM:
    def test_returns_none_when_api_key_missing(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "")
        monkeypatch.setenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
        from app.config import get_settings
        get_settings.cache_clear()
        try:
            llm = build_judge_llm()
            assert llm is None
        finally:
            get_settings.cache_clear()

    def test_returns_none_when_base_url_missing(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        monkeypatch.setenv("OPENAI_BASE_URL", "")
        from app.config import get_settings
        get_settings.cache_clear()
        try:
            llm = build_judge_llm()
            assert llm is None
        finally:
            get_settings.cache_clear()

    def test_returns_llm_when_configured(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        monkeypatch.setenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
        from app.config import get_settings
        get_settings.cache_clear()
        try:
            with patch("app.evaluation.generation_metrics.ChatOpenAI") as mock_llm:
                build_judge_llm()
                mock_llm.assert_called_once()
        finally:
            get_settings.cache_clear()


class TestMetricFunctions:
    def _mock_judge_llm(self):
        llm = MagicMock()
        response = MagicMock()
        response.content = '{"score": 4, "justification": "good"}'
        llm.invoke.return_value = response
        return llm

    def test_faithfulness_prompt_contains_question_and_context(self):
        llm = self._mock_judge_llm()
        evaluate_faithfulness("What is X?", "X is Y.", "Context about X.", llm)
        call_args = llm.invoke.call_args[0][0]
        assert "What is X?" in call_args
        assert "X is Y." in call_args
        assert "Context about X." in call_args
        assert "FAITHFULNESS" in call_args

    def test_answer_relevance_prompt_contains_rubric(self):
        llm = self._mock_judge_llm()
        evaluate_answer_relevance("Q?", "A.", "Ctx.", llm)
        call_args = llm.invoke.call_args[0][0]
        assert "ANSWER RELEVANCE" in call_args

    def test_context_usage_prompt_contains_rubric(self):
        llm = self._mock_judge_llm()
        evaluate_context_usage("Q?", "A.", "Ctx.", llm)
        call_args = llm.invoke.call_args[0][0]
        assert "CONTEXT USAGE" in call_args

    def test_noise_sensitivity_returns_none_when_empty_noise(self):
        llm = self._mock_judge_llm()
        result = evaluate_noise_sensitivity("Q?", "A.", "Ctx.", [], llm)
        assert result is None

    def test_noise_sensitivity_prompt_contains_noise_block(self):
        llm = self._mock_judge_llm()
        evaluate_noise_sensitivity("Q?", "A.", "Ctx.", ["noise1", "noise2"], llm)
        call_args = llm.invoke.call_args[0][0]
        assert "noise1" in call_args
        assert "noise2" in call_args
        assert "Noise Context" in call_args

    def test_noise_sensitivity_score_parsed(self):
        llm = self._mock_judge_llm()
        score, just = evaluate_noise_sensitivity("Q?", "A.", "Ctx.", ["noise"], llm)
        assert score == 4
        assert just == "good"


class TestEvaluateAllGenerationMetrics:
    def _mock_judge_llm(self):
        llm = MagicMock()
        response = MagicMock()
        response.content = '{"score": 4, "justification": "solid"}'
        llm.invoke.return_value = response
        return llm

    def test_returns_generation_eval_result(self):
        llm = self._mock_judge_llm()
        docs = _make_retrieved_docs(["wiki"])
        result = evaluate_all_generation_metrics(
            query_id="q1",
            question="What is RAG?",
            answer="RAG is retrieval augmented generation.",
            retrieved_docs=docs,
            noise_contexts=["Buy cheap sunglasses now!"],
            judge_llm=llm,
        )
        assert isinstance(result, GenerationEvalResult)
        assert result.query_id == "q1"
        assert result.faithfulness_score == 4
        assert result.answer_relevance_score == 4
        assert result.context_usage_score == 4
        assert result.noise_sensitivity_score == 4

    def test_noise_skipped_when_empty(self):
        llm = self._mock_judge_llm()
        result = evaluate_all_generation_metrics(
            query_id="q2",
            question="Q?",
            answer="A.",
            retrieved_docs=[],
            noise_contexts=[],
            judge_llm=llm,
        )
        assert result.noise_sensitivity_score is None
        assert result.noise_sensitivity_justification is None


class TestAggregateGeneration:
    def test_aggregates_multiple_results(self):
        results = [
            GenerationEvalResult(
                query_id="q1",
                faithfulness_score=4,
                faithfulness_justification="good",
                answer_relevance_score=5,
                answer_relevance_justification="perfect",
                context_usage_score=3,
                context_usage_justification="ok",
                noise_sensitivity_score=4,
                noise_sensitivity_justification="fine",
            ),
            GenerationEvalResult(
                query_id="q2",
                faithfulness_score=2,
                faithfulness_justification="bad",
                answer_relevance_score=3,
                answer_relevance_justification="ok",
                context_usage_score=1,
                context_usage_justification="none",
                noise_sensitivity_score=None,
                noise_sensitivity_justification=None,
            ),
        ]
        agg = aggregate_generation(results)
        assert agg["avg_faithfulness"] == 3.0
        assert agg["avg_answer_relevance"] == 4.0
        assert agg["avg_context_usage"] == 2.0
        assert agg["avg_noise_sensitivity"] == 4.0  # only one non-None value

    def test_empty_results(self):
        agg = aggregate_generation([])
        assert agg["avg_faithfulness"] is None
        assert agg["avg_answer_relevance"] is None
        assert agg["avg_context_usage"] is None
        assert agg["avg_noise_sensitivity"] is None

    def test_all_noise_none(self):
        results = [
            GenerationEvalResult(
                query_id="q1",
                faithfulness_score=3,
                faithfulness_justification="ok",
                answer_relevance_score=3,
                answer_relevance_justification="ok",
                context_usage_score=3,
                context_usage_justification="ok",
                noise_sensitivity_score=None,
                noise_sensitivity_justification=None,
            ),
        ]
        agg = aggregate_generation(results)
        assert agg["avg_faithfulness"] == 3.0
        assert agg["avg_noise_sensitivity"] is None
