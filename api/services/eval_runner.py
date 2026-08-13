from __future__ import annotations

import logging

from langchain_openai import ChatOpenAI

from api.core.config import Settings, get_settings
from api.services.gen_metrics import (
    aggregate_generation,
    build_judge_llm,
    evaluate_all_generation_metrics,
)
from api.services.retrieval_metrics import aggregate_retrieval, compute_retrieval_metrics, normalize_docs
from api.models.eval_schemas import EvalQuery, EvalReport
from api.core.agent_runtime import ReActQAAgent
from api.core.scene_classifier import SceneClassifier
from api.services.knowledge import get_search_knowledge, set_search_knowledge, SearchKnowledge

logger = logging.getLogger(__name__)

_UNSET = object()


class EvalRunner:
    """Orchestrates production-path evaluation across retrieval and generation phases.

    Pipeline per query:
      1. SceneClassifier.classify() with user_id from query → determines scene
      2. SearchKnowledge.search_structured() → retrieves scene-specific knowledge
      3. ReActQAAgent.ask() → generates answer using the same tool
    """

    def __init__(self, settings: Settings | None = None) -> None:
        self._settings = settings or get_settings()
        self._agent: ReActQAAgent | None = None
        self._judge_llm: ChatOpenAI | None | object = _UNSET
        self._scene_classifier = SceneClassifier()

        if get_search_knowledge() is None:
            set_search_knowledge(SearchKnowledge())

    def _get_agent(self) -> ReActQAAgent:
        if self._agent is None:
            self._agent = ReActQAAgent()
        return self._agent

    def _get_judge_llm(self) -> ChatOpenAI | None:
        if self._judge_llm is _UNSET:
            self._judge_llm = build_judge_llm()
        return self._judge_llm  # type: ignore[return-value]

    def run(
        self,
        dataset: list[EvalQuery],
        dataset_name: str = "unnamed",
        k: int | None = None,
        run_generation_eval: bool = True,
        max_queries: int | None = None,
    ) -> EvalReport:
        k = k if k is not None else self._settings.rag_top_k

        if max_queries is not None and max_queries < len(dataset):
            dataset = dataset[:max_queries]
            logger.info("Dataset truncated to %d queries (max_queries=%d)", len(dataset), max_queries)

        judge_llm = self._get_judge_llm() if run_generation_eval else None
        if run_generation_eval and judge_llm is None:
            logger.warning("Generation evaluation requested but LLM judge unavailable. Skipping.")

        sk = get_search_knowledge()
        agent = self._get_agent() if (run_generation_eval and judge_llm is not None) else None

        retrieval_results = []
        generation_results = []

        for query in dataset:
            logger.info("Evaluating query_id=%s", query.query_id)

            # Step 1: classify scene from order data (or fall back to query.expected_scene)
            dependencies = (
                {"customer_uid": query.user_id, "user_id": query.user_id}
                if query.user_id
                else {}
            )
            session_id = f"eval-{dataset_name}-{query.query_id}"
            classified_scene = self._scene_classifier.classify(
                dependencies=dependencies,
                question=query.question,
                session_id=session_id,
            )

            if query.expected_scene and classified_scene != query.expected_scene:
                logger.warning(
                    "Scene mismatch for %s: classified=%s expected=%s user_id=%s",
                    query.query_id, classified_scene, query.expected_scene, query.user_id,
                )

            # Step 2: production-path retrieval via SearchKnowledge
            sk.set_context(shop_id=1, scene=classified_scene, goods_id=None)
            retrieved = sk.search_structured(query.question)

            retrieval_result = compute_retrieval_metrics(
                query_id=query.query_id,
                retrieved_docs=retrieved,
                relevant_patterns=query.relevant_source_patterns,
                k=k,
            )
            retrieval_results.append(retrieval_result)

            if agent is not None and judge_llm is not None:
                try:
                    answer = agent.ask(
                        session_id=session_id,
                        question=query.question,
                        scene=classified_scene,
                    )
                except Exception:
                    logger.exception("Agent failed for query_id=%s", query.query_id)
                    answer = "[Agent error: generation failed]"

                normalized_docs = normalize_docs(retrieved, k)
                gen_result = evaluate_all_generation_metrics(
                    query_id=query.query_id,
                    question=query.question,
                    answer=answer,
                    retrieved_docs=normalized_docs,
                    noise_contexts=query.noise_contexts,
                    judge_llm=judge_llm,
                )
                generation_results.append(gen_result)

        ret_agg = aggregate_retrieval(retrieval_results)
        gen_agg = aggregate_generation(generation_results) if generation_results else {
            "avg_faithfulness": None,
            "avg_answer_relevance": None,
            "avg_context_usage": None,
            "avg_noise_sensitivity": None,
        }

        return EvalReport(
            dataset_name=dataset_name,
            num_queries=len(dataset),
            k=k,
            avg_recall_at_k=ret_agg["avg_recall_at_k"],
            avg_hit_rate_at_k=ret_agg["avg_hit_rate_at_k"],
            avg_mrr=ret_agg["avg_mrr"],
            avg_precision_at_k=ret_agg["avg_precision_at_k"],
            avg_faithfulness=gen_agg["avg_faithfulness"],
            avg_answer_relevance=gen_agg["avg_answer_relevance"],
            avg_context_usage=gen_agg["avg_context_usage"],
            avg_noise_sensitivity=gen_agg["avg_noise_sensitivity"],
            retrieval_details=retrieval_results,
            generation_details=generation_results if generation_results else None,
        )