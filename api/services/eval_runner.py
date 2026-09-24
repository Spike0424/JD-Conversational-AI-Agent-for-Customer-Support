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
from api.models.eval_schemas import EvalQuery, EvalReport, GenerationEvalResult
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

    @staticmethod
    def _resolve_default_shop_id() -> int | None:
        """Fallback: pick the shop with the most aftersale_knowledge rows.

        Returns None if no shop has any knowledge.
        """
        from sqlalchemy import func

        from api.models.business import AftersaleKnowledge, Shop
        from api.models.db import create_session

        session = create_session()
        try:
            row = (
                session.query(Shop, func.count(AftersaleKnowledge.id).label("n"))
                .outerjoin(AftersaleKnowledge, AftersaleKnowledge.shop_id == Shop.id)
                .group_by(Shop.id)
                .order_by(func.count(AftersaleKnowledge.id).desc(), Shop.id.asc())
                .first()
            )
            if row is None:
                return None
            shop, n = row[0], row[1]
            logger.info("Auto-resolved shop_id=%s (%s) with %d aftersale rows", shop.id, shop.shop_name, n)
            return shop.id
        finally:
            session.close()

    def _get_judge_llm(self) -> ChatOpenAI | None:
        if self._judge_llm is _UNSET:
            self._judge_llm = build_judge_llm()
        return self._judge_llm  # type: ignore[return-value]

    async def run(
        self,
        dataset: list[EvalQuery],
        dataset_name: str = "unnamed",
        k: int | None = None,
        run_generation_eval: bool = True,
        max_queries: int | None = None,
        shop_id: int | None = None,
    ) -> EvalReport:
        k = k if k is not None else self._settings.rag_top_k

        if max_queries is not None and max_queries < len(dataset):
            dataset = dataset[:max_queries]
            logger.info("Dataset truncated to %d queries (max_queries=%d)", len(dataset), max_queries)

        judge_llm = self._get_judge_llm() if run_generation_eval else None
        if run_generation_eval and judge_llm is None:
            logger.warning("Generation evaluation requested but LLM judge unavailable. Skipping.")

        sk = get_search_knowledge()
        # The agent is always needed: even when the judge is skipped, we record
        # the raw LLM answer in the JSON report for manual inspection.
        agent = self._get_agent()

        # Clear SceneClassifier's session-level cache so each run exercises the
        # real order-context lookup (cache TTL is 30 min, and EvalRunner uses
        # deterministic session_ids f"eval-{dataset_name}-{query.query_id}",
        # which would otherwise short-circuit scene classification on re-runs).
        from api.core.scene_classifier import SceneClassifier

        cleared = SceneClassifier.clear_cache()
        logger.info("Cleared SceneClassifier cache: %d entries", cleared)

        # Resolve shop_id: explicit arg → settings.default_shop_id → shop with most knowledge.
        # Hardcoding shop_id=1 was a footgun when seed data uses different PKs.
        if shop_id is None:
            shop_id = self._settings.default_shop_id
        if shop_id is None:
            shop_id = self._resolve_default_shop_id()
            if shop_id is None:
                logger.error("No shop found in DB and DEFAULT_SHOP_ID not set; aborting.")
                raise RuntimeError("EvalRunner needs at least one shop to retrieve against.")
        logger.info("EvalRunner using shop_id=%s", shop_id)

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
            classified_scene = await self._scene_classifier.classify(
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
            sk.set_context(shop_id=shop_id, scene=classified_scene, goods_id=None)
            total_relevant = sk.count_relevant_documents(query.relevant_source_patterns)
            retrieved = sk.search_structured(query.question)

            retrieval_result = compute_retrieval_metrics(
                query_id=query.query_id,
                retrieved_docs=retrieved,
                relevant_patterns=query.relevant_source_patterns,
                k=k,
                total_relevant_override=total_relevant,
            )
            # Attach raw retrieval output for JSON dump inspection.
            # normalize_docs() returns list[RetrievedDoc]; dump to plain dicts so
            # the schema field (list[dict]) serializes without Pydantic warnings.
            retrieval_result.retrieved_docs = [
                d.model_dump() for d in normalize_docs(retrieved, k)
            ]
            retrieval_result.relevant_source_patterns = list(query.relevant_source_patterns)
            # Record the raw question + the search terms for JSON inspection.
            retrieval_result.query = query.question
            retrieval_result.search_terms = list(sk.last_search_terms)
            retrieval_results.append(retrieval_result)

            # Step 3: generate the answer on the production path. Scores are
            # attached only when a judge LLM is available; otherwise the entry
            # is answers-only (scores=None) but still lands in the JSON report.
            try:
                answer = await agent.ask(
                    session_id=session_id,
                    question=query.question,
                    scene=classified_scene,
                )
            except Exception:
                logger.exception("Agent failed for query_id=%s", query.query_id)
                answer = "[Agent error: generation failed]"

            if judge_llm is not None:
                normalized_docs = normalize_docs(retrieved, k)
                gen_result = evaluate_all_generation_metrics(
                    query_id=query.query_id,
                    question=query.question,
                    answer=answer,
                    retrieved_docs=normalized_docs,
                    noise_contexts=query.noise_contexts,
                    judge_llm=judge_llm,
                )
            else:
                gen_result = GenerationEvalResult(query_id=query.query_id)
            # Attach raw generation output + reference answer for diff inspection.
            gen_result.agent_answer = answer
            gen_result.ground_truth_answer = query.ground_truth_answer
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