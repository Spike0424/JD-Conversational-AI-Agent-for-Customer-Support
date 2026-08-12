"""Search knowledge base with hybrid retrieval: alias matching + keyword scoring + vector semantic fusion."""

import logging
import re
import time
import uuid
from typing import Optional, Union

import numpy as np
from pydantic import BaseModel, Field

from app.business_models import (
    AftersaleKnowledge,
    InsaleKnowledge,
    PresaleKnowledge,
)
from app.config import SUB_SCENE_RULES, get_settings
from app.db import create_session
from app.evaluation.rag_logger import RAGLogger
from app.retrieval.aftersale_retriever import match_aftersale_exact
from app.retrieval.embedding import get_embeddings
from util.agent_tool import agent_tool

logger = logging.getLogger(__name__)

_search_knowledge_description = (
    "Search the internal knowledge base for after-sales policies, "
    "product specifications, usage instructions, or shipping information. "
    "Use this for ANY e-commerce customer service question. "
    "Results are automatically filtered by the current shop and scene."
)

_SCENE_TABLE = {
    "presale": PresaleKnowledge,
    "insale": InsaleKnowledge,
    "aftersale": AftersaleKnowledge,
}

# "mixed" is a virtual scene for clarification prompts — skip DB retrieval
_SCENE_NO_RETRIEVAL = {"mixed"}

_VECTOR_MATCH_THRESHOLD = 0.45
_MATCH_TYPE_RULE = "rule"
_MATCH_TYPE_HYBRID = "hybrid"
_NON_WORD_RE = re.compile(r"[^\w]")
_SCENE_LABELS = {"presale": "售前", "insale": "售中", "aftersale": "售后"}

# ── Pydantic param model ──────────────────────────────────────────────


class SearchKnowledgeParams(BaseModel):
    """Unified knowledge search params."""

    query: str = Field(..., description="客户原始问题")
    shop_id: Union[str, int] = Field(..., description="店铺ID")
    user_id: Optional[Union[str, int]] = Field(None, description="当前客服账号ID")
    recipient_uid: Optional[str] = Field(None, description="客户UID")
    goods_id: Optional[int] = Field(None, description="当前商品ID")
    scene: Optional[str] = Field(None, description="售前/售中/售后")


# ── SearchKnowledge ───────────────────────────────────────────────────


class SearchKnowledge:
    """Hybrid knowledge-base search with alias/keyword/vector scoring.

    Session context (shop_id, scene, goods_id) is set before each LLM
    invocation so the tool can filter retrieval to the correct shop and scene.
    """

    def __init__(self) -> None:
        self._shop_id: int | None = None
        self._scene: str = "presale"
        self._goods_id: int | None = None
        self._goods_name: str = ""
        self._settings = get_settings()
        self._recent_user_messages: list[str] = []
        self._tracker = None
        self._shop_cache: dict[tuple[int, str], tuple[float, str]] = {}
        self._product_cache: dict[tuple[int, str, int], tuple[float, str]] = {}

    def set_context(
        self, shop_id: int | str | None, scene: str = "presale",
        goods_id: int | None = None, goods_name: str = "",
    ) -> None:
        self._shop_id = self._coerce_shop_id(shop_id)
        self._scene = scene
        self._goods_id = goods_id
        self._goods_name = goods_name

    def get_shop_id(self) -> int | None:
        return self._shop_id

    @staticmethod
    def _coerce_shop_id(shop_id: int | str | None) -> int | None:
        if shop_id is None:
            return None
        if isinstance(shop_id, int):
            return shop_id
        try:
            return int(str(shop_id).strip())
        except (ValueError, TypeError):
            logger.warning("shop_id %r is not int-coercible; ignoring shop filter", shop_id)
            return None

    def set_history(self, recent_user_messages: list[str]) -> None:
        self._recent_user_messages = recent_user_messages[-3:]

    def set_tracker(self, tracker) -> None:
        self._tracker = tracker

    def _contextual_complete(self, query: str) -> str:
        clean = query.strip()
        if not clean:
            return query
        signal_terms = [t for t in re.split(r"[\s,，、；;]+", clean) if len(t) >= 2]
        needs_context = (
            len(clean) <= 6
            or ("这款" in clean and len(clean) <= 14)
            or len(signal_terms) <= 1
        )
        if not needs_context or not self._recent_user_messages:
            return query
        history_keywords: list[str] = []
        for msg in self._recent_user_messages:
            for m in re.finditer(r"[一-鿿]{2,}", msg):
                kw = m.group()
                if kw not in history_keywords and kw not in self._settings.search_stop_words:
                    history_keywords.append(kw)
        if not history_keywords:
            return query
        return f"{clean} {' '.join(history_keywords[:5])}"

    # ── helpers ──────────────────────────────────────────────────

    @staticmethod
    def _normalize_match_text(text: str) -> str:
        return _NON_WORD_RE.sub("", text.lower().strip())

    # ── alias match scoring (highest weight) ──────────────────────

    def _alias_match_score(self, query_clean: str, aliases: str) -> int:
        if not query_clean or not aliases:
            return 0

        best_score = 0
        for alias in re.split(r"[/|;；\n\r]+", aliases):
            alias_clean = self._normalize_match_text(alias)
            if len(alias_clean) < 2:
                continue

            if alias_clean == query_clean:
                best_score = max(best_score, 240 + min(len(alias_clean), 30))
            elif len(alias_clean) >= 4 and alias_clean in query_clean:
                best_score = max(best_score, 115 + min(len(alias_clean), 12))
            elif len(query_clean) >= 6 and query_clean in alias_clean:
                best_score = max(best_score, 80 + min(len(query_clean), 12))
            elif alias_clean in query_clean or query_clean in alias_clean:
                best_score = max(best_score, 8 + min(len(alias_clean), 6))

        return best_score

    # ── search-term extraction (Layer 3: jieba + phrase + synonym - stop words) ──

    def _search_terms(self, query: str) -> list[str]:
        query_lower = query.lower()
        terms: list[str] = []
        term_lower_set: set[str] = set()
        stop_words = set(getattr(self._settings, 'search_stop_words', []))

        try:
            import jieba
            jieba_tokens = list(jieba.cut_for_search(query))
        except Exception:
            jieba_tokens = re.findall(r"[一-鿿]{2,}|[a-zA-Z]+|\d+", query)

        for token in jieba_tokens:
            token = token.strip()
            if len(token) >= 2 and token not in stop_words and token.lower() not in term_lower_set:
                terms.append(token)
                term_lower_set.add(token.lower())

        for phrase in self._settings.search_phrase_candidates:
            if phrase.lower() in query_lower and phrase.lower() not in term_lower_set:
                terms.append(phrase)
                term_lower_set.add(phrase.lower())

        for key, variants in self._settings.search_synonym_expansions.items():
            if key.lower() in query_lower:
                for v in variants:
                    v_lower = v.lower()
                    if v_lower not in term_lower_set:
                        terms.append(v)
                        term_lower_set.add(v_lower)

        return terms

    # ── keyword match scoring (~120 points) ───────────────────────

    def _bm25_like_match_score(self, words: list[str], entry: dict) -> int:
        if not words:
            return 0

        text_fields = [
            entry.get("aliases", ""),
            entry.get("answer", ""),
            entry.get("tags", ""),
            entry.get("product_family", ""),
            entry.get("section_title", ""),
        ]
        text = " ".join(f for f in text_fields if f).lower()

        score = 0
        for word in words:
            count = text.count(word.lower())
            if count > 0:
                score += min(10 + count * 3, 25)
        return min(score, 120)

    def _keyword_match_score(self, query: str, entry: dict) -> int:
        words = self._search_terms(query)
        if not words:
            return 0
        return self._bm25_like_match_score(words, entry)

    # ── goods_id consistency ──────────────────────────────────────

    def _goods_id_bonus(self, entry_goods_id: int | None) -> int:
        if self._goods_id is not None and entry_goods_id == self._goods_id:
            return 50
        if entry_goods_id is not None and self._goods_id is not None and entry_goods_id != self._goods_id:
            return -30
        return 0

    # ── data fetching ─────────────────────────────────────────────

    def _fetch_candidates(
        self, shop_id: int, scene: str, goods_id: int | None,
        product_only: bool = False,
    ) -> list[dict]:
        """Fetch knowledge entries. By default returns product-specific + shop-general
        when goods_id is given; pass product_only=True to exclude shop-general rows."""
        table = _SCENE_TABLE.get(scene, PresaleKnowledge)
        session = create_session()
        try:
            from sqlalchemy import or_

            filters = [
                table.shop_id == shop_id,
                table.enabled == True,
            ]
            if goods_id is not None:
                if product_only:
                    filters.append(table.goods_id == goods_id)
                else:
                    filters.append(
                        or_(table.goods_id == goods_id, table.goods_id.is_(None))
                    )
            else:
                filters.append(table.goods_id.is_(None))

            rows = session.query(table).filter(*filters).all()

            candidates: list[dict] = []
            for row in rows:
                candidates.append({
                    "id": row.id,
                    "goods_id": row.goods_id,
                    "aliases": row.aliases or "",
                    "answer": row.answer or "",
                    "sub_intent": row.sub_intent,
                    "product_family": row.product_family or "",
                    "tags": row.tags or "",
                    "section_title": row.section_title or "",
                    "priority": row.priority or 0,
                })

            return candidates
        except Exception:
            logger.exception("_fetch_candidates failed shop=%s scene=%s", shop_id, scene)
            return []
        finally:
            session.close()

    # ── prompt injection (pre-RAG) ─────────────────────────────────

    @staticmethod
    def _format_for_prompt(entries: list[dict]) -> str:
        """Format KB entries as a bullet list for system-prompt injection."""
        if not entries:
            return ""
        lines: list[str] = []
        for e in entries:
            aliases = (e.get("aliases") or "").strip()
            answer = (e.get("answer") or "").strip()
            if not answer:
                continue
            if aliases:
                lines.append(f"- 【{aliases}】{answer}")
            else:
                lines.append(f"- {answer}")
        return "\n".join(lines)

    _PRE_RAG_CACHE_TTL = 60.0

    def fetch_shop_advantages(self, shop_id: int, scene: str) -> str:
        """Shop-general knowledge (goods_id IS NULL) for [DB_SHOP_ADVANTAGES] placeholder. Cached 60s."""
        key = (shop_id, scene)
        now = time.time()
        cached = self._shop_cache.get(key)
        if cached and now - cached[0] < self._PRE_RAG_CACHE_TTL:
            return cached[1]
        result = self._format_for_prompt(self._fetch_candidates(shop_id, scene, goods_id=None))
        self._shop_cache[key] = (now, result)
        return result

    def fetch_product_knowledge(self, shop_id: int, scene: str, goods_id: int) -> str:
        """Product-specific knowledge (goods_id == current) for [DB_PRODUCT_KNOWLEDGE] placeholder. Cached 60s."""
        key = (shop_id, scene, goods_id)
        now = time.time()
        cached = self._product_cache.get(key)
        if cached and now - cached[0] < self._PRE_RAG_CACHE_TTL:
            return cached[1]
        result = self._format_for_prompt(
            self._fetch_candidates(shop_id, scene, goods_id=goods_id, product_only=True)
        )
        self._product_cache[key] = (now, result)
        return result

    # ── vector semantic scoring ───────────────────────────────────

    def _vector_semantic_score(
        self, query: str, candidates: list[dict]
    ) -> list[float]:
        """Batch-embed query + candidates, return cosine similarity per entry."""
        if not candidates:
            return []

        try:
            embeddings = get_embeddings()
            query_vec = np.array(embeddings.embed_query(query), dtype=np.float32)

            texts = []
            for entry in candidates:
                text = entry.get("aliases", "") or entry.get("answer", "")
                texts.append(text[:2048])

            candidate_vecs = np.array(embeddings.embed_documents(texts), dtype=np.float32)

            query_norm = query_vec / (np.linalg.norm(query_vec) or 1.0)
            candidate_norms = candidate_vecs / (
                np.linalg.norm(candidate_vecs, axis=1, keepdims=True) + 1e-10
            )
            scores = np.dot(candidate_norms, query_norm)

            return scores.tolist()
        except Exception:
            logger.exception("_vector_semantic_score failed, returning zeros")
            return [0.0] * len(candidates)

    # ── ranking ───────────────────────────────────────────────────

    def _rank(self, query: str, candidates: list[dict]) -> list[dict]:
        if not candidates:
            return []

        query_clean = self._normalize_match_text(query)

        results: list[dict] = []
        vector_scores = self._vector_semantic_score(query, candidates)

        for i, entry in enumerate(candidates):
            alias_score = self._alias_match_score(query_clean, entry.get("aliases", ""))
            keyword_score = self._keyword_match_score(query, entry)
            goods_bonus = self._goods_id_bonus(entry.get("goods_id"))
            vector_sim = vector_scores[i]

            rule_total = alias_score + keyword_score + goods_bonus

            if vector_sim >= _VECTOR_MATCH_THRESHOLD:
                match_type = _MATCH_TYPE_HYBRID
                final_score = rule_total + vector_sim * 100
            else:
                match_type = _MATCH_TYPE_RULE
                final_score = rule_total

            results.append({
                **entry,
                "alias_score": alias_score,
                "keyword_score": keyword_score,
                "goods_bonus": goods_bonus,
                "vector_similarity": round(vector_sim, 4),
                "match_type": match_type,
                "final_score": round(final_score, 2),
            })

        results.sort(key=lambda r: r["final_score"], reverse=True)
        return results

    # ── sub-scene scoring (kept from original) ────────────────────

    @staticmethod
    def score_sub_scene(query: str) -> dict[str, int]:
        query_lower = query.lower()
        scores: dict[str, int] = {}
        for scene, keywords in SUB_SCENE_RULES.items():
            scores[scene] = 1 if any(kw in query_lower for kw in keywords) else -1
        return scores

    @staticmethod
    def best_sub_scene(query: str) -> str | None:
        scores = SearchKnowledge.score_sub_scene(query)
        best = max(scores, key=lambda k: scores[k])
        return best if scores[best] > 0 else None

    # ── main search ───────────────────────────────────────────────

    def search(self, query: str, trace_id: str = "") -> str:
        from app.orchestrator.request_state import RequestState

        # Layer 2: contextual completion for short queries
        query_before = query
        query = self._contextual_complete(query)
        # Layer 3: search terms
        search_terms = self._search_terms(query)

        if query != query_before:
            logger.info("RAG query optimization: query=%r -> complete=%r -> terms=%s",
                        query_before[:80], query[:80], search_terms)

        if self._tracker:
            self._tracker.transition(RequestState.RAG_QUERY_CLEANED, query=query[:50], terms=len(search_terms))
        tid = trace_id or uuid.uuid4().hex[:12]
        RAGLogger.log_query(tid, query)

        best = self.best_sub_scene(query)
        logger.info(
            "Sub-scene best=%s trace=%s shop=%s scene=%s goods=%s",
            best, tid, self._shop_id, self._scene, self._goods_id,
        )

        if self._shop_id is None:
            return "No shop context set. Cannot search knowledge base."

        if self._scene in _SCENE_NO_RETRIEVAL:
            return ""

        # ── Aftersale fast-path: exact alias match ──
        if self._scene == "aftersale":
            exact = match_aftersale_exact(query, self._shop_id, self._goods_id)
            if exact:
                return f"[aftersale exact] {exact}"

        # ── Hybrid retrieval ──
        if self._tracker:
            self._tracker.transition(RequestState.RAG_FETCHING, scene=self._scene)
        candidates = self._fetch_candidates(
            shop_id=self._shop_id,
            scene=self._scene,
            goods_id=self._goods_id,
        )

        if not candidates:
            if self._tracker:
                self._tracker.transition(RequestState.RAG_NO_HIT)
            label = _SCENE_LABELS.get(self._scene, self._scene)
            return f"No {label} knowledge found for shop={self._shop_id} goods={self._goods_id}."

        if self._tracker:
            self._tracker.transition(RequestState.RAG_RANKING, candidates=len(candidates))
        ranked = self._rank(query, candidates)

        # ── Format results for LLM (clean, no debug scores) ──
        lines: list[str] = []
        top_n = min(len(ranked), 2)
        for i, r in enumerate(ranked[:top_n]):
            lines.append(
                f"## 知识条目 {i + 1}\n"
                f"关键词：{r['aliases']}\n"
                f"内容：{r['answer']}"
            )

        header = f"共找到 {len(ranked)} 条相关知识，以下是前 {top_n} 条："

        # ── Structured RAG trace log ──
        # top_k is what the LLM sees; logger records ALL ranked candidates for debugging/eval.
        retrieval_hit = len(ranked) > 0 and ranked[0]["final_score"] > 0
        RAGLogger.log_trace(
            tid,
            phase="retrieval",
            user_query=query,
            scene=self._scene,
            shop_id=self._shop_id,
            goods_id=self._goods_id,
            candidates_count=len(candidates),
            ranked_count=len(ranked),
            filters={"shop_id": self._shop_id, "goods_id": self._goods_id, "scene": self._scene},
            top_k=min(len(ranked), 2),
            retrieved_chunks=[
                {"source": r.get("aliases", ""), "score": r.get("final_score", 0), "snippet": r.get("answer", "")}
                for r in ranked
            ],
            rerank_result=[
                {"rank": i + 1, "source": r.get("aliases", ""),
                 "sub_intent": r.get("sub_intent", ""),
                 "match_type": r.get("match_type", ""), "final_score": r.get("final_score", 0),
                 "alias_score": r.get("alias_score", 0), "keyword_score": r.get("keyword_score", 0),
                 "vector_similarity": r.get("vector_similarity", 0),
                 "goods_bonus": r.get("goods_bonus", 0)}
                for i, r in enumerate(ranked)
            ],
            retrieval_hit=retrieval_hit,
        )

        if self._tracker:
            self._tracker.transition(RequestState.RAG_RESULT_FORMATTED, top_k=min(len(ranked), 2))

        return header + "\n\n" + "\n\n".join(lines)

    def search_structured(self, query: str) -> list[dict]:
        """Return structured ranked results for eval (no formatted text)."""
        if self._shop_id is None:
            return []
        if self._scene in _SCENE_NO_RETRIEVAL:
            return []

        candidates = self._fetch_candidates(
            shop_id=self._shop_id,
            scene=self._scene,
            goods_id=self._goods_id,
        )
        if not candidates:
            return []

        ranked = self._rank(query, candidates)
        results = []
        for r in ranked:
            results.append({
                "source": r.get("aliases", ""),
                "score": r.get("final_score", 0),
                "snippet": r.get("answer", "")[:300],
            })
        return results


# ── singleton ─────────────────────────────────────────────────────────

_search_knowledge_instance: SearchKnowledge | None = None


def set_search_knowledge(instance: SearchKnowledge) -> None:
    global _search_knowledge_instance
    _search_knowledge_instance = instance


def get_search_knowledge() -> SearchKnowledge | None:
    return _search_knowledge_instance


# ── tool registration ─────────────────────────────────────────────────


@agent_tool(
    name="search_knowledge",
    description=_search_knowledge_description,
    param_model=SearchKnowledgeParams,
)
def search_knowledge(**kwargs: object) -> str:
    if _search_knowledge_instance is None:
        return "search_knowledge is not initialized."

    params = SearchKnowledgeParams(**{k: v for k, v in kwargs.items() if v is not None})

    if params.shop_id:
        try:
            sid = int(params.shop_id) if isinstance(params.shop_id, str) else int(params.shop_id)
        except (ValueError, TypeError):
            sid = None
        if sid is not None:
            _search_knowledge_instance.set_context(
                shop_id=sid,
                scene=params.scene or "presale",
                goods_id=params.goods_id,
            )

    return _search_knowledge_instance.search(params.query)
