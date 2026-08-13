"""Layer 2: Semantic vector matching against seed script database."""

import logging
from typing import Any, Optional

import numpy as np
from langchain_core.embeddings import Embeddings
from langchain_community.embeddings import FastEmbedEmbeddings

from api.core.config import get_settings
from api.models.seed import SemanticRouteResult, get_seed_count, get_seed_index

logger = logging.getLogger(__name__)

# L2 distance threshold: below this → high confidence match
MATCH_THRESHOLD = 0.5

_embeddings: Embeddings | None = None


def _get_embeddings() -> Embeddings:
    global _embeddings
    if _embeddings is None:
        settings = get_settings()
        _embeddings = FastEmbedEmbeddings(model_name=settings.local_embedding_model_name)
    return _embeddings


def semantic_route(query: str, scenario: str | None = None) -> Optional[SemanticRouteResult]:
    """Try to match query against seed script database.

    If scenario is provided, only search seeds matching that scenario.
    Returns matched template dict if L2 distance < threshold, else None.
    """
    seed_array, seed_metas = get_seed_index(scenario=scenario)
    if seed_array is None or len(seed_array) == 0:
        logger.debug("Seed index empty, skipping semantic route")
        return None

    emb = _get_embeddings()
    query_vec = np.array(emb.embed_query(query), dtype=np.float32)
    scores = np.linalg.norm(seed_array - query_vec, axis=1)

    best_idx = int(np.argmin(scores))
    best_score = float(scores[best_idx])

    if best_score < MATCH_THRESHOLD:
        meta = seed_metas[best_idx]
        logger.info(
            "Semantic match: score=%.4f platform=%s scenario=%s trigger=%s",
            best_score, meta["platform"], meta["scenario"], meta["trigger_text"][:60],
        )
        return SemanticRouteResult(
            platform=meta["platform"],
            scenario=meta["scenario"],
            trigger_text=meta["trigger_text"],
            response_template=meta["response_template"],
            score=best_score,
        )

    logger.debug("Semantic no match: best_score=%.4f (threshold=%.2f)", best_score, MATCH_THRESHOLD)
    return None
