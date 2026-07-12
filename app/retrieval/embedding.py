"""Shared embedding singleton for all retrieval modules."""

import os

from langchain_community.embeddings import FastEmbedEmbeddings

from app.config import get_settings

_embeddings_instance: FastEmbedEmbeddings | None = None


def get_embeddings() -> FastEmbedEmbeddings:
    global _embeddings_instance
    if _embeddings_instance is None:
        settings = get_settings()
        cache_dir = os.getenv("FASTEMBED_CACHE_DIR", None)
        _embeddings_instance = FastEmbedEmbeddings(
            model_name=settings.local_embedding_model_name,
            cache_dir=cache_dir,
        )
    return _embeddings_instance
