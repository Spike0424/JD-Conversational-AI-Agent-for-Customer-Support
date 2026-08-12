"""Shared embedding singleton for all retrieval modules."""

import os
from pathlib import Path

from langchain_community.embeddings import FastEmbedEmbeddings

from app.config import get_settings

DEFAULT_CACHE_DIR = str(Path.home() / ".cache" / "fastembed")

_embeddings_instance: FastEmbedEmbeddings | None = None


def get_embeddings() -> FastEmbedEmbeddings:
    global _embeddings_instance
    if _embeddings_instance is None:
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        cache_dir = os.getenv("FASTEMBED_CACHE_DIR", DEFAULT_CACHE_DIR)
        settings = get_settings()
        _embeddings_instance = FastEmbedEmbeddings(
            model_name=settings.local_embedding_model_name,
            cache_dir=cache_dir,
        )
    return _embeddings_instance
