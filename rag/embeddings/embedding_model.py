"""Sentence-transformer embeddings wrapped for LangChain."""

import logging
import sys

from config import EMBEDDING_MODEL

logger = logging.getLogger(__name__)


class EmbeddingError(Exception):
    """Raised when the embedding model cannot be loaded or used."""


def get_embeddings(model_name: str = EMBEDDING_MODEL):
    """Load all-MiniLM-L6-v2 on CPU with normalized vectors (cosine-ready)."""
    try:
        from langchain_huggingface import HuggingFaceEmbeddings
    except ImportError:  # older installs
        logger.warning("langchain_huggingface not importable in %s; trying langchain_community", sys.executable,
                       exc_info=True)
        try:
            from langchain_community.embeddings import HuggingFaceEmbeddings
        except ImportError as exc:
            logger.exception("No HuggingFace embeddings class could be imported (python: %s)", sys.executable)
            raise EmbeddingError(
                f"Embedding packages are missing in {sys.executable} ({type(exc).__name__}: {exc}). "
                "Run: pip install -r requirements.txt"
            ) from exc
    try:
        return HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )
    except Exception as exc:
        logger.exception("Failed to load embedding model")
        raise EmbeddingError(
            f"Could not load embedding model '{model_name}'. The first run downloads it from Hugging Face, "
            f"so check your internet connection. Details: {exc}"
        ) from exc
