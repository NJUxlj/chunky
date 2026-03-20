"""Embedding generation for chunks.

Provides class-based embedders (SentenceTransformer and Bag-of-Words) along
with the legacy functional helpers retained for backward compatibility with
the pipeline runner.
"""

from __future__ import annotations

import logging

import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer

from chunky.config.settings import EmbeddingConfig
from chunky.utils.models import Chunk

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# SentenceTransformer-based embedder
# ---------------------------------------------------------------------------

class SentenceTransformerEmbedder:
    """Dense embedder backed by ``sentence-transformers``.

    Parameters
    ----------
    config : EmbeddingConfig
        Embedding configuration carrying the model name, target device and
        batch size.
    """

    def __init__(self, config: EmbeddingConfig) -> None:
        self.config = config

        from sentence_transformers import SentenceTransformer

        logger.info(
            "Loading SentenceTransformer model: %s (device=%s)",
            config.model_name,
            config.device,
        )
        self._model = SentenceTransformer(config.model_name, device=config.device)

    # -- public API ---------------------------------------------------------

    def embed(self, chunks: list[Chunk]) -> list[Chunk]:
        """Embed all *chunks* in batches, setting ``chunk.embedding``.

        Returns the same list of chunks with embeddings populated.
        """
        if not chunks:
            return chunks

        texts = [c.text for c in chunks]
        logger.info(
            "Encoding %d chunks (batch_size=%d)",
            len(texts),
            self.config.batch_size,
        )
        embeddings = self._model.encode(
            texts,
            batch_size=self.config.batch_size,
            show_progress_bar=False,
        )

        for i, chunk in enumerate(chunks):
            chunk.embedding = embeddings[i].tolist()

        logger.info(
            "Embedding complete, dim=%d",
            len(chunks[0].embedding) if chunks else 0,
        )
        return chunks

    def get_dim(self) -> int:
        """Return the dimensionality of the embedding vectors."""
        return self._model.get_sentence_embedding_dimension()


# ---------------------------------------------------------------------------
# Bag-of-Words (TF-IDF + SVD) embedder — lightweight, no model download
# ---------------------------------------------------------------------------

class BagOfWordsEmbedder:
    """Lightweight dense embedder using TF-IDF followed by Truncated SVD.

    Suitable for test / offline scenarios where downloading a full
    transformer model is not desired.

    Parameters
    ----------
    dim : int
        Target dimensionality for the output embeddings.
    """

    def __init__(self, dim: int = 128) -> None:
        self.dim = dim

    # -- public API ---------------------------------------------------------

    def embed(self, chunks: list[Chunk]) -> list[Chunk]:
        """Embed all *chunks* using TF-IDF + TruncatedSVD.

        The TF-IDF matrix is reduced to ``self.dim`` dimensions via SVD.
        If the corpus is smaller than the requested dimensionality the
        actual number of components is clamped accordingly.

        Returns the same list of chunks with embeddings populated.
        """
        if not chunks:
            return chunks

        texts = [c.text for c in chunks]

        vectorizer = TfidfVectorizer()
        tfidf_matrix = vectorizer.fit_transform(texts)

        # TruncatedSVD requires n_components < min(n_samples, n_features).
        max_components = min(tfidf_matrix.shape[0], tfidf_matrix.shape[1]) - 1
        actual_dim = max(1, min(self.dim, max_components))

        svd = TruncatedSVD(n_components=actual_dim, random_state=42)
        reduced = svd.fit_transform(tfidf_matrix)  # (n_docs, actual_dim)

        # If actual_dim < self.dim we pad with zeros so every vector has
        # exactly self.dim elements, keeping the interface predictable.
        if actual_dim < self.dim:
            padding = np.zeros((reduced.shape[0], self.dim - actual_dim))
            reduced = np.hstack([reduced, padding])

        for i, chunk in enumerate(chunks):
            chunk.embedding = reduced[i].tolist()

        logger.info(
            "BagOfWords embedding complete: %d chunks, dim=%d",
            len(chunks),
            len(chunks[0].embedding) if chunks else 0,
        )
        return chunks

    def get_dim(self) -> int:
        """Return the configured embedding dimensionality."""
        return self.dim


# ---------------------------------------------------------------------------
# API-based embedder — OpenAI-compatible embedding API
# ---------------------------------------------------------------------------

class APIEmbedder:
    """Embedder that calls an OpenAI-compatible embedding API.

    Requires ``api_base`` and ``api_key`` to be set in the config.
    Falls back to SentenceTransformerEmbedder if api_base is not set.

    Parameters
    ----------
    config : EmbeddingConfig
        Embedding configuration with api_base, api_key, and model_name.
    """

    def __init__(self, config: EmbeddingConfig) -> None:
        self.config = config

        if not config.api_base or not config.api_key:
            # Fall back to local model
            logger.info("API base/key not set, falling back to SentenceTransformer")
            self._fallback = SentenceTransformerEmbedder(config)
        else:
            self._fallback = None
            logger.info(
                "Using API embedder: %s (model=%s)",
                config.api_base,
                config.model_name,
            )

    def embed(self, chunks: list[Chunk]) -> list[Chunk]:
        """Embed all *chunks* via API call."""
        if not chunks:
            return chunks

        if self._fallback is not None:
            return self._fallback.embed(chunks)

        import httpx

        headers = {
            "Authorization": f"Bearer {self.config.api_key}",
            "Content-Type": "application/json",
        }

        texts = [c.text for c in chunks]
        payload = {
            "input": texts,
            "model": self.config.model_name,
        }

        api_url = f"{self.config.api_base.rstrip('/')}/embeddings"

        try:
            with httpx.Client(timeout=60.0) as client:
                response = client.post(api_url, json=payload, headers=headers)
                response.raise_for_status()
                result = response.json()

            embeddings = result["data"]
            # Sort by index to maintain order
            embeddings_sorted = sorted(embeddings, key=lambda x: x["index"])

            for i, chunk in enumerate(chunks):
                chunk.embedding = embeddings_sorted[i]["embedding"]

            dim = len(embeddings_sorted[0]["embedding"]) if embeddings_sorted else 0
            logger.info(
                "API embedding complete: %d chunks, dim=%d",
                len(chunks),
                dim,
            )
        except Exception as e:
            logger.error("API embedding failed: %s, falling back to local model", e)
            self._fallback = SentenceTransformerEmbedder(self.config)
            return self._fallback.embed(chunks)

        return chunks

    def get_dim(self) -> int:
        """Return the dimensionality of the embedding vectors.

        Note: This is an approximation; actual dimension is known after
        the first API call. Returns 0 to indicate unknown.
        """
        return 0


# ---------------------------------------------------------------------------
# Legacy functional API (kept for backward compatibility with the pipeline)
# ---------------------------------------------------------------------------

def embed_chunks(chunks: list[Chunk], config: EmbeddingConfig) -> list[Chunk]:
    """Generate embeddings for chunks using sentence-transformers.

    Thin wrapper around :class:`SentenceTransformerEmbedder` that mirrors the
    original functional interface used by the pipeline runner.

    Args:
        chunks: List of Chunk objects.
        config: Embedding configuration (model_name, device, batch_size).

    Returns:
        The same list of chunks with embeddings populated.
    """
    embedder = SentenceTransformerEmbedder(config)
    return embedder.embed(chunks)


def embed_chunks_test(chunks: list[Chunk], dim: int = 128) -> list[Chunk]:
    """Generate lightweight TF-IDF + SVD embeddings for test mode.

    Thin wrapper around :class:`BagOfWordsEmbedder`.

    Args:
        chunks: List of Chunk objects.
        dim: Target dimensionality of the output vectors.

    Returns:
        The same list of chunks with embeddings populated.
    """
    embedder = BagOfWordsEmbedder(dim=dim)
    return embedder.embed(chunks)
