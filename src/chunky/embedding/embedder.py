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
