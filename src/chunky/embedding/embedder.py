"""Embedding generation for chunks.

Provides class-based embedders (SentenceTransformer and Bag-of-Words) along
with the legacy functional helpers retained for backward compatibility with
the pipeline runner.
"""

from __future__ import annotations

# CRITICAL: Set HF_ENDPOINT BEFORE importing sentence_transformers
from chunky.utils.hf_setup import HF_ENDPOINT, ensure_hf_endpoint

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

    Supports loading from local path or downloading from Hugging Face.
    Auto-detects available HF endpoint (hf-mirror.com or huggingface.co).

    Parameters
    ----------
    config : EmbeddingConfig
        Embedding configuration carrying the model name, target device and
        batch size. If local_model_path is set, will try loading from there first.
    """

    def __init__(self, config: EmbeddingConfig) -> None:
        self.config = config

        # HF_ENDPOINT is already set by hf_setup module import
        logger.info(f"Using Hugging Face endpoint: {HF_ENDPOINT}")

        from sentence_transformers import SentenceTransformer

        # Determine model to load
        model_path = self._resolve_model_path()
        
        logger.info(
            "Loading SentenceTransformer model: %s (device=%s)",
            model_path,
            config.device,
        )
        
        try:
            self._model = SentenceTransformer(model_path, device=config.device)
        except Exception as e:
            logger.error(f"Failed to load model from {model_path}: {e}")
            raise

    def _resolve_model_path(self) -> str:
        """Resolve the model path to use.
        
        Priority:
        1. Explicit local model path (if set and valid)
        2. Default cache path (check if already downloaded)
        3. Model name (download from HF)
        
        Returns:
            Path or model name to load
        """
        from chunky.utils.network import validate_model_path
        from chunky.utils.model_downloader import ModelDownloadManager
        
        local_path = self.config.local_model_path
        model_name = self.config.model_name
        
        # Priority 1: Check explicit local path
        if local_path:
            is_valid, msg = validate_model_path(local_path)
            if is_valid:
                logger.info(f"Using local model from: {local_path}")
                return local_path
            else:
                # Warning about invalid local path
                logger.warning(f"⚠️  Local model path invalid: {msg}")
                logger.warning(f"⚠️  Falling back to cache/default path")
        
        # Priority 2: Check default cache path
        cache_manager = ModelDownloadManager()
        cache_path = cache_manager.get_model_cache_path(model_name)
        
        if cache_manager.is_model_cached(model_name):
            logger.info(f"Using cached model from: {cache_path}")
            return str(cache_path)
        
        # Priority 3: Download from HF
        logger.info(f"Model not in cache, will download: {model_name}")
        return model_name

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
# API-based embedder — OpenAI-compatible / vLLM embedding API
# ---------------------------------------------------------------------------

class APIEmbedder:
    """Embedder that calls an OpenAI-compatible or vLLM embedding API.

    Supports three modes:
    - sentence_transformers: Local model (default fallback)
    - openai: OpenAI-compatible API (e.g., text-embedding-3-small)
    - vllm: vLLM server (e.g., e5-mistral-7b-instruct)

    Parameters
    ----------
    config : EmbeddingConfig
        Embedding configuration with api_base, api_key, model_name, and api_type.
    """

    def __init__(self, config: EmbeddingConfig) -> None:
        self.config = config
        self._fallback = None

        # Determine which embedder to use based on api_type
        if config.api_type == "vllm" or config.api_type == "openai":
            if not config.api_base:
                logger.warning("API base not set, falling back to SentenceTransformer")
                self._fallback = SentenceTransformerEmbedder(config)
            else:
                logger.info(
                    "Using %s embedder: %s (model=%s)",
                    config.api_type.upper(),
                    config.api_base,
                    config.model_name,
                )
        else:
            # sentence_transformers or unknown
            logger.info("Using SentenceTransformer embedder")
            self._fallback = SentenceTransformerEmbedder(config)

    def embed(self, chunks: list[Chunk]) -> list[Chunk]:
        """Embed all *chunks* via API call."""
        if not chunks:
            return chunks

        if self._fallback is not None:
            return self._fallback.embed(chunks)

        import httpx

        texts = [c.text for c in chunks]

        if self.config.api_type == "vllm":
            return self._embed_vllm(chunks, texts)
        else:
            return self._embed_openai(chunks, texts)

    def _embed_openai(self, chunks: list[Chunk], texts: list[str]) -> list[Chunk]:
        """Embed using OpenAI-compatible API."""
        headers = {
            "Authorization": f"Bearer {self.config.api_key}",
            "Content-Type": "application/json",
        }

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
            embeddings_sorted = sorted(embeddings, key=lambda x: x["index"])

            for i, chunk in enumerate(chunks):
                chunk.embedding = embeddings_sorted[i]["embedding"]

            dim = len(embeddings_sorted[0]["embedding"]) if embeddings_sorted else 0
            logger.info(
                "OpenAI embedding complete: %d chunks, dim=%d",
                len(chunks),
                dim,
            )
        except Exception as e:
            logger.error("OpenAI embedding failed: %s, falling back to local model", e)
            self._fallback = SentenceTransformerEmbedder(self.config)
            return self._fallback.embed(chunks)

        return chunks

    def _embed_vllm(self, chunks: list[Chunk], texts: list[str]) -> list[Chunk]:
        """Embed using vLLM server API.

        vLLM embedding API format:
        POST /v1/embeddings
        {
            "input": ["text1", "text2"],
            "model": "e5-mistral-7b-instruct"
        }

        Response:
        {
            "data": [{"embedding": [...], "index": 0}],
            "model": "...",
            "usage": {...}
        }
        """
        headers = {
            "Authorization": f"Bearer {self.config.api_key}",
            "Content-Type": "application/json",
        }

        payload = {
            "input": texts,
            "model": self.config.model_name,
        }

        api_url = f"{self.config.api_base.rstrip('/')}/v1/embeddings"

        try:
            with httpx.Client(timeout=120.0) as client:
                response = client.post(api_url, json=payload, headers=headers)
                response.raise_for_status()
                result = response.json()

            embeddings = result.get("data", [])
            embeddings_sorted = sorted(embeddings, key=lambda x: x.get("index", 0))

            for i, chunk in enumerate(chunks):
                embedding_data = embeddings_sorted[i] if i < len(embeddings_sorted) else None
                if embedding_data and "embedding" in embedding_data:
                    chunk.embedding = embedding_data["embedding"]
                else:
                    logger.warning("No embedding found for chunk %d", i)
                    chunk.embedding = []

            dim = len(embeddings_sorted[0]["embedding"]) if embeddings_sorted and "embedding" in embeddings_sorted[0] else 0
            logger.info(
                "vLLM embedding complete: %d chunks, dim=%d",
                len(chunks),
                dim,
            )
        except Exception as e:
            logger.error("vLLM embedding failed: %s, falling back to local model", e)
            self._fallback = SentenceTransformerEmbedder(self.config)
            return self._fallback.embed(chunks)

        return chunks

    def get_dim(self) -> int:
        """Return the dimensionality of the embedding vectors.

        Returns 0 to indicate dimension is unknown until first API call.
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
