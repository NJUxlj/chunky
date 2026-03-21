"""Reranker implementations for chunky.

Supports:
- Local reranker using sentence-transformers CrossEncoder
- API-based reranker using vLLM / OpenAI-compatible rerank API
"""

from __future__ import annotations

# CRITICAL: Set HF_ENDPOINT BEFORE importing sentence_transformers
from chunky.utils.hf_setup import HF_ENDPOINT, ensure_hf_endpoint

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import httpx
import numpy as np

from chunky.config.settings import RerankerConfig

logger = logging.getLogger(__name__)


@dataclass
class RerankResult:
    """Result of reranking a single document."""
    index: int  # Original index in the input list
    text: str  # Document text
    score: float  # Relevance score (higher is better)


class Reranker(ABC):
    """Abstract base class for rerankers."""

    @abstractmethod
    def rerank(
        self,
        query: str,
        documents: list[str],
        top_k: int | None = None,
    ) -> list[RerankResult]:
        """Rerank documents based on relevance to query.

        Args:
            query: The search query.
            documents: List of document texts to rerank.
            top_k: Number of top results to return. If None, return all.

        Returns:
            List of RerankResult sorted by relevance score (descending).
        """
        ...


class LocalReranker(Reranker):
    """Local reranker using sentence-transformers CrossEncoder.

    Supports loading from local path or downloading from Hugging Face.
    Auto-detects available HF endpoint (hf-mirror.com or huggingface.co).
    """

    def __init__(self, config: RerankerConfig) -> None:
        self.config = config
        self._model = None

        # Lazy import to avoid hard dependency
        try:
            # HF_ENDPOINT is already set by hf_setup module import
            logger.info(f"Using Hugging Face endpoint: {HF_ENDPOINT}")
            
            from sentence_transformers import CrossEncoder

            if config.model_name:
                model_path = self._resolve_model_path()
                logger.info("Loading CrossEncoder model: %s", model_path)
                self._model = CrossEncoder(model_path, device=config.device)
            else:
                logger.warning("No model_name specified for LocalReranker")
        except ImportError:
            logger.error(
                "sentence-transformers not installed. "
                "Install with: pip install sentence-transformers"
            )
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

    def rerank(
        self,
        query: str,
        documents: list[str],
        top_k: int | None = None,
    ) -> list[RerankResult]:
        """Rerank documents using local CrossEncoder model."""
        if not documents:
            return []

        if self._model is None:
            logger.error("CrossEncoder model not loaded")
            return [RerankResult(index=i, text=doc, score=0.0) for i, doc in enumerate(documents)]

        # Create query-document pairs
        pairs = [[query, doc] for doc in documents]

        # Get scores from cross-encoder
        logger.info("Reranking %d documents with local model", len(documents))
        scores = self._model.predict(pairs)

        # Build results
        results = [
            RerankResult(index=i, text=doc, score=float(scores[i]))
            for i, doc in enumerate(documents)
        ]

        # Sort by score (descending)
        results.sort(key=lambda x: x.score, reverse=True)

        # Return top_k if specified
        if top_k is not None:
            results = results[:top_k]

        logger.info("Reranking complete, returning %d results", len(results))
        return results


class APIReranker(Reranker):
    """API-based reranker using vLLM or OpenAI-compatible rerank API.

    Compatible with:
    - vLLM reranker API (/v1/rerank)
    - Jina AI rerank API
    - Cohere rerank API

    vLLM reranker server example:
        vllm serve BAAI/bge-reranker-v2-m3 --runner pooling
    """

    def __init__(self, config: RerankerConfig) -> None:
        self.config = config

        if not config.api_base:
            raise ValueError("api_base is required for APIReranker")
        if not config.model_name:
            raise ValueError("model_name is required for APIReranker")

        self.api_base = config.api_base.rstrip("/")
        self.api_key = config.api_key or "EMPTY"

        logger.info(
            "Initialized APIReranker (type=%s, model=%s, base=%s)",
            config.api_type,
            config.model_name,
            self.api_base,
        )

    def _make_request(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Make HTTP request to rerank API."""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        api_url = f"{self.api_base}/v1/rerank"

        try:
            with httpx.Client(timeout=60.0) as client:
                response = client.post(api_url, json=payload, headers=headers)
                response.raise_for_status()
                return response.json()
        except httpx.HTTPStatusError as e:
            logger.error("HTTP error from rerank API: %s - %s", e.response.status_code, e.response.text)
            raise
        except Exception as e:
            logger.error("Failed to call rerank API: %s", e)
            raise

    def rerank(
        self,
        query: str,
        documents: list[str],
        top_k: int | None = None,
    ) -> list[RerankResult]:
        """Rerank documents using vLLM/OpenAI-compatible rerank API.

        API Format (vLLM/Jina AI/Cohere compatible):
        {
            "model": "BAAI/bge-reranker-base",
            "query": "What is the capital of France?",
            "documents": ["doc1", "doc2", ...],
            "top_n": 10
        }

        Response Format:
        {
            "results": [
                {
                    "index": 1,
                    "document": {"text": "..."},
                    "relevance_score": 0.998
                },
                ...
            ]
        }
        """
        if not documents:
            return []

        # Build request payload
        payload = {
            "model": self.config.model_name,
            "query": query,
            "documents": documents,
        }

        if top_k is not None:
            payload["top_n"] = top_k

        logger.info(
            "Calling %s rerank API for %d documents (model=%s)",
            self.config.api_type,
            len(documents),
            self.config.model_name,
        )

        try:
            result = self._make_request(payload)

            # Parse response
            api_results = result.get("results", [])

            # Build RerankResult list
            results = []
            for item in api_results:
                index = item.get("index", 0)
                score = item.get("relevance_score", item.get("score", 0.0))

                # Get document text (handle different API formats)
                document = item.get("document", {})
                if isinstance(document, dict):
                    text = document.get("text", documents[index] if index < len(documents) else "")
                else:
                    text = document if isinstance(document, str) else documents[index] if index < len(documents) else ""

                results.append(RerankResult(index=index, text=text, score=float(score)))

            logger.info("Reranking complete, got %d results", len(results))
            return results

        except Exception as e:
            logger.error("Rerank API call failed: %s", e)
            # Fallback: return documents in original order with zero scores
            return [RerankResult(index=i, text=doc, score=0.0) for i, doc in enumerate(documents)]


class TestReranker(Reranker):
    """Test reranker that returns documents in original order.

    Useful for testing and when no reranking is desired.
    """

    def rerank(
        self,
        query: str,
        documents: list[str],
        top_k: int | None = None,
    ) -> list[RerankResult]:
        """Return documents in original order (no actual reranking)."""
        results = [
            RerankResult(index=i, text=doc, score=1.0 - (i * 0.01))  # Slight descending score
            for i, doc in enumerate(documents)
        ]

        if top_k is not None:
            results = results[:top_k]

        return results


def get_reranker(config: RerankerConfig) -> Reranker:
    """Factory function to create appropriate reranker based on config.

    Args:
        config: RerankerConfig with api_type and other settings.

    Returns:
        Reranker instance (LocalReranker, APIReranker, or TestReranker).

    Raises:
        ValueError: If api_type is not recognized.
    """
    api_type = config.api_type.lower()

    if api_type in ("vllm", "openai", "api"):
        return APIReranker(config)
    elif api_type == "local":
        if not config.model_name:
            logger.warning("No reranker model specified, using TestReranker")
            return TestReranker()
        return LocalReranker(config)
    elif api_type == "test" or not config.model_name:
        return TestReranker()
    else:
        raise ValueError(f"Unknown reranker api_type: {config.api_type!r}")
