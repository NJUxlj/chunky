"""Connectivity tests for LLM and Embedding services."""

from __future__ import annotations

# CRITICAL: Set HF_ENDPOINT BEFORE importing sentence_transformers
from chunky.utils.hf_setup import HF_ENDPOINT, ensure_hf_endpoint

import logging

import httpx

from chunky.config.settings import ChunkyConfig, EmbeddingConfig, LLMConfig, RerankerConfig

logger = logging.getLogger(__name__)


class ConnectivityTestResult:
    """Result of a connectivity test."""

    def __init__(self, name: str, success: bool, message: str = "") -> None:
        self.name = name
        self.success = success
        self.message = message

    def __repr__(self) -> str:
        status = "✅" if self.success else "❌"
        return f"{status} {self.name}: {self.message}"


def test_llm_connectivity(config: LLMConfig) -> ConnectivityTestResult:
    """Test connectivity to the LLM service.

    Returns a ConnectivityTestResult indicating success or failure.
    """
    name = "LLM Connection"

    if not config.api_base:
        return ConnectivityTestResult(
            name=name,
            success=False,
            message="API base URL not configured",
        )

    if not config.model:
        return ConnectivityTestResult(
            name=name,
            success=False,
            message="Model name not configured",
        )

    try:
        import openai

        client = openai.OpenAI(
            base_url=config.api_base or None,
            api_key=config.api_key or "EMPTY",
            timeout=30.0,
        )

        response = client.chat.completions.create(
            model=config.model,
            messages=[{"role": "user", "content": "Hi"}],
            max_tokens=5,
        )

        if response and response.choices:
            return ConnectivityTestResult(
                name=name,
                success=True,
                message=f"Connected to {config.model}",
            )
        else:
            return ConnectivityTestResult(
                name=name,
                success=False,
                message="Unexpected response format",
            )

    except Exception as e:
        error_msg = str(e)
        # Provide helpful error messages for common issues
        if "Connection" in error_msg or "connect" in error_msg.lower():
            hint = "Please check if the API base URL is correct and the server is running."
        elif "401" in error_msg or "Authentication" in error_msg:
            hint = "Please check your API key."
        elif "404" in error_msg or "Not Found" in error_msg:
            hint = "Please check if the model name is correct."
        elif "timeout" in error_msg.lower():
            hint = "Connection timed out. Please try again."
        else:
            hint = f"Error: {error_msg}"

        return ConnectivityTestResult(
            name=name,
            success=False,
            message=hint,
        )


def _is_valid_model_dir(path: Path) -> bool:
    """Check if a directory contains a valid model."""
    if not (path / "config.json").exists():
        return False
    model_files = list(path.glob("*.bin")) + list(path.glob("*.safetensors"))
    return len(model_files) > 0


def _find_cached_model_fuzzy(model_id: str, cache_dir: Path) -> Path | None:
    """Find a cached model using fuzzy matching (fast version without network checks).
    
    Matching priority:
    1. Exact match: model_id (e.g., "Qwen/Qwen3-Embedding-0.6B")
    2. Short name match: last part after "/" (e.g., "Qwen3-Embedding-0.6B")
    3. Case-insensitive match of both above
    """
    if not cache_dir.exists():
        return None
    
    # Extract short name (part after "/")
    short_name = model_id.split("/")[-1] if "/" in model_id else model_id
    
    candidates = []
    
    for cache_path in cache_dir.iterdir():
        if not cache_path.is_dir():
            continue
        
        cache_name = cache_path.name
        cache_name_lower = cache_name.lower()
        model_id_lower = model_id.lower().replace("/", "--")
        short_name_lower = short_name.lower()
        
        # Check for exact match
        if cache_name_lower == model_id_lower:
            candidates.append((cache_path, 1))
        # Check for short name match
        elif short_name_lower in cache_name_lower or cache_name_lower.endswith(short_name_lower):
            candidates.append((cache_path, 2))
    
    if candidates:
        candidates.sort(key=lambda x: x[1])
        best_match = candidates[0][0]
        if _is_valid_model_dir(best_match):
            return best_match
    
    return None


def _resolve_model_path(config_model_name: str, config_local_path: str | None) -> str:
    """Resolve the actual model path to use.
    
    Priority:
    1. If config.local_model_path is set and valid, use it
    2. Search cache for model using fuzzy matching
    3. Fall back to model_name (will trigger download)
    """
    from pathlib import Path
    
    # 1. Check user-specified local_model_path
    if config_local_path:
        local_path = Path(config_local_path)
        if local_path.exists() and _is_valid_model_dir(local_path):
            return str(local_path)
    
    # 2. Search cache using fuzzy matching (fast, no network calls)
    cache_dir = Path.home() / ".cache" / "chunky" / "models"
    cached_path = _find_cached_model_fuzzy(config_model_name, cache_dir)
    if cached_path:
        return str(cached_path)
    
    # 3. Fall back to model_name (will trigger HF download)
    return config_model_name


def test_embedding_connectivity(config: EmbeddingConfig) -> ConnectivityTestResult:
    """Test connectivity to the Embedding service.

    Returns a ConnectivityTestResult indicating success or failure.
    """
    name = "Embedding Connection"

    if config.api_type == "sentence_transformers":
        # Local model - use cached path if available
        try:
            from sentence_transformers import SentenceTransformer
            
            # Resolve the actual model path
            model_path = _resolve_model_path(config.model_name, config.local_model_path)
            
            model = SentenceTransformer(model_path, device=config.device)
            dim = model.get_sentence_embedding_dimension()
            
            if model_path == config.model_name:
                msg = f"Local model loaded: {config.model_name} (dim={dim})"
            else:
                msg = f"Local model loaded from cache: {model_path} (dim={dim})"
            
            return ConnectivityTestResult(
                name=name,
                success=True,
                message=msg,
            )
        except Exception as e:
            return ConnectivityTestResult(
                name=name,
                success=False,
                message=f"Failed to load local model: {e}",
            )

    elif config.api_type in ("openai", "vllm"):
        # API-based embedding
        if not config.api_base:
            return ConnectivityTestResult(
                name=name,
                success=False,
                message="API base URL not configured",
            )

        if not config.api_key and config.api_type == "openai":
            return ConnectivityTestResult(
                name=name,
                success=False,
                message="API key not configured",
            )

        try:
            headers = {
                "Authorization": f"Bearer {config.api_key}",
                "Content-Type": "application/json",
            }

            payload = {
                "input": ["test"],
                "model": config.model_name,
            }

            # Ensure api_base has a scheme
            api_base = config.api_base
            if not api_base.startswith("http://") and not api_base.startswith("https://"):
                api_base = "http://" + api_base

            if config.api_type == "vllm":
                api_url = f"{api_base.rstrip('/')}/v1/embeddings"
            else:
                api_url = f"{api_base.rstrip('/')}/embeddings"

            with httpx.Client(timeout=30.0) as client:
                response = client.post(api_url, json=payload, headers=headers)
                response.raise_for_status()
                result = response.json()

            if "data" in result and len(result["data"]) > 0:
                dim = len(result["data"][0].get("embedding", []))
                return ConnectivityTestResult(
                    name=name,
                    success=True,
                    message=f"Connected to {config.api_type.upper()} API (dim={dim})",
                )
            else:
                return ConnectivityTestResult(
                    name=name,
                    success=False,
                    message="Unexpected response format",
                )

        except Exception as e:
            error_msg = str(e)
            if "Connection" in error_msg or "connect" in error_msg.lower():
                hint = "Please check if the API base URL is correct and the server is running."
            elif "401" in error_msg or "403" in error_msg:
                hint = "Please check your API key."
            elif "404" in error_msg or "Not Found" in error_msg:
                hint = "Please check if the model name is correct."
            elif "timeout" in error_msg.lower():
                hint = "Connection timed out. Please try again."
            else:
                hint = f"Error: {error_msg}"

            return ConnectivityTestResult(
                name=name,
                success=False,
                message=hint,
            )

    else:
        return ConnectivityTestResult(
            name=name,
            success=False,
            message=f"Unknown API type: {config.api_type}",
        )


def test_reranker_connectivity(config: RerankerConfig) -> ConnectivityTestResult:
    """Test connectivity to the Reranker service.

    Returns a ConnectivityTestResult indicating success or failure.
    """
    name = "Reranker Connection"

    if not config.model_name:
        return ConnectivityTestResult(
            name=name,
            success=True,
            message="No reranker model configured (optional)",
        )

    if config.api_type == "local":
        # Local model - use cached path if available
        try:
            from sentence_transformers import CrossEncoder
            
            # Resolve the actual model path
            model_path = _resolve_model_path(config.model_name, config.local_model_path)
            
            model = CrossEncoder(model_path, device=config.device)
            
            if model_path == config.model_name:
                msg = f"Local reranker loaded: {config.model_name}"
            else:
                msg = f"Local reranker loaded from cache: {model_path}"
            
            return ConnectivityTestResult(
                name=name,
                success=True,
                message=msg,
            )
        except Exception as e:
            return ConnectivityTestResult(
                name=name,
                success=False,
                message=f"Failed to load local reranker: {e}",
            )

    elif config.api_type in ("vllm", "openai"):
        # API-based reranker
        if not config.api_base:
            return ConnectivityTestResult(
                name=name,
                success=False,
                message="API base URL not configured",
            )

        try:
            headers = {
                "Authorization": f"Bearer {config.api_key or 'EMPTY'}",
                "Content-Type": "application/json",
            }

            payload = {
                "model": config.model_name,
                "query": "test query",
                "documents": ["test document"],
                "top_n": 1,
            }

            # Ensure api_base has a scheme
            api_base = config.api_base
            if not api_base.startswith("http://") and not api_base.startswith("https://"):
                api_base = "http://" + api_base

            api_url = f"{api_base.rstrip('/')}/v1/rerank"

            with httpx.Client(timeout=30.0) as client:
                response = client.post(api_url, json=payload, headers=headers)
                response.raise_for_status()
                result = response.json()

            if "results" in result and len(result["results"]) > 0:
                return ConnectivityTestResult(
                    name=name,
                    success=True,
                    message=f"Connected to {config.api_type.upper()} reranker API",
                )
            else:
                return ConnectivityTestResult(
                    name=name,
                    success=False,
                    message="Unexpected response format",
                )

        except Exception as e:
            error_msg = str(e)
            if "Connection" in error_msg or "connect" in error_msg.lower():
                hint = "Please check if the API base URL is correct and the server is running."
            elif "401" in error_msg or "403" in error_msg:
                hint = "Please check your API key."
            elif "404" in error_msg or "Not Found" in error_msg:
                hint = "Please check if the model name is correct."
            elif "timeout" in error_msg.lower():
                hint = "Connection timed out. Please try again."
            else:
                hint = f"Error: {error_msg}"

            return ConnectivityTestResult(
                name=name,
                success=False,
                message=hint,
            )

    else:
        return ConnectivityTestResult(
            name=name,
            success=False,
            message=f"Unknown reranker API type: {config.api_type}",
        )


def run_connectivity_tests(config: ChunkyConfig) -> list[ConnectivityTestResult]:
    """Run connectivity tests for LLM, Embedding, and Reranker services.

    In test mode (--test):
    - LLM, Embedding, and Reranker use test implementations, so tests are skipped.

    Args:
        config: The ChunkyConfig to test.

    Returns:
        A list of ConnectivityTestResult for each test.
    """
    results = []

    # In test mode, use test implementations
    if config.test_mode:
        results.append(ConnectivityTestResult(
            name="LLM Connection",
            success=True,
            message="Test mode: using bag-of-words (skipped)",
        ))
        results.append(ConnectivityTestResult(
            name="Embedding Connection",
            success=True,
            message="Test mode: using bag-of-words (skipped)",
        ))
        results.append(ConnectivityTestResult(
            name="Reranker Connection",
            success=True,
            message="Test mode: using test reranker (skipped)",
        ))
    else:
        # Non-test mode: check all
        results.append(test_llm_connectivity(config.llm))
        results.append(test_embedding_connectivity(config.embedding))
        results.append(test_reranker_connectivity(config.reranker))

    return results
