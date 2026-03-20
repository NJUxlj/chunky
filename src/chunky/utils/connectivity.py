"""Connectivity tests for LLM and Embedding services."""

from __future__ import annotations

import logging

import httpx

from chunky.config.settings import ChunkyConfig, EmbeddingConfig, LLMConfig

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


def test_embedding_connectivity(config: EmbeddingConfig) -> ConnectivityTestResult:
    """Test connectivity to the Embedding service.

    Returns a ConnectivityTestResult indicating success or failure.
    """
    name = "Embedding Connection"

    if config.api_type == "sentence_transformers":
        # Local model - just check if it can be loaded
        try:
            from sentence_transformers import SentenceTransformer

            model = SentenceTransformer(config.model_name, device=config.device)
            dim = model.get_sentence_embedding_dimension()
            return ConnectivityTestResult(
                name=name,
                success=True,
                message=f"Local model loaded: {config.model_name} (dim={dim})",
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


def run_connectivity_tests(config: ChunkyConfig) -> list[ConnectivityTestResult]:
    """Run connectivity tests for LLM and Embedding services.

    In test mode (--test):
    - Both LLM and Embedding use bag-of-words, so both tests are skipped.

    Args:
        config: The ChunkyConfig to test.

    Returns:
        A list of ConnectivityTestResult for each test.
    """
    results = []

    # In test mode, both use bag-of-words, so skip both tests
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
    else:
        # Non-test mode: check both
        results.append(test_llm_connectivity(config.llm))
        results.append(test_embedding_connectivity(config.embedding))

    return results
