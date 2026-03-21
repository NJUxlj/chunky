"""Hugging Face setup - MUST be imported before any HF-related imports.

This module ensures HF_ENDPOINT is set correctly BEFORE importing huggingface_hub
or sentence_transformers.
"""

from __future__ import annotations

import os
import logging

logger = logging.getLogger(__name__)

# Hugging Face mirror endpoints to try
HF_ENDPOINTS = [
    "https://hf-mirror.com",  # China mirror (preferred in China)
    "https://huggingface.co",  # Official endpoint
]


def _test_endpoint(url: str, timeout: float = 5.0) -> bool:
    """Test if an endpoint is reachable."""
    try:
        import socket
        import httpx
        
        # Extract host from URL
        parsed = httpx.URL(url)
        host = parsed.host
        
        # Try DNS resolution first (fast)
        socket.gethostbyname(host)
        
        # Try HTTP GET request
        with httpx.Client(timeout=timeout, follow_redirects=True) as client:
            response = client.get(url)
            return response.status_code < 500
    except Exception as e:
        logger.debug(f"Endpoint {url} unreachable: {e}")
        return False


def _get_available_hf_endpoint() -> str:
    """Get the first available Hugging Face endpoint."""
    # First, check if user has set HF_ENDPOINT explicitly
    env_endpoint = os.environ.get("HF_ENDPOINT", "").strip()
    if env_endpoint:
        logger.info(f"Using HF_ENDPOINT from environment: {env_endpoint}")
        return env_endpoint
    
    # Test each endpoint
    for endpoint in HF_ENDPOINTS:
        logger.debug(f"Testing endpoint: {endpoint}")
        if _test_endpoint(endpoint):
            logger.info(f"Using available endpoint: {endpoint}")
            return endpoint
    
    # If none available, default to huggingface.co
    logger.warning("No Hugging Face endpoint reachable, defaulting to huggingface.co")
    return "https://huggingface.co"


def ensure_hf_endpoint() -> str:
    """Ensure HF_ENDPOINT is set in environment.
    
    This function MUST be called BEFORE importing huggingface_hub
    or sentence_transformers.
    
    Returns:
        The endpoint URL being used
    """
    # If already set and valid, return it
    current = os.environ.get("HF_ENDPOINT", "").strip()
    if current and current.startswith("http"):
        return current
    
    # Find available endpoint
    endpoint = _get_available_hf_endpoint()
    
    # Set environment variable
    os.environ["HF_ENDPOINT"] = endpoint
    
    # Also set HUGGINGFACE_HUB_CONFIG if needed
    os.environ["HUGGINGFACE_HUB_NO_SYMLINKS"] = "1"
    
    logger.info(f"HF_ENDPOINT set to: {endpoint}")
    return endpoint


# Auto-setup when this module is imported
HF_ENDPOINT = ensure_hf_endpoint()
