"""Network utilities for detecting available services."""

from __future__ import annotations

import logging
import os
import socket
from typing import Optional

import httpx

logger = logging.getLogger(__name__)


# Hugging Face mirror endpoints to try
HF_ENDPOINTS = [
    "https://hf-mirror.com",  # China mirror (preferred in China)
    "https://huggingface.co",  # Official endpoint
]


def test_endpoint(url: str, timeout: float = 5.0) -> bool:
    """Test if an endpoint is reachable.
    
    Args:
        url: URL to test (e.g., https://huggingface.co)
        timeout: Timeout in seconds
        
    Returns:
        True if endpoint is reachable, False otherwise
    """
    try:
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


def get_available_hf_endpoint() -> str:
    """Get the first available Hugging Face endpoint.
    
    Checks endpoints in order:
    1. Environment variable HF_ENDPOINT (if set)
    2. hf-mirror.com (China mirror)
    3. huggingface.co (official)
    
    Returns:
        URL of available endpoint
    """
    # First, check if user has set HF_ENDPOINT explicitly
    env_endpoint = os.environ.get("HF_ENDPOINT", "").strip()
    if env_endpoint:
        logger.info(f"Using HF_ENDPOINT from environment: {env_endpoint}")
        return env_endpoint
    
    # Test each endpoint
    for endpoint in HF_ENDPOINTS:
        logger.debug(f"Testing endpoint: {endpoint}")
        if test_endpoint(endpoint):
            logger.info(f"Using available endpoint: {endpoint}")
            return endpoint
    
    # If none available, default to huggingface.co
    # (it will fail later with proper error message)
    logger.warning("No Hugging Face endpoint reachable, defaulting to huggingface.co")
    return "https://huggingface.co"


def setup_hf_endpoint() -> str:
    """Setup Hugging Face endpoint for the current session.
    
    Sets HF_ENDPOINT environment variable if not already set.
    
    Returns:
        The endpoint URL being used
    """
    # If already set by user, respect it
    if "HF_ENDPOINT" in os.environ:
        return os.environ["HF_ENDPOINT"]
    
    # Find available endpoint
    endpoint = get_available_hf_endpoint()
    
    # Set environment variable for transformers/sentence-transformers
    os.environ["HF_ENDPOINT"] = endpoint
    
    return endpoint


def is_local_model_path(path: str) -> bool:
    """Check if a path looks like a local model path.
    
    Args:
        path: Path to check
        
    Returns:
        True if it exists and looks like a model directory
    """
    if not path or not os.path.exists(path):
        return False
    
    # Check if it's a directory
    if not os.path.isdir(path):
        return False
    
    # Check for common model files
    model_files = [
        "pytorch_model.bin",
        "model.safetensors",
        "config.json",
        "tokenizer_config.json",
    ]
    
    for filename in model_files:
        if os.path.exists(os.path.join(path, filename)):
            return True
    
    return False


def validate_model_path(path: str) -> tuple[bool, str]:
    """Validate a local model path.
    
    Args:
        path: Path to validate
        
    Returns:
        Tuple of (is_valid, message)
    """
    if not path:
        return False, "Path is empty"
    
    if not os.path.exists(path):
        return False, f"Path does not exist: {path}"
    
    if not os.path.isdir(path):
        return False, f"Path is not a directory: {path}"
    
    # Check for config.json (minimum requirement)
    if not os.path.exists(os.path.join(path, "config.json")):
        return False, f"No config.json found in {path}"
    
    return True, "Valid model directory"
