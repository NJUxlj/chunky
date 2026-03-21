"""Model downloader with support for multiple sources (Hugging Face & ModelScope).

Supports:
- Hugging Face (huggingface.co / hf-mirror.com)
- ModelScope (modelscope.cn) - 国内完整镜像，推荐国内用户使用
"""

from __future__ import annotations

from chunky.utils.hf_setup import HF_ENDPOINT

import logging
import os
from dataclasses import dataclass
from pathlib import Path

from rich.console import Console
from rich import print as rprint

logger = logging.getLogger(__name__)
console = Console()

# Disable symlink warnings
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"


@dataclass
class DownloadResult:
    """Result of a model download attempt."""
    name: str
    model_id: str
    success: bool
    message: str
    local_path: str | None = None


class ModelDownloadManager:
    """Manages downloading of embedding and reranker models.
    
    Supports multiple sources:
    - Hugging Face (with hf-mirror.com as fallback)
    - ModelScope (魔搭社区) - 国内完整镜像
    """
    
    def __init__(self, cache_dir: str | None = None, source: str = "auto"):
        """Initialize download manager.
        
        Args:
            cache_dir: Base directory for model cache
            source: Download source - "huggingface", "modelscope", or "auto"
        """
        if cache_dir:
            self.cache_dir = Path(cache_dir)
        else:
            self.cache_dir = Path.home() / ".cache" / "chunky" / "models"
        
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.hf_endpoint = HF_ENDPOINT
        
        # Determine source
        if source == "auto":
            # Auto-detect: use ModelScope if in China (no proxy)
            self.source = self._detect_best_source()
        else:
            self.source = source
        
        logger.info(f"Using model source: {self.source}")
    
    def _detect_best_source(self) -> str:
        """Detect the best download source based on actual download tests."""
        # Test actual HF download capability (CDN files)
        hf_works = _test_hf_download(proxy_url=None, timeout=5.0)
        
        # Test ModelScope
        ms_works = _test_modelscope_download(timeout=5.0)
        
        # Check if proxy is set and test download through it
        proxy_url, proxy_works = _get_proxy_info()
        
        if hf_works or (proxy_url and proxy_works):
            # Hugging Face CDN directly accessible or proxy works for downloads
            return "huggingface"
        elif ms_works:
            # Hugging Face blocked, but ModelScope works
            return "modelscope"
        else:
            # Both seem blocked, but default to ModelScope for China users
            logger.warning("Download tests failed for both sources, defaulting to ModelScope")
            return "modelscope"
    
    def get_model_cache_path(self, model_id: str) -> Path:
        """Get the local cache path for a model."""
        safe_name = model_id.replace("/", "--")
        return self.cache_dir / safe_name
    
    def is_model_cached(self, model_id: str) -> bool:
        """Check if a model is already cached locally."""
        cache_path = self.get_model_cache_path(model_id)
        
        if not cache_path.exists():
            return False
        
        # Check for required files
        required_files = ["config.json"]
        for filename in required_files:
            if not (cache_path / filename).exists():
                return False
        
        # Check for at least one model file
        model_files = list(cache_path.glob("*.bin")) + list(cache_path.glob("*.safetensors"))
        return len(model_files) > 0
    
    def find_cached_model(self, model_id: str) -> Path | None:
        """Find a cached model using fuzzy matching.
        
        Matching priority:
        1. Exact match: model_id (e.g., "Qwen/Qwen3-Embedding-0.6B")
        2. Short name match: last part after "/" (e.g., "Qwen3-Embedding-0.6B")
        3. Case-insensitive match of both above
        
        Returns:
            Path to cached model directory if found, None otherwise
        """
        if not self.cache_dir.exists():
            return None
        
        # Extract short name (part after "/")
        short_name = model_id.split("/")[-1] if "/" in model_id else model_id
        
        candidates = []
        
        for cache_path in self.cache_dir.iterdir():
            if not cache_path.is_dir():
                continue
            
            # Get the directory name (safe_name format: "org--model-name")
            cache_name = cache_path.name
            
            # Convert to lowercase for case-insensitive comparison
            cache_name_lower = cache_name.lower()
            model_id_lower = model_id.lower().replace("/", "--")
            short_name_lower = short_name.lower()
            
            # Check for exact match
            if cache_name_lower == model_id_lower:
                candidates.append((cache_path, 1))  # Priority 1: exact match
            # Check for short name match (e.g., "qwen3-embedding-0.6b" in "qwen--qwen3-embedding-0.6b")
            elif short_name_lower in cache_name_lower or cache_name_lower.endswith(short_name_lower):
                candidates.append((cache_path, 2))  # Priority 2: short name match
        
        # Sort by priority and return the best match
        if candidates:
            candidates.sort(key=lambda x: x[1])
            best_match = candidates[0][0]
            
            # Verify it has model files
            if self._is_valid_model_dir(best_match):
                return best_match
        
        return None
    
    def _is_valid_model_dir(self, path: Path) -> bool:
        """Check if a directory contains a valid model."""
        if not (path / "config.json").exists():
            return False
        
        model_files = list(path.glob("*.bin")) + list(path.glob("*.safetensors"))
        return len(model_files) > 0
    
    def _download_from_huggingface(self, model_id: str, cache_path: Path) -> DownloadResult:
        """Download from Hugging Face."""
        try:
            from huggingface_hub import snapshot_download
            from huggingface_hub.utils import HfHubHTTPError
            
            rprint(f"  [dim]Source: Hugging Face ({self.hf_endpoint})[/dim]")
            
            downloaded_path = snapshot_download(
                repo_id=model_id,
                local_dir=str(cache_path),
                local_dir_use_symlinks=False,
                endpoint=self.hf_endpoint,
            )
            
            return DownloadResult(
                name=model_id,
                model_id=model_id,
                success=True,
                message="Downloaded from Hugging Face",
                local_path=str(cache_path),
            )
            
        except Exception as e:
            return DownloadResult(
                name=model_id,
                model_id=model_id,
                success=False,
                message=f"Hugging Face failed: {e}",
            )
    
    def _download_from_modelscope(self, model_id: str, cache_path: Path) -> DownloadResult:
        """Download from ModelScope (魔搭社区).
        
        ModelScope is Alibaba's model hub with full China hosting.
        """
        try:
            # Try to import modelscope
            try:
                from modelscope import snapshot_download as ms_snapshot_download
            except ImportError:
                # ModelScope not installed, suggest installation
                return DownloadResult(
                    name=model_id,
                    model_id=model_id,
                    success=False,
                    message="ModelScope not installed. Run: pip install modelscope",
                )
            
            rprint(f"  [dim]Source: ModelScope (魔搭社区)[/dim]")
            
            # Map HF model ID to ModelScope if needed
            # ModelScope uses the same format: organization/model-name
            ms_model_id = model_id
            
            # Some common models have different naming
            ms_model_id = self._map_to_modelscope(ms_model_id)
            
            downloaded_path = ms_snapshot_download(
                model_id=ms_model_id,
                cache_dir=str(self.cache_dir),
                local_dir=str(cache_path),
            )
            
            return DownloadResult(
                name=model_id,
                model_id=model_id,
                success=True,
                message="Downloaded from ModelScope",
                local_path=str(cache_path),
            )
            
        except Exception as e:
            return DownloadResult(
                name=model_id,
                model_id=model_id,
                success=False,
                message=f"ModelScope failed: {e}",
            )
    
    def _map_to_modelscope(self, hf_model_id: str) -> str:
        """Map Hugging Face model ID to ModelScope equivalent."""
        # Common model mappings (HF -> ModelScope)
        mappings = {
            "BAAI/bge-small-zh-v1.5": "AI-ModelScope/bge-small-zh-v1.5",
            "BAAI/bge-base-zh-v1.5": "AI-ModelScope/bge-base-zh-v1.5",
            "BAAI/bge-large-zh-v1.5": "AI-ModelScope/bge-large-zh-v1.5",
            "BAAI/bge-reranker-base": "AI-ModelScope/bge-reranker-base",
            "BAAI/bge-reranker-large": "AI-ModelScope/bge-reranker-large",
        }
        
        return mappings.get(hf_model_id, hf_model_id)
    
    def download_model(self, model_id: str) -> DownloadResult:
        """Download a single model from the best available source."""
        cache_path = self.get_model_cache_path(model_id)
        
        # Check if already cached
        if self.is_model_cached(model_id):
            return DownloadResult(
                name=model_id,
                model_id=model_id,
                success=True,
                message="Already cached",
                local_path=str(cache_path),
            )
        
        # Show download start
        rprint(f"  [cyan]⬇ Downloading {model_id}...[/cyan]")
        rprint(f"    [dim]This may take several minutes for large models...[/dim]")
        
        # Try sources based on priority
        if self.source == "modelscope":
            # Try ModelScope first, then Hugging Face
            result = self._download_from_modelscope(model_id, cache_path)
            if not result.success:
                rprint(f"    [yellow]ModelScope failed, trying Hugging Face...[/yellow]")
                result = self._download_from_huggingface(model_id, cache_path)
        else:
            # Try Hugging Face first, then ModelScope
            result = self._download_from_huggingface(model_id, cache_path)
            if not result.success:
                rprint(f"    [yellow]Hugging Face failed, trying ModelScope...[/yellow]")
                result = self._download_from_modelscope(model_id, cache_path)
        
        if result.success:
            rprint(f"  [green]✓ {model_id} downloaded[/green]")
        else:
            rprint(f"  [red]✗ {model_id} failed[/red]")
        
        return result
    
    def download_models(self, models: list[tuple[str, str]]) -> list[DownloadResult]:
        """Download multiple models sequentially."""
        results = []
        
        for name, model_id in models:
            result = self.download_model(model_id)
            result.name = name
            results.append(result)
        
        return results


def _test_hf_download(proxy_url: str | None = None, timeout: float = 8.0) -> bool:
    """Test if Hugging Face CDN (cas-bridge.xethub.hf.co) is accessible.
    
    This is the actual domain where model files are stored.
    hf-mirror.com only proxies API, not these file downloads.
    """
    try:
        import httpx
        
        # Test URL: BGE model config.json on Hugging Face CDN
        # This is a small file (~500 bytes) that tests actual download capability
        test_url = "https://huggingface.co/BAAI/bge-small-zh-v1.5/resolve/main/config.json"
        
        client_kwargs = {"timeout": timeout, "follow_redirects": True}
        if proxy_url:
            client_kwargs["proxy"] = proxy_url
            
        with httpx.Client(**client_kwargs) as client:
            response = client.get(test_url)
            return response.status_code == 200
    except Exception:
        return False


def _test_modelscope_download(timeout: float = 8.0) -> bool:
    """Test if ModelScope is accessible for downloads."""
    try:
        import httpx
        
        # Test URL: ModelScope API endpoint
        test_url = "https://www.modelscope.cn/api/v1/models/AI-ModelScope/bge-small-zh-v1.5"
        
        with httpx.Client(timeout=timeout, follow_redirects=True) as client:
            response = client.get(test_url)
            return response.status_code == 200
    except Exception:
        return False


def _get_proxy_info() -> tuple[str | None, bool]:
    """Get proxy information and test if HF downloads work through it."""
    proxy_vars = ['HTTPS_PROXY', 'https_proxy', 'HTTP_PROXY', 'http_proxy', 'ALL_PROXY']
    
    for var in proxy_vars:
        value = os.environ.get(var, '').strip()
        if value:
            # Test if HF download works through this proxy
            is_working = _test_hf_download(proxy_url=value)
            return value, is_working
    
    return None, False


def ensure_models_downloaded(config, force_source: str = "auto") -> bool:
    """Ensure all required models are downloaded.
    
    Args:
        config: Configuration object
        force_source: Force specific source - "huggingface", "modelscope", or "auto"
    """
    from rich.prompt import Confirm
    
    # Skip if test mode
    if getattr(config, 'test_mode', False):
        return True
    
    # Check what we need
    need_embedding = (
        config.embedding.api_type == "sentence_transformers" 
        and not config.embedding.local_model_path
    )
    need_reranker = (
        config.reranker.api_type == "local" 
        and config.reranker.model_name
        and not config.reranker.local_model_path
    )
    
    if not need_embedding and not need_reranker:
        return True
    
    # Initialize manager with auto-detected or forced source
    manager = ModelDownloadManager(source=force_source)
    
    # Check cache status using fuzzy matching
    embedding_cache_path = None
    reranker_cache_path = None
    
    if need_embedding:
        # First try exact match, then fuzzy match
        if manager.is_model_cached(config.embedding.model_name):
            embedding_cache_path = manager.get_model_cache_path(config.embedding.model_name)
        else:
            embedding_cache_path = manager.find_cached_model(config.embedding.model_name)
    
    if need_reranker:
        if manager.is_model_cached(config.reranker.model_name):
            reranker_cache_path = manager.get_model_cache_path(config.reranker.model_name)
        else:
            reranker_cache_path = manager.find_cached_model(config.reranker.model_name)
    
    embedding_cached = embedding_cache_path is not None
    reranker_cached = reranker_cache_path is not None
    
    # If all cached, just update paths
    if (not need_embedding or embedding_cached) and (not need_reranker or reranker_cached):
        if need_embedding and embedding_cache_path:
            config.embedding.local_model_path = str(embedding_cache_path)
            console.print(f"[dim]Using cached embedding model: {embedding_cache_path.name}[/dim]")
        if need_reranker and reranker_cache_path:
            config.reranker.local_model_path = str(reranker_cache_path)
            console.print(f"[dim]Using cached reranker model: {reranker_cache_path.name}[/dim]")
        return True
    
    # Show what needs to be downloaded
    models_to_download = []
    if need_embedding and not embedding_cached:
        models_to_download.append(("Embedding", config.embedding.model_name))
    if need_reranker and not reranker_cached:
        models_to_download.append(("Reranker", config.reranker.model_name))
    
    console.print(f"\n[bold]Models required:[/bold]")
    for name, model_id in models_to_download:
        console.print(f"  • {name}: [cyan]{model_id}[/cyan]")
    
    console.print(f"\n[dim]Cache directory: {manager.cache_dir}[/dim]")
    
    # Show network info
    proxy_url, has_proxy = _get_proxy_info()
    proxy_status = "[green]✓[/green]" if has_proxy else "[red]✗[/red]"
    
    console.print(f"[dim]Detected source: {manager.source}[/dim]")
    if proxy_url:
        console.print(f"[dim]Proxy: {proxy_url} {proxy_status}[/dim]")
    else:
        console.print(f"[dim]Proxy: (none) {proxy_status}[/dim]")
    
    # Source selection and warnings
    if manager.source == "modelscope":
        console.print("[dim]ℹ Using ModelScope (魔搭社区) for better China connectivity[/dim]")
    else:
        # Hugging Face selected - explain the situation
        if proxy_url:
            if has_proxy:
                console.print("[dim]ℹ Proxy configured and HF download test passed[/dim]")
                console.print("[dim]  (Note: Large files may still fail if proxy times out)[/dim]")
            else:
                # Proxy is set but doesn't work for HF downloads
                console.print("\n[yellow]⚠ Your proxy is set but HF CDN download test failed[/yellow]")
                console.print(f"[dim]  Proxy {proxy_url} may not route HF file CDN traffic[/dim]")
                console.print("[dim]  Model files are hosted on cas-bridge.xethub.hf.co (US)[/dim]")
        else:
            console.print("\n[yellow]⚠ Warning: No proxy detected for Hugging Face downloads[/yellow]")
            console.print("[dim]  huggingface.co file CDN may be blocked in China[/dim]")
        
        # Always offer ModelScope as alternative
        console.print("\n[dim]Alternative:[/dim]")
        console.print("  • ModelScope (阿里云国内镜像) - works without VPN")
        
        # Always ask if user wants to switch, even if modelscope not installed
        if Confirm.ask("\nSwitch to ModelScope (魔搭社区)?", default=False):
            try:
                import modelscope
                manager = ModelDownloadManager(source="modelscope")
                console.print("[green]✓ Switched to ModelScope[/green]")
            except ImportError:
                console.print("\n[yellow]ModelScope not installed.[/yellow]")
                console.print("Install with: [cyan]pip install modelscope[/cyan]")
                if not Confirm.ask("Continue with Hugging Face anyway?", default=True):
                    console.print("[yellow]Download cancelled.[/yellow]")
                    return False
    
    # Ask user to proceed
    if not Confirm.ask("\nDownload models now?", default=True):
        console.print("[yellow]Download skipped. You can:[/yellow]")
        console.print("  1. Configure local model paths: chunky embedding config / chunky reranker config")
        console.print("  2. Use test mode: chunky config --test-mode on")
        console.print("  3. Install ModelScope: pip install modelscope")
        console.print("  4. Set proxy and retry: export HTTPS_PROXY=http://your-proxy:port")
        return False
    
    console.print()
    
    # Download models
    results = manager.download_models(models_to_download)
    
    # Check results
    failed = [r for r in results if not r.success]
    successful = [r for r in results if r.success]
    
    console.print()
    
    if successful:
        console.print(f"[green]✓ {len(successful)} model(s) ready[/green]")
    
    if failed:
        console.print(f"[red]✗ {len(failed)} model(s) failed[/red]")
        console.print("\n[bold red]Errors:[/bold red]")
        for r in failed:
            console.print(f"  [red]• {r.name} ({r.model_id}): {r.message}[/red]")
        
        console.print("\n[yellow]Suggestions:[/yellow]")
        console.print("  1. Check your internet connection")
        console.print("  2. Install ModelScope: pip install modelscope")
        console.print("  3. Set HTTPS_PROXY if you have a VPN")
        console.print("  4. Download models manually from https://www.modelscope.cn")
        return False
    
    # Update config with cache paths
    for result in successful:
        if result.local_path:
            if need_embedding and result.model_id == config.embedding.model_name:
                config.embedding.local_model_path = result.local_path
            elif need_reranker and result.model_id == config.reranker.model_name:
                config.reranker.local_model_path = result.local_path
    
    return True
