"""Main CLI entry point for chunky."""

from __future__ import annotations

from pathlib import Path

import click
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Confirm, Prompt
from rich.table import Table

from chunky.config.settings import (
    ChunkyConfig,
    ChromaConfig,
    EmbeddingConfig,
    LLMConfig,
    MilvusConfig,
    RerankerConfig,
    load_config,
    save_config,
)

console = Console()


# ───────────────────────────────────────────────────────────────────
# Helpers
# ───────────────────────────────────────────────────────────────────


def _mask_key(key: str) -> str:
    """Return a masked version of an API key for display."""
    if not key:
        return "(not set)"
    if len(key) <= 8:
        return "****"
    return key[:4] + "****" + key[-4:]


def _prompt_llm_config(existing: LLMConfig | None = None) -> LLMConfig:
    """Interactively prompt the user for LLM configuration values.

    If *existing* is provided its values are used as defaults.
    """
    defaults = existing or LLMConfig()

    console.print("\n[bold]LLM Configuration[/bold]")
    api_type = Prompt.ask(
        "  API type (openai/vllm)",
        choices=["openai", "vllm"],
        default=defaults.api_type or "openai",
    )
    api_base = Prompt.ask(
        "  API base URL",
        default=defaults.api_base or "https://api.openai.com/v1",
    )
    api_key = Prompt.ask(
        "  API key",
        password=True,
        default=defaults.api_key or "",
    )
    model = Prompt.ask(
        "  Model name",
        default=defaults.model or "gpt-4o-mini",
    )

    return LLMConfig(
        api_base=api_base,
        api_key=api_key,
        model=model,
        api_type=api_type,
        max_tokens=defaults.max_tokens,
        temperature=defaults.temperature,
    )


def _prompt_milvus_config(existing: MilvusConfig | None = None) -> MilvusConfig:
    """Interactively prompt the user for Milvus configuration values."""
    defaults = existing or MilvusConfig()

    console.print("\n[bold]Milvus Configuration[/bold]")
    
    # First ask if using Lite or Standalone
    use_lite = Confirm.ask(
        "  Use Milvus Lite?",
        default=defaults.use_lite,
    )
    
    if use_lite:
        console.print("\n[cyan]Milvus Lite mode selected[/cyan]")
        console.print("  Milvus Lite runs locally without Docker.")
        uri = Prompt.ask(
            "  Database file path",
            default=defaults.uri or "milvus_lite.db",
        )
    else:
        console.print("\n[cyan]Milvus Standalone mode selected[/cyan]")
        console.print("  Connect to a remote Milvus server or Docker container.")
        uri = Prompt.ask(
            "  Milvus URI (host:port)",
            default=defaults.uri if not defaults.use_lite else "localhost:19530",
        )
    
    default_collection = Prompt.ask(
        "  Default collection name",
        default=defaults.default_collection or "chunky_default",
    )
    
    dim_str = Prompt.ask(
        "  Embedding dimension",
        default=str(defaults.dim or 512),
    )
    
    try:
        dim = int(dim_str)
    except ValueError:
        dim = 512

    return MilvusConfig(
        uri=uri,
        use_lite=use_lite,
        default_collection=default_collection,
        dim=dim,
    )


def _prompt_chroma_config(existing: ChromaConfig | None = None) -> ChromaConfig:
    """Interactively prompt the user for ChromaDB configuration values."""
    defaults = existing or ChromaConfig()

    console.print("\n[bold]ChromaDB Configuration[/bold]")
    
    persist_dir = Prompt.ask(
        "  Persist directory",
        default=defaults.persist_directory or "~/.chunky/chroma_db",
    )
    
    default_collection = Prompt.ask(
        "  Default collection name",
        default=defaults.default_collection or "chunky_default",
    )
    
    dim_str = Prompt.ask(
        "  Embedding dimension",
        default=str(defaults.dim or 384),
    )
    
    try:
        dim = int(dim_str)
    except ValueError:
        dim = 384

    return ChromaConfig(
        persist_directory=persist_dir,
        default_collection=default_collection,
        dim=dim,
    )


def _count_supported_files(directory: Path) -> dict[str, int]:
    """Walk *directory* and return a {extension: count} mapping of supported files."""
    from chunky.parsers.registry import SUPPORTED_EXTENSIONS

    counts: dict[str, int] = {}
    for path in sorted(directory.rglob("*")):
        if not path.is_file():
            continue
        ext = path.suffix.lower()
        if ext in SUPPORTED_EXTENSIONS:
            counts[ext] = counts.get(ext, 0) + 1
    return counts


# ───────────────────────────────────────────────────────────────────
# Top-level group
# ───────────────────────────────────────────────────────────────────


@click.group()
def cli() -> None:
    """chunky -- Build local knowledge bases from documents."""


# ───────────────────────────────────────────────────────────────────
# chunky config  (subgroup)
# ───────────────────────────────────────────────────────────────────


@cli.command("config")
@click.option(
    "--list",
    "list_all",
    is_flag=True,
    help="List all configuration settings.",
)
@click.option(
    "--test-mode",
    type=click.Choice(["on", "off"], case_sensitive=False),
    help="Toggle test mode on or off (without interactive configuration).",
)
def config_cmd(list_all: bool, test_mode: str | None) -> None:
    """Show or modify configuration settings.
    
    Examples:
        chunky config                  # Show all configuration
        chunky config --list           # Same as above
        chunky config --test-mode on   # Enable test mode
        chunky config --test-mode off  # Disable test mode
    """
    config = load_config()

    # Handle test-mode toggle
    if test_mode is not None:
        config.test_mode = (test_mode.lower() == "on")
        save_config(config)
        status = "enabled" if config.test_mode else "disabled"
        console.print(f"[bold green]Test mode {status}![/bold green]")
        console.print()
        if config.test_mode:
            console.print("[dim]Test mode uses lightweight implementations:[/dim]")
            console.print("  • Embedding: bag-of-words (TF-IDF + SVD)")
            console.print("  • LLM Labeling: keyword extraction")
            console.print("  • Reranker: disabled")
        else:
            console.print("[dim]Normal mode uses configured services:[/dim]")
            console.print("  • Embedding: neural models via API or local")
            console.print("  • LLM Labeling: LLM API calls")
            console.print("  • Reranker: configured cross-encoder model")
        return

    # Default: show all configuration
    if list_all or True:
        _print_full_config(config)


def _print_full_config(config: ChunkyConfig) -> None:
    """Print the complete configuration."""
    console.print()
    console.print(Panel("[bold cyan]chunky Configuration[/bold cyan]", expand=False))
    console.print()

    # LLM
    llm_table = Table(title="LLM Configuration", box=None, show_lines=False)
    llm_table.add_column("Setting", style="cyan")
    llm_table.add_column("Value", style="white")
    llm_table.add_row("API Type", config.llm.api_type)
    llm_table.add_row("API Base", config.llm.api_base or "(not set)")
    llm_table.add_row("API Key", _mask_key(config.llm.api_key))
    llm_table.add_row("Model", config.llm.model or "(not set)")
    llm_table.add_row("Max Tokens", str(config.llm.max_tokens))
    llm_table.add_row("Temperature", str(config.llm.temperature))
    console.print(llm_table)
    console.print()

    # Embedding
    emb_table = Table(title="Embedding Configuration", box=None, show_lines=False)
    emb_table.add_column("Setting", style="cyan")
    emb_table.add_column("Value", style="white")
    emb_table.add_row("API Type", config.embedding.api_type)
    emb_table.add_row("Model Name", config.embedding.model_name)
    if config.embedding.local_model_path:
        emb_table.add_row("Local Path", config.embedding.local_model_path)
    emb_table.add_row("API Base", config.embedding.api_base or "(local model)")
    emb_table.add_row("API Key", _mask_key(config.embedding.api_key))
    emb_table.add_row("Device", config.embedding.device)
    emb_table.add_row("Batch Size", str(config.embedding.batch_size))
    console.print(emb_table)
    console.print()

    # Reranker
    rer_table = Table(title="Reranker Configuration", box=None, show_lines=False)
    rer_table.add_column("Setting", style="cyan")
    rer_table.add_column("Value", style="white")
    rer_table.add_row("API Type", config.reranker.api_type)
    rer_table.add_row("Model Name", config.reranker.model_name or "(not set)")
    if config.reranker.local_model_path:
        rer_table.add_row("Local Path", config.reranker.local_model_path)
    if config.reranker.api_type in ("vllm", "openai"):
        rer_table.add_row("API Base", config.reranker.api_base or "(not set)")
        rer_table.add_row("API Key", _mask_key(config.reranker.api_key))
    else:
        rer_table.add_row("Device", config.reranker.device)
    console.print(rer_table)
    console.print()

    # Vector Store
    vs_table = Table(title="Vector Store Configuration", box=None, show_lines=False)
    vs_table.add_column("Setting", style="cyan")
    vs_table.add_column("Value", style="white")
    vs_table.add_row("Type", config.vector_store_type)
    
    if config.vector_store_type == "chroma":
        vs_table.add_row("Persist Directory", config.chroma.persist_directory)
        vs_table.add_row("Default Collection", config.chroma.default_collection)
        vs_table.add_row("Dimension", str(config.chroma.dim))
    else:
        vs_table.add_row("URI", config.milvus.uri)
        vs_table.add_row("Lite Mode", str(config.milvus.use_lite))
        vs_table.add_row("Default Collection", config.milvus.default_collection)
        vs_table.add_row("Dimension", str(config.milvus.dim))
    console.print(vs_table)
    console.print()

    # General
    gen_table = Table(title="General", box=None, show_lines=False)
    gen_table.add_column("Setting", style="cyan")
    gen_table.add_column("Value", style="white")
    gen_table.add_row("Test Mode", "ON" if config.test_mode else "OFF")
    console.print(gen_table)
    console.print()


# ───────────────────────────────────────────────────────────────────
# chunky init
# ───────────────────────────────────────────────────────────────────


@cli.command()
@click.option(
    "--test",
    is_flag=True,
    help="Enable test mode (bag-of-words embedding, simple LDA topic modeling).",
)
def init(test: bool) -> None:
    """Interactive setup for chunky configuration."""
    console.print(
        Panel(
            "[bold cyan]chunky init[/bold cyan] -- interactive setup",
            expand=False,
        )
    )

    # Load existing config to preserve settings
    existing_config = load_config()

    if test:
        console.print(
            "[yellow]Test mode enabled.[/yellow]  "
            "Embedding will use bag-of-words; topic modeling will use simple LDA.\n"
        )

    # --- LLM config (always prompted, with existing values) ---
    llm = _prompt_llm_config(existing=existing_config.llm)

    if test:
        # In test mode: skip embedding model (use bag-of-words), skip reranker
        embedding = EmbeddingConfig(model_name="bag-of-words")
        reranker = RerankerConfig()
        console.print("\n[dim]  Embedding model set to bag-of-words (test mode)[/dim]")
        console.print("[dim]  Reranker skipped (test mode)[/dim]")
    else:
        # --- Embedding config (interactive, with existing values) ---
        embedding = _prompt_embedding_config(existing=existing_config.embedding)

        # --- Reranker config ---
        reranker = _prompt_reranker_config()

    # --- Vector Store Type ---
    console.print("\n[bold]Vector Store Configuration[/bold]")
    use_chroma = Confirm.ask(
        "  Use ChromaDB? (No = use Milvus)",
        default=True,
    )
    
    if use_chroma:
        # ChromaDB config (with existing values)
        chroma = _prompt_chroma_config(existing=existing_config.chroma)
        milvus = existing_config.milvus  # Preserve Milvus config
        vector_store_type = "chroma"
        console.print("\n[green]Using ChromaDB as vector store[/green]")
    else:
        # Milvus config (with existing values)
        milvus = _prompt_milvus_config(existing=existing_config.milvus)
        chroma = existing_config.chroma  # Preserve ChromaDB config
        vector_store_type = "milvus"
        console.print("\n[green]Using Milvus as vector store[/green]")

    # --- Assemble and save ---
    config = ChunkyConfig(
        llm=llm,
        embedding=embedding,
        reranker=reranker,
        milvus=milvus,
        chroma=chroma,
        vector_store_type=vector_store_type,
        test_mode=test,
    )
    save_config(config)

    console.print("\n[bold green]Configuration saved![/bold green]")
    _print_config_summary(config)


def _print_config_summary(config: ChunkyConfig) -> None:
    """Print a compact summary table after saving config."""
    table = Table(title="Configuration Summary", show_lines=True)
    table.add_column("Section", style="cyan", no_wrap=True)
    table.add_column("Setting", style="white")
    table.add_column("Value", style="green")

    # LLM
    table.add_row("LLM", "API Type", config.llm.api_type)
    table.add_row("LLM", "API Base", config.llm.api_base or "(not set)")
    table.add_row("LLM", "API Key", _mask_key(config.llm.api_key))
    table.add_row("LLM", "Model", config.llm.model or "(not set)")
    table.add_row("LLM", "Max Tokens", str(config.llm.max_tokens))
    table.add_row("LLM", "Temperature", str(config.llm.temperature))

    # Embedding
    table.add_row("Embedding", "API Type", config.embedding.api_type)
    table.add_row("Embedding", "Model", config.embedding.model_name)
    if config.embedding.local_model_path:
        table.add_row("Embedding", "Local Path", config.embedding.local_model_path)
    table.add_row("Embedding", "API Base", config.embedding.api_base or "(local)")
    table.add_row("Embedding", "Device", config.embedding.device)
    table.add_row("Embedding", "Batch Size", str(config.embedding.batch_size))

    # Reranker
    table.add_row("Reranker", "API Type", config.reranker.api_type)
    table.add_row("Reranker", "Model", config.reranker.model_name or "(not set)")
    if config.reranker.local_model_path:
        table.add_row("Reranker", "Local Path", config.reranker.local_model_path)
    if config.reranker.api_type in ("vllm", "openai"):
        table.add_row("Reranker", "API Base", config.reranker.api_base or "(not set)")
        table.add_row("Reranker", "API Key", _mask_key(config.reranker.api_key))
    else:
        table.add_row("Reranker", "Device", config.reranker.device)

    # Milvus
    table.add_row("Milvus", "URI", config.milvus.uri)
    table.add_row("Milvus", "Lite Mode", str(config.milvus.use_lite))
    table.add_row("Milvus", "Collection", config.milvus.default_collection)
    table.add_row("Milvus", "Dimension", str(config.milvus.dim))

    # General
    table.add_row("General", "Test Mode", str(config.test_mode))

    console.print(table)


# ───────────────────────────────────────────────────────────────────
# chunky models  (subgroup)
# ───────────────────────────────────────────────────────────────────


@cli.group()
def models() -> None:
    """Manage LLM model configuration."""


@models.command("config")
def models_config() -> None:
    """Re-configure the LLM settings interactively."""
    config = load_config()

    console.print(
        Panel("[bold cyan]LLM Model Configuration[/bold cyan]", expand=False)
    )

    config.llm = _prompt_llm_config(existing=config.llm)
    save_config(config)

    console.print("\n[bold green]LLM configuration saved![/bold green]")


@models.command("list")
def models_list() -> None:
    """Display the current LLM configuration."""
    config = load_config()

    table = Table(title="LLM Configuration")
    table.add_column("Setting", style="cyan", no_wrap=True)
    table.add_column("Value", style="white")

    table.add_row("API Type", config.llm.api_type)
    table.add_row("API Base", config.llm.api_base or "(not set)")
    table.add_row("API Key", _mask_key(config.llm.api_key))
    table.add_row("Model", config.llm.model or "(not set)")
    table.add_row("Max Tokens", str(config.llm.max_tokens))
    table.add_row("Temperature", str(config.llm.temperature))
    table.add_row("Embedding Model", config.embedding.model_name)
    table.add_row("Embedding Device", config.embedding.device)
    table.add_row("Reranker Type", config.reranker.api_type)
    table.add_row("Reranker Model", config.reranker.model_name or "(not set)")
    if config.reranker.api_type in ("vllm", "openai"):
        table.add_row("Reranker API Base", config.reranker.api_base or "(not set)")
    else:
        table.add_row("Reranker Device", config.reranker.device)

    console.print(table)


# ───────────────────────────────────────────────────────────────────
# chunky embedding  (subgroup)
# ───────────────────────────────────────────────────────────────────


def _prompt_reranker_config(existing: RerankerConfig | None = None) -> RerankerConfig:
    """Interactively prompt the user for reranker configuration values."""
    defaults = existing or RerankerConfig()

    console.print("\n[bold]Reranker Configuration[/bold]")
    
    # API type selection
    console.print("\n  [cyan]Reranker Type:[/cyan]")
    console.print("    1. local (local cross-encoder model, default)")
    console.print("    2. vllm (vLLM reranker API server)")
    console.print("    3. openai (OpenAI-compatible reranker API)")
    api_type_choice = Prompt.ask(
        "  Select reranker type",
        choices=["1", "2", "3"],
        default="1",
    )
    api_type_map = {"1": "local", "2": "vllm", "3": "openai"}
    api_type = api_type_map[api_type_choice]

    # Show cached models in cache directory
    if api_type == "local":
        cache_dir = Path.home() / ".cache" / "chunky" / "models"
        if cache_dir.exists():
            # Only show valid model dirs (contain config.json)
            cached_models = [
                d.name for d in cache_dir.iterdir() 
                if d.is_dir() and not d.name.startswith(".") and (d / "config.json").exists()
            ]
            if cached_models:
                console.print("    [dim]Cached models (use short name or full HF ID):[/dim]")
                for model in sorted(cached_models):
                    # Convert cache name back to readable format
                    readable = model.replace("--", "/")
                    console.print(f"    [dim]• {readable}[/dim]")

    model_name = Prompt.ask(
        "  Model name (Hugging Face model ID)",
        default=defaults.model_name or "BAAI/bge-reranker-base",
    )
    
    # Local model path (only for local mode)
    local_model_path = ""
    if api_type == "local":
        use_local_path = Confirm.ask(
            "  Use local model path?",
            default=bool(defaults.local_model_path),
        )
        if use_local_path:
            local_model_path = Prompt.ask(
                "  Local model path",
                default=defaults.local_model_path or "",
            )
    
    api_base = ""
    api_key = ""
    device = defaults.device or "cpu"
    
    if api_type in ("vllm", "openai"):
        api_base = Prompt.ask(
            "  API base URL",
            default=defaults.api_base or "http://localhost:8000",
        )
        api_key = Prompt.ask(
            "  API key (optional)",
            password=True,
            default=defaults.api_key or "",
        )
    else:
        device = Prompt.ask("  Device", default=defaults.device or "cpu")

    return RerankerConfig(
        model_name=model_name,
        api_type=api_type,
        api_base=api_base,
        api_key=api_key,
        device=device,
        local_model_path=local_model_path,
    )


def _prompt_embedding_config(existing: EmbeddingConfig | None = None) -> EmbeddingConfig:
    """Interactively prompt the user for embedding configuration values."""
    defaults = existing or EmbeddingConfig()

    console.print("\n[bold]Embedding Configuration[/bold]")
    
    # API type selection
    console.print("\n  [cyan]API Type:[/cyan]")
    console.print("    1. sentence_transformers (local, default)")
    console.print("    2. openai (OpenAI-compatible API)")
    console.print("    3. vllm (vLLM server)")
    api_type_choice = Prompt.ask(
        "  Select API type",
        choices=["1", "2", "3"],
        default="1",
    )
    api_type_map = {"1": "sentence_transformers", "2": "openai", "3": "vllm"}
    api_type = api_type_map[api_type_choice]

    # Show cached models in cache directory
    if api_type == "sentence_transformers":
        cache_dir = Path.home() / ".cache" / "chunky" / "models"
        if cache_dir.exists():
            # Only show valid model dirs (contain config.json)
            cached_models = [
                d.name for d in cache_dir.iterdir() 
                if d.is_dir() and not d.name.startswith(".") and (d / "config.json").exists()
            ]
            if cached_models:
                console.print("    [dim]Cached models (use short name or full HF ID):[/dim]")
                for model in sorted(cached_models):
                    # Convert cache name back to readable format
                    readable = model.replace("--", "/")
                    console.print(f"    [dim]• {readable}[/dim]")

    model_name = Prompt.ask(
        "  Model name (Hugging Face model ID)",
        default=defaults.model_name or "BAAI/bge-small-zh-v1.5",
    )
    
    # Local model path (only for local mode)
    local_model_path = ""
    if api_type == "sentence_transformers":
        use_local_path = Confirm.ask(
            "  Use local model path?",
            default=bool(defaults.local_model_path),
        )
        if use_local_path:
            local_model_path = Prompt.ask(
                "  Local model path",
                default=defaults.local_model_path or "",
            )
    
    # Only show API base/key for API-based modes
    api_base = ""
    api_key = ""
    if api_type in ("openai", "vllm"):
        api_base = Prompt.ask(
            "  API base URL",
            default=defaults.api_base or "http://localhost:8000",
        )
        api_key = Prompt.ask(
            "  API key (optional)",
            password=True,
            default=defaults.api_key or "",
        )
    
    device = Prompt.ask("  Device", default=defaults.device or "cpu")
    batch_size_str = Prompt.ask(
        "  Batch size",
        default=str(defaults.batch_size or 32),
    )
    try:
        batch_size = int(batch_size_str)
    except ValueError:
        batch_size = 32

    return EmbeddingConfig(
        model_name=model_name,
        api_type=api_type,
        api_base=api_base,
        api_key=api_key,
        device=device,
        batch_size=batch_size,
        local_model_path=local_model_path,
    )


@cli.group()
def embedding() -> None:
    """Manage embedding model configuration."""


@embedding.command("config")
def embedding_config() -> None:
    """Re-configure the embedding settings interactively."""
    config = load_config()

    console.print(
        Panel("[bold cyan]Embedding Configuration[/bold cyan]", expand=False)
    )

    config.embedding = _prompt_embedding_config(existing=config.embedding)
    save_config(config)

    console.print("\n[bold green]Embedding configuration saved![/bold green]")
    _print_embedding_summary(config)


@embedding.command("list")
def embedding_list() -> None:
    """Display the current embedding configuration."""
    config = load_config()
    _print_embedding_summary(config)


def _print_embedding_summary(config: ChunkyConfig) -> None:
    """Print embedding configuration table."""
    table = Table(title="Embedding Configuration")
    table.add_column("Setting", style="cyan", no_wrap=True)
    table.add_column("Value", style="white")

    table.add_row("API Type", config.embedding.api_type)
    table.add_row("Model Name", config.embedding.model_name)
    if config.embedding.local_model_path:
        table.add_row("Local Path", config.embedding.local_model_path)
    table.add_row("API Base", config.embedding.api_base or "(local model)" if config.embedding.api_type != "sentence_transformers" else "(local model)")
    table.add_row("API Key", _mask_key(config.embedding.api_key))
    table.add_row("Device", config.embedding.device)
    table.add_row("Batch Size", str(config.embedding.batch_size))

    console.print(table)


# ───────────────────────────────────────────────────────────────────
# chunky reranker  (subgroup)
# ───────────────────────────────────────────────────────────────────


@cli.group()
def reranker() -> None:
    """Manage reranker model configuration."""


@reranker.command("config")
def reranker_config() -> None:
    """Re-configure the reranker settings interactively."""
    config = load_config()

    console.print(
        Panel("[bold cyan]Reranker Configuration[/bold cyan]", expand=False)
    )

    config.reranker = _prompt_reranker_config(existing=config.reranker)
    save_config(config)

    console.print("\n[bold green]Reranker configuration saved![/bold green]")
    _print_reranker_summary(config)


@reranker.command("list")
def reranker_list() -> None:
    """Display the current reranker configuration."""
    config = load_config()
    _print_reranker_summary(config)


def _print_reranker_summary(config: ChunkyConfig) -> None:
    """Print reranker configuration table."""
    table = Table(title="Reranker Configuration")
    table.add_column("Setting", style="cyan", no_wrap=True)
    table.add_column("Value", style="white")

    table.add_row("API Type", config.reranker.api_type)
    table.add_row("Model Name", config.reranker.model_name or "(not set)")
    if config.reranker.local_model_path:
        table.add_row("Local Path", config.reranker.local_model_path)
    if config.reranker.api_type in ("vllm", "openai"):
        table.add_row("API Base", config.reranker.api_base or "(not set)")
        table.add_row("API Key", _mask_key(config.reranker.api_key))
    else:
        table.add_row("Device", config.reranker.device)

    console.print(table)


# ───────────────────────────────────────────────────────────────────
# chunky milvus  (subgroup)
# ───────────────────────────────────────────────────────────────────


@cli.group(invoke_without_command=True)
@click.option(
    "--collection",
    default=None,
    help="Print or set the default Milvus collection name.",
)
@click.option(
    "--delete",
    "delete_flag",
    is_flag=True,
    default=False,
    help="Delete the specified collection.",
)
@click.pass_context
def milvus(ctx: click.Context, collection: str | None, delete_flag: bool) -> None:
    """Manage Milvus vector store configuration."""
    # If a subcommand is being invoked, let it handle things.
    if ctx.invoked_subcommand is not None:
        return

    config = load_config()

    if delete_flag:
        # Delete collection mode
        if collection is None:
            collection = config.milvus.default_collection
        
        if not collection:
            console.print("[red]Error:[/red] No collection specified. Use --collection or set default.")
            raise SystemExit(1)
        
        from rich.prompt import Confirm
        if not Confirm.ask(f"Delete collection '{collection}'? This cannot be undone.", default=False):
            console.print("[dim]Cancelled.[/dim]")
            return
        
        # Perform deletion
        try:
            from chunky.vectorstore.milvus_store import MilvusStore
            store = MilvusStore(config.milvus)
            store.connect()
            
            if store.collection_exists(collection):
                store.drop_collection(collection)
                console.print(f"[bold green]✓ Collection '{collection}' deleted.[/bold green]")
            else:
                console.print(f"[yellow]Collection '{collection}' does not exist.[/yellow]")
            
            store.close()
        except Exception as e:
            console.print(f"[red]Failed to delete collection:[/red] {e}")
            raise SystemExit(1)
        return

    if collection is not None:
        # Set collection override and switch to Milvus
        config.milvus.default_collection = collection
        config.vector_store_type = "milvus"
        save_config(config)
        console.print(
            f"[bold green]Default collection set to:[/bold green] {collection}"
        )
        console.print(
            f"[dim]Vector store switched to: Milvus[/dim]"
        )
    else:
        # No subcommand and no --collection: show current collection
        console.print(
            f"[bold cyan]Current default collection:[/bold cyan] "
            f"{config.milvus.default_collection}"
        )


@milvus.command("config")
def milvus_config() -> None:
    """Re-configure Milvus connection settings interactively."""
    config = load_config()

    console.print(
        Panel("[bold cyan]Milvus Configuration[/bold cyan]", expand=False)
    )

    config.milvus = _prompt_milvus_config(existing=config.milvus)
    
    # Ensure vector_store_type is set to milvus
    config.vector_store_type = "milvus"
    
    save_config(config)

    console.print("\n[bold green]Milvus configuration saved![/bold green]")
    
    # Show connection info
    if config.milvus.use_lite:
        console.print(f"\n[dim]URI: {config.milvus.uri}[/dim]")
    else:
        console.print(f"\n[dim]Server: {config.milvus.uri}[/dim]")


# ───────────────────────────────────────────────────────────────────
# chunky chroma  (subgroup)
# ───────────────────────────────────────────────────────────────────


@cli.group(invoke_without_command=True)
@click.option(
    "--collection",
    "collection_name",
    default=None,
    help="Print or set the default ChromaDB collection name.",
)
@click.option(
    "--delete",
    "delete_flag",
    is_flag=True,
    default=False,
    help="Delete the specified collection.",
)
@click.pass_context
def chroma(ctx: click.Context, collection_name: str | None, delete_flag: bool) -> None:
    """Manage ChromaDB vector store configuration."""
    # If a subcommand is being invoked, let it handle things.
    if ctx.invoked_subcommand is not None:
        return

    config = load_config()

    if delete_flag:
        # Delete collection mode
        target_collection = collection_name or config.chroma.default_collection
        
        if not target_collection:
            console.print("[red]Error:[/red] No collection specified. Use --collection or set default.")
            raise SystemExit(1)
        
        from rich.prompt import Confirm
        if not Confirm.ask(f"Delete collection '{target_collection}'? This cannot be undone.", default=False):
            console.print("[dim]Cancelled.[/dim]")
            return
        
        # Perform deletion
        try:
            from chunky.vectorstore.chroma_store import ChromaStore
            store = ChromaStore(config.chroma)
            store.connect()
            
            if store.collection_exists(target_collection):
                store.drop_collection(target_collection)
                console.print(f"[bold green]✓ Collection '{target_collection}' deleted.[/bold green]")
            else:
                console.print(f"[yellow]Collection '{target_collection}' does not exist.[/yellow]")
            
            store.close()
        except Exception as e:
            console.print(f"[red]Failed to delete collection:[/red] {e}")
            raise SystemExit(1)
        return

    if collection_name is not None:
        # Set collection override and switch to ChromaDB
        config.chroma.default_collection = collection_name
        config.vector_store_type = "chroma"
        save_config(config)
        console.print(
            f"[bold green]Default collection set to:[/bold green] {collection_name}"
        )
        console.print(
            f"[dim]Vector store switched to: ChromaDB[/dim]"
        )
    else:
        # No subcommand and no --collection: show current collection
        console.print(
            f"[bold cyan]Current default collection:[/bold cyan] "
            f"{config.chroma.default_collection}"
        )


@chroma.command("config")
def chroma_config() -> None:
    """Re-configure ChromaDB connection settings interactively."""
    config = load_config()

    console.print(
        Panel("[bold cyan]ChromaDB Configuration[/bold cyan]", expand=False)
    )

    config.chroma = _prompt_chroma_config(existing=config.chroma)
    
    # Ensure vector_store_type is set to chroma
    config.vector_store_type = "chroma"
    
    save_config(config)

    console.print("\n[bold green]ChromaDB configuration saved![/bold green]")
    console.print(f"\n[dim]Persist directory: {config.chroma.persist_directory}[/dim]")


# ───────────────────────────────────────────────────────────────────
# chunky build
# ───────────────────────────────────────────────────────────────────


@cli.command()
@click.option(
    "--dir",
    "dir_path",
    required=True,
    type=click.Path(exists=True, file_okay=False, readable=True),
    help="Directory of documents to process.",
)
@click.option(
    "--collection",
    default=None,
    help="Override the Milvus collection name for this build.",
)
def build(dir_path: str, collection: str | None) -> None:
    """Build a knowledge base from a directory of documents."""
    # CRITICAL: Import hf_setup first to ensure HF_ENDPOINT is set
    # before any Hugging Face related imports
    from chunky.utils import hf_setup
    
    directory = Path(dir_path).resolve()
    config = load_config()
    
    # Get the correct default collection based on vector store type
    if collection:
        collection_name = collection
    elif config.vector_store_type == "chroma":
        collection_name = config.chroma.default_collection
    else:
        collection_name = config.milvus.default_collection

    # ── Validate ──────────────────────────────────────────────────
    if not directory.is_dir():
        console.print(f"[bold red]Error:[/bold red] '{directory}' is not a valid directory.")
        raise SystemExit(1)

    # ── Summary ───────────────────────────────────────────────────
    console.print(
        Panel(
            "[bold cyan]chunky build[/bold cyan] -- knowledge base builder",
            expand=False,
        )
    )

    # ── Download Models (if needed) ───────────────────────────────
    if not config.test_mode:
        from chunky.utils.model_downloader import ensure_models_downloaded
        
        console.print("[bold]Step 0/N[/bold] Checking models ...")
        
        models_ready = ensure_models_downloaded(config)
        
        if not models_ready:
            console.print("\n[bold red]Model download failed![/bold red]")
            console.print("[yellow]Please check your network connection or model IDs.[/yellow]")
            raise SystemExit(1)
        
        console.print("[bold green]Models ready![/bold green]")
        console.print()

    # ── Connectivity Tests ─────────────────────────────────────────
    console.print("[bold]Step 1/N[/bold] Testing connectivity ...")
    from chunky.utils.connectivity import run_connectivity_tests

    results = run_connectivity_tests(config)
    all_passed = True

    for result in results:
        if result.success:
            console.print(f"  [green]✓[/green] {result.name}: {result.message}")
        else:
            all_passed = False
            console.print(f"  [red]✗[/red] {result.name}: {result.message}")

    console.print()

    if not all_passed:
        console.print("[bold red]Connectivity test failed![/bold red]")
        console.print("[yellow]Please fix the issues above and try again.[/yellow]")
        raise SystemExit(1)

    console.print("[bold green]Connectivity test passed![/bold green]")
    console.print()

    file_counts = _count_supported_files(directory)
    total_files = sum(file_counts.values())

    if total_files == 0:
        console.print(
            f"[bold red]No supported files found in:[/bold red] {directory}"
        )
        console.print(
            "[dim]Supported extensions: .pdf, .docx, .doc, .pptx, .ppt, .txt, .md[/dim]"
        )
        raise SystemExit(1)

    summary_table = Table(title="Build Summary", show_lines=False)
    summary_table.add_column("Parameter", style="cyan", no_wrap=True)
    summary_table.add_column("Value", style="white")

    summary_table.add_row("Source directory", str(directory))
    summary_table.add_row("Collection", collection_name)
    summary_table.add_row("Test mode", "ON" if config.test_mode else "OFF")
    summary_table.add_row("Total files", str(total_files))

    # Per-extension breakdown
    for ext in sorted(file_counts):
        summary_table.add_row(f"  {ext}", str(file_counts[ext]))

    console.print(summary_table)
    console.print()

    # ── Run pipeline ──────────────────────────────────────────────
    try:
        from chunky.pipeline.runner import PipelineRunner

        runner = PipelineRunner(config)
        runner.run(str(directory), collection_name=collection_name)
    except KeyboardInterrupt:
        console.print("\n[bold yellow]Build interrupted by user.[/bold yellow]")
        raise SystemExit(130)
    except Exception as exc:
        console.print(f"\n[bold red]Pipeline failed:[/bold red] {exc}")
        raise SystemExit(1)


# ───────────────────────────────────────────────────────────────────
# chunky search
# ───────────────────────────────────────────────────────────────────


@cli.command()
@click.argument("query")
@click.option(
    "--collection",
    "-c",
    default=None,
    help="Collection to search in. Defaults to configured default collection.",
)
@click.option(
    "--top-k", "-k",
    default=5,
    type=int,
    help="Number of results to return (default: 5).",
)
@click.option(
    "--vector-weight", "-vw",
    default=0.5,
    type=float,
    help="Weight for vector search score (0-1, default: 0.5).",
)
@click.option(
    "--bm25-weight", "-bw",
    default=0.5,
    type=float,
    help="Weight for BM25 score (0-1, default: 0.5).",
)
@click.option(
    "--fusion", "-f",
    "fusion_method",
    type=click.Choice(["rrf", "weighted_sum", "relative_score"], case_sensitive=False),
    default="rrf",
    help="Fusion method: rrf (Reciprocal Rank Fusion), weighted_sum, relative_score (default: rrf).",
)
@click.option(
    "--no-labels",
    is_flag=True,
    help="Hide labels from results.",
)
@click.option(
    "--no-topics",
    is_flag=True,
    help="Hide topics from results.",
)
@click.option(
    "--verbose", "-v",
    is_flag=True,
    help="Show detailed scores for each result.",
)
@click.option(
    "--rerank", "-r",
    is_flag=True,
    help="Use reranker to rerank results.",
)
@click.option(
    "--rerank-top-k",
    default=20,
    type=int,
    help="Number of initial results to rerank (default: 20).",
)
def search(
    query: str,
    collection: str | None,
    top_k: int,
    vector_weight: float,
    bm25_weight: float,
    fusion_method: str,
    no_labels: bool,
    no_topics: bool,
    verbose: bool,
    rerank: bool,
    rerank_top_k: int,
) -> None:
    """Search the knowledge base using hybrid search (vector + BM25).
    
    QUERY: The search query string.
    
    Examples:
    
        chunky search "machine learning"
        chunky search "neural networks" -c my_collection -k 10
        chunky search "transformer attention" -vw 0.7 -bw 0.3 -f rrf
    """
    config = load_config()
    
    # Select default collection based on vector store type
    if collection is None:
        if config.vector_store_type == "chroma":
            collection_name = config.chroma.default_collection
        else:
            collection_name = config.milvus.default_collection
    else:
        collection_name = collection

    # Print header
    print("=" * 50)
    print("chunky search -- hybrid search (vector + BM25)")
    print("=" * 50)

    # Validate weights
    if not (0 <= vector_weight <= 1) or not (0 <= bm25_weight <= 1):
        print("ERROR: Weights must be between 0 and 1.")
        raise SystemExit(1)

    # Show search config
    print("\nSearch Configuration:")
    print(f"  Query:         {query}")
    print(f"  Collection:    {collection_name}")
    print(f"  Top K:         {top_k}")
    print(f"  Vector Weight: {vector_weight}")
    print(f"  BM25 Weight:   {bm25_weight}")
    print(f"  Fusion Method: {fusion_method.upper()}")
    print()

    try:
        from chunky.search import SearchManager
        
        print("Connecting to vector store...")
        manager = SearchManager(config)
        manager.connect()
        
        print("Building BM25 index...")
        chunks_count = manager.build_index(collection_name)
        
        if chunks_count == 0:
            print(f"WARNING: No chunks found in collection '{collection_name}'")
            print("Run 'chunky build' first to populate the knowledge base.")
            manager.close()
            raise SystemExit(1)
        
        print(f"Searching {chunks_count} chunks...")
        
        # Determine initial top_k for search (need more for reranking)
        search_top_k = rerank_top_k if rerank else top_k
        
        results = manager.search(
            query=query,
            collection_name=collection_name,
            top_k=search_top_k,
            vector_weight=vector_weight,
            bm25_weight=bm25_weight,
            fusion_method=fusion_method,
        )
        
        # Apply reranker if enabled
        if rerank and results:
            if config.reranker.model_name:
                print(f"Reranking {len(results)} results with {config.reranker.api_type} reranker...")
                from chunky.reranker import get_reranker
                
                try:
                    reranker = get_reranker(config.reranker)
                    documents = [r.text for r in results]
                    reranked = reranker.rerank(query, documents, top_k=top_k)
                    
                    # Update results based on reranker output
                    new_results = []
                    for i, rr in enumerate(reranked, 1):
                        orig_result = results[rr.index]
                        orig_result.rank = i
                        # Update combined score to reflect reranker score
                        orig_result.combined_score = rr.score
                        new_results.append(orig_result)
                    
                    results = new_results
                    print(f"Reranking complete.")
                except Exception as e:
                    print(f"Warning: Reranking failed: {e}")
                    # Fallback to original results, truncate to top_k
                    results = results[:top_k]
            else:
                print("Warning: Reranker enabled but no model configured. Skipping rerank.")
                results = results[:top_k]
        
        manager.close()

        if not results:
            print("\nNo results found.")
            print("Try a different query or adjust the search parameters.")
            raise SystemExit(0)

        # Display results
        print(f"\n{'='*60}")
        print(f"Found {len(results)} results:")
        print(f"{'='*60}\n")

        for result in results:
            # Score info
            score_info = ""
            if verbose:
                score_info = f" (vec:{result.vector_score:.3f}, bm25:{result.bm25_score:.3f}, combined:{result.combined_score:.3f})"
            
            print(f">>> Result #{result.rank}: {result.source_file} (chunk-{result.chunk_index}){score_info}")
            print("-" * 40)
            
            # Text preview (first 300 chars)
            text_preview = result.text[:300] + "..." if len(result.text) > 300 else result.text
            print(f"  {text_preview}")
            
            if not no_labels and result.labels:
                print(f"  Labels: {', '.join(result.labels[:5])}")
            
            if not no_topics and result.topics:
                print(f"  Topics: {', '.join(result.topics[:5])}")
            
            print()

    except KeyboardInterrupt:
        print("\n[Search interrupted by user]")
        raise SystemExit(130)
    except Exception as exc:
        print(f"\nSearch failed: {exc}")
        if verbose:
            import traceback
            traceback.print_exc()
        raise SystemExit(1)


# ───────────────────────────────────────────────────────────────────
# chunky collections
# ───────────────────────────────────────────────────────────────────


@cli.command("collections")
def collections_list() -> None:
    """List all collections in the vector store."""
    config = load_config()
    
    try:
        from chunky.search import SearchManager
        
        manager = SearchManager(config)
        manager.connect()
        
        collections = manager.get_collections()
        manager.close()
        
        if not collections:
            console.print("[yellow]No collections found.[/yellow]")
            console.print("[dim]Run 'chunky build' to create your first collection.[/dim]")
            return
        
        table = Table(title=f"Collections ({len(collections)} found)")
        table.add_column("Name", style="cyan")
        
        default_col = config.milvus.default_collection
        for col in sorted(collections):
            marker = " [dim](default)[/dim]" if col == default_col else ""
            table.add_row(f"{col}{marker}")
        
        console.print(table)
        
    except Exception as exc:
        console.print(f"[bold red]Failed to list collections:[/bold red] {exc}")
        raise SystemExit(1)
