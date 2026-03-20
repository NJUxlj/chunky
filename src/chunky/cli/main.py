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

    if test:
        console.print(
            "[yellow]Test mode enabled.[/yellow]  "
            "Embedding will use bag-of-words; topic modeling will use simple LDA.\n"
        )

    # --- LLM config (always prompted) ---
    llm = _prompt_llm_config()

    if test:
        # In test mode: skip embedding model (use bag-of-words), skip reranker
        embedding = EmbeddingConfig(model_name="bag-of-words")
        reranker = RerankerConfig()
        console.print("\n[dim]  Embedding model set to bag-of-words (test mode)[/dim]")
        console.print("[dim]  Reranker skipped (test mode)[/dim]")
    else:
        # --- Embedding config ---
        console.print("\n[bold]Embedding Configuration[/bold]")
        emb_model = Prompt.ask(
            "  Model name",
            default="BAAI/bge-small-zh-v1.5",
        )
        emb_device = Prompt.ask("  Device", default="cpu")
        embedding = EmbeddingConfig(model_name=emb_model, device=emb_device)

        # --- Reranker config ---
        console.print("\n[bold]Reranker Configuration[/bold]")
        reranker_model = Prompt.ask(
            "  Model name",
            default="BAAI/bge-reranker-base",
        )
        reranker_device = Prompt.ask("  Device", default="cpu")
        reranker = RerankerConfig(model_name=reranker_model, device=reranker_device)

    # --- Vector Store Type ---
    console.print("\n[bold]Vector Store Configuration[/bold]")
    use_chroma = Confirm.ask(
        "  Use ChromaDB? (No = use Milvus)",
        default=True,
    )
    
    if use_chroma:
        # ChromaDB config
        chroma = _prompt_chroma_config()
        milvus = MilvusConfig()  # Use defaults
        vector_store_type = "chroma"
        console.print("\n[green]Using ChromaDB as vector store[/green]")
    else:
        # Milvus config
        milvus = _prompt_milvus_config()
        chroma = ChromaConfig()  # Use defaults
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
    table.add_row("Embedding", "Model", config.embedding.model_name)
    table.add_row("Embedding", "Device", config.embedding.device)
    table.add_row("Embedding", "Batch Size", str(config.embedding.batch_size))

    # Reranker
    table.add_row("Reranker", "Model", config.reranker.model_name or "(not set)")
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
    table.add_row("Reranker Model", config.reranker.model_name or "(not set)")
    table.add_row("Reranker Device", config.reranker.device)

    console.print(table)


# ───────────────────────────────────────────────────────────────────
# chunky embedding  (subgroup)
# ───────────────────────────────────────────────────────────────────


def _prompt_embedding_config(existing: EmbeddingConfig | None = None) -> EmbeddingConfig:
    """Interactively prompt the user for embedding configuration values."""
    defaults = existing or EmbeddingConfig()

    console.print("\n[bold]Embedding Configuration[/bold]")
    model_name = Prompt.ask(
        "  Model name",
        default=defaults.model_name or "BAAI/bge-small-zh-v1.5",
    )
    api_base = Prompt.ask(
        "  API base URL (optional, for API-based embedding)",
        default=defaults.api_base or "",
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
        api_base=api_base,
        api_key=api_key,
        device=device,
        batch_size=batch_size,
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

    table.add_row("Model Name", config.embedding.model_name)
    table.add_row("API Base", config.embedding.api_base or "(not set)")
    table.add_row("API Key", _mask_key(config.embedding.api_key))
    table.add_row("Device", config.embedding.device)
    table.add_row("Batch Size", str(config.embedding.batch_size))

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
@click.pass_context
def milvus(ctx: click.Context, collection: str | None) -> None:
    """Manage Milvus vector store configuration."""
    # If a subcommand is being invoked, let it handle things.
    if ctx.invoked_subcommand is not None:
        return

    config = load_config()

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
@click.pass_context
def chroma(ctx: click.Context, collection_name: str | None) -> None:
    """Manage ChromaDB vector store configuration."""
    # If a subcommand is being invoked, let it handle things.
    if ctx.invoked_subcommand is not None:
        return

    config = load_config()

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
    directory = Path(dir_path).resolve()
    config = load_config()
    collection_name = collection or config.milvus.default_collection

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
) -> None:
    """Search the knowledge base using hybrid search (vector + BM25).
    
    QUERY: The search query string.
    
    Examples:
    
        chunky search "machine learning"
        chunky search "neural networks" -c my_collection -k 10
        chunky search "transformer attention" -vw 0.7 -bw 0.3 -f rrf
    """
    config = load_config()
    collection_name = collection or config.milvus.default_collection

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
        
        results = manager.search(
            query=query,
            collection_name=collection_name,
            top_k=top_k,
            vector_weight=vector_weight,
            bm25_weight=bm25_weight,
            fusion_method=fusion_method,
        )
        
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
