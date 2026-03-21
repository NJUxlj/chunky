"""Pipeline runner -- orchestrates the full knowledge-base build with progress bars."""

from __future__ import annotations

import logging
import os
from pathlib import Path

from rich.console import Console

from chunky.config.settings import ChunkyConfig, load_config
from chunky.progress.manager import ChunkingProgress
from chunky.utils.models import Chunk

logger = logging.getLogger(__name__)

SUPPORTED_EXTENSIONS: set[str] = {".pdf", ".docx", ".doc", ".pptx", ".ppt", ".txt", ".md"}


def get_vector_store(config: ChunkyConfig):
    """Factory function to get the appropriate vector store."""
    if config.vector_store_type == "chroma":
        from chunky.vectorstore.chroma_store import ChromaStore
        return ChromaStore(config.chroma)
    else:
        from chunky.vectorstore.milvus_store import MilvusStore
        return MilvusStore(config.milvus)


def get_default_dim(config: ChunkyConfig) -> int:
    """Get default embedding dimension based on vector store type."""
    if config.vector_store_type == "chroma":
        return config.chroma.dim or 384
    return config.milvus.dim or 512


class PipelineRunner:
    """End-to-end pipeline with 5 progress bars."""

    def __init__(self, config: ChunkyConfig) -> None:
        self.config = config
        self.console = Console()
        self.progress: ChunkingProgress | None = None

    def run(self, directory: str, collection_name: str | None = None) -> None:
        """运行完整的知识库构建流水线"""
        collection = collection_name or (
            self.config.chroma.default_collection if self.config.vector_store_type == "chroma" 
            else self.config.milvus.default_collection
        )

        self.console.print()
        self.console.print(f"[bold cyan]🏗️  Building knowledge base:[/bold cyan] {collection}")
        self.console.print(f"[bold cyan]📁 Source directory:[/bold cyan] {directory}")
        self.console.print(f"[bold cyan]🧪 Test mode:[/bold cyan] {'ON' if self.config.test_mode else 'OFF'}")
        self.console.print()

        self.console.print("[bold]Step 1/6[/bold] Discovering and parsing files ...")
        documents = self._parse_files(directory)
        if not documents:
            self.console.print("[red]No documents found.[/red]")
            return
        self.console.print(f"  Parsed [green]{len(documents)}[/green] documents.\n")

        self.progress = ChunkingProgress(self.console)
        self.progress.start()

        try:
            self.console.print("[bold]Step 2/6[/bold] Splitting text into chunks ...")
            all_chunks = self._chunk_documents_with_progress(documents)
            if not all_chunks:
                self.console.print("[red]No chunks produced.[/red]")
                return
            self.console.print(f"\n  Created [green]{len(all_chunks)}[/green] chunks.\n")

            self.progress.setup_processing(len(all_chunks))

            self.console.print("[bold]Step 3-6/6[/bold] Processing chunks ...")
            dim = self._process_chunks_streaming(all_chunks, collection)
            self.console.print()

            self.console.print("[bold]Step 4/6[/bold] LDA Topic Modeling ...")
            self.progress.setup_lda(len(all_chunks))
            self._assign_topics_with_progress(collection, len(all_chunks))
            self.console.print()

            self.console.print(f"\n[bold green]✅ Done![/bold green] Processed {len(all_chunks)} chunks.\n")
        finally:
            if self.progress:
                self.progress.stop()

    def _parse_files(self, directory: str) -> list[tuple[str, str]]:
        """解析目录中的所有文件"""
        from chunky.parsers.registry import get_parser
        root = Path(directory)
        if not root.is_dir():
            self.console.print(f"[red]Error:[/red] '{directory}' is not a valid directory.")
            return []

        results: list[tuple[str, str]] = []
        for dirpath, _dirnames, filenames in os.walk(root):
            for fname in sorted(filenames):
                fpath = os.path.join(dirpath, fname)
                ext = os.path.splitext(fname)[1].lower()
                if ext not in SUPPORTED_EXTENSIONS:
                    continue
                parser = get_parser(fpath)
                if parser is None:
                    continue
                try:
                    text = parser.parse(fpath)
                    if text and text.strip():
                        results.append((fpath, text))
                except Exception as exc:
                    logger.exception("Failed to parse %s", fpath)
        return results

    def _chunk_documents_with_progress(self, documents: list[tuple[str, str]]) -> list[Chunk]:
        """带进度条的文档 chunking"""
        from chunky.chunking.splitter import chunk_text
        self.progress.setup_chunking(len(documents))
        all_chunks: list[Chunk] = []
        for file_path, text in documents:
            try:
                chunks = chunk_text(text, source_file=file_path)
                all_chunks.extend(chunks)
                self.progress.update_chunking(advance=1)
            except Exception as exc:
                logger.exception("Failed to chunk %s", file_path)
        return all_chunks

    def _process_chunks_streaming(self, chunks: list[Chunk], collection: str) -> int:
        """流式处理 chunks"""
        store = get_vector_store(self.config)
        dim = get_default_dim(self.config)
        try:
            store.connect()
            store.create_collection(collection, dim)
            
            # Step 1: Embedding (sequential, can be slow on CPU)
            self.console.print("  [dim]Embedding chunks...[/dim]")
            for chunk in chunks:
                chunk = self._embed_single_chunk(chunk)
                self.progress.update_embedding(advance=1)
            
            # Step 2: LLM Labeling (concurrent for better performance)
            self.console.print("  [dim]Labeling chunks with LLM (concurrent)...[/dim]")
            chunks = self._label_chunks_concurrent(chunks)
            
            # Step 3: Insert to vector store in batches
            batch_size = 100
            total_inserted = 0
            for i in range(0, len(chunks), batch_size):
                batch = chunks[i : i + batch_size]
                inserted = store.insert(collection, batch)
                total_inserted += inserted
                for _ in range(inserted):
                    self.progress.update_milvus(advance=1)
            
            store_type = "ChromaDB" if self.config.vector_store_type == "chroma" else "Milvus"
            self.console.print(f"\n  [green]Inserted {total_inserted} chunks into {store_type}[/green]")
            return dim
        finally:
            store.close()

    def _embed_single_chunk(self, chunk: Chunk) -> Chunk:
        if self.config.test_mode:
            from chunky.embedding.embedder import embed_chunks_test
            return embed_chunks_test([chunk], dim=self._get_embedding_dim())[0]
        else:
            from chunky.embedding.embedder import APIEmbedder
            embedder = APIEmbedder(self.config.embedding)
            return embedder.embed([chunk])[0]

    def _label_single_chunk(self, chunk: Chunk) -> Chunk:
        """Label a single chunk (kept for compatibility)."""
        if self.config.test_mode:
            from chunky.llm.test_labeler import label_chunks_test
            return label_chunks_test([chunk])[0]
        else:
            from chunky.llm.labeler import label_chunks
            return label_chunks([chunk], self.config.llm)[0]

    def _label_chunks_concurrent(self, chunks: list[Chunk]) -> list[Chunk]:
        """Label all chunks concurrently using thread pool.
        
        Uses LLMLabeler with concurrent processing for better performance.
        """
        if self.config.test_mode:
            # Test mode doesn't benefit from concurrency
            from chunky.llm.test_labeler import TestLabeler
            labeler = TestLabeler(top_k=5)
            return labeler.label_chunks(chunks)
        else:
            from chunky.llm.labeler import LLMLabeler
            labeler = LLMLabeler(self.config.llm)
            
            # Use progress callback for thread-safe progress updates
            def progress_callback():
                self.progress.update_llm_labeling(advance=1)
            
            return labeler.label_chunks(chunks, progress_callback=progress_callback)

    def _assign_topics_with_progress(self, collection: str, total: int) -> None:
        from chunky.topics.modeler import assign_topics_lda_batch
        store = get_vector_store(self.config)
        try:
            store.connect()
            self.console.print("  [cyan]Fetching all chunks...[/cyan]")
            chunks = store.get_all_chunks(collection)
            if not chunks:
                self.console.print("  [yellow]No chunks found[/yellow]")
                return
            self.console.print(f"  [cyan]Found {len(chunks)} chunks, running LDA...[/cyan]")
            chunks_with_topics = assign_topics_lda_batch(chunks)
            for i, chunk in enumerate(chunks_with_topics):
                self.progress.update_lda(advance=1)
                if (i + 1) % 100 == 0:
                    self.console.print(f"  [cyan]LDA: {i + 1}/{len(chunks_with_topics)}[/cyan]")
            store.update_lda_topics(collection, chunks_with_topics)
            self.console.print(f"  [green]LDA complete, updated {len(chunks_with_topics)} chunks[/green]")
        except Exception as exc:
            self.console.print(f"  [red]LDA failed:[/red] {exc}")
            logger.exception("LDA failed")
        finally:
            store.close()

    def _get_embedding_dim(self) -> int:
        return get_default_dim(self.config)


def run_pipeline(dir_path: str, collection_name: str | None = None) -> None:
    """CLI entry point."""
    config = load_config()
    runner = PipelineRunner(config)
    runner.run(dir_path, collection_name=collection_name)
