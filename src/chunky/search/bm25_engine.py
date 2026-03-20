"""BM25 search engine for chunky."""

from __future__ import annotations

import logging
from dataclasses import dataclass

from rank_bm25 import BM25Plus, BM25Okapi

from chunky.utils.models import Chunk

logger = logging.getLogger(__name__)


@dataclass
class BM25Result:
    """BM25 search result."""
    chunk_id: str
    text: str
    score: float
    source_file: str
    chunk_index: int


class BM25Engine:
    """BM25 search engine for keyword-based document retrieval."""

    def __init__(self, algorithm: str = "bm25okapi") -> None:
        """Initialize BM25 engine.
        
        Args:
            algorithm: "bm25okapi" or "bm25plus". 
                      BM25Okapi is the standard BM25 algorithm.
                      BM25Plus is a variant with better robustness.
        """
        self.algorithm = algorithm
        self._tokenized_corpus: list[list[str]] = []
        self._bm25: BM25Okapi | BM25Plus | None = None
        self._chunks: list[dict] = []  # Store chunk metadata

    def build_index(self, chunks: list[Chunk]) -> int:
        """Build BM25 index from chunks.
        
        Args:
            chunks: List of chunks to index.
            
        Returns:
            Number of chunks indexed.
        """
        if not chunks:
            logger.warning("No chunks provided for BM25 indexing")
            return 0

        # Tokenize corpus (simple whitespace tokenization + lowercase)
        self._tokenized_corpus = []
        self._chunks = []

        for chunk in chunks:
            # Tokenize: lowercase and split by whitespace
            tokens = chunk.text.lower().split()
            self._tokenized_corpus.append(tokens)
            self._chunks.append({
                "chunk_id": getattr(chunk, 'milvus_id', f"{chunk.source_file}_{chunk.chunk_index}"),
                "text": chunk.text,
                "source_file": chunk.source_file,
                "chunk_index": chunk.chunk_index,
            })

        # Build BM25 index
        if self.algorithm == "bm25plus":
            self._bm25 = BM25Plus(self._tokenized_corpus)
        else:
            self._bm25 = BM25Okapi(self._tokenized_corpus)

        logger.info("Built %s index with %d chunks", self.algorithm, len(self._chunks))
        return len(chunks)

    def search(self, query: str, top_k: int = 10) -> list[BM25Result]:
        """Search for relevant chunks using BM25.
        
        Args:
            query: Search query string.
            top_k: Number of top results to return.
            
        Returns:
            List of BM25Result sorted by relevance score.
        """
        if self._bm25 is None:
            logger.error("BM25 index not built. Call build_index() first.")
            return []

        # Tokenize query
        query_tokens = query.lower().split()
        
        # Get BM25 scores
        scores = self._bm25.get_scores(query_tokens)
        
        # Get top-k indices
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
        
        # Build results
        results = []
        for idx in top_indices:
            if scores[idx] > 0:  # Only return positive scores
                results.append(BM25Result(
                    chunk_id=self._chunks[idx]["chunk_id"],
                    text=self._chunks[idx]["text"],
                    score=float(scores[idx]),
                    source_file=self._chunks[idx]["source_file"],
                    chunk_index=self._chunks[idx]["chunk_index"],
                ))

        logger.info("BM25 search returned %d results for query: %s", len(results), query[:50])
        return results

    def save(self, path: str) -> None:
        """Save BM25 index to disk."""
        import pickle
        
        data = {
            "algorithm": self.algorithm,
            "tokenized_corpus": self._tokenized_corpus,
            "chunks": self._chunks,
        }
        with open(path, 'wb') as f:
            pickle.dump(data, f)
        logger.info("BM25 index saved to: %s", path)

    def load(self, path: str) -> None:
        """Load BM25 index from disk."""
        import pickle
        
        with open(path, 'rb') as f:
            data = pickle.load(f)
        
        self.algorithm = data["algorithm"]
        self._tokenized_corpus = data["tokenized_corpus"]
        self._chunks = data["chunks"]
        
        if self.algorithm == "bm25plus":
            self._bm25 = BM25Plus(self._tokenized_corpus)
        else:
            self._bm25 = BM25Okapi(self._tokenized_corpus)
        
        logger.info("BM25 index loaded from: %s", path)
