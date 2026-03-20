"""Search manager for hybrid search combining vector and BM25."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from chunky.config.settings import ChunkyConfig, load_config
from chunky.embedding.embedder import SentenceTransformerEmbedder, BagOfWordsEmbedder
from chunky.search.bm25_engine import BM25Engine, BM25Result
from chunky.search.hybrid_search import HybridSearcher, HybridSearchResult

logger = logging.getLogger(__name__)


class SearchManager:
    """Manages hybrid search operations."""

    def __init__(self, config: ChunkyConfig | None = None) -> None:
        self.config = config or load_config()
        self._vector_store: Any = None
        self._embedder: Any = None
        self._bm25: BM25Engine | None = None
        self._hybrid_searcher: HybridSearcher | None = None
        self._bm25_cache_dir = Path.home() / ".chunky" / "bm25_cache"
        self._bm25_cache_dir.mkdir(parents=True, exist_ok=True)

    def connect(self) -> None:
        from chunky.vectorstore.chroma_store import ChromaStore
        from chunky.vectorstore.milvus_store import MilvusStore

        if self.config.vector_store_type == "chroma":
            self._vector_store = ChromaStore(self.config.chroma)
        else:
            self._vector_store = MilvusStore(self.config.milvus)
        
        self._vector_store.connect()
        
        if self.config.test_mode or self.config.embedding.model_name == "bag-of-words":
            dim = self.config.chroma.dim if self.config.vector_store_type == "chroma" else self.config.milvus.dim
            self._embedder = BagOfWordsEmbedder(dim=dim)
        else:
            self._embedder = SentenceTransformerEmbedder(self.config.embedding)
        
        self._bm25 = BM25Engine(algorithm="bm25okapi")
        self._hybrid_searcher = HybridSearcher(vector_weight=0.5, bm25_weight=0.5, fusion_method="rrf")

    def build_index(self, collection_name: str, force_rebuild: bool = False) -> int:
        if self._vector_store is None or self._bm25 is None:
            raise RuntimeError("Not connected. Call connect() first.")

        cache_file = self._bm25_cache_dir / f"{self.config.vector_store_type}_{collection_name}.pkl"
        if cache_file.exists() and not force_rebuild:
            self._bm25.load(str(cache_file))
            return len(self._bm25._chunks)

        chunks = self._vector_store.get_all_chunks(collection_name)
        if not chunks:
            return 0

        count = self._bm25.build_index(chunks)
        self._bm25.save(str(cache_file))
        return count

    def search(self, query: str, collection_name: str, top_k: int = 10,
               vector_weight: float = 0.5, bm25_weight: float = 0.5,
               fusion_method: str = "rrf") -> list[HybridSearchResult]:
        if self._vector_store is None or self._embedder is None:
            raise RuntimeError("Not connected. Call connect() first.")

        if not self._vector_store.collection_exists(collection_name):
            return []

        if self._hybrid_searcher:
            self._hybrid_searcher.vector_weight = vector_weight
            self._hybrid_searcher.bm25_weight = bm25_weight
            self._hybrid_searcher.fusion_method = fusion_method
        else:
            self._hybrid_searcher = HybridSearcher(vector_weight=vector_weight, bm25_weight=bm25_weight, fusion_method=fusion_method)

        vector_results = self._vector_search(query, collection_name, top_k * 2)
        bm25_results = self._bm25_search(query, top_k * 2)
        return self._hybrid_searcher.search(query=query, vector_results=vector_results, bm25_results=bm25_results, top_k=top_k)

    def _vector_search(self, query: str, collection_name: str, top_k: int) -> list[dict[str, Any]]:
        if self._vector_store is None or self._embedder is None:
            return []
        try:
            from chunky.utils.models import Chunk
            temp_chunk = Chunk(text=query, source_file="query", chunk_index=0)
            self._embedder.embed([temp_chunk])
            query_embedding = temp_chunk.embedding
            if query_embedding is None:
                return []
            query_vec = query_embedding.tolist() if hasattr(query_embedding, 'tolist') else list(query_embedding)
            if self.config.vector_store_type == "chroma":
                return self._chroma_vector_search(collection_name, query_vec, top_k)
            else:
                return self._milvus_vector_search(collection_name, query_vec, top_k)
        except Exception as e:
            logger.error("Vector search failed: %s", e)
            return []

    def _chroma_vector_search(self, collection_name: str, query_vec: list, top_k: int) -> list[dict[str, Any]]:
        try:
            collection = self._vector_store._get_collection(collection_name)
            results = collection.query(query_embeddings=[query_vec], n_results=top_k, include=["documents", "metadatas", "distances"])
            vector_results = []
            if results and results.get('ids') and len(results['ids']) > 0:
                for i in range(len(results['ids'][0])):
                    vector_results.append({
                        "chunk_id": results['ids'][0][i],
                        "text": results['documents'][0][i] if i < len(results['documents'][0]) else "",
                        "source_file": results['metadatas'][0][i].get("source_file", "") if i < len(results['metadatas'][0]) else "",
                        "chunk_index": results['metadatas'][0][i].get("chunk_index", 0) if i < len(results['metadatas'][0]) else 0,
                        "distance": results['distances'][0][i] if i < len(results['distances'][0]) else 0.0,
                        "labels": results['metadatas'][0][i].get("labels", "[]") if i < len(results['metadatas'][0]) else "[]",
                        "topics": results['metadatas'][0][i].get("topics", "[]") if i < len(results['metadatas'][0]) else "[]",
                    })
            return vector_results
        except Exception as e:
            logger.error("ChromaDB vector search failed: %s", e)
            return []

    def _milvus_vector_search(self, collection_name: str, query_vec: list, top_k: int) -> list[dict[str, Any]]:
        try:
            client = self._vector_store._get_client()
            results = client.search(collection_name=collection_name, data=[query_vec], limit=top_k,
                                   output_fields=["id", "text", "source_file", "chunk_index", "labels", "topics"])
            vector_results = []
            if results and len(results) > 0 and len(results[0]) > 0:
                for item in results[0]:
                    entity = item.get("entity", {})
                    vector_results.append({
                        "chunk_id": str(item.get("id", "")),
                        "text": entity.get("text", ""),
                        "source_file": entity.get("source_file", ""),
                        "chunk_index": entity.get("chunk_index", 0),
                        "distance": item.get("distance", 0.0),
                        "labels": entity.get("labels", "[]"),
                        "topics": entity.get("topics", "[]"),
                    })
            return vector_results
        except Exception as e:
            logger.error("Milvus vector search failed: %s", e)
            return []

    def _bm25_search(self, query: str, top_k: int) -> list[BM25Result]:
        if self._bm25 is None or len(self._bm25._chunks) == 0:
            return []
        return self._bm25.search(query, top_k=top_k)

    def close(self) -> None:
        if self._vector_store:
            self._vector_store.close()
        self._vector_store = None
        self._embedder = None
        self._bm25 = None
        self._hybrid_searcher = None

    def get_collections(self) -> list[str]:
        if self._vector_store is None:
            raise RuntimeError("Not connected. Call connect() first.")
        try:
            if self.config.vector_store_type == "chroma":
                client = self._vector_store._get_client()
                collections = client.list_collections()
                return [c.name for c in collections]
            else:
                client = self._vector_store._get_client()
                return []  # MilvusClient doesn't have list_collections in older versions
        except Exception as e:
            logger.error("Failed to list collections: %s", e)
            return []
