"""ChromaDB vector storage for chunks."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import chromadb
from chromadb.config import Settings

from chunky.config.settings import ChromaConfig
from chunky.utils.models import Chunk

logger = logging.getLogger(__name__)


class ChromaStore:
    """Manages a ChromaDB connection for storing chunks."""

    def __init__(self, config: ChromaConfig) -> None:
        self.config = config
        self._client: chromadb.PersistentClient | None = None
        self._collection = None

    def connect(self) -> None:
        """Connect to ChromaDB using PersistentClient."""
        db_path = self.config.persist_directory
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        logger.info("Connecting to ChromaDB at: %s", db_path)
        
        self._client = chromadb.PersistentClient(
            path=db_path,
            settings=Settings(anonymized_telemetry=False, allow_reset=True)
        )
        logger.info("ChromaDB connection established")

    def create_collection(self, collection_name: str, dim: int) -> None:
        """Create a collection."""
        client = self._get_client()
        try:
            existing = client.get_collection(collection_name)
            if existing:
                logger.info("Collection '%s' already exists", collection_name)
                self._collection = existing
                return
        except Exception:
            pass

        self._collection = client.create_collection(
            name=collection_name,
            metadata={"description": "chunky knowledge base"},
            get_or_create=True,
        )
        logger.info("Created/loaded collection '%s'", collection_name)

    def collection_exists(self, collection_name: str) -> bool:
        """Return True if collection exists."""
        client = self._get_client()
        try:
            client.get_collection(collection_name)
            return True
        except Exception:
            return False

    def drop_collection(self, collection_name: str) -> None:
        """Drop a collection."""
        client = self._get_client()
        try:
            client.delete_collection(collection_name)
            logger.info("Dropped collection '%s'", collection_name)
        except Exception as e:
            logger.warning("Failed to drop collection: %s", e)

    def insert_one(self, collection_name: str, chunk: Chunk) -> int:
        """Insert a single chunk."""
        collection = self._get_or_create_collection(collection_name)
        collection.add(
            documents=[chunk.text],
            metadatas=[{
                "source_file": chunk.source_file,
                "chunk_index": chunk.chunk_index,
                "labels": json.dumps(chunk.labels, ensure_ascii=False),
                "topics": json.dumps(chunk.topics, ensure_ascii=False),
            }],
            ids=[f"{chunk.source_file}_{chunk.chunk_index}"],
        )
        return 1

    def insert(self, collection_name: str, chunks: list[Chunk]) -> int:
        """Insert chunks into collection."""
        if not chunks:
            logger.warning("No chunks to insert")
            return 0

        collection = self._get_or_create_collection(collection_name)

        documents = []
        metadatas = []
        ids = []

        for i, chunk in enumerate(chunks):
            documents.append(chunk.text)
            metadatas.append({
                "source_file": chunk.source_file,
                "chunk_index": chunk.chunk_index,
                "labels": json.dumps(chunk.labels, ensure_ascii=False),
                "topics": json.dumps(chunk.topics, ensure_ascii=False),
            })
            ids.append(f"chunk_{i}_{chunk.source_file}_{chunk.chunk_index}")

        collection.add(documents=documents, metadatas=metadatas, ids=ids)
        logger.info("Inserted %d chunks into '%s'", len(chunks), collection_name)
        return len(chunks)

    def query_all(self, collection_name: str, output_fields: list[str] | None = None) -> list[dict]:
        """Return all rows from collection."""
        collection = self._get_collection(collection_name)
        
        try:
            results = collection.get(limit=100000)
            items = []
            
            for i in range(len(results.get('ids', []))):
                metadata = results['metadatas'][i] if i < len(results.get('metadatas', [])) else {}
                items.append({
                    "text": results['documents'][i] if i < len(results.get('documents', [])) else "",
                    "source_file": metadata.get('source_file', ''),
                    "chunk_index": metadata.get('chunk_index', 0),
                    "labels": metadata.get('labels', '[]'),
                    "topics": metadata.get('topics', '[]'),
                    "id": results['ids'][i] if i < len(results.get('ids', [])) else '',
                })
            
            logger.info("Queried %d rows from '%s'", len(items), collection_name)
            return items
        except Exception as e:
            logger.error("Failed to query collection: %s", e)
            return []

    def update_topics(self, collection_name: str, updates: list[dict]) -> int:
        """Update topics for rows."""
        if not updates:
            return 0
            
        collection = self._get_collection(collection_name)
        
        for update in updates:
            chunk_id = update.get("id")
            topics = update.get("topics", [])
            
            if chunk_id:
                try:
                    existing = collection.get(ids=[chunk_id])
                    if existing and existing['metadatas']:
                        metadata = existing['metadatas'][0]
                        metadata['topics'] = json.dumps(topics, ensure_ascii=False)
                        collection.update(ids=[chunk_id], metadatas=[metadata])
                except Exception as e:
                    logger.warning("Failed to update chunk %s: %s", chunk_id, e)
        
        return len(updates)

    def close(self) -> None:
        """Close the connection."""
        self._client = None
        self._collection = None
        logger.info("ChromaDB connection closed")

    def get_all_chunks(self, collection_name: str) -> list[Chunk]:
        """Get all chunks from collection."""
        collection = self._get_collection(collection_name)
        
        try:
            results = collection.get(limit=100000)
            chunks = []
            
            for i in range(len(results.get('ids', []))):
                metadata = results['metadatas'][i] if i < len(results.get('metadatas', [])) else {}
                chunks.append(Chunk(
                    text=results['documents'][i] if i < len(results.get('documents', [])) else "",
                    source_file=metadata.get('source_file', ''),
                    chunk_index=metadata.get('chunk_index', 0),
                    labels=json.loads(metadata.get('labels', '[]')),
                    topics=json.loads(metadata.get('topics', '[]')),
                    embedding=None,
                ))
                chunks[-1].milvus_id = results['ids'][i]
            
            logger.info("Retrieved %d chunks from '%s'", len(chunks), collection_name)
            return chunks
        except Exception as e:
            logger.error("Failed to get chunks: %s", e)
            return []

    def update_lda_topics(self, collection_name: str, chunks: list[Chunk]) -> int:
        """Update LDA topic labels."""
        if not chunks:
            return 0

        updates = []
        for chunk in chunks:
            if hasattr(chunk, 'milvus_id') and chunk.milvus_id:
                updates.append({"id": chunk.milvus_id, "topics": chunk.topics})
        
        if updates:
            return self.update_topics(collection_name, updates)
        return 0

    def _get_client(self) -> chromadb.PersistentClient:
        """Return the active client."""
        if self._client is None:
            raise RuntimeError("ChromaStore is not connected. Call connect() first.")
        return self._client

    def _get_collection(self, collection_name: str) -> chromadb.Collection:
        """Get collection by name."""
        client = self._get_client()
        return client.get_collection(collection_name)

    def _get_or_create_collection(self, collection_name: str) -> chromadb.Collection:
        """Get or create collection."""
        if self._collection and self._collection.name == collection_name:
            return self._collection
        
        client = self._get_client()
        self._collection = client.get_or_create_collection(
            name=collection_name,
            metadata={"description": "chunky knowledge base"},
        )
        return self._collection
