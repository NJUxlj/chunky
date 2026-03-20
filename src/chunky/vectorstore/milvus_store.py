"""Milvus / Milvus-Lite vector storage for chunks."""

from __future__ import annotations

import json
import logging

from pymilvus import CollectionSchema, DataType, FieldSchema, MilvusClient

from chunky.config.settings import MilvusConfig
from chunky.utils.models import Chunk

logger = logging.getLogger(__name__)


class MilvusStore:
    """Manages a Milvus (or Milvus-Lite) connection for storing chunks.

    Uses the modern pymilvus MilvusClient API.  For Milvus-Lite the *uri* is a
    local file path (e.g. ``"milvus_lite.db"``).  For a standalone Milvus server
    the *uri* should be ``"host:port"`` and ``use_lite`` should be ``False``.
    """

    def __init__(self, config: MilvusConfig) -> None:
        self.config = config
        self._client: MilvusClient | None = None

    # ------------------------------------------------------------------
    # Connection
    # ------------------------------------------------------------------

    def connect(self) -> None:
        """Connect to Milvus using the pymilvus MilvusClient API.

        * Milvus-Lite (``use_lite=True``): ``uri`` is treated as a local file
          path and passed directly to ``MilvusClient(uri=...)``.
        * Standalone Milvus (``use_lite=False``): ``uri`` is expected in
          ``host:port`` form, which is converted to
          ``http://host:port`` for the client.
        """
        from pathlib import Path

        if self.config.use_lite:
            # For milvus-lite, ensure we have an absolute path or a proper file path
            uri = self.config.uri
            if not uri.startswith('/') and not uri.startswith('./'):
                # Use a path in the current working directory
                uri = str(Path.cwd() / uri)
            logger.info("Connecting to Milvus-Lite: %s", uri)
        else:
            # Parse host:port from uri
            raw = self.config.uri
            if "://" not in raw:
                uri = f"http://{raw}"
            else:
                uri = raw
            logger.info("Connecting to Milvus server: %s", uri)

        self._client = MilvusClient(uri=uri)
        logger.info("Milvus connection established")

    # ------------------------------------------------------------------
    # Collection management
    # ------------------------------------------------------------------

    def create_collection(self, collection_name: str, dim: int) -> None:
        """Create a collection with the chunky schema.

        Fields
        ------
        * **id** -- INT64, primary key, auto-generated
        * **text** -- VARCHAR(65 535)
        * **source_file** -- VARCHAR(1 024)
        * **chunk_index** -- INT64
        * **labels** -- VARCHAR(2 048) -- JSON-encoded list[str]
        * **topics** -- VARCHAR(2 048) -- JSON-encoded list[str]
        * **embedding** -- FLOAT_VECTOR(*dim*)

        An IVF_FLAT index with ``metric_type="COSINE"`` is built on the
        *embedding* field.
        """
        client = self._get_client()

        if client.has_collection(collection_name):
            logger.info(
                "Collection '%s' already exists -- skipping creation",
                collection_name,
            )
            return

        fields = [
            FieldSchema(
                name="id",
                dtype=DataType.INT64,
                is_primary=True,
                auto_id=True,
            ),
            FieldSchema(
                name="text",
                dtype=DataType.VARCHAR,
                max_length=65535,
            ),
            FieldSchema(
                name="source_file",
                dtype=DataType.VARCHAR,
                max_length=1024,
            ),
            FieldSchema(
                name="chunk_index",
                dtype=DataType.INT64,
            ),
            FieldSchema(
                name="labels",
                dtype=DataType.VARCHAR,
                max_length=2048,
            ),
            FieldSchema(
                name="topics",
                dtype=DataType.VARCHAR,
                max_length=2048,
            ),
            FieldSchema(
                name="embedding",
                dtype=DataType.FLOAT_VECTOR,
                dim=dim,
            ),
        ]
        schema = CollectionSchema(fields=fields, enable_dynamic_field=True)

        index_params = client.prepare_index_params()
        index_params.add_index(
            field_name="embedding",
            index_type="IVF_FLAT",
            metric_type="COSINE",
            params={"nlist": 128},
        )

        client.create_collection(
            collection_name=collection_name,
            schema=schema,
            index_params=index_params,
        )
        logger.info("Created collection '%s' (dim=%d)", collection_name, dim)

    def collection_exists(self, collection_name: str) -> bool:
        """Return ``True`` if the collection already exists in Milvus."""
        return self._get_client().has_collection(collection_name)

    def drop_collection(self, collection_name: str) -> None:
        """Drop (delete) a collection from Milvus."""
        client = self._get_client()
        if client.has_collection(collection_name):
            client.drop_collection(collection_name)
            logger.info("Dropped collection '%s'", collection_name)
        else:
            logger.warning(
                "Collection '%s' does not exist -- nothing to drop",
                collection_name,
            )

    # ------------------------------------------------------------------
    # Insert
    # ------------------------------------------------------------------

    def insert_one(self, collection_name: str, chunk: Chunk) -> int:
        """Insert a single chunk into *collection_name*.

        Returns the primary key of the inserted row.
        """
        client = self._get_client()
        data = [
            {
                "text": chunk.text,
                "source_file": chunk.source_file,
                "chunk_index": chunk.chunk_index,
                "labels": json.dumps(chunk.labels, ensure_ascii=False),
                "topics": json.dumps(chunk.topics, ensure_ascii=False),
                "embedding": chunk.embedding,
            }
        ]
        result = client.insert(collection_name=collection_name, data=data)
        return result.get("insert_count", 1) if isinstance(result, dict) else 1

    def insert(self, collection_name: str, chunks: list[Chunk]) -> int:
        """Insert chunks into *collection_name*.

        ``labels`` and ``topics`` lists are serialised to JSON strings before
        insertion.

        Args:
            collection_name: Target collection.
            chunks: Chunk objects with embeddings already populated.

        Returns:
            Number of successfully inserted rows.
        """
        if not chunks:
            logger.warning("No chunks to insert")
            return 0

        client = self._get_client()

        data = [
            {
                "text": chunk.text,
                "source_file": chunk.source_file,
                "chunk_index": chunk.chunk_index,
                "labels": json.dumps(chunk.labels, ensure_ascii=False),
                "topics": json.dumps(chunk.topics, ensure_ascii=False),
                "embedding": chunk.embedding,
            }
            for chunk in chunks
        ]

        result = client.insert(collection_name=collection_name, data=data)
        inserted = result.get("insert_count", len(chunks)) if isinstance(result, dict) else len(chunks)
        logger.info(
            "Inserted %d chunks into '%s'",
            inserted,
            collection_name,
        )
        return inserted

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------

    def query_all(self, collection_name: str, output_fields: list[str] | None = None) -> list[dict]:
        """Return all rows from *collection_name*.

        Uses an always-true filter expression to fetch everything.
        """
        client = self._get_client()
        if output_fields is None:
            output_fields = ["id", "text", "source_file", "chunk_index", "labels", "topics"]
        results = client.query(
            collection_name=collection_name,
            filter='chunk_index >= 0',
            output_fields=output_fields,
            limit=16384,
        )
        logger.info("Queried %d rows from '%s'", len(results), collection_name)
        return results

    # ------------------------------------------------------------------
    # Update
    # ------------------------------------------------------------------

    def update_topics(self, collection_name: str, updates: list[dict]) -> int:
        """Update the *topics* field for rows identified by *id*.

        Parameters
        ----------
        updates:
            Each dict must contain ``"id"`` (int) and ``"topics"`` (list[str]).

        Returns the number of rows upserted.
        """
        if not updates:
            return 0
        client = self._get_client()
        data = [
            {"id": u["id"], "topics": json.dumps(u["topics"], ensure_ascii=False)}
            for u in updates
        ]
        # Milvus upsert requires all fields; use per-row delete+insert via
        # the simpler approach of deleting then re-inserting just the topics.
        # Actually, MilvusClient supports upsert only when we supply all fields.
        # Instead, we delete old rows and re-insert — but that loses embeddings.
        #
        # Best approach: iterate and use the low-level approach.
        # For MilvusClient, the simplest path is to delete by ids and re-insert
        # full rows.  However, that requires fetching full rows first.
        #
        # Alternative: since we already have the chunks in memory during the
        # pipeline, we can just re-insert from memory.  But the task says
        # "update topics in milvus".
        #
        # Let's use the Milvus "upsert-like" approach: fetch full rows,
        # patch topics, delete old, insert new.

        ids = [u["id"] for u in updates]
        topics_map = {u["id"]: u["topics"] for u in updates}

        # Fetch existing rows
        rows = client.query(
            collection_name=collection_name,
            filter=f"id in {ids}",
            output_fields=["id", "text", "source_file", "chunk_index", "labels", "topics", "embedding"],
            limit=len(ids),
        )

        if not rows:
            logger.warning("No rows found to update")
            return 0

        # Patch topics - only keep defined fields (exclude dynamic fields and auto_id id)
        # Note: auto_id fields should not be inserted
        defined_fields = {"text", "source_file", "chunk_index", "labels", "topics", "embedding"}
        patched = []
        for row in rows:
            clean_row = {k: v for k, v in row.items() if k in defined_fields}
            clean_row["topics"] = json.dumps(topics_map.get(row["id"], []), ensure_ascii=False)
            # labels may already be a string (from milvus); keep as-is
            if isinstance(clean_row.get("labels"), list):
                clean_row["labels"] = json.dumps(clean_row["labels"], ensure_ascii=False)
            patched.append(clean_row)

        # Delete old rows
        client.delete(collection_name=collection_name, ids=ids)

        # Re-insert patched rows
        result = client.insert(collection_name=collection_name, data=patched)
        count = result.get("insert_count", len(patched)) if isinstance(result, dict) else len(patched)
        logger.info("Updated topics for %d rows in '%s'", count, collection_name)
        return count

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def close(self) -> None:
        """Close the Milvus connection."""
        if self._client is not None:
            self._client.close()
            self._client = None
            logger.info("Milvus connection closed")

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # LDA Topic Updates
    # ------------------------------------------------------------------

    def get_all_chunks(self, collection_name: str) -> list[Chunk]:
        """Get all chunks from a collection.

        Returns a list of Chunk objects with text, embedding, and metadata.
        """
        client = self._get_client()

        # Flush and load to ensure data is queryable
        client.flush(collection_name=collection_name)
        client.load_collection(collection_name=collection_name)
        
        # Query all rows with embeddings
        rows = client.query(
            collection_name=collection_name,
            filter="chunk_index >= 0",
            output_fields=["id", "text", "source_file", "chunk_index", "labels", "topics", "embedding"],
            limit=16384,
        )

        chunks = []
        for row in rows:
            # Parse labels and topics from JSON strings
            labels = json.loads(row.get("labels", "[]"))
            topics = json.loads(row.get("topics", "[]"))

            # Get embedding (might be None if not yet inserted)
            embedding = row.get("embedding", [])

            chunk = Chunk(
                text=row["text"],
                source_file=row.get("source_file", ""),
                chunk_index=row.get("chunk_index", 0),
                labels=labels,
                topics=topics,
                embedding=embedding if embedding else None,
            )
            chunk.milvus_id = row.get("id")
            chunks.append(chunk)

        logger.info("Retrieved %d chunks from '%s'", len(chunks), collection_name)
        return chunks

    def update_lda_topics(self, collection_name: str, chunks: list[Chunk]) -> int:
        """Update LDA topic labels for chunks in Milvus.

        This method updates the 'topics' field for each chunk in the collection.

        Args:
            collection_name: Target collection name
            chunks: List of Chunk objects with updated topics

        Returns:
            Number of updated chunks
        """
        if not chunks:
            return 0

        client = self._get_client()

        # Get all chunk IDs and their topics
        updates = []
        for chunk in chunks:
            if hasattr(chunk, 'milvus_id') and chunk.milvus_id is not None:
                updates.append({
                    "id": chunk.milvus_id,
                    "topics": chunk.topics,
                })

        if not updates:
            # If no IDs, try to match by text + source_file + chunk_index
            # First, query all existing chunks
            existing_rows = client.query(
                collection_name=collection_name,
                filter="chunk_index >= 0",
                output_fields=["id", "text", "source_file", "chunk_index"],
                limit=16384,
            )

            # Build lookup by (text, source_file, chunk_index)
            lookup = {}
            for row in existing_rows:
                key = (row["text"], row.get("source_file", ""), row.get("chunk_index", 0))
                lookup[key] = row["id"]

            for chunk in chunks:
                key = (chunk.text, chunk.source_file, chunk.chunk_index)
                if key in lookup:
                    updates.append({
                        "id": lookup[key],
                        "topics": chunk.topics,
                    })

        if not updates:
            logger.warning("No chunks found to update LDA topics")
            return 0

        # Use update_topics to do the actual update
        return self.update_topics(collection_name, updates)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _get_client(self) -> MilvusClient:
        """Return the active client, raising if not connected."""
        if self._client is None:
            raise RuntimeError(
                "MilvusStore is not connected. Call connect() first."
            )
        return self._client
