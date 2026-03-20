"""Data models shared across chunky modules."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Chunk:
    """A single chunk of text from a document."""
    text: str
    source_file: str
    chunk_index: int
    metadata: dict = field(default_factory=dict)
    # Populated during pipeline
    labels: list[str] = field(default_factory=list)
    topics: list[str] = field(default_factory=list)
    embedding: list[float] = field(default_factory=list)
    # Milvus-specific fields
    milvus_id: int | None = field(default=None, repr=False)
