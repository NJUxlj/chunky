"""Tests for chunky.utils.models."""

from __future__ import annotations

from chunky.utils.models import Chunk


class TestChunkCreation:
    def test_basic_creation(self):
        chunk = Chunk(text="hello world", source_file="test.txt", chunk_index=0)
        assert chunk.text == "hello world"
        assert chunk.source_file == "test.txt"
        assert chunk.chunk_index == 0

    def test_creation_with_metadata(self):
        meta = {"page": 1, "section": "intro"}
        chunk = Chunk(text="some text", source_file="doc.pdf", chunk_index=3, metadata=meta)
        assert chunk.metadata == {"page": 1, "section": "intro"}

    def test_creation_with_all_fields(self):
        chunk = Chunk(
            text="full chunk",
            source_file="f.txt",
            chunk_index=5,
            metadata={"k": "v"},
            labels=["label1", "label2"],
            topics=["topic1"],
            embedding=[0.1, 0.2, 0.3],
        )
        assert chunk.labels == ["label1", "label2"]
        assert chunk.topics == ["topic1"]
        assert chunk.embedding == [0.1, 0.2, 0.3]


class TestChunkDefaults:
    def test_default_metadata_is_empty_dict(self):
        chunk = Chunk(text="t", source_file="f", chunk_index=0)
        assert chunk.metadata == {}

    def test_default_labels_is_empty_list(self):
        chunk = Chunk(text="t", source_file="f", chunk_index=0)
        assert chunk.labels == []

    def test_default_topics_is_empty_list(self):
        chunk = Chunk(text="t", source_file="f", chunk_index=0)
        assert chunk.topics == []

    def test_default_embedding_is_empty_list(self):
        chunk = Chunk(text="t", source_file="f", chunk_index=0)
        assert chunk.embedding == []

    def test_default_factory_independence(self):
        """Each Chunk should get its own default list/dict instances."""
        c1 = Chunk(text="a", source_file="f", chunk_index=0)
        c2 = Chunk(text="b", source_file="f", chunk_index=1)
        c1.labels.append("tag")
        assert c2.labels == []  # c2 should not be affected

    def test_mutable_fields_are_independent(self):
        c1 = Chunk(text="a", source_file="f", chunk_index=0)
        c2 = Chunk(text="b", source_file="f", chunk_index=1)
        c1.metadata["key"] = "val"
        assert "key" not in c2.metadata
