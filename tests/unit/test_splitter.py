"""Tests for chunky.chunking.splitter."""

import pytest

from chunky.chunking.splitter import TextSplitter, chunk_text


class TestTextSplitter:
    def test_empty_text(self):
        splitter = TextSplitter(chunk_size=100)
        assert splitter.split("", "test.txt") == []
        assert splitter.split("   ", "test.txt") == []

    def test_short_text_single_chunk(self):
        splitter = TextSplitter(chunk_size=200, chunk_overlap=0)
        text = "Hello world. This is a short text."
        chunks = splitter.split(text, "test.txt")
        assert len(chunks) == 1
        assert chunks[0].text == text
        assert chunks[0].source_file == "test.txt"
        assert chunks[0].chunk_index == 0

    def test_splits_on_paragraph_boundary(self):
        splitter = TextSplitter(chunk_size=50, chunk_overlap=0)
        text = "First paragraph.\n\nSecond paragraph.\n\nThird paragraph."
        chunks = splitter.split(text, "test.txt")
        assert len(chunks) >= 2
        assert chunks[0].chunk_index == 0
        assert chunks[1].chunk_index == 1

    def test_chunk_indices_sequential(self):
        splitter = TextSplitter(chunk_size=30, chunk_overlap=0)
        text = "A " * 100
        chunks = splitter.split(text, "test.txt")
        for i, chunk in enumerate(chunks):
            assert chunk.chunk_index == i

    def test_source_file_preserved(self):
        splitter = TextSplitter(chunk_size=30, chunk_overlap=0)
        text = "A " * 100
        chunks = splitter.split(text, "my/doc.pdf")
        for chunk in chunks:
            assert chunk.source_file == "my/doc.pdf"

    def test_invalid_chunk_size(self):
        with pytest.raises(ValueError):
            TextSplitter(chunk_size=0)
        with pytest.raises(ValueError):
            TextSplitter(chunk_size=-1)

    def test_overlap_must_be_less_than_size(self):
        with pytest.raises(ValueError):
            TextSplitter(chunk_size=100, chunk_overlap=100)
        with pytest.raises(ValueError):
            TextSplitter(chunk_size=100, chunk_overlap=200)

    def test_chunk_text_wrapper(self):
        text = "Hello world. This is a test."
        chunks = chunk_text(text, "file.txt", chunk_size=200)
        assert len(chunks) >= 1
        assert chunks[0].source_file == "file.txt"
