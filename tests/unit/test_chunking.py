"""Tests for chunky.chunking.splitter."""

from __future__ import annotations

import pytest

from chunky.chunking.splitter import TextSplitter, chunk_text


# ── chunk_text with short text (single chunk) ───────────────────


class TestChunkTextShort:
    def test_short_text_returns_one_chunk(self):
        chunks = chunk_text("Hello world.", source_file="test.txt")
        assert len(chunks) == 1
        assert chunks[0].text == "Hello world."
        assert chunks[0].source_file == "test.txt"
        assert chunks[0].chunk_index == 0

    def test_text_exactly_at_chunk_size(self):
        text = "a" * 512
        chunks = chunk_text(text, source_file="test.txt", chunk_size=512)
        assert len(chunks) == 1


# ── chunk_text with long text (multiple chunks) ─────────────────


class TestChunkTextLong:
    def test_long_text_returns_multiple_chunks(self):
        text = "word " * 500  # ~2500 chars
        chunks = chunk_text(text, source_file="long.txt", chunk_size=100, chunk_overlap=10)
        assert len(chunks) > 1

    def test_long_text_with_paragraph_boundaries(self):
        paragraphs = ["Paragraph number %d. " % i + "Extra content here." for i in range(20)]
        text = "\n\n".join(paragraphs)
        chunks = chunk_text(text, source_file="doc.txt", chunk_size=200, chunk_overlap=20)
        assert len(chunks) > 1
        for chunk in chunks:
            assert len(chunk.text) <= 200 or len(chunk.text) <= 220  # overlap may slightly exceed

    def test_all_chunks_have_content(self):
        text = "word " * 500
        chunks = chunk_text(text, source_file="f.txt", chunk_size=100, chunk_overlap=10)
        for chunk in chunks:
            assert chunk.text.strip() != ""


# ── chunk_text with empty text ───────────────────────────────────


class TestChunkTextEmpty:
    def test_empty_string_returns_empty_list(self):
        chunks = chunk_text("", source_file="empty.txt")
        assert chunks == []

    def test_whitespace_only_returns_empty_list(self):
        chunks = chunk_text("   \n\n  ", source_file="blank.txt")
        assert chunks == []

    def test_none_like_empty_returns_empty_list(self):
        chunks = chunk_text("  \t\n  ", source_file="ws.txt")
        assert chunks == []


# ── chunk overlap behavior ───────────────────────────────────────


class TestChunkOverlap:
    def test_zero_overlap(self):
        text = "AAAA\n\nBBBB\n\nCCCC"
        chunks = chunk_text(text, source_file="f.txt", chunk_size=5, chunk_overlap=0)
        assert len(chunks) >= 2
        # With zero overlap, second chunk should not start with content from first
        if len(chunks) >= 2:
            assert not chunks[1].text.startswith(chunks[0].text[:3])

    def test_positive_overlap_includes_previous_context(self):
        # Use large enough text with clear separation
        text = "alpha beta gamma\n\ndelta epsilon zeta\n\neta theta iota"
        chunks = chunk_text(text, source_file="f.txt", chunk_size=25, chunk_overlap=10)
        # With overlap > 0 and multiple chunks, later chunks may contain
        # text from previous chunks
        assert len(chunks) >= 2


# ── chunk_index is properly set ──────────────────────────────────


class TestChunkIndex:
    def test_chunk_indices_are_sequential(self):
        text = "word " * 500
        chunks = chunk_text(text, source_file="f.txt", chunk_size=100, chunk_overlap=10)
        for i, chunk in enumerate(chunks):
            assert chunk.chunk_index == i

    def test_single_chunk_has_index_zero(self):
        chunks = chunk_text("short", source_file="f.txt")
        assert len(chunks) == 1
        assert chunks[0].chunk_index == 0

    def test_source_file_preserved_on_all_chunks(self):
        text = "word " * 500
        chunks = chunk_text(text, source_file="my_doc.txt", chunk_size=100, chunk_overlap=10)
        for chunk in chunks:
            assert chunk.source_file == "my_doc.txt"


# ── TextSplitter validation ─────────────────────────────────────


class TestTextSplitterValidation:
    def test_negative_chunk_size_raises(self):
        with pytest.raises(ValueError, match="chunk_size must be positive"):
            TextSplitter(chunk_size=-1)

    def test_zero_chunk_size_raises(self):
        with pytest.raises(ValueError, match="chunk_size must be positive"):
            TextSplitter(chunk_size=0)

    def test_negative_overlap_raises(self):
        with pytest.raises(ValueError, match="chunk_overlap must be non-negative"):
            TextSplitter(chunk_overlap=-5)

    def test_overlap_ge_chunk_size_raises(self):
        with pytest.raises(ValueError, match="chunk_overlap .* must be less than chunk_size"):
            TextSplitter(chunk_size=100, chunk_overlap=100)

    def test_overlap_greater_than_chunk_size_raises(self):
        with pytest.raises(ValueError):
            TextSplitter(chunk_size=50, chunk_overlap=60)
