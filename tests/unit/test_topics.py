"""Tests for chunky.topics.modeler (assign_topics with LDA)."""

from __future__ import annotations

from chunky.topics.modeler import LDAModeler, assign_topics, get_modeler
from chunky.utils.models import Chunk


def _make_chunk(text: str, index: int = 0) -> Chunk:
    return Chunk(text=text, source_file="test.txt", chunk_index=index)


# ── assign_topics with LDA on simple texts ───────────────────────


class TestAssignTopicsLDA:
    def test_assigns_topics_to_chunks(self):
        chunks = [
            _make_chunk("Machine learning algorithms are used in data science.", 0),
            _make_chunk("Neural networks enable deep learning applications.", 1),
            _make_chunk("Database systems manage structured data efficiently.", 2),
            _make_chunk("Web servers handle HTTP requests and responses.", 3),
            _make_chunk("Cloud computing provides scalable infrastructure.", 4),
        ]
        result = assign_topics(chunks, method="lda", n_topics=2)
        assert result is chunks
        for chunk in result:
            assert isinstance(chunk.topics, list)
            assert len(chunk.topics) > 0

    def test_topic_words_are_strings(self):
        chunks = [
            _make_chunk("Python programming language syntax variables", 0),
            _make_chunk("JavaScript framework React component rendering", 1),
            _make_chunk("SQL database query optimization indexing", 2),
        ]
        assign_topics(chunks, method="lda", n_topics=2)
        for chunk in chunks:
            for topic_word in chunk.topics:
                assert isinstance(topic_word, str)


# ── Topics are populated on each chunk ───────────────────────────


class TestTopicsPopulated:
    def test_all_chunks_get_topics(self):
        texts = [
            "Machine learning algorithms classification regression",
            "Database systems relational SQL queries indexing",
            "Web development frontend backend JavaScript React",
            "Cloud computing infrastructure deployment containers",
            "Natural language processing tokenization parsing",
            "Computer vision image recognition object detection",
            "Operating systems process scheduling memory management",
            "Network protocols TCP UDP routing packets",
        ]
        chunks = [_make_chunk(t, i) for i, t in enumerate(texts)]
        assign_topics(chunks, method="lda", n_topics=3)
        for chunk in chunks:
            assert hasattr(chunk, "topics")
            assert isinstance(chunk.topics, list)
            assert len(chunk.topics) > 0

    def test_topics_populated_via_lda_modeler_directly(self):
        modeler = LDAModeler(n_topics=2, n_words=3)
        chunks = [
            _make_chunk("Computer science algorithms data structures", 0),
            _make_chunk("Biology genetics DNA protein synthesis", 1),
            _make_chunk("Mathematics calculus linear algebra vectors", 2),
        ]
        result = modeler.fit_transform(chunks)
        assert result is chunks
        for chunk in result:
            assert len(chunk.topics) > 0
            assert len(chunk.topics) <= 3  # n_words=3


# ── Empty chunk list ─────────────────────────────────────────────


class TestTopicsEmptyChunks:
    def test_assign_topics_empty_list(self):
        result = assign_topics([], method="lda", n_topics=5)
        assert result == []

    def test_lda_modeler_empty_list(self):
        modeler = LDAModeler(n_topics=5)
        result = modeler.fit_transform([])
        assert result == []


# ── get_modeler factory ──────────────────────────────────────────


class TestGetModeler:
    def test_lda_method(self):
        modeler = get_modeler("lda")
        assert isinstance(modeler, LDAModeler)

    def test_unknown_method_raises(self):
        import pytest

        with pytest.raises(ValueError, match="Unknown topic modeling method"):
            get_modeler("unknown_method")


# ── n_topics clamping ────────────────────────────────────────────


class TestNTopicsClamping:
    def test_n_topics_larger_than_docs(self):
        """LDAModeler should clamp n_topics when it exceeds doc count."""
        chunks = [
            _make_chunk("Only two documents here", 0),
            _make_chunk("This is the second document", 1),
        ]
        # n_topics=50 but only 2 docs -- should not crash
        modeler = LDAModeler(n_topics=50, n_words=3)
        result = modeler.fit_transform(chunks)
        for chunk in result:
            assert isinstance(chunk.topics, list)
