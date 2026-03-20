"""Tests for chunky.llm.labeler (TestLabeler / label_chunks_test)."""

from __future__ import annotations

from chunky.llm.labeler import TestLabeler, label_chunks_test
from chunky.utils.models import Chunk


def _make_chunk(text: str, index: int = 0) -> Chunk:
    return Chunk(text=text, source_file="test.txt", chunk_index=index)


# ── label_chunks_test basic behavior ─────────────────────────────


class TestLabelChunksTest:
    def test_sets_labels_on_chunks(self):
        chunks = [
            _make_chunk("Machine learning is a subset of artificial intelligence."),
            _make_chunk("Python programming language is widely used in data science."),
        ]
        result = label_chunks_test(chunks)
        assert result is chunks  # modifies in-place and returns same list
        for chunk in result:
            assert isinstance(chunk.labels, list)
            assert len(chunk.labels) > 0

    def test_labels_are_lowercase_strings(self):
        chunks = [_make_chunk("Natural Language Processing uses deep learning models.")]
        label_chunks_test(chunks)
        for label in chunks[0].labels:
            assert isinstance(label, str)
            assert label == label.lower()

    def test_respects_top_k(self):
        text = "word1 word2 word3 word4 word5 word6 word7 word8 word9 word10"
        chunks = [_make_chunk(text)]
        label_chunks_test(chunks, top_k=3)
        assert len(chunks[0].labels) <= 3

    def test_empty_chunk_list(self):
        result = label_chunks_test([])
        assert result == []


# ── Chinese + English mixed text ─────────────────────────────────


class TestLabelChunksTestChinese:
    def test_chinese_text_gets_labels(self):
        chunks = [_make_chunk("机器学习是人工智能的一个重要分支领域")]
        label_chunks_test(chunks)
        assert len(chunks[0].labels) > 0

    def test_mixed_chinese_english_text(self):
        chunks = [
            _make_chunk("Python是一种流行的编程语言，广泛用于machine learning和数据分析。")
        ]
        label_chunks_test(chunks)
        labels = chunks[0].labels
        assert len(labels) > 0
        # Should contain either Chinese chars or English words
        for label in labels:
            assert isinstance(label, str)
            assert len(label) > 0

    def test_chinese_stop_words_filtered(self):
        # Text composed mostly of stop words + a few real words
        chunks = [_make_chunk("机器学习是在这个领域的重要技术")]
        label_chunks_test(chunks)
        labels = chunks[0].labels
        # Common stop words like 的, 是, 在, 这 should be filtered
        chinese_stop = set("的了是在不有和人这中大为上个国我以要他时来用们生到作地于出会")
        for label in labels:
            if len(label) == 1:
                assert label not in chinese_stop


# ── TestLabeler class directly ───────────────────────────────────


class TestTestLabelerClass:
    def test_label_chunks_method(self):
        labeler = TestLabeler(top_k=3)
        chunks = [_make_chunk("Deep learning neural network training optimization")]
        result = labeler.label_chunks(chunks)
        assert result is chunks
        assert len(chunks[0].labels) > 0
        assert len(chunks[0].labels) <= 3

    def test_extract_tokens(self):
        tokens = TestLabeler._extract_tokens("Hello World 你好")
        assert "hello" in tokens
        assert "world" in tokens
        assert "你" in tokens
        assert "好" in tokens

    def test_multiple_chunks(self):
        labeler = TestLabeler(top_k=5)
        chunks = [
            _make_chunk("Artificial intelligence research paper", index=0),
            _make_chunk("Database systems and query optimization", index=1),
            _make_chunk("Web development with JavaScript frameworks", index=2),
        ]
        labeler.label_chunks(chunks)
        for chunk in chunks:
            assert len(chunk.labels) > 0
