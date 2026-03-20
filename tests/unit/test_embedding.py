"""Tests for chunky.embedding.embedder (embed_chunks_test / BagOfWordsEmbedder)."""

from __future__ import annotations

from chunky.embedding.embedder import BagOfWordsEmbedder, embed_chunks_test
from chunky.utils.models import Chunk


def _make_chunk(text: str, index: int = 0) -> Chunk:
    return Chunk(text=text, source_file="test.txt", chunk_index=index)


# ── embed_chunks_test basic behavior ─────────────────────────────


class TestEmbedChunksTest:
    def test_embeddings_are_populated(self):
        chunks = [
            _make_chunk("Machine learning is great.", 0),
            _make_chunk("Deep learning uses neural networks.", 1),
        ]
        result = embed_chunks_test(chunks, dim=64)
        assert result is chunks  # modifies in-place and returns same list
        for chunk in result:
            assert isinstance(chunk.embedding, list)
            assert len(chunk.embedding) > 0

    def test_embedding_dimension_matches(self):
        chunks = [
            _make_chunk("Alpha beta gamma delta.", 0),
            _make_chunk("Epsilon zeta eta theta.", 1),
            _make_chunk("Iota kappa lambda mu.", 2),
        ]
        embed_chunks_test(chunks, dim=128)
        for chunk in chunks:
            assert len(chunk.embedding) == 128

    def test_custom_dimension(self):
        chunks = [
            _make_chunk("Hello world example text.", 0),
            _make_chunk("Another piece of text here.", 1),
        ]
        embed_chunks_test(chunks, dim=256)
        for chunk in chunks:
            assert len(chunk.embedding) == 256

    def test_embeddings_are_floats(self):
        chunks = [
            _make_chunk("Test embedding generation.", 0),
            _make_chunk("Another test sentence.", 1),
        ]
        embed_chunks_test(chunks, dim=32)
        for chunk in chunks:
            for val in chunk.embedding:
                assert isinstance(val, float)


# ── Empty chunk list ─────────────────────────────────────────────


class TestEmbedChunksTestEmpty:
    def test_empty_list_returns_empty(self):
        result = embed_chunks_test([], dim=128)
        assert result == []

    def test_empty_list_via_class(self):
        embedder = BagOfWordsEmbedder(dim=64)
        result = embedder.embed([])
        assert result == []


# ── BagOfWordsEmbedder class ─────────────────────────────────────


class TestBagOfWordsEmbedder:
    def test_embed_populates_embeddings(self):
        embedder = BagOfWordsEmbedder(dim=64)
        chunks = [
            _make_chunk("Natural language processing.", 0),
            _make_chunk("Computer vision deep learning.", 1),
        ]
        result = embedder.embed(chunks)
        assert result is chunks
        for chunk in result:
            assert len(chunk.embedding) == 64

    def test_get_dim(self):
        embedder = BagOfWordsEmbedder(dim=256)
        assert embedder.get_dim() == 256

    def test_single_chunk_embedding(self):
        embedder = BagOfWordsEmbedder(dim=32)
        chunks = [_make_chunk("Single document for testing embeddings.", 0)]
        embedder.embed(chunks)
        # With a single doc, TruncatedSVD will clamp components
        assert len(chunks[0].embedding) == 32

    def test_dim_larger_than_corpus_pads_with_zeros(self):
        embedder = BagOfWordsEmbedder(dim=500)
        chunks = [
            _make_chunk("Short text.", 0),
            _make_chunk("Another short text.", 1),
        ]
        embedder.embed(chunks)
        # The dim should still be 500 due to zero padding
        for chunk in chunks:
            assert len(chunk.embedding) == 500

    def test_different_texts_produce_different_embeddings(self):
        embedder = BagOfWordsEmbedder(dim=64)
        chunks = [
            _make_chunk("Machine learning algorithms and data.", 0),
            _make_chunk("Cooking recipes and kitchen utensils.", 1),
        ]
        embedder.embed(chunks)
        # The two embeddings should not be identical
        assert chunks[0].embedding != chunks[1].embedding
