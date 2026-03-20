"""Topic modeling for chunk collections.

Provides class-based topic modelers (LDA and BERTopic) along with a
factory function for convenient instantiation.  The legacy functional
API (``assign_topics``) is retained for backward compatibility.
"""

from __future__ import annotations

import logging
from typing import Union

import numpy as np
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer

from chunky.utils.models import Chunk

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# LDA-based topic modeler
# ---------------------------------------------------------------------------

class LDAModeler:
    """Latent Dirichlet Allocation topic modeler backed by scikit-learn.

    Parameters
    ----------
    n_topics : int
        Number of topics to extract.
    n_words : int
        Number of top words per topic used as the topic label.
    """

    def __init__(self, n_topics: int = 10, n_words: int = 5) -> None:
        self.n_topics = n_topics
        self.n_words = n_words

    # -- public API ---------------------------------------------------------

    def fit_transform(self, chunks: list[Chunk]) -> list[Chunk]:
        """Fit LDA on *chunks* and populate each chunk's ``topics`` field.

        Returns the same list of chunks with ``chunk.topics`` set to the
        top words of the dominant topic for that chunk.
        """
        if not chunks:
            return chunks

        texts = [c.text for c in chunks]
        logger.info(
            "LDAModeler: fitting %d topics on %d chunks",
            self.n_topics,
            len(chunks),
        )

        # Vectorize — use max_df=1.0 to avoid pruning terms that appear in
        # every document (common when chunks are similar or duplicated).
        vectorizer = CountVectorizer(max_features=5000, max_df=1.0, min_df=1)
        doc_term = vectorizer.fit_transform(texts)
        feature_names = vectorizer.get_feature_names_out()

        if len(feature_names) == 0:
            logger.warning("No features extracted from chunks — skipping topic modeling")
            return chunks

        # Clamp n_topics so it does not exceed the number of documents or
        # features — LDA requires n_components <= min(n_docs, n_features).
        actual_topics = max(1, min(self.n_topics, doc_term.shape[0], len(feature_names)))

        lda = LatentDirichletAllocation(
            n_components=actual_topics,
            random_state=42,
            max_iter=20,
        )
        doc_topics = lda.fit_transform(doc_term)  # (n_docs, n_topics)

        # Collect the top words for each topic
        topic_words: list[list[str]] = []
        for topic_dist in lda.components_:
            top_indices = topic_dist.argsort()[: -self.n_words - 1 : -1]
            topic_words.append([feature_names[i] for i in top_indices])

        # Assign the dominant topic's words to each chunk
        for i, chunk in enumerate(chunks):
            dominant = int(np.argmax(doc_topics[i]))
            chunk.topics = topic_words[dominant]
            logger.debug("Chunk %d -> topic %d: %s", i, dominant, chunk.topics)

        return chunks


# ---------------------------------------------------------------------------
# BERTopic-based topic modeler
# ---------------------------------------------------------------------------

class BERTopicModeler:
    """Topic modeler backed by `BERTopic <https://maartengr.github.io/BERTopic/>`_.

    The ``bertopic`` library is imported lazily so the rest of the package
    works even when BERTopic is not installed.

    Parameters
    ----------
    n_topics : int or "auto"
        Number of topics.  Pass ``"auto"`` to let BERTopic decide.
    """

    def __init__(self, n_topics: int | str = "auto") -> None:
        self.n_topics = n_topics

    # -- public API ---------------------------------------------------------

    def fit_transform(self, chunks: list[Chunk]) -> list[Chunk]:
        """Fit BERTopic on *chunks* and populate each chunk's ``topics`` field.

        If ``bertopic`` is not installed an :class:`ImportError` is logged and
        the method falls back to :class:`LDAModeler` transparently.
        """
        if not chunks:
            return chunks

        try:
            from bertopic import BERTopic  # noqa: WPS433 — lazy import by design
        except ImportError:
            logger.warning(
                "BERTopic not installed. Install with: pip install 'chunky[bertopic]'"
            )
            logger.warning("Falling back to LDA.")
            fallback = LDAModeler(
                n_topics=self.n_topics if isinstance(self.n_topics, int) else 10,
            )
            return fallback.fit_transform(chunks)

        texts = [c.text for c in chunks]
        nr_topics = self.n_topics if isinstance(self.n_topics, int) else None
        logger.info(
            "BERTopicModeler: fitting on %d chunks (nr_topics=%s)",
            len(chunks),
            nr_topics,
        )

        model = BERTopic(nr_topics=nr_topics, verbose=False)
        topics_list, _ = model.fit_transform(texts)

        topic_info = model.get_topics()
        for i, chunk in enumerate(chunks):
            tid = topics_list[i]
            if tid == -1 or tid not in topic_info:
                chunk.topics = []
            else:
                chunk.topics = [word for word, _ in topic_info[tid][:5]]
            logger.debug("Chunk %d -> BERTopic %d: %s", i, tid, chunk.topics)

        return chunks


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def get_modeler(method: str = "lda", **kwargs) -> LDAModeler | BERTopicModeler:
    """Return a topic modeler instance for the given *method*.

    Parameters
    ----------
    method : str
        ``"lda"`` or ``"bertopic"``.
    **kwargs
        Forwarded to the chosen modeler's ``__init__``.

    Raises
    ------
    ValueError
        If *method* is not recognised.
    """
    method = method.lower()
    if method == "lda":
        return LDAModeler(**kwargs)
    if method in ("bertopic", "bert"):
        return BERTopicModeler(**kwargs)
    raise ValueError(
        f"Unknown topic modeling method: {method!r}. Choose 'lda' or 'bertopic'."
    )


# ---------------------------------------------------------------------------
# Legacy functional API (kept for backward compatibility with the pipeline)
# ---------------------------------------------------------------------------

def assign_topics(
    chunks: list[Chunk],
    method: str = "lda",
    n_topics: int = 10,
) -> list[Chunk]:
    """Assign topic words to chunks using LDA or BERTopic.

    This is a thin wrapper around :func:`get_modeler` that mirrors the
    original functional interface used by the pipeline runner.

    Args:
        chunks: List of Chunk objects.
        method: ``"lda"`` or ``"bertopic"``.
        n_topics: Number of topics to extract.

    Returns:
        The same list of chunks with topics populated.
    """
    if not chunks:
        return chunks

    logger.info(
        "Assigning topics using %s (n_topics=%d) to %d chunks",
        method,
        n_topics,
        len(chunks),
    )

    if method == "bertopic":
        modeler: Union[LDAModeler, BERTopicModeler] = BERTopicModeler(n_topics=n_topics)
    else:
        modeler = LDAModeler(n_topics=n_topics)

    return modeler.fit_transform(chunks)


def assign_topics_lda_batch(chunks: list[Chunk], n_topics: int = 10) -> list[Chunk]:
    """Assign LDA topics to chunks using batch processing.

    This function is specifically designed for the streaming pipeline where
    chunks are processed in batches. It uses LDA to assign topic labels
    to all chunks at once.

    Args:
        chunks: List of Chunk objects (can be empty or with partial data).
        n_topics: Number of topics to extract.

    Returns:
        The same list of chunks with topics populated.
    """
    if not chunks:
        logger.warning("No chunks provided for LDA topic modeling")
        return chunks

    logger.info(
        "LDA batch topic modeling: %d chunks, %d topics",
        len(chunks),
        n_topics,
    )

    # Use the LDA modeler
    modeler = LDAModeler(n_topics=n_topics)
    return modeler.fit_transform(chunks)
