"""Recursive character text splitting for chunking documents."""

from __future__ import annotations

import logging
import re

from chunky.utils.models import Chunk

logger = logging.getLogger(__name__)

# Separators ordered from coarsest to finest granularity:
#   paragraph boundaries -> sentence boundaries -> word boundaries -> character
_SEPARATORS = ["\n\n", "\n", ". ", ".", " ", ""]


def _split_by_separator(text: str, separator: str) -> list[str]:
    """Split text by a separator, keeping non-empty pieces."""
    if separator == "":
        return list(text)
    return [part for part in text.split(separator) if part]


def _recursive_split(text: str, chunk_size: int, separators: list[str]) -> list[str]:
    """Recursively split *text* using progressively finer separators.

    The algorithm tries the coarsest separator first.  Pieces that fit within
    *chunk_size* are accumulated; pieces that are still too large are split
    again with the next finer separator.
    """
    if len(text) <= chunk_size:
        return [text]

    sep = separators[0]
    remaining_seps = separators[1:] if len(separators) > 1 else separators

    pieces = _split_by_separator(text, sep)

    chunks: list[str] = []
    current = ""

    for piece in pieces:
        # Determine what the combined text would be
        candidate = (current + sep + piece) if current else piece

        if len(candidate) <= chunk_size:
            current = candidate
        else:
            # Flush current buffer if non-empty
            if current:
                chunks.append(current)
            # If this single piece exceeds chunk_size, split it further
            if len(piece) > chunk_size:
                sub_chunks = _recursive_split(piece, chunk_size, remaining_seps)
                chunks.extend(sub_chunks)
                current = ""
            else:
                current = piece

    if current:
        chunks.append(current)

    return chunks


class TextSplitter:
    """Recursive character-level text splitter.

    Splits on paragraph boundaries first (``\\n\\n``), then line breaks,
    then sentence-ending periods, then individual words, and finally
    individual characters.  Chunk overlap is applied by prepending the
    tail of the preceding chunk to the next one, snapping to the nearest
    word boundary to avoid mid-word cuts.

    Parameters
    ----------
    chunk_size:
        Maximum number of **characters** per chunk.  Defaults to 512.
    chunk_overlap:
        Number of overlapping characters between consecutive chunks.
        Defaults to 50.
    """

    def __init__(self, chunk_size: int = 512, chunk_overlap: int = 50) -> None:
        if chunk_size <= 0:
            raise ValueError(f"chunk_size must be positive, got {chunk_size}")
        if chunk_overlap < 0:
            raise ValueError(f"chunk_overlap must be non-negative, got {chunk_overlap}")
        if chunk_overlap >= chunk_size:
            raise ValueError(
                f"chunk_overlap ({chunk_overlap}) must be less than chunk_size ({chunk_size})"
            )
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def split(self, text: str, source_file: str) -> list[Chunk]:
        """Split *text* into :class:`Chunk` objects.

        Parameters
        ----------
        text:
            The full document text to split.
        source_file:
            Path or identifier of the source document (stored in each
            ``Chunk.source_file``).

        Returns
        -------
        list[Chunk]
            Ordered list of chunks with ``chunk_index`` starting at 0.
        """
        if not text or not text.strip():
            return []

        # Normalize whitespace slightly -- collapse runs of 3+ newlines
        text = re.sub(r"\n{3,}", "\n\n", text)

        raw_chunks = _recursive_split(text, self.chunk_size, _SEPARATORS)
        logger.debug(
            "Recursive split produced %d raw pieces from %s",
            len(raw_chunks),
            source_file,
        )

        # Apply overlap: prepend tail of previous chunk to each subsequent chunk
        final_texts: list[str] = []
        for i, chunk_text in enumerate(raw_chunks):
            if i == 0 or self.chunk_overlap <= 0:
                final_texts.append(chunk_text)
            else:
                prev = raw_chunks[i - 1]
                overlap_text = prev[-self.chunk_overlap :]
                # Snap to word boundary to avoid slicing mid-word
                space_idx = overlap_text.find(" ")
                if space_idx != -1:
                    overlap_text = overlap_text[space_idx + 1 :]
                merged = overlap_text + " " + chunk_text
                # Trim back to chunk_size if overlap pushed it over
                if len(merged) > self.chunk_size:
                    merged = merged[: self.chunk_size]
                final_texts.append(merged)

        results = [
            Chunk(
                text=t.strip(),
                source_file=source_file,
                chunk_index=idx,
            )
            for idx, t in enumerate(final_texts)
            if t.strip()
        ]
        logger.debug(
            "Produced %d chunks (overlap=%d) from %s",
            len(results),
            self.chunk_overlap,
            source_file,
        )
        return results


# ------------------------------------------------------------------
# Backward-compatible free function
# ------------------------------------------------------------------

def chunk_text(
    text: str,
    source_file: str,
    chunk_size: int = 512,
    chunk_overlap: int = 64,
) -> list[Chunk]:
    """Split text into chunks using recursive character splitting.

    This is a convenience wrapper around :class:`TextSplitter` kept for
    backward compatibility with existing call sites (e.g. the pipeline
    runner).

    Args:
        text: The full document text to split.
        source_file: Path or name of the source file.
        chunk_size: Maximum number of characters per chunk.
        chunk_overlap: Number of overlapping characters between consecutive chunks.

    Returns:
        List of Chunk objects with proper chunk_index and source_file.
    """
    splitter = TextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return splitter.split(text, source_file)
