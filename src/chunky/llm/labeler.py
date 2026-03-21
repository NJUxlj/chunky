"""LLM-based and rule-based chunk labeling.

This module provides two labeler classes:

* :class:`LLMLabeler` -- sends each chunk to an OpenAI-compatible LLM
  endpoint and parses the returned comma-separated labels.
* :class:`TestLabeler` -- a lightweight, offline alternative that
  extracts top-frequency keywords as labels (no LLM call required).

Both classes expose the same ``label_chunks`` interface so they can be
used interchangeably in the pipeline.
"""

from __future__ import annotations

import logging
import re
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
from typing import Callable

import openai

from chunky.config.settings import LLMConfig
from chunky.utils.models import Chunk

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Language detection helpers
# ---------------------------------------------------------------------------

_CJK_RE = re.compile(r"[\u4e00-\u9fff\u3400-\u4dbf]")


def _is_mainly_chinese(text: str) -> bool:
    """Return True when > 30 % of the non-whitespace characters are CJK."""
    stripped = re.sub(r"\s+", "", text)
    if not stripped:
        return False
    cjk_count = len(_CJK_RE.findall(stripped))
    return cjk_count / len(stripped) > 0.30


# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------

_PROMPT_ZH = (
    "请为以下文本片段生成3-5个描述性标签/关键词，用逗号分隔。"
    "只返回标签，不要加任何解释。\n\n"
    "文本：{text}\n\n"
    "标签："
)

_PROMPT_EN = (
    "Generate 3-5 descriptive labels/tags for the following text. "
    "Return only the labels, separated by commas. No explanations.\n\n"
    "Text: {text}\n\n"
    "Labels:"
)


# ---------------------------------------------------------------------------
# LLMLabeler
# ---------------------------------------------------------------------------

class LLMLabeler:
    """Label chunks by sending them to an OpenAI-compatible LLM.

    Works seamlessly with both OpenAI and vLLM endpoints -- vLLM exposes
    an OpenAI-compatible API, so only the ``api_base`` / ``base_url``
    differs.

    Supports concurrent processing using thread pool for improved performance.

    Parameters
    ----------
    config:
        An :class:`LLMConfig` instance carrying endpoint, model, and
        generation parameters.
    """

    def __init__(self, config: LLMConfig) -> None:
        self.config = config
        self._client = openai.OpenAI(
            base_url=config.api_base or None,
            api_key=config.api_key or "EMPTY",
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def label_chunks(
        self,
        chunks: list[Chunk],
        progress_callback: Callable[[], None] | None = None,
    ) -> list[Chunk]:
        """Label every chunk via the configured LLM.

        Each chunk's ``.labels`` list is replaced with the parsed
        response. Chunks are modified **in-place** and also returned
        for convenience.

        Uses thread pool for concurrent processing to speed up labeling.

        Parameters
        ----------
        chunks:
            The chunks to label.
        progress_callback:
            Optional callback function called after each chunk is labeled.
            Must be thread-safe.

        Returns
        -------
        list[Chunk]
            The same list with ``.labels`` populated.
        """
        if not chunks:
            return chunks

        max_workers = max(1, self.config.max_concurrent)
        
        # For small batches, use sequential processing
        if len(chunks) <= max_workers:
            return self._label_chunks_sequential(chunks, progress_callback)
        
        # For large batches, use concurrent processing
        return self._label_chunks_concurrent(chunks, max_workers, progress_callback)

    def _label_chunks_sequential(
        self,
        chunks: list[Chunk],
        progress_callback: Callable[[], None] | None = None,
    ) -> list[Chunk]:
        """Label chunks sequentially (for small batches)."""
        for i, chunk in enumerate(chunks):
            self._label_single_chunk(chunk, i, len(chunks))
            if progress_callback:
                progress_callback()
        return chunks

    def _label_chunks_concurrent(
        self,
        chunks: list[Chunk],
        max_workers: int,
        progress_callback: Callable[[], None] | None = None,
    ) -> list[Chunk]:
        """Label chunks concurrently using thread pool."""
        progress_lock = Lock()
        
        def label_with_index(args: tuple[int, Chunk]) -> tuple[int, Chunk]:
            """Label a chunk and return its index for ordering."""
            idx, chunk = args
            self._label_single_chunk(chunk, idx, len(chunks))
            
            if progress_callback:
                with progress_lock:
                    progress_callback()
            
            return idx, chunk

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks with their original indices
            future_to_idx = {
                executor.submit(label_with_index, (i, chunk)): i
                for i, chunk in enumerate(chunks)
            }
            
            # Collect results as they complete
            results: list[tuple[int, Chunk]] = []
            for future in as_completed(future_to_idx):
                try:
                    idx, chunk = future.result()
                    results.append((idx, chunk))
                except Exception:
                    # If task failed, find the original chunk and set empty labels
                    idx = future_to_idx[future]
                    logger.exception("Failed to label chunk %d", idx)
                    chunks[idx].labels = []
                    results.append((idx, chunks[idx]))
                    if progress_callback:
                        with progress_lock:
                            progress_callback()
        
        # Sort by original index to maintain order
        results.sort(key=lambda x: x[0])
        
        # Copy labels back to original chunks
        for idx, chunk in results:
            chunks[idx] = chunk
        
        return chunks

    def _label_single_chunk(self, chunk: Chunk, index: int, total: int) -> None:
        """Label a single chunk using LLM.
        
        Modifies chunk in-place.
        """
        # Choose prompt language based on the chunk content
        if _is_mainly_chinese(chunk.text):
            prompt = _PROMPT_ZH.format(text=chunk.text)
        else:
            prompt = _PROMPT_EN.format(text=chunk.text)

        try:
            response = self._client.chat.completions.create(
                model=self.config.model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=self.config.max_tokens,
                temperature=self.config.temperature,
            )
            raw = response.choices[0].message.content or ""
            # Parse comma-separated labels, stripping whitespace and
            # empty strings
            labels = [lbl.strip() for lbl in raw.split(",") if lbl.strip()]
            chunk.labels = labels
            logger.debug(
                "Chunk %d/%d labeled: %s", index + 1, total, labels
            )
        except Exception:
            logger.warning(
                "LLM labeling failed for chunk %d, setting empty labels",
                index,
                exc_info=True,
            )
            chunk.labels = []


# ---------------------------------------------------------------------------
# TestLabeler -- rule-based, no LLM required
# ---------------------------------------------------------------------------

# Common stop words for English and Chinese
_STOP_WORDS = frozenset(
    # English
    "a an the is are was were be been being have has had do does did will would "
    "shall should may might can could this that these those it its i me my we our "
    "you your he him his she her they them their what which who whom how when where "
    "why and or but if then else for to of in on at by with from not no nor as so".split()
    +
    # Chinese
    list("的了是在不有和人这中大为上个国我以要他时来用们生到作地于出会")
    + list("就也而对其能所说还可又将与自从之已更做如此被但更然没很当已经最让")
)

_TOKEN_RE = re.compile(r"[\u4e00-\u9fff]|[a-zA-Z]+")


class TestLabeler:
    """Simple keyword-frequency labeler that requires no LLM.

    Useful for local testing, CI pipelines, or environments where an
    LLM endpoint is unavailable.  It tokenizes the chunk text into
    Chinese characters and English words, removes stop words, and
    returns the *top_k* most frequent tokens as labels.

    Parameters
    ----------
    top_k:
        Number of keywords to extract per chunk.  Defaults to 5.
    """

    def __init__(self, top_k: int = 5) -> None:
        self.top_k = top_k

    @staticmethod
    def _extract_tokens(text: str) -> list[str]:
        """Tokenize text into Chinese characters and English words."""
        return [tok.lower() for tok in _TOKEN_RE.findall(text)]

    def label_chunks(self, chunks: list[Chunk]) -> list[Chunk]:
        """Label every chunk using keyword frequency.

        Each chunk's ``.labels`` list is replaced with the top-frequency
        keywords.  Chunks are modified **in-place** and also returned
        for convenience.

        Parameters
        ----------
        chunks:
            The chunks to label.

        Returns
        -------
        list[Chunk]
            The same list with ``.labels`` populated.
        """
        for i, chunk in enumerate(chunks):
            tokens = self._extract_tokens(chunk.text)
            filtered = [t for t in tokens if t not in _STOP_WORDS and len(t) > 1]
            if not filtered:
                # Fallback: use single-char tokens
                filtered = [t for t in tokens if t not in _STOP_WORDS]
            counts = Counter(filtered)
            labels = [word for word, _ in counts.most_common(self.top_k)]
            chunk.labels = labels
            logger.debug(
                "Chunk %d/%d test-labeled: %s", i + 1, len(chunks), labels
            )

        return chunks


# ---------------------------------------------------------------------------
# Backward-compatible free functions
# ---------------------------------------------------------------------------

def label_chunks(
    chunks: list[Chunk],
    config: LLMConfig,
    progress_callback: Callable[[], None] | None = None,
) -> list[Chunk]:
    """Label chunks using an LLM (backward-compatible wrapper).

    Kept so that existing call sites (e.g. the pipeline runner) continue
    to work without changes.
    """
    labeler = LLMLabeler(config)
    return labeler.label_chunks(chunks, progress_callback=progress_callback)


def label_chunks_test(chunks: list[Chunk], top_k: int = 5) -> list[Chunk]:
    """Label chunks using keyword frequency (backward-compatible wrapper)."""
    labeler = TestLabeler(top_k=top_k)
    return labeler.label_chunks(chunks)
