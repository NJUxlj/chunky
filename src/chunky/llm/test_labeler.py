"""Simple keyword extraction for test mode (no LLM required).

This module re-exports from :mod:`chunky.llm.labeler` for backward
compatibility.  All logic now lives in ``labeler.py``.
"""

from chunky.llm.labeler import TestLabeler, label_chunks_test

__all__ = ["TestLabeler", "label_chunks_test"]
