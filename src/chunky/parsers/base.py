"""Base parser interface for chunky."""

from __future__ import annotations

from abc import ABC, abstractmethod


class BaseParser(ABC):
    @abstractmethod
    def parse(self, file_path: str) -> str:
        """Parse a file and return its text content."""
        ...

    @abstractmethod
    def supports(self, file_path: str) -> bool:
        """Check if this parser supports the given file."""
        ...
