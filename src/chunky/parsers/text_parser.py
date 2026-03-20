"""Plain text and Markdown file parser with encoding detection."""

from __future__ import annotations

from chunky.parsers.base import BaseParser


class TextParser(BaseParser):
    SUPPORTED_EXTENSIONS = {".txt", ".md"}

    def supports(self, file_path: str) -> bool:
        return any(file_path.lower().endswith(ext) for ext in self.SUPPORTED_EXTENSIONS)

    def parse(self, file_path: str) -> str:
        import chardet

        with open(file_path, "rb") as f:
            raw = f.read()

        detected = chardet.detect(raw)
        encoding = detected.get("encoding") or "utf-8"

        return raw.decode(encoding)
