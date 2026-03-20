"""DOCX file parser using python-docx."""

from __future__ import annotations

from chunky.parsers.base import BaseParser


class DocxParser(BaseParser):
    SUPPORTED_EXTENSIONS = {".docx", ".doc"}

    def supports(self, file_path: str) -> bool:
        return any(file_path.lower().endswith(ext) for ext in self.SUPPORTED_EXTENSIONS)

    def parse(self, file_path: str) -> str:
        from docx import Document

        doc = Document(file_path)
        paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
        return "\n\n".join(paragraphs)
