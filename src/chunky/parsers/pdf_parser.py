"""PDF file parser using pdfplumber."""

from __future__ import annotations

from chunky.parsers.base import BaseParser


class PdfParser(BaseParser):
    SUPPORTED_EXTENSIONS = {".pdf"}

    def supports(self, file_path: str) -> bool:
        return any(file_path.lower().endswith(ext) for ext in self.SUPPORTED_EXTENSIONS)

    def parse(self, file_path: str) -> str:
        import pdfplumber

        pages: list[str] = []
        with pdfplumber.open(file_path) as pdf:
            for page in pdf.pages:
                text = page.extract_text()
                if text:
                    pages.append(text)
        return "\n\n".join(pages)
