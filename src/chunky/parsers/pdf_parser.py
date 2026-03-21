"""PDF file parser using pdfplumber."""

from __future__ import annotations

import logging
import warnings

# Suppress FontBBox warnings BEFORE importing pdfplumber
# These are non-critical PDF metadata issues that don't affect text extraction
warnings.filterwarnings("ignore", message=".*FontBBox.*")
warnings.filterwarnings("ignore", message=".*font descriptor.*")
warnings.filterwarnings("ignore", message=".*cannot be parsed as 4 floats.*")

# Also suppress via logging (some warnings come from pdfminer via logging)
logging.getLogger("pdfminer").setLevel(logging.ERROR)

from chunky.parsers.base import BaseParser


class PdfParser(BaseParser):
    SUPPORTED_EXTENSIONS = {".pdf"}

    def supports(self, file_path: str) -> bool:
        return any(file_path.lower().endswith(ext) for ext in self.SUPPORTED_EXTENSIONS)

    def parse(self, file_path: str) -> str:
        # Import here to ensure warning filters are applied
        import pdfplumber

        pages: list[str] = []
        with pdfplumber.open(file_path) as pdf:
            for page in pdf.pages:
                text = page.extract_text()
                if text:
                    pages.append(text)
        return "\n\n".join(pages)
