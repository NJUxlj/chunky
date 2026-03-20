"""Parser registry — maps file extensions to parser instances."""

from __future__ import annotations

import os
from pathlib import Path

from rich.console import Console

from chunky.parsers.base import BaseParser
from chunky.parsers.docx_parser import DocxParser
from chunky.parsers.pdf_parser import PdfParser
from chunky.parsers.pptx_parser import PptxParser
from chunky.parsers.text_parser import TextParser

console = Console()

_PARSERS: list[BaseParser] = [
    PdfParser(),
    DocxParser(),
    PptxParser(),
    TextParser(),
]

SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".doc", ".pptx", ".ppt", ".txt", ".md"}


def get_parser(file_path: str) -> BaseParser | None:
    """Get the appropriate parser for a file based on its extension."""
    for parser in _PARSERS:
        if parser.supports(file_path):
            return parser
    return None


def parse_directory(dir_path: str, file_types: str = "all") -> list[tuple[str, str]]:
    """Parse all supported files in a directory.

    Returns list of (file_path, text) tuples.
    """
    root = Path(dir_path)
    if not root.is_dir():
        console.print(f"[red]Error:[/red] '{dir_path}' is not a valid directory.")
        return []

    # Determine which extensions to include
    if file_types == "all":
        allowed = SUPPORTED_EXTENSIONS
    else:
        allowed = {ext.strip() if ext.strip().startswith(".") else f".{ext.strip()}"
                   for ext in file_types.split(",")}

    results: list[tuple[str, str]] = []
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        ext = path.suffix.lower()
        if ext not in allowed:
            continue

        parser = get_parser(str(path))
        if parser is None:
            console.print(f"[yellow]Skipping[/yellow] {path} (no parser available)")
            continue

        try:
            text = parser.parse(str(path))
            if text.strip():
                results.append((str(path), text))
            else:
                console.print(f"[yellow]Skipping[/yellow] {path} (empty content)")
        except Exception as e:
            console.print(f"[red]Error parsing[/red] {path}: {e}")

    return results
