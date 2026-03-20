"""Tests for chunky.parsers (TextParser, supports(), registry, parse_directory)."""

from __future__ import annotations

from chunky.parsers.docx_parser import DocxParser
from chunky.parsers.pdf_parser import PdfParser
from chunky.parsers.pptx_parser import PptxParser
from chunky.parsers.registry import get_parser, parse_directory
from chunky.parsers.text_parser import TextParser


# ── TextParser with real tmp files ───────────────────────────────


class TestTextParser:
    def test_parse_txt_file(self, tmp_path):
        f = tmp_path / "sample.txt"
        f.write_text("Hello, world!", encoding="utf-8")

        parser = TextParser()
        result = parser.parse(str(f))
        assert result == "Hello, world!"

    def test_parse_md_file(self, tmp_path):
        f = tmp_path / "readme.md"
        f.write_text("# Title\n\nSome markdown content.", encoding="utf-8")

        parser = TextParser()
        result = parser.parse(str(f))
        assert "# Title" in result
        assert "Some markdown content." in result

    def test_parse_utf8_chinese_content(self, tmp_path):
        f = tmp_path / "chinese.txt"
        f.write_text("你好世界", encoding="utf-8")

        parser = TextParser()
        result = parser.parse(str(f))
        assert result == "你好世界"

    def test_supports_txt(self):
        parser = TextParser()
        assert parser.supports("document.txt") is True
        assert parser.supports("NOTES.TXT") is True

    def test_supports_md(self):
        parser = TextParser()
        assert parser.supports("readme.md") is True
        assert parser.supports("CHANGELOG.MD") is True

    def test_does_not_support_other_extensions(self):
        parser = TextParser()
        assert parser.supports("image.png") is False
        assert parser.supports("data.csv") is False
        assert parser.supports("report.pdf") is False


# ── PdfParser.supports() ─────────────────────────────────────────


class TestPdfParserSupports:
    def test_supports_pdf(self):
        parser = PdfParser()
        assert parser.supports("report.pdf") is True
        assert parser.supports("REPORT.PDF") is True

    def test_does_not_support_non_pdf(self):
        parser = PdfParser()
        assert parser.supports("report.txt") is False
        assert parser.supports("report.docx") is False
        assert parser.supports("report.pptx") is False


# ── DocxParser.supports() ────────────────────────────────────────


class TestDocxParserSupports:
    def test_supports_docx(self):
        parser = DocxParser()
        assert parser.supports("report.docx") is True
        assert parser.supports("REPORT.DOCX") is True

    def test_supports_doc(self):
        parser = DocxParser()
        assert parser.supports("old_file.doc") is True
        assert parser.supports("OLD_FILE.DOC") is True

    def test_does_not_support_non_docx(self):
        parser = DocxParser()
        assert parser.supports("report.pdf") is False
        assert parser.supports("notes.txt") is False


# ── PptxParser.supports() ────────────────────────────────────────


class TestPptxParserSupports:
    def test_supports_pptx(self):
        parser = PptxParser()
        assert parser.supports("slides.pptx") is True
        assert parser.supports("SLIDES.PPTX") is True

    def test_supports_ppt(self):
        parser = PptxParser()
        assert parser.supports("old_slides.ppt") is True
        assert parser.supports("OLD_SLIDES.PPT") is True

    def test_does_not_support_non_pptx(self):
        parser = PptxParser()
        assert parser.supports("report.pdf") is False
        assert parser.supports("notes.txt") is False


# ── registry.get_parser() ────────────────────────────────────────


class TestGetParser:
    def test_returns_text_parser_for_txt(self):
        parser = get_parser("notes.txt")
        assert isinstance(parser, TextParser)

    def test_returns_text_parser_for_md(self):
        parser = get_parser("readme.md")
        assert isinstance(parser, TextParser)

    def test_returns_pdf_parser_for_pdf(self):
        parser = get_parser("report.pdf")
        assert isinstance(parser, PdfParser)

    def test_returns_docx_parser_for_docx(self):
        parser = get_parser("report.docx")
        assert isinstance(parser, DocxParser)

    def test_returns_pptx_parser_for_pptx(self):
        parser = get_parser("slides.pptx")
        assert isinstance(parser, PptxParser)

    def test_returns_none_for_unsupported(self):
        parser = get_parser("image.png")
        assert parser is None

    def test_returns_none_for_csv(self):
        parser = get_parser("data.csv")
        assert parser is None


# ── parse_directory() with tmp dir ───────────────────────────────


class TestParseDirectory:
    def test_parse_directory_with_txt_file(self, tmp_path):
        f = tmp_path / "hello.txt"
        f.write_text("Hello from test!", encoding="utf-8")

        results = parse_directory(str(tmp_path))
        assert len(results) == 1
        file_path, text = results[0]
        assert file_path.endswith("hello.txt")
        assert text == "Hello from test!"

    def test_parse_directory_with_multiple_txt_files(self, tmp_path):
        (tmp_path / "a.txt").write_text("File A", encoding="utf-8")
        (tmp_path / "b.txt").write_text("File B", encoding="utf-8")

        results = parse_directory(str(tmp_path))
        assert len(results) == 2

    def test_parse_directory_skips_unsupported_files(self, tmp_path):
        (tmp_path / "data.csv").write_text("a,b,c", encoding="utf-8")
        (tmp_path / "notes.txt").write_text("some notes", encoding="utf-8")

        results = parse_directory(str(tmp_path))
        assert len(results) == 1
        assert results[0][0].endswith("notes.txt")

    def test_parse_directory_with_empty_dir(self, tmp_path):
        results = parse_directory(str(tmp_path))
        assert results == []

    def test_parse_directory_returns_empty_for_invalid_path(self):
        results = parse_directory("/nonexistent/path/abc123")
        assert results == []

    def test_parse_directory_with_file_types_filter(self, tmp_path):
        (tmp_path / "notes.txt").write_text("text content", encoding="utf-8")
        (tmp_path / "readme.md").write_text("markdown content", encoding="utf-8")

        results = parse_directory(str(tmp_path), file_types=".txt")
        assert len(results) == 1
        assert results[0][0].endswith("notes.txt")

    def test_parse_directory_skips_empty_files(self, tmp_path):
        (tmp_path / "empty.txt").write_text("", encoding="utf-8")
        (tmp_path / "nonempty.txt").write_text("content", encoding="utf-8")

        results = parse_directory(str(tmp_path))
        assert len(results) == 1
        assert results[0][0].endswith("nonempty.txt")
