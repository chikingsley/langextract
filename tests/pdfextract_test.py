"""Tests for the pdfextract module."""

import tempfile
from pathlib import Path

import pymupdf
import pytest

from pdfextract import extract_document
from pdfextract.quality import check_page_quality
from pdfextract.types import DocumentResult


def _create_test_pdf(text_pages: list[str]) -> Path:
    """Create a simple PDF with text pages for testing."""
    doc = pymupdf.open()
    for text in text_pages:
        page = doc.new_page()
        page.insert_text((72, 72), text, fontsize=12)
    path = Path(tempfile.mktemp(suffix=".pdf"))
    doc.save(str(path))
    doc.close()
    return path


class TestQualityCheck:
    def test_empty_text_needs_ocr(self):
        result = check_page_quality("")
        assert result.needs_ocr is True
        assert result.non_whitespace == 0

    def test_whitespace_only_needs_ocr(self):
        result = check_page_quality("   \n\t  ")
        assert result.needs_ocr is True

    def test_good_text_passes(self):
        text = "This is a valid contract document with sufficient meaningful words and content."
        result = check_page_quality(text)
        assert result.needs_ocr is False
        assert result.meaningful_words >= 3

    def test_garbled_text_fails(self):
        # Simulate garbled binary extraction — too few meaningful words
        text = "ÿØÿà\x00\x10JFIF\x00\x01ÿÛ\x00C\x00"
        result = check_page_quality(text)
        assert result.needs_ocr is True

    def test_short_text_fails(self):
        text = "hi"
        result = check_page_quality(text)
        assert result.needs_ocr is True

    def test_borderline_text(self):
        # Enough chars but no meaningful words
        text = "a b c d e f g h i j k l m n o p q r s t u v w x y z 1 2 3 4 5 6 7 8 9 0"
        result = check_page_quality(text)
        # Has non-whitespace but no meaningful words (4+ alnum chars)
        assert result.meaningful_words == 0


class TestNativeExtraction:
    def test_extract_single_page(self):
        path = _create_test_pdf(["Hello World"])
        try:
            result = extract_document(str(path), skip_ocr=True)
            assert isinstance(result, DocumentResult)
            assert len(result.pages) == 1
            assert result.pages[0].source == "native"
            assert "Hello" in result.full_text
            assert "World" in result.full_text
            assert len(result.word_map) >= 2
        finally:
            path.unlink(missing_ok=True)

    def test_extract_multi_page(self):
        path = _create_test_pdf([
            "Page one has some contract text here.",
            "Page two has more detailed terms and conditions.",
        ])
        try:
            result = extract_document(str(path), skip_ocr=True)
            assert len(result.pages) == 2
            assert result.pages[0].page_num == 0
            assert result.pages[1].page_num == 1
            # Full text should contain both pages
            assert "Page one" in result.full_text
            assert "Page two" in result.full_text
        finally:
            path.unlink(missing_ok=True)

    def test_word_map_char_offsets_are_consistent(self):
        path = _create_test_pdf(["Contract Agreement between Desert Services and Client"])
        try:
            result = extract_document(str(path), skip_ocr=True)
            for word in result.word_map:
                # Each word's text should match the slice of full_text
                assert result.full_text[word.char_start : word.char_end] == word.text, (
                    f"Word '{word.text}' at [{word.char_start}:{word.char_end}] "
                    f"doesn't match full_text slice '{result.full_text[word.char_start:word.char_end]}'"
                )
        finally:
            path.unlink(missing_ok=True)

    def test_words_in_range(self):
        path = _create_test_pdf(["The contract value is seven thousand nine hundred thirty dollars"])
        try:
            result = extract_document(str(path), skip_ocr=True)
            # Find "contract" in full text
            idx = result.full_text.find("contract")
            assert idx >= 0
            words = result.words_in_range(idx, idx + len("contract"))
            assert any(w.text == "contract" for w in words)
        finally:
            path.unlink(missing_ok=True)

    def test_bbox_for_range(self):
        path = _create_test_pdf(["Total Amount Due"])
        try:
            result = extract_document(str(path), skip_ocr=True)
            idx = result.full_text.find("Amount")
            assert idx >= 0
            bbox = result.bbox_for_range(idx, idx + len("Amount"))
            assert bbox is not None
            page, x0, y0, x1, y1 = bbox
            assert page == 0
            assert x0 < x1  # valid bbox
            assert y0 < y1
        finally:
            path.unlink(missing_ok=True)

    def test_empty_page_detected(self):
        """A blank page should be flagged as needing OCR."""
        doc = pymupdf.open()
        doc.new_page()  # blank page, no text
        path = Path(tempfile.mktemp(suffix=".pdf"))
        doc.save(str(path))
        doc.close()

        try:
            result = extract_document(str(path), skip_ocr=True)
            assert len(result.pages) == 1
            # Blank page with skip_ocr still produces a page, just empty
            assert result.pages[0].text.strip() == ""
        finally:
            path.unlink(missing_ok=True)


class TestPageDimensions:
    def test_page_dimensions_recorded(self):
        path = _create_test_pdf(["Some text"])
        try:
            result = extract_document(str(path), skip_ocr=True)
            page = result.pages[0]
            assert page.width > 0
            assert page.height > 0
            # Default PyMuPDF page is A4 (595 x 842 points)
            assert abs(page.width - 595) < 1
            assert abs(page.height - 842) < 1
        finally:
            path.unlink(missing_ok=True)
