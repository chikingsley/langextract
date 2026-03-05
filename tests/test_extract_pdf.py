# Copyright 2025 Google LLC.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for lx.extract_pdf() wiring."""

from unittest import mock

from langextract.extraction import PdfExtractionResult, extract_pdf
from pdfextract.types import DocumentResult, PageResult, WordBbox


def _make_doc_result(text: str = "Patient takes Lisinopril 10mg daily.") -> DocumentResult:
    """Build a minimal DocumentResult for testing."""
    words = [
        WordBbox(text="Patient", page=0, x0=72, y0=100, x1=130, y1=112, char_start=0, char_end=7),
        WordBbox(text="takes", page=0, x0=135, y0=100, x1=170, y1=112, char_start=8, char_end=13),
        WordBbox(
            text="Lisinopril",
            page=0, x0=175, y0=100, x1=250, y1=112,
            char_start=14, char_end=24,
        ),
        WordBbox(text="10mg", page=0, x0=255, y0=100, x1=285, y1=112, char_start=25, char_end=29),
        WordBbox(
            text="daily.",
            page=0, x0=290, y0=100, x1=325, y1=112,
            char_start=30, char_end=36,
        ),
    ]
    page = PageResult(page_num=0, source="native", text=text, words=words, width=612, height=792)
    return DocumentResult(pages=[page], full_text=text, word_map=words)


class TestPdfExtractionResult:
    def test_has_document_and_pdf_result(self):
        from langextract.core.data import AnnotatedDocument

        doc = AnnotatedDocument(text="hello")
        pdf = _make_doc_result("hello")
        result = PdfExtractionResult(document=doc, pdf_result=pdf)
        assert result.document is doc
        assert result.pdf_result is pdf

    def test_bbox_for_range_works_through_result(self):
        pdf = _make_doc_result()
        bbox = pdf.bbox_for_range(14, 24)  # "Lisinopril"
        assert bbox is not None
        page, x0, _y0, _x1, y1 = bbox
        assert page == 0
        assert x0 == 175
        assert y1 == 112


class TestExtractPdfWiring:
    @mock.patch("pdfextract.extract_document")
    @mock.patch("langextract.extraction.extract")
    def test_chains_pdfextract_to_extract(self, mock_extract, mock_extract_doc):
        """extract_pdf should call extract_document then extract."""
        from langextract.core.data import AnnotatedDocument

        doc_result = _make_doc_result()
        mock_extract_doc.return_value = doc_result

        annotated = AnnotatedDocument(text=doc_result.full_text, extractions=[])
        mock_extract.return_value = annotated

        result = extract_pdf(
            "test.pdf",
            prompt_description="Extract medications",
            examples=[mock.MagicMock()],
            model_id="gpt-4o-mini",
            api_key="test-key",
        )

        # Verify pdfextract was called with the path
        mock_extract_doc.assert_called_once_with(
            "test.pdf",
            ocr_dpi=300,
            force_ocr=False,
            skip_ocr=False,
            ocr_backend="rapidocr",
            ocr_backend_options=None,
        )

        # Verify extract was called with the full text
        mock_extract.assert_called_once()
        call_args = mock_extract.call_args
        assert call_args[0][0] == doc_result.full_text
        assert call_args[1]["fetch_urls"] is False

        assert isinstance(result, PdfExtractionResult)
        assert result.document is annotated
        assert result.pdf_result is doc_result

    @mock.patch("pdfextract.extract_document")
    @mock.patch("langextract.extraction.extract")
    def test_forwards_ocr_options(self, mock_extract, mock_extract_doc):
        """OCR-related kwargs go to extract_document, not extract."""
        from langextract.core.data import AnnotatedDocument

        mock_extract_doc.return_value = _make_doc_result()
        mock_extract.return_value = AnnotatedDocument(text="x", extractions=[])

        extract_pdf(
            "scan.pdf",
            ocr_dpi=600,
            force_ocr=True,
            ocr_backend="ollama",
            ocr_backend_options={"model": "glm-ocr"},
            prompt_description="Extract",
            examples=[mock.MagicMock()],
            model_id="gpt-4o-mini",
            api_key="test-key",
        )

        mock_extract_doc.assert_called_once_with(
            "scan.pdf",
            ocr_dpi=600,
            force_ocr=True,
            skip_ocr=False,
            ocr_backend="ollama",
            ocr_backend_options={"model": "glm-ocr"},
        )

        # extract should NOT get ocr params
        call_kwargs = mock_extract.call_args[1]
        assert "ocr_dpi" not in call_kwargs
        assert "force_ocr" not in call_kwargs
        assert "ocr_backend" not in call_kwargs

    @mock.patch("pdfextract.extract_document")
    @mock.patch("langextract.extraction.extract")
    def test_handles_list_return_from_extract(self, mock_extract, mock_extract_doc):
        """extract() sometimes returns a list; extract_pdf should unwrap it."""
        from langextract.core.data import AnnotatedDocument

        mock_extract_doc.return_value = _make_doc_result()
        annotated = AnnotatedDocument(text="x", extractions=[])
        mock_extract.return_value = [annotated]  # list form

        result = extract_pdf(
            "test.pdf",
            prompt_description="Extract",
            examples=[mock.MagicMock()],
            model_id="gpt-4o-mini",
            api_key="test-key",
        )

        assert result.document is annotated


class TestExtractPdfImport:
    def test_importable_from_top_level(self):
        import langextract as lx

        assert hasattr(lx, "extract_pdf")
        assert hasattr(lx, "PdfExtractionResult")

    def test_extract_pdf_in_all(self):
        import langextract as lx

        assert "extract_pdf" in lx.__all__
