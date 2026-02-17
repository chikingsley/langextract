"""PDF text extraction with word-level bounding boxes.

Extracts text from native and scanned PDFs with position mapping,
enabling langextract extractions to be projected back onto PDF pages.

Usage:
    from pdfextract import extract_document

    result = extract_document("contract.pdf")
    # result.full_text  → flat string for langextract
    # result.word_map   → char_index → WordBbox lookup
    # result.pages      → per-page results with source (native/ocr)
"""

from pdfextract.extract import extract_document
from pdfextract.quality import check_page_quality
from pdfextract.types import DocumentResult, PageResult, QualityCheck, WordBbox

__all__ = [
    "extract_document",
    "check_page_quality",
    "DocumentResult",
    "PageResult",
    "QualityCheck",
    "WordBbox",
]
