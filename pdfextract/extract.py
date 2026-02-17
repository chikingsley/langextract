"""Main extraction pipeline: native → quality check → OCR fallback → merge."""

from __future__ import annotations

import logging
from pathlib import Path

from pdfextract.native import extract_native
from pdfextract.quality import check_page_quality
from pdfextract.types import DocumentResult, PageResult, WordBbox

logger = logging.getLogger(__name__)


def extract_document(
    path: str | Path,
    *,
    ocr_dpi: int = 300,
    force_ocr: bool = False,
    skip_ocr: bool = False,
) -> DocumentResult:
    """Extract text and word bounding boxes from a PDF.

    Pipeline per page:
      1. Native extraction via PyMuPDF (fast, character-level precision)
      2. Quality check on extracted text
      3. If quality fails → OCR via PaddleOCR (word-level bboxes)

    Args:
        path: Path to the PDF file.
        ocr_dpi: Resolution for OCR page rendering. Higher = better accuracy, slower.
        force_ocr: OCR every page regardless of native text quality.
        skip_ocr: Never run OCR (useful when PaddleOCR is not installed).

    Returns:
        DocumentResult with full_text, word_map, and per-page results.
    """
    path = str(path)

    # Step 1: Extract all pages natively
    native_pages, doc = extract_native(path)

    # Step 2: Quality check each page, OCR if needed
    final_pages: list[PageResult] = []
    ocr_needed: list[int] = []

    for page_result in native_pages:
        if force_ocr:
            ocr_needed.append(page_result.page_num)
            continue

        quality = check_page_quality(page_result.text)

        if quality.needs_ocr and not skip_ocr:
            logger.info(
                "Page %d: quality check failed "
                "(non_ws=%d, alnum_ratio=%.2f, meaningful_words=%d) → OCR",
                page_result.page_num,
                quality.non_whitespace,
                quality.alnum_ratio,
                quality.meaningful_words,
            )
            ocr_needed.append(page_result.page_num)
        else:
            if quality.needs_ocr and skip_ocr:
                logger.warning(
                    "Page %d: quality check failed but OCR skipped (skip_ocr=True)",
                    page_result.page_num,
                )
            final_pages.append(page_result)

    # Step 3: OCR pages that failed quality check
    if ocr_needed:
        from pdfextract.ocr import extract_page_ocr

        for page_num in ocr_needed:
            # Recalculate char_offset based on pages processed so far
            char_offset = _calculate_offset(final_pages, native_pages, page_num)
            ocr_result = extract_page_ocr(doc, page_num, char_offset, dpi=ocr_dpi)
            final_pages.append(ocr_result)

    doc.close()

    # Sort pages back into order (OCR pages may have been appended out of order)
    final_pages.sort(key=lambda p: p.page_num)

    # Step 4: Build unified text and word map with correct char offsets
    return _build_document_result(final_pages)


def _calculate_offset(
    completed_pages: list[PageResult],
    native_pages: list[PageResult],
    target_page: int,
) -> int:
    """Calculate char offset for a page based on preceding pages."""
    offset = 0
    for page_num in range(target_page):
        # Find this page in completed or native pages
        page = next(
            (p for p in completed_pages if p.page_num == page_num),
            next((p for p in native_pages if p.page_num == page_num), None),
        )
        if page:
            offset += len(page.text) + 1  # +1 for page separator
    return offset


def _build_document_result(pages: list[PageResult]) -> DocumentResult:
    """Build final DocumentResult with recalculated char offsets.

    Recalculates all character offsets to be consistent with the
    concatenated full_text string.
    """
    text_parts: list[str] = []
    all_words: list[WordBbox] = []
    char_offset = 0

    for i, page in enumerate(pages):
        if i > 0:
            text_parts.append("\n")
            char_offset += 1

        # Recalculate word offsets relative to full_text
        page_start = char_offset
        new_words: list[WordBbox] = []

        for word in page.words:
            # Shift word offsets by the difference between page start and original offset
            shift = page_start - (page.words[0].char_start if page.words else 0)
            new_word = WordBbox(
                text=word.text,
                page=word.page,
                x0=word.x0,
                y0=word.y0,
                x1=word.x1,
                y1=word.y1,
                char_start=word.char_start + shift,
                char_end=word.char_end + shift,
            )
            new_words.append(new_word)

        page.words = new_words  # frozen field override during build
        all_words.extend(new_words)
        text_parts.append(page.text)
        char_offset += len(page.text)

    full_text = "".join(text_parts)

    return DocumentResult(
        pages=pages,
        full_text=full_text,
        word_map=all_words,
    )
