"""OCR extraction for scanned PDF pages.

Uses RapidOCR (ONNX-based PP-OCRv4) for word-level bounding boxes on pages
where native text extraction failed the quality check.

RapidOCR is imported lazily — only needed when scanned pages are detected.
Models ship bundled with the package (no runtime download).
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import pymupdf

from pdfextract.types import PageResult, WordBbox

if TYPE_CHECKING:
    from rapidocr import RapidOCR

logger = logging.getLogger(__name__)

# RapidOCR instance cache (lazy init)
_ocr_engine: RapidOCR | None = None


def _get_ocr_engine() -> RapidOCR:
    """Lazy-init RapidOCR. Only imports when first scanned page is found."""
    global _ocr_engine
    if _ocr_engine is None:
        try:
            from rapidocr import RapidOCR

            _ocr_engine = RapidOCR()
        except ImportError:
            raise ImportError(
                "RapidOCR is required for scanned PDF pages. "
                "Install with: uv add rapidocr onnxruntime"
            ) from None
    return _ocr_engine


def _parse_word_results(
    word_results: Any,
    zoom: float,
    page_num: int,
    char_offset: int,
) -> tuple[list[str], list[WordBbox], int]:
    """Parse RapidOCR word_results into text parts and WordBbox list.

    word_results is a tuple of tuples grouped by line:
        ((("Hello", 0.99, [[x0,y0],[x1,y1],[x2,y2],[x3,y3]]), ...), ...)

    Returns:
        (text_parts, word_bboxes, final_offset)
    """
    text_parts: list[str] = []
    word_bboxes: list[WordBbox] = []
    current_offset = char_offset

    for line_idx, line_words in enumerate(word_results):
        if line_idx > 0:
            text_parts.append("\n")
            current_offset += 1

        for word_idx, word_entry in enumerate(line_words):
            word_text: str = str(word_entry[0])
            bbox_points: list[list[int]] = word_entry[2]

            if not word_text.strip():
                continue

            if word_idx > 0:
                text_parts.append(" ")
                current_offset += 1

            # 4-point polygon in pixel coords → axis-aligned bbox in PDF points
            xs = [float(p[0]) / zoom for p in bbox_points]
            ys = [float(p[1]) / zoom for p in bbox_points]
            x0, x1 = min(xs), max(xs)
            y0, y1 = min(ys), max(ys)

            word_start = current_offset
            word_end = word_start + len(word_text)

            word_bboxes.append(
                WordBbox(
                    text=word_text,
                    page=page_num,
                    x0=x0,
                    y0=y0,
                    x1=x1,
                    y1=y1,
                    char_start=word_start,
                    char_end=word_end,
                )
            )

            text_parts.append(word_text)
            current_offset = word_end

    # Trailing newline after last line
    text_parts.append("\n")
    current_offset += 1

    return text_parts, word_bboxes, current_offset


def extract_page_ocr(
    doc: pymupdf.Document,
    page_num: int,
    char_offset: int,
    dpi: int = 300,
) -> PageResult:
    """OCR a single PDF page and return word-level bounding boxes.

    Renders the page to an image, runs RapidOCR with return_word_box=True,
    and maps the detected word bounding boxes back to PDF coordinate space.

    Args:
        doc: Open PyMuPDF document.
        page_num: 0-indexed page number.
        char_offset: Current character offset in the full document text.
        dpi: Resolution for page rendering (higher = better OCR, slower).

    Returns:
        PageResult with OCR-detected text and word bboxes.
    """
    page = doc[page_num]
    rect = page.rect
    engine = _get_ocr_engine()

    # Render page to image for OCR
    zoom = dpi / 72  # PyMuPDF default is 72 DPI
    mat = pymupdf.Matrix(zoom, zoom)
    pix = page.get_pixmap(matrix=mat)

    import numpy as np

    img_array = np.frombuffer(pix.samples, dtype=np.uint8).reshape(
        pix.height, pix.width, pix.n
    )

    # Run OCR with word-level bounding boxes
    result = engine(img_array, return_word_box=True)

    page_text_parts: list[str] = []
    word_bboxes: list[WordBbox] = []

    if result.word_results:  # type: ignore[union-attr]
        page_text_parts, word_bboxes, _ = _parse_word_results(
            result.word_results,  # type: ignore[union-attr]
            zoom=zoom,
            page_num=page_num,
            char_offset=char_offset,
        )

    page_text = "".join(page_text_parts)

    logger.info(
        "OCR page %d: %d words detected (%.0fx%.0f at %d DPI)",
        page_num,
        len(word_bboxes),
        rect.width,
        rect.height,
        dpi,
    )

    return PageResult(
        page_num=page_num,
        source="ocr",
        text=page_text,
        words=word_bboxes,
        width=rect.width,
        height=rect.height,
    )
