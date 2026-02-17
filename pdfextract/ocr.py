"""OCR extraction for scanned PDF pages.

Uses PaddleOCR for word-level bounding boxes on pages where native
text extraction failed the quality check.

PaddleOCR is imported lazily — only needed when scanned pages are detected.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import pymupdf

from pdfextract.types import PageResult, WordBbox

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)

# PaddleOCR instance cache (lazy init)
_paddle_ocr = None


def _get_paddle_ocr():
    """Lazy-init PaddleOCR. Only imports when first scanned page is found."""
    global _paddle_ocr
    if _paddle_ocr is None:
        try:
            from paddleocr import PaddleOCR

            _paddle_ocr = PaddleOCR(
                use_angle_cls=True,
                lang="en",
                show_log=False,
            )
        except ImportError:
            raise ImportError(
                "PaddleOCR is required for scanned PDF pages. "
                "Install with: uv add paddleocr paddlepaddle"
            ) from None
    return _paddle_ocr


def extract_page_ocr(
    doc: pymupdf.Document,
    page_num: int,
    char_offset: int,
    dpi: int = 300,
) -> PageResult:
    """OCR a single PDF page and return word-level bounding boxes.

    Renders the page to an image, runs PaddleOCR, and maps the detected
    word bounding boxes back to PDF coordinate space.

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
    ocr = _get_paddle_ocr()

    # Render page to image for OCR
    zoom = dpi / 72  # PyMuPDF default is 72 DPI
    mat = pymupdf.Matrix(zoom, zoom)
    pix = page.get_pixmap(matrix=mat)

    # PaddleOCR expects numpy array or file path
    import numpy as np

    img_array = np.frombuffer(pix.samples, dtype=np.uint8).reshape(
        pix.height, pix.width, pix.n
    )

    # Run OCR
    results = ocr.ocr(img_array, cls=True)

    page_text_parts: list[str] = []
    word_bboxes: list[WordBbox] = []
    current_offset = char_offset

    if results and results[0]:
        for line in results[0]:
            bbox_points, (text, confidence) = line

            if not text.strip():
                continue

            # PaddleOCR returns 4-point polygon: [[x0,y0],[x1,y1],[x2,y2],[x3,y3]]
            # Convert to axis-aligned bbox in PDF coordinates (divide by zoom)
            xs = [p[0] / zoom for p in bbox_points]
            ys = [p[1] / zoom for p in bbox_points]
            x0, x1 = min(xs), max(xs)
            y0, y1 = min(ys), max(ys)

            # Split OCR line into words for finer granularity
            words_in_line = text.split()
            if not words_in_line:
                continue

            # Distribute the line bbox across words proportionally
            line_width = x1 - x0
            total_chars = sum(len(w) for w in words_in_line)

            word_x = x0
            for i, word_text in enumerate(words_in_line):
                # Add space between words
                if i > 0:
                    page_text_parts.append(" ")
                    current_offset += 1

                # Proportional width based on character count
                word_width = (len(word_text) / total_chars) * line_width if total_chars > 0 else line_width
                word_x1 = word_x + word_width

                word_start = current_offset
                word_end = word_start + len(word_text)

                word_bboxes.append(
                    WordBbox(
                        text=word_text,
                        page=page_num,
                        x0=word_x,
                        y0=y0,
                        x1=word_x1,
                        y1=y1,
                        char_start=word_start,
                        char_end=word_end,
                    )
                )

                page_text_parts.append(word_text)
                current_offset = word_end
                word_x = word_x1

            # Line break after each OCR line
            page_text_parts.append("\n")
            current_offset += 1

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
