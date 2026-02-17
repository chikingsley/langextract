"""OCR orchestration for scanned PDF pages."""

from __future__ import annotations

import logging
from typing import Any

import pymupdf

from pdfextract.ocr_backends import create_ocr_backend
from pdfextract.types import PageResult, WordBbox

logger = logging.getLogger(__name__)

DEFAULT_OCR_PROMPT = (
    "Extract all visible text verbatim from this page. "
    "Preserve line breaks. Output text only."
)


def extract_page_ocr(
    doc: pymupdf.Document,
    page_num: int,
    char_offset: int,
    *,
    dpi: int = 300,
    backend: str = "rapidocr",
    backend_options: dict[str, Any] | None = None,
    prompt: str = DEFAULT_OCR_PROMPT,
) -> PageResult:
    """Run OCR on one page using the configured backend."""
    page = doc[page_num]
    rect = page.rect
    zoom = dpi / 72
    pixmap = page.get_pixmap(matrix=pymupdf.Matrix(zoom, zoom))

    selected_backend = create_ocr_backend(backend, **(backend_options or {}))
    backend_result = selected_backend.extract_page(pixmap=pixmap, dpi=dpi, prompt=prompt)

    word_bboxes = [
        WordBbox(
            text=word.text,
            page=page_num,
            x0=word.x0,
            y0=word.y0,
            x1=word.x1,
            y1=word.y1,
            char_start=char_offset + word.char_start,
            char_end=char_offset + word.char_end,
        )
        for word in backend_result.words
    ]

    logger.info(
        "OCR page %d via %s: %d words (%.0fx%.0f at %d DPI)",
        page_num,
        selected_backend.name,
        len(word_bboxes),
        rect.width,
        rect.height,
        dpi,
    )

    return PageResult(
        page_num=page_num,
        source="ocr",
        text=backend_result.text,
        words=word_bboxes,
        width=rect.width,
        height=rect.height,
    )
