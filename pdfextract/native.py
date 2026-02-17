"""Native PDF text extraction using PyMuPDF.

Extracts text with word-level bounding boxes from digital/native PDFs.
This is the fast path — no OCR needed, coordinates come directly from
the PDF structure.
"""

import pymupdf

from pdfextract.types import PageResult, WordBbox


def extract_page_native(page: pymupdf.Page, page_num: int, char_offset: int) -> PageResult:
    """Extract text and word bboxes from a single native PDF page.

    Args:
        page: PyMuPDF page object.
        page_num: 0-indexed page number.
        char_offset: Current character offset in the full document text.

    Returns:
        PageResult with text, word bboxes, and page dimensions.
    """
    words = page.get_text("words")  # list of (x0, y0, x1, y1, "word", block, line, word_n)
    rect = page.rect

    page_text_parts: list[str] = []
    word_bboxes: list[WordBbox] = []
    current_offset = char_offset

    # get_text("words") returns words in reading order
    prev_block = -1
    prev_line = -1

    for x0, y0, x1, y1, word_text, block_no, line_no, _word_no in words:
        # Add line/block separators
        if prev_block >= 0:
            if block_no != prev_block:
                page_text_parts.append("\n\n")
                current_offset += 2
            elif line_no != prev_line:
                page_text_parts.append("\n")
                current_offset += 1
            else:
                page_text_parts.append(" ")
                current_offset += 1

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

        page_text_parts.append(word_text)
        current_offset = word_end
        prev_block = block_no
        prev_line = line_no

    page_text = "".join(page_text_parts)

    return PageResult(
        page_num=page_num,
        source="native",
        text=page_text,
        words=word_bboxes,
        width=rect.width,
        height=rect.height,
    )


def extract_native(path: str) -> tuple[list[PageResult], pymupdf.Document]:
    """Extract all pages from a PDF using native text extraction.

    Args:
        path: Path to the PDF file.

    Returns:
        Tuple of (page results, pymupdf document) — document kept open
        for potential OCR page rendering later.
    """
    doc = pymupdf.open(path)
    pages: list[PageResult] = []
    char_offset = 0

    for page_num in range(len(doc)):
        page = doc[page_num]
        result = extract_page_native(page, page_num, char_offset)
        pages.append(result)
        # +1 for the page separator we'll add between pages
        char_offset += len(result.text) + 1

    return pages, doc
