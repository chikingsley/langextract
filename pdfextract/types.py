"""Core types for PDF extraction results."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal


@dataclass(frozen=True)
class WordBbox:
    """A word with its bounding box on a PDF page."""

    text: str
    page: int  # 0-indexed page number
    x0: float  # left
    y0: float  # top
    x1: float  # right
    y1: float  # bottom
    char_start: int  # start index in full_text
    char_end: int  # end index in full_text (exclusive)


@dataclass
class PageResult:
    """Extraction result for a single page."""

    page_num: int  # 0-indexed
    source: Literal["native", "ocr"]
    text: str
    words: list[WordBbox] = field(default_factory=list)
    width: float = 0.0  # page width in points
    height: float = 0.0  # page height in points


@dataclass
class QualityCheck:
    """Result of text quality evaluation for a page."""

    non_whitespace: int
    alnum: int
    meaningful_words: int
    alnum_ratio: float
    needs_ocr: bool


@dataclass
class DocumentResult:
    """Complete extraction result for a PDF document."""

    pages: list[PageResult]
    full_text: str  # concatenated text from all pages
    word_map: list[WordBbox]  # all words in document order, char offsets into full_text

    def words_in_range(self, start: int, end: int) -> list[WordBbox]:
        """Find all words overlapping a character range in full_text.

        Use this to map langextract char_interval back to PDF coordinates.
        """
        return [
            w
            for w in self.word_map
            if w.char_end > start and w.char_start < end
        ]

    def bbox_for_range(self, start: int, end: int) -> tuple[int, float, float, float, float] | None:
        """Get the bounding box enclosing a character range.

        Returns (page, x0, y0, x1, y1) or None if no words found.
        """
        words = self.words_in_range(start, end)
        if not words:
            return None
        page = words[0].page
        x0 = min(w.x0 for w in words if w.page == page)
        y0 = min(w.y0 for w in words if w.page == page)
        x1 = max(w.x1 for w in words if w.page == page)
        y1 = max(w.y1 for w in words if w.page == page)
        return (page, x0, y0, x1, y1)
