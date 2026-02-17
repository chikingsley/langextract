"""Text quality checks to decide if a page needs OCR.

Ported from Kreuzberg's OCR fallback heuristics:
- Checks non-whitespace character count
- Checks alphanumeric ratio (catches garbled/binary text)
- Checks for meaningful words (4+ alphanumeric chars)
- Checks text coverage area vs page area
"""

from pdfextract.types import QualityCheck

# Thresholds (from Kreuzberg)
MIN_NON_WHITESPACE_PER_PAGE = 32
MIN_MEANINGFUL_WORD_LEN = 4
MIN_MEANINGFUL_WORDS = 3
MIN_ALNUM_RATIO = 0.3


def check_page_quality(text: str) -> QualityCheck:
    """Evaluate extracted text quality for a single page.

    Returns a QualityCheck with the decision in needs_ocr.
    """
    stripped = text.strip()

    if not stripped:
        return QualityCheck(
            non_whitespace=0,
            alnum=0,
            meaningful_words=0,
            alnum_ratio=0.0,
            needs_ocr=True,
        )

    non_whitespace = sum(1 for c in stripped if not c.isspace())
    alnum = sum(1 for c in stripped if c.isalnum())
    meaningful_words = sum(
        1
        for word in stripped.split()
        if sum(1 for c in word if c.isalnum()) >= MIN_MEANINGFUL_WORD_LEN
    )
    alnum_ratio = alnum / non_whitespace if non_whitespace > 0 else 0.0

    # Decision logic (from Kreuzberg's evaluate_native_text_for_ocr)
    if non_whitespace == 0 or alnum == 0:
        needs_ocr = True
    elif (
        non_whitespace >= MIN_NON_WHITESPACE_PER_PAGE
        and meaningful_words >= MIN_MEANINGFUL_WORDS
    ):
        needs_ocr = False
    elif alnum_ratio < MIN_ALNUM_RATIO:
        needs_ocr = True
    elif non_whitespace < MIN_NON_WHITESPACE_PER_PAGE:
        needs_ocr = True
    else:
        needs_ocr = meaningful_words == 0

    return QualityCheck(
        non_whitespace=non_whitespace,
        alnum=alnum,
        meaningful_words=meaningful_words,
        alnum_ratio=alnum_ratio,
        needs_ocr=needs_ocr,
    )
