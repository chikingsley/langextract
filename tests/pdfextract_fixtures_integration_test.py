"""Integration tests against real contract PDF fixtures."""

from __future__ import annotations

from pathlib import Path

import pytest
from pdfextract import extract_document

_FIXTURE_DIR = Path(__file__).parent / "fixtures"
_FIXTURE_PDFS = sorted(_FIXTURE_DIR.glob("*.pdf"))


@pytest.mark.integration
@pytest.mark.parametrize(
    "pdf_path",
    _FIXTURE_PDFS,
    ids=[path.name for path in _FIXTURE_PDFS],
)
def test_extract_document_real_contract_fixture(pdf_path: Path):
    result = extract_document(pdf_path, skip_ocr=True)

    assert result.pages
    assert result.word_map
    assert len(result.full_text.strip()) > 500

    text_lower = result.full_text.lower()
    assert any(
        marker in text_lower
        for marker in ("development agreement", "agreement", "contract")
    )

    for word in result.word_map[:2000]:
        assert result.full_text[word.char_start : word.char_end] == word.text
