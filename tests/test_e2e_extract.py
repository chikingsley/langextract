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

"""End-to-end extraction tests against real LLM backends.

These tests hit real APIs and verify the full pipeline:
text → chunking → prompting → LLM → resolver → AnnotatedDocument.

Run with:
    uv run pytest tests/test_e2e_extract.py -v -m live_api

Requires OPENROUTER_API_KEY in .env or environment.
"""

from __future__ import annotations

import os

import langextract as lx
import pytest
from langextract.core.data import ExampleData, Extraction
from langextract.factory import ModelConfig

pytestmark = pytest.mark.live_api

EXPECTED_MEDICATIONS = {"Lisinopril", "Metformin", "Atorvastatin", "Aspirin"}

# Few-shot example for medication extraction (different text from the test input)
MEDICATION_EXAMPLES = [
    ExampleData(
        text=(
            "The patient was prescribed Lisinopril 10mg daily for hypertension "
            "and Metformin 500mg twice daily for type 2 diabetes."
        ),
        extractions=[
            Extraction(
                extraction_class="medication",
                extraction_text="Lisinopril 10mg",
                attributes={"name": "Lisinopril", "dosage": "10mg", "frequency": "daily"},
            ),
            Extraction(
                extraction_class="medication",
                extraction_text="Metformin 500mg",
                attributes={
                    "name": "Metformin",
                    "dosage": "500mg",
                    "frequency": "twice daily",
                },
            ),
        ],
    ),
]


def _get_openrouter_config() -> ModelConfig:
    """Build a ModelConfig for OpenRouter, or skip if key is missing."""
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        pytest.skip("OPENROUTER_API_KEY not set")
    base_url = os.environ.get("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
    model_id = os.environ.get("OPENROUTER_MODEL", "google/gemini-2.5-flash")
    # Explicit provider="openai" because OpenRouter model IDs (e.g. google/gemini-*)
    # don't match OpenAI's router patterns — the router would try to use Gemini directly.
    return ModelConfig(
        model_id=model_id,
        provider="openai",
        provider_kwargs={"api_key": api_key, "base_url": base_url},
    )


def _extract_medication_names(result: lx.core.data.AnnotatedDocument) -> set[str]:
    """Pull medication names from extractions."""
    names = set()
    for ext in result.extractions:
        if ext.extraction_class == "medication":
            # Try the 'name' attribute first, fall back to extraction_text
            name = (ext.attributes or {}).get("name", ext.extraction_text)
            # Normalize: just take the first word (drug name without dosage)
            first_word = name.split()[0].rstrip(",.")
            names.add(first_word)
    return names


class TestOpenRouterExtraction:
    """Full pipeline extraction via OpenRouter."""

    def test_extract_medications_basic(self, medical_text):
        """Extract medications from medical note — no schema constraints."""
        config = _get_openrouter_config()

        result = lx.extract(
            medical_text,
            prompt_description="Extract all medications with their name, dosage, and frequency.",
            examples=MEDICATION_EXAMPLES,
            config=config,
            temperature=0.0,
            use_schema_constraints=False,
            show_progress=False,
        )

        assert isinstance(result, lx.core.data.AnnotatedDocument)
        assert len(result.extractions) >= 3, (
            f"Expected at least 3 medications, got {len(result.extractions)}: "
            f"{[e.extraction_text for e in result.extractions]}"
        )

        found = _extract_medication_names(result)
        overlap = found & EXPECTED_MEDICATIONS
        assert len(overlap) >= 3, (
            f"Expected at least 3 of {EXPECTED_MEDICATIONS}, found {found}"
        )

    def test_extract_medications_with_schema(self, medical_text):
        """Extract medications with OpenAISchema structured outputs."""
        config = _get_openrouter_config()

        result = lx.extract(
            medical_text,
            prompt_description="Extract all medications with their name, dosage, and frequency.",
            examples=MEDICATION_EXAMPLES,
            config=config,
            temperature=0.0,
            use_schema_constraints=True,
            show_progress=False,
        )

        assert isinstance(result, lx.core.data.AnnotatedDocument)
        assert len(result.extractions) >= 3

        found = _extract_medication_names(result)
        overlap = found & EXPECTED_MEDICATIONS
        assert len(overlap) >= 3, (
            f"Expected at least 3 of {EXPECTED_MEDICATIONS}, found {found}"
        )

    def test_extractions_have_char_intervals(self, medical_text):
        """Extractions should be grounded — aligned to positions in the text."""
        config = _get_openrouter_config()

        result = lx.extract(
            medical_text,
            prompt_description="Extract all medications with their name, dosage, and frequency.",
            examples=MEDICATION_EXAMPLES,
            config=config,
            temperature=0.0,
            use_schema_constraints=False,
            show_progress=False,
        )

        aligned = [e for e in result.extractions if e.char_interval is not None]
        assert len(aligned) >= 1, (
            "Expected at least 1 extraction with char_interval, got 0. "
            f"Extractions: "
            f"{[(e.extraction_text, e.alignment_status) for e in result.extractions]}"
        )

        # Verify char_intervals actually point to text in the source
        for ext in aligned:
            ci = ext.char_interval
            span = medical_text[ci.start_pos : ci.end_pos]
            assert len(span) > 0, f"Empty span for {ext.extraction_text}"

    def test_extract_simple_entities(self):
        """Minimal extraction — just named entities from a short string."""
        config = _get_openrouter_config()

        text = "Dr. Jane Wilson prescribed Aspirin 81mg to John Smith."
        examples = [
            ExampleData(
                text="Dr. Bob gave Mary Tylenol 500mg.",
                extractions=[
                    Extraction(extraction_class="person", extraction_text="Dr. Bob"),
                    Extraction(extraction_class="person", extraction_text="Mary"),
                    Extraction(
                        extraction_class="medication", extraction_text="Tylenol 500mg"
                    ),
                ],
            ),
        ]

        result = lx.extract(
            text,
            prompt_description="Extract all people and medications.",
            examples=examples,
            config=config,
            temperature=0.0,
            use_schema_constraints=False,
            show_progress=False,
        )

        assert isinstance(result, lx.core.data.AnnotatedDocument)
        classes = {e.extraction_class for e in result.extractions}
        texts = {e.extraction_text for e in result.extractions}

        # Should find at least one person and one medication
        assert "person" in classes or "medication" in classes, (
            f"Expected person/medication classes, got {classes} with texts {texts}"
        )
        assert len(result.extractions) >= 2
