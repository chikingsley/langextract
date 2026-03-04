"""Shared test fixtures for langextract tests."""

import os
import pathlib

import pytest

FIXTURES_DIR = pathlib.Path(__file__).parent / "fixtures"


@pytest.fixture
def medical_text():
    """Canonical medical text for extraction tests."""
    return (FIXTURES_DIR / "medical_note.txt").read_text()


@pytest.fixture
def sample_examples():
    """Standard few-shot examples for medication extraction."""
    from langextract.core.data import ExampleData, Extraction

    return [
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
                    attributes={"name": "Metformin", "dosage": "500mg", "frequency": "twice daily"},
                ),
            ],
        )
    ]


def _check_url(url: str, timeout: float = 2.0) -> bool:
    """Check if a URL is reachable."""
    try:
        import httpx
        resp = httpx.get(url, timeout=timeout)
        return resp.status_code == 200
    except Exception:
        return False


@pytest.fixture(scope="session")
def ollama_available():
    """Check if Ollama is available on home-mac via Tailscale."""
    url = os.environ.get("OLLAMA_BASE_URL", "http://100.113.195.95:11434")
    return _check_url(f"{url}/api/tags")


@pytest.fixture(scope="session")
def openrouter_key():
    """Get OpenRouter API key if available."""
    return os.environ.get("OPENROUTER_API_KEY")


@pytest.fixture(scope="session")
def gemini_key():
    """Get Gemini API key if available."""
    return os.environ.get("GEMINI_API_KEY")
