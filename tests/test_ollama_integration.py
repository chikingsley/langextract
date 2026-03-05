"""Integration tests for real Ollama inference."""

from __future__ import annotations

import os

import httpx
import pytest
from langextract.core import data
from langextract.providers.ollama import OllamaLanguageModel

pytestmark = pytest.mark.integration


def _model_available(base_url: str, model_id: str) -> bool:
    try:
        response = httpx.get(f"{base_url}/api/tags", timeout=5.0)
        response.raise_for_status()
        payload = response.json()
    except Exception:
        return False

    models = payload.get("models", [])
    names = {m.get("name", "") for m in models if isinstance(m, dict)}
    return model_id in names or any(name.startswith(f"{model_id}:") for name in names)


def test_ollama_infer_live(ollama_available):
    if not ollama_available:
        pytest.skip("Ollama endpoint not reachable.")

    base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    model_id = os.getenv("OLLAMA_TEST_MODEL", "gemma3:latest")

    if not _model_available(base_url, model_id):
        pytest.skip(f"Ollama model '{model_id}' is not available on {base_url}.")

    model = OllamaLanguageModel(
        model_id=model_id,
        base_url=base_url,
        format_type=data.FormatType.JSON,
    )

    prompt = "Return JSON with one key named status."
    outputs = list(model.infer([prompt]))

    assert len(outputs) == 1
    assert len(outputs[0]) == 1

    scored = outputs[0][0]
    assert isinstance(scored.output, str)
    assert scored.output.strip()
    assert scored.usage is not None
    assert scored.usage["total_tokens"] >= 0
