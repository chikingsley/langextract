"""Live API tests.

These tests intentionally hit real endpoints and should be run only when
credentials are configured.
"""

from __future__ import annotations

import os

import pytest
from langextract.providers.openai import OpenAILanguageModel

pytestmark = pytest.mark.live_api


def _require_openrouter_credentials() -> tuple[str, str, str]:
    api_key = os.getenv("OPENROUTER_API_KEY") or os.getenv("OPENAI_API_KEY")
    if not api_key:
        pytest.skip("Missing OPENROUTER_API_KEY (or OPENAI_API_KEY) for live API tests.")

    base_url = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
    model_id = os.getenv("OPENROUTER_MODEL", "google/gemini-2.5-flash")
    return api_key, base_url, model_id


def test_openrouter_openai_compat_json_roundtrip():
    api_key, base_url, model_id = _require_openrouter_credentials()
    model = OpenAILanguageModel(
        model_id=model_id,
        api_key=api_key,
        base_url=base_url,
        temperature=0.0,
        max_workers=1,
    )

    prompt = "Return JSON object {\"status\":\"ok\"} and nothing else."
    results = list(model.infer([prompt], max_output_tokens=64))

    assert len(results) == 1
    assert len(results[0]) == 1

    output = results[0][0].output
    parsed = model.parse_output(output)
    assert isinstance(parsed, dict)
    assert str(parsed.get("status", "")).lower() == "ok"
