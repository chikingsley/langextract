"""Tests for OpenAI reasoning_effort normalization (fork-specific feature)."""

from unittest import mock

import pytest


@pytest.fixture
def openai_model():
    """Create an OpenAILanguageModel with a mocked OpenAI client."""
    with mock.patch("openai.OpenAI"):
        from langextract.providers.openai import OpenAILanguageModel

        return OpenAILanguageModel(model_id="gpt-4o-mini", api_key="test-key")


class TestNormalizeReasoningParams:
    def test_flat_reasoning_effort_converts_to_nested(self, openai_model):
        config = {"reasoning_effort": "high"}
        result = openai_model._normalize_reasoning_params(config)
        assert result == {"reasoning": {"effort": "high"}}
        assert "reasoning_effort" not in result

    def test_existing_reasoning_dict_merged(self, openai_model):
        config = {
            "reasoning_effort": "medium",
            "reasoning": {"type": "chain_of_thought"},
        }
        result = openai_model._normalize_reasoning_params(config)
        assert result["reasoning"]["type"] == "chain_of_thought"
        assert result["reasoning"]["effort"] == "medium"
        assert "reasoning_effort" not in result

    def test_no_reasoning_effort_passes_through(self, openai_model):
        config = {"temperature": 0.7, "top_p": 0.9}
        result = openai_model._normalize_reasoning_params(config)
        assert result == {"temperature": 0.7, "top_p": 0.9}
        assert "reasoning" not in result

    def test_existing_effort_not_overwritten(self, openai_model):
        """setdefault should not overwrite an existing 'effort' key."""
        config = {
            "reasoning_effort": "low",
            "reasoning": {"effort": "high"},
        }
        result = openai_model._normalize_reasoning_params(config)
        assert result["reasoning"]["effort"] == "high"

    def test_does_not_mutate_input(self, openai_model):
        config = {"reasoning_effort": "medium"}
        original = config.copy()
        openai_model._normalize_reasoning_params(config)
        assert config == original

    def test_none_reasoning_creates_fresh_dict(self, openai_model):
        """When reasoning key exists but is None, creates fresh dict."""
        config = {"reasoning_effort": "low", "reasoning": None}
        result = openai_model._normalize_reasoning_params(config)
        assert result["reasoning"] == {"effort": "low"}
