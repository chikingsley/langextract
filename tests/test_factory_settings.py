"""Tests for environment-driven factory/settings behavior."""

from __future__ import annotations

from unittest import mock

from langextract import factory
from langextract import providers as providers_pkg
from langextract.providers import openai, router
from langextract.settings import LangExtractSettings, get_settings, resolve_api_key


def test_resolve_api_key_prefers_explicit_value():
    settings = LangExtractSettings(gemini_api_key="env-gemini-key")
    resolved = resolve_api_key(
        explicit="explicit-key",
        env_var_names=("GEMINI_API_KEY",),
        settings=settings,
    )
    assert resolved == "explicit-key"


def test_get_settings_uses_langextract_env_file(monkeypatch, tmp_path):
    env_file = tmp_path / "custom.env"
    env_file.write_text(
        "OPENAI_API_KEY=file-openai-key\nOLLAMA_BASE_URL=http://127.0.0.1:22434\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("LANGEXTRACT_ENV_FILE", str(env_file))

    settings = get_settings()
    assert settings.openai_api_key == "file-openai-key"
    assert settings.ollama_base_url == "http://127.0.0.1:22434"


def test_create_model_reads_api_key_and_base_url_from_environment(monkeypatch):
    monkeypatch.delenv("LANGEXTRACT_ENV_FILE", raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "env-openai-key")
    monkeypatch.setenv("OLLAMA_BASE_URL", "http://127.0.0.1:44444")

    providers_pkg._reset_for_testing()
    router.clear()

    with mock.patch("openai.OpenAI") as mock_openai_client:
        model = factory.create_model(
            factory.ModelConfig(provider="openai"),
            return_fence_output=False,
        )

    assert isinstance(model, openai.OpenAILanguageModel)
    assert model.api_key == "env-openai-key"
    assert model.base_url == "http://127.0.0.1:44444"
    mock_openai_client.assert_called_once_with(
        api_key="env-openai-key",
        base_url="http://127.0.0.1:44444",
        organization=None,
    )
