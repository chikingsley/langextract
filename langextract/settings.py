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

"""Runtime settings for environment-backed configuration."""


import os

from pydantic_settings import BaseSettings, SettingsConfigDict


class LangExtractSettings(BaseSettings):
    """Configuration loaded from environment variables and optional .env file."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    gemini_api_key: str | None = None
    openai_api_key: str | None = None
    ollama_base_url: str = "http://localhost:11434"


def get_settings() -> LangExtractSettings:
    """Build runtime settings, optionally overriding env file via env var."""
    env_file = os.getenv("LANGEXTRACT_ENV_FILE")
    if env_file:
        runtime_kwargs: dict[str, str] = {"_env_file": env_file}
        return LangExtractSettings(**runtime_kwargs)
    return LangExtractSettings()


def resolve_api_key(
    *,
    explicit: str | None,
    env_var_names: tuple[str, ...],
    settings: LangExtractSettings,
) -> str | None:
    """Resolve an API key from explicit value or provider env vars.

    Priority:
      1. Explicit value (passed directly by the caller)
      2. Provider-specific env var (e.g. GEMINI_API_KEY)

    Args:
      explicit: API key passed explicitly by the user, or None.
      env_var_names: Provider-declared env var names to check, in order.
      settings: Resolved LangExtractSettings instance.

    Returns:
      The resolved API key, or None if nothing is found.
    """
    if explicit is not None:
        return explicit

    for name in env_var_names:
        value = getattr(settings, name.lower(), None)
        if value:
            return value

    return None
