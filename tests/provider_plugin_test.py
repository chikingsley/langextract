"""Plugin discovery and registration tests."""

from __future__ import annotations

import subprocess
import sys
import textwrap
from importlib import util as importlib_util
from typing import TYPE_CHECKING

import pytest
from langextract import factory, providers
from langextract import plugins as legacy_plugins
from langextract.core import base_model
from langextract.providers import router

if TYPE_CHECKING:
    from pathlib import Path


class TestPluginSmoke:
    """Fast smoke checks for built-in provider discovery."""

    def test_available_providers_includes_builtins(self):
        available = legacy_plugins.available_providers(include_optional=True)
        assert "gemini" in available
        assert "ollama" in available
        assert "openai" in available

    def test_get_provider_class_returns_language_model_class(self):
        provider_class = legacy_plugins.get_provider_class("gemini")
        assert issubclass(provider_class, base_model.BaseLanguageModel)

    def test_unknown_provider_raises_key_error(self):
        with pytest.raises(KeyError):
            legacy_plugins.get_provider_class("this-does-not-exist")


@pytest.mark.requires_pip
class TestPluginE2E:
    """End-to-end plugin install/discovery path using a throwaway package."""

    @staticmethod
    def _write_plugin_package(package_root: Path) -> None:
        package_root.mkdir(parents=True, exist_ok=True)
        module_dir = package_root / "mock_provider_plugin"
        module_dir.mkdir(parents=True, exist_ok=True)

        (module_dir / "__init__.py").write_text("", encoding="utf-8")
        (module_dir / "provider.py").write_text(
            textwrap.dedent(
                """\
                from langextract.core import base_model, schema, types


                class MockProviderLanguageModel(base_model.BaseLanguageModel):
                    pattern_priority = 42
                    ENV_API_KEY_NAMES = ()

                    @classmethod
                    def get_model_patterns(cls):
                        return (r"^mock-plugin-model$",)

                    def __init__(self, model_id="mock-plugin-model", **kwargs):
                        self.model_id = model_id
                        super().__init__(constraint=schema.Constraint(), **kwargs)

                    def infer(self, batch_prompts, **kwargs):
                        for _prompt in batch_prompts:
                            yield [types.ScoredOutput(score=1.0, output='{"result":"plugin-ok"}')]
                """
            ),
            encoding="utf-8",
        )

        (package_root / "pyproject.toml").write_text(
            textwrap.dedent(
                """\
                [build-system]
                requires = ["setuptools>=68"]
                build-backend = "setuptools.build_meta"

                [project]
                name = "langextract-mock-provider-plugin"
                version = "0.0.1"
                description = "Test plugin for LangExtract provider discovery."

                [project.entry-points."langextract.providers"]
                mock-plugin = "mock_provider_plugin.provider:MockProviderLanguageModel"
                """
            ),
            encoding="utf-8",
        )

    def test_plugin_can_be_installed_and_resolved(self, tmp_path):
        if importlib_util.find_spec("pip") is None:
            pytest.skip("pip module unavailable in this environment.")

        plugin_root = tmp_path / "mock_plugin_pkg"
        self._write_plugin_package(plugin_root)

        subprocess.run(
            [sys.executable, "-m", "pip", "install", str(plugin_root)],
            check=True,
            capture_output=True,
            text=True,
        )

        # Legacy plugin discovery cache.
        legacy_plugins._discovered.cache_clear()

        # Runtime registry state for the modern provider loading path.
        providers._reset_for_testing()
        router.clear()

        discovered = legacy_plugins.available_providers(
            allow_override=True,
            include_optional=True,
        )
        assert "mock-plugin" in discovered

        legacy_cls = legacy_plugins.get_provider_class(
            "mock-plugin",
            allow_override=True,
            include_optional=True,
        )
        assert legacy_cls.__name__ == "MockProviderLanguageModel"

        providers.load_plugins_once()
        resolved_cls = router.resolve("mock-plugin-model")
        assert resolved_cls.__name__ == "MockProviderLanguageModel"

        model = factory.create_model(
            factory.ModelConfig(model_id="mock-plugin-model"),
            return_fence_output=False,
        )
        outputs = list(model.infer(["test prompt"]))
        assert outputs[0][0].output == '{"result":"plugin-ok"}'


# Keep tox target compatibility (tests/provider_plugin_test.py::PluginSmokeTest
# and ::PluginE2ETest) while honoring pytest's Test* discovery pattern.
PluginSmokeTest = TestPluginSmoke
PluginE2ETest = TestPluginE2E
