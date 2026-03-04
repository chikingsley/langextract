"""Tests for the centralized FormatHandler (fork-specific feature)."""

import json

import pytest
import yaml

from langextract.core.data import ATTRIBUTE_SUFFIX, EXTRACTIONS_KEY, Extraction, FormatType
from langextract.core.exceptions import FormatParseError
from langextract.core.format_handler import FormatHandler


# ---------------------------------------------------------------------------
# Init defaults
# ---------------------------------------------------------------------------


class TestFormatHandlerDefaults:
    def test_default_format_type(self):
        h = FormatHandler()
        assert h.format_type is FormatType.JSON

    def test_default_use_wrapper(self):
        h = FormatHandler()
        assert h.use_wrapper is True

    def test_default_use_fences(self):
        h = FormatHandler()
        assert h.use_fences is True

    def test_default_allow_top_level_list(self):
        h = FormatHandler()
        assert h.allow_top_level_list is True

    def test_default_wrapper_key_when_wrapper_enabled(self):
        h = FormatHandler(use_wrapper=True)
        assert h.wrapper_key == EXTRACTIONS_KEY

    def test_wrapper_key_none_when_wrapper_disabled(self):
        h = FormatHandler(use_wrapper=False)
        assert h.wrapper_key is None

    def test_custom_wrapper_key(self):
        h = FormatHandler(use_wrapper=True, wrapper_key="results")
        assert h.wrapper_key == "results"

    def test_default_attribute_suffix(self):
        h = FormatHandler()
        assert h.attribute_suffix == ATTRIBUTE_SUFFIX

    def test_default_strict_fences(self):
        h = FormatHandler()
        assert h.strict_fences is False


# ---------------------------------------------------------------------------
# Think-tag stripping
# ---------------------------------------------------------------------------


class TestThinkTagStripping:
    """Test that <think>...</think> blocks from reasoning models get stripped."""

    def _handler(self, **kw):
        return FormatHandler(use_fences=False, use_wrapper=False, **kw)

    def test_json_with_think_tags_parses(self):
        h = self._handler()
        text = '<think>Let me reason about this...</think>[{"person": "Alice"}]'
        result = h.parse_output(text)
        assert result == [{"person": "Alice"}]

    def test_multiline_think_tags(self):
        h = self._handler()
        text = (
            "<think>\nStep 1: analyze\nStep 2: extract\n</think>\n"
            '[{"city": "Paris"}]'
        )
        result = h.parse_output(text)
        assert result == [{"city": "Paris"}]

    def test_no_think_tags_works_normally(self):
        h = self._handler()
        text = '[{"animal": "cat"}]'
        result = h.parse_output(text)
        assert result == [{"animal": "cat"}]

    def test_strict_mode_does_not_strip_think_tags(self):
        h = self._handler()
        text = '<think>reasoning</think>  [{"x": 1}]'
        with pytest.raises(FormatParseError):
            h.parse_output(text, strict=True)


# ---------------------------------------------------------------------------
# Top-level list acceptance
# ---------------------------------------------------------------------------


class TestTopLevelList:
    def test_bare_list_accepted_when_allowed(self):
        h = FormatHandler(
            use_fences=False,
            use_wrapper=False,
            allow_top_level_list=True,
        )
        text = '[{"entity": "Apple"}]'
        result = h.parse_output(text)
        assert result == [{"entity": "Apple"}]

    def test_bare_list_rejected_when_disallowed(self):
        h = FormatHandler(
            use_fences=False,
            use_wrapper=False,
            allow_top_level_list=False,
        )
        text = '[{"entity": "Apple"}]'
        with pytest.raises(FormatParseError):
            h.parse_output(text)


# ---------------------------------------------------------------------------
# JSON / YAML roundtrip
# ---------------------------------------------------------------------------


class TestFormatRoundtrip:
    _extractions = [
        Extraction("person", "Alice", attributes={"role": "engineer"}),
        Extraction("location", "Berlin", attributes={"country": "Germany"}),
    ]

    def test_json_roundtrip(self):
        h = FormatHandler(format_type=FormatType.JSON, use_fences=False, use_wrapper=True)
        formatted = h.format_extraction_example(self._extractions)
        items = h.parse_output(formatted)

        assert len(items) == 2
        assert items[0]["person"] == "Alice"
        assert items[0]["person_attributes"] == {"role": "engineer"}
        assert items[1]["location"] == "Berlin"

    def test_yaml_roundtrip(self):
        h = FormatHandler(format_type=FormatType.YAML, use_fences=False, use_wrapper=True)
        formatted = h.format_extraction_example(self._extractions)
        items = h.parse_output(formatted)

        assert len(items) == 2
        assert items[0]["person"] == "Alice"
        assert items[1]["location"] == "Berlin"

    def test_json_roundtrip_with_fences(self):
        h = FormatHandler(format_type=FormatType.JSON, use_fences=True, use_wrapper=True)
        formatted = h.format_extraction_example(self._extractions)
        items = h.parse_output(formatted)
        assert len(items) == 2

    def test_yaml_roundtrip_with_fences(self):
        h = FormatHandler(format_type=FormatType.YAML, use_fences=True, use_wrapper=True)
        formatted = h.format_extraction_example(self._extractions)
        items = h.parse_output(formatted)
        assert len(items) == 2


# ---------------------------------------------------------------------------
# Fence parsing
# ---------------------------------------------------------------------------


class TestFenceParsing:
    def test_json_in_fences_parses(self):
        h = FormatHandler(format_type=FormatType.JSON, use_fences=True, use_wrapper=False)
        text = '```json\n[{"k": "v"}]\n```'
        result = h.parse_output(text)
        assert result == [{"k": "v"}]

    def test_no_fence_fallback(self):
        """When use_fences=True but no fences present, falls back to raw parse."""
        h = FormatHandler(format_type=FormatType.JSON, use_fences=True, use_wrapper=False)
        text = '[{"k": "v"}]'
        result = h.parse_output(text)
        assert result == [{"k": "v"}]

    def test_multiple_fences_rejected(self):
        h = FormatHandler(format_type=FormatType.JSON, use_fences=True, use_wrapper=False)
        text = '```json\n[{"a": 1}]\n```\n\n```json\n[{"b": 2}]\n```'
        with pytest.raises(FormatParseError, match="Multiple fenced blocks"):
            h.parse_output(text)

    def test_empty_input_raises(self):
        h = FormatHandler(format_type=FormatType.JSON, use_fences=True)
        with pytest.raises(FormatParseError, match="Empty"):
            h.parse_output("")

    def test_strict_fences_require_exactly_one_block(self):
        h = FormatHandler(
            format_type=FormatType.JSON,
            use_fences=True,
            use_wrapper=False,
            strict_fences=True,
        )
        # No fences at all
        with pytest.raises(FormatParseError, match="does not contain valid fence"):
            h.parse_output('[{"a": 1}]')

    def test_yaml_fence_tag_accepted(self):
        h = FormatHandler(format_type=FormatType.YAML, use_fences=True, use_wrapper=False)
        text = "```yaml\n- entity: test\n```"
        result = h.parse_output(text)
        assert result == [{"entity": "test"}]
