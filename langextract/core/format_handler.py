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

"""Centralized format handler for prompts and parsing."""


import json
import re
import typing

import yaml

from langextract.core import data, exceptions

if typing.TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

ExtractionValueType = str | int | float | dict | list | None

_JSON_FORMAT = "json"
_YAML_FORMAT = "yaml"
_YML_FORMAT = "yml"

_FENCE_START = r"```"
_LANGUAGE_TAG = r"(?P<lang>[A-Za-z0-9_+-]+)?"
_FENCE_NEWLINE = r"(?:\s*\n)?"
_FENCE_BODY = r"(?P<body>[\s\S]*?)"
_FENCE_END = r"```"

_FENCE_RE = re.compile(
    _FENCE_START + _LANGUAGE_TAG + _FENCE_NEWLINE + _FENCE_BODY + _FENCE_END,
    re.MULTILINE,
)

_THINK_TAG_RE = re.compile(r"<think>[\s\S]*?</think>\s*", re.IGNORECASE)


class FormatHandler:
    """Handles all format-specific logic for prompts and parsing.

    This class centralizes format handling for JSON and YAML outputs,
    including fence detection, wrapper management, and parsing.

    Attributes:
      format_type: The output format ('json' or 'yaml').
      use_wrapper: Whether to wrap extractions in a container dictionary.
      wrapper_key: The key name for the container dictionary (e.g., creates
        {"extractions": [...]} instead of just [...]).
      use_fences: Whether to use code fences in formatted output.
      attribute_suffix: Suffix for attribute fields in extractions.
      strict_fences: Whether to enforce strict fence validation.
      allow_top_level_list: Whether to allow top-level lists in parsing.
    """

    def __init__(
        self,
        format_type: data.FormatType = data.FormatType.JSON,
        use_wrapper: bool = True,
        wrapper_key: str | None = None,
        use_fences: bool = True,
        attribute_suffix: str = data.ATTRIBUTE_SUFFIX,
        strict_fences: bool = False,
        allow_top_level_list: bool = True,
    ) -> None:
        """Initialize format handler.

        Args:
          format_type: Output format type enum.
          use_wrapper: Whether to wrap extractions in a container dictionary.
            True: {"extractions": [...]}, False: [...]
          wrapper_key: Key name for the container dictionary. When use_wrapper=True:
            - If None: defaults to EXTRACTIONS_KEY ("extractions")
            - If provided: uses the specified key as container
            When use_wrapper=False, this parameter is ignored.
          use_fences: Whether to use ```json or ```yaml fences.
          attribute_suffix: Suffix for attribute fields.
          strict_fences: If True, require exact fence format. If False, be lenient
            with model output variations.
          allow_top_level_list: Allow top-level list when not strict and
            wrapper not required.
        """
        self.format_type = format_type
        self.use_wrapper = use_wrapper
        if use_wrapper:
            self.wrapper_key = wrapper_key if wrapper_key is not None else data.EXTRACTIONS_KEY
        else:
            self.wrapper_key = None
        self.use_fences = use_fences
        self.attribute_suffix = attribute_suffix
        self.strict_fences = strict_fences
        self.allow_top_level_list = allow_top_level_list

    def __repr__(self) -> str:
        return (
            "FormatHandler("
            f"format_type={self.format_type!r}, use_wrapper={self.use_wrapper}, "
            f"wrapper_key={self.wrapper_key!r}, use_fences={self.use_fences}, "
            f"attribute_suffix={self.attribute_suffix!r}, "
            f"strict_fences={self.strict_fences}, "
            f"allow_top_level_list={self.allow_top_level_list})"
        )

    def format_extraction_example(self, extractions: list[data.Extraction]) -> str:
        """Format extractions for a prompt example.

        Args:
          extractions: List of extractions to format

        Returns:
          Formatted string for the prompt
        """
        items = [
            {
                ext.extraction_class: ext.extraction_text,
                f"{ext.extraction_class}{self.attribute_suffix}": (ext.attributes or {}),
            }
            for ext in extractions
        ]

        payload = {self.wrapper_key: items} if self.use_wrapper and self.wrapper_key else items

        if self.format_type == data.FormatType.YAML:
            formatted = yaml.safe_dump(payload, default_flow_style=False, sort_keys=False)
        else:
            formatted = json.dumps(payload, indent=2, ensure_ascii=False)

        return self._add_fences(formatted) if self.use_fences else formatted

    def parse_output(
        self, text: str, *, strict: bool | None = None
    ) -> Sequence[Mapping[str, ExtractionValueType]]:
        """Parse model output to extract data.

        Args:
          text: Raw model output.
          strict: If True, enforce strict schema validation. When strict is
            True, always require wrapper object if wrapper_key is configured,
            reject top-level lists even if allow_top_level_list is True, and
            enforce exact format compliance.

        Returns:
          List of extraction dictionaries.

        Raises:
          FormatError: Various subclasses for specific parsing failures.
        """
        if not text:
            raise exceptions.FormatParseError("Empty or invalid input string.")

        content = self._extract_content(text)

        try:
            parsed = self._parse_with_fallback(content, bool(strict))
        except (yaml.YAMLError, json.JSONDecodeError) as e:
            msg = f"Failed to parse {self.format_type.value.upper()} content: {str(e)[:200]}"
            raise exceptions.FormatParseError(msg) from e

        if parsed is None:
            if self.use_wrapper:
                raise exceptions.FormatParseError(
                    f"Content must be a mapping with an '{self.wrapper_key}' key."
                )
            raise exceptions.FormatParseError(
                "Content must be a list of extractions or a dict."
            )

        require_wrapper = self.wrapper_key is not None and (self.use_wrapper or bool(strict))

        if isinstance(parsed, dict):
            if require_wrapper:
                if self.wrapper_key not in parsed:
                    raise exceptions.FormatParseError(
                        f"Content must contain an '{self.wrapper_key}' key."
                    )
                items = parsed[self.wrapper_key]
            else:
                if data.EXTRACTIONS_KEY in parsed:
                    items = parsed[data.EXTRACTIONS_KEY]
                elif self.wrapper_key and self.wrapper_key in parsed:
                    items = parsed[self.wrapper_key]
                else:
                    items = [parsed]
        elif isinstance(parsed, list):
            if require_wrapper and (strict or not self.allow_top_level_list):
                raise exceptions.FormatParseError(
                    f"Content must be a mapping with an '{self.wrapper_key}' key."
                )
            if strict and self.use_wrapper:
                raise exceptions.FormatParseError("Strict mode requires a wrapper object.")
            if not self.allow_top_level_list:
                raise exceptions.FormatParseError("Top-level list is not allowed.")
            # Some models return [...] instead of {"extractions": [...]}.
            items = parsed
        else:
            raise exceptions.FormatParseError(f"Expected list or dict, got {type(parsed)}")

        if not isinstance(items, list):
            raise exceptions.FormatParseError(
                "The extractions must be a sequence (list) of mappings."
            )

        for item in items:
            if not isinstance(item, dict):
                raise exceptions.FormatParseError("Each item in the sequence must be a mapping.")
            for k in item:
                if not isinstance(k, str):
                    raise exceptions.FormatParseError(
                        "All extraction keys must be strings (got a non-string key)."
                    )

        return items

    def _add_fences(self, content: str) -> str:
        """Add code fences around content."""
        fence_type = self.format_type.value
        return f"```{fence_type}\n{content.strip()}\n```"

    def _is_valid_language_tag(
        self, lang: str | None, valid_tags: dict[data.FormatType, set[str]]
    ) -> bool:
        """Check if language tag is valid for the format type."""
        if lang is None:
            return True
        tag = lang.strip().lower()
        return tag in valid_tags.get(self.format_type, set())

    def _parse_with_fallback(self, content: str, strict: bool):
        """Parse content, retrying without <think> tags on failure."""
        try:
            if self.format_type == data.FormatType.YAML:
                return yaml.safe_load(content)
            return json.loads(content)
        except (yaml.YAMLError, json.JSONDecodeError):
            if strict:
                raise
            # Reasoning models (DeepSeek-R1, QwQ) emit <think> tags before JSON.
            if _THINK_TAG_RE.search(content):
                stripped = _THINK_TAG_RE.sub("", content).strip()
                if self.format_type == data.FormatType.YAML:
                    return yaml.safe_load(stripped)
                return json.loads(stripped)
            raise

    def _extract_content(self, text: str) -> str:
        """Extract content from text, handling fences if configured.

        Args:
          text: Input text that may contain fenced blocks

        Returns:
          Extracted content

        Raises:
          FormatParseError: When fences required but not found or multiple
            blocks found.
        """
        if not self.use_fences:
            return text.strip()

        matches = list(_FENCE_RE.finditer(text))

        valid_tags = {
            data.FormatType.YAML: {_YAML_FORMAT, _YML_FORMAT},
            data.FormatType.JSON: {_JSON_FORMAT},
        }

        candidates = [
            m for m in matches if self._is_valid_language_tag(m.group("lang"), valid_tags)
        ]

        if self.strict_fences:
            if len(candidates) != 1:
                if len(candidates) == 0:
                    raise exceptions.FormatParseError(
                        "Input string does not contain valid fence markers."
                    )
                raise exceptions.FormatParseError(
                    "Multiple fenced blocks found. Expected exactly one."
                )
            return candidates[0].group("body").strip()

        if len(candidates) == 1:
            return candidates[0].group("body").strip()
        if len(candidates) > 1:
            raise exceptions.FormatParseError("Multiple fenced blocks found. Expected exactly one.")

        if matches:
            if not self.strict_fences and len(matches) == 1:
                return matches[0].group("body").strip()
            raise exceptions.FormatParseError(f"No {self.format_type.value} code block found.")

        return text.strip()

    # ---- Format handler construction from resolver params ----

    @classmethod
    def from_resolver_params(
        cls,
        *,
        resolver_params: dict | None,
        base_format_type: data.FormatType,
        base_use_fences: bool,
        base_attribute_suffix: str = data.ATTRIBUTE_SUFFIX,
        base_use_wrapper: bool = True,
        base_wrapper_key: str | None = data.EXTRACTIONS_KEY,
    ) -> tuple[FormatHandler, dict]:
        """Create FormatHandler from resolver_params.

        Args:
          resolver_params: May contain a 'format_handler'.
          base_format_type: Default format when not overridden.
          base_use_fences: Default fence usage from the model.
          base_attribute_suffix: Default attribute suffix.
          base_use_wrapper: Default wrapper behavior.
          base_wrapper_key: Default wrapper key.
        Returns:
          (format_handler, remaining_resolver_params)
        """
        rp = dict(resolver_params or {})

        if rp.get("format_handler") is not None:
            handler = rp.pop("format_handler")
            return typing.cast("FormatHandler", handler), rp

        handler = cls(
            format_type=base_format_type,
            use_wrapper=base_use_wrapper,
            wrapper_key=base_wrapper_key if base_use_wrapper else None,
            use_fences=base_use_fences,
            attribute_suffix=base_attribute_suffix,
            strict_fences=False,
        )
        return handler, rp
