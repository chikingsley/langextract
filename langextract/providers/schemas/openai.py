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

"""OpenAI provider schema implementation for Structured Outputs."""

from __future__ import annotations

import dataclasses
import warnings
from typing import TYPE_CHECKING, Any

from langextract.core import data, schema
from langextract.core import format_handler as fh

if TYPE_CHECKING:
    from collections.abc import Sequence


@dataclasses.dataclass
class OpenAISchema(schema.BaseSchema):
    """Schema implementation for OpenAI Structured Outputs (json_schema mode).

    Converts ExampleData objects into a JSON Schema that OpenAI can enforce
    via ``response_format: {type: "json_schema", ...}``. This enables
    token-level schema enforcement — the model is guaranteed to produce
    valid JSON matching the schema.

    OpenAI strict mode requirements:
    - ``additionalProperties: false`` on every object
    - All properties listed in ``required``
    - No ``nullable`` — use ``anyOf`` with ``{type: "null"}`` instead
    """

    _schema_dict: dict[str, Any]

    @property
    def schema_dict(self) -> dict[str, Any]:
        """Returns the schema dictionary."""
        return self._schema_dict

    @schema_dict.setter
    def schema_dict(self, schema_dict: dict[str, Any]) -> None:
        """Sets the schema dictionary."""
        self._schema_dict = schema_dict

    def to_provider_config(self) -> dict[str, Any]:
        """Convert schema to OpenAI-specific configuration.

        Returns:
          Dictionary with response_format for OpenAI's json_schema mode.
        """
        return {
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "extractions",
                    "strict": True,
                    "schema": self._schema_dict,
                },
            },
        }

    @property
    def requires_raw_output(self) -> bool:
        """OpenAI json_schema mode outputs raw JSON without fences."""
        return True

    def validate_format(self, format_handler: fh.FormatHandler) -> None:
        """Validate OpenAI's format requirements.

        OpenAI json_schema mode requires:
        - No fence markers (outputs raw JSON)
        - Wrapper with EXTRACTIONS_KEY (built into the schema)
        """
        if format_handler.use_fences:
            warnings.warn(
                "OpenAI Structured Outputs produces native JSON via"
                " response_format='json_schema'. Using fence_output=True may"
                " cause parsing issues. Set fence_output=False.",
                UserWarning,
                stacklevel=3,
            )

        if not format_handler.use_wrapper or format_handler.wrapper_key != data.EXTRACTIONS_KEY:
            warnings.warn(
                "OpenAI's json_schema expects"
                f" wrapper_key='{data.EXTRACTIONS_KEY}'. Current settings:"
                f" use_wrapper={format_handler.use_wrapper},"
                f" wrapper_key='{format_handler.wrapper_key}'",
                UserWarning,
                stacklevel=3,
            )

    @classmethod
    def from_examples(
        cls,
        examples_data: Sequence[data.ExampleData],
        attribute_suffix: str = data.ATTRIBUTE_SUFFIX,
    ) -> OpenAISchema:
        """Creates an OpenAISchema from example extractions.

        Builds a JSON Schema with a top-level "extractions" array. Each
        element is an object containing the extraction class name and an
        accompanying "<class>_attributes" object for its attributes.

        All objects use ``additionalProperties: false`` and list every
        property in ``required`` as mandated by OpenAI strict mode.
        Attribute objects are wrapped in ``anyOf`` with ``{type: "null"}``
        to allow null when the category doesn't apply to a given item.

        Args:
          examples_data: A sequence of ExampleData objects containing
            extraction classes and attributes.
          attribute_suffix: String appended to each class name to form the
            attributes field name (defaults to "_attributes").

        Returns:
          An OpenAISchema ready for use with OpenAI's Structured Outputs.
        """
        extraction_categories: dict[str, dict[str, set[type]]] = {}
        for example in examples_data:
            for extraction in example.extractions:
                category = extraction.extraction_class
                if category not in extraction_categories:
                    extraction_categories[category] = {}

                if extraction.attributes:
                    for attr_name, attr_value in extraction.attributes.items():
                        if attr_name not in extraction_categories[category]:
                            extraction_categories[category][attr_name] = set()
                        extraction_categories[category][attr_name].add(type(attr_value))

        extraction_properties: dict[str, dict[str, Any]] = {}

        for category, attrs in extraction_categories.items():
            extraction_properties[category] = {"type": "string"}

            attributes_field = f"{category}{attribute_suffix}"
            attr_properties: dict[str, Any] = {}

            if not attrs:
                attr_properties["_unused"] = {"type": "string"}
            else:
                for attr_name, attr_types in attrs.items():
                    if list in attr_types:
                        attr_properties[attr_name] = {
                            "type": "array",
                            "items": {"type": "string"},
                        }
                    else:
                        attr_properties[attr_name] = {"type": "string"}

            # Wrap in anyOf with null to allow missing attribute objects
            # (when the category doesn't apply to a given extraction item)
            extraction_properties[attributes_field] = {
                "anyOf": [
                    {
                        "type": "object",
                        "properties": attr_properties,
                        "required": sorted(attr_properties.keys()),
                        "additionalProperties": False,
                    },
                    {"type": "null"},
                ],
            }

        extraction_schema = {
            "type": "object",
            "properties": extraction_properties,
            "required": sorted(extraction_properties.keys()),
            "additionalProperties": False,
        }

        schema_dict = {
            "type": "object",
            "properties": {
                data.EXTRACTIONS_KEY: {
                    "type": "array",
                    "items": extraction_schema,
                }
            },
            "required": [data.EXTRACTIONS_KEY],
            "additionalProperties": False,
        }

        return cls(_schema_dict=schema_dict)
