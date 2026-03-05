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

"""Tests for OpenAISchema structured output support."""

import warnings
from unittest import mock

from langextract.core import data
from langextract.core import format_handler as fh
from langextract.providers.schemas.openai import OpenAISchema


def _make_examples(
    categories: dict[str, dict[str, object]] | None = None,
) -> list[data.ExampleData]:
    """Build ExampleData list from a category→attributes mapping."""
    if categories is None:
        categories = {
            "medication": {"dosage": "10mg", "route": "oral"},
        }

    extractions = []
    for cat, attrs in categories.items():
        extractions.append(
            data.Extraction(
                extraction_class=cat,
                extraction_text=cat,
                attributes=attrs if attrs else None,
            )
        )

    return [data.ExampleData(text="sample text", extractions=extractions)]


class TestFromExamples:
    def test_basic_schema_structure(self):
        schema = OpenAISchema.from_examples(_make_examples())
        sd = schema.schema_dict

        assert sd["type"] == "object"
        assert sd["additionalProperties"] is False
        assert sd["required"] == [data.EXTRACTIONS_KEY]

        items = sd["properties"][data.EXTRACTIONS_KEY]["items"]
        assert items["type"] == "object"
        assert items["additionalProperties"] is False

    def test_category_and_attributes_present(self):
        schema = OpenAISchema.from_examples(_make_examples())
        items = schema.schema_dict["properties"][data.EXTRACTIONS_KEY]["items"]
        props = items["properties"]

        assert "medication" in props
        assert props["medication"]["type"] == "string"

        attr_key = "medication_attributes"
        assert attr_key in props

        # Attributes wrapped in anyOf for nullable support
        any_of = props[attr_key]["anyOf"]
        assert len(any_of) == 2
        obj_schema = any_of[0]
        assert obj_schema["type"] == "object"
        assert "dosage" in obj_schema["properties"]
        assert "route" in obj_schema["properties"]
        assert obj_schema["additionalProperties"] is False
        assert any_of[1] == {"type": "null"}

    def test_all_properties_in_required(self):
        schema = OpenAISchema.from_examples(_make_examples())
        items = schema.schema_dict["properties"][data.EXTRACTIONS_KEY]["items"]

        # All extraction item properties must be in required
        assert sorted(items["properties"].keys()) == items["required"]

    def test_attribute_properties_in_required(self):
        schema = OpenAISchema.from_examples(_make_examples())
        items = schema.schema_dict["properties"][data.EXTRACTIONS_KEY]["items"]
        attr_obj = items["properties"]["medication_attributes"]["anyOf"][0]

        assert sorted(attr_obj["properties"].keys()) == attr_obj["required"]

    def test_list_attribute_becomes_array(self):
        examples = _make_examples({"symptom": {"related_conditions": ["flu", "cold"]}})
        schema = OpenAISchema.from_examples(examples)
        items = schema.schema_dict["properties"][data.EXTRACTIONS_KEY]["items"]
        attr_obj = items["properties"]["symptom_attributes"]["anyOf"][0]

        assert attr_obj["properties"]["related_conditions"] == {
            "type": "array",
            "items": {"type": "string"},
        }

    def test_no_attributes_uses_unused_placeholder(self):
        examples = _make_examples({"tag": {}})
        schema = OpenAISchema.from_examples(examples)
        items = schema.schema_dict["properties"][data.EXTRACTIONS_KEY]["items"]
        attr_obj = items["properties"]["tag_attributes"]["anyOf"][0]

        assert "_unused" in attr_obj["properties"]
        assert attr_obj["required"] == ["_unused"]

    def test_multiple_categories(self):
        examples = _make_examples({
            "medication": {"dosage": "10mg"},
            "condition": {"severity": "mild"},
        })
        schema = OpenAISchema.from_examples(examples)
        items = schema.schema_dict["properties"][data.EXTRACTIONS_KEY]["items"]
        props = items["properties"]

        assert "medication" in props
        assert "medication_attributes" in props
        assert "condition" in props
        assert "condition_attributes" in props
        assert len(items["required"]) == 4


class TestToProviderConfig:
    def test_returns_json_schema_format(self):
        schema = OpenAISchema.from_examples(_make_examples())
        config = schema.to_provider_config()

        rf = config["response_format"]
        assert rf["type"] == "json_schema"
        assert rf["json_schema"]["name"] == "extractions"
        assert rf["json_schema"]["strict"] is True
        assert "schema" in rf["json_schema"]

    def test_schema_matches_schema_dict(self):
        schema = OpenAISchema.from_examples(_make_examples())
        config = schema.to_provider_config()

        assert config["response_format"]["json_schema"]["schema"] is schema.schema_dict


class TestProperties:
    def test_requires_raw_output(self):
        schema = OpenAISchema.from_examples(_make_examples())
        assert schema.requires_raw_output is True

    def test_schema_dict_getter_setter(self):
        schema = OpenAISchema.from_examples(_make_examples())
        original = schema.schema_dict
        schema.schema_dict = {"type": "object"}
        assert schema.schema_dict == {"type": "object"}
        assert schema.schema_dict != original


class TestValidateFormat:
    def test_warns_on_fence_output(self):
        schema = OpenAISchema.from_examples(_make_examples())
        handler = fh.FormatHandler(
            format_type=data.FormatType.JSON,
            use_fences=True,
            attribute_suffix=data.ATTRIBUTE_SUFFIX,
            use_wrapper=True,
            wrapper_key=data.EXTRACTIONS_KEY,
        )
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            schema.validate_format(handler)
            fence_warns = [x for x in w if "fence_output" in str(x.message)]
            assert len(fence_warns) == 1

    def test_warns_on_wrong_wrapper(self):
        schema = OpenAISchema.from_examples(_make_examples())
        handler = fh.FormatHandler(
            format_type=data.FormatType.JSON,
            use_fences=False,
            attribute_suffix=data.ATTRIBUTE_SUFFIX,
            use_wrapper=False,
            wrapper_key="other",
        )
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            schema.validate_format(handler)
            wrapper_warns = [x for x in w if "wrapper_key" in str(x.message)]
            assert len(wrapper_warns) == 1

    def test_no_warnings_when_correct(self):
        schema = OpenAISchema.from_examples(_make_examples())
        handler = fh.FormatHandler(
            format_type=data.FormatType.JSON,
            use_fences=False,
            attribute_suffix=data.ATTRIBUTE_SUFFIX,
            use_wrapper=True,
            wrapper_key=data.EXTRACTIONS_KEY,
        )
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            schema.validate_format(handler)
            assert len(w) == 0


class TestOpenAIProviderIntegration:
    def test_get_schema_class_returns_openai_schema(self):
        from langextract.providers.openai import OpenAILanguageModel

        assert OpenAILanguageModel.get_schema_class() is OpenAISchema

    @mock.patch("openai.OpenAI")
    def test_schema_response_format_flows_to_api(self, mock_openai_cls):
        """Schema's response_format reaches _process_single_prompt."""
        from langextract.providers.openai import OpenAILanguageModel

        mock_client = mock.MagicMock()
        mock_openai_cls.return_value = mock_client

        # Simulate what the factory does: schema → to_provider_config → kwargs
        schema = OpenAISchema.from_examples(_make_examples())
        schema_config = schema.to_provider_config()

        model = OpenAILanguageModel(
            model_id="gpt-4o-mini",
            api_key="test-key",
            **schema_config,
        )
        model.apply_schema(schema)

        # Mock response
        mock_choice = mock.MagicMock()
        mock_choice.message.content = '{"extractions": []}'
        mock_response = mock.MagicMock()
        mock_response.choices = [mock_choice]
        mock_response.usage = None
        mock_client.chat.completions.create.return_value = mock_response

        # Run inference
        list(model.infer(["Extract medications"]))

        call_kwargs = mock_client.chat.completions.create.call_args[1]
        rf = call_kwargs["response_format"]
        assert rf["type"] == "json_schema"
        assert rf["json_schema"]["strict"] is True

    @mock.patch("openai.OpenAI")
    def test_openrouter_adds_require_parameters(self, mock_openai_cls):
        """OpenRouter base_url triggers require_parameters in extra_body."""
        from langextract.providers.openai import OpenAILanguageModel

        mock_client = mock.MagicMock()
        mock_openai_cls.return_value = mock_client

        schema = OpenAISchema.from_examples(_make_examples())
        schema_config = schema.to_provider_config()

        model = OpenAILanguageModel(
            model_id="gpt-4o-mini",
            api_key="test-key",
            base_url="https://openrouter.ai/api/v1",
            **schema_config,
        )
        model.apply_schema(schema)

        mock_choice = mock.MagicMock()
        mock_choice.message.content = '{"extractions": []}'
        mock_response = mock.MagicMock()
        mock_response.choices = [mock_choice]
        mock_response.usage = None
        mock_client.chat.completions.create.return_value = mock_response

        list(model.infer(["Extract medications"]))

        call_kwargs = mock_client.chat.completions.create.call_args[1]
        assert call_kwargs["extra_body"]["provider"]["require_parameters"] is True

    @mock.patch("openai.OpenAI")
    def test_non_openrouter_no_extra_body(self, mock_openai_cls):
        """Non-OpenRouter URLs should not add extra_body."""
        from langextract.providers.openai import OpenAILanguageModel

        mock_client = mock.MagicMock()
        mock_openai_cls.return_value = mock_client

        schema = OpenAISchema.from_examples(_make_examples())
        schema_config = schema.to_provider_config()

        model = OpenAILanguageModel(
            model_id="gpt-4o-mini",
            api_key="test-key",
            **schema_config,
        )
        model.apply_schema(schema)

        mock_choice = mock.MagicMock()
        mock_choice.message.content = '{"extractions": []}'
        mock_response = mock.MagicMock()
        mock_response.choices = [mock_choice]
        mock_response.usage = None
        mock_client.chat.completions.create.return_value = mock_response

        list(model.infer(["Extract medications"]))

        call_kwargs = mock_client.chat.completions.create.call_args[1]
        assert "extra_body" not in call_kwargs


class TestOpenAISchemaImport:
    def test_importable_from_schemas_package(self):
        from langextract.providers.schemas import OpenAISchema as imported

        assert imported is OpenAISchema
