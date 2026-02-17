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

"""Tests for Gemini Batch API functionality (inline batch submission)."""

from unittest import mock

from absl.testing import absltest, parameterized
from google import genai
from langextract.providers import gemini
from langextract.providers import gemini_batch as gb


def _create_mock_batch_job(
    state=genai.types.JobState.JOB_STATE_SUCCEEDED,
):
    """Create a mock BatchJob for testing."""
    job = mock.create_autospec(genai.types.BatchJob, instance=True)
    job.name = "batches/123"
    job.state = state
    job.responses = []
    return job


def _create_mock_response(text_content):
    """Create a mock response with text."""
    resp = mock.MagicMock()
    resp.text = text_content
    return resp


class TestGeminiBatchAPI(absltest.TestCase):
    """Test Gemini Batch API routing and inline submission."""

    @mock.patch.object(genai, "Client", autospec=True)
    def test_batch_routing(self, mock_client_cls):
        """Test that batch API is used when enabled and threshold is met."""
        mock_client = mock_client_cls.return_value
        mock_client.vertexai = True

        job = _create_mock_batch_job()
        job.responses = [_create_mock_response('{"ok":1}'), _create_mock_response('{"ok":2}')]

        mock_client.batches.create.return_value = job
        mock_client.batches.get.return_value = job

        model = gemini.GeminiLanguageModel(
            model_id="gemini-2.5-flash",
            vertexai=True,
            project="test-project",
            location="us-central1",
            batch={
                "enabled": True,
                "threshold": 2,
                "poll_interval": 1,
            },
        )
        outs = list(model.infer(["p1", "p2"]))

        self.assertLen(outs, 2)
        self.assertEqual(outs[0][0].output, '{"ok":1}')
        self.assertEqual(outs[1][0].output, '{"ok":2}')
        mock_client.batches.create.assert_called()

    @mock.patch.object(genai, "Client", autospec=True)
    def test_realtime_when_disabled(self, mock_client_cls):
        """Test that real-time API is used when batch is disabled."""
        mock_client = mock_client_cls.return_value
        mock_client.vertexai = True
        mock_response = mock.create_autospec(genai.types.GenerateContentResponse, instance=True)
        mock_response.text = '{"ok":1}'
        mock_client.models.generate_content.return_value = mock_response

        model = gemini.GeminiLanguageModel(
            model_id="gemini-2.5-flash",
            vertexai=True,
            project="p",
            location="l",
            batch={"enabled": False},
        )
        outs = list(model.infer(["hello"]))

        self.assertLen(outs, 1)
        self.assertEqual(outs[0][0].output, '{"ok":1}')
        mock_client.models.generate_content.assert_called()
        mock_client.batches.create.assert_not_called()

    @mock.patch.object(genai, "Client", autospec=True)
    def test_cached_content_passed_to_realtime_config(self, mock_client_cls):
        """Test cached_content is forwarded for real-time generate_content calls."""
        mock_client = mock_client_cls.return_value
        mock_client.vertexai = True
        mock_response = mock.create_autospec(genai.types.GenerateContentResponse, instance=True)
        mock_response.text = '{"ok":1}'
        mock_client.models.generate_content.return_value = mock_response

        model = gemini.GeminiLanguageModel(
            model_id="gemini-2.5-flash",
            vertexai=True,
            project="p",
            location="l",
            batch={"enabled": False},
        )
        outs = list(model.infer(["hello"], cached_content="cachedContents/abc"))

        self.assertLen(outs, 1)
        self.assertEqual(outs[0][0].output, '{"ok":1}')
        call_kwargs = mock_client.models.generate_content.call_args.kwargs
        config_arg = call_kwargs["config"]
        self.assertEqual(config_arg.get("cached_content"), "cachedContents/abc")

    @mock.patch.object(genai, "Client", autospec=True)
    def test_cached_content_bypasses_batch(self, mock_client_cls):
        """Test cached_content disables batch path and falls back to real-time."""
        mock_client = mock_client_cls.return_value
        mock_client.vertexai = True

        mock_response_1 = mock.create_autospec(genai.types.GenerateContentResponse, instance=True)
        mock_response_1.text = '{"ok":1}'
        mock_response_2 = mock.create_autospec(genai.types.GenerateContentResponse, instance=True)
        mock_response_2.text = '{"ok":2}'
        mock_client.models.generate_content.side_effect = [mock_response_1, mock_response_2]

        model = gemini.GeminiLanguageModel(
            model_id="gemini-2.5-flash",
            vertexai=True,
            project="p",
            location="l",
            batch={
                "enabled": True,
                "threshold": 1,
            },
        )
        outs = list(model.infer(["p1", "p2"], cached_content="cachedContents/abc"))

        self.assertLen(outs, 2)
        self.assertEqual(outs[0][0].output, '{"ok":1}')
        self.assertEqual(outs[1][0].output, '{"ok":2}')
        mock_client.batches.create.assert_not_called()
        self.assertEqual(mock_client.models.generate_content.call_count, 2)

    @mock.patch.object(genai, "Client", autospec=True)
    def test_realtime_when_below_threshold(self, mock_client_cls):
        """Test that real-time API is used when prompt count is below threshold."""
        mock_client = mock_client_cls.return_value
        mock_client.vertexai = True
        mock_response = mock.create_autospec(genai.types.GenerateContentResponse, instance=True)
        mock_response.text = '{"ok":1}'
        mock_client.models.generate_content.return_value = mock_response

        model = gemini.GeminiLanguageModel(
            model_id="gemini-2.5-flash",
            vertexai=True,
            project="p",
            location="l",
            batch={
                "enabled": True,
                "threshold": 10,
            },
        )
        outs = list(model.infer(["hello"]))

        self.assertLen(outs, 1)
        self.assertEqual(outs[0][0].output, '{"ok":1}')
        mock_client.models.generate_content.assert_called()
        mock_client.batches.create.assert_not_called()

    @mock.patch.object(genai, "Client", autospec=True)
    def test_batch_with_schema(self, mock_client_cls):
        """Test that batch API properly includes schema when configured."""
        mock_client = mock_client_cls.return_value
        mock_client.vertexai = True

        job = _create_mock_batch_job()
        job.responses = [_create_mock_response('{"name":"test"}')]

        mock_client.batches.create.return_value = job
        mock_client.batches.get.return_value = job

        from langextract.providers import schemas

        mock_schema = mock.create_autospec(schemas.gemini.GeminiSchema, instance=True)
        mock_schema.schema_dict = {
            "type": "object",
            "properties": {"name": {"type": "string"}},
        }

        model = gemini.GeminiLanguageModel(
            model_id="gemini-2.5-flash",
            vertexai=True,
            project="p",
            location="l",
            gemini_schema=mock_schema,
            batch={
                "enabled": True,
                "threshold": 1,
            },
        )

        with mock.patch.object(gb, "_submit_inline", autospec=True) as mock_submit:
            mock_submit.return_value = job

            outs = list(model.infer(["test prompt"]))

            self.assertLen(outs, 1)
            self.assertEqual(outs[0][0].output, '{"name":"test"}')
            mock_submit.assert_called_once()

        assert model.gemini_schema is not None
        self.assertEqual(model.gemini_schema.schema_dict, mock_schema.schema_dict)

    @mock.patch.object(genai, "Client", autospec=True)
    def test_batch_error_handling(self, mock_client_cls):
        """Test that batch errors are properly handled and raised."""
        mock_client = mock_client_cls.return_value
        mock_client.vertexai = True
        mock_client.batches.create.side_effect = Exception("Batch API error")

        model = gemini.GeminiLanguageModel(
            model_id="gemini-2.5-flash",
            vertexai=True,
            project="p",
            location="l",
            batch={
                "enabled": True,
                "threshold": 1,
            },
        )

        with self.assertRaisesRegex(Exception, "Gemini Batch API error"):
            list(model.infer(["test prompt"]))

    @mock.patch.object(genai, "Client", autospec=True)
    def test_max_prompts_per_job(self, mock_client_cls):
        """Test request splitting into multiple jobs when max_prompts_per_job is exceeded."""
        mock_client = mock_client_cls.return_value
        mock_client.vertexai = True

        prompts = ["p1", "p2", "p3", "p4", "p5"]

        # 3 jobs: [p1,p2], [p3,p4], [p5]
        job0 = _create_mock_batch_job()
        job0.responses = [_create_mock_response("r1"), _create_mock_response("r2")]
        job1 = _create_mock_batch_job()
        job1.responses = [_create_mock_response("r3"), _create_mock_response("r4")]
        job2 = _create_mock_batch_job()
        job2.responses = [_create_mock_response("r5")]

        mock_client.batches.create.side_effect = [job0, job1, job2]
        mock_client.batches.get.side_effect = [job0, job1, job2]

        model = gemini.GeminiLanguageModel(
            model_id="gemini-2.5-flash",
            vertexai=True,
            project="p",
            location="l",
            batch={
                "enabled": True,
                "threshold": 1,
                "max_prompts_per_job": 2,
            },
        )

        results = list(model.infer(prompts))

        self.assertEqual(mock_client.batches.create.call_count, 3)
        self.assertListEqual([r[0].output for r in results], ["r1", "r2", "r3", "r4", "r5"])


class BatchConfigValidationTest(parameterized.TestCase):
    """Test BatchConfig validation logic."""

    @parameterized.named_parameters(
        {"testcase_name": "threshold_lt_1", "threshold": 0},
        {"testcase_name": "poll_interval_le_0", "poll_interval": 0},
        {"testcase_name": "timeout_le_0", "timeout": 0},
        {"testcase_name": "max_prompts_per_job_le_0", "max_prompts_per_job": 0},
    )
    def test_validation_errors(self, **overrides):
        """Verify validation errors for invalid config values."""
        with self.assertRaises(ValueError):
            gb.BatchConfig(**overrides)


class EmptyAndPaddingTest(absltest.TestCase):
    """Test empty prompt handling and result padding."""

    @mock.patch.object(genai, "Client", autospec=True)
    def test_empty_prompts_fast_path(self, mock_client_cls):
        """Verify empty prompts return immediately without API calls."""
        outs = gb.infer_batch(
            client=mock_client_cls.return_value,
            model_id="m",
            prompts=[],
            schema_dict=None,
            gen_config={},
            cfg=gb.BatchConfig(enabled=True, poll_interval=1),
        )
        self.assertEqual(outs, [])

    @mock.patch.object(genai, "Client", autospec=True)
    def test_pad_to_expected_count(self, mock_client_cls):
        """Verify padding to maintain 1:1 alignment with input prompts."""
        mock_client = mock_client_cls.return_value

        # Job returns only 1 response for 2 prompts
        job = _create_mock_batch_job()
        job.responses = [_create_mock_response("only_one")]

        mock_client.batches.create.return_value = job
        mock_client.batches.get.return_value = job

        cfg = gb.BatchConfig(enabled=True, threshold=1, poll_interval=1)
        outs = gb.infer_batch(
            client=mock_client,
            model_id="m",
            prompts=["p1", "p2"],
            schema_dict=None,
            gen_config={},
            cfg=cfg,
        )
        self.assertLen(outs, 2)
        self.assertEqual(outs[0][0], "only_one")
        self.assertEqual(outs[1][0], "")  # padded


class BuildInlineRequestTest(absltest.TestCase):
    """Test inline request construction."""

    def test_basic_request(self):
        """Test simple prompt without schema or config."""
        req = gb._build_inline_request("hello", None, {})
        self.assertEqual(req.contents, "hello")

    def test_request_with_schema(self):
        """Test request with JSON schema."""
        schema = {"type": "object", "properties": {"name": {"type": "string"}}}
        req = gb._build_inline_request("hello", schema, {"temperature": 0.5})
        config_obj = req.config
        self.assertIsNotNone(config_obj)
        assert config_obj is not None
        self.assertEqual(config_obj.response_mime_type, "application/json")
        self.assertEqual(config_obj.response_schema, schema)
        self.assertEqual(config_obj.temperature, 0.5)

    def test_request_with_system_instruction(self):
        """Test request with system instruction."""
        req = gb._build_inline_request("hello", None, {}, system_instruction="Be helpful")
        config_obj = req.config
        self.assertIsNotNone(config_obj)
        assert config_obj is not None
        self.assertEqual(config_obj.system_instruction, "Be helpful")


if __name__ == "__main__":
    absltest.main()
