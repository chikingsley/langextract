"""Tests for extraction runtime branches that are easy to miss with mocks."""

from __future__ import annotations

import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import langextract as lx
import pytest
from langextract import prompt_validation as pv
from langextract.core import base_model, data, schema, types


class _StaticTextHandler(BaseHTTPRequestHandler):
    response_text = "Alice visited Seattle."

    def do_GET(self):
        payload = self.response_text.encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/plain; charset=utf-8")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, _format, *_args):
        return


@pytest.fixture
def local_text_url():
    server = ThreadingHTTPServer(("127.0.0.1", 0), _StaticTextHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    url = f"http://127.0.0.1:{server.server_port}/source.txt"
    try:
        yield url
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


class _EchoExtractorModel(base_model.BaseLanguageModel):
    """Small deterministic model for runtime-path extraction tests."""

    def __init__(self):
        super().__init__(constraint=schema.Constraint())
        self.format_type = data.FormatType.JSON

    def infer(self, batch_prompts, **kwargs):
        for prompt in batch_prompts:
            user_question = prompt.rsplit("\nQ: ", maxsplit=1)[-1]
            if user_question.startswith("Alice visited Seattle."):
                output = '{"extractions":[{"entity":"Alice"}]}'
            else:
                output = '{"extractions":[]}'
            yield [types.ScoredOutput(score=1.0, output=output)]


def _example_data():
    return [
        lx.data.ExampleData(
            text="Alice visited Seattle.",
            extractions=[lx.data.Extraction(extraction_class="entity", extraction_text="Alice")],
        )
    ]


def test_extract_fetches_url_content_when_enabled(local_text_url):
    result = lx.extract(
        text_or_documents=local_text_url,
        prompt_description="Extract named entities.",
        examples=_example_data(),
        model=_EchoExtractorModel(),
        fetch_urls=True,
        use_schema_constraints=False,
        prompt_validation_level=pv.PromptValidationLevel.OFF,
        show_progress=False,
    )

    assert isinstance(result, data.AnnotatedDocument)
    assert result.text == "Alice visited Seattle."
    assert any(extraction.extraction_text == "Alice" for extraction in result.extractions)


def test_extract_treats_url_as_literal_when_fetch_disabled(local_text_url):
    result = lx.extract(
        text_or_documents=local_text_url,
        prompt_description="Extract named entities.",
        examples=_example_data(),
        model=_EchoExtractorModel(),
        fetch_urls=False,
        use_schema_constraints=False,
        prompt_validation_level=pv.PromptValidationLevel.OFF,
        show_progress=False,
    )

    assert isinstance(result, data.AnnotatedDocument)
    assert result.text == local_text_url
    assert result.extractions == []


def test_extract_raises_with_prompt_validation_error_level():
    bad_examples = [
        lx.data.ExampleData(
            text="Alice visited Seattle.",
            extractions=[
                lx.data.Extraction(
                    extraction_class="entity",
                    extraction_text="NOT_IN_SOURCE_TEXT",
                )
            ],
        )
    ]

    with pytest.raises(pv.PromptAlignmentError):
        lx.extract(
            text_or_documents="any text",
            prompt_description="Extract entities.",
            examples=bad_examples,
            model=_EchoExtractorModel(),
            fetch_urls=False,
            use_schema_constraints=False,
            prompt_validation_level=pv.PromptValidationLevel.ERROR,
            show_progress=False,
        )
