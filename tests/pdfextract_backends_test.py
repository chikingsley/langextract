"""Integration tests for configurable OCR backends."""

from __future__ import annotations

import base64
import json
import threading
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import TYPE_CHECKING, Any

import pymupdf
import pytest
from pdfextract import extract_document
from pdfextract.ocr_backends import available_ocr_backends, create_ocr_backend

if TYPE_CHECKING:
    from pathlib import Path


def _create_test_pdf(path: Path, pages: list[str]) -> Path:
    doc = pymupdf.open()
    for page_text in pages:
        page = doc.new_page()
        page.insert_text((72, 72), page_text, fontsize=12)
    doc.save(str(path))
    doc.close()
    return path


@contextmanager
def _json_test_server(response_payload: dict[str, Any], status_code: int = 200):
    requests: list[dict[str, Any]] = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self) -> None:
            content_length = int(self.headers.get("Content-Length", "0"))
            raw_body = self.rfile.read(content_length)
            payload = json.loads(raw_body.decode("utf-8")) if raw_body else {}
            requests.append(
                {
                    "path": self.path,
                    "headers": dict(self.headers),
                    "json": payload,
                }
            )
            encoded = json.dumps(response_payload).encode("utf-8")
            self.send_response(status_code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(encoded)))
            self.end_headers()
            self.wfile.write(encoded)

        def log_message(self, format: str, *args: Any) -> None:
            del format, args

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        host = str(server.server_address[0])
        port = int(server.server_address[1])
        yield f"http://{host}:{port}", requests
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def test_available_ocr_backends_contains_expected_names():
    names = set(available_ocr_backends())
    assert {"rapidocr", "ollama", "vllm"}.issubset(names)


def test_create_ocr_backend_rejects_unknown_backend():
    with pytest.raises(ValueError, match="Unknown OCR backend"):
        create_ocr_backend("does-not-exist")


@pytest.mark.integration
def test_extract_document_uses_ollama_backend_via_live_http(tmp_path: Path):
    path = _create_test_pdf(
        tmp_path / "ollama-backend.pdf",
        ["Native text should be ignored because OCR is forced."],
    )

    with _json_test_server({"message": {"content": "OCR text from Ollama"}}) as (
        base_url,
        requests,
    ):
        result = extract_document(
            path,
            force_ocr=True,
            ocr_backend="ollama",
            ocr_backend_options={
                "base_url": base_url,
                "model": "qwen2.5vl:7b",
                "timeout": 30.0,
            },
        )

    assert len(requests) == 1
    request = requests[0]
    assert request["path"] == "/api/chat"
    assert request["json"]["model"] == "qwen2.5vl:7b"
    assert request["json"]["messages"][0]["role"] == "user"
    image_payload = request["json"]["messages"][0]["images"][0]
    base64.b64decode(image_payload, validate=True)

    assert result.pages[0].source == "ocr"
    assert "OCR text from Ollama" in result.full_text
    assert result.word_map == []


@pytest.mark.integration
def test_extract_document_uses_vllm_backend_via_live_http(tmp_path: Path):
    path = _create_test_pdf(
        tmp_path / "vllm-backend.pdf",
        ["Native text should be ignored because OCR is forced."],
    )

    with _json_test_server({"choices": [{"message": {"content": "OCR text from vLLM"}}]}) as (
        base_url,
        requests,
    ):
        result = extract_document(
            path,
            force_ocr=True,
            ocr_backend="vllm",
            ocr_backend_options={
                "base_url": f"{base_url}/v1",
                "model": "Qwen/Qwen2.5-VL-7B-Instruct",
                "api_key": "test-key",
                "timeout": 30.0,
            },
        )

    assert len(requests) == 1
    request = requests[0]
    assert request["path"] == "/v1/chat/completions"
    assert request["headers"]["Authorization"] == "Bearer test-key"
    content = request["json"]["messages"][0]["content"]
    assert content[0]["type"] == "text"
    assert content[1]["type"] == "image_url"

    image_url = content[1]["image_url"]["url"]
    assert image_url.startswith("data:image/png;base64,")
    base64.b64decode(image_url.split(",", maxsplit=1)[1], validate=True)

    assert result.pages[0].source == "ocr"
    assert "OCR text from vLLM" in result.full_text
    assert result.word_map == []


@pytest.mark.integration
def test_extract_document_raises_when_backend_http_request_fails(tmp_path: Path):
    path = _create_test_pdf(
        tmp_path / "backend-error.pdf",
        ["Native text should be ignored because OCR is forced."],
    )

    with (
        _json_test_server({"error": "upstream failure"}, status_code=500) as (base_url, _),
        pytest.raises(Exception, match="500"),
    ):
        extract_document(
            path,
            force_ocr=True,
            ocr_backend="ollama",
            ocr_backend_options={"base_url": base_url, "model": "qwen2.5vl:7b"},
        )
