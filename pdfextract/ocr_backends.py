"""OCR backend contract and provider implementations."""

from __future__ import annotations

import abc
import base64
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import httpx

if TYPE_CHECKING:
    import pymupdf
    from rapidocr import RapidOCR


@dataclass(frozen=True, slots=True)
class OcrWord:
    """Word-level OCR result in page coordinate space."""

    text: str
    x0: float
    y0: float
    x1: float
    y1: float
    char_start: int
    char_end: int


@dataclass(frozen=True, slots=True)
class OcrBackendResult:
    """Normalized OCR output from a backend."""

    text: str
    words: list[OcrWord]


class OcrBackend(abc.ABC):
    """Contract implemented by OCR providers."""

    name: str = ""
    supports_word_bboxes: bool = False

    @abc.abstractmethod
    def extract_page(
        self,
        *,
        pixmap: pymupdf.Pixmap,
        dpi: int,
        prompt: str,
    ) -> OcrBackendResult:
        """Extract OCR text (and optional word boxes) from a rendered page."""


def _normalize_text(text: str) -> str:
    stripped = text.strip()
    if not stripped:
        return ""
    return f"{stripped}\n"


class RapidOcrBackend(OcrBackend):
    """RapidOCR backend (ONNX)."""

    name = "rapidocr"
    supports_word_bboxes = True

    def __init__(self) -> None:
        self._engine: RapidOCR | None = None

    def _get_engine(self) -> RapidOCR:
        if self._engine is None:
            try:
                from rapidocr import RapidOCR
            except ImportError as exc:  # pragma: no cover - dependency guard
                raise ImportError(
                    "RapidOCR is required for rapidocr backend. "
                    "Install with: uv add rapidocr onnxruntime"
                ) from exc
            self._engine = RapidOCR()
        return self._engine

    def _parse_word_results(
        self,
        word_results: Any,
        *,
        zoom: float,
    ) -> OcrBackendResult:
        text_parts: list[str] = []
        words: list[OcrWord] = []
        current_offset = 0

        for line_idx, line_words in enumerate(word_results):
            if line_idx > 0:
                text_parts.append("\n")
                current_offset += 1

            for word_idx, word_entry in enumerate(line_words):
                if len(word_entry) < 3:
                    continue

                word_text = str(word_entry[0])
                bbox_points = word_entry[2]
                if not word_text.strip():
                    continue

                if word_idx > 0:
                    text_parts.append(" ")
                    current_offset += 1

                xs = [float(point[0]) / zoom for point in bbox_points]
                ys = [float(point[1]) / zoom for point in bbox_points]
                word_start = current_offset
                word_end = word_start + len(word_text)

                words.append(
                    OcrWord(
                        text=word_text,
                        x0=min(xs),
                        y0=min(ys),
                        x1=max(xs),
                        y1=max(ys),
                        char_start=word_start,
                        char_end=word_end,
                    )
                )
                text_parts.append(word_text)
                current_offset = word_end

        text = "".join(text_parts)
        if text:
            text = f"{text}\n"
        return OcrBackendResult(text=text, words=words)

    def _parse_txts_and_boxes(
        self,
        txts: Any,
        boxes: Any,
        *,
        zoom: float,
    ) -> OcrBackendResult:
        text_parts: list[str] = []
        words: list[OcrWord] = []
        current_offset = 0

        txt_list = list(txts or [])
        box_list = list(boxes or [])
        count = min(len(txt_list), len(box_list))

        for idx in range(count):
            word_text = str(txt_list[idx]).strip()
            if not word_text:
                continue

            if text_parts:
                text_parts.append("\n")
                current_offset += 1

            raw_box = box_list[idx]
            if hasattr(raw_box, "tolist"):
                raw_box = raw_box.tolist()
            points = list(raw_box)
            if len(points) < 4:
                continue

            xs = [float(point[0]) / zoom for point in points]
            ys = [float(point[1]) / zoom for point in points]
            word_start = current_offset
            word_end = word_start + len(word_text)

            words.append(
                OcrWord(
                    text=word_text,
                    x0=min(xs),
                    y0=min(ys),
                    x1=max(xs),
                    y1=max(ys),
                    char_start=word_start,
                    char_end=word_end,
                )
            )
            text_parts.append(word_text)
            current_offset = word_end

        text = "".join(text_parts)
        if text:
            text = f"{text}\n"
        return OcrBackendResult(text=text, words=words)

    def extract_page(
        self,
        *,
        pixmap: pymupdf.Pixmap,
        dpi: int,
        prompt: str,
    ) -> OcrBackendResult:
        del prompt
        import numpy as np

        zoom = dpi / 72
        image_array = np.frombuffer(pixmap.samples, dtype=np.uint8).reshape(
            pixmap.height,
            pixmap.width,
            pixmap.n,
        )
        result = self._get_engine()(image_array, return_word_box=True)
        txts = getattr(result, "txts", None)
        boxes = getattr(result, "boxes", None)
        if txts and boxes:
            return self._parse_txts_and_boxes(txts, boxes, zoom=zoom)
        word_results = getattr(result, "word_results", None)
        if not word_results:
            return OcrBackendResult(text="", words=[])
        return self._parse_word_results(word_results, zoom=zoom)


class OllamaOcrBackend(OcrBackend):
    """Ollama vision backend via `/api/chat`."""

    name = "ollama"
    supports_word_bboxes = False

    def __init__(
        self,
        *,
        base_url: str = "http://localhost:11434",
        model: str = "qwen2.5vl:7b",
        timeout: float = 120.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout

    def extract_page(
        self,
        *,
        pixmap: pymupdf.Pixmap,
        dpi: int,
        prompt: str,
    ) -> OcrBackendResult:
        del dpi
        image_b64 = base64.b64encode(pixmap.tobytes("png")).decode("ascii")
        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "user",
                    "content": prompt,
                    "images": [image_b64],
                }
            ],
            "stream": False,
            "options": {"temperature": 0},
        }
        response = httpx.post(
            f"{self.base_url}/api/chat",
            json=payload,
            timeout=self.timeout,
        )
        response.raise_for_status()
        data = response.json()

        text = ""
        message = data.get("message")
        if isinstance(message, dict):
            message_content = message.get("content")
            if isinstance(message_content, str):
                text = message_content
        if not text:
            response_text = data.get("response")
            if isinstance(response_text, str):
                text = response_text

        return OcrBackendResult(text=_normalize_text(text), words=[])


class VllmOcrBackend(OcrBackend):
    """vLLM OpenAI-compatible vision backend."""

    name = "vllm"
    supports_word_bboxes = False

    def __init__(
        self,
        *,
        base_url: str = "http://localhost:8000/v1",
        model: str,
        api_key: str = "EMPTY",
        timeout: float = 120.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key
        self.timeout = timeout

    def extract_page(
        self,
        *,
        pixmap: pymupdf.Pixmap,
        dpi: int,
        prompt: str,
    ) -> OcrBackendResult:
        del dpi
        image_b64 = base64.b64encode(pixmap.tobytes("png")).decode("ascii")
        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/png;base64,{image_b64}"},
                        },
                    ],
                }
            ],
            "stream": False,
            "temperature": 0,
        }
        headers = {"Authorization": f"Bearer {self.api_key}"}
        response = httpx.post(
            f"{self.base_url}/chat/completions",
            json=payload,
            headers=headers,
            timeout=self.timeout,
        )
        response.raise_for_status()
        data = response.json()

        text = ""
        choices = data.get("choices")
        if isinstance(choices, list) and choices:
            first_choice = choices[0]
            if isinstance(first_choice, dict):
                message = first_choice.get("message")
                if isinstance(message, dict):
                    content = message.get("content")
                    if isinstance(content, str):
                        text = content
                    elif isinstance(content, list):
                        text_parts = [
                            str(item.get("text"))
                            for item in content
                            if isinstance(item, dict)
                            and item.get("type") == "text"
                            and isinstance(item.get("text"), str)
                        ]
                        text = "\n".join(text_parts)

        return OcrBackendResult(text=_normalize_text(text), words=[])


_BACKENDS: dict[str, type[OcrBackend]] = {
    RapidOcrBackend.name: RapidOcrBackend,
    OllamaOcrBackend.name: OllamaOcrBackend,
    VllmOcrBackend.name: VllmOcrBackend,
}

# Instance cache: backends with no kwargs are reused across calls.
# This avoids reloading ONNX models (RapidOCR) or reconnecting HTTP clients
# on every page when the same backend+options combination is used.
_INSTANCE_CACHE: dict[str, OcrBackend] = {}


def available_ocr_backends() -> tuple[str, ...]:
    """Return supported OCR backend names."""
    return tuple(sorted(_BACKENDS))


def create_ocr_backend(name: str = "rapidocr", **kwargs: Any) -> OcrBackend:
    """Return an OCR backend instance by registry name.

    Instances are cached by name when no kwargs are provided, so repeated
    calls for the same no-config backend (e.g. "rapidocr") reuse the same
    object and avoid reloading ONNX models on every page.
    """
    backend_name = name.strip().lower()
    backend_cls = _BACKENDS.get(backend_name)
    if backend_cls is None:
        available = ", ".join(available_ocr_backends())
        raise ValueError(f"Unknown OCR backend '{name}'. Available backends: {available}")
    if not kwargs:
        cached = _INSTANCE_CACHE.get(backend_name)
        if cached is None:
            cached = backend_cls()
            _INSTANCE_CACHE[backend_name] = cached
        return cached
    return backend_cls(**kwargs)


__all__ = [
    "OcrBackend",
    "OcrBackendResult",
    "OcrWord",
    "OllamaOcrBackend",
    "RapidOcrBackend",
    "VllmOcrBackend",
    "available_ocr_backends",
    "create_ocr_backend",
]
