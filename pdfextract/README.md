# `pdfextract` OCR Backends

`extract_document(...)` supports pluggable OCR backends via:

- `ocr_backend`: backend name (`rapidocr`, `ollama`, `vllm`)
- `ocr_backend_options`: provider-specific init kwargs

Example:

```python
from pdfextract import extract_document

result = extract_document(
    "contract.pdf",
    force_ocr=True,
    ocr_backend="vllm",
    ocr_backend_options={
        "base_url": "http://localhost:8000/v1",
        "model": "Qwen/Qwen2.5-VL-7B-Instruct",
        "api_key": "EMPTY",
    },
)
```

## Backend Contract

Backends implement `OcrBackend` from `pdfextract/ocr_backends.py` and return
normalized `OcrBackendResult`:

- `text`: OCR text for the page
- `words`: optional word boxes (`OcrWord`) with page-coordinate geometry and
  page-local character spans

`rapidocr` returns text + word boxes. `ollama` and `vllm` currently return text
only (`words=[]`), because those APIs do not return word-level bounding boxes by default.

## Adding a New Backend

1. Implement an `OcrBackend` subclass in `pdfextract/ocr_backends.py`.
2. Register it in `_BACKENDS`.
3. Add integration tests under `tests/pdfextract_backends_test.py`.
