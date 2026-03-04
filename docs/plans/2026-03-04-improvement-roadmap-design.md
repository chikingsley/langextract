# LangExtract Improvement Roadmap — Design Document

**Date**: 2026-03-04
**Status**: Approved
**Author**: Claude + Chi

---

## Context

This fork (lingextract-plus-plus v2.0.0) diverged from Google's langextract at commit `3638fe4`
(cross-chunk context awareness, Dec 2025). Our modernization pass added:

- Python 3.14 target, modern typing (`X | Y`, `TYPE_CHECKING` guards)
- `pdfextract/` sub-package with pluggable OCR backends (RapidOCR, Ollama vision, vLLM vision)
- `core/format_handler.py` — centralized fence/format parsing with think-tag stripping
- `settings.py` — pydantic-settings config, centralized API key resolution
- `prompt_validation.py` — few-shot example alignment validation
- `gemini_batch.py` — Gemini Batch API support
- `cli.py` — command-line interface
- httpx replacing requests, ruff replacing pylint, ty replacing pytype

Upstream's `feature/refactor-architecture` branch (never merged) targeted the same layered
architecture we already have. We are ahead of upstream's architectural vision.

**Key gap**: zero test coverage. Upstream had 24 test files with ~377 methods that were lost
during our modernization.

---

## Testing Policy: Real Endpoints First

**Core principle**: Test with actual LLM endpoints and real data wherever possible. Mocks are
acceptable only for unit-testing internal logic (parsing, chunking, alignment) where the LLM
call is not the thing under test.

### Available Test Endpoints

| Endpoint | Host | URL | Models | Use For |
|----------|------|-----|--------|---------|
| **Ollama (home-mac)** | Tailscale | `http://100.113.195.95:11434` | qwen3.5:4b, glm-ocr | Ollama provider tests, OCR vision tests |
| **Ollama (gmk-server)** | localhost | `http://localhost:11434` | See CLAUDE.md | Local tests when available |
| **OpenRouter** | cloud | `https://openrouter.ai/api/v1` | gemini-2.5-flash, gpt-4o-mini, etc. | OpenAI provider tests, Gemini via OpenAI-compat |
| **Gemini API** | cloud | direct via google-genai SDK | gemini-2.5-flash-lite | Gemini provider tests |

### Environment Variables for Tests

```bash
# Required for live tests
OLLAMA_BASE_URL=http://100.113.195.95:11434    # home-mac via Tailscale
OPENROUTER_API_KEY=sk-or-...                    # OpenRouter for multi-model
GEMINI_API_KEY=...                              # Direct Gemini API
OPENAI_API_KEY=...                              # Or OpenRouter key with base_url override

# Test markers
# pytest -m live_api       — runs tests hitting real endpoints
# pytest -m integration    — runs integration tests (may use endpoints)
# pytest -m "not live_api" — unit tests only (mocked)
```

### OpenRouter Integration

OpenRouter provides a single API key for accessing multiple models via an OpenAI-compatible
endpoint. This enables testing our OpenAI provider against various models:

```python
model = OpenAILanguageModel(
    model_id="google/gemini-2.5-flash",  # or "openai/gpt-4o-mini", etc.
    api_key=os.environ["OPENROUTER_API_KEY"],
    base_url="https://openrouter.ai/api/v1",
)
```

**Setup**: Sign up at openrouter.ai, get API key, set `OPENROUTER_API_KEY` env var.

### Test Data

Use real documents — not synthetic strings. Maintain a `tests/fixtures/` directory with:

- `sample.pdf` — a short multi-page PDF with mixed native text + scanned pages
- `medical_note.txt` — canonical medical text (medication extraction domain)
- `invoice.pdf` — structured document for table/field extraction
- `multilingual.txt` — mixed-script text for tokenizer tests

---

## Item 1: Test Foundation (Hybrid Port + New)

### 1A: Port Critical Upstream Tests

Port these 6 upstream test files, adapting from absl-testing to work with our codebase:

| File | Methods | What It Covers | Priority |
|------|---------|----------------|----------|
| `resolver_test.py` | 26 | YAML/JSON parsing, alignment, fence handling | P0 |
| `annotation_test.py` | 17 | End-to-end annotation pipeline with mocked LLM | P0 |
| `tokenizer_test.py` | 38 | Unicode + ASCII tokenization, sentence boundaries | P0 |
| `inference_test.py` | 29 | All 3 provider infer() implementations | P0 |
| `init_test.py` | 14 | Public lx.extract() API wiring | P1 |
| `chunking_test.py` | 16 | Text chunking + sentence splitting | P1 |

**Adaptation strategy**:

- Keep `absl-testing` as a dev dep (already in pyproject.toml) for the ported tests
- Update import paths (`langextract.core.data` vs `langextract.data`, etc.)
- Fix any Python 3.14 incompatibilities
- Replace `requests.post` mocks with `httpx.post` mocks for Ollama
- Update Gemini mocks to match our genai SDK version

**Source**: `git show upstream/main:tests/<filename>` for each file.

### 1B: Write New pytest Tests

New tests for fork-specific features, using pytest natively:

| File | Target Module | Test Type |
|------|---------------|-----------|
| `tests/test_format_handler.py` | `core/format_handler.py` | Unit — think-tag stripping, fence parsing, JSON/YAML roundtrip |
| `tests/test_settings.py` | `settings.py` | Unit — API key resolution, env var precedence |
| `tests/test_pdfextract.py` | `pdfextract/` | Integration — OCR backend init, native extraction, quality scoring |
| `tests/test_pdfextract_ocr.py` | `pdfextract/ocr_backends.py` | Live — actual OCR on test images via RapidOCR |
| `tests/test_gemini_batch.py` | `providers/gemini_batch.py` | Unit — batch config, routing logic |
| `tests/test_openai_reasoning.py` | `providers/openai.py` | Unit — `_normalize_reasoning_params` |
| `tests/test_prompt_validation.py` | `prompt_validation.py` | Unit — alignment policies, validation levels |
| `tests/test_extract_pdf.py` | `extraction.py` (extract_pdf) | Integration — full PDF-to-extraction pipeline |

### 1C: Live API Test Suite

End-to-end tests against real endpoints:

| File | Endpoint | What It Tests |
|------|----------|---------------|
| `tests/live/test_ollama_live.py` | home-mac Ollama | Full extraction with qwen3.5:4b |
| `tests/live/test_openrouter_live.py` | OpenRouter | Extraction with gemini-flash, gpt-4o-mini |
| `tests/live/test_gemini_live.py` | Gemini API | Extraction with gemini-2.5-flash-lite |
| `tests/live/test_ocr_vision_live.py` | home-mac Ollama | glm-ocr vision OCR on test PDF pages |
| `tests/live/test_extract_pdf_live.py` | home-mac Ollama | Full PDF → OCR → LLM → extraction pipeline |

All live tests use `@pytest.mark.live_api` and skip gracefully if the endpoint is unreachable.

### Pass/Fail Criteria

- All ported tests pass on Python 3.14
- New unit tests achieve >80% line coverage on target modules
- Live tests pass when endpoints are available, skip cleanly when not
- `uv run pytest -m "not live_api"` passes in CI with zero failures

---

## Item 2: Wire pdfextract into Wheel + `lx.extract_pdf()`

### Problem

1. `pdfextract/` is excluded from the wheel (`pyproject.toml:65` only lists `langextract`)
2. No convenience bridge — users must manually call `pdfextract.extract_document()` then pass
   `.full_text` to `lx.extract()`
3. The `word_map` (page coordinate projection) is unused by the extraction pipeline

### Changes

**pyproject.toml**:

```toml
packages = ["langextract", "pdfextract"]
```

**New function** in `langextract/extraction.py`:

```python
def extract_pdf(
    path: str | pathlib.Path,
    *,
    ocr_backend: str = "rapidocr",
    dpi: int = 300,
    # All lx.extract() params forwarded
    prompt_description: str,
    examples: list[data.ExampleData],
    model_id: str | None = None,
    model: base_model.BaseLanguageModel | None = None,
    **kwargs,
) -> AnnotatedDocument:
```

**Data flow**:

```text
lx.extract_pdf("invoice.pdf", prompt_description="...", examples=[...])
  → pdfextract.extract_document(path, ocr_backend, dpi)
  → DocumentResult(full_text, pages, word_map)
  → lx.extract(full_text, prompt_description=..., examples=...)
  → AnnotatedDocument with word_map attached
```

**AnnotatedDocument enrichment**: Add optional `word_map` field. When present,
`doc.bbox_for_extraction(extraction)` returns the PDF page bounding box for any extraction
via char_interval → word_map projection.

### Expose in `__init__.py`

Add `extract_pdf` to lazy-loaded public API: `lx.extract_pdf(...)`.

### Tests

| Test | Type | What |
|------|------|------|
| Native PDF extraction | Integration | PDF with digital text → correct full_text |
| Scanned PDF + OCR | Live | Scanned PDF → OCR → text (needs RapidOCR) |
| Full pipeline | Live | PDF → OCR → LLM extraction → AnnotatedDocument with word_map |
| Empty PDF | Unit | Graceful error on empty/corrupt PDF |
| Bbox projection | Unit | char_interval → word_map → page coordinates correct |

---

## Item 3: OpenAI Structured Outputs (Schema Constraints)

### Problem

When `use_schema_constraints=True`:

- **Gemini**: Uses `GeminiSchema` → field-level JSON schema constraints → model forced to conform
- **Ollama**: Uses `FormatModeSchema` → `format: "json"` → JSON mode only
- **OpenAI**: Falls back to `Constraint.NONE` → only `response_format: {"type": "json_object"}`

OpenAI's Structured Outputs API supports `response_format: {"type": "json_schema", "json_schema": {...}}`
which provides the same field-level constraint enforcement as Gemini. Not implemented.

### Changes

**New file**: `langextract/providers/schemas/openai.py`

```python
class OpenAISchema(BaseSchema):
    """JSON Schema constraints for OpenAI Structured Outputs."""

    @classmethod
    def from_examples(cls, examples: list[data.ExampleData], ...) -> OpenAISchema:
        # Derive JSON Schema from extraction classes + attributes
        ...

    def to_provider_config(self) -> dict:
        return {
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "extraction_output",
                    "strict": True,
                    "schema": self._schema,
                },
            },
        }

    @property
    def requires_raw_output(self) -> bool:
        return True  # No fences needed with structured output
```

**Update** `langextract/providers/openai.py`:

- Override `get_schema_class()` → return `OpenAISchema`
- `_process_single_prompt` → use schema's `response_format` when available

**Update** `langextract/providers/schemas/__init__.py`:

- Export `OpenAISchema`

### Tests

| Test | Type | What |
|------|------|------|
| Schema derivation | Unit | from_examples() → correct JSON schema |
| API params | Unit | to_provider_config() in API call params |
| Live extraction | Live (OpenRouter) | Extraction with schema vs without — compare quality |
| Fallback | Unit | Schema disabled → falls back to json_object mode |

---

## Item 4: Dead Code Cleanup

### Removals

| File | What | Ref |
|------|------|-----|
| `langextract/inference.py` | Dead re-export module. No internal callers. `InferenceType` enum used nowhere. | Full file |
| `langextract/registry.py` | Dead `__getattr__` redirect to `plugins.py`. No callers. | Full file |
| `langextract/core/data.py:44` | `AlignmentStatus.MATCH_GREATER` — never assigned by any code path | 1 line |
| `langextract/progress.py:255-282` | `format_extraction_stats()` + `create_extraction_postfix()` — duplicate, uncalled | 28 lines |

### Updates

- `langextract/__init__.py` — remove `"inference"` and `"registry"` from `_LAZY_MODULES` and `__all__`
- Verify: `uv run python -c "import langextract; dir(langextract)"` still works

### Tests

Import smoke test: verify `import langextract` and all public API functions still work after cleanup.

---

## Item 5: Streaming Extraction — `lx.extract_iter()`

### Problem

`lx.extract()` calls `list(result)` at `extraction.py:327`, materializing all documents before
returning. For large corpora, the caller can't process results incrementally.

Additionally, `annotation.py:393` calls `list(outputs)` on the infer iterator, preventing
per-prompt streaming within a batch.

### Changes

**New function**: `extract_iter()` in `extraction.py` — returns the generator directly.

```python
def extract_iter(
    text_or_documents,
    **kwargs,
) -> Iterator[AnnotatedDocument]:
    """Like extract(), but yields documents lazily as they complete."""
    ...
    yield from annotator.annotate_documents(...)
```

**Expose**: `lx.extract_iter(...)` in `__init__.py`.

`lx.extract()` stays unchanged (returns `list`) for backward compat.

### Tests

| Test | Type | What |
|------|------|------|
| Lazy yield | Unit | First doc yields before generator exhausted |
| Equivalence | Unit | `list(lx.extract_iter(...))` == `lx.extract(...)` |
| Large batch | Integration | 100-doc extraction streams without OOM |

---

## Item 6: Visualization Multi-Document Support

### Problem

`visualization.py:581-585` loads all documents from JSONL but silently renders only `documents[0]`.

### Changes

```python
def visualize(
    data_source: ...,
    *,
    document_index: int = 0,  # NEW
    animation_speed: float = 1.0,
    show_legend: bool = True,
    gif_optimized: bool = True,
) -> object | str:
```

- Bounds check on `document_index`
- `UserWarning` when JSONL has multiple docs and `document_index` not explicitly set

### Tests

| Test | Type | What |
|------|------|------|
| Index selection | Unit | Renders correct doc by index |
| Warning | Unit | Warning emitted for multi-doc JSONL |
| Out of bounds | Unit | Clear error for invalid index |

---

## Item 7: PEP 561 `py.typed` Marker

### Changes

- Create `langextract/py.typed` (empty file)
- Add to `pyproject.toml` package data if needed

### Tests

`ty` type-check pass on the package.

---

## Item 8: Fuzzy Alignment Performance

### Problem

`WordAligner` fuzzy path in `resolver.py` is O(n·m²) worst case — sliding window over source
text for each unaligned extraction across multiple window sizes.

### Approach

1. **Profile first** — benchmark with a 10K-char document and 50 extractions
2. **If slow**: add ngram index for candidate window selection (sublinear lookup)
3. **If acceptable**: document the complexity and add a regression benchmark

### Tests

Benchmark test: 10K chars, 50 extractions, assert wall-clock < 2s.

---

## Speculative Items (Future)

### S1: Vision-Model Direct Extraction from PDF Pages

**Idea**: Instead of OCR → text → LLM extraction, send the rendered page image directly to
a vision-capable LLM and extract in one step. For complex layouts (tables, forms, diagrams),
this would be far more accurate.

**Architecture**: `pdfextract` already has `OllamaOcrBackend` and `VllmOcrBackend` that send
page images to vision models. The extension would be a new extraction mode:

```python
lx.extract_pdf("invoice.pdf", mode="vision", model_id="glm-ocr")
```

Instead of converting to text first, each page image goes directly to the LLM with the
extraction prompt. The `glm-ocr` model on home-mac Ollama (`100.113.195.95:11434`) already
accepts images and could serve this role.

**Challenges**: Token limits per page, multi-page documents need page-level chunking instead
of text-level chunking, word_map/char_interval semantics change (page coordinates instead of
text offsets).

**Test endpoint**: home-mac Ollama with `glm-ocr:latest`.

### S2: Cross-Document Extraction

**Idea**: Extract entities/relationships that span a set of related documents (contract +
amendments, medical record across visits, etc.).

**Architecture**: A document-set mode where the prompt can reference entities resolved in
previous documents. Extends the cross-chunk coreference approach to cross-document.

```python
docs = [Document(text=t, document_id=f"page_{i}") for i, t in enumerate(texts)]
results = lx.extract(docs, context_window_docs=2, ...)
```

**Challenges**: Context window limits, entity resolution across documents, ordering semantics.

### S3: OpenRouter as First-Class Provider

**Idea**: Register OpenRouter patterns in the router so `model_id="openrouter/google/gemini-2.5-flash"`
auto-routes to the OpenRouter endpoint. This would let users access any model via a single API key.

**Architecture**: New `OpenRouterLanguageModel` that wraps `OpenAILanguageModel` with
`base_url="https://openrouter.ai/api/v1"` and model ID normalization.

### S4: Ollama Schema Constraints

**Idea**: Ollama supports JSON schema in the `format` parameter. Implement `OllamaSchema`
similar to `OpenAISchema` that derives a schema from examples and passes it as
`format: {"type": "object", "properties": {...}}`.

**Architecture**: New `OllamaSchema(BaseSchema)` in `providers/schemas/ollama.py`.

---

## Implementation Order

| Phase | Items | Estimated Effort |
|-------|-------|-----------------|
| **Phase 1** | Item 4 (dead code cleanup) + Item 7 (py.typed) | Small — cleanup |
| **Phase 2** | Item 1A (port upstream tests) | Medium — adapt 6 test files |
| **Phase 3** | Item 2 (pdfextract integration) + Item 1B (new tests) | Large — new feature + tests |
| **Phase 4** | Item 3 (OpenAI schema) | Medium — new schema + tests |
| **Phase 5** | Item 5 (streaming) + Item 6 (viz multi-doc) | Small — incremental |
| **Phase 6** | Item 8 (alignment perf) | Small — profile-driven |
| **Phase 7** | Item 1C (live API tests) + OpenRouter setup | Medium — endpoint integration |
| **Speculative** | S1–S4 | Future phases |
