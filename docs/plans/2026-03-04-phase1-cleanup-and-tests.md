# Phase 1–2: Dead Code Cleanup + Test Foundation

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Remove dead code, add PEP 561 marker, then establish a test suite by porting 6 critical upstream test files and writing new tests for fork-specific features.

**Architecture:** Phase 1 is pure cleanup — delete dead modules and unused symbols, update `__init__.py`. Phase 2 ports upstream tests (absl-testing based, adapted for our import paths and Python 3.14) and adds new pytest tests for `format_handler`, `settings`, `prompt_validation`, and the `openai` reasoning normalization. Live API tests use real Ollama on home-mac (`100.113.195.95:11434`) and OpenRouter.

**Tech Stack:** pytest, absl-py (for ported tests), unittest.mock (for provider mocking), httpx (replacing requests mocks), ruff (linting)

---

## Task 1: Delete `langextract/inference.py`

**Files:**

- Delete: `langextract/inference.py`
- Modify: `langextract/__init__.py:37,71`

**Step 1: Verify no internal callers**

Run: `grep -r "from langextract.inference" langextract/ && grep -r "import langextract.inference" langextract/`
Expected: No output (zero callers confirmed)

**Step 2: Delete the file**

```bash
rm langextract/inference.py
```

**Step 3: Remove from `__init__.py`**

In `langextract/__init__.py`:

- Remove `"inference"` from the `__all__` list (line 37)
- Remove `"inference": "langextract.inference"` from `_LAZY_MODULES` (line 71)

**Step 4: Verify import still works**

Run: `uv run python -c "import langextract; print(langextract.extract)"`
Expected: `<function extract at 0x...>`

Run: `uv run python -c "from langextract.core.base_model import BaseLanguageModel; print(BaseLanguageModel)"`
Expected: `<class 'langextract.core.base_model.BaseLanguageModel'>` — the canonical import path still works.

**Step 5: Commit**

```bash
git add -u langextract/inference.py langextract/__init__.py
git commit -m "chore: remove dead inference.py re-export module"
```

---

## Task 2: Delete `langextract/registry.py`

**Files:**

- Delete: `langextract/registry.py`
- Modify: `langextract/__init__.py:82`

**Step 1: Verify no internal callers**

Run: `grep -r "from langextract.registry" langextract/ && grep -r "langextract.registry" langextract/`
Expected: Only the lazy-module entry in `__init__.py`

**Step 2: Delete and update**

```bash
rm langextract/registry.py
```

In `langextract/__init__.py`:

- Remove `"registry": "langextract.registry"` from `_LAZY_MODULES` (line 82)
- Do NOT remove `"plugins"` — that's the real module

**Step 3: Verify**

Run: `uv run python -c "import langextract; print(langextract.plugins)"`
Expected: `<module 'langextract.plugins' ...>`

**Step 4: Commit**

```bash
git add -u langextract/registry.py langextract/__init__.py
git commit -m "chore: remove dead registry.py redirect module"
```

---

## Task 3: Remove `MATCH_GREATER` and dead progress functions

**Files:**

- Modify: `langextract/core/data.py:44`
- Modify: `langextract/progress.py:255-282`

**Step 1: Remove `MATCH_GREATER` from `AlignmentStatus`**

In `langextract/core/data.py`, delete line 44:

```python
    MATCH_GREATER = "match_greater"
```

**Step 2: Verify no callers**

Run: `grep -r "MATCH_GREATER" langextract/`
Expected: No output

**Step 3: Remove duplicate dead functions from `progress.py`**

Delete `format_extraction_stats` (lines 255-267) and `create_extraction_postfix` (lines 270-282) from `langextract/progress.py`.

**Step 4: Verify no callers**

Run: `grep -r "format_extraction_stats\|create_extraction_postfix" langextract/`
Expected: No output

**Step 5: Verify import**

Run: `uv run python -c "from langextract.core.data import AlignmentStatus; print(list(AlignmentStatus))"`
Expected: `[<AlignmentStatus.MATCH_EXACT: 'match_exact'>, <AlignmentStatus.MATCH_LESSER: 'match_lesser'>, <AlignmentStatus.MATCH_FUZZY: 'match_fuzzy'>]`

**Step 6: Commit**

```bash
git add langextract/core/data.py langextract/progress.py
git commit -m "chore: remove unused MATCH_GREATER enum + dead progress functions"
```

---

## Task 4: Add `py.typed` PEP 561 marker

**Files:**

- Create: `langextract/py.typed`

**Step 1: Create the marker**

```bash
touch langextract/py.typed
```

This is an empty file that tells type checkers (ty, mypy, pyright) the package ships type information.

**Step 2: Verify ty recognizes it**

Run: `uv run ty check langextract/__init__.py`
Expected: Should run without "package is not typed" warnings.

**Step 3: Commit**

```bash
git add langextract/py.typed
git commit -m "chore: add PEP 561 py.typed marker"
```

---

## Task 5: Create test infrastructure

**Files:**

- Create: `tests/__init__.py`
- Create: `tests/conftest.py`
- Create: `tests/fixtures/medical_note.txt`

**Step 1: Create tests directory and conftest**

`tests/__init__.py` — empty file.

`tests/conftest.py`:

```python
"""Shared test fixtures for langextract tests."""

import os
import pathlib

import pytest

# Test data directory
FIXTURES_DIR = pathlib.Path(__file__).parent / "fixtures"


@pytest.fixture
def medical_text():
    """Canonical medical text for extraction tests."""
    return (FIXTURES_DIR / "medical_note.txt").read_text()


@pytest.fixture
def sample_examples():
    """Standard few-shot examples for medication extraction."""
    from langextract.core.data import ExampleData, Extraction

    return [
        ExampleData(
            text=(
                "The patient was prescribed Lisinopril 10mg daily for hypertension "
                "and Metformin 500mg twice daily for type 2 diabetes."
            ),
            extractions=[
                Extraction(
                    extraction_class="medication",
                    extraction_text="Lisinopril 10mg",
                    attributes={"name": "Lisinopril", "dosage": "10mg", "frequency": "daily"},
                ),
                Extraction(
                    extraction_class="medication",
                    extraction_text="Metformin 500mg",
                    attributes={"name": "Metformin", "dosage": "500mg", "frequency": "twice daily"},
                ),
            ],
        )
    ]


# -- Endpoint availability helpers --

def _check_url(url: str, timeout: float = 2.0) -> bool:
    """Check if a URL is reachable."""
    try:
        import httpx
        resp = httpx.get(url, timeout=timeout)
        return resp.status_code == 200
    except Exception:
        return False


@pytest.fixture(scope="session")
def ollama_available():
    """Check if Ollama is available on home-mac via Tailscale."""
    url = os.environ.get("OLLAMA_BASE_URL", "http://100.113.195.95:11434")
    return _check_url(f"{url}/api/tags")


@pytest.fixture(scope="session")
def openrouter_key():
    """Get OpenRouter API key if available."""
    return os.environ.get("OPENROUTER_API_KEY")


@pytest.fixture(scope="session")
def gemini_key():
    """Get Gemini API key if available."""
    return os.environ.get("GEMINI_API_KEY")
```

`tests/fixtures/medical_note.txt`:

```text
Patient: John Smith, DOB: 03/15/1965, MRN: 123456

Chief Complaint: Follow-up for hypertension and type 2 diabetes mellitus.

History of Present Illness:
Mr. Smith is a 60-year-old male presenting for routine follow-up. He reports good compliance
with his medication regimen. Blood pressure has been well-controlled at home readings of
125/80 mmHg. He denies any episodes of hypoglycemia. His last HbA1c was 7.2%.

Current Medications:
1. Lisinopril 20mg daily for hypertension
2. Metformin 1000mg twice daily for type 2 diabetes
3. Atorvastatin 40mg at bedtime for hyperlipidemia
4. Aspirin 81mg daily for cardiovascular prophylaxis

Assessment and Plan:
- Hypertension: Well-controlled on current regimen. Continue Lisinopril 20mg daily.
- Type 2 Diabetes: HbA1c at goal. Continue Metformin 1000mg BID.
- Hyperlipidemia: Continue Atorvastatin 40mg QHS. Recheck lipid panel in 3 months.
- Follow-up in 3 months.

Signed: Dr. Jane Wilson, MD
```

**Step 2: Verify pytest discovers the directory**

Run: `uv run pytest tests/ --collect-only`
Expected: Should collect 0 tests (no test files yet) but not error.

**Step 3: Commit**

```bash
git add tests/
git commit -m "test: add test infrastructure (conftest, fixtures, markers)"
```

---

## Task 6: Port `resolver_test.py` from upstream

This is the most critical test file — 2407 lines testing YAML/JSON parsing, alignment, and fence handling.

**Files:**

- Create: `tests/resolver_test.py`

**Step 1: Extract the upstream file**

```bash
git show upstream/main:tests/resolver_test.py > tests/resolver_test.py
```

**Step 2: Adapt imports for our codebase**

The upstream file uses these imports that need updating:

```python
# Upstream uses:
from langextract import resolver as resolver_lib
from langextract import chunking
from langextract.core import data
from langextract.core import tokenizer
```

These should work as-is in our fork (we have the same module paths). However, check for:

- Any reference to `langextract.core.format_handler.FormatHandler` — our fork centralizes this in `core/`, upstream distributes it. Tests should reference `from langextract.core import format_handler`.
- `from absl.testing import absltest` and `from absl.testing import parameterized` — keep these, `absl-py` is in our dev deps.

**Step 3: Run and fix iteratively**

Run: `uv run pytest tests/resolver_test.py -v --tb=short -x 2>&1 | head -80`

Fix import errors one at a time. Common fixes:

- `resolver_lib.Resolver` constructor signature may differ (check `resolver.py` for current `__init__` params)
- `data.FormatType` may be in `core.types` now (check `from langextract.core.types import FormatType`)
- `tokenizer.RegexTokenizer` vs `tokenizer.Tokenizer`

Keep fixing until all 26 tests pass.

**Step 4: Commit**

```bash
git add tests/resolver_test.py
git commit -m "test: port resolver_test.py from upstream (26 tests)"
```

---

## Task 7: Port `tokenizer_test.py` from upstream

**Files:**

- Create: `tests/tokenizer_test.py`

**Step 1: Extract and adapt**

```bash
git show upstream/main:tests/tokenizer_test.py > tests/tokenizer_test.py
```

**Step 2: Run and fix**

Run: `uv run pytest tests/tokenizer_test.py -v --tb=short -x 2>&1 | head -80`

The tokenizer tests should be the most stable since our `core/tokenizer.py` is close to upstream. Fix any import path issues.

**Step 3: All 38 tests pass**

Run: `uv run pytest tests/tokenizer_test.py -v`
Expected: 38 passed

**Step 4: Commit**

```bash
git add tests/tokenizer_test.py
git commit -m "test: port tokenizer_test.py from upstream (38 tests)"
```

---

## Task 8: Port `annotation_test.py` from upstream

**Files:**

- Create: `tests/annotation_test.py`

**Step 1: Extract and adapt**

```bash
git show upstream/main:tests/annotation_test.py > tests/annotation_test.py
```

**Step 2: Run and fix**

Run: `uv run pytest tests/annotation_test.py -v --tb=short -x 2>&1 | head -80`

Key adaptation points:

- The annotation tests mock the language model — check that `mock.patch.object` targets match our class paths
- `ScoredOutput` is in `langextract.core.types` (our fork) vs possibly `langextract.inference` (upstream)
- `show_progress` parameter may exist in our fork but not upstream's test expectations

**Step 3: All 17 tests pass**

**Step 4: Commit**

```bash
git add tests/annotation_test.py
git commit -m "test: port annotation_test.py from upstream (17 tests)"
```

---

## Task 9: Port `inference_test.py` from upstream

**Files:**

- Create: `tests/inference_test.py`

**Step 1: Extract and adapt**

```bash
git show upstream/main:tests/inference_test.py > tests/inference_test.py
```

**Step 2: Key adaptations**

- Ollama tests: replace `mock.patch("requests.post")` with `mock.patch("httpx.post")` — our fork uses httpx
- Ollama: `structured_output_format` parameter renamed to `format` in our fork
- Ollama: `model_url` parameter renamed to `base_url` in our fork
- Gemini: our fork has `ENV_API_KEY_NAMES` class attr, upstream may not
- OpenAI: our fork has `_normalize_reasoning_params` (add tests for that too)

**Step 3: Run and fix iteratively**

Run: `uv run pytest tests/inference_test.py -v --tb=short -x 2>&1 | head -80`

**Step 4: All 29 tests pass**

**Step 5: Commit**

```bash
git add tests/inference_test.py
git commit -m "test: port inference_test.py from upstream (29 tests)"
```

---

## Task 10: Port `init_test.py` from upstream

**Files:**

- Create: `tests/init_test.py`

**Step 1: Extract and adapt**

```bash
git show upstream/main:tests/init_test.py > tests/init_test.py
```

**Step 2: Key adaptations**

- Remove tests for `inference` and `registry` module access (we deleted those)
- Update default model ID expectations (`gemini-2.5-flash-lite` in our fork)
- `show_progress` parameter exists in our fork
- Schema validation warnings may differ

**Step 3: Run and fix**

Run: `uv run pytest tests/init_test.py -v --tb=short -x 2>&1 | head -80`

**Step 4: Commit**

```bash
git add tests/init_test.py
git commit -m "test: port init_test.py from upstream (14 tests)"
```

---

## Task 11: Port `chunking_test.py` from upstream

**Files:**

- Create: `tests/chunking_test.py`

**Step 1: Extract and adapt**

```bash
git show upstream/main:tests/chunking_test.py > tests/chunking_test.py
```

**Step 2: Run and fix**

Run: `uv run pytest tests/chunking_test.py -v --tb=short -x 2>&1 | head -80`

Chunking is core infrastructure and should be very stable across forks.

**Step 3: All 16 tests pass**

**Step 4: Commit**

```bash
git add tests/chunking_test.py
git commit -m "test: port chunking_test.py from upstream (16 tests)"
```

---

## Task 12: Write new pytest tests for `format_handler.py`

**Files:**

- Create: `tests/test_format_handler.py`

**Step 1: Write the tests**

```python
"""Tests for langextract.core.format_handler — think-tag stripping, fence parsing, roundtrip."""

import json

import pytest
import yaml

from langextract.core import data, format_handler


class TestFormatHandlerInit:
    def test_defaults(self):
        fh = format_handler.FormatHandler()
        assert fh.format_type == data.FormatType.JSON
        assert fh.use_wrapper is True
        assert fh.use_fences is True
        assert fh.allow_top_level_list is True

    def test_repr(self):
        fh = format_handler.FormatHandler()
        r = repr(fh)
        assert "FormatHandler(" in r
        assert "format_type=" in r


class TestThinkTagStripping:
    """Tests that <think>...</think> blocks from reasoning models are stripped."""

    def test_strips_think_tags_from_json(self):
        fh = format_handler.FormatHandler(use_fences=False, use_wrapper=False)
        text = '<think>Let me analyze this...</think>\n[{"medication": "Aspirin"}]'
        result = fh.parse_output(text)
        assert len(result) == 1
        assert result[0]["medication"] == "Aspirin"

    def test_strips_multiline_think_tags(self):
        fh = format_handler.FormatHandler(use_fences=False, use_wrapper=False)
        text = (
            "<think>\nI need to extract medications.\n"
            "The text mentions Aspirin.\n</think>\n"
            '[{"medication": "Aspirin"}]'
        )
        result = fh.parse_output(text)
        assert len(result) == 1

    def test_no_think_tags_works_normally(self):
        fh = format_handler.FormatHandler(use_fences=False, use_wrapper=False)
        text = '[{"medication": "Aspirin"}]'
        result = fh.parse_output(text)
        assert len(result) == 1

    def test_strict_mode_does_not_strip_think_tags(self):
        fh = format_handler.FormatHandler(use_fences=False, use_wrapper=False)
        text = '<think>analysis</think>\n[{"medication": "Aspirin"}]'
        with pytest.raises(Exception):
            fh.parse_output(text, strict=True)


class TestTopLevelListAcceptance:
    """Tests that top-level lists are accepted when allow_top_level_list=True."""

    def test_accepts_bare_list(self):
        fh = format_handler.FormatHandler(
            use_fences=False, use_wrapper=True, wrapper_key="extractions",
            allow_top_level_list=True,
        )
        text = '[{"medication": "Aspirin"}]'
        result = fh.parse_output(text)
        assert len(result) == 1

    def test_rejects_bare_list_when_disabled(self):
        fh = format_handler.FormatHandler(
            use_fences=False, use_wrapper=True, wrapper_key="extractions",
            allow_top_level_list=False,
        )
        text = '[{"medication": "Aspirin"}]'
        with pytest.raises(Exception):
            fh.parse_output(text)


class TestJsonYamlRoundtrip:
    """Tests that format → parse roundtrips correctly."""

    def test_json_roundtrip(self):
        fh = format_handler.FormatHandler(
            format_type=data.FormatType.JSON, use_wrapper=True,
            wrapper_key="extractions", use_fences=True,
        )
        extractions = [
            data.Extraction(
                extraction_class="medication",
                extraction_text="Aspirin 81mg",
                attributes={"name": "Aspirin", "dosage": "81mg"},
            )
        ]
        formatted = fh.format_extraction_example(extractions)
        result = fh.parse_output(formatted)
        assert len(result) == 1
        assert result[0]["medication"] == "Aspirin 81mg"

    def test_yaml_roundtrip(self):
        fh = format_handler.FormatHandler(
            format_type=data.FormatType.YAML, use_wrapper=True,
            wrapper_key="extractions", use_fences=True,
        )
        extractions = [
            data.Extraction(
                extraction_class="medication",
                extraction_text="Aspirin 81mg",
                attributes={"name": "Aspirin", "dosage": "81mg"},
            )
        ]
        formatted = fh.format_extraction_example(extractions)
        result = fh.parse_output(formatted)
        assert len(result) == 1
        assert result[0]["medication"] == "Aspirin 81mg"


class TestFenceParsing:
    def test_json_fence(self):
        fh = format_handler.FormatHandler(use_fences=True, use_wrapper=False)
        text = '```json\n[{"a": 1}]\n```'
        result = fh.parse_output(text)
        assert result[0]["a"] == 1

    def test_no_fence_fallback(self):
        fh = format_handler.FormatHandler(use_fences=True, use_wrapper=False, strict_fences=False)
        text = '[{"a": 1}]'
        result = fh.parse_output(text)
        assert result[0]["a"] == 1

    def test_multiple_fences_rejected(self):
        fh = format_handler.FormatHandler(use_fences=True, use_wrapper=False)
        text = '```json\n[{"a": 1}]\n```\n```json\n[{"b": 2}]\n```'
        with pytest.raises(Exception):
            fh.parse_output(text)

    def test_empty_input_raises(self):
        fh = format_handler.FormatHandler()
        with pytest.raises(Exception):
            fh.parse_output("")
```

**Step 2: Run tests**

Run: `uv run pytest tests/test_format_handler.py -v`
Expected: All pass

**Step 3: Commit**

```bash
git add tests/test_format_handler.py
git commit -m "test: add format_handler tests (think-tag stripping, fences, roundtrip)"
```

---

## Task 13: Write new pytest tests for OpenAI reasoning normalization

**Files:**

- Create: `tests/test_openai_reasoning.py`

**Step 1: Write the tests**

```python
"""Tests for OpenAI reasoning_effort normalization."""

from unittest import mock

import pytest

from langextract.core import data, exceptions, schema
from langextract.core import types as core_types


class TestNormalizeReasoningParams:
    """Test _normalize_reasoning_params on OpenAILanguageModel."""

    def _make_model(self):
        with mock.patch("openai.OpenAI"):
            from langextract.providers.openai import OpenAILanguageModel
            return OpenAILanguageModel(model_id="gpt-4o-mini", api_key="test-key")

    def test_flat_reasoning_effort_converted(self):
        model = self._make_model()
        result = model._normalize_reasoning_params({"reasoning_effort": "minimal"})
        assert "reasoning_effort" not in result
        assert result["reasoning"] == {"effort": "minimal"}

    def test_existing_reasoning_dict_merged(self):
        model = self._make_model()
        result = model._normalize_reasoning_params({
            "reasoning_effort": "minimal",
            "reasoning": {"summary": "auto"},
        })
        assert result["reasoning"] == {"effort": "minimal", "summary": "auto"}

    def test_no_reasoning_effort_passthrough(self):
        model = self._make_model()
        result = model._normalize_reasoning_params({"temperature": 0.5})
        assert result == {"temperature": 0.5}

    def test_existing_effort_not_overwritten(self):
        model = self._make_model()
        result = model._normalize_reasoning_params({
            "reasoning_effort": "low",
            "reasoning": {"effort": "high"},
        })
        # setdefault means existing "effort" key is preserved
        assert result["reasoning"]["effort"] == "high"
```

**Step 2: Run tests**

Run: `uv run pytest tests/test_openai_reasoning.py -v`
Expected: All pass

**Step 3: Commit**

```bash
git add tests/test_openai_reasoning.py
git commit -m "test: add OpenAI reasoning_effort normalization tests"
```

---

## Task 14: Write new pytest tests for `prompt_validation.py`

**Files:**

- Create: `tests/test_prompt_validation.py`

**Step 1: Read current prompt_validation module**

Check: `langextract/prompt_validation.py` — understand the public API (`validate_examples`, `AlignmentPolicy`, `PromptValidationLevel`, etc.)

**Step 2: Write tests covering**

- Exact alignment passes validation
- Fuzzy alignment triggers warning at appropriate level
- No-match extraction raises at error level
- Empty examples list passes
- Validation doesn't mutate input examples
- Different `AlignmentPolicy` configurations

**Step 3: Run and verify all pass**

Run: `uv run pytest tests/test_prompt_validation.py -v`

**Step 4: Commit**

```bash
git add tests/test_prompt_validation.py
git commit -m "test: add prompt_validation tests"
```

---

## Task 15: Full test suite verification + ruff check

**Step 1: Run the complete test suite**

Run: `uv run pytest tests/ -v --tb=short -m "not live_api"`
Expected: All tests pass

**Step 2: Run ruff**

Run: `uv run ruff check tests/`
Expected: Clean (or fix any issues)

**Step 3: Count coverage**

Run: `uv run pytest tests/ --co -q | tail -5`
Report: total test count

**Step 4: Final commit if any fixes needed**

```bash
git add -A tests/
git commit -m "test: fix lint issues in test suite"
```

---

## Success Criteria

| Metric | Target |
|--------|--------|
| Ported upstream tests passing | 140+ (from 6 files) |
| New pytest tests passing | 20+ (format_handler, openai, prompt_validation) |
| Dead code removed | 4 items (inference.py, registry.py, MATCH_GREATER, progress dupes) |
| `py.typed` marker | Present |
| `uv run pytest -m "not live_api"` | 0 failures |
| `uv run ruff check langextract/ tests/` | 0 errors |
