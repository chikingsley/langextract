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

"""Gemini Batch API helper module for LangExtract.

This module provides batch inference support using the google-genai SDK's
inline batch API. It handles:
- Inline batch submission (no GCS required)
- Job polling and result extraction
- Schema-based structured output
- Order preservation across batch processing
"""

from __future__ import annotations

import dataclasses
import logging
import time
from typing import TYPE_CHECKING, Any, Protocol

from google import genai

from langextract.core import exceptions

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

_UNSET = object()


@dataclasses.dataclass(slots=True, frozen=True)
class BatchConfig:
    """Define and validate Gemini Batch API configuration.

    Attributes:
      enabled: Whether batch mode is enabled.
      threshold: Minimum prompts to trigger batch processing.
      poll_interval: Seconds between job status checks.
      timeout: Maximum seconds to wait for job completion.
      max_prompts_per_job: Max prompts allowed in one batch job.
      ignore_item_errors: If True, continue on per-item errors.
    """

    enabled: bool = False
    threshold: int = 50
    poll_interval: int = 30
    timeout: int = 3600
    max_prompts_per_job: int = 20000
    ignore_item_errors: bool = False
    on_job_create: Callable[[Any], None] | None = None

    def __post_init__(self):
        """Validate numeric knobs early."""
        validations = [
            (self.threshold >= 1, "batch.threshold must be >= 1"),
            (self.poll_interval > 0, "batch.poll_interval must be > 0"),
            (self.timeout > 0, "batch.timeout must be > 0"),
            (
                self.max_prompts_per_job > 0,
                "batch.max_prompts_per_job must be > 0",
            ),
        ]
        for is_valid, error_msg in validations:
            if not is_valid:
                raise ValueError(error_msg)

    @classmethod
    def from_dict(cls, d: dict | None) -> BatchConfig:
        """Create BatchConfig from dictionary, using defaults for missing keys."""
        if d is None:
            return cls()
        valid_keys = {f.name for f in dataclasses.fields(cls)}
        filtered_dict = {k: v for k, v in d.items() if k in valid_keys}

        unknown = sorted(set(d.keys()) - valid_keys)
        if unknown:
            logging.warning("Ignoring unknown batch config keys: %s", ", ".join(unknown))
        cfg = cls(**filtered_dict)
        if cfg.on_job_create is None:
            object.__setattr__(cfg, "on_job_create", _default_job_create_callback)
        return cfg


_TERMINAL_FAIL = frozenset(
    {
        genai.types.JobState.JOB_STATE_FAILED,
        genai.types.JobState.JOB_STATE_CANCELLED,
        genai.types.JobState.JOB_STATE_EXPIRED,
    }
)
_TERMINAL_OK = frozenset(
    {
        genai.types.JobState.JOB_STATE_SUCCEEDED,
        genai.types.JobState.JOB_STATE_PAUSED,
    }
)


def _default_job_create_callback(job: Any) -> None:
    """Default callback to log batch job details."""
    logging.info("Batch job created: %s (state: %s)", job.name, job.state)


def _build_inline_request(
    prompt: str,
    schema_dict: dict | None,
    gen_config: dict,
    system_instruction: str | None = None,
    safety_settings: Sequence[Any] | None = None,
) -> genai.types.InlinedRequest:
    """Build an InlinedRequest for batch submission.

    Args:
      prompt: The text prompt.
      schema_dict: Optional JSON schema for structured output.
      gen_config: Generation config parameters (temperature, top_p, etc).
      system_instruction: Optional system instruction.
      safety_settings: Optional safety settings.

    Returns:
      InlinedRequest ready for batch submission.
    """
    config_kwargs: dict[str, Any] = {}

    if gen_config:
        config_kwargs.update(gen_config)

    if schema_dict:
        config_kwargs["response_mime_type"] = "application/json"
        config_kwargs["response_schema"] = schema_dict

    if system_instruction:
        config_kwargs["system_instruction"] = system_instruction

    if safety_settings:
        config_kwargs["safety_settings"] = safety_settings

    return genai.types.InlinedRequest(
        contents=prompt,
        config=genai.types.GenerateContentConfig(**config_kwargs) if config_kwargs else None,
    )


def _submit_inline(
    client: genai.Client,
    model_id: str,
    requests: Sequence[genai.types.InlinedRequest],
    display: str,
) -> genai.types.BatchJob:
    """Submit an inline batch job.

    Args:
      client: google.genai.Client instance.
      model_id: Model identifier (e.g., "gemini-2.5-flash").
      requests: List of InlinedRequest objects.
      display: Display name for the batch job.

    Returns:
      BatchJob object that can be polled for completion.
    """
    return client.batches.create(
        model=model_id,
        src=list(requests),
        config=genai.types.CreateBatchJobConfig(display_name=display),
    )


class _TextResponse(Protocol):
    """Protocol for inline response objects with text attribute."""

    text: str


def _extract_text(
    resp: _TextResponse | dict[str, Any] | None,
) -> tuple[str | None, dict[str, int] | None]:
    """Extract text and usage from batch API response.

    Args:
      resp: Response object containing text.

    Returns:
      Tuple of (text string, usage dict) or (None, None) if invalid.
    """
    if resp is None:
        return None, None

    text = None
    usage = None

    if hasattr(resp, "text"):
        val = getattr(resp, "text", None)
        if isinstance(val, str):
            text = val

    if isinstance(resp, dict):
        text = _safe_get_nested(resp, "candidates", 0, "content", "parts", 0, "text")
        usage_meta = _safe_get_nested(resp, "usageMetadata")
        if isinstance(usage_meta, dict):
            usage = {
                "prompt_tokens": int(usage_meta.get("promptTokenCount", 0)),
                "completion_tokens": int(usage_meta.get("candidatesTokenCount", 0)),
                "total_tokens": int(usage_meta.get("totalTokenCount", 0)),
            }

    return (text if isinstance(text, str) else None), usage


def _safe_get_nested(data: dict, *keys) -> Any:
    """Safely traverse nested dictionaries/lists."""
    current = data
    for key in keys:
        if current is None:
            return None
        if isinstance(key, int):
            if not isinstance(current, list) or len(current) <= key:
                return None
            current = current[key]
        else:
            if not isinstance(current, dict):
                return None
            current = current.get(key)
    return current


def _poll_completion(
    client: genai.Client, job: genai.types.BatchJob, cfg: BatchConfig
) -> genai.types.BatchJob:
    """Poll batch job until completion or timeout.

    Args:
      client: google.genai.Client instance for polling job status.
      job: Batch job object returned from client.batches.create().
      cfg: Batch configuration including timeout and poll_interval.

    Returns:
      Completed batch job object.

    Raises:
      InferenceRuntimeError: If the job fails or times out.
    """
    start = time.time()
    name = job.name
    if not isinstance(name, str):
        raise exceptions.InferenceRuntimeError("Batch job missing name; cannot poll completion.")

    while True:
        job = client.batches.get(name=name)
        state = job.state

        if state in _TERMINAL_OK:
            return job

        if state in _TERMINAL_FAIL:
            error_details = job.error or "(no error details)"
            raise exceptions.InferenceRuntimeError(
                f"Batch job failed: state={state.name}, name={name}, error={error_details}"
            )

        if time.time() - start > cfg.timeout:
            try:
                client.batches.cancel(name=name)
            except Exception as e:
                logging.warning("Failed to cancel timed-out batch job %s: %s", name, e)
            raise exceptions.InferenceRuntimeError(
                f"Batch job timed out after {cfg.timeout}s: {name}"
            )

        time.sleep(cfg.poll_interval)
        logging.info("Batch job running... (state: %s)", state.name)


def _extract_inline_results(
    job: genai.types.BatchJob,
    cfg: BatchConfig,
    expected_count: int,
) -> list[tuple[str, dict[str, int] | None]]:
    """Extract text outputs from inline batch results, preserving order.

    Args:
      job: Completed batch job object.
      cfg: Batch configuration including error handling settings.
      expected_count: Number of prompts submitted.

    Returns:
      List of (text, usage) tuples corresponding 1:1 to input prompts.
    """
    outputs: list[tuple[str, dict[str, int] | None]] = []

    # Inline batch results are available on the job object directly
    # via job.dest or the response attribute, in submission order
    responses = getattr(job, "responses", None) or []

    for resp in responses:
        text, usage = _extract_text(resp)
        outputs.append((text or "", usage))

    # Pad if responses are fewer than expected
    while len(outputs) < expected_count:
        outputs.append(("", None))

    return outputs[:expected_count]


def infer_batch(
    client: genai.Client,
    model_id: str,
    prompts: Sequence[str],
    schema_dict: dict | None,
    gen_config: dict,
    cfg: BatchConfig,
    system_instruction: str | None = None,
    safety_settings: Sequence[Any] | None = None,
    **_kwargs: Any,
) -> list[tuple[str, dict[str, int] | None]]:
    """Execute batch inference on multiple prompts using the Gemini Batch API.

    Uses inline batch submission (no GCS/Vertex AI required).

    Args:
      client: google.genai.Client instance.
      model_id: Model identifier (e.g., "gemini-2.5-flash").
      prompts: Sequence of prompts to process in batch.
      schema_dict: Optional JSON schema for structured output.
      gen_config: Generation configuration parameters.
      cfg: Batch configuration including thresholds and timeouts.
      system_instruction: Optional system instruction text.
      safety_settings: Optional safety settings sequence.

    Returns:
      List of (text, usage) tuples corresponding 1:1 to input prompts.

    Raises:
      InferenceRuntimeError: If batch job fails or times out.
    """
    if not prompts:
        return []

    # Suppress verbose HTTP logs
    for logger_name in (
        "google.auth.transport.requests",
        "urllib3.connectionpool",
        "httpx",
        "httpcore",
    ):
        logging.getLogger(logger_name).setLevel(logging.WARNING)

    logging.info("Batch API: Processing %d prompts", len(prompts))

    display_base = f"langextract-batch-{int(time.time())}"

    def _process_batch(
        batch_items: Sequence[tuple[int, str]], display: str
    ) -> dict[int, tuple[str, dict[str, int] | None]]:
        """Submit batch job, poll completion, and extract results."""
        requests = [
            _build_inline_request(p, schema_dict, gen_config, system_instruction, safety_settings)
            for _, p in batch_items
        ]
        job = _submit_inline(client, model_id, requests, display)

        if cfg.on_job_create:
            try:
                cfg.on_job_create(job)
            except Exception as e:
                logging.warning("Batch job creation callback failed: %s", e)

        job = _poll_completion(client, job, cfg)
        logging.info("Batch job completed successfully.")

        results = _extract_inline_results(job, cfg, expected_count=len(requests))

        return {
            orig_idx: result
            for (orig_idx, _), result in zip(batch_items, results, strict=True)
        }

    all_items = list(enumerate(prompts))
    new_results: dict[int, tuple[str, dict[str, int] | None]] = {}

    if cfg.max_prompts_per_job and len(all_items) > cfg.max_prompts_per_job:
        chunk_size = cfg.max_prompts_per_job
        for chunk_num, i in enumerate(range(0, len(all_items), chunk_size)):
            chunk_items = all_items[i : i + chunk_size]
            chunk_results = _process_batch(chunk_items, f"{display_base}-part-{chunk_num}")
            new_results.update(chunk_results)
    else:
        new_results = _process_batch(all_items, display_base)

    return [new_results.get(i, ("", None)) for i in range(len(prompts))]
