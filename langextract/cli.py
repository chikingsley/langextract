# Copyright 2026 Peacockery Studio LLC.
#
# Author: Chi Ejimofor <chi@peacockery.studio>
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

"""General LangExtract CLI."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import langextract as lx
from langextract import prompt_validation as pv

DEFAULT_MODEL = "gemini-2.5-flash-lite"
DEFAULT_MAX_CHAR_BUFFER = 2_000
DEFAULT_EXTRACTION_PASSES = 2
DEFAULT_MAX_WORKERS = 8

CONTRACTS_PROMPT = """\
Extract every clause, sentence, or phrase that falls into these categories. \
Be EXHAUSTIVE. Flag everything. False positives acceptable. Missing something is not.

Categories:
- scope_creep: Language adding maintenance, repair, removal, or replacement
  obligations beyond install-and-inspect.
- mobilization_trap: "all mobilizations included" or single mobilization for
  multi-month project.
- inapplicable_section: Sections not applicable to SWPPP (hoisting,
  scaffolding, fire stopping, caulking, sealants, sleeves, penetrations, roof
  flashings).
- overbroad_language: "all permits", "all fees", "all licenses" beyond what's in the estimate.
- noi_not_filing: Requirement to file NOI, NOT, MSGP, or stormwater documentation.
- record_retention: Requirement to retain records or maintain documents for a period.
- inspection_terms: Inspection frequency, quantity, schedule, rain event
  triggers. Include exact numbers.
- payment_terms: Payment timing, retainage, pay-when-paid, net terms, billing windows.
- liability_language: Indemnification, hold harmless, fines, penalties, citations.
- company_name_issue: Misspelling of "Desert Services" or former name "IDG" /
  "Innovative Development Group".
- missing_quantity: Where quantities should be specified but aren't.

For each extraction, use attributes to specify severity
(critical/warning/info) and recommended_action.
"""

CONTRACTS_EXAMPLE = lx.data.ExampleData(
    text=(
        "Subcontractor shall furnish, install, maintain, and remove the stabilized "
        "construction entrance per Detail EC-5. Subcontractor shall include all costs "
        "of all permits, fees, and licenses. Payment terms are net 45 from receipt of "
        "approved pay application."
    ),
    extractions=[
        lx.data.Extraction(
            extraction_class="scope_creep",
            extraction_text="maintain, and remove",
            description="Adds maintenance and removal beyond install scope.",
            attributes={
                "severity": "critical",
                "recommended_action": "Strike 'maintain, and remove'. Install only.",
            },
        ),
        lx.data.Extraction(
            extraction_class="overbroad_language",
            extraction_text="all costs of all permits, fees, and licenses",
            description="Too broad - limit to permits in estimate only.",
            attributes={
                "severity": "critical",
                "recommended_action": "Replace with 'only permits included in estimate'.",
            },
        ),
        lx.data.Extraction(
            extraction_class="payment_terms",
            extraction_text="net 45 from receipt of approved pay application",
            description="Payment terms identified.",
            attributes={
                "severity": "info",
                "recommended_action": "Verify net 45 is acceptable.",
            },
        ),
    ],
)

PROFILE_LIBRARY: dict[str, tuple[str, list[lx.data.ExampleData]]] = {
    "contracts": (CONTRACTS_PROMPT, [CONTRACTS_EXAMPLE]),
}


def _safe_attributes(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        return {}
    return {str(k): v for k, v in value.items()}


def _read_source_text(args: argparse.Namespace) -> str:
    if args.text_file:
        path = args.text_file
        if not path.exists():
            raise FileNotFoundError(f"text file not found: {path}")
        return path.read_text(encoding="utf-8")
    if args.text is not None:
        return args.text
    if args.stdin:
        return sys.stdin.read()
    raise ValueError("one input source is required")


def _parse_examples_payload(payload: Any) -> list[lx.data.ExampleData]:
    raw_examples = payload.get("examples") if isinstance(payload, dict) else payload
    if not isinstance(raw_examples, list) or not raw_examples:
        raise ValueError("examples payload must be a non-empty list or {\"examples\": [...]}")

    parsed_examples: list[lx.data.ExampleData] = []
    for idx, raw_example in enumerate(raw_examples):
        if not isinstance(raw_example, dict):
            raise ValueError(f"example[{idx}] must be an object")
        text = raw_example.get("text")
        if not isinstance(text, str) or not text.strip():
            raise ValueError(f"example[{idx}].text must be a non-empty string")

        raw_extractions = raw_example.get("extractions")
        if not isinstance(raw_extractions, list) or not raw_extractions:
            raise ValueError(f"example[{idx}].extractions must be a non-empty list")

        extractions: list[lx.data.Extraction] = []
        for ex_idx, raw_extraction in enumerate(raw_extractions):
            if not isinstance(raw_extraction, dict):
                raise ValueError(f"example[{idx}].extractions[{ex_idx}] must be an object")

            extraction_class = raw_extraction.get("class") or raw_extraction.get(
                "extraction_class"
            )
            extraction_text = raw_extraction.get("text") or raw_extraction.get("extraction_text")
            if not isinstance(extraction_class, str) or not extraction_class.strip():
                raise ValueError(
                    f"example[{idx}].extractions[{ex_idx}] missing class/extraction_class"
                )
            if not isinstance(extraction_text, str) or not extraction_text.strip():
                raise ValueError(
                    f"example[{idx}].extractions[{ex_idx}] missing text/extraction_text"
                )

            description = raw_extraction.get("description")
            if description is not None and not isinstance(description, str):
                raise ValueError(
                    f"example[{idx}].extractions[{ex_idx}].description must be string"
                )

            attributes = raw_extraction.get("attributes")
            if attributes is not None and not isinstance(attributes, dict):
                raise ValueError(
                    f"example[{idx}].extractions[{ex_idx}].attributes must be object"
                )

            extractions.append(
                lx.data.Extraction(
                    extraction_class=extraction_class.strip(),
                    extraction_text=extraction_text.strip(),
                    description=description,
                    attributes=_safe_attributes(attributes),
                )
            )

        parsed_examples.append(lx.data.ExampleData(text=text, extractions=extractions))

    return parsed_examples


def _load_examples(args: argparse.Namespace) -> list[lx.data.ExampleData]:
    if args.examples_file:
        path = args.examples_file
        if not path.exists():
            raise FileNotFoundError(f"examples file not found: {path}")
        payload = json.loads(path.read_text(encoding="utf-8"))
        return _parse_examples_payload(payload)

    if args.profile in PROFILE_LIBRARY:
        return PROFILE_LIBRARY[args.profile][1]

    raise ValueError(
        f"unknown profile '{args.profile}'. Provide --examples-file or choose from: "
        + ", ".join(sorted(PROFILE_LIBRARY))
    )


def _load_prompt(args: argparse.Namespace) -> str:
    if args.prompt_file is not None:
        if not args.prompt_file.exists():
            raise FileNotFoundError(f"prompt file not found: {args.prompt_file}")
        prompt = args.prompt_file.read_text(encoding="utf-8").strip()
        if prompt:
            return prompt
        raise ValueError("prompt file is empty")

    if args.profile in PROFILE_LIBRARY:
        return PROFILE_LIBRARY[args.profile][0]

    raise ValueError(
        f"unknown profile '{args.profile}'. Provide --prompt-file or choose from: "
        + ", ".join(sorted(PROFILE_LIBRARY))
    )


def _serialize_result(
    *,
    model_id: str,
    profile: str,
    doc: lx.data.AnnotatedDocument,
) -> dict[str, Any]:
    entities: list[dict[str, Any]] = []
    for ext in doc.extractions or []:
        start = ext.char_interval.start_pos if ext.char_interval else None
        end = ext.char_interval.end_pos if ext.char_interval else None
        entities.append(
            {
                "class": ext.extraction_class,
                "text": ext.extraction_text,
                "attributes": _safe_attributes(ext.attributes),
                "start": start,
                "end": end,
            }
        )

    entities.sort(
        key=lambda item: (
            item["start"] is None,
            item["start"] if isinstance(item["start"], int) else 0,
        )
    )
    return {
        "ok": True,
        "profile": profile,
        "model": model_id,
        "extraction_count": len(entities),
        "entities": entities,
    }


def _render_json(payload: dict[str, Any], output: Path | None, pretty: bool) -> None:
    text = json.dumps(payload, indent=2 if pretty else None, ensure_ascii=True)
    if pretty:
        text += "\n"

    if output is None:
        sys.stdout.write(text)
        if not text.endswith("\n"):
            sys.stdout.write("\n")
        return

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(text + ("" if text.endswith("\n") else "\n"), encoding="utf-8")


def run_extract(args: argparse.Namespace) -> dict[str, Any]:
    text = _read_source_text(args)
    if not text.strip():
        raise ValueError("source text is empty")
    if args.max_char_buffer <= 0:
        raise ValueError("--max-char-buffer must be > 0")
    if args.extraction_passes <= 0:
        raise ValueError("--extraction-passes must be > 0")
    if args.max_workers <= 0:
        raise ValueError("--max-workers must be > 0")

    prompt = _load_prompt(args)
    examples = _load_examples(args)

    doc = lx.extract(
        text,
        prompt_description=prompt,
        examples=examples,
        model_id=args.model,
        extraction_passes=args.extraction_passes,
        max_char_buffer=args.max_char_buffer,
        max_workers=args.max_workers,
        temperature=args.temperature,
        show_progress=args.show_progress,
        prompt_validation_level=pv.PromptValidationLevel.ERROR,
    )
    first = doc[0] if isinstance(doc, list) else doc
    if first is None:
        first = lx.data.AnnotatedDocument(extractions=[], text=text)
    return _serialize_result(model_id=args.model, profile=args.profile, doc=first)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="General LangExtract CLI for profile-based and custom extraction tasks."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    profiles_parser = subparsers.add_parser("profiles", help="List built-in extraction profiles.")
    profiles_parser.add_argument(
        "--format",
        choices=("text", "json"),
        default="text",
        help="Output format for profile list.",
    )

    extract_parser = subparsers.add_parser("extract", help="Run structured extraction.")
    source_group = extract_parser.add_mutually_exclusive_group(required=True)
    source_group.add_argument("--text-file", type=Path, help="Path to UTF-8 source text file.")
    source_group.add_argument("--text", help="Raw source text.")
    source_group.add_argument("--stdin", action="store_true", help="Read source text from stdin.")

    extract_parser.add_argument(
        "--profile",
        default="contracts",
        help="Built-in extraction profile (default: contracts).",
    )
    extract_parser.add_argument(
        "--prompt-file",
        type=Path,
        help="Override prompt from file. Required for unknown profiles.",
    )
    extract_parser.add_argument(
        "--examples-file",
        type=Path,
        help="Examples JSON file. Required for unknown profiles.",
    )
    extract_parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f"Model id (default: {DEFAULT_MODEL}).",
    )
    extract_parser.add_argument(
        "--max-char-buffer",
        type=int,
        default=DEFAULT_MAX_CHAR_BUFFER,
        help=f"Chunk size in chars (default: {DEFAULT_MAX_CHAR_BUFFER}).",
    )
    extract_parser.add_argument(
        "--extraction-passes",
        type=int,
        default=DEFAULT_EXTRACTION_PASSES,
        help=f"Sequential extraction passes (default: {DEFAULT_EXTRACTION_PASSES}).",
    )
    extract_parser.add_argument(
        "--max-workers",
        type=int,
        default=DEFAULT_MAX_WORKERS,
        help=f"Concurrent workers for chunk inference (default: {DEFAULT_MAX_WORKERS}).",
    )
    extract_parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature (default: 0.0).",
    )
    extract_parser.add_argument(
        "--show-progress",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Show LangExtract progress output (default: false).",
    )
    extract_parser.add_argument(
        "--output",
        "-o",
        type=Path,
        help="Optional output file path. Defaults to stdout.",
    )
    extract_parser.add_argument(
        "--pretty",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Pretty-print output JSON (default: false).",
    )
    return parser


def _run_profiles(args: argparse.Namespace) -> int:
    names = sorted(PROFILE_LIBRARY.keys())
    if args.format == "json":
        print(json.dumps({"profiles": names}))
    else:
        for name in names:
            print(name)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "profiles":
            return _run_profiles(args)
        if args.command == "extract":
            payload = run_extract(args)
            _render_json(payload, args.output, args.pretty)
            return 0
        raise ValueError(f"unsupported command: {args.command}")
    except Exception as err:  # pragma: no cover - error path CLI behavior.
        print(f"[langextract-cli] {err}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
