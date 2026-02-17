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

"""Tests for the general LangExtract CLI behavior."""

from __future__ import annotations

import contextlib
import io
import json
import tempfile
from pathlib import Path
from unittest import mock

import langextract as lx
import langextract.cli as cli
from absl.testing import absltest

CONTRACTS_ROOT = Path(__file__).resolve().parents[3]
REAL_FIXTURE = (
    CONTRACTS_ROOT
    / "ground-truth"
    / "test-fixtures"
    / "modera-paradise-valley"
    / "reconciled.md"
)


def _fake_doc() -> lx.data.AnnotatedDocument:
    return lx.data.AnnotatedDocument(
        text="Subcontractor shall include all permits.",
        extractions=[
            lx.data.Extraction(
                extraction_class="overbroad_language",
                extraction_text="all permits",
                char_interval=lx.data.CharInterval(start_pos=27, end_pos=38),
                attributes={
                    "severity": "critical",
                    "recommended_action": "Limit to estimate scope.",
                },
            ),
            lx.data.Extraction(
                extraction_class="payment_terms",
                extraction_text="Net 30",
                char_interval=lx.data.CharInterval(start_pos=40, end_pos=46),
                attributes={"severity": "info"},
            ),
        ],
    )


class LangExtractCliTest(absltest.TestCase):
    def test_uses_real_fixture_text_file(self):
        self.assertTrue(REAL_FIXTURE.exists(), f"missing fixture: {REAL_FIXTURE}")
        fixture_text = REAL_FIXTURE.read_text(encoding="utf-8")
        self.assertTrue(fixture_text.strip())

        stdout = io.StringIO()
        with (
            mock.patch("langextract.cli.lx.extract", return_value=_fake_doc()) as mock_extract,
            contextlib.redirect_stdout(stdout),
        ):
            exit_code = cli.main(
                [
                    "extract",
                    "--profile",
                    "contracts",
                    "--text-file",
                    str(REAL_FIXTURE),
                    "--model",
                    "gemini-2.5-flash-lite",
                ]
            )

        self.assertEqual(exit_code, 0)
        payload = json.loads(stdout.getvalue())
        self.assertTrue(payload["ok"])
        self.assertEqual(payload["model"], "gemini-2.5-flash-lite")
        self.assertEqual(payload["extraction_count"], 2)
        self.assertLen(payload["entities"], 2)
        self.assertEqual(payload["entities"][0]["class"], "overbroad_language")

        args = mock_extract.call_args
        self.assertIsNotNone(args)
        self.assertEqual(args.args[0], fixture_text)

    def test_writes_output_file(self):
        self.assertTrue(REAL_FIXTURE.exists(), f"missing fixture: {REAL_FIXTURE}")

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "cli-output.json"
            with mock.patch("langextract.cli.lx.extract", return_value=_fake_doc()):
                exit_code = cli.main(
                    [
                        "extract",
                        "--profile",
                        "contracts",
                        "--text-file",
                        str(REAL_FIXTURE),
                        "--output",
                        str(output_path),
                        "--pretty",
                    ]
                )

            self.assertEqual(exit_code, 0)
            self.assertTrue(output_path.exists())
            payload = json.loads(output_path.read_text(encoding="utf-8"))
            self.assertTrue(payload["ok"])
            self.assertEqual(payload["extraction_count"], 2)

    def test_returns_error_for_empty_source(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            empty_path = Path(tmpdir) / "empty.txt"
            empty_path.write_text("", encoding="utf-8")

            stderr = io.StringIO()
            with contextlib.redirect_stderr(stderr):
                exit_code = cli.main(
                    [
                        "extract",
                        "--profile",
                        "contracts",
                        "--text-file",
                        str(empty_path),
                    ]
                )

            self.assertEqual(exit_code, 1)
            self.assertIn("source text is empty", stderr.getvalue())

    def test_profiles_command(self):
        stdout = io.StringIO()
        with contextlib.redirect_stdout(stdout):
            exit_code = cli.main(["profiles"])

        self.assertEqual(exit_code, 0)
        profile_names = {line.strip() for line in stdout.getvalue().splitlines() if line.strip()}
        self.assertIn("contracts", profile_names)


if __name__ == "__main__":
    absltest.main()
