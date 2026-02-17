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

"""Live CLI smoke test against a real fixture."""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONTRACTS_ROOT = Path(__file__).resolve().parents[3]
REAL_FIXTURE = (
    CONTRACTS_ROOT
    / "ground-truth"
    / "test-fixtures"
    / "modera-paradise-valley"
    / "reconciled.md"
)

skip_if_no_gemini = pytest.mark.skipif(
    not os.environ.get("GEMINI_API_KEY"),
    reason="Gemini API key not available (set GEMINI_API_KEY)",
)

live_api = pytest.mark.live_api


@live_api
@skip_if_no_gemini
def test_langextract_cli_live_smoke(tmp_path):
    assert REAL_FIXTURE.exists(), f"missing fixture: {REAL_FIXTURE}"

    sample_text = REAL_FIXTURE.read_text(encoding="utf-8")[:4500]
    input_path = tmp_path / "contract-sample.txt"
    input_path.write_text(sample_text, encoding="utf-8")

    proc = subprocess.run(
        [
            "uv",
            "run",
            "langextract-cli",
            "extract",
            "--profile",
            "contracts",
            "--text-file",
            str(input_path),
            "--model",
            os.environ.get("CONTRACT_LANGEXTRACT_MODEL", "gemini-2.5-flash-lite"),
            "--max-char-buffer",
            "1500",
            "--extraction-passes",
            "1",
            "--max-workers",
            "4",
        ],
        cwd=PROJECT_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert proc.returncode == 0, proc.stderr[-1000:]
    payload = json.loads(proc.stdout)
    assert payload["ok"] is True
    assert isinstance(payload["entities"], list)
    assert payload["model"]
