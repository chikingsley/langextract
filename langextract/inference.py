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

"""Public inference API."""


import enum

from langextract.core.base_model import BaseLanguageModel
from langextract.core.exceptions import InferenceOutputError
from langextract.core.types import ScoredOutput
from langextract.providers.gemini import GeminiLanguageModel
from langextract.providers.ollama import OllamaLanguageModel
from langextract.providers.openai import OpenAILanguageModel


class InferenceType(enum.Enum):
    ITERATIVE = "iterative"
    MULTIPROCESS = "multiprocess"


__all__ = [
    "BaseLanguageModel",
    "GeminiLanguageModel",
    "InferenceOutputError",
    "InferenceType",
    "OllamaLanguageModel",
    "OpenAILanguageModel",
    "ScoredOutput",
]
