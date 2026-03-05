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

"""Ling Extract Plus Plus: extract structured information from text with LLMs.

This package provides the main extract and visualize functions,
with lazy loading for other submodules accessed via attribute access.
"""


import importlib
import sys
from typing import Any

from langextract import visualization
from langextract.extraction import PdfExtractionResult
from langextract.extraction import extract as extract_func
from langextract.extraction import extract_pdf as extract_pdf_func

__all__ = [
    "annotation",
    "cli",
    "core",
    "data",
    "exceptions",
    "extract",
    "extract_pdf",
    "factory",
    "io",
    "plugins",
    "prompting",
    "providers",
    "resolver",
    "schema",
    "visualization",
    "visualize",
]

_CACHE: dict[str, Any] = {}


def extract(*args: Any, **kwargs: Any):
    """Top-level API: lx.extract(...)."""
    return extract_func(*args, **kwargs)


def extract_pdf(*args: Any, **kwargs: Any) -> PdfExtractionResult:
    """Top-level API: lx.extract_pdf(path, ...)."""
    return extract_pdf_func(*args, **kwargs)


def visualize(*args: Any, **kwargs: Any):
    """Top-level API: lx.visualize(...)."""
    return visualization.visualize(*args, **kwargs)


# PEP 562 lazy loading
_LAZY_MODULES = {
    "annotation": "langextract.annotation",
    "cli": "langextract.cli",
    "chunking": "langextract.chunking",
    "data": "langextract.data",
    "data_lib": "langextract.data_lib",
    "debug_utils": "langextract.core.debug_utils",
    "exceptions": "langextract.exceptions",
    "factory": "langextract.factory",
    "io": "langextract.io",
    "progress": "langextract.progress",
    "prompting": "langextract.prompting",
    "providers": "langextract.providers",
    "resolver": "langextract.resolver",
    "schema": "langextract.schema",
    "tokenizer": "langextract.tokenizer",
    "visualization": "langextract.visualization",
    "core": "langextract.core",
    "plugins": "langextract.plugins",
}


def __getattr__(name: str) -> Any:
    if name in _CACHE:
        return _CACHE[name]
    modpath = _LAZY_MODULES.get(name)
    if modpath is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = importlib.import_module(modpath)
    # ensure future 'import langextract.<name>' returns the same module
    sys.modules[f"{__name__}.{name}"] = module
    setattr(sys.modules[__name__], name, module)
    _CACHE[name] = module
    return module


def __dir__():
    return sorted(__all__)
