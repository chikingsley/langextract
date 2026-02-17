#!/bin/bash
# Copyright 2025 Google LLC
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

# Autoformat LangExtract codebase
#
# Usage: ./autoformat.sh [target_directory ...]
#        If no target is specified, formats the current directory
#
# This script runs:
# 1. ruff format for code formatting
# 2. ruff check --fix for lint + import sorting autofixes
# 3. pre-commit hooks for additional formatting (trailing whitespace, end-of-file, etc.)

set -e

echo "LangExtract Auto-formatter"
echo "=========================="
echo

# Check for required tools
check_tool() {
    if ! command -v "$1" &> /dev/null; then
        echo "Error: $1 not found. Please install with: pip install $1"
        exit 1
    fi
}

check_tool "ruff"
check_tool "pre-commit"

# Parse command line arguments
show_usage() {
    echo "Usage: $0 [target_directory ...]"
    echo
    echo "Formats Python code using Ruff."
    echo
    echo "Arguments:"
    echo "  target_directory    One or more directories to format (default: langextract tests)"
    echo
    echo "Examples:"
    echo "  $0                  # Format langextract and tests directories"
    echo "  $0 langextract      # Format only langextract directory"
    echo "  $0 src tests        # Format multiple specific directories"
}

# Check for help flag
if [ "$1" = "-h" ] || [ "$1" = "--help" ]; then
    show_usage
    exit 0
fi

# Determine target directories
if [ $# -eq 0 ]; then
    TARGETS="langextract tests"
    echo "No target specified. Formatting default directories: langextract tests"
else
    TARGETS="$@"
    echo "Formatting targets: $TARGETS"
fi

# Run Ruff formatter
echo "Running ruff format..."
if ruff format $TARGETS; then
    echo "Formatting complete"
else
    echo "Formatting failed"
    exit 1
fi

echo

# Run Ruff lint autofix
echo "Running ruff check --fix..."
if ruff check --fix $TARGETS; then
    echo "Lint autofixes complete"
else
    echo "Lint autofixes failed"
    exit 1
fi

echo

# Run pre-commit hooks for additional formatting
echo "Running pre-commit hooks for additional formatting..."
if pre-commit run --all-files; then
    echo "Pre-commit hooks passed"
else
    echo "Pre-commit hooks made changes - please review"
    # Exit with success since formatting was applied
    exit 0
fi

echo
echo "All formatting complete!"
echo
echo "Next steps:"
echo "  - Run: ruff check $TARGETS"
echo "  - Commit your changes"
