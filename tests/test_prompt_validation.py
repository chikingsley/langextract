"""Tests for prompt alignment validation (fork-specific feature)."""

import copy
import logging

import pytest
from langextract.core.data import AlignmentStatus, ExampleData, Extraction
from langextract.prompt_validation import (
    AlignmentPolicy,
    PromptAlignmentError,
    PromptValidationLevel,
    ValidationIssue,
    ValidationReport,
    handle_alignment_report,
    validate_prompt_alignment,
)

# ---------------------------------------------------------------------------
# Exports
# ---------------------------------------------------------------------------


class TestExports:
    def test_all_exports_importable(self):
        from langextract import prompt_validation

        for name in prompt_validation.__all__:
            assert hasattr(prompt_validation, name), f"Missing export: {name}"

    def test_expected_exports(self):
        from langextract import prompt_validation

        expected = {
            "AlignmentPolicy",
            "PromptAlignmentError",
            "PromptValidationLevel",
            "ValidationIssue",
            "ValidationReport",
            "handle_alignment_report",
            "validate_prompt_alignment",
        }
        assert set(prompt_validation.__all__) == expected


# ---------------------------------------------------------------------------
# Validation levels
# ---------------------------------------------------------------------------


class TestValidationLevels:
    def test_off_level_does_nothing(self):
        """OFF level should not log or raise, even with failed issues."""
        issue = ValidationIssue(
            example_index=0,
            example_id=None,
            extraction_class="entity",
            extraction_text_preview="some text",
            alignment_status=None,
            issue_kind=ValidationIssue.__dataclass_fields__["issue_kind"].type.__args__[0]
            if hasattr(ValidationIssue.__dataclass_fields__["issue_kind"].type, "__args__")
            else None,
        )
        # Build the issue_kind properly from the module internals
        from langextract.prompt_validation import _IssueKind

        issue = ValidationIssue(
            example_index=0,
            example_id=None,
            extraction_class="entity",
            extraction_text_preview="some text",
            alignment_status=None,
            issue_kind=_IssueKind.FAILED,
        )
        report = ValidationReport(issues=[issue])

        # Should not raise
        handle_alignment_report(report, PromptValidationLevel.OFF)

    def test_warning_level_logs(self, caplog):
        from langextract.prompt_validation import _IssueKind

        issue = ValidationIssue(
            example_index=0,
            example_id=None,
            extraction_class="entity",
            extraction_text_preview="test entity",
            alignment_status=None,
            issue_kind=_IssueKind.FAILED,
        )
        report = ValidationReport(issues=[issue])

        with caplog.at_level(logging.WARNING):
            handle_alignment_report(report, PromptValidationLevel.WARNING)

        assert any("FAILED" in rec.message for rec in caplog.records)

    def test_error_level_raises_on_failed(self):
        from langextract.prompt_validation import _IssueKind

        issue = ValidationIssue(
            example_index=0,
            example_id=None,
            extraction_class="entity",
            extraction_text_preview="test entity",
            alignment_status=None,
            issue_kind=_IssueKind.FAILED,
        )
        report = ValidationReport(issues=[issue])

        with pytest.raises(PromptAlignmentError, match="could not be aligned"):
            handle_alignment_report(report, PromptValidationLevel.ERROR)

    def test_error_level_no_raise_on_non_exact_without_strict(self):
        from langextract.prompt_validation import _IssueKind

        issue = ValidationIssue(
            example_index=0,
            example_id=None,
            extraction_class="entity",
            extraction_text_preview="test entity",
            alignment_status=AlignmentStatus.MATCH_FUZZY,
            issue_kind=_IssueKind.NON_EXACT,
        )
        report = ValidationReport(issues=[issue])

        # Should not raise without strict_non_exact
        handle_alignment_report(report, PromptValidationLevel.ERROR)

    def test_error_level_raises_on_non_exact_with_strict(self):
        from langextract.prompt_validation import _IssueKind

        issue = ValidationIssue(
            example_index=0,
            example_id=None,
            extraction_class="entity",
            extraction_text_preview="test entity",
            alignment_status=AlignmentStatus.MATCH_FUZZY,
            issue_kind=_IssueKind.NON_EXACT,
        )
        report = ValidationReport(issues=[issue])

        with pytest.raises(PromptAlignmentError, match="strict mode"):
            handle_alignment_report(
                report, PromptValidationLevel.ERROR, strict_non_exact=True
            )


# ---------------------------------------------------------------------------
# Empty examples
# ---------------------------------------------------------------------------


class TestEmptyExamples:
    def test_empty_examples_returns_clean_report(self):
        report = validate_prompt_alignment([])
        assert report.issues == []
        assert report.has_failed is False
        assert report.has_non_exact is False


# ---------------------------------------------------------------------------
# Validation does not mutate input
# ---------------------------------------------------------------------------


class TestNoMutation:
    def test_validation_does_not_mutate_examples(self):
        examples = [
            ExampleData(
                text="Alice went to Paris",
                extractions=[
                    Extraction("person", "Alice"),
                    Extraction("location", "Paris"),
                ],
            )
        ]
        originals = copy.deepcopy(examples)
        validate_prompt_alignment(examples)

        for orig, ex in zip(originals, examples, strict=False):
            assert orig.text == ex.text
            assert len(orig.extractions) == len(ex.extractions)
            for o_ext, e_ext in zip(orig.extractions, ex.extractions, strict=False):
                assert o_ext.extraction_class == e_ext.extraction_class
                assert o_ext.extraction_text == e_ext.extraction_text
                assert o_ext.alignment_status == e_ext.alignment_status
                assert o_ext.char_interval == e_ext.char_interval


# ---------------------------------------------------------------------------
# Exact alignment
# ---------------------------------------------------------------------------


class TestAlignment:
    def test_exact_match_produces_no_issues(self):
        """Extraction text that appears verbatim in the source should align exactly."""
        examples = [
            ExampleData(
                text="Alice went to Paris last summer.",
                extractions=[
                    Extraction("person", "Alice"),
                    Extraction("location", "Paris"),
                ],
            )
        ]
        report = validate_prompt_alignment(examples)
        # Exact matches do not produce issues
        assert report.has_failed is False

    def test_no_match_produces_failed_issue(self):
        """Extraction text not present at all should fail alignment."""
        examples = [
            ExampleData(
                text="Alice went to Paris last summer.",
                extractions=[
                    Extraction("person", "NONEXISTENT_TEXT_12345"),
                ],
            )
        ]
        report = validate_prompt_alignment(examples)
        assert report.has_failed is True
        assert len(report.issues) >= 1
        assert report.issues[0].extraction_class == "person"


# ---------------------------------------------------------------------------
# ValidationReport properties
# ---------------------------------------------------------------------------


class TestValidationReport:
    def test_has_failed_false_when_no_issues(self):
        report = ValidationReport(issues=[])
        assert report.has_failed is False
        assert report.has_non_exact is False

    def test_has_failed_true(self):
        from langextract.prompt_validation import _IssueKind

        issue = ValidationIssue(
            example_index=0,
            example_id=None,
            extraction_class="x",
            extraction_text_preview="y",
            alignment_status=None,
            issue_kind=_IssueKind.FAILED,
        )
        report = ValidationReport(issues=[issue])
        assert report.has_failed is True
        assert report.has_non_exact is False

    def test_has_non_exact_true(self):
        from langextract.prompt_validation import _IssueKind

        issue = ValidationIssue(
            example_index=0,
            example_id=None,
            extraction_class="x",
            extraction_text_preview="y",
            alignment_status=AlignmentStatus.MATCH_FUZZY,
            issue_kind=_IssueKind.NON_EXACT,
        )
        report = ValidationReport(issues=[issue])
        assert report.has_failed is False
        assert report.has_non_exact is True


# ---------------------------------------------------------------------------
# AlignmentPolicy defaults
# ---------------------------------------------------------------------------


class TestAlignmentPolicy:
    def test_defaults(self):
        policy = AlignmentPolicy()
        assert policy.enable_fuzzy_alignment is True
        assert policy.fuzzy_alignment_threshold == 0.75
        assert policy.accept_match_lesser is True
