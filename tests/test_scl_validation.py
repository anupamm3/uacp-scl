"""Unit tests for SCL semantic validation rules."""

import pytest

from uacp.scl.ast import Document, Example, Identity, Output, Variable
from uacp.scl.errors import SCLValidationError
from uacp.scl.validate import validate, validate_strict


def test_valid_document_has_no_errors() -> None:
    doc = Document(
        identity=Identity(profile="auditor", body="You are an auditor."),
        variables=[
            Variable(name="data", source="runtime.data", template="Data: {{data}}"),
        ],
        examples=[
            Example(input="Input: {{data}}", output="Output text"),
        ],
        output=Output(
            format="json_schema",
            strict=True,
            body='{"type": "object", "properties": {"res": {"type": "string"}}}',
            schema={"type": "object", "properties": {"res": {"type": "string"}}},
        ),
    )
    diagnostics = validate(doc)
    assert len(diagnostics) == 0
    validate_strict(doc)  # should not raise


def test_identity_empty_profile_error() -> None:
    doc = Document(
        identity=Identity(profile="   ", body="Body"),
    )
    diagnostics = validate(doc)
    assert any("profile" in d.message for d in diagnostics)
    with pytest.raises(SCLValidationError):
        validate_strict(doc)


def test_invalid_variable_name() -> None:
    doc = Document(
        identity=Identity(profile="auditor", body="Body"),
        variables=[
            Variable(name="Invalid-Name!", source="runtime.data"),
        ],
    )
    diagnostics = validate(doc)
    assert any("must match" in d.message for d in diagnostics)


def test_duplicate_variable_name() -> None:
    doc = Document(
        identity=Identity(profile="auditor", body="Body"),
        variables=[
            Variable(name="user_id", source="runtime.user1"),
            Variable(name="user_id", source="runtime.user2"),
        ],
    )
    diagnostics = validate(doc)
    assert any("Duplicate variable declaration" in d.message for d in diagnostics)


def test_undeclared_placeholder_in_identity() -> None:
    doc = Document(
        identity=Identity(profile="auditor", body="Hello {{missing_var}}"),
        variables=[],
    )
    diagnostics = validate(doc)
    assert any("references undeclared variable 'missing_var'" in d.message for d in diagnostics)


def test_undeclared_placeholder_in_template() -> None:
    doc = Document(
        identity=Identity(profile="auditor", body="Auditor body"),
        variables=[
            Variable(name="var_a", source="runtime.a", template="{{var_a}} and {{var_b}}"),
        ],
    )
    diagnostics = validate(doc)
    assert any("references undeclared variable 'var_b'" in d.message for d in diagnostics)


def test_undeclared_placeholder_in_example() -> None:
    doc = Document(
        identity=Identity(profile="auditor", body="Auditor body"),
        variables=[],
        examples=[
            Example(input="{{unknown_input}}", output="Output"),
        ],
    )
    diagnostics = validate(doc)
    assert any("references undeclared variable 'unknown_input'" in d.message for d in diagnostics)


def test_strict_true_requires_json_schema() -> None:
    doc = Document(
        identity=Identity(profile="auditor", body="Body"),
        output=Output(format="markdown", strict=True, body="Format in markdown"),
    )
    diagnostics = validate(doc)
    assert any("strict='true' requires format='json_schema'" in d.message for d in diagnostics)


def test_strict_true_requires_object_root_type() -> None:
    doc = Document(
        identity=Identity(profile="auditor", body="Body"),
        output=Output(
            format="json_schema",
            strict=True,
            body='{"type": "string"}',
            schema={"type": "string"},
        ),
    )
    diagnostics = validate(doc)
    assert any("root type is 'object'" in d.message for d in diagnostics)


def test_invalid_draft07_schema() -> None:
    doc = Document(
        identity=Identity(profile="auditor", body="Body"),
        output=Output(
            format="json_schema",
            strict=False,
            body='{"type": 123}',
            schema={"type": 123},  # type must be string or array in Draft-07
        ),
    )
    diagnostics = validate(doc)
    assert any("Invalid Draft-07 JSON Schema" in d.message for d in diagnostics)


def test_example_output_mismatch_emits_warning() -> None:
    doc = Document(
        identity=Identity(profile="auditor", body="Body"),
        examples=[
            Example(
                input="Sample input",
                output='{"score": "not_a_number"}',  # violates type integer
            )
        ],
        output=Output(
            format="json_schema",
            strict=True,
            body='{"type": "object", "properties": {"score": {"type": "integer"}}}',
            schema={"type": "object", "properties": {"score": {"type": "integer"}}},
        ),
    )
    diagnostics = validate(doc)
    warnings = [d for d in diagnostics if d.severity == "warning"]
    errors = [d for d in diagnostics if d.severity == "error"]

    assert len(errors) == 0
    assert len(warnings) == 1
    assert "does not conform to schema" in warnings[0].message
    # Warnings must not cause validate_strict to fail
    validate_strict(doc)


def test_example_output_non_json_is_ignored_for_schema_validation() -> None:
    doc = Document(
        identity=Identity(profile="auditor", body="Body"),
        examples=[
            Example(input="Sample input", output="Plain text non-JSON demonstration"),
        ],
        output=Output(
            format="json_schema",
            strict=True,
            body='{"type": "object"}',
            schema={"type": "object"},
        ),
    )
    diagnostics = validate(doc)
    assert len(diagnostics) == 0

