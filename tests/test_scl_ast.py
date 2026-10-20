"""Tests for SCL AST models, placeholder extraction, and error formatting."""

import pytest
from pydantic import ValidationError

from uacp.scl.ast import (
    Document,
    Example,
    Identity,
    Output,
    Span,
    Variable,
    placeholders,
)
from uacp.scl.errors import (
    Diagnostic,
    MissingVariableError,
    SCLError,
    SCLSyntaxError,
    SCLValidationError,
    format_source_caret,
)


def test_span_construction() -> None:
    span = Span(line=1, col=1, end_line=1, end_col=10)
    assert span.line == 1
    assert span.col == 1
    assert span.end_line == 1
    assert span.end_col == 10


def test_identity_construction_and_defaults() -> None:
    ident = Identity(profile="auditor", body="You are an auditor.")
    assert ident.profile == "auditor"
    assert ident.tone is None
    assert ident.body == "You are an auditor."
    assert ident.span is None

    ident_with_tone = Identity(
        profile="auditor",
        tone="strict",
        body="Verify all logs.",
        span=Span(line=2, col=3, end_line=4, end_col=15),
    )
    assert ident_with_tone.tone == "strict"
    assert ident_with_tone.span is not None
    assert ident_with_tone.span.line == 2


def test_variable_construction_and_defaults() -> None:
    var = Variable(name="tx_log", source="runtime.payload.tx")
    assert var.name == "tx_log"
    assert var.source == "runtime.payload.tx"
    assert var.type == "text"
    assert var.template is None

    var_json = Variable(
        name="config",
        source="runtime.config",
        type="json",
        template="Config: {{config}}",
    )
    assert var_json.type == "json"
    assert var_json.template == "Config: {{config}}"


def test_example_construction_and_defaults() -> None:
    ex = Example(input="input text", output="output text")
    assert ex.pair is True
    assert ex.input == "input text"
    assert ex.output == "output text"


def test_output_construction_and_schema_alias() -> None:
    schema_dict = {
        "type": "object",
        "properties": {"verdict": {"type": "string"}},
    }
    out = Output(
        format="json_schema",
        strict=True,
        body='{"type": "object"}',
        schema=schema_dict,
    )
    assert out.format == "json_schema"
    assert out.strict is True
    assert out.schema == schema_dict

    out_md = Output(format="markdown", body="Markdown report format")
    assert out_md.format == "markdown"
    assert out_md.strict is False
    assert out_md.schema is None


def test_document_construction_and_to_dict() -> None:
    ident = Identity(profile="researcher", body="Research things.")
    var = Variable(name="query", source="runtime.payload.query")
    ex = Example(input="q", output="a")
    out = Output(format="markdown", body="Answer:")

    doc = Document(
        identity=ident,
        variables=[var],
        examples=[ex],
        output=out,
    )

    assert doc.version == "1"
    assert len(doc.variables) == 1
    assert len(doc.examples) == 1
    assert doc.output is not None

    d = doc.to_dict()
    assert d["version"] == "1"
    assert d["identity"]["profile"] == "researcher"
    assert d["variables"][0]["name"] == "query"
    assert d["examples"][0]["input"] == "q"
    assert d["output"]["format"] == "markdown"


def test_ast_immutability() -> None:
    ident = Identity(profile="bot", body="Hello")
    with pytest.raises(ValidationError):
        ident.profile = "new_bot"

    doc = Document(identity=ident)
    with pytest.raises(ValidationError):
        doc.version = "2"

    span = Span(line=1, col=1, end_line=1, end_col=5)
    with pytest.raises(ValidationError):
        span.line = 2


def test_placeholder_extraction() -> None:
    text = "Hello {{user_name}}, welcome to {{service_name}}! Contact {{user_name}} for help."
    extracted = placeholders(text)
    assert extracted == ["user_name", "service_name"]


def test_placeholder_extraction_with_whitespace() -> None:
    text = "{{  first_var   }} and {{second_var}} plus {{   first_var  }}"
    assert placeholders(text) == ["first_var", "second_var"]


def test_placeholder_extraction_empty_and_invalid() -> None:
    assert placeholders("No placeholders here") == []
    assert placeholders("Invalid {{123bad}} and {{}}") == []


def test_format_source_caret() -> None:
    source = "line one\nline two with error\nline three"
    formatted = format_source_caret(source, line=2, col=6)
    assert "2 | line two with error" in formatted
    assert "     |      ^" in formatted

    out_of_range = format_source_caret(source, line=10, col=1)
    assert "out of range" in out_of_range


def test_syntax_error_formatting() -> None:
    source = "<scl>\n  <identity>No profile</identity>\n</scl>"
    err = SCLSyntaxError(
        message="Missing required attribute 'profile'",
        line=2,
        col=3,
        hint="Add profile='...'",
    )
    formatted = err.format_with_source(source)
    assert "Syntax Error at 2:3: Missing required attribute 'profile'" in formatted
    assert "2 |   <identity>No profile</identity>" in formatted
    assert "^" in formatted
    assert "Hint: Add profile='...'" in formatted
    assert "Line 2:3" in str(err)
    assert isinstance(err, SCLError)


def test_validation_error_formatting() -> None:
    source = "line 1\nline 2"
    d1 = Diagnostic(message="First error", line=1, col=2, severity="error")
    d2 = Diagnostic(
        message="Second warning",
        line=2,
        col=1,
        severity="warning",
        hint="Be careful",
    )
    val_err = SCLValidationError(diagnostics=[d1, d2])
    formatted = val_err.format_with_source(source)

    assert "[ERROR] at 1:2: First error" in formatted
    assert "[WARNING] at 2:1: Second warning" in formatted
    assert "Hint: Be careful" in formatted
    assert "2 validation error(s)" in str(val_err)


def test_missing_variable_error_formatting() -> None:
    source = "Template with {{undeclared_var}} inside"
    err = MissingVariableError(name="undeclared_var", line=1, col=15)
    formatted = err.format_with_source(source)
    assert "Variable 'undeclared_var' was referenced but not declared" in formatted
    assert "1 | Template with {{undeclared_var}} inside" in formatted

    err_no_coords = MissingVariableError(name="other_var")
    assert "Undeclared variable 'other_var' referenced" in str(err_no_coords)
    assert err_no_coords.format_with_source(source) == str(err_no_coords)
