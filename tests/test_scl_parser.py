"""Unit and snapshot tests for the SCL parser."""

import json
from pathlib import Path

import jsonschema  # type: ignore[import-untyped]
import pytest

from uacp.scl.errors import SCLSyntaxError
from uacp.scl.parser import format, parse, parse_file


def test_parse_valid_minimal_document_implicit_root() -> None:
    text = '<identity profile="test_bot">You are a test assistant.</identity>'
    doc = parse(text)
    assert doc.version == "1"
    assert doc.identity.profile == "test_bot"
    assert doc.identity.body == "You are a test assistant."
    assert doc.variables == []
    assert doc.examples == []
    assert doc.output is None


def test_parse_valid_minimal_document_explicit_root() -> None:
    text = (
        '<scl version="1">\n'
        '  <identity profile="test_bot" tone="neutral">\n'
        '    System persona.\n'
        '  </identity>\n'
        '</scl>'
    )
    doc = parse(text)
    assert doc.version == "1"
    assert doc.identity.profile == "test_bot"
    assert doc.identity.tone == "neutral"
    assert "System persona." in doc.identity.body


def test_parse_multiple_examples() -> None:
    text = (
        '<scl>\n'
        '  <identity profile="coder">Coding assistant.</identity>\n'
        '  <example pair="true">\n'
        '    <input>2 + 2</input>\n'
        '    <output>4</output>\n'
        '  </example>\n'
        '  <example pair="false">\n'
        '    <input>3 + 3</input>\n'
        '    <output>6</output>\n'
        '  </example>\n'
        '</scl>'
    )
    doc = parse(text)
    assert len(doc.examples) == 2
    assert doc.examples[0].pair is True
    assert doc.examples[0].input == "2 + 2"
    assert doc.examples[0].output == "4"
    assert doc.examples[1].pair is False
    assert doc.examples[1].input == "3 + 3"
    assert doc.examples[1].output == "6"


def test_parse_crlf_input() -> None:
    text = (
        "<scl>\r\n"
        "  <identity profile=\"auditor\">\r\n"
        "    Audit transactions.\r\n"
        "  </identity>\r\n"
        "  <variable name=\"tx\" source=\"runtime.tx\" type=\"json\">\r\n"
        "    {{tx}}\r\n"
        "  </variable>\r\n"
        "</scl>"
    )
    doc = parse(text)
    assert doc.identity.profile == "auditor"
    assert len(doc.variables) == 1
    assert doc.variables[0].name == "tx"


def test_parse_unicode_text() -> None:
    text = (
        '<scl version="1">\n'
        '  <identity profile="linguist" tone="calm, empathetic">\n'
        '    Multilingual assistant: こんにちは世界, Bonjour le monde, 🚀✨\n'
        '  </identity>\n'
        '</scl>'
    )
    doc = parse(text)
    assert "こんにちは世界" in doc.identity.body
    assert "🚀✨" in doc.identity.body


def test_parse_json_schema_output_nested_brackets() -> None:
    schema = {
        "$schema": "http://json-schema.org/draft-07/schema#",
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "tag <test> with <brackets>"}
        },
        "required": ["query"],
    }
    text = (
        '<scl>\n'
        '  <identity profile="bot">Bot</identity>\n'
        '  <output format="json_schema" strict="true">\n'
        f'    {json.dumps(schema)}\n'
        '  </output>\n'
        '</scl>'
    )
    doc = parse(text)
    assert doc.output is not None
    assert doc.output.format == "json_schema"
    assert doc.output.strict is True
    assert doc.output.schema == schema


def test_financial_auditor_example_file() -> None:
    path = Path("examples/financial_auditor.scl")
    assert path.exists()
    doc = parse_file(path)
    assert doc.identity.profile == "financial_auditor"
    assert doc.identity.tone == "objective, meticulous"
    assert len(doc.variables) == 1
    assert doc.variables[0].name == "transaction_manifest"
    assert doc.variables[0].source == "runtime.payload.ledger_v1"
    assert doc.variables[0].type == "json"

    assert len(doc.examples) == 1
    assert doc.output is not None
    assert doc.output.format == "json_schema"
    assert doc.output.strict is True
    assert doc.output.schema is not None

    # Verify example output satisfies the schema
    example_output_raw = doc.examples[0].output.strip()
    example_data = json.loads(example_output_raw)
    jsonschema.validate(instance=example_data, schema=doc.output.schema)


def test_format_pretty_printer_roundtrip() -> None:
    path = Path("examples/financial_auditor.scl")
    doc1 = parse_file(path)
    formatted = format(doc1)
    doc2 = parse(formatted)
    assert doc2 == doc1


def test_parse_with_store_spans() -> None:
    text = '<identity profile="bot">Hello</identity>'
    doc = parse(text, store_spans=True)
    assert doc.identity.span is not None
    assert doc.identity.span.line == 1
    assert doc.identity.span.col == 1


# ── Snapshot Tests for Invalid Inputs ─────────────────────────────────

INVALID_INPUT_CASES = [
    ("missing_identity_profile", "<identity>You are a bot.</identity>"),
    (
        "missing_variable_name",
        '<identity profile="a">b</identity><variable source="runtime.x">text</variable>',
    ),
    (
        "missing_variable_source",
        '<identity profile="a">b</identity><variable name="v">text</variable>',
    ),
    (
        "invalid_variable_name",
        '<identity profile="a">b</identity>'
        '<variable name="123_bad" source="runtime.x">text</variable>',
    ),
    (
        "invalid_variable_source",
        '<identity profile="a">b</identity>'
        '<variable name="v" source="runtime..bad">text</variable>',
    ),
    (
        "invalid_variable_type",
        '<identity profile="a">b</identity>'
        '<variable name="v" source="runtime.x" type="csv">text</variable>',
    ),
    (
        "missing_output_format",
        '<identity profile="a">b</identity><output>text</output>',
    ),
    (
        "invalid_output_format",
        '<identity profile="a">b</identity><output format="yaml">text</output>',
    ),
    (
        "duplicate_identity",
        '<identity profile="a">first</identity><identity profile="b">second</identity>',
    ),
    (
        "duplicate_output",
        '<identity profile="a">b</identity>'
        '<output format="markdown">first</output><output format="markdown">second</output>',
    ),
    (
        "unclosed_identity_tag",
        '<identity profile="a">unclosed',
    ),
    (
        "unclosed_scl_wrapper",
        '<scl><identity profile="a">b</identity>',
    ),
    (
        "mismatched_tag_output_excel",
        '<identity profile="a">b</identity><output format="markdown">text</excel>',
    ),
    (
        "mismatched_tag_identity_variable",
        '<identity profile="a">b</variable>',
    ),
    (
        "invalid_json_schema_body",
        '<identity profile="a">b</identity>\n'
        '<output format="json_schema">\n{\n  "bad": \n}\n</output>',
    ),
    (
        "json_schema_not_object",
        '<identity profile="a">b</identity><output format="json_schema">"just a string"</output>',
    ),
    (
        "duplicate_attribute",
        '<identity profile="a" profile="b">b</identity>',
    ),
    (
        "unknown_attribute",
        '<identity profile="a" unknown="value">b</identity>',
    ),
    (
        "invalid_boolean_attribute",
        '<identity profile="a">b</identity>'
        '<example pair="yes"><input>in</input><output>out</output></example>',
    ),
    (
        "example_pair_missing_output",
        '<identity profile="a">b</identity><example pair="true"><input>in</input></example>',
    ),
    (
        "missing_identity_element",
        '<variable name="x" source="runtime.x">template</variable>',
    ),
    (
        "unexpected_text_outside",
        '<identity profile="a">b</identity>Stray text',
    ),
]


@pytest.mark.parametrize("case_name, scl_source", INVALID_INPUT_CASES)
def test_invalid_scl_snapshots(case_name: str, scl_source: str, snapshot: object) -> None:
    with pytest.raises(SCLSyntaxError) as exc_info:
        parse(scl_source)
    # Check that error includes line and column coordinates
    assert exc_info.value.line >= 1
    assert exc_info.value.col >= 1
    assert str(exc_info.value) == snapshot
