"""Tests for adapter schema tools, output contract enforcer, and repair loop."""

from __future__ import annotations

from typing import Any

import hypothesis.strategies as st
import pytest
from hypothesis import given

from uacp.adapters.enforcer import (
    LLMResult,
    OutputViolation,
    enforce,
    invoke_with_repair,
)
from uacp.adapters.schema_tools import (
    extract_json,
    extract_xml_block,
    normalize_schema_for,
)
from uacp.scl.ast import Output

# ───────────────────────────────────────────────────────────────────
# Fake Adapter for Repair Loop Testing
# ───────────────────────────────────────────────────────────────────


class SequenceAdapter:
    """Fake adapter that returns predefined responses on successive invoke calls."""

    def __init__(self, responses: list[str]) -> None:
        self.responses = list(responses)
        self.call_count = 0
        self.received_requests: list[Any] = []

    async def invoke(self, request: Any) -> LLMResult:
        self.call_count += 1
        self.received_requests.append(request)
        if not self.responses:
            raise RuntimeError("No more responses queued in fake adapter")
        return LLMResult(content=self.responses.pop(0))


# ───────────────────────────────────────────────────────────────────
# Tests: extract_json
# ───────────────────────────────────────────────────────────────────


def test_extract_json_fenced_with_language() -> None:
    text = """
Here is the JSON you requested:
```json
{
  "status": "success",
  "items": [1, 2, 3]
}
```
Let me know if you need more.
"""
    result = extract_json(text)
    assert result == {"status": "success", "items": [1, 2, 3]}


def test_extract_json_fenced_without_language() -> None:
    text = """
```
[{"id": "a"}, {"id": "b"}]
```
"""
    result = extract_json(text)
    assert result == [{"id": "a"}, {"id": "b"}]


def test_extract_json_trailing_prose() -> None:
    text = '{"name": "Alice", "role": "admin"} Note: This user has escalated privileges.'
    result = extract_json(text)
    assert result == {"name": "Alice", "role": "admin"}


def test_extract_json_leading_and_trailing_prose() -> None:
    text = 'Result of calculation: {"total": 42.5} (Calculated with 2 decimals).'
    result = extract_json(text)
    assert result == {"total": 42.5}


def test_extract_json_nested_braces_in_strings() -> None:
    text = '{"message": "Hello {user}, formatting {foo: bar} here", "count": 2}'
    result = extract_json(text)
    assert result["message"] == "Hello {user}, formatting {foo: bar} here"
    assert result["count"] == 2


def test_extract_json_escaped_quotes_and_brackets_in_strings() -> None:
    text = r'{"template": "Pattern: [a-z]+ \"nested {1}\"", "valid": true}'
    result = extract_json(text)
    assert result["template"] == 'Pattern: [a-z]+ "nested {1}"'
    assert result["valid"] is True


def test_extract_json_lenient_trailing_comma() -> None:
    text = """
    {
      "key1": "val1",
      "key2": "val2",
    }
    """
    with pytest.raises(ValueError):
        extract_json(text, lenient=False)

    lenient_result = extract_json(text, lenient=True)
    assert lenient_result == {"key1": "val1", "key2": "val2"}


def test_extract_json_no_json_raises() -> None:
    with pytest.raises(ValueError, match="No valid JSON"):
        extract_json("Just plain text with no brackets anywhere.")


# ───────────────────────────────────────────────────────────────────
# Tests: extract_xml_block
# ───────────────────────────────────────────────────────────────────


def test_extract_xml_block_basic() -> None:
    text = '<identity profile="tester">You are a tester.</identity>'
    assert extract_xml_block(text, "identity") == "You are a tester."


def test_extract_xml_block_multiline() -> None:
    text = '<output format="markdown">\n# Header\nBody\n</output>'
    assert extract_xml_block(text, "output") == "\n# Header\nBody\n"


def test_extract_xml_block_missing() -> None:
    text = "<identity>Agent</identity>"
    assert extract_xml_block(text, "variable") is None


# ───────────────────────────────────────────────────────────────────
# Tests: enforce
# ───────────────────────────────────────────────────────────────────


def test_enforce_json_schema_valid() -> None:
    schema = {
        "type": "object",
        "required": ["id", "score"],
        "properties": {
            "id": {"type": "string"},
            "score": {"type": "number"},
        },
    }
    node = Output(format="json_schema", body="", schema=schema)
    raw = '{"id": "doc_1", "score": 0.95}'
    result = enforce(node, raw)
    assert result == {"id": "doc_1", "score": 0.95}


def test_enforce_json_schema_violation_missing_required() -> None:
    schema = {
        "type": "object",
        "required": ["id", "score"],
        "properties": {
            "id": {"type": "string"},
            "score": {"type": "number"},
        },
    }
    node = Output(format="json_schema", body="", schema=schema)
    raw = '{"id": "doc_1"}'

    with pytest.raises(OutputViolation) as exc_info:
        enforce(node, raw)

    assert any("'score' is a required property" in err for err in exc_info.value.errors)


def test_enforce_json_schema_violation_invalid_type() -> None:
    schema = {
        "type": "object",
        "required": ["score"],
        "properties": {
            "score": {"type": "number"},
        },
    }
    node = Output(format="json_schema", body="", schema=schema)
    raw = '{"score": "not_a_number"}'

    with pytest.raises(OutputViolation) as exc_info:
        enforce(node, raw)

    assert any("is not of type 'number'" in err for err in exc_info.value.errors)


def test_enforce_markdown_strict_valid() -> None:
    node = Output(format="markdown", strict=True, body="")
    raw = "# Summary\nAll checks passed."
    assert enforce(node, raw) == raw


def test_enforce_markdown_strict_empty_raises() -> None:
    node = Output(format="markdown", strict=True, body="")
    with pytest.raises(OutputViolation, match="Output cannot be empty"):
        enforce(node, "   \n\t  ")


# ───────────────────────────────────────────────────────────────────
# Tests: invoke_with_repair
# ───────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_invoke_with_repair_success_first_try() -> None:
    schema = {
        "type": "object",
        "required": ["count"],
        "properties": {"count": {"type": "integer"}},
    }
    node = Output(format="json_schema", body="", schema=schema)

    adapter = SequenceAdapter(['{"count": 10}'])
    request = {"messages": [{"role": "user", "content": "Compute count"}]}

    result = await invoke_with_repair(adapter, request, node, max_repairs=2)
    assert result == {"count": 10}
    assert adapter.call_count == 1
    assert len(request["messages"]) == 1


@pytest.mark.asyncio
async def test_invoke_with_repair_success_second_try() -> None:
    schema = {
        "type": "object",
        "required": ["count"],
        "properties": {"count": {"type": "integer"}},
    }
    node = Output(format="json_schema", body="", schema=schema)

    # First attempt: invalid type (string instead of integer)
    # Second attempt: valid integer
    adapter = SequenceAdapter(['{"count": "invalid"}', '{"count": 42}'])
    request = {"messages": [{"role": "user", "content": "Compute count"}]}

    result = await invoke_with_repair(adapter, request, node, max_repairs=2)
    assert result == {"count": 42}
    assert adapter.call_count == 2
    # Check that bad assistant response and user repair messages were appended
    assert len(request["messages"]) == 3
    assert request["messages"][1]["role"] == "assistant"
    assert request["messages"][1]["content"] == '{"count": "invalid"}'
    assert request["messages"][2]["role"] == "user"
    assert "corrected JSON" in request["messages"][2]["content"]


@pytest.mark.asyncio
async def test_invoke_with_repair_exhaustion() -> None:
    schema = {
        "type": "object",
        "required": ["count"],
        "properties": {"count": {"type": "integer"}},
    }
    node = Output(format="json_schema", body="", schema=schema)

    # All attempts return invalid data
    adapter = SequenceAdapter(
        [
            '{"bad": 1}',
            '{"bad": 2}',
            '{"bad": 3}',
        ]
    )
    request = {"messages": [{"role": "user", "content": "Compute count"}]}

    with pytest.raises(OutputViolation):
        await invoke_with_repair(adapter, request, node, max_repairs=2)

    # 1 initial call + 2 repairs = 3 total calls
    assert adapter.call_count == 3


# ───────────────────────────────────────────────────────────────────
# Tests: normalize_schema_for
# ───────────────────────────────────────────────────────────────────


def test_normalize_schema_plain() -> None:
    schema = {
        "$schema": "http://json-schema.org/draft-07/schema#",
        "type": "object",
        "properties": {"name": {"type": "string", "minLength": 3}},
    }
    normalized = normalize_schema_for("plain", schema)
    assert normalized == schema
    assert normalized is not schema  # Must be deep copy


def test_normalize_schema_openai_strict() -> None:
    schema = {
        "$schema": "http://json-schema.org/draft-07/schema#",
        "type": "object",
        "properties": {
            "id": {"type": "string", "pattern": "^[0-9]+$"},
            "optional_desc": {"type": "string", "maxLength": 100},
        },
        "required": ["id"],
    }
    normalized = normalize_schema_for("openai_strict", schema)

    # Unsupported keywords removed
    assert "$schema" not in normalized
    assert "pattern" not in normalized["properties"]["id"]
    assert "maxLength" not in normalized["properties"]["optional_desc"]["anyOf"][0]

    # Objects have additionalProperties: False
    assert normalized["additionalProperties"] is False

    # Optional properties converted to anyOf nullable unions
    assert normalized["properties"]["optional_desc"] == {
        "anyOf": [{"type": "string"}, {"type": "null"}]
    }

    # All properties are in required
    assert sorted(normalized["required"]) == ["id", "optional_desc"]


def test_normalize_schema_gemini() -> None:
    schema = {
        "$schema": "http://json-schema.org/draft-07/schema#",
        "$id": "https://example.com/schema.json",
        "type": "object",
        "patternProperties": {"^s_": {"type": "string"}},
        "properties": {
            "name": {"type": "string", "description": "User name"},
        },
        "required": ["name"],
    }
    normalized = normalize_schema_for("gemini", schema)

    assert "$schema" not in normalized
    assert "$id" not in normalized
    assert "patternProperties" not in normalized
    assert normalized["properties"]["name"]["description"] == "User name"
    assert normalized["required"] == ["name"]


# ───────────────────────────────────────────────────────────────────
# Hypothesis Property Test: Idempotence of normalize_schema_for
# ───────────────────────────────────────────────────────────────────

# Recursive strategy to generate arbitrary JSON Schemas
schema_primitives = st.sampled_from(["string", "integer", "number", "boolean", "null"])

json_schema_strategy = st.recursive(
    st.fixed_dictionaries(
        {},
        optional={
            "type": schema_primitives,
            "description": st.text(max_size=20),
            "minLength": st.integers(min_value=0, max_value=50),
            "maxLength": st.integers(min_value=50, max_value=100),
            "pattern": st.just("^[a-z]+$"),
            "minimum": st.integers(min_value=0, max_value=10),
            "default": st.text(max_size=10),
            "$schema": st.just("http://json-schema.org/draft-07/schema#"),
        },
    ),
    lambda children: st.fixed_dictionaries(
        {"type": st.just("object")},
        optional={
            "properties": st.dictionaries(
                st.text(min_size=1, max_size=8, alphabet="abcdefghijklmnopqrstuvwxyz"),
                children,
                max_size=3,
            ),
            "required": st.lists(
                st.text(min_size=1, max_size=8, alphabet="abcdefghijklmnopqrstuvwxyz"),
                max_size=3,
            ),
            "additionalProperties": st.booleans(),
        },
    ),
    max_leaves=10,
)


@given(
    style=st.sampled_from(["openai_strict", "gemini", "plain"]),
    schema=json_schema_strategy,
)
def test_normalize_schema_idempotence(style: str, schema: dict[str, Any]) -> None:
    first_pass = normalize_schema_for(style, schema)
    second_pass = normalize_schema_for(style, first_pass)
    assert first_pass == second_pass
