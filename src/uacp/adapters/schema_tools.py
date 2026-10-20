"""Schema conversion and extraction utilities for adapter payloads."""

from __future__ import annotations

import copy
import json
import re
from typing import Any

# Table-driven keyword exclusion lists.
# Verified against OpenAI Structured Outputs official documentation (strict: true):
# Rejects pattern, minLength, maxLength, minimum, maximum, multipleOf, format,
# patternProperties, unevaluatedProperties, propertyNames, minProperties, maxProperties,
# unevaluatedItems, contains, minContains, maxContains, minItems, maxItems, uniqueItems,
# default, $schema, and $id.
OPENAI_STRICT_UNSUPPORTED_KEYWORDS: frozenset[str] = frozenset(
    {
        "$schema",
        "$id",
        "minLength",
        "maxLength",
        "pattern",
        "format",
        "minimum",
        "maximum",
        "exclusiveMinimum",
        "exclusiveMaximum",
        "multipleOf",
        "patternProperties",
        "unevaluatedProperties",
        "propertyNames",
        "minProperties",
        "maxProperties",
        "unevaluatedItems",
        "contains",
        "minContains",
        "maxContains",
        "minItems",
        "maxItems",
        "uniqueItems",
        "default",
    }
)

# Verified against Google Gemini API / OpenAPI 3.0 Schema official documentation:
# Rejects $schema, $id, patternProperties, dependentRequired, dependentSchemas,
# if, then, else, readOnly, writeOnly, unevaluatedProperties, unevaluatedItems,
# contains, minContains, maxContains, propertyNames, minProperties, maxProperties.
GEMINI_UNSUPPORTED_KEYWORDS: frozenset[str] = frozenset(
    {
        "$schema",
        "$id",
        "patternProperties",
        "dependentRequired",
        "dependentSchemas",
        "if",
        "then",
        "else",
        "readOnly",
        "writeOnly",
        "unevaluatedProperties",
        "unevaluatedItems",
        "contains",
        "minContains",
        "maxContains",
        "propertyNames",
        "minProperties",
        "maxProperties",
    }
)

_CODE_FENCE_REGEX = re.compile(r"```(?:json)?\s*([\s\S]*?)```", re.IGNORECASE)


def _clean_lenient_json(raw: str) -> str:
    """Clean common LLM JSON syntax issues such as comments and trailing commas."""
    # Remove single-line comments
    cleaned = re.sub(r"//.*$", "", raw, flags=re.MULTILINE)
    # Remove multi-line comments
    cleaned = re.sub(r"/\*[\s\S]*?\*/", "", cleaned)
    # Remove trailing commas before closing braces/brackets
    cleaned = re.sub(r",\s*([}\]])", r"\1", cleaned)
    return cleaned


def _find_balanced_json(text: str, lenient: bool = False) -> Any:
    """Find and parse the first balanced JSON object or array in text.

    Correctly tracks quotes and backslash escapes to prevent braces or brackets
    inside string literals from corrupting depth counts.
    """
    length = len(text)
    idx = 0

    while idx < length:
        char = text[idx]
        if char not in ("{", "["):
            idx += 1
            continue

        start_idx = idx
        stack: list[str] = []
        in_string = False
        escape = False
        matched = False

        scan_idx = start_idx
        while scan_idx < length:
            c = text[scan_idx]
            if in_string:
                if escape:
                    escape = False
                elif c == "\\":
                    escape = True
                elif c == '"':
                    in_string = False
            else:
                if c == '"':
                    in_string = True
                elif c in ("{", "["):
                    stack.append(c)
                elif c == "}":
                    if not stack or stack[-1] != "{":
                        break
                    stack.pop()
                    if not stack:
                        matched = True
                        break
                elif c == "]":
                    if not stack or stack[-1] != "[":
                        break
                    stack.pop()
                    if not stack:
                        matched = True
                        break
            scan_idx += 1

        if matched:
            candidate = text[start_idx : scan_idx + 1]
            try:
                return json.loads(candidate)
            except json.JSONDecodeError:
                if lenient:
                    try:
                        return json.loads(_clean_lenient_json(candidate))
                    except json.JSONDecodeError:
                        pass
        idx += 1

    return None


def extract_json(text: str, lenient: bool = False) -> Any:
    """Extract and parse the first balanced JSON object or array from text.

    Strips markdown code fences if present and correctly handles nested braces
    and brackets inside string literals.

    Args:
        text: String containing JSON, potentially surrounded by markdown fences or prose.
        lenient: If True, attempts cleanup of trailing commas and comments.

    Returns:
        Parsed JSON object (dict or list).

    Raises:
        ValueError: If no valid balanced JSON object or array can be parsed.
    """
    # 1. If markdown code fences exist, try extracting from inside them first
    for match in _CODE_FENCE_REGEX.finditer(text):
        fence_content = match.group(1).strip()
        try:
            return json.loads(fence_content)
        except json.JSONDecodeError:
            result = _find_balanced_json(fence_content, lenient=lenient)
            if result is not None:
                return result

    # 2. Search entire text for first balanced JSON object or array
    result = _find_balanced_json(text, lenient=lenient)
    if result is not None:
        return result

    # 3. If lenient, attempt whole-text cleanup as a final fallback
    if lenient:
        try:
            return json.loads(_clean_lenient_json(text.strip()))
        except json.JSONDecodeError:
            pass

    raise ValueError("No valid JSON object or array found in text")


def extract_xml_block(text: str, tag: str) -> str | None:
    """Return the inner text of an XML block <tag>...</tag>.

    Args:
        text: Input string containing XML tags.
        tag: Tag name to locate (e.g. 'identity', 'output').

    Returns:
        Inner content of the tag, or None if the tag is not present.
    """
    pattern = re.compile(rf"<{re.escape(tag)}(?:\s+[^>]*)?>(.*?)</{re.escape(tag)}>", re.DOTALL)
    match = pattern.search(text)
    if match:
        return match.group(1)
    return None


def _normalize_openai_strict_node(node: dict[str, Any]) -> None:
    """Recursively adapt a schema node for OpenAI strict structured outputs."""
    # Strip unsupported keywords
    for kw in OPENAI_STRICT_UNSUPPORTED_KEYWORDS:
        node.pop(kw, None)

    # Objects in strict mode require additionalProperties: false and all properties required
    is_object = node.get("type") == "object" or "properties" in node
    if is_object:
        node["additionalProperties"] = False
        if "properties" in node and isinstance(node["properties"], dict):
            existing_required = set(node.get("required", []))
            all_props = list(node["properties"].keys())

            # Convert optional properties to nullable unions
            for prop_name, prop_schema in node["properties"].items():
                if isinstance(prop_schema, dict) and prop_name not in existing_required:
                    # Check if already a nullable union
                    if "anyOf" in prop_schema and isinstance(prop_schema["anyOf"], list):
                        has_null = any(
                            isinstance(item, dict) and item.get("type") == "null"
                            for item in prop_schema["anyOf"]
                        )
                        if not has_null:
                            prop_schema["anyOf"].append({"type": "null"})
                    else:
                        node["properties"][prop_name] = {
                            "anyOf": [copy.deepcopy(prop_schema), {"type": "null"}]
                        }

            # All properties must now be listed in required
            node["required"] = all_props


def _normalize_gemini_node(node: dict[str, Any]) -> None:
    """Recursively strip keywords unsupported by Google Gemini."""
    for kw in GEMINI_UNSUPPORTED_KEYWORDS:
        node.pop(kw, None)


def _walk_and_normalize(schema: Any, provider_style: str) -> None:
    """Traverse and normalize all schema nodes recursively."""
    if not isinstance(schema, dict):
        return

    if provider_style == "openai_strict":
        _normalize_openai_strict_node(schema)
    elif provider_style == "gemini":
        _normalize_gemini_node(schema)

    # Recurse into known nested schema containers
    if "properties" in schema and isinstance(schema["properties"], dict):
        for child in schema["properties"].values():
            _walk_and_normalize(child, provider_style)

    if "items" in schema:
        if isinstance(schema["items"], dict):
            _walk_and_normalize(schema["items"], provider_style)
        elif isinstance(schema["items"], list):
            for child in schema["items"]:
                _walk_and_normalize(child, provider_style)

    for union_key in ("anyOf", "allOf", "oneOf"):
        if union_key in schema and isinstance(schema[union_key], list):
            for child in schema[union_key]:
                _walk_and_normalize(child, provider_style)

    for defs_key in ("$defs", "definitions"):
        if defs_key in schema and isinstance(schema[defs_key], dict):
            for child in schema[defs_key].values():
                _walk_and_normalize(child, provider_style)

    if "additionalProperties" in schema and isinstance(schema["additionalProperties"], dict):
        _walk_and_normalize(schema["additionalProperties"], provider_style)

    if "prefixItems" in schema and isinstance(schema["prefixItems"], list):
        for child in schema["prefixItems"]:
            _walk_and_normalize(child, provider_style)


def normalize_schema_for(provider_style: str, schema: dict[str, Any]) -> dict[str, Any]:
    """Normalize a JSON schema according to provider-specific constraints.

    Verification details:
    - "openai_strict": Verified against official OpenAI Structured Outputs docs.
      Requires additionalProperties: false on all objects, all properties must be in
      required with optional fields represented as anyOf nullable unions, and rejects
      constraints like minLength, maxLength, pattern, minimum, maximum, format, etc.
    - "gemini": Verified against Google GenAI / OpenAPI 3.0 subset docs. Rejects $schema,
      $id, patternProperties, dependentRequired, dependentSchemas, if/then/else, etc.
    - "plain": Passes through schema unchanged.

    Args:
        provider_style: Target style ("openai_strict", "gemini", or "plain").
        schema: Input JSON schema dictionary.

    Returns:
        Deep copy of the adapted JSON schema.
    """
    adapted = copy.deepcopy(schema)
    if provider_style == "plain":
        return adapted

    _walk_and_normalize(adapted, provider_style)
    return adapted
