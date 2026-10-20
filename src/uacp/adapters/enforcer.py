"""Output contract validation and repair enforcer for LLM adapters."""

from __future__ import annotations

import json
from typing import Any, Protocol, cast, runtime_checkable

from jsonschema import Draft7Validator  # type: ignore[import-untyped]

from uacp.adapters.base import Adapter, LLMResult
from uacp.adapters.schema_tools import extract_json
from uacp.scl.ast import Output


class OutputViolation(Exception):  # noqa: N818
    """Raised when LLM output violates the required SCL output contract."""

    def __init__(self, errors: list[str]) -> None:
        self.errors = errors
        summary = "; ".join(errors) if errors else "Output contract violation"
        super().__init__(summary)

    def __str__(self) -> str:
        return f"OutputViolation: {'; '.join(self.errors)}"


@runtime_checkable
class LLMAdapter(Protocol):
    """Protocol for LLM adapter invocations."""

    async def invoke(self, request: Any) -> LLMResult:
        """Invoke the LLM with a request payload."""
        ...


__all__ = [
    "Adapter",
    "LLMAdapter",
    "LLMResult",
    "OutputViolation",
    "enforce",
    "invoke_with_repair",
]


def enforce(output_node: Output, raw: str) -> dict[str, Any] | list[Any] | str:
    """Validate raw model output against an SCL Output specification.

    Args:
        output_node: SCL Output AST node containing format, strict, and schema.
        raw: Raw textual output returned by an LLM.

    Returns:
        Parsed JSON dictionary/list for 'json_schema' format, or string for 'markdown'.

    Raises:
        OutputViolation: If the output does not conform to the schema or constraints.
    """
    if output_node.format == "json_schema":
        try:
            parsed = extract_json(raw)
        except ValueError as exc:
            raise OutputViolation([f"JSON extraction failed: {exc}"]) from exc

        # Determine target schema from AST node
        schema = output_node.schema
        if schema is None and output_node.body.strip():
            try:
                schema = json.loads(output_node.body)
            except json.JSONDecodeError as exc:
                raise OutputViolation([f"Invalid schema in output node: {exc}"]) from exc

        if schema is not None:
            validator = Draft7Validator(schema)
            validation_errors = [
                f"{err.json_path or 'root'}: {err.message}"
                for err in validator.iter_errors(parsed)
            ]
            if validation_errors:
                raise OutputViolation(validation_errors)

        return cast(dict[str, Any] | list[Any], parsed)

    if output_node.format == "markdown":
        if output_node.strict and not raw.strip():
            raise OutputViolation(["Output cannot be empty"])
        return raw

    raise OutputViolation([f"Unsupported output format: {output_node.format}"])


def _append_repair_messages(request: Any, bad_answer: str, error_messages: list[str]) -> None:
    """Append bad response and corrective message to request message history."""
    errors_formatted = "\n".join(f"- {e}" for e in error_messages)
    corrective_text = (
        f"Your previous response violated the required schema:\n{errors_formatted}\n"
        "Please output ONLY the corrected JSON matching the required schema, without commentary."
    )

    bad_msg = {"role": "assistant", "content": bad_answer}
    repair_msg = {"role": "user", "content": corrective_text}

    if hasattr(request, "messages") and isinstance(request.messages, list):
        request.messages.append(bad_msg)
        request.messages.append(repair_msg)
    elif (
        isinstance(request, dict)
        and "messages" in request
        and isinstance(request["messages"], list)
    ):
        request["messages"].append(bad_msg)
        request["messages"].append(repair_msg)
    elif isinstance(request, list):
        request.append(bad_msg)
        request.append(repair_msg)


async def invoke_with_repair(
    adapter: Adapter | LLMAdapter,
    request: Any,
    output_node: Output,
    max_repairs: int = 2,
) -> dict[str, Any] | list[Any] | str:
    """Invoke adapter and automatically attempt corrective repairs on output contract violations.

    Args:
        adapter: LLM adapter adhering to LLMAdapter protocol.
        request: Request payload (e.g. object or dict with 'messages' list).
        output_node: Expected output contract.
        max_repairs: Maximum number of corrective repair attempts before raising.

    Returns:
        Enforced output (parsed dict/list or string).

    Raises:
        OutputViolation: When output fails enforcement after max_repairs attempts.
    """
    repairs_attempted = 0

    while True:
        result = await adapter.invoke(request)
        raw_content = getattr(result, "content", None)
        raw_text: str = raw_content if isinstance(raw_content, str) else str(result)

        try:
            return enforce(output_node, raw_text)
        except OutputViolation as violation:
            if repairs_attempted >= max_repairs:
                raise violation

            repairs_attempted += 1
            _append_repair_messages(request, raw_text, violation.errors)
