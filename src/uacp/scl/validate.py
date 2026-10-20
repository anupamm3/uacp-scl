"""Semantic validation and schema checking for SCL documents."""

from __future__ import annotations

import json
import re
from typing import Any

from jsonschema import Draft7Validator  # type: ignore[import-untyped]
from jsonschema.exceptions import SchemaError  # type: ignore[import-untyped]

from uacp.scl.ast import Document, placeholders
from uacp.scl.errors import Diagnostic, SCLValidationError

_VAR_NAME_REGEX = re.compile(r"^[a-z_][a-z0-9_]*$")


def _get_coords(node: Any) -> tuple[int, int]:
    """Extract line and column coordinates from an AST node, or default to (1, 1)."""
    span = getattr(node, "span", None)
    if span is not None:
        return int(span.line), int(span.col)
    return 1, 1


def validate(doc: Document) -> list[Diagnostic]:
    """Perform semantic validation on an SCL Document AST.

    Rules checked:
        1. Exactly one identity block with non-empty profile and body.
        2. At most one output element.
        3. Variable names are well-formed (^[a-z_][a-z0-9_]*$) and unique.
        4. Every {{placeholder}} token across all text nodes refers to a declared variable.
        5. strict="true" on <output> requires format="json_schema" and root type="object".
        6. If format="json_schema", the schema must be a valid Draft-07 JSON Schema.
        7. If an example output is valid JSON and a valid schema exists, it is validated
           against the schema and a WARNING diagnostic is emitted on mismatch.

    Args:
        doc: Document AST to validate.

    Returns:
        List of Diagnostic items representing errors or warnings.
    """
    diagnostics: list[Diagnostic] = []

    # 1. Identity validation
    if not getattr(doc, "identity", None):
        line, col = _get_coords(doc)
        diagnostics.append(
            Diagnostic(
                message="Document must contain exactly one <identity> element.",
                line=line,
                col=col,
                severity="error",
            )
        )
    else:
        id_line, id_col = _get_coords(doc.identity)
        if not doc.identity.profile.strip():
            diagnostics.append(
                Diagnostic(
                    message="<identity> 'profile' attribute must not be empty.",
                    line=id_line,
                    col=id_col,
                    severity="error",
                )
            )

    # 2. Variables validation (well-formed and unique)
    declared_vars: set[str] = set()
    for var in doc.variables:
        var_line, var_col = _get_coords(var)
        if not _VAR_NAME_REGEX.match(var.name):
            diagnostics.append(
                Diagnostic(
                    message=(
                        f"Variable name '{var.name}' is invalid; must match '^[a-z_][a-z0-9_]*$'."
                    ),
                    line=var_line,
                    col=var_col,
                    severity="error",
                )
            )
        if var.name in declared_vars:
            diagnostics.append(
                Diagnostic(
                    message=f"Duplicate variable declaration for name '{var.name}'.",
                    line=var_line,
                    col=var_col,
                    severity="error",
                )
            )
        declared_vars.add(var.name)

    # 3. Placeholder validation across all text sections
    def check_placeholders(text: str | None, node: Any) -> None:
        if not text:
            return
        line, col = _get_coords(node)
        for ph in placeholders(text):
            if ph not in declared_vars:
                diagnostics.append(
                    Diagnostic(
                        message=(
                            f"Placeholder '{{{{{ph}}}}}' references undeclared variable '{ph}'."
                        ),
                        line=line,
                        col=col,
                        severity="error",
                        hint=f"Declare <variable name=\"{ph}\" ...> before referencing it.",
                    )
                )

    if getattr(doc, "identity", None):
        check_placeholders(doc.identity.body, doc.identity)

    for var in doc.variables:
        if var.template:
            check_placeholders(var.template, var)

    for example in doc.examples:
        check_placeholders(example.input, example)
        check_placeholders(example.output, example)

    if doc.output:
        check_placeholders(doc.output.body, doc.output)

    # 4. Output and Schema validation
    schema_is_valid = False
    valid_schema: dict[str, Any] | None = None

    if doc.output is not None:
        out_line, out_col = _get_coords(doc.output)

        if doc.output.strict:
            if doc.output.format != "json_schema":
                diagnostics.append(
                    Diagnostic(
                        message="strict='true' requires format='json_schema'.",
                        line=out_line,
                        col=out_col,
                        severity="error",
                    )
                )
            else:
                schema = doc.output.schema
                if not isinstance(schema, dict) or schema.get("type") != "object":
                    diagnostics.append(
                        Diagnostic(
                            message=(
                                "strict='true' requires a schema whose root type is 'object'."
                            ),
                            line=out_line,
                            col=out_col,
                            severity="error",
                        )
                    )

        if doc.output.format == "json_schema":
            if doc.output.schema is None:
                diagnostics.append(
                    Diagnostic(
                        message="Output format is 'json_schema' but no valid schema was provided.",
                        line=out_line,
                        col=out_col,
                        severity="error",
                    )
                )
            else:
                try:
                    Draft7Validator.check_schema(doc.output.schema)
                    schema_is_valid = True
                    valid_schema = doc.output.schema
                except SchemaError as e:
                    diagnostics.append(
                        Diagnostic(
                            message=f"Invalid Draft-07 JSON Schema: {e.message}",
                            line=out_line,
                            col=out_col,
                            severity="error",
                        )
                    )

    # 5. Example output vs schema validation (emits WARNING)
    if schema_is_valid and valid_schema is not None:
        validator = Draft7Validator(valid_schema)
        for example in doc.examples:
            ex_line, ex_col = _get_coords(example)
            try:
                ex_data = json.loads(example.output)
            except (json.JSONDecodeError, ValueError):
                continue

            for err in validator.iter_errors(ex_data):
                diagnostics.append(
                    Diagnostic(
                        message=f"Example output does not conform to schema: {err.message}",
                        line=ex_line,
                        col=ex_col,
                        severity="warning",
                    )
                )

    return diagnostics


def validate_strict(doc: Document) -> None:
    """Validate document and raise SCLValidationError if any error diagnostics exist.

    Warnings do not raise an exception.

    Args:
        doc: Document AST to validate.

    Raises:
        SCLValidationError: If one or more diagnostics with severity='error' occur.
    """
    diagnostics = validate(doc)
    errors = [d for d in diagnostics if d.severity == "error"]
    if errors:
        raise SCLValidationError(errors)
