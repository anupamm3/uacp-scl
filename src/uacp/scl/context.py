"""SCL compilation context and variable resolution."""

from __future__ import annotations

import difflib
import json
import re
from typing import Any, Literal

from uacp.scl.ast import Document, Example, Identity, SCLNode, Variable
from uacp.scl.errors import MissingVariableError
from uacp.scl.parser import format as format_doc


def escape_xml_data(value: str) -> str:
    """Escape XML tag delimiters to prevent prompt injection and markup breakout.

    Replaces '<' with '&lt;' and '>' with '&gt;'.
    """
    return value.replace("<", "&lt;").replace(">", "&gt;")


class RuntimeContext:
    """Context holding runtime variables and providing dotted-path resolution."""

    def __init__(self, data: dict[str, Any] | None = None) -> None:
        self.data: dict[str, Any] = data if data is not None else {}

    def _collect_keys(self, current: Any, prefix: str = "") -> list[str]:
        keys: list[str] = []
        if isinstance(current, dict):
            for k, v in current.items():
                full_path = f"{prefix}.{k}" if prefix else str(k)
                keys.append(full_path)
                keys.extend(self._collect_keys(v, full_path))
        return keys

    def resolve(self, path: str) -> Any:
        """Resolve a dotted-path lookup into context data.

        Args:
            path: Dotted key path, e.g. 'runtime.payload.ledger_v1'.

        Returns:
            The resolved value at the specified path.

        Raises:
            MissingVariableError: If path does not exist, listing the closest available keys.
        """
        # Direct key lookup
        if path in self.data:
            return self.data[path]

        # Dotted path traversal
        parts = path.split(".")
        current: Any = self.data
        found = True
        for part in parts:
            if isinstance(current, dict) and part in current:
                current = current[part]
            else:
                found = False
                break

        if found:
            return current

        # Search for closest matches
        all_keys = self._collect_keys(self.data)
        candidates = difflib.get_close_matches(path, all_keys, n=3, cutoff=0.3)
        raise MissingVariableError(name=path, candidates=candidates)


class BoundVariable(SCLNode):
    """A declared SCL variable bound to resolved and serialized runtime data."""

    name: str
    source: str
    type: Literal["json", "text"]
    raw_value: Any
    rendered_value: str


class BoundDocument(SCLNode):
    """An SCL document with all declared variables bound and templates rendered."""

    document: Document
    variables: dict[str, BoundVariable]
    rendered_identity: str
    rendered_examples: list[Example]

    def to_document(self) -> Document:
        """Construct an AST Document with variables and templates substituted."""
        bound_vars = [
            Variable(
                name=var.name,
                source=var.source,
                type=var.type,
                template=self.variables[var.name].rendered_value,
                span=var.span,
            )
            for var in self.document.variables
        ]

        bound_identity = Identity(
            profile=self.document.identity.profile,
            tone=self.document.identity.tone,
            body=self.rendered_identity,
            span=self.document.identity.span,
        )

        return Document(
            version=self.document.version,
            identity=bound_identity,
            variables=bound_vars,
            examples=self.rendered_examples,
            output=self.document.output,
            span=self.document.span,
        )

    def render_scl(self) -> str:
        """Render the bound document back to canonical SCL markup string."""
        return format_doc(self.to_document())


def _substitute_placeholders(text: str, values: dict[str, str]) -> str:
    """Replace all {{placeholder}} tokens with their corresponding rendered values."""
    if not text:
        return text

    def replacer(match: re.Match[str]) -> str:
        var_name = match.group(1).strip()
        return values.get(var_name, match.group(0))

    return re.sub(r"\{\{\s*([a-z_][a-z0-9_]*)\s*\}\}", replacer, text)


def bind(
    doc: Document,
    ctx: RuntimeContext,
    *,
    max_variable_chars: int = 200_000,
    overflow: Literal["error", "truncate"] = "error",
) -> BoundDocument:
    """Resolve and bind runtime variables to an SCL Document AST.

    Args:
        doc: Parsed SCL Document.
        ctx: RuntimeContext holding data payloads.
        max_variable_chars: Max serialized characters permitted per variable.
        overflow: Strategy when variable exceeds max length: 'error' or 'truncate'.

    Returns:
        BoundDocument containing resolved variables and rendered sections.

    Raises:
        ValueError: If overflow strategy is invalid or variable size exceeds limit.
        MissingVariableError: If a variable's source cannot be resolved.
    """
    if overflow not in ("error", "truncate"):
        raise ValueError(
            f"Invalid overflow strategy '{overflow}'. Must be 'error' or 'truncate'."
        )

    # 1. Resolve, coerce, size check, and escape all variables
    raw_values: dict[str, Any] = {}
    escaped_values: dict[str, str] = {}

    for var in doc.variables:
        raw_val = ctx.resolve(var.source)
        raw_values[var.name] = raw_val

        if var.type == "json":
            coerced = json.dumps(raw_val, sort_keys=True, ensure_ascii=False)
        else:
            coerced = str(raw_val)

        if len(coerced) > max_variable_chars:
            if overflow == "error":
                raise ValueError(
                    f"Variable '{var.name}' size ({len(coerced)} chars) exceeds maximum limit "
                    f"of {max_variable_chars} chars."
                )
            coerced = coerced[:max_variable_chars]

        # Security escape injected markup
        escaped_values[var.name] = escape_xml_data(coerced)

    # 2. Render templates for each variable
    bound_variables: dict[str, BoundVariable] = {}
    for var in doc.variables:
        if var.template:
            rendered = _substitute_placeholders(var.template, escaped_values)
        else:
            rendered = escaped_values[var.name]

        bound_variables[var.name] = BoundVariable(
            name=var.name,
            source=var.source,
            type=var.type,
            raw_value=raw_values[var.name],
            rendered_value=rendered,
        )

    # 3. Render placeholders in identity and examples
    rendered_identity = _substitute_placeholders(doc.identity.body, escaped_values)

    rendered_examples: list[Example] = []
    for ex in doc.examples:
        rendered_examples.append(
            Example(
                pair=ex.pair,
                input=_substitute_placeholders(ex.input, escaped_values),
                output=_substitute_placeholders(ex.output, escaped_values),
                span=ex.span,
            )
        )

    return BoundDocument(
        document=doc,
        variables=bound_variables,
        rendered_identity=rendered_identity,
        rendered_examples=rendered_examples,
    )
