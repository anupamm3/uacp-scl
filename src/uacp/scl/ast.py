"""AST (Abstract Syntax Tree) models for SCL (Semantic Context Layer)."""

from __future__ import annotations

import re
import warnings
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

_PLACEHOLDER_REGEX = re.compile(r"\{\{\s*([a-z_][a-z0-9_]*)\s*\}\}")


def placeholders(text: str) -> list[str]:
    """Extract unique variable placeholder names from a template string.

    Variable placeholders follow the syntax `{{name}}`, where `name` conforms
    to the identifier regex `[a-z_][a-z0-9_]*`.

    Args:
        text: Input text containing potential placeholders.

    Returns:
        Deduplicated list of extracted placeholder names in order of appearance.
    """
    matches = _PLACEHOLDER_REGEX.findall(text)
    return list(dict.fromkeys(matches))


class SCLNode(BaseModel):
    """Base class for all SCL AST nodes."""

    model_config = ConfigDict(frozen=True, extra="forbid")


class Span(SCLNode):
    """Source text coordinate span for an AST element."""

    line: int
    col: int
    end_line: int
    end_col: int


class Identity(SCLNode):
    """Agent identity persona and communication style."""

    profile: str
    tone: str | None = None
    body: str
    span: Span | None = None


class Variable(SCLNode):
    """Runtime variable declaration and binding template."""

    name: str
    source: str
    type: Literal["json", "text"] = "text"
    template: str | None = None
    span: Span | None = None


class Example(SCLNode):
    """Few-shot input/output demonstration pair."""

    pair: bool = True
    input: str
    output: str
    span: Span | None = None


with warnings.catch_warnings():
    warnings.filterwarnings(
        "ignore",
        message=r'Field name "schema" in "Output" shadows an attribute in parent',
        category=UserWarning,
    )

    class Output(SCLNode):
        """Output format requirements and optional structured schema."""

        format: Literal["json_schema", "markdown"]
        strict: bool = False
        body: str
        schema: dict[str, Any] | None = None  # type: ignore[assignment]
        span: Span | None = None


class Document(SCLNode):
    """Root node of a parsed SCL specification."""

    version: str | None = "1"
    identity: Identity
    variables: list[Variable] = Field(default_factory=list)
    examples: list[Example] = Field(default_factory=list)
    output: Output | None = None
    span: Span | None = None

    def to_dict(self) -> dict[str, Any]:
        """Serialize AST document to a Python dictionary."""
        return self.model_dump(by_alias=True)
