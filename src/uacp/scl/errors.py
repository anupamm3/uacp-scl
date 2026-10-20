"""Error types and diagnostic formatting for SCL (Semantic Context Layer)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


def format_source_caret(source: str, line: int, col: int) -> str:
    """Render a source line with a caret pointing to the offending column.

    Args:
        source: Full source text.
        line: 1-based line number.
        col: 1-based column number.

    Returns:
        Formatted multi-line string containing the numbered line and caret pointer.
    """
    lines = source.splitlines()
    if 1 <= line <= len(lines):
        offending_line = lines[line - 1]
        caret_offset = max(0, col - 1)
        caret_line = " " * caret_offset + "^"
        return f"{line:4d} | {offending_line}\n     | {caret_line}"
    return f"Line {line}, Col {col} (out of range)"


@dataclass(frozen=True)
class Diagnostic:
    """A diagnostic message associated with source coordinates."""

    message: str
    line: int
    col: int
    severity: Literal["error", "warning"] = "error"
    hint: str | None = None

    def format_with_source(self, source: str) -> str:
        """Render diagnostic message with source snippet and caret."""
        snippet = format_source_caret(source, self.line, self.col)
        hint_str = f"\nHint: {self.hint}" if self.hint else ""
        return (
            f"[{self.severity.upper()}] at {self.line}:{self.col}: {self.message}\n"
            f"{snippet}{hint_str}"
        )


class SCLError(Exception):
    """Base exception for all SCL parsing, validation, and compilation errors."""


class SCLSyntaxError(SCLError):
    """Raised when SCL markup contains invalid syntax or malformed tokens."""

    def __init__(
        self,
        message: str,
        line: int,
        col: int,
        hint: str | None = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.line = line
        self.col = col
        self.hint = hint

    def format_with_source(self, source: str) -> str:
        """Render a friendly message with the offending source line and caret."""
        snippet = format_source_caret(source, self.line, self.col)
        hint_str = f"\nHint: {self.hint}" if self.hint else ""
        return f"Syntax Error at {self.line}:{self.col}: {self.message}\n{snippet}{hint_str}"

    def __str__(self) -> str:
        hint_str = f" (hint: {self.hint})" if self.hint else ""
        return f"Line {self.line}:{self.col}: {self.message}{hint_str}"


class SCLValidationError(SCLError):
    """Raised when SCL document violates semantic or schema constraints."""

    def __init__(self, diagnostics: list[Diagnostic]) -> None:
        self.diagnostics = diagnostics
        summary = f"{len(diagnostics)} validation error(s)"
        super().__init__(summary)

    def format_with_source(self, source: str) -> str:
        """Render all diagnostics with source snippets and carets."""
        return "\n\n".join(d.format_with_source(source) for d in self.diagnostics)

    def __str__(self) -> str:
        if not self.diagnostics:
            return "Validation failed with no diagnostics."
        details = "; ".join(f"{d.line}:{d.col}: {d.message}" for d in self.diagnostics)
        return f"{len(self.diagnostics)} validation error(s): {details}"


class MissingVariableError(SCLError):
    """Raised when a template references an undeclared variable or context key."""

    def __init__(
        self,
        name: str,
        line: int | None = None,
        col: int | None = None,
        candidates: list[str] | None = None,
    ) -> None:
        self.name = name
        self.line = line
        self.col = col
        self.candidates = candidates or []
        loc = f" at {line}:{col}" if line is not None and col is not None else ""
        hint = f" (did you mean: {', '.join(self.candidates)}?)" if self.candidates else ""
        super().__init__(f"Undeclared variable '{name}' referenced{loc}{hint}")

    def format_with_source(self, source: str) -> str:
        """Render friendly message if coordinates are available."""
        if self.line is not None and self.col is not None:
            snippet = format_source_caret(source, self.line, self.col)
            hint_str = (
                f"\nHint: Did you mean: {', '.join(self.candidates)}?"
                if self.candidates
                else ""
            )
            return (
                f"MissingVariableError: Variable '{self.name}' was referenced but not declared.\n"
                f"{snippet}{hint_str}"
            )
        return str(self)
