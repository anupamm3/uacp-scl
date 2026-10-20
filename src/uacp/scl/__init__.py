"""Semantic Context Layer (SCL) package."""

from uacp.scl.ast import (
    Document,
    Example,
    Identity,
    Output,
    SCLNode,
    Span,
    Variable,
    placeholders,
)
from uacp.scl.context import (
    BoundDocument,
    BoundVariable,
    RuntimeContext,
    bind,
    escape_xml_data,
)
from uacp.scl.errors import (
    Diagnostic,
    MissingVariableError,
    SCLError,
    SCLSyntaxError,
    SCLValidationError,
    format_source_caret,
)
from uacp.scl.lexer import Lexer, Token, TokenType, tokenize
from uacp.scl.parser import Parser, format, parse, parse_file
from uacp.scl.validate import validate, validate_strict

__all__ = [
    "BoundDocument",
    "BoundVariable",
    "Diagnostic",
    "Document",
    "Example",
    "Identity",
    "Lexer",
    "MissingVariableError",
    "Output",
    "Parser",
    "RuntimeContext",
    "SCLError",
    "SCLNode",
    "SCLSyntaxError",
    "SCLValidationError",
    "Span",
    "Token",
    "TokenType",
    "Variable",
    "bind",
    "escape_xml_data",
    "format",
    "format_source_caret",
    "parse",
    "parse_file",
    "placeholders",
    "tokenize",
    "validate",
    "validate_strict",
]
