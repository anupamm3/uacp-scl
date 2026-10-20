"""Hand-written character scanner for SCL (Semantic Context Layer)."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import StrEnum

from uacp.scl.errors import SCLSyntaxError


class TokenType(StrEnum):
    """Enumeration of SCL token kinds."""

    OPEN_TAG = "OPEN_TAG"
    CLOSE_TAG = "CLOSE_TAG"
    TEXT = "TEXT"
    EOF = "EOF"


@dataclass(frozen=True)
class Token:
    """Represents a scanned lexical unit in SCL."""

    type: TokenType
    value: str
    line: int
    col: int
    end_line: int
    end_col: int
    attrs: dict[str, str] = field(default_factory=dict)


_IDENT_REGEX = re.compile(r"[a-zA-Z_][a-zA-Z0-9_]*")
_JSON_SCHEMA_CLOSE_REGEX = re.compile(r"</output\s*>")


class Lexer:
    """Lexical scanner for SCL markup with line and column tracking."""

    def __init__(self, text: str) -> None:
        self.text = text
        self.pos = 0
        self.line = 1
        self.col = 1
        self.raw_until_tag: str | None = None

    def _peek(self, offset: int = 0) -> str:
        idx = self.pos + offset
        if idx < len(self.text):
            return self.text[idx]
        return ""

    def _advance(self) -> str:
        if self.pos >= len(self.text):
            return ""
        ch = self.text[self.pos]
        if ch == "\r" and self._peek(1) == "\n":
            self.pos += 2
            self.line += 1
            self.col = 1
            return "\r\n"
        if ch in ("\n", "\r"):
            self.pos += 1
            self.line += 1
            self.col = 1
            return ch
        self.pos += 1
        self.col += 1
        return ch

    def _advance_by(self, count: int) -> None:
        for _ in range(count):
            self._advance()

    def _skip_whitespace(self) -> None:
        while self.pos < len(self.text) and self.text[self.pos].isspace():
            self._advance()

    def _scan_ident(self) -> str:
        match = _IDENT_REGEX.match(self.text, self.pos)
        if not match:
            return ""
        ident = match.group(0)
        self._advance_by(len(ident))
        return ident

    def _scan_raw_text(self, tag_name: str) -> Token:
        start_line = self.line
        start_col = self.col
        start_pos = self.pos

        pattern = _JSON_SCHEMA_CLOSE_REGEX if tag_name == "output" else re.compile(
            rf"</{tag_name}\s*>"
        )
        match = pattern.search(self.text, self.pos)
        target_pos = match.start() if match else len(self.text)

        while self.pos < target_pos:
            self._advance()

        content = self.text[start_pos:target_pos]
        self.raw_until_tag = None
        return Token(
            type=TokenType.TEXT,
            value=content,
            line=start_line,
            col=start_col,
            end_line=self.line,
            end_col=self.col,
        )

    def next_token(self) -> Token:
        """Scan and return the next lexical token."""
        if self.raw_until_tag is not None:
            return self._scan_raw_text(self.raw_until_tag)

        if self.pos >= len(self.text):
            return Token(
                type=TokenType.EOF,
                value="",
                line=self.line,
                col=self.col,
                end_line=self.line,
                end_col=self.col,
            )

        if self._peek() == "<":
            if self._peek(1) == "/":
                return self._scan_close_tag()
            return self._scan_open_tag()

        return self._scan_text()

    def _scan_close_tag(self) -> Token:
        start_line = self.line
        start_col = self.col
        # Advance past '</'
        self._advance()
        self._advance()

        self._skip_whitespace()
        name_line, name_col = self.line, self.col
        tag_name = self._scan_ident()
        if not tag_name:
            found = repr(self._peek()) if self._peek() else "EOF"
            raise SCLSyntaxError(
                f"Expected tag name after '</', found {found}",
                name_line,
                name_col,
                hint="Specify tag name like </output>",
            )

        self._skip_whitespace()
        if self._peek() != ">":
            found = repr(self._peek()) if self._peek() else "EOF"
            raise SCLSyntaxError(
                f"Expected '>' to close '</{tag_name}>', found {found}",
                self.line,
                self.col,
                hint="Close the tag with '>'",
            )
        self._advance()

        return Token(
            type=TokenType.CLOSE_TAG,
            value=tag_name,
            line=start_line,
            col=start_col,
            end_line=self.line,
            end_col=self.col,
        )

    def _scan_open_tag(self) -> Token:
        start_line = self.line
        start_col = self.col
        # Advance past '<'
        self._advance()

        self._skip_whitespace()
        name_line, name_col = self.line, self.col
        tag_name = self._scan_ident()
        if not tag_name:
            found = repr(self._peek()) if self._peek() else "EOF"
            raise SCLSyntaxError(
                f"Expected tag name after '<', found {found}",
                name_line,
                name_col,
                hint="Specify tag name like <identity>",
            )

        attrs: dict[str, str] = {}
        while True:
            self._skip_whitespace()
            if self.pos >= len(self.text):
                raise SCLSyntaxError(
                    f"Unclosed tag '<{tag_name}>' at EOF",
                    start_line,
                    start_col,
                    hint="Close tag with '>'",
                )

            if self._peek() == ">":
                self._advance()
                break

            attr_line, attr_col = self.line, self.col
            attr_name = self._scan_ident()
            if not attr_name:
                found = repr(self._peek())
                raise SCLSyntaxError(
                    f"Expected attribute name or '>' in <{tag_name}>, found {found}",
                    self.line,
                    self.col,
                    hint="Attributes must follow name=\"value\"",
                )

            self._skip_whitespace()
            if self._peek() != "=":
                found = repr(self._peek()) if self._peek() else "EOF"
                raise SCLSyntaxError(
                    f"Expected '=' after attribute '{attr_name}' in <{tag_name}>, found {found}",
                    self.line,
                    self.col,
                    hint=f"Use {attr_name}=\"value\"",
                )
            self._advance()  # consume '='

            self._skip_whitespace()
            quote_char = self._peek()
            if quote_char not in ('"', "'"):
                found = repr(quote_char) if quote_char else "EOF"
                raise SCLSyntaxError(
                    f"Expected quote ('\"' or '\\'') for attribute '{attr_name}', found {found}",
                    self.line,
                    self.col,
                    hint=f"Enclose value in quotes: {attr_name}=\"value\"",
                )
            self._advance()  # consume open quote

            val_chars: list[str] = []
            quote_closed = False
            while self.pos < len(self.text):
                ch = self._peek()
                if ch == quote_char:
                    self._advance()
                    quote_closed = True
                    break
                val_chars.append(self._advance())

            if not quote_closed:
                raise SCLSyntaxError(
                    f"Unterminated quote for attribute '{attr_name}' in <{tag_name}>",
                    attr_line,
                    attr_col,
                    hint="Close attribute quote",
                )

            if attr_name in attrs:
                raise SCLSyntaxError(
                    f"Duplicate attribute '{attr_name}' in <{tag_name}>",
                    attr_line,
                    attr_col,
                    hint=f"Remove duplicate '{attr_name}' attribute",
                )

            attrs[attr_name] = "".join(val_chars)

        if tag_name == "output" and attrs.get("format") == "json_schema":
            self.raw_until_tag = "output"

        return Token(
            type=TokenType.OPEN_TAG,
            value=tag_name,
            line=start_line,
            col=start_col,
            end_line=self.line,
            end_col=self.col,
            attrs=attrs,
        )

    def _scan_text(self) -> Token:
        start_line = self.line
        start_col = self.col
        chars: list[str] = []

        while self.pos < len(self.text) and self._peek() != "<":
            chars.append(self._advance())

        return Token(
            type=TokenType.TEXT,
            value="".join(chars),
            line=start_line,
            col=start_col,
            end_line=self.line,
            end_col=self.col,
        )


def tokenize(text: str) -> list[Token]:
    """Tokenize full SCL source into a list of tokens ending with EOF.

    Args:
        text: SCL source text.

    Returns:
        List of scanned tokens.
    """
    lexer = Lexer(text)
    tokens: list[Token] = []
    while True:
        tok = lexer.next_token()
        tokens.append(tok)
        if tok.type == TokenType.EOF:
            break
    return tokens
