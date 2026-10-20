"""SCL recursive-descent parser: tokens -> AST."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from uacp.scl.ast import Document, Example, Identity, Output, Span, Variable
from uacp.scl.errors import SCLSyntaxError
from uacp.scl.lexer import Token, TokenType, tokenize

_VAR_NAME_REGEX = re.compile(r"^[a-z_][a-z0-9_]*$")
_DOTTED_PATH_REGEX = re.compile(r"^[a-zA-Z_][a-zA-Z0-9_]*(\.[a-zA-Z_][a-zA-Z0-9_]*)*$")

_ALLOWED_ATTRS: dict[str, set[str]] = {
    "scl": {"version"},
    "identity": {"profile", "tone"},
    "variable": {"name", "source", "type"},
    "example": {"pair"},
    "input": set(),
    "output": {"format", "strict"},
}


class Parser:
    """Recursive-descent parser for SCL tokens producing a Document AST."""

    def __init__(
        self,
        tokens: list[Token],
        *,
        source_name: str = "<string>",
        store_spans: bool = False,
    ) -> None:
        self.tokens = tokens
        self.source_name = source_name
        self.store_spans = store_spans
        self.pos = 0
        self.tag_stack: list[Token] = []

    def _peek(self) -> Token:
        if self.pos < len(self.tokens):
            return self.tokens[self.pos]
        return self.tokens[-1]

    def _advance(self) -> Token:
        tok = self._peek()
        if self.pos < len(self.tokens):
            self.pos += 1
        return tok

    def _check_attributes(self, tok: Token) -> None:
        allowed = _ALLOWED_ATTRS.get(tok.value)
        if allowed is None:
            return
        for attr in tok.attrs:
            if attr not in allowed:
                hint = (
                    f"Allowed attributes: {', '.join(sorted(allowed))}"
                    if allowed
                    else "No attributes allowed"
                )
                raise SCLSyntaxError(
                    f"Unknown attribute '{attr}' on <{tok.value}>",
                    tok.line,
                    tok.col,
                    hint=hint,
                )

    def _parse_bool(self, val: str, attr_name: str, tok: Token) -> bool:
        if val == "true":
            return True
        if val == "false":
            return False
        raise SCLSyntaxError(
            f"Invalid boolean value '{val}' for attribute '{attr_name}' in <{tok.value}>",
            tok.line,
            tok.col,
            hint=f"Use {attr_name}=\"true\" or {attr_name}=\"false\"",
        )

    def _make_span(self, start_tok: Token, end_tok: Token) -> Span | None:
        if not self.store_spans:
            return None
        return Span(
            line=start_tok.line,
            col=start_tok.col,
            end_line=end_tok.end_line,
            end_col=end_tok.end_col,
        )

    def parse_document(self) -> Document:
        """Parse full token stream into a Document AST."""
        self._skip_whitespace_text()

        has_scl_wrapper = False
        scl_version: str | None = None
        scl_open_tok: Token | None = None

        if self._peek().type == TokenType.OPEN_TAG and self._peek().value == "scl":
            scl_open_tok = self._advance()
            self._check_attributes(scl_open_tok)
            scl_version = scl_open_tok.attrs.get("version", "1")
            self.tag_stack.append(scl_open_tok)
            has_scl_wrapper = True

        identity: Identity | None = None
        variables: list[Variable] = []
        examples: list[Example] = []
        output: Output | None = None

        while True:
            self._skip_whitespace_text()
            tok = self._peek()

            if tok.type == TokenType.EOF:
                break

            if tok.type == TokenType.CLOSE_TAG:
                if has_scl_wrapper and tok.value == "scl":
                    break
                # Mismatched or unexpected closing tag
                expected = f"</{self.tag_stack[-1].value}>" if self.tag_stack else "no tag"
                raise SCLSyntaxError(
                    f"Mismatched closing tag: expected {expected} but found </{tok.value}>",
                    tok.line,
                    tok.col,
                    hint=f"expected {expected} but found </{tok.value}>",
                )

            if tok.type == TokenType.TEXT:
                stripped = tok.value.strip()
                if stripped:
                    raise SCLSyntaxError(
                        f"Unexpected text outside element: {stripped!r}",
                        tok.line,
                        tok.col,
                    )
                self._advance()
                continue

            if tok.type == TokenType.OPEN_TAG:
                if tok.value == "identity":
                    if identity is not None:
                        raise SCLSyntaxError(
                            "Duplicate <identity> tag; exactly one <identity> is allowed",
                            tok.line,
                            tok.col,
                            hint="Remove duplicate <identity>",
                        )
                    identity = self._parse_identity()
                elif tok.value == "variable":
                    variables.append(self._parse_variable())
                elif tok.value == "example":
                    examples.append(self._parse_example())
                elif tok.value == "output":
                    if output is not None:
                        raise SCLSyntaxError(
                            "Duplicate <output> tag; at most one <output> is allowed",
                            tok.line,
                            tok.col,
                            hint="Remove duplicate <output>",
                        )
                    output = self._parse_output()
                elif tok.value == "scl":
                    raise SCLSyntaxError(
                        "Unexpected nested <scl> tag",
                        tok.line,
                        tok.col,
                    )
                else:
                    raise SCLSyntaxError(
                        f"Unknown element <{tok.value}>",
                        tok.line,
                        tok.col,
                        hint="Allowed elements: <identity>, <variable>, <example>, <output>",
                    )

        if has_scl_wrapper:
            self._skip_whitespace_text()
            tok = self._peek()
            if tok.type == TokenType.EOF:
                assert scl_open_tok is not None
                raise SCLSyntaxError(
                    "Unclosed tag <scl> at EOF",
                    scl_open_tok.line,
                    scl_open_tok.col,
                    hint="expected </scl>",
                )
            if tok.type != TokenType.CLOSE_TAG or tok.value != "scl":
                expected = f"</{self.tag_stack[-1].value}>" if self.tag_stack else "</scl>"
                found = f"</{tok.value}>" if tok.type == TokenType.CLOSE_TAG else repr(tok.value)
                raise SCLSyntaxError(
                    f"Mismatched closing tag: expected {expected} but found {found}",
                    tok.line,
                    tok.col,
                    hint=f"expected {expected} but found {found}",
                )
            close_scl = self._advance()
            self.tag_stack.pop()

            self._skip_whitespace_text()
            if self._peek().type != TokenType.EOF:
                tok = self._peek()
                raise SCLSyntaxError(
                    f"Unexpected content after </scl>: {tok.value!r}",
                    tok.line,
                    tok.col,
                )
        else:
            scl_version = "1"
            if self._peek().type != TokenType.EOF:
                tok = self._peek()
                raise SCLSyntaxError(
                    f"Unexpected token at end of document: {tok.value!r}",
                    tok.line,
                    tok.col,
                )

        if self.tag_stack:
            unclosed = self.tag_stack[-1]
            raise SCLSyntaxError(
                f"Unclosed tag <{unclosed.value}> at EOF",
                unclosed.line,
                unclosed.col,
                hint=f"expected </{unclosed.value}>",
            )

        if identity is None:
            raise SCLSyntaxError(
                "Missing required <identity> element",
                1,
                1,
                hint="Add <identity profile=\"...\">...</identity>",
            )

        doc_span = (
            self._make_span(scl_open_tok, close_scl)
            if (has_scl_wrapper and scl_open_tok and self.store_spans)
            else None
        )

        return Document(
            version=scl_version,
            identity=identity,
            variables=variables,
            examples=examples,
            output=output,
            span=doc_span,
        )

    def _skip_whitespace_text(self) -> None:
        while self._peek().type == TokenType.TEXT and not self._peek().value.strip():
            self._advance()

    def _parse_identity(self) -> Identity:
        open_tok = self._advance()
        self._check_attributes(open_tok)
        self.tag_stack.append(open_tok)

        profile = open_tok.attrs.get("profile")
        if not profile:
            raise SCLSyntaxError(
                "Missing required attribute 'profile' on <identity>",
                open_tok.line,
                open_tok.col,
                hint="Add profile=\"...\"",
            )
        tone = open_tok.attrs.get("tone")

        body_parts: list[str] = []
        while True:
            tok = self._peek()
            if tok.type == TokenType.TEXT:
                body_parts.append(self._advance().value)
            elif tok.type == TokenType.CLOSE_TAG:
                if tok.value != "identity":
                    expected = f"</{self.tag_stack[-1].value}>"
                    raise SCLSyntaxError(
                        f"Mismatched closing tag: expected {expected} but found </{tok.value}>",
                        tok.line,
                        tok.col,
                        hint=f"expected {expected} but found </{tok.value}>",
                    )
                close_tok = self._advance()
                self.tag_stack.pop()
                break
            elif tok.type == TokenType.EOF:
                raise SCLSyntaxError(
                    "Unclosed tag <identity> at EOF",
                    open_tok.line,
                    open_tok.col,
                    hint="expected </identity>",
                )
            else:
                raise SCLSyntaxError(
                    f"Unexpected tag <{tok.value}> inside <identity>",
                    tok.line,
                    tok.col,
                )

        body = "".join(body_parts)
        return Identity(
            profile=profile,
            tone=tone,
            body=body,
            span=self._make_span(open_tok, close_tok),
        )

    def _parse_variable(self) -> Variable:
        open_tok = self._advance()
        self._check_attributes(open_tok)
        self.tag_stack.append(open_tok)

        name = open_tok.attrs.get("name")
        if not name:
            raise SCLSyntaxError(
                "Missing required attribute 'name' on <variable>",
                open_tok.line,
                open_tok.col,
                hint="Add name=\"...\"",
            )
        if not _VAR_NAME_REGEX.match(name):
            raise SCLSyntaxError(
                f"Invalid variable name '{name}'; must match regex '^[a-z_][a-z0-9_]*$'",
                open_tok.line,
                open_tok.col,
            )

        source = open_tok.attrs.get("source")
        if not source:
            raise SCLSyntaxError(
                "Missing required attribute 'source' on <variable>",
                open_tok.line,
                open_tok.col,
                hint="Add source=\"...\"",
            )
        if not _DOTTED_PATH_REGEX.match(source):
            raise SCLSyntaxError(
                f"Invalid variable source '{source}'; must be a dotted path",
                open_tok.line,
                open_tok.col,
            )

        var_type = open_tok.attrs.get("type", "text")
        if var_type not in ("json", "text"):
            raise SCLSyntaxError(
                f"Invalid variable type '{var_type}'; must be 'json' or 'text'",
                open_tok.line,
                open_tok.col,
                hint="Set type=\"json\" or type=\"text\"",
            )

        body_parts: list[str] = []
        while True:
            tok = self._peek()
            if tok.type == TokenType.TEXT:
                body_parts.append(self._advance().value)
            elif tok.type == TokenType.CLOSE_TAG:
                if tok.value != "variable":
                    expected = f"</{self.tag_stack[-1].value}>"
                    raise SCLSyntaxError(
                        f"Mismatched closing tag: expected {expected} but found </{tok.value}>",
                        tok.line,
                        tok.col,
                        hint=f"expected {expected} but found </{tok.value}>",
                    )
                close_tok = self._advance()
                self.tag_stack.pop()
                break
            elif tok.type == TokenType.EOF:
                raise SCLSyntaxError(
                    "Unclosed tag <variable> at EOF",
                    open_tok.line,
                    open_tok.col,
                    hint="expected </variable>",
                )
            else:
                raise SCLSyntaxError(
                    f"Unexpected tag <{tok.value}> inside <variable>",
                    tok.line,
                    tok.col,
                )

        body = "".join(body_parts)
        template = body if body.strip() else None

        return Variable(
            name=name,
            source=source,
            type=var_type,  # type: ignore[arg-type]
            template=template,
            span=self._make_span(open_tok, close_tok),
        )

    def _parse_example(self) -> Example:
        open_tok = self._advance()
        self._check_attributes(open_tok)
        self.tag_stack.append(open_tok)

        pair_attr = open_tok.attrs.get("pair", "true")
        pair = self._parse_bool(pair_attr, "pair", open_tok)

        inputs: list[str] = []
        outputs: list[str] = []
        close_tok: Token | None = None

        while True:
            self._skip_whitespace_text()
            tok = self._peek()

            if tok.type == TokenType.CLOSE_TAG:
                if tok.value != "example":
                    expected = f"</{self.tag_stack[-1].value}>"
                    raise SCLSyntaxError(
                        f"Mismatched closing tag: expected {expected} but found </{tok.value}>",
                        tok.line,
                        tok.col,
                        hint=f"expected {expected} but found </{tok.value}>",
                    )
                close_tok = self._advance()
                self.tag_stack.pop()
                break

            if tok.type == TokenType.EOF:
                raise SCLSyntaxError(
                    "Unclosed tag <example> at EOF",
                    open_tok.line,
                    open_tok.col,
                    hint="expected </example>",
                )

            if tok.type == TokenType.OPEN_TAG:
                if tok.value == "input":
                    inputs.append(self._parse_inner_text_element("input"))
                elif tok.value == "output":
                    outputs.append(self._parse_inner_text_element("output"))
                else:
                    raise SCLSyntaxError(
                        f"Unexpected tag <{tok.value}> inside <example>",
                        tok.line,
                        tok.col,
                        hint="<example> only allows <input> and <output>",
                    )
            elif tok.type == TokenType.TEXT:
                stripped = tok.value.strip()
                if stripped:
                    raise SCLSyntaxError(
                        f"Unexpected text inside <example>: {stripped!r}",
                        tok.line,
                        tok.col,
                    )
                self._advance()

        if pair and (len(inputs) != 1 or len(outputs) != 1):
            msg = (
                f"<example pair=\"true\"> must contain exactly one <input> and one <output> "
                f"(found {len(inputs)} inputs, {len(outputs)} outputs)"
            )
            raise SCLSyntaxError(
                msg,
                open_tok.line,
                open_tok.col,
                hint="Ensure exactly one <input> and one <output> are inside <example>",
            )

        input_text = inputs[0] if inputs else ""
        output_text = outputs[0] if outputs else ""

        assert close_tok is not None
        return Example(
            pair=pair,
            input=input_text,
            output=output_text,
            span=self._make_span(open_tok, close_tok),
        )

    def _parse_inner_text_element(self, expected_name: str) -> str:
        open_tok = self._advance()
        self._check_attributes(open_tok)
        self.tag_stack.append(open_tok)

        parts: list[str] = []
        while True:
            tok = self._peek()
            if tok.type == TokenType.TEXT:
                parts.append(self._advance().value)
            elif tok.type == TokenType.CLOSE_TAG:
                if tok.value != expected_name:
                    expected = f"</{self.tag_stack[-1].value}>"
                    raise SCLSyntaxError(
                        f"Mismatched closing tag: expected {expected} but found </{tok.value}>",
                        tok.line,
                        tok.col,
                        hint=f"expected {expected} but found </{tok.value}>",
                    )
                self._advance()
                self.tag_stack.pop()
                break
            elif tok.type == TokenType.EOF:
                raise SCLSyntaxError(
                    f"Unclosed tag <{expected_name}> at EOF",
                    open_tok.line,
                    open_tok.col,
                    hint=f"expected </{expected_name}>",
                )
            else:
                raise SCLSyntaxError(
                    f"Unexpected tag <{tok.value}> inside <{expected_name}>",
                    tok.line,
                    tok.col,
                )
        return "".join(parts)

    def _parse_output(self) -> Output:
        open_tok = self._advance()
        self._check_attributes(open_tok)
        self.tag_stack.append(open_tok)

        fmt = open_tok.attrs.get("format")
        if not fmt:
            raise SCLSyntaxError(
                "Missing required attribute 'format' on <output>",
                open_tok.line,
                open_tok.col,
                hint="Add format=\"json_schema\" or format=\"markdown\"",
            )
        if fmt not in ("json_schema", "markdown"):
            raise SCLSyntaxError(
                f"Invalid output format '{fmt}'; must be 'json_schema' or 'markdown'",
                open_tok.line,
                open_tok.col,
                hint="Use format=\"json_schema\" or format=\"markdown\"",
            )

        strict_attr = open_tok.attrs.get("strict", "false")
        strict = self._parse_bool(strict_attr, "strict", open_tok)

        body_tok: Token | None = None
        body_parts: list[str] = []

        while True:
            tok = self._peek()
            if tok.type == TokenType.TEXT:
                if body_tok is None:
                    body_tok = tok
                body_parts.append(self._advance().value)
            elif tok.type == TokenType.CLOSE_TAG:
                if tok.value != "output":
                    expected = f"</{self.tag_stack[-1].value}>"
                    raise SCLSyntaxError(
                        f"Mismatched closing tag: expected {expected} but found </{tok.value}>",
                        tok.line,
                        tok.col,
                        hint=f"expected {expected} but found </{tok.value}>",
                    )
                close_tok = self._advance()
                self.tag_stack.pop()
                break
            elif tok.type == TokenType.EOF:
                raise SCLSyntaxError(
                    "Unclosed tag <output> at EOF",
                    open_tok.line,
                    open_tok.col,
                    hint="expected </output>",
                )
            else:
                raise SCLSyntaxError(
                    f"Unexpected tag <{tok.value}> inside <output>",
                    tok.line,
                    tok.col,
                )

        body = "".join(body_parts)
        schema_dict: dict[str, Any] | None = None

        if fmt == "json_schema":
            try:
                schema_dict = json.loads(body)
                if not isinstance(schema_dict, dict):
                    err_line = body_tok.line if body_tok else open_tok.line
                    err_col = body_tok.col if body_tok else open_tok.col
                    raise SCLSyntaxError(
                        "JSON schema in <output> must be a JSON object",
                        err_line,
                        err_col,
                        hint="Enclose schema in { ... }",
                    )
            except json.JSONDecodeError as exc:
                if body_tok is not None:
                    err_line = body_tok.line + (exc.lineno - 1)
                    err_col = (body_tok.col + exc.colno - 1) if exc.lineno == 1 else exc.colno
                else:
                    err_line = open_tok.line
                    err_col = open_tok.col
                raise SCLSyntaxError(
                    f"Invalid JSON schema in <output>: {exc.msg}",
                    err_line,
                    err_col,
                    hint="Check JSON syntax inside <output>",
                ) from exc

        return Output(
            format=fmt,  # type: ignore[arg-type]
            strict=strict,
            body=body,
            schema=schema_dict,
            span=self._make_span(open_tok, close_tok),
        )


def parse(
    text: str,
    *,
    source_name: str = "<string>",
    store_spans: bool = False,
) -> Document:
    """Parse SCL text markup into a Document AST.

    Args:
        text: Source SCL markup string.
        source_name: Optional origin identifier for error reporting.
        store_spans: Whether to populate Span coordinates on AST nodes.

    Returns:
        Constructed Document AST node.
    """
    tokens = tokenize(text)
    parser = Parser(tokens, source_name=source_name, store_spans=store_spans)
    return parser.parse_document()


def parse_file(
    path: str | Path,
    *,
    store_spans: bool = False,
) -> Document:
    """Read and parse an SCL file into a Document AST.

    Args:
        path: Path to the .scl file.
        store_spans: Whether to populate Span coordinates on AST nodes.

    Returns:
        Constructed Document AST node.
    """
    path_obj = Path(path)
    text = path_obj.read_text(encoding="utf-8")
    return parse(text, source_name=str(path_obj), store_spans=store_spans)


def format(doc: Document) -> str:
    """Pretty-print an SCL AST Document back into canonical SCL markup.

    Args:
        doc: Document AST node.

    Returns:
        Formatted SCL markup string.
    """
    lines: list[str] = []
    has_wrapper = doc.version is not None
    if has_wrapper:
        lines.append(f'<scl version="{doc.version}">')

    indent = "  " if has_wrapper else ""

    # Identity
    tone_str = f' tone="{doc.identity.tone}"' if doc.identity.tone is not None else ""
    lines.append(
        f'{indent}<identity profile="{doc.identity.profile}"{tone_str}>'
        f'{doc.identity.body}</identity>'
    )

    # Variables
    for var in doc.variables:
        type_str = f' type="{var.type}"' if var.type != "text" else ""
        template_content = var.template if var.template is not None else ""
        lines.append(
            f'{indent}<variable name="{var.name}" source="{var.source}"{type_str}>'
            f'{template_content}</variable>'
        )

    # Examples
    for ex in doc.examples:
        pair_str = ' pair="false"' if not ex.pair else ""
        lines.append(f"{indent}<example{pair_str}>")
        lines.append(f"{indent}  <input>{ex.input}</input>")
        lines.append(f"{indent}  <output>{ex.output}</output>")
        lines.append(f"{indent}</example>")

    # Output
    if doc.output is not None:
        strict_str = ' strict="true"' if doc.output.strict else ""
        lines.append(
            f'{indent}<output format="{doc.output.format}"{strict_str}>{doc.output.body}</output>'
        )

    if has_wrapper:
        lines.append("</scl>")

    return "\n".join(lines) + "\n"
