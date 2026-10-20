"""Unit tests for the SCL lexer."""

import pytest

from uacp.scl.errors import SCLSyntaxError
from uacp.scl.lexer import TokenType, tokenize


def test_lexer_minimal_tokens() -> None:
    text = '<identity profile="bot">Hello</identity>'
    tokens = tokenize(text)
    assert len(tokens) == 4
    assert tokens[0].type == TokenType.OPEN_TAG
    assert tokens[0].value == "identity"
    assert tokens[0].attrs == {"profile": "bot"}
    assert tokens[1].type == TokenType.TEXT
    assert tokens[1].value == "Hello"
    assert tokens[2].type == TokenType.CLOSE_TAG
    assert tokens[2].value == "identity"
    assert tokens[3].type == TokenType.EOF


def test_lexer_line_col_tracking_crlf_and_lf() -> None:
    text = "<scl>\r\n  <identity profile='auditor'>\r\nline 2\r\n</identity>\r\n</scl>"
    tokens = tokenize(text)
    open_scl = tokens[0]
    assert open_scl.line == 1
    assert open_scl.col == 1

    open_ident = next(t for t in tokens if t.type == TokenType.OPEN_TAG and t.value == "identity")
    assert open_ident.line == 2
    assert open_ident.col == 3


def test_lexer_quoted_attributes() -> None:
    text = '<identity profile="double" tone=\'single\'>body</identity>'
    tokens = tokenize(text)
    assert tokens[0].attrs == {"profile": "double", "tone": "single"}


def test_lexer_whitespace_in_attributes() -> None:
    text = '<identity   profile  =  "test"   tone  =  \'formal\'  >body</identity>'
    tokens = tokenize(text)
    assert tokens[0].attrs == {"profile": "test", "tone": "formal"}


def test_lexer_duplicate_attribute_raises() -> None:
    text = '<identity profile="a" profile="b">body</identity>'
    with pytest.raises(SCLSyntaxError) as exc_info:
        tokenize(text)
    assert "Duplicate attribute 'profile'" in str(exc_info.value)
    assert exc_info.value.line == 1


def test_lexer_unterminated_quote_raises() -> None:
    text = '<identity profile="unterminated>body</identity>'
    with pytest.raises(SCLSyntaxError) as exc_info:
        tokenize(text)
    assert "Unterminated quote" in str(exc_info.value)


def test_lexer_unclosed_tag_raises() -> None:
    text = "<identity profile='test'"
    with pytest.raises(SCLSyntaxError) as exc_info:
        tokenize(text)
    assert "Unclosed tag '<identity>' at EOF" in str(exc_info.value)


def test_lexer_json_schema_raw_text_nested_brackets() -> None:
    text = (
        '<output format="json_schema">\n'
        '{"desc": "<tag> a < b > c </unknown>"}\n'
        '</output>'
    )
    tokens = tokenize(text)
    assert tokens[0].type == TokenType.OPEN_TAG
    assert tokens[0].value == "output"
    assert tokens[1].type == TokenType.TEXT
    assert '{"desc": "<tag> a < b > c </unknown>"}\n' in tokens[1].value
    assert tokens[2].type == TokenType.CLOSE_TAG
    assert tokens[2].value == "output"


def test_lexer_close_tag_missing_name_raises() -> None:
    text = "</>"
    with pytest.raises(SCLSyntaxError) as exc_info:
        tokenize(text)
    assert "Expected tag name after '</'" in str(exc_info.value)


def test_lexer_close_tag_unclosed_raises() -> None:
    text = "</output"
    with pytest.raises(SCLSyntaxError) as exc_info:
        tokenize(text)
    assert "Expected '>' to close '</output>'" in str(exc_info.value)

