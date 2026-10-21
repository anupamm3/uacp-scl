"""Unit tests for SCL and UACP command-line interface."""

from __future__ import annotations

import json
from pathlib import Path

from typer.testing import CliRunner

from uacp.adapters.fake_adapter import FakeAdapter
from uacp.adapters.registry import register_adapter
from uacp.cli import scl_app, uacp_app

runner = CliRunner()

VALID_SCL = """<scl version="1">
  <identity profile="assistant" tone="helpful">You are a helpful assistant.</identity>
  <variable name="user_name" source="user.name" type="text">{{user_name}}</variable>
  <example>
    <input>Hi</input>
    <output>{"reply": "Hello"}</output>
  </example>
  <output format="json_schema">
    {
      "type": "object",
      "properties": {
        "reply": {"type": "string"}
      },
      "required": ["reply"]
    }
  </output>
</scl>
"""

SYNTAX_ERROR_SCL = """<scl version="1">
  <identity profile="assistant">Missing closing tag
</scl>
"""

SEMANTIC_ERROR_SCL = """<scl version="1">
  <identity profile="assistant">Hello {{undeclared}}</identity>
</scl>
"""

WARNING_SCL = """<scl version="1">
  <identity profile="assistant">Test</identity>
  <example>
    <input>Hi</input>
    <output>{"reply": 123}</output>
  </example>
  <output format="json_schema">
    {
      "type": "object",
      "properties": {
        "reply": {"type": "string"}
      },
      "required": ["reply"]
    }
  </output>
</scl>
"""


def test_lint_valid_file(tmp_path: Path) -> None:
    scl_file = tmp_path / "valid.scl"
    scl_file.write_text(VALID_SCL, encoding="utf-8")

    result = runner.invoke(scl_app, ["lint", str(scl_file)])
    assert result.exit_code == 0
    assert "passed linting" in result.stdout


def test_lint_syntax_error(tmp_path: Path) -> None:
    scl_file = tmp_path / "syntax_err.scl"
    scl_file.write_text(SYNTAX_ERROR_SCL, encoding="utf-8")

    result = runner.invoke(scl_app, ["lint", str(scl_file)])
    assert result.exit_code == 1
    assert "Syntax Error" in result.stdout
    assert "^" in result.stdout


def test_lint_semantic_error(tmp_path: Path) -> None:
    scl_file = tmp_path / "semantic_err.scl"
    scl_file.write_text(SEMANTIC_ERROR_SCL, encoding="utf-8")

    result = runner.invoke(scl_app, ["lint", str(scl_file)])
    assert result.exit_code == 1
    assert "ERROR" in result.stdout
    assert "^" in result.stdout


def test_lint_warnings_non_strict(tmp_path: Path) -> None:
    scl_file = tmp_path / "warning.scl"
    scl_file.write_text(WARNING_SCL, encoding="utf-8")

    result = runner.invoke(scl_app, ["lint", str(scl_file)])
    assert result.exit_code == 0
    assert "WARNING" in result.stdout
    assert "^" in result.stdout


def test_lint_warnings_strict(tmp_path: Path) -> None:
    scl_file = tmp_path / "warning_strict.scl"
    scl_file.write_text(WARNING_SCL, encoding="utf-8")

    result = runner.invoke(scl_app, ["lint", str(scl_file), "--strict"])
    assert result.exit_code == 1
    assert "WARNING" in result.stdout


def test_ast_tree(tmp_path: Path) -> None:
    scl_file = tmp_path / "valid.scl"
    scl_file.write_text(VALID_SCL, encoding="utf-8")

    result = runner.invoke(scl_app, ["ast", str(scl_file)])
    assert result.exit_code == 0
    assert "Document" in result.stdout
    assert "Identity" in result.stdout
    assert "Variables" in result.stdout
    assert "user_name" in result.stdout


def test_ast_json(tmp_path: Path) -> None:
    scl_file = tmp_path / "valid.scl"
    scl_file.write_text(VALID_SCL, encoding="utf-8")

    result = runner.invoke(scl_app, ["ast", str(scl_file), "--json"])
    assert result.exit_code == 0
    data = json.loads(result.stdout)
    assert data["identity"]["profile"] == "assistant"
    assert data["variables"][0]["name"] == "user_name"


def test_ast_syntax_error(tmp_path: Path) -> None:
    scl_file = tmp_path / "syntax_err.scl"
    scl_file.write_text(SYNTAX_ERROR_SCL, encoding="utf-8")

    result = runner.invoke(scl_app, ["ast", str(scl_file)])
    assert result.exit_code == 1
    assert "Syntax Error" in result.stdout


def test_compile_success_and_redaction(tmp_path: Path) -> None:
    scl_file = tmp_path / "test.scl"
    scl_file.write_text(VALID_SCL, encoding="utf-8")

    data_file = tmp_path / "data.json"
    long_string = "A" * 300
    data_file.write_text(
        json.dumps({"user": {"name": "Alice"}, "extra": long_string}),
        encoding="utf-8",
    )

    result = runner.invoke(
        scl_app,
        [
            "compile",
            str(scl_file),
            "--alias",
            "fake",
            "--data",
            str(data_file),
            "--user-message",
            "Say hello",
        ],
    )
    assert result.exit_code == 0
    compiled = json.loads(result.stdout)
    assert compiled["provider"] == "fake"
    assert compiled["model"] == "fake-echo-v1"
    assert any("Alice" in m["content"] for m in compiled["messages"])


def test_compile_truncates_long_data_and_masks_keys(tmp_path: Path) -> None:
    # SCL with long variable
    scl = """<scl version="1">
      <identity profile="assistant">Assistant</identity>
      <variable name="huge" source="huge_key" type="text">{{huge}}</variable>
    </scl>
    """
    scl_file = tmp_path / "huge.scl"
    scl_file.write_text(scl, encoding="utf-8")

    huge_text = "X" * 350
    data_file = tmp_path / "data.json"
    data_file.write_text(
        json.dumps({"huge_key": huge_text, "api_key": "AIzaSySecretKey1234567890"}),
        encoding="utf-8",
    )

    result = runner.invoke(
        scl_app,
        ["compile", str(scl_file), "--alias", "fake", "--data", str(data_file)],
    )
    assert result.exit_code == 0
    assert "[truncated" in result.stdout
    assert "AIzaSy" not in result.stdout


def test_compile_unknown_alias(tmp_path: Path) -> None:
    scl_file = tmp_path / "test.scl"
    scl_file.write_text(VALID_SCL, encoding="utf-8")
    data_file = tmp_path / "data.json"
    data_file.write_text(json.dumps({"user": {"name": "Alice"}}), encoding="utf-8")

    result = runner.invoke(
        scl_app,
        ["compile", str(scl_file), "--alias", "unknown_alias", "--data", str(data_file)],
    )
    assert result.exit_code == 1
    assert "Adapter Error" in result.stdout


def test_compile_missing_variable(tmp_path: Path) -> None:
    scl_file = tmp_path / "test.scl"
    scl_file.write_text(VALID_SCL, encoding="utf-8")
    data_file = tmp_path / "data.json"
    data_file.write_text(json.dumps({"wrong": "field"}), encoding="utf-8")

    result = runner.invoke(
        scl_app,
        ["compile", str(scl_file), "--alias", "fake", "--data", str(data_file)],
    )
    assert result.exit_code == 1
    assert "Binding Error" in result.stdout


def test_run_success_offline(tmp_path: Path) -> None:
    scl_file = tmp_path / "test.scl"
    scl_file.write_text(VALID_SCL, encoding="utf-8")
    data_file = tmp_path / "data.json"
    data_file.write_text(json.dumps({"user": {"name": "Alice"}}), encoding="utf-8")

    result = runner.invoke(
        scl_app,
        [
            "run",
            str(scl_file),
            "--alias",
            "fake",
            "--data",
            str(data_file),
            "--user-message",
            "Hello",
        ],
    )
    assert result.exit_code == 0
    assert "Input tokens:" in result.stdout
    assert "Output tokens:" in result.stdout
    assert "Repairs: 0" in result.stdout


def test_run_with_repair(tmp_path: Path) -> None:
    scl_file = tmp_path / "test.scl"
    scl_file.write_text(VALID_SCL, encoding="utf-8")
    data_file = tmp_path / "data.json"
    data_file.write_text(json.dumps({"user": {"name": "Alice"}}), encoding="utf-8")

    # Register a scripted fake adapter that fails once then succeeds
    class ScriptedFakeAdapter(FakeAdapter):
        def __init__(self, **kwargs):
            super().__init__(
                script=[
                    {"bad": "schema"},  # violates output schema ('reply' is required)
                    {"reply": "Fixed on repair"},  # conforms to schema
                ],
                **kwargs,
            )

    register_adapter("fake", ScriptedFakeAdapter)

    try:
        result = runner.invoke(
            scl_app,
            [
                "run",
                str(scl_file),
                "--alias",
                "fake",
                "--data",
                str(data_file),
                "--user-message",
                "Hello",
            ],
        )
        assert result.exit_code == 0
        assert "Fixed on repair" in result.stdout
        assert "Repairs: 1" in result.stdout
    finally:
        register_adapter("fake", FakeAdapter)


def test_run_fatal_violation(tmp_path: Path) -> None:
    scl_file = tmp_path / "test.scl"
    scl_file.write_text(VALID_SCL, encoding="utf-8")
    data_file = tmp_path / "data.json"
    data_file.write_text(json.dumps({"user": {"name": "Alice"}}), encoding="utf-8")

    class FatalFakeAdapter(FakeAdapter):
        def __init__(self, **kwargs):
            super().__init__(
                script=[
                    {"bad": 1},
                    {"bad": 2},
                    {"bad": 3},
                    {"bad": 4},
                ],
                **kwargs,
            )

    register_adapter("fake", FatalFakeAdapter)

    try:
        result = runner.invoke(
            scl_app,
            [
                "run",
                str(scl_file),
                "--alias",
                "fake",
                "--data",
                str(data_file),
                "--user-message",
                "Hello",
            ],
        )
        assert result.exit_code == 1
        assert "Output Violation" in result.stdout
    finally:
        register_adapter("fake", FakeAdapter)


def test_uacp_cli_mount(tmp_path: Path) -> None:
    scl_file = tmp_path / "valid.scl"
    scl_file.write_text(VALID_SCL, encoding="utf-8")

    result = runner.invoke(uacp_app, ["scl", "lint", str(scl_file)])
    assert result.exit_code == 0
    assert "passed linting" in result.stdout
