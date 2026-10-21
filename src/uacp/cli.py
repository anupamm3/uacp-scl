"""UACP and SCL command-line interface powered by Typer and Rich."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Annotated, Any

import typer
from rich.console import Console
from rich.tree import Tree

from uacp.adapters.base import Adapter, CompiledRequest, LLMResult, Message
from uacp.adapters.enforcer import OutputViolation, invoke_with_repair
from uacp.adapters.fake_adapter import FakeAdapter
from uacp.adapters.registry import get_adapter, load_env
from uacp.scl.ast import Document
from uacp.scl.ast import Output as SCLOutput
from uacp.scl.context import RuntimeContext, bind
from uacp.scl.errors import SCLSyntaxError, format_source_caret
from uacp.scl.parser import parse
from uacp.scl.validate import validate

console = Console()

scl_app = typer.Typer(
    name="scl",
    help="Semantic Context Layer (SCL) command-line interface.",
    no_args_is_help=True,
)
uacp_app = typer.Typer(
    name="uacp",
    help="Unified Agent Communication Protocol (UACP) command-line interface.",
    no_args_is_help=True,
)
uacp_app.add_typer(scl_app, name="scl")

# Backward compatibility alias
app = scl_app

_SENSITIVE_KEY_SUBSTRINGS = (
    "key",
    "token",
    "secret",
    "auth",
    "credential",
    "password",
    "api_key",
)


def _sanitize_for_display(obj: Any, max_str_len: int = 200) -> Any:
    """Recursively mask API keys/secrets and truncate long string payloads for display."""
    if isinstance(obj, dict):
        sanitized: dict[str, Any] = {}
        for k, v in obj.items():
            k_lower = str(k).lower()
            if any(sub in k_lower for sub in _SENSITIVE_KEY_SUBSTRINGS):
                sanitized[k] = "***REDACTED***"
            else:
                sanitized[k] = _sanitize_for_display(v, max_str_len)
        return sanitized
    if isinstance(obj, list):
        return [_sanitize_for_display(item, max_str_len) for item in obj]
    if isinstance(obj, str):
        if len(obj) > 20 and (
            obj.startswith("AIzaSy") or obj.startswith("gsk_") or obj.startswith("sk-")
        ):
            return "***REDACTED***"
        if len(obj) > max_str_len:
            return obj[:max_str_len] + f"... [truncated {len(obj) - max_str_len} chars]"
        return obj
    return obj


def _build_ast_tree(doc: Document) -> Tree:
    """Construct a hierarchical Rich Tree visualization for an SCL Document AST."""
    root = Tree(f"[bold cyan]Document[/bold cyan] (version={doc.version or 'none'})")

    # Identity
    id_label = f'[bold green]Identity[/bold green] (profile="{doc.identity.profile}"'
    if doc.identity.tone:
        id_label += f', tone="{doc.identity.tone}"'
    id_label += ")"
    id_node = root.add(id_label)
    body_snippet = doc.identity.body.strip()
    if len(body_snippet) > 80:
        body_snippet = body_snippet[:77] + "..."
    id_node.add(f"[dim]{body_snippet}[/dim]")

    # Variables
    if doc.variables:
        vars_node = root.add(f"[bold yellow]Variables[/bold yellow] ({len(doc.variables)})")
        for v in doc.variables:
            var_label = f'[yellow]{v.name}[/yellow] (source="{v.source}", type="{v.type}")'
            v_branch = vars_node.add(var_label)
            if v.template:
                tmpl = v.template.strip()
                if len(tmpl) > 60:
                    tmpl = tmpl[:57] + "..."
                v_branch.add(f"[dim]template: {tmpl}[/dim]")

    # Examples
    if doc.examples:
        exs_node = root.add(f"[bold magenta]Examples[/bold magenta] ({len(doc.examples)})")
        for i, ex in enumerate(doc.examples, start=1):
            ex_branch = exs_node.add(f"[magenta]Example {i}[/magenta] (pair={ex.pair})")
            in_snip = ex.input.strip().replace("\n", " ")
            if len(in_snip) > 50:
                in_snip = in_snip[:47] + "..."
            ex_branch.add(f"[cyan]input:[/cyan] {in_snip}")
            out_snip = ex.output.strip().replace("\n", " ")
            if len(out_snip) > 50:
                out_snip = out_snip[:47] + "..."
            ex_branch.add(f"[green]output:[/green] {out_snip}")

    # Output
    if doc.output:
        out_label = (
            f'[bold blue]Output[/bold blue] (format="{doc.output.format}", '
            f"strict={doc.output.strict})"
        )
        out_node = root.add(out_label)
        if doc.output.schema:
            out_node.add(f"[dim]schema keys: {list(doc.output.schema.keys())}[/dim]")
        elif doc.output.body.strip():
            b_snip = doc.output.body.strip().replace("\n", " ")
            if len(b_snip) > 60:
                b_snip = b_snip[:57] + "..."
            out_node.add(f"[dim]{b_snip}[/dim]")

    return root


class TrackingAdapter:
    """Delegating wrapper around an Adapter that accumulates token usage and call counts."""

    def __init__(self, inner: Adapter) -> None:
        self.inner = inner
        self.invocations = 0
        self.total_input_tokens = 0
        self.total_output_tokens = 0

    @property
    def capabilities(self) -> Any:
        return self.inner.capabilities

    @property
    def name(self) -> str:
        return self.inner.name

    def compile(self, bound_doc: Any, user_message: Any) -> CompiledRequest:
        return self.inner.compile(bound_doc, user_message)

    async def invoke(self, request: Any) -> LLMResult:
        self.invocations += 1
        if hasattr(request, "messages") and isinstance(request.messages, list):
            request.messages = [
                Message(role=m["role"], content=m["content"])
                if isinstance(m, dict) and "role" in m and "content" in m
                else m
                for m in request.messages
            ]
        result = await self.inner.invoke(request)
        if hasattr(result, "usage") and result.usage:
            self.total_input_tokens += getattr(result.usage, "input_tokens", 0)
            self.total_output_tokens += getattr(result.usage, "output_tokens", 0)
        return result


@scl_app.command("lint")
def lint_cmd(
    file: Annotated[
        Path,
        typer.Argument(
            help="Path to SCL file to lint",
            exists=True,
            file_okay=True,
            dir_okay=False,
            readable=True,
        ),
    ],
    strict: Annotated[
        bool,
        typer.Option("--strict", help="Treat warnings as errors"),
    ] = False,
) -> None:
    """Parse + validate an SCL file and display diagnostics with line, column, and caret."""
    source = file.read_text(encoding="utf-8")
    try:
        doc = parse(source, source_name=str(file), store_spans=True)
    except SCLSyntaxError as exc:
        console.print(
            f"[bold red]Syntax Error:[/bold red] {exc.message} at line {exc.line}, col {exc.col}"
        )
        console.print(format_source_caret(source, exc.line, exc.col))
        if exc.hint:
            console.print(f"[dim]Hint: {exc.hint}[/dim]")
        raise typer.Exit(code=1) from exc
    except Exception as exc:
        console.print(f"[bold red]Error parsing file:[/bold red] {exc}")
        raise typer.Exit(code=1) from exc

    diagnostics = validate(doc)
    has_errors = any(d.severity == "error" for d in diagnostics)
    has_warnings = any(d.severity == "warning" for d in diagnostics)

    if not diagnostics:
        console.print(f"[bold green]✓[/bold green] {file} passed linting.")
        return

    for diag in diagnostics:
        color = "red" if diag.severity == "error" else "yellow"
        console.print(
            f"[bold {color}][{diag.severity.upper()}][/bold {color}] "
            f"line {diag.line}, col {diag.col}: {diag.message}"
        )
        console.print(format_source_caret(source, diag.line, diag.col))
        if diag.hint:
            console.print(f"[dim]Hint: {diag.hint}[/dim]")

    if has_errors or (strict and has_warnings):
        raise typer.Exit(code=1)


@scl_app.command("ast")
def ast_cmd(
    file: Annotated[
        Path,
        typer.Argument(
            help="Path to SCL file",
            exists=True,
            file_okay=True,
            dir_okay=False,
            readable=True,
        ),
    ],
    as_json: Annotated[
        bool,
        typer.Option("--json", help="Print the AST as JSON"),
    ] = False,
) -> None:
    """Print the parsed SCL AST as a Rich tree or formatted JSON."""
    source = file.read_text(encoding="utf-8")
    try:
        doc = parse(source, source_name=str(file), store_spans=True)
    except SCLSyntaxError as exc:
        console.print(f"[bold red]Syntax Error:[/bold red] {exc.message}")
        console.print(format_source_caret(source, exc.line, exc.col))
        raise typer.Exit(code=1) from exc

    if as_json:
        console.print_json(doc.model_dump_json(indent=2))
    else:
        tree = _build_ast_tree(doc)
        console.print(tree)


@scl_app.command("compile")
def compile_cmd(
    file: Annotated[
        Path,
        typer.Argument(
            help="Path to SCL file",
            exists=True,
            file_okay=True,
            dir_okay=False,
            readable=True,
        ),
    ],
    alias: Annotated[
        str,
        typer.Option(
            "--alias",
            "-a",
            help="Model alias name from configuration",
        ),
    ],
    data: Annotated[
        Path,
        typer.Option(
            "--data",
            "-d",
            help="Path to JSON data file",
            exists=True,
            file_okay=True,
            dir_okay=False,
            readable=True,
        ),
    ],
    user_message: Annotated[
        str,
        typer.Option(
            "--user-message",
            "-u",
            help="Optional user message prompt",
        ),
    ] = "",
) -> None:
    """Bind data and compile for adapter alias without executing network calls."""
    load_env()
    source = file.read_text(encoding="utf-8")
    try:
        doc = parse(source, source_name=str(file))
    except SCLSyntaxError as exc:
        console.print(f"[bold red]Syntax Error:[/bold red] {exc.message}")
        raise typer.Exit(code=1) from exc

    try:
        data_payload = json.loads(data.read_text(encoding="utf-8"))
    except Exception as exc:
        console.print(f"[bold red]Error reading data JSON:[/bold red] {exc}")
        raise typer.Exit(code=1) from exc

    ctx = RuntimeContext(data_payload)
    try:
        bound_doc = bind(doc, ctx)
    except Exception as exc:
        console.print(f"[bold red]Binding Error:[/bold red] {exc}")
        raise typer.Exit(code=1) from exc

    try:
        adapter = get_adapter(alias)
    except Exception as exc:
        console.print(f"[bold red]Adapter Error:[/bold red] {exc}")
        raise typer.Exit(code=1) from exc

    compiled_req = adapter.compile(bound_doc, user_message)
    sanitized = _sanitize_for_display(compiled_req.model_dump())
    console.print_json(json.dumps(sanitized))


@scl_app.command("run")
def run_cmd(
    file: Annotated[
        Path,
        typer.Argument(
            help="Path to SCL file",
            exists=True,
            file_okay=True,
            dir_okay=False,
            readable=True,
        ),
    ],
    alias: Annotated[
        str,
        typer.Option(
            "--alias",
            "-a",
            help="Model alias name from configuration",
        ),
    ],
    data: Annotated[
        Path,
        typer.Option(
            "--data",
            "-d",
            help="Path to JSON data file",
            exists=True,
            file_okay=True,
            dir_okay=False,
            readable=True,
        ),
    ],
    user_message: Annotated[
        str,
        typer.Option(
            "--user-message",
            "-u",
            help="User message prompt",
        ),
    ],
) -> None:
    """Compile, invoke through invoke_with_repair, and print result and metrics."""
    load_env()
    source = file.read_text(encoding="utf-8")
    try:
        doc = parse(source, source_name=str(file))
    except SCLSyntaxError as exc:
        console.print(f"[bold red]Syntax Error:[/bold red] {exc.message}")
        raise typer.Exit(code=1) from exc

    try:
        data_payload = json.loads(data.read_text(encoding="utf-8"))
    except Exception as exc:
        console.print(f"[bold red]Error reading data JSON:[/bold red] {exc}")
        raise typer.Exit(code=1) from exc

    ctx = RuntimeContext(data_payload)
    try:
        bound_doc = bind(doc, ctx)
    except Exception as exc:
        console.print(f"[bold red]Binding Error:[/bold red] {exc}")
        raise typer.Exit(code=1) from exc

    try:
        adapter = get_adapter(alias)
    except Exception as exc:
        console.print(f"[bold red]Adapter Error:[/bold red] {exc}")
        raise typer.Exit(code=1) from exc

    if isinstance(adapter, FakeAdapter) and not adapter.script:
        if doc.examples:
            adapter.script = [doc.examples[0].output]
        elif doc.output and doc.output.format == "json_schema":
            adapter.script = ["{}"]
        else:
            adapter.script = ["OK"]

    compiled_req = adapter.compile(bound_doc, user_message)
    tracker = TrackingAdapter(adapter)
    output_node = (
        doc.output
        if doc.output is not None
        else SCLOutput(format="markdown", strict=False, body="")
    )

    try:
        result = asyncio.run(invoke_with_repair(tracker, compiled_req, output_node))
    except OutputViolation as exc:
        console.print(f"[bold red]Output Violation:[/bold red] {exc}")
        console.print(f"Repairs: {max(0, tracker.invocations - 1)}")
        raise typer.Exit(code=1) from exc
    except Exception as exc:
        console.print(f"[bold red]Invocation Error:[/bold red] {exc}")
        raise typer.Exit(code=1) from exc

    # Print parsed result
    if isinstance(result, (dict, list)):
        console.print_json(json.dumps(result))
    else:
        console.print(str(result))

    repairs = max(0, tracker.invocations - 1)
    console.print(f"[bold cyan]Input tokens:[/bold cyan] {tracker.total_input_tokens}")
    console.print(f"[bold cyan]Output tokens:[/bold cyan] {tracker.total_output_tokens}")
    console.print(f"[bold cyan]Repairs:[/bold cyan] {repairs}")
