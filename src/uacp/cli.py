"""UACP command-line interface."""

import typer

app = typer.Typer(help="UACP command-line interface.")


@app.callback()
def main() -> None:
    """UACP command-line interface."""
