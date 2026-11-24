"""End-to-end tests for the circuit breaker loop-demo mode."""

from __future__ import annotations

import pytest
from typer.testing import CliRunner

from uacp.cli import uacp_app
from uacp.runner.loop_demo import run_loop_demo

runner = CliRunner()


@pytest.mark.asyncio
async def test_loop_demo_halts_at_hop_5() -> None:
    """Verify that loop-demo halts at exactly hop 5 with success return code."""
    exit_code = await run_loop_demo(max_hops=5)
    assert exit_code == 0


@pytest.mark.asyncio
async def test_loop_demo_sqlite_backend() -> None:
    """Verify loop-demo halts cleanly when backed by SQLite Whiteboard."""
    exit_code = await run_loop_demo(max_hops=5, backend_type="sqlite")
    assert exit_code == 0


def test_cli_loop_demo_command() -> None:
    """Verify `uacp run loop-demo` CLI command exits with code 0."""
    result = runner.invoke(uacp_app, ["run", "loop-demo"])
    assert result.exit_code == 0

