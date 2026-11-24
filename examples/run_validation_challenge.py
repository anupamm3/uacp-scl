#!/usr/bin/env python3
"""Run the forensic validation challenge end-to-end."""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from uacp.adapters.registry import load_env  # noqa: E402
from uacp.runner.validation_challenge import run_validation_challenge  # noqa: E402


def main() -> None:
    load_env()
    parser = argparse.ArgumentParser(
        description="Run the end-to-end UACP forensic audit and remediation challenge."
    )
    parser.add_argument(
        "--alpha",
        default="alpha_default",
        help="Model alias for Agent Alpha (default: 'alpha_default')",
    )
    parser.add_argument(
        "--beta",
        default="beta_default",
        help="Model alias for Agent Beta (default: 'beta_default')",
    )
    parser.add_argument(
        "--backend",
        choices=["inmem", "sqlite"],
        default="inmem",
        help="Whiteboard backend: 'inmem' or 'sqlite' (default: 'inmem')",
    )
    parser.add_argument(
        "--replay",
        default=None,
        help="Path to pre-recorded JSON file for offline replay",
    )
    parser.add_argument(
        "--record",
        default=None,
        help="Path to save recorded LLM interactions for later replay",
    )
    parser.add_argument(
        "--offline",
        action="store_true",
        help="Run offline using built-in recorded fixtures (no API keys or internet needed)",
    )
    parser.add_argument(
        "--yes",
        "-y",
        action="store_true",
        help="Automatically confirm quota warning for live runs",
    )
    parser.add_argument(
        "--metrics-out",
        default=None,
        help="Optional path to write run metrics JSON",
    )

    args = parser.parse_args()

    exit_code, _, _ = asyncio.run(
        run_validation_challenge(
            alpha_alias=args.alpha,
            beta_alias=args.beta,
            backend_type=args.backend,
            replay_file=args.replay,
            record_file=args.record,
            offline=args.offline,
            yes=args.yes,
            metrics_out=args.metrics_out,
        )
    )
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
