"""Smoke test script for GeminiAdapter.

Performs ONE real call with a tiny prompt and prints the parsed result,
only when the GEMINI_API_KEY environment variable is set.
Never logs or prints the API key.
"""

from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path

import dotenv

from uacp.adapters.registry import get_adapter
from uacp.scl.context import RuntimeContext, bind
from uacp.scl.parser import parse

TINY_SCL = """<scl version="1">
  <identity profile="health_checker">
    You are a concise JSON health-check agent.
  </identity>
  <output format="json_schema" strict="true">
    {
      "type": "object",
      "required": ["status", "alive"],
      "properties": {
        "status": {"type": "string"},
        "alive": {"type": "boolean"}
      }
    }
  </output>
</scl>"""


async def main() -> int:
    """Execute smoke test if GEMINI_API_KEY is present."""
    project_root = Path(__file__).resolve().parent.parent
    dotenv.load_dotenv(project_root / ".env")

    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("[INFO] GEMINI_API_KEY not set. Skipping live Gemini smoke test.")
        return 0

    print("[INFO] GEMINI_API_KEY detected. Initializing GeminiAdapter for 'alpha_default'...")
    try:
        adapter = get_adapter("alpha_default")
        doc = parse(TINY_SCL)
        bound_doc = bind(doc, RuntimeContext({}))
        compiled = adapter.compile(bound_doc, "Perform a brief system status check.")

        print(f"[INFO] Invoking model '{compiled.model}' via GeminiAdapter...")
        result = await adapter.invoke(compiled)

        print("[SUCCESS] Live call completed!")
        print(f"  Parsed JSON : {result.parsed}")
        print(f"  Input Tokens: {result.usage.input_tokens}")
        print(f"  Output Tokens: {result.usage.output_tokens}")
        return 0
    except Exception as exc:
        print(f"[ERROR] Live smoke test failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
