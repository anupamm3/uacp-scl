"""Smoke test script for OpenAICompatAdapter targeting Groq.

Performs ONE real call with a tiny prompt and prints the parsed result,
only when the GROQ_API_KEY environment variable is set.
Never logs or prints the API key.
"""

from __future__ import annotations

import asyncio
import os
import sys

from uacp.adapters.registry import get_adapter, load_env
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
    """Execute smoke test if GROQ_API_KEY is present."""
    load_env()

    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        print("[INFO] GROQ_API_KEY not set. Skipping live Groq smoke test.")
        return 0

    print("[INFO] GROQ_API_KEY detected. Initializing OpenAICompatAdapter for 'beta_default'...")
    try:
        adapter = get_adapter("beta_default")
        doc = parse(TINY_SCL)
        bound_doc = bind(doc, RuntimeContext({}))
        compiled = adapter.compile(bound_doc, "Perform a brief system status check.")

        print(f"[INFO] Invoking model '{compiled.model}' via {adapter.__class__.__name__}...")
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

