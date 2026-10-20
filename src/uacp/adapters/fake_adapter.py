"""Fake adapter for testing and deterministic offline replays."""

from __future__ import annotations

import contextlib
import json
from pathlib import Path
from typing import Any

from uacp.adapters.base import (
    Adapter,
    AdapterCapabilities,
    CompiledRequest,
    LLMResult,
    Message,
    Usage,
    build_data_blocks,
    build_identity_system,
    render_examples,
)
from uacp.scl.context import BoundDocument


class FakeAdapter(Adapter):
    """Deterministic LLM adapter that serves scripted replies and records invocations."""

    def __init__(
        self,
        script: list[str | dict[str, Any] | Exception] | None = None,
        *,
        name: str = "fake",
        model_id: str = "fake-echo-v1",
        capabilities: AdapterCapabilities | None = None,
    ) -> None:
        self.name = name
        self.model_id = model_id
        self.capabilities = capabilities or AdapterCapabilities(
            supports_native_schema=True,
            supports_system_role=True,
            max_context=32768,
        )
        self.script: list[str | dict[str, Any] | Exception] = (
            list(script) if script is not None else []
        )
        self.recorded_requests: list[CompiledRequest] = []

    def compile(self, bound_doc: BoundDocument, user_message: str | Message) -> CompiledRequest:
        """Compile a bound SCL document into a canonical CompiledRequest."""
        user_content = (
            user_message.content if isinstance(user_message, Message) else str(user_message)
        )

        messages: list[Message] = []
        system: str | None = None

        if self.capabilities.supports_system_role:
            sys_text = build_identity_system(bound_doc)
            system = sys_text if sys_text else None
        else:
            sys_text = build_identity_system(bound_doc)
            if sys_text:
                messages.append(Message(role="user", content=sys_text))
                messages.append(Message(role="assistant", content="Acknowledged."))

        # Add examples
        for inp, outp in render_examples(bound_doc):
            messages.append(Message(role="user", content=inp))
            messages.append(Message(role="assistant", content=outp))

        # Add bound data blocks and user message
        data_blocks = build_data_blocks(bound_doc)
        if data_blocks:
            full_user = f"{data_blocks}\n\n{user_content}" if user_content else data_blocks
        else:
            full_user = user_content

        if full_user:
            messages.append(Message(role="user", content=full_user))

        # Determine output mode and response schema
        schema: dict[str, Any] | None = None
        output_mode: Any = "prompt_only"

        output_node = bound_doc.document.output
        if output_node is not None:
            schema = output_node.schema
            if schema is None and output_node.body.strip():
                with contextlib.suppress(Exception):
                    schema = json.loads(output_node.body)

            if output_node.format == "json_schema":
                output_mode = (
                    "native_schema" if self.capabilities.supports_native_schema else "json_mode"
                )
            else:
                output_mode = "prompt_only"

        return CompiledRequest(
            provider=self.name,
            model=self.model_id,
            messages=messages,
            system=system,
            response_schema=schema,
            output_mode=output_mode,
        )

    async def invoke(self, request: CompiledRequest) -> LLMResult:
        """Record the request and return the next scripted result or raise a scripted exception."""
        self.recorded_requests.append(request)

        if not self.script:
            raise IndexError("No scripted replies remaining in FakeAdapter")

        reply = self.script.pop(0)
        if isinstance(reply, Exception):
            raise reply

        parsed: dict[str, Any] | list[Any] | None = None
        if isinstance(reply, dict):
            text = json.dumps(reply)
            parsed = reply
        else:
            text = str(reply)
            try:
                candidate = json.loads(text)
                if isinstance(candidate, (dict, list)):
                    parsed = candidate
            except (json.JSONDecodeError, ValueError):
                parsed = None

        # Calculate synthetic token counts (~4 chars per token)
        total_input_chars = sum(len(m.content) for m in request.messages)
        if request.system:
            total_input_chars += len(request.system)

        input_tokens = max(1, total_input_chars // 4)
        output_tokens = max(1, len(text) // 4)
        usage = Usage(input_tokens=input_tokens, output_tokens=output_tokens, requests=1)

        return LLMResult(
            text=text,
            parsed=parsed if isinstance(parsed, dict) else None,
            usage=usage,
            raw_meta={"scripted": True, "remaining_script_items": len(self.script)},
        )

    @classmethod
    def from_file(cls, path: str | Path, **kwargs: Any) -> FakeAdapter:
        """Load scripted replies from a JSON file for playback demonstrations."""
        file_path = Path(path)
        with open(file_path, encoding="utf-8") as f:
            data = json.load(f)

        if isinstance(data, list):
            script = data
        elif isinstance(data, dict):
            script = data.get("replies", data.get("script", []))
        else:
            raise ValueError(f"Unexpected JSON structure in {path}: expected list or dict")

        return cls(script=script, **kwargs)
