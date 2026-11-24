"""Recording and deterministic replay adapters for offline challenge reproduction."""

from __future__ import annotations

import hashlib
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
)
from uacp.adapters.fake_adapter import FakeAdapter
from uacp.memory.whiteboard.base import compute_canonical_json
from uacp.scl.context import BoundDocument


def compute_request_hash(request: CompiledRequest) -> str:
    """Compute a deterministic SHA-256 hash of a compiled request without secrets."""
    canonical_dict = {
        "provider": request.provider,
        "model": request.model,
        "system": request.system,
        "messages": [{"role": m.role, "content": m.content} for m in request.messages],
        "response_schema": request.response_schema,
        "output_mode": request.output_mode,
    }
    canonical_str = compute_canonical_json(canonical_dict)
    return hashlib.sha256(canonical_str.encode("utf-8")).hexdigest()


class RecordingAdapter(Adapter):
    """Adapter wrapper that records every invocation request hash and reply."""

    def __init__(self, inner: Adapter, alias: str, manager: RecordManager) -> None:
        self.inner = inner
        self.alias = alias
        self.manager = manager
        self.name = inner.name
        self.capabilities = inner.capabilities

    def compile(self, bound_doc: BoundDocument, user_message: str | Message) -> CompiledRequest:
        return self.inner.compile(bound_doc, user_message)

    async def invoke(self, request: CompiledRequest) -> LLMResult:
        result = await self.inner.invoke(request)
        req_hash = compute_request_hash(request)
        record = {
            "alias": self.alias,
            "provider": request.provider,
            "model": request.model,
            "request_hash": req_hash,
            "reply": {
                "text": result.text,
                "parsed": result.parsed,
                "usage": {
                    "input_tokens": result.usage.input_tokens,
                    "output_tokens": result.usage.output_tokens,
                    "requests": result.usage.requests,
                },
            },
        }
        self.manager.add_record(record)
        return result


class ReplayAdapter(FakeAdapter):
    """Plays back pre-recorded LLM results matching request hash or alias order."""

    def __init__(
        self,
        alias: str,
        records: list[dict[str, Any]],
        *,
        name: str | None = None,
        model_id: str | None = None,
    ) -> None:
        self.alias = alias
        self.records = [dict(r) for r in records if r.get("alias") == alias or not r.get("alias")]
        # Also maintain a hash lookup table
        self.hash_lookup: dict[str, list[dict[str, Any]]] = {}
        for r in self.records:
            h = r.get("request_hash")
            if h:
                self.hash_lookup.setdefault(h, []).append(r)

        matched_name = name or (self.records[0].get("provider") if self.records else "fake")
        matched_model = model_id or (
            self.records[0].get("model") if self.records else "replay-model"
        )

        super().__init__(
            script=[],
            name=matched_name or "fake",
            model_id=matched_model or "replay-model",
            capabilities=AdapterCapabilities(
                supports_native_schema=True,
                supports_system_role=True,
                max_context=131072,
            ),
        )

    async def invoke(self, request: CompiledRequest) -> LLMResult:
        self.recorded_requests.append(request)
        req_hash = compute_request_hash(request)

        # 1. Look up by exact request hash if possible
        matched_record: dict[str, Any] | None = None
        if self.hash_lookup.get(req_hash):
            matched_record = self.hash_lookup[req_hash].pop(0)
            if matched_record in self.records:
                self.records.remove(matched_record)
        elif self.records:
            # 2. Fall back to sequential order for this alias
            matched_record = self.records.pop(0)

        if matched_record is None:
            raise IndexError(
                f"No replay records remaining for alias '{self.alias}' "
                f"(request_hash={req_hash[:12]})"
            )

        reply_dict = matched_record.get("reply", {})
        text = str(reply_dict.get("text", ""))
        parsed = reply_dict.get("parsed")
        usage_dict = reply_dict.get("usage", {})

        usage = Usage(
            input_tokens=int(usage_dict.get("input_tokens", max(1, len(text) // 4))),
            output_tokens=int(usage_dict.get("output_tokens", max(1, len(text) // 4))),
            requests=int(usage_dict.get("requests", 1)),
        )

        return LLMResult(
            text=text,
            parsed=parsed if isinstance(parsed, (dict, list)) else None,
            usage=usage,
            raw_meta={"replayed": True, "alias": self.alias, "request_hash": req_hash},
        )


class RecordManager:
    """Manages recording sessions and offline replay instantiation."""

    def __init__(self) -> None:
        self.records: list[dict[str, Any]] = []

    def add_record(self, record: dict[str, Any]) -> None:
        """Add an execution record."""
        self.records.append(record)

    def wrap_adapter(self, alias: str, adapter: Adapter) -> RecordingAdapter:
        """Wrap an adapter with recording capability."""
        return RecordingAdapter(inner=adapter, alias=alias, manager=self)

    def save(self, path: str | Path) -> None:
        """Save recorded interactions to a JSON file."""
        file_path = Path(path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "version": "1.0",
            "records": self.records,
        }
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)

    @classmethod
    def load_records(cls, path: str | Path) -> list[dict[str, Any]]:
        """Load records list from JSON file."""
        file_path = Path(path)
        with open(file_path, encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict):
            return list(data.get("records") or [])
        if isinstance(data, list):
            return data
        return []

    @classmethod
    def create_replay_adapter(cls, alias: str, path: str | Path) -> ReplayAdapter:
        """Create a ReplayAdapter for a given alias from a recorded file."""
        records = cls.load_records(path)
        return ReplayAdapter(alias=alias, records=records)
