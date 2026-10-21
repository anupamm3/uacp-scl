"""Script to auto-generate JSON schemas for UACP models."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from uacp.protocol.jsonrpc import JSONRPCRequest, JSONRPCResponse
from uacp.protocol.messages import (
    CapabilityQueryPayload,
    CapabilityResponsePayload,
    CircuitBreakerState,
    ErrorPayload,
    HandoffPayload,
    HeartbeatPayload,
    NegotiatePayload,
    Party,
    TaskDelegationPayload,
    TaskResultPayload,
    UACPMessage,
)

MODELS: dict[str, type[BaseModel]] = {
    "uacp_message.json": UACPMessage,
    "party.json": Party,
    "circuit_breaker_state.json": CircuitBreakerState,
    "jsonrpc_request.json": JSONRPCRequest,
    "jsonrpc_response.json": JSONRPCResponse,
    "payload_task_delegation.json": TaskDelegationPayload,
    "payload_task_result.json": TaskResultPayload,
    "payload_negotiate.json": NegotiatePayload,
    "payload_error.json": ErrorPayload,
    "payload_handoff.json": HandoffPayload,
    "payload_capability_query.json": CapabilityQueryPayload,
    "payload_capability_response.json": CapabilityResponsePayload,
    "payload_heartbeat.json": HeartbeatPayload,
}


def generate_schemas(output_dir: Path) -> None:
    """Generate JSON schema files for all registered models."""
    output_dir.mkdir(parents=True, exist_ok=True)
    for filename, model_cls in MODELS.items():
        schema: dict[str, Any] = model_cls.model_json_schema()
        file_path = output_dir / filename
        file_path.write_text(json.dumps(schema, indent=2) + "\n", encoding="utf-8")
        print(f"Generated {file_path}")


if __name__ == "__main__":
    target_dir = Path(__file__).resolve().parent.parent / "docs" / "schemas"
    generate_schemas(target_dir)

