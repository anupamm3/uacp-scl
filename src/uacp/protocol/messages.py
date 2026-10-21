"""UACP message types and typed intent payload models."""

from __future__ import annotations

import time
import uuid
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator


def _utc_now() -> datetime:
    """Return the current timezone-aware UTC datetime."""
    return datetime.now(UTC)


def generate_message_id() -> str:
    """Generate a monotonically sortable unique message ID.

    The format begins with a zero-padded 14-digit millisecond timestamp (UTC),
    followed by a random hex suffix, ensuring lexicographical sorting matches
    chronological generation order.
    """
    ts_ms = int(time.time() * 1000)
    rand_suffix = uuid.uuid4().hex[:12]
    return f"{ts_ms:014d}-{rand_suffix}"


class Intent(StrEnum):
    """Enumeration of standard UACP message intents."""

    task_delegation = "task_delegation"
    task_result = "task_result"
    handoff = "handoff"
    negotiate = "negotiate"
    capability_query = "capability_query"
    capability_response = "capability_response"
    error = "error"
    heartbeat = "heartbeat"


class Party(BaseModel):
    """Represents an agent participant (sender or recipient)."""

    model_config = ConfigDict(frozen=True)

    agent_id: str = Field(..., description="Unique agent identifier.")
    engine: str = Field(..., description="LLM provider or engine moniker backing the agent.")


class CircuitBreakerState(BaseModel):
    """Tracks circuit-breaker hop counts across agent delegation chains."""

    model_config = ConfigDict(frozen=True)

    max_hops: int = Field(default=5, ge=1, description="Maximum hops before breaker trips.")
    current_hop: int = Field(default=0, ge=0, description="Current hop index in delegation path.")

    @property
    def is_exhausted(self) -> bool:
        """Return True if current hop reached or exceeded maximum allowed hops."""
        return self.current_hop >= self.max_hops

    def next_hop(self) -> CircuitBreakerState:
        """Return a new CircuitBreakerState incremented by one hop."""
        return CircuitBreakerState(max_hops=self.max_hops, current_hop=self.current_hop + 1)


class UACPMessage(BaseModel):
    """Core message model for the Unified Agent Communication Protocol (UACP)."""

    model_config = ConfigDict(extra="ignore")

    uacp_version: str = Field(default="1.0", description="UACP protocol specification version.")
    message_id: str = Field(
        default_factory=generate_message_id,
        description="Sortable unique message identifier.",
    )
    correlation_id: str = Field(
        ...,
        description="Correlation ID shared across a multi-turn conversation or task chain.",
    )
    trace_id: str = Field(..., description="Distributed tracing identifier.")
    sender: Party = Field(..., description="Sender party identification.")
    recipient: Party = Field(..., description="Recipient party identification.")
    intent: Intent = Field(..., description="Intent of this message.")
    payload: dict[str, Any] = Field(
        default_factory=dict,
        description="Intent-specific payload payload dictionary.",
    )
    circuit_breaker: CircuitBreakerState = Field(
        default_factory=CircuitBreakerState,
        description="Circuit breaker hop tracking state.",
    )
    created_at: datetime = Field(
        default_factory=_utc_now,
        description="Timezone-aware UTC timestamp when message was created.",
    )

    @field_validator("created_at", mode="before")
    @classmethod
    def _ensure_utc_aware(cls, v: Any) -> datetime:
        """Ensure created_at datetime is timezone-aware and converted to UTC."""
        if isinstance(v, str):
            dt = datetime.fromisoformat(v)
        elif isinstance(v, datetime):
            dt = v
        else:
            raise ValueError(f"Invalid datetime format: {v}")

        if dt.tzinfo is None:
            return dt.replace(tzinfo=UTC)
        return dt.astimezone(UTC)


# ── Typed Intent Payloads ───────────────────────────────────────────────


class TaskDelegationPayload(BaseModel):
    """Payload for delegating a task to another agent."""

    instruction: str = Field(..., description="Task instructions and prompt context.")
    whiteboard_session_ref: str = Field(
        ...,
        description="Whiteboard session key holding shared data context.",
    )
    expected_output_schema: dict[str, Any] | None = Field(
        default=None,
        description="Optional JSON Schema the recipient's output should conform to.",
    )
    deadline_s: float | None = Field(
        default=None,
        gt=0,
        description="Optional task execution deadline in seconds.",
    )


class TaskResultPayload(BaseModel):
    """Payload returning the outcome of a delegated task."""

    status: str = Field(..., description="Execution status ('success', 'failure', 'partial').")
    summary: str = Field(..., description="Summary explanation of findings or execution.")
    result_ref: str | None = Field(
        default=None,
        description="Reference key on Whiteboard or artifact location holding result data.",
    )


class NegotiatePayload(BaseModel):
    """Payload for bilateral capability or parameter negotiation."""

    proposal: dict[str, Any] | str = Field(..., description="Proposed parameters or constraints.")
    constraints: dict[str, Any] | list[str] | None = Field(
        default=None,
        description="Active constraints or requirements.",
    )
    round: int = Field(default=1, ge=1, description="Negotiation round index.")


class ErrorPayload(BaseModel):
    """Payload conveying protocol or application-level errors."""

    code: int = Field(..., description="Numeric error code.")
    message: str = Field(..., description="Descriptive human-readable error message.")
    data: Any = Field(default=None, description="Optional diagnostic or trace details.")


class HandoffPayload(BaseModel):
    """Payload for transferring session ownership or responsibility to another agent."""

    target_agent: str = Field(..., description="Agent ID receiving responsibility.")
    reason: str = Field(..., description="Reason for initiating the handoff.")
    context_ref: str | None = Field(
        default=None,
        description="Whiteboard session or hash reference holding state.",
    )


class CapabilityQueryPayload(BaseModel):
    """Payload querying an agent's supported intents or domain capabilities."""

    query: list[str] = Field(
        default_factory=list,
        description="Optional list of specific capability keys queried.",
    )


class CapabilityResponsePayload(BaseModel):
    """Payload advertising supported capabilities and metadata."""

    capabilities: list[str] | dict[str, Any] = Field(
        ...,
        description="Supported capabilities list or detailed capability mapping.",
    )


class HeartbeatPayload(BaseModel):
    """Payload for liveness and health checks."""

    status: str = Field(default="ok", description="Status string (e.g. 'ok', 'degraded').")
    timestamp: datetime = Field(
        default_factory=_utc_now,
        description="Heartbeat generation timestamp.",
    )

    @field_validator("timestamp", mode="before")
    @classmethod
    def _ensure_utc_aware(cls, v: Any) -> datetime:
        if isinstance(v, str):
            dt = datetime.fromisoformat(v)
        elif isinstance(v, datetime):
            dt = v
        else:
            raise ValueError(f"Invalid datetime format: {v}")

        if dt.tzinfo is None:
            return dt.replace(tzinfo=UTC)
        return dt.astimezone(UTC)


INTENT_PAYLOAD_MAP: dict[Intent, type[BaseModel]] = {
    Intent.task_delegation: TaskDelegationPayload,
    Intent.task_result: TaskResultPayload,
    Intent.handoff: HandoffPayload,
    Intent.negotiate: NegotiatePayload,
    Intent.capability_query: CapabilityQueryPayload,
    Intent.capability_response: CapabilityResponsePayload,
    Intent.error: ErrorPayload,
    Intent.heartbeat: HeartbeatPayload,
}


def validate_payload(message: UACPMessage) -> BaseModel:
    """Validate a message's dictionary payload against the model registered for its intent.

    Args:
        message: The UACPMessage containing an intent and payload dict.

    Returns:
        The validated typed BaseModel instance.

    Raises:
        pydantic.ValidationError: If payload fails validation against the intent model.
        KeyError: If intent is not registered in INTENT_PAYLOAD_MAP.
    """
    model_cls = INTENT_PAYLOAD_MAP[message.intent]
    return model_cls.model_validate(message.payload)
