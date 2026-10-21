"""Unit tests for UACP message models and intent payloads."""

from __future__ import annotations

import time
from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from uacp.protocol.messages import (
    CircuitBreakerState,
    Intent,
    Party,
    TaskDelegationPayload,
    TaskResultPayload,
    UACPMessage,
    generate_message_id,
    validate_payload,
)


def test_blueprint_sample_message_validates() -> None:
    """Verify that the standard sample message from the blueprint validates cleanly."""
    raw = {
        "uacp_version": "1.0",
        "message_id": "01729555200000-8f921ab39c4d",
        "correlation_id": "corr-audit-session-202610",
        "trace_id": "trace-7729-381a",
        "sender": {
            "agent_id": "agent_alpha",
            "engine": "alpha_default:gemini-2.5-flash",
        },
        "recipient": {
            "agent_id": "agent_beta",
            "engine": "beta_default:llama-3.3-70b-versatile",
        },
        "intent": "task_delegation",
        "payload": {
            "instruction": "Mitigate duplicate transaction anomalies in batch 4.",
            "whiteboard_session_ref": "wb-session-alpha-4421",
            "expected_output_schema": {
                "type": "object",
                "properties": {
                    "code": {"type": "string"},
                    "rationale": {"type": "string"},
                },
                "required": ["code", "rationale"],
            },
            "deadline_s": 30.0,
        },
        "circuit_breaker": {
            "max_hops": 5,
            "current_hop": 1,
        },
        "created_at": "2026-10-22T00:50:00Z",
    }

    msg = UACPMessage.model_validate(raw)
    assert msg.uacp_version == "1.0"
    assert msg.message_id == "01729555200000-8f921ab39c4d"
    assert msg.sender.agent_id == "agent_alpha"
    assert msg.recipient.agent_id == "agent_beta"
    assert msg.intent == Intent.task_delegation
    assert msg.circuit_breaker.current_hop == 1
    assert msg.circuit_breaker.max_hops == 5
    assert not msg.circuit_breaker.is_exhausted
    assert msg.created_at.tzinfo == UTC

    # Typed payload validation
    payload_model = validate_payload(msg)
    assert isinstance(payload_model, TaskDelegationPayload)
    assert payload_model.whiteboard_session_ref == "wb-session-alpha-4421"
    assert payload_model.deadline_s == 30.0


def test_sortable_message_id_order() -> None:
    """Ensure generated message IDs are unique and lexicographically sortable by time."""
    id1 = generate_message_id()
    time.sleep(0.005)
    id2 = generate_message_id()
    assert id1 != id2
    assert id1 < id2


def test_utc_timezone_awareness() -> None:
    """Ensure created_at timestamps are always timezone-aware in UTC."""
    # Default created_at
    msg = UACPMessage(
        correlation_id="c1",
        trace_id="t1",
        sender=Party(agent_id="a1", engine="gemini"),
        recipient=Party(agent_id="a2", engine="groq"),
        intent=Intent.heartbeat,
        payload={},
    )
    assert msg.created_at.tzinfo is not None
    assert msg.created_at.tzinfo == UTC

    # Naive datetime provided
    naive_dt = datetime(2026, 10, 22, 12, 0, 0)
    msg_naive = UACPMessage(
        correlation_id="c1",
        trace_id="t1",
        sender=Party(agent_id="a1", engine="gemini"),
        recipient=Party(agent_id="a2", engine="groq"),
        intent=Intent.heartbeat,
        payload={},
        created_at=naive_dt,
    )
    assert msg_naive.created_at.tzinfo == UTC
    assert msg_naive.created_at.hour == 12


def test_circuit_breaker_behavior() -> None:
    """Verify CircuitBreakerState defaults, exhaustion checks, and incrementation."""
    cb = CircuitBreakerState()
    assert cb.max_hops == 5
    assert cb.current_hop == 0
    assert not cb.is_exhausted

    # Advance hops
    cb1 = cb.next_hop()
    assert cb1.current_hop == 1
    assert cb1.max_hops == 5

    exhausted = CircuitBreakerState(max_hops=3, current_hop=3)
    assert exhausted.is_exhausted

    over_exhausted = CircuitBreakerState(max_hops=3, current_hop=4)
    assert over_exhausted.is_exhausted


def test_payload_validation_per_intent() -> None:
    """Verify payload validation against registered intent models."""
    base_msg = {
        "correlation_id": "c1",
        "trace_id": "t1",
        "sender": {"agent_id": "a", "engine": "e"},
        "recipient": {"agent_id": "b", "engine": "e"},
    }

    # 1. Task Delegation
    msg_delegation = UACPMessage.model_validate(
        {
            **base_msg,
            "intent": "task_delegation",
            "payload": {
                "instruction": "Run tests",
                "whiteboard_session_ref": "wb-1",
            },
        }
    )
    payload_del = validate_payload(msg_delegation)
    assert isinstance(payload_del, TaskDelegationPayload)

    # 2. Task Result
    msg_result = UACPMessage.model_validate(
        {
            **base_msg,
            "intent": "task_result",
            "payload": {
                "status": "success",
                "summary": "Completed successfully",
                "result_ref": "wb-res-1",
            },
        }
    )
    payload_res = validate_payload(msg_result)
    assert isinstance(payload_res, TaskResultPayload)
    assert payload_res.status == "success"

    # 3. Invalid payload schema triggers ValidationError
    bad_msg = UACPMessage.model_validate(
        {
            **base_msg,
            "intent": "task_delegation",
            "payload": {"invalid_field": 123},  # missing instruction and whiteboard_session_ref
        }
    )
    with pytest.raises(ValidationError):
        validate_payload(bad_msg)
