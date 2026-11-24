"""Task ownership transfer (handoff) verification, protocol helpers, and event logging."""

from __future__ import annotations

import contextlib
from typing import Any

from uacp.memory.whiteboard.base import WhiteboardBackend
from uacp.protocol.jsonrpc import JSONRPCResponse
from uacp.protocol.messages import (
    CircuitBreakerState,
    HandoffPayload,
    Intent,
    Party,
    UACPMessage,
)
from uacp.protocol.router import Router


async def verify_and_accept_handoff(
    whiteboard: WhiteboardBackend,
    session_id: str,
    previous_owner: str,
    new_owner: str,
    payload: HandoffPayload | dict[str, Any],
) -> tuple[bool, str]:
    """Verify hash chain integrity and latest tip hash, transferring ownership if valid.

    Args:
        whiteboard: Whiteboard backend holding session state and hash chain.
        session_id: Target Whiteboard session identifier.
        previous_owner: Agent ID transferring ownership.
        new_owner: Agent ID receiving ownership.
        payload: HandoffPayload or dictionary containing last_hash, summary, and metadata.

    Returns:
        Tuple of (is_valid: bool, reason_or_details: str).
    """
    handoff_data = (
        HandoffPayload.model_validate(payload)
        if isinstance(payload, dict)
        else payload
    )

    # 1. Verify cryptographic hash chain integrity and state tamper status
    report = await whiteboard.verify_chain(session_id)
    if not report.is_valid:
        with contextlib.suppress(Exception):
            await whiteboard.append_event(
                session_id,
                {
                    "event_type": "session.handoff.rejected",
                    "data": {
                        "previous_owner": previous_owner,
                        "new_owner": new_owner,
                        "reason": f"Tampered hash chain (TamperDetected): {report.details}",
                        "corrupted_event_id": report.corrupted_event_id,
                    },
                },
            )
        return False, f"Tampered hash chain (TamperDetected): {report.details}"

    # 2. Confirm current chain tip matches last_hash
    current_tip = await whiteboard.get_latest_hash(session_id)
    if handoff_data.last_hash and current_tip != handoff_data.last_hash:
        err_msg = (
            f"Hash chain tip mismatch: expected '{handoff_data.last_hash}', "
            f"found '{current_tip}'"
        )
        with contextlib.suppress(Exception):
            await whiteboard.append_event(
                session_id,
                {
                    "event_type": "session.handoff.rejected",
                    "data": {
                        "previous_owner": previous_owner,
                        "new_owner": new_owner,
                        "reason": err_msg,
                        "expected_hash": handoff_data.last_hash,
                        "actual_hash": current_tip,
                    },
                },
            )
        return False, err_msg

    # 3. Record session.handoff event on Whiteboard
    await whiteboard.append_event(
        session_id,
        {
            "event_type": "session.handoff",
            "data": {
                "previous_owner": previous_owner,
                "new_owner": new_owner,
                "summary": handoff_data.summary,
                "last_hash": handoff_data.last_hash,
                "next_action": handoff_data.next_action,
                "metadata": handoff_data.metadata,
            },
        },
    )

    # 4. Mark previous owner as an observer for this session
    await whiteboard.add_observer(session_id, previous_owner)

    return True, f"Handoff accepted: ownership transferred to {new_owner}"


async def execute_handoff(
    router: Router,
    session_id: str,
    sender_agent_id: str,
    recipient_agent_id: str,
    summary: str,
    next_action: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> JSONRPCResponse:
    """Execute a task handoff message through the router to a recipient agent.

    Args:
        router: Router for message dispatch.
        session_id: Active Whiteboard session being handed off.
        sender_agent_id: Current owner agent ID.
        recipient_agent_id: Recipient agent ID receiving ownership.
        summary: Summary of progress, results, or rationale for handoff.
        next_action: Optional suggested next action for the recipient.
        metadata: Optional auxiliary parameters.

    Returns:
        JSONRPCResponse from the recipient agent.
    """
    last_hash = await router.whiteboard.get_latest_hash(session_id)

    payload = HandoffPayload(
        whiteboard_session_ref=session_id,
        last_hash=last_hash,
        summary=summary,
        next_action=next_action,
        metadata=metadata or {},
        target_agent=recipient_agent_id,
        reason=summary,
        context_ref=session_id,
    )

    msg = UACPMessage(
        correlation_id=session_id,
        trace_id=f"handoff-{session_id}",
        sender=Party(agent_id=sender_agent_id),
        recipient=Party(agent_id=recipient_agent_id),
        intent=Intent.handoff,
        payload=payload.model_dump(),
        circuit_breaker=CircuitBreakerState(
            max_hops=router.breaker.get_max_hops(session_id),
            current_hop=0,
        ),
    )

    return await router.send(
        msg,
        recipient_agent_id=recipient_agent_id,
        session_id=session_id,
    )
