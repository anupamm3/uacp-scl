"""Unit tests for capability negotiation, task handoff, and observer write protection."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from uacp.memory.whiteboard.base import ObserverWriteError
from uacp.memory.whiteboard.inmem_backend import InMemoryWhiteboardBackend
from uacp.memory.whiteboard.sqlite_backend import SqliteWhiteboardBackend
from uacp.protocol.capabilities import AgentCard, Capability
from uacp.protocol.handoff import execute_handoff, verify_and_accept_handoff
from uacp.protocol.messages import (
    HandoffPayload,
    NegotiatePayload,
)
from uacp.protocol.negotiation import (
    AgentNegotiationContext,
    DefaultNegotiationPolicy,
    NegotiationAction,
)
from uacp.protocol.router import Router
from uacp.protocol.transport.inproc import InProcessTransport


def _build_test_card(agent_id: str = "agent_beta") -> AgentCard:
    """Build an AgentCard with known capabilities and constraints."""
    cap = Capability(
        name="audit_ledger",
        description="Audit double-entry ledgers for anomalies",
        cost_hint=1.0,
        avg_latency_ms=1000.0,
        complexity_level=3,
        strengths=["forensics", "finance"],
        max_input_chars=5000,
    )
    return AgentCard(
        agent_id=agent_id,
        engine="default",
        role="Auditor",
        current_load=1,
        capabilities=[cap],
    )


def test_negotiation_direct_accept() -> None:
    """Verify proposal matching agent capability and constraints is accepted."""
    card = _build_test_card()
    context = AgentNegotiationContext(
        agent_id=card.agent_id,
        complexity_level=3,
        strengths=["forensics", "finance"],
        current_load=card.current_load,
        capabilities=["audit_ledger"],
        card=card,
    )

    policy = DefaultNegotiationPolicy()
    payload = NegotiatePayload(
        task="audit_ledger",
        constraints={"deadline_s": 5.0, "max_tokens": 2000, "complexity": 2},
        round=1,
        action="propose",
    )

    decision = policy.evaluate(payload, context, round_num=1)
    assert decision.action == NegotiationAction.accept
    assert decision.reason_code == "ACCEPTED"


def test_negotiation_counter_then_accept() -> None:
    """Verify short deadline triggers counter-offer, which is accepted in round 2."""
    card = _build_test_card()
    context = AgentNegotiationContext(
        agent_id=card.agent_id,
        complexity_level=3,
        strengths=["forensics", "finance"],
        current_load=card.current_load,
        capabilities=["audit_ledger"],
        card=card,
    )

    policy = DefaultNegotiationPolicy()

    # Round 1: Deadline of 0.5s is below average latency (1000ms = 1.0s)
    payload_r1 = NegotiatePayload(
        task="audit_ledger",
        constraints={"deadline_s": 0.5, "max_tokens": 1000},
        round=1,
        action="propose",
    )
    decision_r1 = policy.evaluate(payload_r1, context, round_num=1)
    assert decision_r1.action == NegotiationAction.counter
    assert decision_r1.reason_code == "DEADLINE_TOO_SHORT"
    assert decision_r1.modified_constraints is not None
    counter_deadline = decision_r1.modified_constraints["deadline_s"]
    assert counter_deadline >= 1.0

    # Round 2: Initiator accepts the countered deadline
    payload_r2 = NegotiatePayload(
        task="audit_ledger",
        constraints={"deadline_s": counter_deadline, "max_tokens": 1000},
        round=2,
        action="propose",
    )
    decision_r2 = policy.evaluate(payload_r2, context, round_num=2)
    assert decision_r2.action == NegotiationAction.accept
    assert decision_r2.reason_code == "ACCEPTED"


def test_negotiation_reject_missing_capability() -> None:
    """Verify proposing an unknown capability causes rejection."""
    card = _build_test_card()
    context = AgentNegotiationContext(
        agent_id=card.agent_id,
        complexity_level=3,
        strengths=["forensics"],
        current_load=1,
        capabilities=["audit_ledger"],
        card=card,
    )

    policy = DefaultNegotiationPolicy()
    payload = NegotiatePayload(
        task="quantum_synthesis",
        constraints={"deadline_s": 10.0},
        round=1,
        action="propose",
    )

    decision = policy.evaluate(payload, context, round_num=1)
    assert decision.action == NegotiationAction.reject
    assert decision.reason_code == "CAPABILITY_NOT_FOUND"


def test_negotiation_reject_infeasible_complexity() -> None:
    """Verify complexity gap > 1 causes rejection."""
    card = _build_test_card()
    context = AgentNegotiationContext(
        agent_id=card.agent_id,
        complexity_level=1,
        strengths=["forensics"],
        current_load=1,
        capabilities=["audit_ledger"],
        card=card,
    )

    policy = DefaultNegotiationPolicy()
    payload = NegotiatePayload(
        task="audit_ledger",
        constraints={"complexity": 5},
        round=1,
        action="propose",
    )

    decision = policy.evaluate(payload, context, round_num=1)
    assert decision.action == NegotiationAction.reject
    assert decision.reason_code == "COMPLEXITY_EXCEEDED"


def test_negotiation_exceeding_max_rounds_halts() -> None:
    """Verify exceeding 2 negotiation rounds terminates with no_agreement."""
    card = _build_test_card()
    context = AgentNegotiationContext(
        agent_id=card.agent_id,
        complexity_level=3,
        strengths=["forensics"],
        current_load=1,
        capabilities=["audit_ledger"],
        card=card,
    )

    policy = DefaultNegotiationPolicy(max_rounds=2)
    payload = NegotiatePayload(
        task="audit_ledger",
        constraints={"deadline_s": 5.0},
        round=3,
        action="propose",
    )

    decision = policy.evaluate(payload, context, round_num=3)
    assert decision.action == NegotiationAction.no_agreement
    assert decision.reason_code == "MAX_ROUNDS_REACHED"


@pytest.mark.asyncio
async def test_handoff_valid_hash_chain_succeeds() -> None:
    """Verify handoff transfers ownership and records session.handoff when chain is valid."""
    wb = InMemoryWhiteboardBackend()
    session_id = await wb.create_session(ttl_s=3600)

    # Put state entries to build a valid hash chain
    await wb.put(session_id, "data.1", {"x": 10}, author="AgentAlpha")
    await wb.put(session_id, "data.2", {"y": 20}, author="AgentAlpha")

    last_hash = await wb.get_latest_hash(session_id)
    payload = HandoffPayload(
        whiteboard_session_ref=session_id,
        last_hash=last_hash,
        summary="Analysis completed, handing off execution",
    )

    is_valid, msg = await verify_and_accept_handoff(
        whiteboard=wb,
        session_id=session_id,
        previous_owner="AgentAlpha",
        new_owner="AgentBeta",
        payload=payload,
    )

    assert is_valid is True
    assert "ownership transferred to AgentBeta" in msg

    # Verify AgentAlpha is marked as observer
    assert await wb.is_observer(session_id, "AgentAlpha") is True
    assert await wb.is_observer(session_id, "AgentBeta") is False

    # Verify session.handoff event was appended
    events = await wb.events(session_id)
    handoff_events = [e for e in events if e.event_type == "session.handoff"]
    assert len(handoff_events) == 1
    assert handoff_events[0].data["previous_owner"] == "AgentAlpha"
    assert handoff_events[0].data["new_owner"] == "AgentBeta"


@pytest.mark.asyncio
async def test_handoff_tampered_whiteboard_rejected() -> None:
    """Verify handoff is rejected if the Whiteboard hash chain has been tampered with."""
    wb = InMemoryWhiteboardBackend()
    session_id = await wb.create_session(ttl_s=3600)

    await wb.put(session_id, "key1", "val1", author="AgentAlpha")
    last_hash = await wb.get_latest_hash(session_id)

    # Tamper with internal event chain hash
    session = wb._get_session(session_id)
    tampered_event = session.events[-1].model_copy(update={"hash": "corrupted_hash_0000"})
    session.events[-1] = tampered_event

    payload = HandoffPayload(
        whiteboard_session_ref=session_id,
        last_hash=last_hash,
        summary="Transferring with tampered log",
    )

    is_valid, msg = await verify_and_accept_handoff(
        whiteboard=wb,
        session_id=session_id,
        previous_owner="AgentAlpha",
        new_owner="AgentBeta",
        payload=payload,
    )

    assert is_valid is False
    assert "Tampered hash chain" in msg
    assert await wb.is_observer(session_id, "AgentAlpha") is False

    # Verify rejection event
    events = await wb.events(session_id)
    rejected_events = [e for e in events if e.event_type == "session.handoff.rejected"]
    assert len(rejected_events) == 1


@pytest.mark.asyncio
async def test_handoff_mismatched_last_hash_rejected() -> None:
    """Verify handoff is rejected if caller supplies mismatched last_hash."""
    wb = InMemoryWhiteboardBackend()
    session_id = await wb.create_session(ttl_s=3600)
    await wb.put(session_id, "key1", "val1", author="AgentAlpha")

    payload = HandoffPayload(
        whiteboard_session_ref=session_id,
        last_hash="mismatched_fake_hash_tip",
        summary="Mismatched tip",
    )

    is_valid, msg = await verify_and_accept_handoff(
        whiteboard=wb,
        session_id=session_id,
        previous_owner="AgentAlpha",
        new_owner="AgentBeta",
        payload=payload,
    )

    assert is_valid is False
    assert "Hash chain tip mismatch" in msg
    assert await wb.is_observer(session_id, "AgentAlpha") is False


@pytest.mark.asyncio
async def test_observer_write_protection_in_memory() -> None:
    """Verify observer cannot write to in-memory Whiteboard session after handoff."""
    wb = InMemoryWhiteboardBackend()
    session_id = await wb.create_session(ttl_s=3600)

    # AgentAlpha writes initially
    await wb.put(session_id, "initial", "ok", author="AgentAlpha")

    # Mark AgentAlpha as observer
    await wb.add_observer(session_id, "AgentAlpha")
    assert await wb.is_observer(session_id, "AgentAlpha") is True

    # Subsequent put by AgentAlpha is blocked
    with pytest.raises(ObserverWriteError, match="is an observer on session"):
        await wb.put(session_id, "new_key", "blocked", author="AgentAlpha")

    # Another agent (AgentBeta) can still write
    ver = await wb.put(session_id, "beta_key", "allowed", author="AgentBeta")
    assert ver >= 1


@pytest.mark.asyncio
async def test_observer_write_protection_sqlite(tmp_path: Path) -> None:
    """Verify observer cannot write to SQLite Whiteboard session after handoff."""
    db_file = tmp_path / "test_wb.db"
    wb = SqliteWhiteboardBackend(db_path=db_file)
    try:
        session_id = await wb.create_session(ttl_s=3600)

        # Initial write
        await wb.put(session_id, "init_key", "init_val", author="AgentAlpha")

        # Add observer
        await wb.add_observer(session_id, "AgentAlpha")
        assert await wb.is_observer(session_id, "AgentAlpha") is True

        # Write attempt by observer raises ObserverWriteError
        with pytest.raises(ObserverWriteError, match="is an observer on session"):
            await wb.put(session_id, "fail_key", "fail_val", author="AgentAlpha")

        # Other agent can write
        ver = await wb.put(session_id, "beta_key", "success", author="AgentBeta")
        assert ver >= 1
    finally:
        wb.close_db()


@pytest.mark.asyncio
async def test_execute_handoff_integration() -> None:
    """Verify execute_handoff dispatches message through router and completes
    ownership transfer."""
    wb = InMemoryWhiteboardBackend()
    session_id = await wb.create_session(ttl_s=3600)
    await wb.put(session_id, "data", "val", author="Alpha")

    transport = InProcessTransport()
    router = Router(whiteboard=wb, transport=transport)

    async def _handle_handoff(req: Any) -> Any:
        msg = req.params
        payload = HandoffPayload.model_validate(msg.payload)
        is_valid, reason = await verify_and_accept_handoff(
            whiteboard=wb,
            session_id=payload.whiteboard_session_ref,
            previous_owner=msg.sender.agent_id,
            new_owner="Beta",
            payload=payload,
        )
        return {
            "status": "success" if is_valid else "rejected",
            "reason": reason,
        }

    transport.register_agent("Beta", _handle_handoff)

    res = await execute_handoff(
        router=router,
        session_id=session_id,
        sender_agent_id="Alpha",
        recipient_agent_id="Beta",
        summary="Task handoff to Beta",
    )
    assert res.error is None
    assert isinstance(res.result, dict)
    assert res.result["status"] == "success"
    assert await wb.is_observer(session_id, "Alpha") is True

