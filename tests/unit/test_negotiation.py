"""Unit tests for bilateral task negotiation, circuit breaker hops, and session handoffs."""

from __future__ import annotations

import pytest

from uacp.agents.base import Agent
from uacp.memory.whiteboard.client import WhiteboardClient
from uacp.memory.whiteboard.inmem_backend import InMemoryWhiteboardBackend
from uacp.protocol.capabilities import Capability, CapabilityRegistry
from uacp.protocol.circuit_breaker import CircuitBreaker
from uacp.protocol.messages import Intent, Party, UACPMessage
from uacp.protocol.negotiation import (
    DefaultNegotiationPolicy,
    NegotiationAction,
    NegotiationDecision,
    NegotiationProposal,
    execute_negotiation,
)
from uacp.protocol.router import Router
from uacp.protocol.transport.inproc import InProcessTransport


@pytest.fixture
def test_setup() -> tuple[InMemoryWhiteboardBackend, Router, CapabilityRegistry]:
    """Fixture providing in-memory whiteboard, router, and registry."""
    wb = InMemoryWhiteboardBackend()
    transport = InProcessTransport()
    registry = CapabilityRegistry()
    router = Router(whiteboard=wb, transport=transport, registry=registry)
    return wb, router, registry


@pytest.mark.asyncio
async def test_negotiation_accept(
    test_setup: tuple[InMemoryWhiteboardBackend, Router, CapabilityRegistry],
) -> None:
    """Verify negotiation immediately accepts when complexity, strengths, and load match."""
    wb, router, registry = test_setup
    session_id = await wb.create_session(ttl_s=300.0)

    cap = Capability(
        name="code_analysis",
        strengths=["python", "security"],
        complexity_level=4,
    )
    receiver = Agent(
        agent_id="security_bot",
        capabilities=[cap],
        current_load=1,
    )
    receiver.register(registry, router)

    proposal = NegotiationProposal(
        subtask_id="task-001",
        subtask_text="Audit authentication endpoint",
        difficulty=3,
        required_strengths=["security"],
        deadline_s=30.0,
        token_budget=2000,
    )

    outcome = await execute_negotiation(
        router=router,
        session_id=session_id,
        initiator_agent_id="coordinator",
        responder_agent_id="security_bot",
        proposal=proposal,
    )

    assert outcome.agreed is True
    assert outcome.final_decision.action == NegotiationAction.accept
    assert outcome.final_decision.reason_code == "ACCEPTED"
    assert outcome.rounds_completed == 1

    # Verify written to Whiteboard
    state = await wb.get_state(session_id)
    assert "negotiation.task-001" in state
    val = state["negotiation.task-001"]["value"]
    assert val["agreed"] is True
    assert val["final_decision"]["action"] == "accept"


@pytest.mark.asyncio
async def test_negotiation_counter_then_accept(
    test_setup: tuple[InMemoryWhiteboardBackend, Router, CapabilityRegistry],
) -> None:
    """Verify receiver counters by 1 difficulty level or load, and initiator accepts."""
    wb, router, registry = test_setup
    session_id = await wb.create_session(ttl_s=300.0)

    # Agent with complexity 3
    cap = Capability(
        name="audit",
        strengths=["accounting"],
        complexity_level=3,
    )
    receiver = Agent(
        agent_id="junior_auditor",
        capabilities=[cap],
        current_load=0,
    )
    receiver.register(registry, router)

    # Proposed difficulty 4 (gap of 1 -> receiver will counter with difficulty 3)
    proposal = NegotiationProposal(
        subtask_id="task-002",
        subtask_text="Audit quarterly report",
        difficulty=4,
        required_strengths=["accounting"],
        deadline_s=60.0,
        token_budget=5000,
    )

    # Custom initiator that accepts counter-offers
    class InitiatorPolicy(DefaultNegotiationPolicy):
        def evaluate(self, prop, ctx, round_num=1):
            if prop.difficulty <= 3:
                return NegotiationDecision(
                    action=NegotiationAction.accept,
                    reason_code="COUNTER_ACCEPTED",
                    round=round_num,
                )
            return super().evaluate(prop, ctx, round_num)

    initiator = Agent(
        agent_id="senior_auditor",
        capabilities=[Capability(name="lead", complexity_level=5)],
        negotiation_policy=InitiatorPolicy(),
    )
    initiator.register(registry, router)

    # Step 1: Responder receives proposal and counters with difficulty 3
    msg = UACPMessage(
        correlation_id=session_id,
        trace_id="tr-neg-1",
        sender=Party(agent_id="senior_auditor"),
        recipient=Party(agent_id="junior_auditor"),
        intent=Intent.negotiate,
        payload={"proposal": proposal.model_dump(), "round": 1},
    )
    res = await router.send(msg, session_id=session_id)
    assert res.result is not None
    dec1 = res.result["payload"]["decision"]
    assert dec1["action"] == "counter"
    assert dec1["proposal"]["difficulty"] == 3
    assert dec1["reason_code"] == "DIFFICULTY_ADJUSTED"

    # Step 2: Initiator evaluates counter proposal and accepts
    counter_prop = NegotiationProposal.model_validate(dec1["proposal"])
    eval_dec = initiator.evaluate_counter(counter_prop, round_num=2)
    assert eval_dec.action == NegotiationAction.accept
    assert eval_dec.reason_code == "COUNTER_ACCEPTED"


@pytest.mark.asyncio
async def test_negotiation_reject_strength_mismatch(
    test_setup: tuple[InMemoryWhiteboardBackend, Router, CapabilityRegistry],
) -> None:
    """Verify rejection when agent has zero matching strengths."""
    wb, router, registry = test_setup
    session_id = await wb.create_session(ttl_s=300.0)

    cap = Capability(
        name="writer",
        strengths=["copywriting", "creative"],
        complexity_level=3,
    )
    receiver = Agent(
        agent_id="writer_agent",
        capabilities=[cap],
    )
    receiver.register(registry, router)

    proposal = NegotiationProposal(
        subtask_id="task-003",
        subtask_text="Run kernel vulnerability scan",
        difficulty=3,
        required_strengths=["kernel_security"],
        deadline_s=30.0,
        token_budget=1000,
    )

    outcome = await execute_negotiation(
        router=router,
        session_id=session_id,
        initiator_agent_id="coordinator",
        responder_agent_id="writer_agent",
        proposal=proposal,
    )

    assert outcome.agreed is False
    assert outcome.final_decision.action == NegotiationAction.reject
    assert outcome.final_decision.reason_code == "STRENGTH_MISMATCH"


@pytest.mark.asyncio
async def test_negotiation_no_agreement_after_round_limit(
    test_setup: tuple[InMemoryWhiteboardBackend, Router, CapabilityRegistry],
) -> None:
    """Verify negotiation terminates with explicit no_agreement after round limit."""
    wb, router, registry = test_setup
    session_id = await wb.create_session(ttl_s=300.0)

    # Receiver always counters
    class AlwaysCounterPolicy(DefaultNegotiationPolicy):
        def evaluate(self, prop, ctx, round_num=1):
            if round_num > 2:
                return NegotiationDecision(
                    action=NegotiationAction.no_agreement,
                    reason_code="MAX_ROUNDS_REACHED",
                    round=round_num,
                )
            new_p = prop.model_copy(update={"deadline_s": prop.deadline_s + 10.0})
            return NegotiationDecision(
                action=NegotiationAction.counter,
                proposal=new_p,
                reason_code="EXTEND_DEADLINE",
                round=round_num,
            )

    receiver = Agent(
        agent_id="counter_agent",
        capabilities=[Capability(name="cap1", strengths=["st1"], complexity_level=3)],
        negotiation_policy=AlwaysCounterPolicy(),
    )
    receiver.register(registry, router)

    proposal = NegotiationProposal(
        subtask_id="task-004",
        subtask_text="Complex task",
        difficulty=3,
        required_strengths=["st1"],
        deadline_s=20.0,
        token_budget=1000,
    )

    outcome = await execute_negotiation(
        router=router,
        session_id=session_id,
        initiator_agent_id="coordinator",
        responder_agent_id="counter_agent",
        proposal=proposal,
        max_rounds=2,
    )

    assert outcome.agreed is False
    assert outcome.final_decision.action == NegotiationAction.no_agreement
    assert outcome.final_decision.reason_code == "MAX_ROUNDS_REACHED"


@pytest.mark.asyncio
async def test_negotiation_circuit_breaker_hop_limit(
    test_setup: tuple[InMemoryWhiteboardBackend, Router, CapabilityRegistry],
) -> None:
    """Verify each message counts as a hop and trips breaker into explicit no_agreement."""
    wb, router, registry = test_setup
    # Session limit set to only 1 hop
    session_id = await wb.create_session(ttl_s=300.0, max_hops=1)
    breaker = CircuitBreaker(whiteboard=wb, max_hops=1)
    router.breaker = breaker

    receiver = Agent(
        agent_id="worker_agent",
        capabilities=[Capability(name="cap", strengths=["tag"], complexity_level=2)],
    )
    receiver.register(registry, router)

    proposal = NegotiationProposal(
        subtask_id="task-005",
        subtask_text="Subtask test",
        difficulty=2,
        required_strengths=["tag"],
        deadline_s=30.0,
        token_budget=1000,
    )

    # First increment authoritative hop to trip the breaker limit
    await wb.incr_hop(session_id)

    outcome = await execute_negotiation(
        router=router,
        session_id=session_id,
        initiator_agent_id="coordinator",
        responder_agent_id="worker_agent",
        proposal=proposal,
        circuit_breaker=breaker,
    )

    assert outcome.agreed is False
    assert outcome.final_decision.action == NegotiationAction.no_agreement
    assert outcome.final_decision.reason_code == "CIRCUIT_BREAKER_TRIPPED"


@pytest.mark.asyncio
async def test_handoff_valid_hash_chain() -> None:
    """Verify handoff succeeds when hash chain integrity is valid."""
    wb = InMemoryWhiteboardBackend()
    session_id = await wb.create_session(ttl_s=300.0)
    client = WhiteboardClient(wb, session_id=session_id, author="agent_a")
    await client.put("key1", {"status": "ready"})

    transport = InProcessTransport()
    router = Router(whiteboard=wb, transport=transport)

    agent_b = Agent(
        agent_id="agent_b",
        whiteboard=WhiteboardClient(wb, session_id=session_id, author="agent_b"),
        router=router,
    )
    transport.register_agent("agent_b", agent_b._on_transport_request)

    agent_a = Agent(
        agent_id="agent_a",
        whiteboard=client,
        router=router,
    )

    res = await agent_a.handoff(
        target_agent_id="agent_b",
        reason="Phase 1 complete",
        session_id=session_id,
    )

    assert res.error is None
    assert res.result is not None
    assert res.result["payload"]["status"] == "success"
    assert res.result["payload"]["is_valid"] is True


@pytest.mark.asyncio
async def test_handoff_tampered_hash_chain() -> None:
    """Verify handoff is rejected when hash chain is tampered."""
    wb = InMemoryWhiteboardBackend()
    session_id = await wb.create_session(ttl_s=300.0)
    client = WhiteboardClient(wb, session_id=session_id, author="agent_a")
    await client.put("key1", {"value": 1})
    await client.put("key2", {"value": 2})

    # Tamper with recorded event log directly
    session = wb._sessions[session_id]
    assert len(session.events) >= 2
    # Modify the second event's recorded state_hash
    tampered_event = session.events[1].model_copy(update={"state_hash": "bad_hash_999"})
    session.events[1] = tampered_event

    transport = InProcessTransport()
    router = Router(whiteboard=wb, transport=transport)

    agent_b = Agent(
        agent_id="agent_b",
        whiteboard=WhiteboardClient(wb, session_id=session_id, author="agent_b"),
        router=router,
    )
    transport.register_agent("agent_b", agent_b._on_transport_request)

    agent_a = Agent(
        agent_id="agent_a",
        whiteboard=client,
        router=router,
    )

    res = await agent_a.handoff(
        target_agent_id="agent_b",
        reason="Handoff after tamper",
        session_id=session_id,
    )

    assert res.result is not None
    assert res.result["payload"]["status"] == "rejected"
    assert "TamperDetected" in res.result["payload"]["summary"]
    assert res.result["payload"]["is_valid"] is False


@pytest.mark.asyncio
async def test_observer_cannot_write_after_handoff() -> None:
    """Verify an agent transitioned to observer mode cannot write to Whiteboard."""
    wb = InMemoryWhiteboardBackend()
    session_id = await wb.create_session(ttl_s=300.0)
    client = WhiteboardClient(wb, session_id=session_id, author="agent_a")

    agent = Agent(
        agent_id="agent_a",
        whiteboard=client,
    )

    assert agent.whiteboard is not None

    # Initial write succeeds
    v1 = await agent.whiteboard.put("init_key", "initial_value")
    assert v1 == 1

    # Transition to observer mode
    agent.set_observer(True)
    assert agent.is_observer is True
    assert agent.whiteboard is not None

    # Reads still work
    val = await agent.whiteboard.get("init_key")
    assert val == "initial_value"

    # Writes are blocked and raise PermissionError
    with pytest.raises(PermissionError, match="Observer agent cannot write to Whiteboard"):
        await agent.whiteboard.put("new_key", "new_value")

    with pytest.raises(PermissionError, match="Observer agent cannot write to Whiteboard"):
        await agent.whiteboard.write_model("model_key", client)  # type: ignore[arg-type]
