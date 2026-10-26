"""Unit tests for UACP message router, circuit breaker, and in-process transport."""

from __future__ import annotations

from pathlib import Path

import pytest

from uacp.memory.whiteboard import InMemoryWhiteboardBackend, SqliteWhiteboardBackend
from uacp.memory.whiteboard.base import WhiteboardBackend
from uacp.protocol.capabilities import AgentCard, Capability, CapabilityRegistry
from uacp.protocol.circuit_breaker import CircuitBreaker
from uacp.protocol.jsonrpc import (
    JSONRPCErrorCode,
    JSONRPCRequest,
    JSONRPCResponse,
    build_response,
)
from uacp.protocol.messages import (
    CircuitBreakerState,
    Intent,
    Party,
    TaskDelegationPayload,
    UACPMessage,
)
from uacp.protocol.router import Router
from uacp.protocol.transport.inproc import InProcessTransport


@pytest.fixture
def inmem_whiteboard() -> InMemoryWhiteboardBackend:
    """Fixture providing a fresh in-memory whiteboard."""
    return InMemoryWhiteboardBackend()


@pytest.fixture
def sqlite_whiteboard(tmp_path: Path) -> SqliteWhiteboardBackend:
    """Fixture providing a fresh SQLite whiteboard."""
    db_path = str(tmp_path / "router_test_wb.db")
    return SqliteWhiteboardBackend(db_path=db_path)


# ── Transport Tests ─────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_inproc_transport_register_and_deliver() -> None:
    transport = InProcessTransport()

    async def echo_handler(req: JSONRPCRequest) -> JSONRPCResponse:
        return build_response(id=req.id, result={"echo": req.params.payload})

    transport.register_agent("agent_echo", echo_handler)
    assert transport.has_agent("agent_echo")
    assert "agent_echo" in transport.registered_agents()

    msg = UACPMessage(
        correlation_id="corr-1",
        trace_id="tr-1",
        sender=Party(agent_id="caller", engine="default"),
        recipient=Party(agent_id="agent_echo", engine="default"),
        intent=Intent.task_delegation,
        payload=TaskDelegationPayload(
            instruction="echo me",
            whiteboard_session_ref="sess-1",
        ).model_dump(),
        circuit_breaker=CircuitBreakerState(current_hop=1),
    )

    resp = await transport.deliver(msg)
    assert resp.error is None
    assert resp.result == {"echo": msg.payload}

    # Unknown agent
    unknown_msg = msg.model_copy(
        update={"recipient": Party(agent_id="unknown_agent", engine="default")}
    )
    unknown_resp = await transport.deliver(unknown_msg)
    assert unknown_resp.error is not None
    assert unknown_resp.error.code == JSONRPCErrorCode.CAPABILITY_NOT_FOUND


# ── Circuit Breaker & Router Delegation Hop Limit Tests ─────────────────────


@pytest.mark.asyncio
async def test_two_agents_delegating_back_and_forth_halt_at_hop_5(
    inmem_whiteboard: InMemoryWhiteboardBackend,
) -> None:
    """Two fake agents delegating back and forth must halt at exactly hop 5."""
    session_id = await inmem_whiteboard.create_session(ttl_s=60.0, max_hops=5)
    transport = InProcessTransport()
    router = Router(whiteboard=inmem_whiteboard, transport=transport, max_hops=5)

    hop_history: list[int] = []

    async def agent_alpha_handler(req: JSONRPCRequest) -> JSONRPCResponse:
        current_hop = req.params.circuit_breaker.current_hop
        hop_history.append(current_hop)
        next_msg = UACPMessage(
            correlation_id=req.params.correlation_id,
            trace_id=req.params.trace_id,
            sender=Party(agent_id="agent_alpha", engine="default"),
            recipient=Party(agent_id="agent_beta", engine="default"),
            intent=Intent.task_delegation,
            payload=TaskDelegationPayload(
                instruction=f"Alpha delegation turn at hop {current_hop + 1}",
                whiteboard_session_ref=session_id,
            ).model_dump(),
            circuit_breaker=req.params.circuit_breaker.next_hop(),
        )
        return await router.send(next_msg)

    async def agent_beta_handler(req: JSONRPCRequest) -> JSONRPCResponse:
        current_hop = req.params.circuit_breaker.current_hop
        hop_history.append(current_hop)
        next_msg = UACPMessage(
            correlation_id=req.params.correlation_id,
            trace_id=req.params.trace_id,
            sender=Party(agent_id="agent_beta", engine="default"),
            recipient=Party(agent_id="agent_alpha", engine="default"),
            intent=Intent.task_delegation,
            payload=TaskDelegationPayload(
                instruction=f"Beta delegation turn at hop {current_hop + 1}",
                whiteboard_session_ref=session_id,
            ).model_dump(),
            circuit_breaker=req.params.circuit_breaker.next_hop(),
        )
        return await router.send(next_msg)

    transport.register_agent("agent_alpha", agent_alpha_handler)
    transport.register_agent("agent_beta", agent_beta_handler)

    initial_msg = UACPMessage(
        correlation_id="corr-loop",
        trace_id="tr-loop",
        sender=Party(agent_id="agent_alpha", engine="default"),
        recipient=Party(agent_id="agent_beta", engine="default"),
        intent=Intent.task_delegation,
        payload=TaskDelegationPayload(
            instruction="Initial start turn at hop 1",
            whiteboard_session_ref=session_id,
        ).model_dump(),
        circuit_breaker=CircuitBreakerState(max_hops=5, current_hop=1),
    )

    final_resp = await router.send(initial_msg)

    # Must trip and halt at exactly hop 5
    assert final_resp.error is not None
    assert final_resp.error.code == JSONRPCErrorCode.HOP_LIMIT_EXCEEDED
    assert int(final_resp.error.code) == -32001
    assert "HopLimitExceeded" in final_resp.error.message

    # Hops delivered were 1, 2, 3, 4 before hop 5 tripped
    assert hop_history == [1, 2, 3, 4]

    # Verify session.halted event on whiteboard
    events = await inmem_whiteboard.events(session_id)
    halt_events = [e for e in events if e.event_type == "session.halted"]
    assert len(halt_events) == 1
    assert halt_events[0].data["code"] == -32001


@pytest.mark.asyncio
async def test_custom_max_hops(inmem_whiteboard: InMemoryWhiteboardBackend) -> None:
    """Configuring custom max_hops (e.g. 3) halts at exactly that limit."""
    session_id = await inmem_whiteboard.create_session(ttl_s=60.0, max_hops=3)
    transport = InProcessTransport()
    router = Router(whiteboard=inmem_whiteboard, transport=transport, max_hops=3)

    hops_received: list[int] = []

    async def ping_handler(req: JSONRPCRequest) -> JSONRPCResponse:
        current_hop = req.params.circuit_breaker.current_hop
        hops_received.append(current_hop)
        next_msg = UACPMessage(
            correlation_id=req.params.correlation_id,
            trace_id=req.params.trace_id,
            sender=Party(agent_id="ping", engine="default"),
            recipient=Party(agent_id="pong", engine="default"),
            intent=Intent.task_delegation,
            payload=TaskDelegationPayload(
                instruction=f"Ping to Pong step {current_hop + 1}",
                whiteboard_session_ref=session_id,
            ).model_dump(),
            circuit_breaker=req.params.circuit_breaker.next_hop(),
        )
        return await router.send(next_msg)

    async def pong_handler(req: JSONRPCRequest) -> JSONRPCResponse:
        current_hop = req.params.circuit_breaker.current_hop
        hops_received.append(current_hop)
        next_msg = UACPMessage(
            correlation_id=req.params.correlation_id,
            trace_id=req.params.trace_id,
            sender=Party(agent_id="pong", engine="default"),
            recipient=Party(agent_id="ping", engine="default"),
            intent=Intent.task_delegation,
            payload=TaskDelegationPayload(
                instruction=f"Pong to Ping step {current_hop + 1}",
                whiteboard_session_ref=session_id,
            ).model_dump(),
            circuit_breaker=req.params.circuit_breaker.next_hop(),
        )
        return await router.send(next_msg)

    transport.register_agent("ping", ping_handler)
    transport.register_agent("pong", pong_handler)

    initial_msg = UACPMessage(
        correlation_id="corr-pingpong",
        trace_id="tr-pingpong",
        sender=Party(agent_id="pong", engine="default"),
        recipient=Party(agent_id="ping", engine="default"),
        intent=Intent.task_delegation,
        payload=TaskDelegationPayload(
            instruction="Start pingpong step 1",
            whiteboard_session_ref=session_id,
        ).model_dump(),
        circuit_breaker=CircuitBreakerState(max_hops=3, current_hop=1),
    )

    final_resp = await router.send(initial_msg)
    assert final_resp.error is not None
    assert final_resp.error.code == JSONRPCErrorCode.HOP_LIMIT_EXCEEDED
    assert hops_received == [1, 2]


# ── Cycle Detection Tests ───────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_cycle_detection_halts_earlier_than_hop_limit(
    inmem_whiteboard: InMemoryWhiteboardBackend,
) -> None:
    """Cycle detection halts when the same (sender, recipient, payload) appears >= 2 times."""
    session_id = await inmem_whiteboard.create_session(ttl_s=60.0, max_hops=10)
    transport = InProcessTransport()
    router = Router(whiteboard=inmem_whiteboard, transport=transport, max_hops=10)

    async def agent_b_handler(req: JSONRPCRequest) -> JSONRPCResponse:
        return build_response(id=req.id, result={"acknowledged": True})

    transport.register_agent("agent_b", agent_b_handler)

    identical_payload = TaskDelegationPayload(
        instruction="Exact duplicate instruction",
        whiteboard_session_ref=session_id,
    ).model_dump()

    # First transmission: sender A -> recipient B (hop 1)
    msg1 = UACPMessage(
        correlation_id="corr-cycle",
        trace_id="tr-cycle",
        sender=Party(agent_id="agent_a", engine="default"),
        recipient=Party(agent_id="agent_b", engine="default"),
        intent=Intent.task_delegation,
        payload=identical_payload,
        circuit_breaker=CircuitBreakerState(max_hops=10, current_hop=1),
    )
    resp1 = await router.send(msg1)
    assert resp1.error is None

    # Intermediate message: different payload (hop 2)
    msg2 = UACPMessage(
        correlation_id="corr-cycle",
        trace_id="tr-cycle",
        sender=Party(agent_id="agent_a", engine="default"),
        recipient=Party(agent_id="agent_b", engine="default"),
        intent=Intent.task_delegation,
        payload=TaskDelegationPayload(
            instruction="Different instruction",
            whiteboard_session_ref=session_id,
        ).model_dump(),
        circuit_breaker=CircuitBreakerState(max_hops=10, current_hop=2),
    )
    resp2 = await router.send(msg2)
    assert resp2.error is None

    # Second transmission of identical payload: sender A -> recipient B (hop 3)
    msg3 = UACPMessage(
        correlation_id="corr-cycle",
        trace_id="tr-cycle",
        sender=Party(agent_id="agent_a", engine="default"),
        recipient=Party(agent_id="agent_b", engine="default"),
        intent=Intent.task_delegation,
        payload=identical_payload,
        circuit_breaker=CircuitBreakerState(max_hops=10, current_hop=3),
    )
    resp3 = await router.send(msg3)

    # Must trip cycle detection at hop 3, well before max_hops 10
    assert resp3.error is not None
    assert resp3.error.code == JSONRPCErrorCode.HOP_LIMIT_EXCEEDED
    assert "CycleDetected" in resp3.error.message

    # Verify session is halted
    assert router.breaker.is_session_halted(session_id)


# ── Anti-Spoofing Tests ─────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_spoofed_hop_value_rejected(
    inmem_whiteboard: InMemoryWhiteboardBackend,
) -> None:
    """Message claiming an advisory hop differing from Whiteboard count is rejected."""
    session_id = await inmem_whiteboard.create_session(ttl_s=60.0, max_hops=5)
    transport = InProcessTransport()
    router = Router(whiteboard=inmem_whiteboard, transport=transport, max_hops=5)

    async def worker_handler(req: JSONRPCRequest) -> JSONRPCResponse:
        return build_response(id=req.id, result={"done": True})

    transport.register_agent("worker", worker_handler)

    # Advance authoritative count to 1
    msg1 = UACPMessage(
        correlation_id="corr-spoof",
        trace_id="tr-spoof",
        sender=Party(agent_id="caller", engine="default"),
        recipient=Party(agent_id="worker", engine="default"),
        intent=Intent.task_delegation,
        payload=TaskDelegationPayload(
            instruction="legit hop 1",
            whiteboard_session_ref=session_id,
        ).model_dump(),
        circuit_breaker=CircuitBreakerState(max_hops=5, current_hop=1),
    )
    resp1 = await router.send(msg1)
    assert resp1.error is None

    # Next message should be hop 2, but malicious sender passes hop 1 (or 99)
    spoofed_msg = UACPMessage(
        correlation_id="corr-spoof",
        trace_id="tr-spoof",
        sender=Party(agent_id="caller", engine="default"),
        recipient=Party(agent_id="worker", engine="default"),
        intent=Intent.task_delegation,
        payload=TaskDelegationPayload(
            instruction="spoofed hop 1",
            whiteboard_session_ref=session_id,
        ).model_dump(),
        circuit_breaker=CircuitBreakerState(max_hops=5, current_hop=1),
    )
    resp_spoof = await router.send(spoofed_msg)

    assert resp_spoof.error is not None
    assert resp_spoof.error.code == JSONRPCErrorCode.HOP_LIMIT_EXCEEDED
    assert "AntiSpoofingViolation" in resp_spoof.error.message


# ── Halted Session Refuses Messages ─────────────────────────────────────────


@pytest.mark.asyncio
async def test_halted_session_refuses_messages(
    inmem_whiteboard: InMemoryWhiteboardBackend,
) -> None:
    """A session halted manually or via trip refuses all subsequent message deliveries."""
    session_id = await inmem_whiteboard.create_session(ttl_s=60.0, max_hops=5)
    transport = InProcessTransport()
    router = Router(whiteboard=inmem_whiteboard, transport=transport, max_hops=5)

    delivery_count = 0

    async def counting_handler(req: JSONRPCRequest) -> JSONRPCResponse:
        nonlocal delivery_count
        delivery_count += 1
        return build_response(id=req.id, result={"count": delivery_count})

    transport.register_agent("service", counting_handler)

    # 1. Normal send works
    msg1 = UACPMessage(
        correlation_id="corr-halt",
        trace_id="tr-halt",
        sender=Party(agent_id="caller", engine="default"),
        recipient=Party(agent_id="service", engine="default"),
        intent=Intent.task_delegation,
        payload=TaskDelegationPayload(
            instruction="initial task",
            whiteboard_session_ref=session_id,
        ).model_dump(),
        circuit_breaker=CircuitBreakerState(max_hops=5, current_hop=1),
    )
    resp1 = await router.send(msg1)
    assert resp1.error is None
    assert delivery_count == 1

    # 2. Manual halt
    await router.halt(session_id, "Manual supervisor emergency halt")
    assert router.breaker.is_session_halted(session_id)

    # 3. Subsequent message is rejected; handler is NOT invoked
    msg2 = UACPMessage(
        correlation_id="corr-halt",
        trace_id="tr-halt",
        sender=Party(agent_id="caller", engine="default"),
        recipient=Party(agent_id="service", engine="default"),
        intent=Intent.task_delegation,
        payload=TaskDelegationPayload(
            instruction="post halt task",
            whiteboard_session_ref=session_id,
        ).model_dump(),
        circuit_breaker=CircuitBreakerState(max_hops=5, current_hop=2),
    )
    resp2 = await router.send(msg2)
    assert resp2.error is not None
    assert resp2.error.code == JSONRPCErrorCode.HOP_LIMIT_EXCEEDED
    assert "Manual supervisor emergency halt" in resp2.error.message
    assert delivery_count == 1


# ── Recipient Resolution Tests ──────────────────────────────────────────────


@pytest.mark.asyncio
async def test_recipient_resolution_agent_id_vs_capability(
    inmem_whiteboard: InMemoryWhiteboardBackend,
) -> None:
    """Router resolves recipient by agent ID or capability (agent ID wins if both given)."""
    session_id = await inmem_whiteboard.create_session(ttl_s=60.0)
    transport = InProcessTransport()
    registry = CapabilityRegistry()
    router = Router(
        whiteboard=inmem_whiteboard,
        transport=transport,
        registry=registry,
    )

    received_by: list[str] = []

    async def math_handler(req: JSONRPCRequest) -> JSONRPCResponse:
        received_by.append("math_agent")
        return build_response(id=req.id, result={"solved": True})

    async def fallback_handler(req: JSONRPCRequest) -> JSONRPCResponse:
        received_by.append("fallback_agent")
        return build_response(id=req.id, result={"solved": True})

    transport.register_agent("math_agent", math_handler)
    transport.register_agent("fallback_agent", fallback_handler)

    card = AgentCard(
        agent_id="math_agent",
        engine="default",
        capabilities=[Capability(name="arithmetic", description="Solves math")],
    )
    registry.register(card)

    # Case A: Resolve by capability name
    cap_msg = UACPMessage(
        correlation_id="corr-res",
        trace_id="tr-res",
        sender=Party(agent_id="caller", engine="default"),
        recipient=Party(agent_id="", capability="arithmetic"),
        intent=Intent.task_delegation,
        payload=TaskDelegationPayload(
            instruction="calc 2+2",
            whiteboard_session_ref=session_id,
        ).model_dump(),
        circuit_breaker=CircuitBreakerState(current_hop=1),
    )
    resp_cap = await router.send(cap_msg)
    assert resp_cap.error is None
    assert received_by[-1] == "math_agent"
    assert "arithmetic" in router.history[-1][0].trace_id if hasattr(router, "history") else True

    # Case B: If both agent_id and capability are given, agent_id wins
    both_msg = UACPMessage(
        correlation_id="corr-res",
        trace_id="tr-res",
        sender=Party(agent_id="caller", engine="default"),
        recipient=Party(agent_id="fallback_agent", capability="arithmetic"),
        intent=Intent.task_delegation,
        payload=TaskDelegationPayload(
            instruction="calc 3+3",
            whiteboard_session_ref=session_id,
        ).model_dump(),
        circuit_breaker=CircuitBreakerState(current_hop=2),
    )
    resp_both = await router.send(both_msg)
    assert resp_both.error is None
    assert received_by[-1] == "fallback_agent"

    # Case C: Unknown capability returns -32002 CAPABILITY_NOT_FOUND
    unknown_cap_msg = UACPMessage(
        correlation_id="corr-res",
        trace_id="tr-res",
        sender=Party(agent_id="caller", engine="default"),
        recipient=Party(agent_id="", capability="quantum_teleportation"),
        intent=Intent.task_delegation,
        payload=TaskDelegationPayload(
            instruction="teleport",
            whiteboard_session_ref=session_id,
        ).model_dump(),
        circuit_breaker=CircuitBreakerState(current_hop=3),
    )
    resp_unknown = await router.send(unknown_cap_msg)
    assert resp_unknown.error is not None
    assert resp_unknown.error.code == JSONRPCErrorCode.CAPABILITY_NOT_FOUND


# ── Provider Quota & Session Expired Tests ──────────────────────────────────


@pytest.mark.asyncio
async def test_provider_quota_exhausted(inmem_whiteboard: InMemoryWhiteboardBackend) -> None:
    """When quota hook raises or returns False, breaker trips with code -32004."""
    session_id = await inmem_whiteboard.create_session(ttl_s=60.0)
    transport = InProcessTransport()

    def quota_hook(sess_id: str, msg: UACPMessage) -> bool:
        return False  # quota exhausted

    breaker = CircuitBreaker(whiteboard=inmem_whiteboard, quota_hook=quota_hook)
    router = Router(
        whiteboard=inmem_whiteboard,
        transport=transport,
        circuit_breaker=breaker,
    )

    msg = UACPMessage(
        correlation_id="corr-quota",
        trace_id="tr-quota",
        sender=Party(agent_id="caller", engine="default"),
        recipient=Party(agent_id="worker", engine="default"),
        intent=Intent.task_delegation,
        payload=TaskDelegationPayload(
            instruction="task",
            whiteboard_session_ref=session_id,
        ).model_dump(),
        circuit_breaker=CircuitBreakerState(current_hop=1),
    )

    resp = await router.send(msg)
    assert resp.error is not None
    assert resp.error.code == JSONRPCErrorCode.QUOTA_EXCEEDED
    assert int(resp.error.code) == -32004


@pytest.mark.asyncio
async def test_session_expired_trips_breaker(
    inmem_whiteboard: InMemoryWhiteboardBackend,
) -> None:
    """Accessing an expired or closed whiteboard session returns code -32005."""
    session_id = await inmem_whiteboard.create_session(ttl_s=60.0)
    await inmem_whiteboard.close(session_id)

    transport = InProcessTransport()
    router = Router(whiteboard=inmem_whiteboard, transport=transport)

    msg = UACPMessage(
        correlation_id="corr-expired",
        trace_id="tr-expired",
        sender=Party(agent_id="caller", engine="default"),
        recipient=Party(agent_id="worker", engine="default"),
        intent=Intent.task_delegation,
        payload=TaskDelegationPayload(
            instruction="task",
            whiteboard_session_ref=session_id,
        ).model_dump(),
        circuit_breaker=CircuitBreakerState(current_hop=1),
    )

    resp = await router.send(msg)
    assert resp.error is not None
    assert resp.error.code == JSONRPCErrorCode.SESSION_EXPIRED
    assert int(resp.error.code) == -32005


# ── Dual Whiteboard Backend Tests ───────────────────────────────────────────


@pytest.mark.asyncio
@pytest.mark.parametrize("backend_type", ["inmem", "sqlite"])
async def test_breaker_works_with_both_whiteboard_backends(
    backend_type: str,
    inmem_whiteboard: InMemoryWhiteboardBackend,
    sqlite_whiteboard: SqliteWhiteboardBackend,
) -> None:
    """Circuit breaker behaves identically on InMemory and SQLite backends."""
    wb: WhiteboardBackend = inmem_whiteboard if backend_type == "inmem" else sqlite_whiteboard
    session_id = await wb.create_session(ttl_s=60.0, max_hops=2)
    transport = InProcessTransport()
    router = Router(whiteboard=wb, transport=transport, max_hops=2)

    async def agent_x_handler(req: JSONRPCRequest) -> JSONRPCResponse:
        return build_response(id=req.id, result={"status": "ok"})

    transport.register_agent("agent_x", agent_x_handler)

    # Hop 1 succeeds
    msg1 = UACPMessage(
        correlation_id="corr-dual",
        trace_id="tr-dual",
        sender=Party(agent_id="agent_y", engine="default"),
        recipient=Party(agent_id="agent_x", engine="default"),
        intent=Intent.task_delegation,
        payload=TaskDelegationPayload(
            instruction="task 1",
            whiteboard_session_ref=session_id,
        ).model_dump(),
        circuit_breaker=CircuitBreakerState(max_hops=2, current_hop=1),
    )
    resp1 = await router.send(msg1)
    assert resp1.error is None

    # Hop 2 reaches max_hops 2 -> trips!
    msg2 = UACPMessage(
        correlation_id="corr-dual",
        trace_id="tr-dual",
        sender=Party(agent_id="agent_y", engine="default"),
        recipient=Party(agent_id="agent_x", engine="default"),
        intent=Intent.task_delegation,
        payload=TaskDelegationPayload(
            instruction="task 2",
            whiteboard_session_ref=session_id,
        ).model_dump(),
        circuit_breaker=CircuitBreakerState(max_hops=2, current_hop=2),
    )
    resp2 = await router.send(msg2)
    assert resp2.error is not None
    assert resp2.error.code == JSONRPCErrorCode.HOP_LIMIT_EXCEEDED

    # Verify session.halted event on the backend
    events = await wb.events(session_id)
    halted = [e for e in events if e.event_type == "session.halted"]
    assert len(halted) == 1
    assert halted[0].data["code"] == -32001
