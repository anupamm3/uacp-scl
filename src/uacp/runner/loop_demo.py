"""Demonstration of runaway delegation prevention via CircuitBreaker at hop 5."""

from __future__ import annotations

import re
import uuid
from typing import Any

from rich.console import Console

from uacp.memory.whiteboard.base import WhiteboardBackend
from uacp.memory.whiteboard.inmem_backend import InMemoryWhiteboardBackend
from uacp.memory.whiteboard.sqlite_backend import SqliteWhiteboardBackend
from uacp.protocol.circuit_breaker import CircuitBreaker
from uacp.protocol.jsonrpc import JSONRPCRequest, JSONRPCResponse
from uacp.protocol.messages import (
    CircuitBreakerState,
    Intent,
    Party,
    TaskDelegationPayload,
    UACPMessage,
)
from uacp.protocol.transport.inproc import InProcessTransport
from uacp.runner.trace import RunnerTracer, TracingRouter, TracingWhiteboardBackend


async def run_loop_demo(
    max_hops: int = 5,
    backend_type: str = "inmem",
    console: Console | None = None,
) -> int:
    """Run two scripted agents delegating back and forth until the CircuitBreaker halts at hop 5.

    Returns:
        0 on successful demonstration (breaker halted at expected hop limit), non-zero otherwise.
    """
    c = console or Console()
    tracer = RunnerTracer(console=c)
    tracer.print_section("UACP Circuit Breaker Runaway Delegation Demo")
    tracer.on_info(
        f"Initializing safety demo with max_hops={max_hops}. "
        f"Two agents will ping-pong delegations."
    )

    # 1. Whiteboard session setup
    raw_backend: WhiteboardBackend
    if backend_type == "sqlite":
        raw_backend = SqliteWhiteboardBackend(":memory:")
    else:
        raw_backend = InMemoryWhiteboardBackend()

    whiteboard = TracingWhiteboardBackend(raw_backend, tracer=tracer)
    session_id = await whiteboard.create_session(ttl_s=3600.0, max_hops=max_hops)

    # 2. Transport, Breaker, and Router setup
    transport = InProcessTransport()
    breaker = CircuitBreaker(whiteboard=whiteboard, max_hops=max_hops)
    router = TracingRouter(
        tracer=tracer,
        whiteboard=whiteboard,
        transport=transport,
        circuit_breaker=breaker,
        max_hops=max_hops,
    )

    # 3. Define mutually delegating ping-pong agent handlers
    def make_handler(agent_name: str, target_name: str) -> Any:
        async def handler(req: JSONRPCRequest) -> JSONRPCResponse:
            msg: UACPMessage = req.params
            inst = str(msg.payload.get("instruction", ""))
            match = re.search(r"step (\d+)", inst)
            step = int(match.group(1)) if match else int(msg.payload.get("step", 1))

            tracer.on_info(
                f"Agent '{agent_name}' received delegation step {step} from {msg.sender.agent_id}"
            )

            # Prepare symmetric return delegation with incremented step
            next_step = step + 1
            payload_dict = TaskDelegationPayload(
                instruction=f"Ping-pong delegation step {next_step}",
                whiteboard_session_ref=session_id,
            ).model_dump()
            payload_dict["step"] = next_step

            next_msg = UACPMessage(
                correlation_id=msg.correlation_id,
                trace_id=msg.trace_id,
                sender=Party(agent_id=agent_name),
                recipient=Party(agent_id=target_name),
                intent=Intent.task_delegation,
                payload=payload_dict,
                circuit_breaker=CircuitBreakerState(
                    max_hops=max_hops,
                    current_hop=msg.circuit_breaker.current_hop + 1,
                ),
            )

            # Send next delegation via router
            res = await router.send(
                next_msg,
                recipient_agent_id=target_name,
                session_id=session_id,
            )
            return res

        return handler

    transport.register_agent("agent_ping", make_handler("agent_ping", "agent_pong"))
    transport.register_agent("agent_pong", make_handler("agent_pong", "agent_ping"))

    tracer.on_info("Agents 'agent_ping' and 'agent_pong' registered with router.")

    # 4. Initiate the runaway loop from orchestrator
    init_payload = TaskDelegationPayload(
        instruction="Ping-pong delegation step 1",
        whiteboard_session_ref=session_id,
    ).model_dump()
    init_payload["step"] = 1

    init_msg = UACPMessage(
        correlation_id=f"loop-corr-{uuid.uuid4().hex[:8]}",
        trace_id=f"loop-trace-{uuid.uuid4().hex[:8]}",
        sender=Party(agent_id="orchestrator"),
        recipient=Party(agent_id="agent_ping"),
        intent=Intent.task_delegation,
        payload=init_payload,
        circuit_breaker=CircuitBreakerState(max_hops=max_hops, current_hop=1),
    )

    tracer.on_info("Orchestrator dispatching initial delegation to agent_ping...")
    await router.send(init_msg, recipient_agent_id="agent_ping", session_id=session_id)

    # 5. Verify the circuit breaker halted the session
    is_halted = breaker.is_session_halted(session_id)
    halt_err = breaker.get_halt_error(session_id)
    final_hop = whiteboard.get_hop_count(session_id)

    tracer.print_section("Demo Results")
    if is_halted and final_hop >= max_hops:
        c.print(
            f"[bold green][PASS][/bold green] Circuit breaker tripped at authoritative hop "
            f"[bold]{final_hop}[/bold] (max={max_hops})."
        )
        if halt_err:
            c.print(f"[bold cyan]Halt Reason:[/bold cyan] {halt_err.message}")
            c.print(f"[bold cyan]Error Code:[/bold cyan] {halt_err.code}")
        c.print("[dim]The runaway multi-agent delegation was halted automatically.[/dim]")
        return 0

    c.print(
        f"[bold red][FAIL][/bold red] Expected session to halt at hop {max_hops}, but "
        f"is_halted={is_halted}, final_hop={final_hop}."
    )
    return 1
