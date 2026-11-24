"""Live readable trace rendering with Rich for UACP multi-agent execution."""

from __future__ import annotations

from collections import defaultdict
from typing import Any

from rich.console import Console
from rich.panel import Panel
from rich.text import Text

from uacp.memory.whiteboard.base import (
    DriftReport,
    ReviewRecord,
    ReviewVerdict,
    VerificationReport,
    WhiteboardBackend,
    WhiteboardEvent,
)
from uacp.protocol.jsonrpc import JSONRPCResponse
from uacp.protocol.messages import CircuitBreakerState, UACPMessage
from uacp.protocol.router import Router


class RunnerTracer:
    """Live Rich console output for messages, Whiteboard writes, and circuit breaker events."""

    def __init__(self, console: Console | None = None) -> None:
        self.console = console or Console()

    def print_section(self, title: str) -> None:
        """Print a visually distinct rule section header."""
        self.console.rule(f"[bold cyan]{title}[/bold cyan]")

    def on_hop(
        self,
        hop: int,
        authoritative_hop: int,
        sender: str,
        recipient: str,
        intent: str,
        summary: str = "",
    ) -> None:
        """Log a message delivery hop."""
        text = Text()
        text.append(f"HOP {hop:02d} ", style="bold cyan")
        text.append(f"[auth:{authoritative_hop:02d}] ", style="dim")
        text.append(f"{sender} ", style="bold green")
        text.append(f"--> [{intent}] --> ", style="bold yellow")
        text.append(f"{recipient}", style="bold blue")
        if summary:
            text.append(f"\n    |-- {summary}", style="dim")
        self.console.print(text)

    def on_whiteboard_write(self, key: str, version: int, author: str, summary: str = "") -> None:
        """Log a write to the Whiteboard state."""
        text = Text()
        text.append("  WB WRITE ", style="bold magenta")
        text.append(f"key='{key}' ", style="bold white")
        text.append(f"v{version} ", style="cyan")
        text.append(f"by {author}", style="yellow")
        if summary:
            text.append(f" ({summary})", style="dim")
        self.console.print(text)

    def on_llm_call(
        self,
        agent_id: str,
        intent: str,
        input_tokens: int,
        output_tokens: int,
        repairs: int = 0,
    ) -> None:
        """Log an LLM invocation and token usage."""
        text = Text()
        text.append("  LLM CALL ", style="bold yellow")
        text.append(f"[{agent_id}] ", style="green")
        text.append(f"intent={intent} ", style="dim")
        text.append(f"tokens: in={input_tokens}, out={output_tokens} ", style="cyan")
        if repairs > 0:
            text.append(f"repairs={repairs}", style="bold red")
        else:
            text.append("repairs=0", style="dim")
        self.console.print(text)

    def on_session_halted(self, session_id: str, reason: str, code: int) -> None:
        """Log a circuit breaker halt event."""
        panel = Panel(
            f"[bold red]CIRCUIT BREAKER TRIPPED[/bold red]\n"
            f"[yellow]Session:[/yellow] {session_id}\n"
            f"[yellow]Code:[/yellow] {code}\n"
            f"[yellow]Reason:[/yellow] {reason}",
            border_style="red",
            title="[bold red]HALT[/bold red]",
        )
        self.console.print(panel)

    def on_info(self, message: str) -> None:
        """Log general runner info."""
        self.console.print(f"[dim][INFO] {message}[/dim]")


class TracingWhiteboardBackend(WhiteboardBackend):
    """Proxy around WhiteboardBackend that notifies RunnerTracer on writes, events, and hops."""

    def __init__(self, inner: WhiteboardBackend, tracer: RunnerTracer) -> None:
        super().__init__(clock=inner.clock)
        self.inner = inner
        self.tracer = tracer
        self._hop_counts: dict[str, int] = defaultdict(int)

    def get_hop_count(self, session_id: str) -> int:
        """Return the authoritative hop count tracked for a session."""
        return self._hop_counts.get(session_id, 0)

    async def create_session(self, ttl_s: float, max_hops: int = 5) -> str:
        sid = await self.inner.create_session(ttl_s=ttl_s, max_hops=max_hops)
        self._hop_counts[sid] = 0
        return sid

    async def close(self, session_id: str) -> None:
        await self.inner.close(session_id)

    async def incr_hop(self, session_id: str) -> int:
        hop = await self.inner.incr_hop(session_id)
        self._hop_counts[session_id] = hop
        return hop

    async def get_state(self, session_id: str) -> dict[str, dict[str, Any]]:
        return await self.inner.get_state(session_id)

    async def put(
        self,
        session_id: str,
        key: str,
        value: Any,
        author: str,
        expected_version: int | None = None,
    ) -> int:
        v = await self.inner.put(
            session_id=session_id,
            key=key,
            value=value,
            author=author,
            expected_version=expected_version,
        )
        summary = ""
        if isinstance(value, dict):
            if "status" in value:
                summary = f"status={value['status']}"
            elif "anomalies_found" in value:
                summary = (
                    f"mitigated={value.get('anomalies_mitigated', 0)}/"
                    f"{value.get('anomalies_found', 0)}"
                )
        self.tracer.on_whiteboard_write(key, v, author, summary)
        return v

    async def append_event(
        self,
        session_id: str,
        event: dict[str, Any] | WhiteboardEvent,
    ) -> WhiteboardEvent:
        ev = await self.inner.append_event(session_id, event)
        ev_type = str(ev.event_type)
        data = ev.data
        if ev_type == "agent.llm_call":
            tokens = data.get("tokens", {})
            self.tracer.on_llm_call(
                agent_id=str(data.get("agent_id", "unknown")),
                intent=str(data.get("intent", "llm_call")),
                input_tokens=int(tokens.get("input_tokens", 0)),
                output_tokens=int(tokens.get("output_tokens", 0)),
                repairs=int(data.get("repairs", 0)),
            )
        elif ev_type == "session.halted":
            self.tracer.on_session_halted(
                session_id=session_id,
                reason=str(data.get("reason", "Unknown halt reason")),
                code=int(data.get("code", -32001)),
            )
        return ev

    async def events(
        self,
        session_id: str,
        since: str | int | None = None,
    ) -> list[WhiteboardEvent]:
        return await self.inner.events(session_id, since=since)

    async def verify_chain(self, session_id: str) -> VerificationReport:
        return await self.inner.verify_chain(session_id)

    async def add_review(
        self,
        session_id: str,
        key: str,
        reviewer: str,
        verdict: ReviewVerdict | str,
        notes: str = "",
    ) -> ReviewRecord:
        return await self.inner.add_review(session_id, key, reviewer, verdict, notes)

    async def drift_report(self, session_id: str) -> DriftReport:
        return await self.inner.drift_report(session_id)


class TracingRouter(Router):
    """Router subclass that synchronizes hops and reports delivery events to RunnerTracer."""

    def __init__(
        self,
        tracer: RunnerTracer,
        whiteboard: WhiteboardBackend,
        transport: Any,
        registry: Any = None,
        circuit_breaker: Any = None,
        max_hops: int | None = None,
    ) -> None:
        super().__init__(
            whiteboard=whiteboard,
            transport=transport,
            registry=registry,
            circuit_breaker=circuit_breaker,
            max_hops=max_hops,
        )
        self.tracer = tracer

    async def send(
        self,
        message: Any,
        *,
        recipient_agent_id: str | None = None,
        capability: str | None = None,
        session_id: str | None = None,
    ) -> JSONRPCResponse:
        # Determine target session id
        sid = session_id
        if hasattr(message, "payload") and isinstance(message.payload, dict):
            sid = sid or message.payload.get("whiteboard_session_ref")
        if hasattr(message, "correlation_id"):
            sid = sid or message.correlation_id

        # Authoritative hop count before send
        auth_hop_before = 0
        if isinstance(self.whiteboard, TracingWhiteboardBackend) and sid:
            auth_hop_before = self.whiteboard.get_hop_count(sid)

        # Synchronize advisory hop count if message was created with hop 0
        if (
            isinstance(message, UACPMessage)
            and message.circuit_breaker.current_hop == 0
            and sid is not None
        ):
            message = message.model_copy(
                update={
                    "circuit_breaker": CircuitBreakerState(
                        max_hops=message.circuit_breaker.max_hops,
                        current_hop=auth_hop_before + 1,
                    )
                }
            )

        res = await super().send(
            message,
            recipient_agent_id=recipient_agent_id,
            capability=capability,
            session_id=session_id,
        )

        auth_hop_after = auth_hop_before
        if isinstance(self.whiteboard, TracingWhiteboardBackend) and sid:
            auth_hop_after = self.whiteboard.get_hop_count(sid)

        sender_id = (
            getattr(message.sender, "agent_id", str(message.sender))
            if hasattr(message, "sender")
            else "unknown"
        )
        recipient_id = recipient_agent_id or (
            getattr(message.recipient, "agent_id", str(message.recipient))
            if hasattr(message, "recipient")
            else "unknown"
        )
        intent = getattr(message, "intent", "unknown")
        if hasattr(intent, "value"):
            intent = intent.value
        msg_hop = getattr(getattr(message, "circuit_breaker", None), "current_hop", auth_hop_after)

        summary = ""
        if res.error:
            summary = f"[red]ERROR: {res.error.message}[/red]"
        elif isinstance(res.result, dict):
            if "status" in res.result:
                summary = f"status={res.result['status']}"
            elif "summary" in res.result:
                summary = str(res.result["summary"])

        self.tracer.on_hop(
            hop=msg_hop,
            authoritative_hop=auth_hop_after,
            sender=str(sender_id),
            recipient=str(recipient_id),
            intent=str(intent),
            summary=summary,
        )
        return res
