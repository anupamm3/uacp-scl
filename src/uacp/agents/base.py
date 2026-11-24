"""Agent base class, lifecycle management, SCL LLM execution, and event tracking."""

from __future__ import annotations

import contextlib
from pathlib import Path
from typing import Any, ClassVar

from uacp.adapters.base import Adapter, CompiledRequest, LLMResult, Message, Usage
from uacp.adapters.enforcer import invoke_with_repair
from uacp.memory.whiteboard.base import WhiteboardBackend
from uacp.memory.whiteboard.client import WhiteboardClient
from uacp.protocol.capabilities import AgentCard, Capability, CapabilityRegistry, build_agent_card
from uacp.protocol.jsonrpc import JSONRPCRequest, JSONRPCResponse
from uacp.protocol.messages import (
    HandoffPayload,
    Intent,
    Party,
    TaskDelegationPayload,
    TaskResultPayload,
    UACPMessage,
)
from uacp.protocol.negotiation import (
    AgentNegotiationContext,
    DefaultNegotiationPolicy,
    NegotiationAction,
    NegotiationDecision,
    NegotiationOutcome,
    NegotiationPolicy,
    NegotiationProposal,
    execute_negotiation,
)
from uacp.protocol.router import Router
from uacp.scl.ast import Output as SCLOutput
from uacp.scl.context import RuntimeContext, bind
from uacp.scl.parser import parse_file
from uacp.scl.validate import validate_strict


class TrackingAdapterWrapper:
    """Tracking wrapper around an Adapter that accumulates token usage and repair invocations."""

    def __init__(self, inner: Adapter) -> None:
        self.inner = inner
        self.invocations = 0
        self.total_input_tokens = 0
        self.total_output_tokens = 0

    @property
    def capabilities(self) -> Any:
        return self.inner.capabilities

    @property
    def name(self) -> str:
        return self.inner.name

    def compile(self, bound_doc: Any, user_message: Any) -> CompiledRequest:
        return self.inner.compile(bound_doc, user_message)

    async def invoke(self, request: Any) -> LLMResult:
        self.invocations += 1
        if hasattr(request, "messages") and isinstance(request.messages, list):
            request.messages = [
                Message(role=m["role"], content=m["content"])
                if isinstance(m, dict) and "role" in m and "content" in m
                else m
                for m in request.messages
            ]
        result = await self.inner.invoke(request)
        if hasattr(result, "usage") and result.usage:
            self.total_input_tokens += getattr(result.usage, "input_tokens", 0)
            self.total_output_tokens += getattr(result.usage, "output_tokens", 0)
        return result


class ReadOnlyWhiteboardClient:
    """Read-only proxy wrapping WhiteboardClient to prevent observer writes."""

    def __init__(self, inner: WhiteboardClient) -> None:
        self._inner = inner

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)

    async def put(self, *args: Any, **kwargs: Any) -> Any:
        raise PermissionError("Observer agent cannot write to Whiteboard after handoff.")

    async def write_model(self, *args: Any, **kwargs: Any) -> Any:
        raise PermissionError("Observer agent cannot write to Whiteboard after handoff.")

    async def incr_hop(self, *args: Any, **kwargs: Any) -> Any:
        raise PermissionError("Observer agent cannot write to Whiteboard after handoff.")

    async def add_review(self, *args: Any, **kwargs: Any) -> Any:
        raise PermissionError("Observer agent cannot write to Whiteboard after handoff.")


class Agent:
    """Base class for all UACP autonomous agents.

    An agent's role is specified by its SCL contract, its abilities are defined
    via an AgentCard, and its communications and events are integrated through
    the Router and Whiteboard.
    """

    abilities: ClassVar[list[Capability]] = []

    def __init__(
        self,
        agent_id: str,
        scl_path: str | Path | None = None,
        adapter: Adapter | None = None,
        whiteboard: WhiteboardClient | None = None,
        router: Router | None = None,
        card: AgentCard | None = None,
        capabilities: list[Capability] | None = None,
        current_load: int = 0,
        negotiation_policy: NegotiationPolicy | None = None,
    ) -> None:
        """Initialize the Agent.

        Args:
            agent_id: Unique identifier for this agent.
            scl_path: Optional path to an SCL specification file defining identity and contracts.
            adapter: Optional LLM adapter for prompt compilation and execution.
            whiteboard: Optional WhiteboardClient for shared state access and trace logging.
            router: Optional Router for sending and receiving protocol messages.
            card: Optional pre-constructed AgentCard manifest.
            capabilities: Optional list of capabilities. Defaults to class-level abilities.
            current_load: Initial task load on the agent (default 0).
            negotiation_policy: Optional custom policy for task difficulty negotiation.
        """
        self.agent_id = agent_id
        self.scl_path = Path(scl_path) if scl_path is not None else None
        self.adapter = adapter
        self.whiteboard = whiteboard
        self.router = router
        self.registry: CapabilityRegistry | None = None
        self.current_load = current_load
        self.negotiation_policy: NegotiationPolicy = (
            negotiation_policy if negotiation_policy is not None else DefaultNegotiationPolicy()
        )
        self.is_observer: bool = False

        # Capabilities defined in one centralized place
        self.capabilities: list[Capability] = (
            list(capabilities) if capabilities is not None else list(self.abilities)
        )
        self.card: AgentCard | None = card

    def _build_agent_card(self) -> AgentCard:
        """Build an AgentCard from the agent's SCL specification and declared abilities."""
        role = ""
        discovered = list(self.capabilities)
        version = "1.0"

        if self.scl_path is not None and self.scl_path.is_file():
            doc = parse_file(str(self.scl_path))
            if doc.identity:
                role = doc.identity.body.strip()
            if doc.version:
                version = doc.version

            if not discovered:
                return build_agent_card(
                    agent_cls=self.__class__,
                    scl_path=self.scl_path,
                    agent_id=self.agent_id,
                    engine=self.adapter.name if self.adapter else "default",
                    role=role or None,
                    current_load=self.current_load,
                )

        engine_name = self.adapter.name if self.adapter else "default"
        return AgentCard(
            agent_id=self.agent_id,
            engine=engine_name,
            role=role,
            current_load=self.current_load,
            capabilities=discovered,
            version=version,
        )

    def register(self, registry: CapabilityRegistry, router: Router) -> None:
        """Register the agent with a CapabilityRegistry and router transport.

        Args:
            registry: CapabilityRegistry instance to publish the agent card to.
            router: Router whose transport will deliver messages to this agent.
        """
        self.registry = registry
        self.router = router
        if self.card is None:
            self.card = self._build_agent_card()
        self.registry.register(self.card)
        self.router.transport.register_agent(self.agent_id, self._on_transport_request)

    def update_load(self, n: int) -> None:
        """Update current task load at runtime and update the registry."""
        self.current_load = n
        if self.card is not None:
            self.card.current_load = n
        if self.registry is not None:
            self.registry.update_load(self.agent_id, n)

    def heartbeat(self) -> bool:
        """Renew the agent's heartbeat lease in the capability registry."""
        if self.registry is not None:
            return self.registry.heartbeat(self.agent_id)
        return False

    async def _record_whiteboard_event(
        self,
        session_id: str | None,
        event_type: str,
        data: dict[str, Any],
    ) -> None:
        """Record an event to the Whiteboard backend if available."""
        backend: WhiteboardBackend | None = None
        if self.whiteboard is not None:
            backend = self.whiteboard.backend
        elif self.router is not None:
            backend = self.router.whiteboard

        if backend is None:
            return

        sid = session_id or (self.whiteboard.session_id if self.whiteboard else None)
        if not sid:
            return

        with contextlib.suppress(Exception):
            await backend.append_event(sid, {"event_type": event_type, "data": data})

    async def run_llm(
        self,
        user_message: str,
        runtime_data: dict[str, Any] | None = None,
        session_id: str | None = None,
        intent: str = "run_llm",
    ) -> tuple[dict[str, Any] | list[Any] | str, Usage]:
        """Parse SCL contract, bind data, invoke LLM with auto-repair, and log usage.

        Args:
            user_message: Plain string user prompt/instruction.
            runtime_data: Optional dictionary payload for SCL variable binding.
            session_id: Optional Whiteboard session reference for audit logging.
            intent: Intent name associated with this LLM call.

        Returns:
            Tuple of (parsed_output, Usage).

        Raises:
            ValueError: If agent has no SCL path or adapter configured.
        """
        if self.is_observer:
            raise PermissionError("Observer agent cannot write to Whiteboard after handoff.")

        if self.scl_path is None:
            raise ValueError(f"Agent '{self.agent_id}' has no scl_path configured.")
        if self.adapter is None:
            raise ValueError(f"Agent '{self.agent_id}' has no adapter configured.")

        # 1. Parse and validate SCL AST
        doc = parse_file(str(self.scl_path))
        validate_strict(doc)

        # 2. Bind runtime variables (supporting dotted paths like data.* and runtime.payload.*)
        data_dict = dict(runtime_data or {})
        if "data" not in data_dict:
            data_dict["data"] = dict(data_dict)
        if "runtime" not in data_dict:
            data_dict["runtime"] = {"payload": dict(data_dict)}
        ctx = RuntimeContext(data_dict)
        bound_doc = bind(doc, ctx)

        # 3. Compile request
        compiled_req = self.adapter.compile(bound_doc, user_message)

        # 4. Wrap with tracking adapter
        tracker = TrackingAdapterWrapper(self.adapter)
        output_node = (
            doc.output
            if doc.output is not None
            else SCLOutput(format="markdown", strict=False, body="")
        )

        # 5. Invoke with repair
        parsed_result = await invoke_with_repair(tracker, compiled_req, output_node)
        repairs = max(0, tracker.invocations - 1)
        usage = Usage(
            input_tokens=tracker.total_input_tokens,
            output_tokens=tracker.total_output_tokens,
            requests=tracker.invocations,
        )

        # 6. Record LLM call event on Whiteboard
        await self._record_whiteboard_event(
            session_id=session_id,
            event_type="agent.llm_call",
            data={
                "agent_id": self.agent_id,
                "intent": intent,
                "tokens": {
                    "input_tokens": usage.input_tokens,
                    "output_tokens": usage.output_tokens,
                    "total_tokens": usage.input_tokens + usage.output_tokens,
                },
                "repairs": repairs,
            },
        )

        return parsed_result, usage

    async def handle(self, message: UACPMessage) -> UACPMessage:
        """Handle incoming UACP message, dispatching on intent.

        Args:
            message: Incoming UACPMessage.

        Returns:
            A response UACPMessage.
        """
        session_id = (
            message.payload.get("whiteboard_session_ref")
            or message.correlation_id
            or (self.whiteboard.session_id if self.whiteboard else None)
        )

        # Log handled message to whiteboard
        await self._record_whiteboard_event(
            session_id=session_id,
            event_type="agent.message_handled",
            data={
                "agent_id": self.agent_id,
                "intent": message.intent.value,
                "message_id": message.message_id,
                "sender": message.sender.agent_id,
                "recipient": message.recipient.agent_id,
                "correlation_id": message.correlation_id,
            },
        )

        match message.intent:
            case Intent.task_delegation:
                return await self.handle_task_delegation(message)
            case Intent.heartbeat:
                return await self.handle_heartbeat(message)
            case Intent.capability_query:
                return await self.handle_capability_query(message)
            case Intent.task_result:
                return await self.handle_task_result(message)
            case Intent.negotiate:
                return await self.handle_negotiate(message)
            case Intent.handoff:
                return await self.handle_handoff(message)
            case _:
                return await self.handle_default(message)

    async def handle_task_delegation(self, message: UACPMessage) -> UACPMessage:
        """Handle task delegation by extracting plain string task and optional payload."""
        delegation = TaskDelegationPayload.model_validate(message.payload)
        instruction = delegation.instruction
        session_ref = delegation.whiteboard_session_ref

        # Extract optional data payload
        runtime_data: dict[str, Any] = message.payload.get("data") or message.payload.get(
            "runtime_data", {}
        )

        result_data: Any = None
        usage = Usage()

        if self.scl_path is not None and self.adapter is not None:
            result_data, usage = await self.run_llm(
                user_message=instruction,
                runtime_data=runtime_data,
                session_id=session_ref,
                intent=Intent.task_delegation.value,
            )
        else:
            result_data = {"status": "completed", "instruction": instruction}

        # Store result on whiteboard if client is configured
        result_ref: str | None = None
        if self.whiteboard is not None:
            result_ref = f"result:{message.message_id}"
            with contextlib.suppress(Exception):
                await self.whiteboard.put(result_ref, result_data)

        task_result = TaskResultPayload(
            status="success",
            summary=f"Task completed by {self.agent_id}",
            result_ref=result_ref,
        )

        payload_dict = task_result.model_dump()
        payload_dict["result"] = result_data
        payload_dict["usage"] = usage.model_dump()

        engine = self.card.engine if self.card else "default"
        return UACPMessage(
            correlation_id=message.correlation_id,
            trace_id=message.trace_id,
            sender=Party(agent_id=self.agent_id, engine=engine),
            recipient=message.sender,
            intent=Intent.task_result,
            payload=payload_dict,
        )

    async def handle_heartbeat(self, message: UACPMessage) -> UACPMessage:
        """Handle incoming heartbeat ping."""
        engine = self.card.engine if self.card else "default"
        return UACPMessage(
            correlation_id=message.correlation_id,
            trace_id=message.trace_id,
            sender=Party(agent_id=self.agent_id, engine=engine),
            recipient=message.sender,
            intent=Intent.heartbeat,
            payload={"status": "ok"},
        )

    async def handle_capability_query(self, message: UACPMessage) -> UACPMessage:
        """Handle capability query advertising supported capabilities."""
        engine = self.card.engine if self.card else "default"
        caps = [c.name for c in self.card.capabilities] if self.card else []
        return UACPMessage(
            correlation_id=message.correlation_id,
            trace_id=message.trace_id,
            sender=Party(agent_id=self.agent_id, engine=engine),
            recipient=message.sender,
            intent=Intent.capability_response,
            payload={"capabilities": caps},
        )

    async def handle_task_result(self, message: UACPMessage) -> UACPMessage:
        """Handle task result received from downstream delegation."""
        engine = self.card.engine if self.card else "default"
        return UACPMessage(
            correlation_id=message.correlation_id,
            trace_id=message.trace_id,
            sender=Party(agent_id=self.agent_id, engine=engine),
            recipient=message.sender,
            intent=Intent.task_result,
            payload={"status": "acknowledged"},
        )

    def get_negotiation_context(self) -> AgentNegotiationContext:
        """Return negotiation context containing complexity level, strengths, and current load."""
        max_comp = 1
        all_strengths: list[str] = []
        for cap in self.capabilities:
            if cap.complexity_level > max_comp:
                max_comp = cap.complexity_level
            for s in cap.strengths:
                if s not in all_strengths:
                    all_strengths.append(s)

        return AgentNegotiationContext(
            agent_id=self.agent_id,
            complexity_level=max_comp,
            strengths=all_strengths,
            current_load=self.current_load,
        )

    def set_observer(self, is_observer: bool = True) -> None:
        """Set agent to observer mode (read-only whiteboard access)."""
        self.is_observer = is_observer
        if (
            is_observer
            and self.whiteboard is not None
            and not isinstance(self.whiteboard, ReadOnlyWhiteboardClient)
        ):
            self.whiteboard = ReadOnlyWhiteboardClient(self.whiteboard)  # type: ignore[assignment]

    async def handle_negotiate(self, message: UACPMessage) -> UACPMessage:
        """Handle incoming bilateral task negotiation proposal or counter-offer."""
        raw_proposal = message.payload.get("proposal", {})
        round_num = int(message.payload.get("round", 1))

        if isinstance(raw_proposal, dict):
            # Check if this is already a decision or a proposal
            if "subtask_text" in raw_proposal:
                proposal = NegotiationProposal.model_validate(raw_proposal)
            elif "action" in raw_proposal:
                # Responder got counter decision from initiator
                return await self.handle_default(message)
            else:
                proposal = NegotiationProposal.model_validate(raw_proposal)
        elif isinstance(raw_proposal, NegotiationProposal):
            proposal = raw_proposal
        else:
            decision = NegotiationDecision(
                action=NegotiationAction.reject,
                reason_code="INVALID_PROPOSAL",
                reason="Invalid proposal payload format",
                round=round_num,
            )
            return self._build_negotiation_reply(message, decision, round_num)

        context = self.get_negotiation_context()
        decision = self.negotiation_policy.evaluate(proposal, context, round_num=round_num)
        return self._build_negotiation_reply(message, decision, round_num)

    def _build_negotiation_reply(
        self,
        message: UACPMessage,
        decision: NegotiationDecision,
        round_num: int,
    ) -> UACPMessage:
        """Build UACPMessage response carrying negotiation decision."""
        engine = self.card.engine if self.card else "default"
        payload_data = {
            "proposal": decision.model_dump(),
            "decision": decision.model_dump(),
            "round": round_num,
        }
        return UACPMessage(
            correlation_id=message.correlation_id,
            trace_id=message.trace_id,
            sender=Party(agent_id=self.agent_id, engine=engine),
            recipient=message.sender,
            intent=Intent.negotiate,
            payload=payload_data,
        )

    def evaluate_counter(
        self,
        counter_proposal: NegotiationProposal,
        round_num: int = 2,
    ) -> NegotiationDecision:
        """Evaluate a received counter-offer from a peer agent."""
        context = self.get_negotiation_context()
        return self.negotiation_policy.evaluate(counter_proposal, context, round_num=round_num)

    async def handle_handoff(self, message: UACPMessage) -> UACPMessage:
        """Handle incoming session handoff, verifying state integrity first."""
        payload = HandoffPayload.model_validate(message.payload)

        # Verify hash chain integrity if Whiteboard is available
        is_valid = True
        verification_details = "Verified"
        if self.whiteboard is not None:
            try:
                report = await self.whiteboard.verify_chain()
                is_valid = report.is_valid
                verification_details = report.details
            except Exception as e:
                is_valid = False
                verification_details = f"Verification error: {e}"

        engine = self.card.engine if self.card else "default"

        if not is_valid:
            return UACPMessage(
                correlation_id=message.correlation_id,
                trace_id=message.trace_id,
                sender=Party(agent_id=self.agent_id, engine=engine),
                recipient=message.sender,
                intent=Intent.task_result,
                payload={
                    "status": "rejected",
                    "summary": f"Handoff rejected: TamperDetected ({verification_details})",
                    "result_ref": payload.context_ref,
                    "is_valid": False,
                },
            )

        return UACPMessage(
            correlation_id=message.correlation_id,
            trace_id=message.trace_id,
            sender=Party(agent_id=self.agent_id, engine=engine),
            recipient=message.sender,
            intent=Intent.task_result,
            payload={
                "status": "success",
                "summary": f"Handoff accepted: {payload.reason}",
                "result_ref": payload.context_ref,
                "is_valid": True,
            },
        )

    async def handoff(
        self,
        target_agent_id: str,
        reason: str,
        session_id: str | None = None,
        context_ref: str | None = None,
    ) -> JSONRPCResponse:
        """Initiate session handoff to target agent and transition caller to observer mode."""
        if self.router is None:
            raise ValueError(f"Agent '{self.agent_id}' has no router configured for handoff.")

        sid = session_id or (self.whiteboard.session_id if self.whiteboard else "default")
        payload = HandoffPayload(
            target_agent=target_agent_id,
            reason=reason,
            context_ref=context_ref,
        )
        engine = self.card.engine if self.card else "default"
        msg = UACPMessage(
            correlation_id=sid,
            trace_id=f"handoff-{self.agent_id}-{target_agent_id}",
            sender=Party(agent_id=self.agent_id, engine=engine),
            recipient=Party(agent_id=target_agent_id),
            intent=Intent.handoff,
            payload=payload.model_dump(),
        )

        response = await self.router.send(msg, session_id=sid)
        # On initiating handoff, caller transitions to observer (read-only)
        self.set_observer(True)
        return response

    async def negotiate(
        self,
        recipient_agent_id: str,
        proposal: NegotiationProposal,
        session_id: str | None = None,
        max_rounds: int = 2,
    ) -> NegotiationOutcome:
        """Initiate bilateral negotiation with recipient agent through router."""
        if self.router is None:
            raise ValueError(f"Agent '{self.agent_id}' has no router configured for negotiation.")

        sid = session_id or (self.whiteboard.session_id if self.whiteboard else "default")
        return await execute_negotiation(
            router=self.router,
            session_id=sid,
            initiator_agent_id=self.agent_id,
            responder_agent_id=recipient_agent_id,
            proposal=proposal,
            max_rounds=max_rounds,
        )

    async def handle_default(self, message: UACPMessage) -> UACPMessage:
        """Default handler returning acknowledgment for unspecialized intents."""
        engine = self.card.engine if self.card else "default"
        return UACPMessage(
            correlation_id=message.correlation_id,
            trace_id=message.trace_id,
            sender=Party(agent_id=self.agent_id, engine=engine),
            recipient=message.sender,
            intent=Intent.task_result,
            payload={"status": "acknowledged", "received_intent": message.intent.value},
        )

    async def _on_transport_request(self, req: JSONRPCRequest) -> UACPMessage:
        """Handler registered with transport, processing JSONRPCRequest and returning response."""
        msg: UACPMessage = req.params
        return await self.handle(msg)
