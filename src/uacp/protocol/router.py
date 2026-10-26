"""Message routing between agents via transport and capability registry."""

from __future__ import annotations

import contextlib
from typing import Any

from uacp.memory.whiteboard.base import WhiteboardBackend
from uacp.protocol.capabilities import CapabilityRegistry
from uacp.protocol.circuit_breaker import CircuitBreaker
from uacp.protocol.jsonrpc import (
    JSONRPCError,
    JSONRPCErrorCode,
    JSONRPCRequest,
    JSONRPCResponse,
    build_error_response,
    parse_request,
)
from uacp.protocol.messages import Party, UACPMessage, validate_payload
from uacp.protocol.transport.inproc import InProcessTransport


class Router:
    """Routes messages between agents, enforcing safety limits, anti-spoofing, and capabilities."""

    def __init__(
        self,
        whiteboard: WhiteboardBackend,
        transport: InProcessTransport,
        registry: CapabilityRegistry | None = None,
        circuit_breaker: CircuitBreaker | None = None,
        max_hops: int | None = None,
    ) -> None:
        """Initialize the Router.

        Args:
            whiteboard: Ephemeral Whiteboard backend for session hops and event logging.
            transport: In-process agent transport for dispatching messages.
            registry: Optional CapabilityRegistry for discovering agents by capability.
            circuit_breaker: Optional CircuitBreaker instance. If None, one is created.
            max_hops: Optional default hop limit used if circuit_breaker is instantiated here.
        """
        self.whiteboard = whiteboard
        self.transport = transport
        self.registry = registry
        self.breaker = circuit_breaker or CircuitBreaker(whiteboard=whiteboard, max_hops=max_hops)
        self.recorded_responses: list[tuple[UACPMessage, JSONRPCResponse]] = []

    async def halt(self, session_id: str, reason: str) -> JSONRPCError:
        """Manually halt a session and refuse further deliveries."""
        return await self.breaker.halt(session_id, reason)

    async def _record_response(
        self,
        session_id: str,
        message: UACPMessage,
        response: JSONRPCResponse,
    ) -> None:
        """Record the delivered response in-memory and on the Whiteboard event log."""
        self.recorded_responses.append((message, response))
        with contextlib.suppress(Exception):
            await self.whiteboard.append_event(
                session_id,
                {
                    "event_type": "message.delivered",
                    "data": {
                        "message_id": message.message_id,
                        "intent": message.intent.value,
                        "sender": message.sender.agent_id,
                        "recipient": message.recipient.agent_id,
                        "hop": message.circuit_breaker.current_hop,
                        "response": response.model_dump(),
                    },
                },
            )

    async def send(
        self,
        message: UACPMessage | JSONRPCRequest | dict[str, Any],
        *,
        recipient_agent_id: str | None = None,
        capability: str | None = None,
        session_id: str | None = None,
    ) -> JSONRPCResponse:
        """Send a message to a recipient agent, enforcing safety invariants.

        Args:
            message: UACPMessage, JSONRPCRequest, or dictionary representation.
            recipient_agent_id: Optional explicit recipient agent ID override.
            capability: Optional capability name to resolve recipient dynamically.
            session_id: Optional Whiteboard session identifier override.

        Returns:
            JSONRPCResponse containing result or error descriptor.
        """
        # 1. Parse / normalize incoming request and message model
        rpc_id: str | int | None = None
        uacp_msg: UACPMessage

        if isinstance(message, JSONRPCRequest):
            rpc_id = message.id
            uacp_msg = message.params
        elif isinstance(message, UACPMessage):
            rpc_id = message.message_id
            uacp_msg = message
        elif isinstance(message, dict):
            if "method" in message:
                parsed = parse_request(message)
                if isinstance(parsed, JSONRPCResponse):
                    return parsed
                rpc_id = parsed.id
                uacp_msg = parsed.params
            else:
                try:
                    uacp_msg = UACPMessage.model_validate(message)
                    rpc_id = uacp_msg.message_id
                except Exception as e:
                    return build_error_response(
                        id=None,
                        code=JSONRPCErrorCode.INVALID_PARAMS,
                        message=f"Invalid message payload: {e}",
                        data=str(e),
                    )
        else:
            return build_error_response(
                id=None,
                code=JSONRPCErrorCode.INVALID_REQUEST,
                message="Expected UACPMessage, JSONRPCRequest, or dict",
            )

        # 2. Validate payload schema against intent contract
        try:
            validate_payload(uacp_msg)
        except Exception as val_err:
            return build_error_response(
                id=rpc_id,
                code=JSONRPCErrorCode.SCHEMA_VIOLATION,
                message=f"SchemaViolation: message payload violates contract schema ({val_err})",
                data=str(val_err),
            )

        # 3. Determine authoritative session_id
        resolved_session_id = (
            session_id or uacp_msg.payload.get("whiteboard_session_ref") or uacp_msg.correlation_id
        )

        # 4. Resolve recipient: if both agent_id and capability are given, agent_id wins
        target_agent_id = recipient_agent_id or uacp_msg.recipient.agent_id or None
        target_capability = capability or uacp_msg.recipient.capability or None

        if target_agent_id:
            # Agent ID takes precedence over capability
            if target_agent_id != uacp_msg.recipient.agent_id:
                uacp_msg = uacp_msg.model_copy(
                    update={
                        "recipient": Party(
                            agent_id=target_agent_id,
                            engine=uacp_msg.recipient.engine or "default",
                            capability=target_capability,
                        )
                    }
                )
        elif target_capability:
            # Resolve via capability registry
            if self.registry is None:
                return build_error_response(
                    id=rpc_id,
                    code=JSONRPCErrorCode.CAPABILITY_NOT_FOUND,
                    message=(
                        f"CapabilityNotFound: registry not configured to "
                        f"resolve '{target_capability}'"
                    ),
                )

            ranked = self.registry.find(target_capability)
            if not ranked:
                return build_error_response(
                    id=rpc_id,
                    code=JSONRPCErrorCode.CAPABILITY_NOT_FOUND,
                    message=(
                        f"CapabilityNotFound: no active agents found offering capability "
                        f"'{target_capability}'"
                    ),
                )

            best = ranked[0]
            ranking_reason = best.reason
            trace_entry = (
                f"[resolved_agent={best.card.agent_id} capability={target_capability} "
                f"reason={ranking_reason}]"
            )
            updated_trace = (
                f"{uacp_msg.trace_id} {trace_entry}" if uacp_msg.trace_id else trace_entry
            )

            uacp_msg = uacp_msg.model_copy(
                update={
                    "recipient": Party(
                        agent_id=best.card.agent_id,
                        engine=best.card.engine,
                        capability=target_capability,
                    ),
                    "trace_id": updated_trace,
                }
            )
        else:
            return build_error_response(
                id=rpc_id,
                code=JSONRPCErrorCode.INVALID_PARAMS,
                message="Recipient must specify either agent_id or capability",
            )

        # 5. Advance message hop if it is an uninitialized (hop 0) message
        if uacp_msg.circuit_breaker.current_hop == 0:
            uacp_msg = uacp_msg.model_copy(
                update={"circuit_breaker": uacp_msg.circuit_breaker.next_hop()}
            )

        # 6. Run circuit breaker check
        breaker_err = await self.breaker.check(resolved_session_id, uacp_msg)
        if breaker_err is not None:
            err_response = breaker_err.to_response()
            err_response.id = rpc_id
            await self._record_response(resolved_session_id, uacp_msg, err_response)
            return err_response

        # 7. Deliver message through transport
        req = JSONRPCRequest(
            id=rpc_id,
            method=f"uacp.{uacp_msg.intent.value}",
            params=uacp_msg,
        )
        response = await self.transport.deliver(req)

        # 8. Record and return the response
        await self._record_response(resolved_session_id, uacp_msg, response)
        return response
