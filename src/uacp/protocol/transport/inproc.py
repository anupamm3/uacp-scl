"""In-process transport for direct agent-to-agent communication in the same process."""

from __future__ import annotations

import inspect
from collections.abc import Awaitable, Callable
from typing import Any, TypeAlias

from uacp.protocol.jsonrpc import (
    JSONRPCError,
    JSONRPCErrorCode,
    JSONRPCRequest,
    JSONRPCResponse,
    build_error_response,
    build_response,
)
from uacp.protocol.messages import UACPMessage

AgentHandler: TypeAlias = Callable[
    [JSONRPCRequest],
    Awaitable[JSONRPCResponse | UACPMessage | dict[str, Any] | None],
]


class InProcessTransport:
    """In-process transport delivering messages directly to registered agent async handlers."""

    def __init__(self) -> None:
        self._handlers: dict[str, AgentHandler] = {}

    def register_agent(self, agent_id: str, handler: AgentHandler) -> None:
        """Register an asynchronous message handler for an agent ID."""
        self._handlers[agent_id] = handler

    def deregister_agent(self, agent_id: str) -> bool:
        """Deregister an agent's message handler.

        Returns True if the agent existed and was removed, False otherwise.
        """
        return self._handlers.pop(agent_id, None) is not None

    def has_agent(self, agent_id: str) -> bool:
        """Return True if an agent handler is registered under agent_id."""
        return agent_id in self._handlers

    def registered_agents(self) -> list[str]:
        """Return a list of all registered agent identifiers."""
        return list(self._handlers.keys())

    async def deliver(self, request: JSONRPCRequest | UACPMessage) -> JSONRPCResponse:
        """Deliver a request or message directly to the recipient agent's registered handler.

        Args:
            request: The JSONRPCRequest or UACPMessage to deliver.

        Returns:
            A validated JSONRPCResponse object.
        """
        rpc_req: JSONRPCRequest
        if isinstance(request, JSONRPCRequest):
            rpc_req = request
        else:
            rpc_req = JSONRPCRequest(
                id=request.message_id,
                method=f"uacp.{request.intent.value}",
                params=request,
            )

        recipient_id = rpc_req.params.recipient.agent_id
        handler = self._handlers.get(recipient_id)

        if handler is None:
            return build_error_response(
                id=rpc_req.id,
                code=JSONRPCErrorCode.CAPABILITY_NOT_FOUND,
                message=(
                    f"CapabilityNotFound: agent '{recipient_id}' is not registered on transport"
                ),
            )

        try:
            raw_res = handler(rpc_req)
            if inspect.isawaitable(raw_res):
                handler_result = await raw_res
            else:
                handler_result = raw_res
        except JSONRPCError as err:
            return build_error_response(
                id=rpc_req.id,
                code=err.code,
                message=err.message,
                data=err.data,
            )
        except Exception as exc:
            return build_error_response(
                id=rpc_req.id,
                code=JSONRPCErrorCode.INTERNAL_ERROR,
                message=f"Internal handler error in agent '{recipient_id}': {exc}",
                data=str(exc),
            )

        if isinstance(handler_result, JSONRPCResponse):
            return handler_result

        if isinstance(handler_result, UACPMessage):
            return build_response(id=rpc_req.id, result=handler_result.model_dump())

        if handler_result is None:
            return build_response(id=rpc_req.id, result={"status": "acknowledged"})

        return build_response(id=rpc_req.id, result=handler_result)
