"""HTTP transport for cross-process agent communication."""

from __future__ import annotations

import asyncio
import inspect

import httpx

from uacp.protocol.jsonrpc import (
    JSONRPCError,
    JSONRPCErrorCode,
    JSONRPCRequest,
    JSONRPCResponse,
    build_error_response,
    build_response,
)
from uacp.protocol.messages import UACPMessage
from uacp.protocol.transport.inproc import AgentHandler, InProcessTransport


class HTTPTransport(InProcessTransport):
    """HTTP transport delivering JSON-RPC UACP messages over HTTP using httpx."""

    def __init__(
        self,
        endpoint: str = "http://127.0.0.1:8000",
        *,
        transport: httpx.AsyncBaseTransport | None = None,
        auth_secret: str | None = None,
        timeout: float = 30.0,
        max_retries: int = 3,
        backoff_factor: float = 0.2,
        agent_endpoints: dict[str, str] | None = None,
        agent_transports: dict[str, httpx.AsyncBaseTransport] | None = None,
    ) -> None:
        """Initialize HTTP transport with optional authentication, retries, and endpoints."""
        super().__init__()
        self.endpoint = endpoint.rstrip("/")
        self.transport = transport
        self.auth_secret = auth_secret
        self.timeout = timeout
        self.max_retries = max(1, max_retries)
        self.backoff_factor = backoff_factor
        self.agent_endpoints: dict[str, str] = {
            k: v.rstrip("/") for k, v in (agent_endpoints or {}).items()
        }
        self.agent_transports: dict[str, httpx.AsyncBaseTransport] = agent_transports or {}

    def register_agent(self, agent_id: str, handler: AgentHandler) -> None:
        """Register an asynchronous message handler for a local agent ID."""
        self._handlers[agent_id] = handler

    def deregister_agent(self, agent_id: str) -> bool:
        """Deregister a local agent's message handler."""
        return self._handlers.pop(agent_id, None) is not None

    def has_agent(self, agent_id: str) -> bool:
        """Return True if an agent handler is registered locally."""
        return agent_id in self._handlers

    def registered_agents(self) -> list[str]:
        """Return a list of all registered local agent identifiers."""
        return list(self._handlers.keys())

    async def deliver(self, request: JSONRPCRequest | UACPMessage) -> JSONRPCResponse:
        """Deliver a request or message to the target recipient over HTTP or local handler."""
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

        # 1. Dispatch locally if handler exists
        if recipient_id in self._handlers:
            handler = self._handlers[recipient_id]
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

        # 2. Dispatch over HTTP
        target_base = self.agent_endpoints.get(recipient_id, self.endpoint)
        target_url = target_base if target_base.endswith("/rpc") else f"{target_base}/rpc"
        target_transport = self.agent_transports.get(recipient_id, self.transport)

        headers = {"Content-Type": "application/json"}
        if self.auth_secret:
            headers["X-UACP-Secret"] = self.auth_secret
            headers["Authorization"] = f"Bearer {self.auth_secret}"

        payload = rpc_req.model_dump(mode="json")

        last_error: Exception | None = None
        for attempt in range(self.max_retries):
            try:
                async with httpx.AsyncClient(
                    transport=target_transport,
                    timeout=self.timeout,
                ) as client:
                    resp = await client.post(target_url, json=payload, headers=headers)

                if resp.status_code == 200:
                    data = resp.json()
                    return JSONRPCResponse.model_validate(data)
                if resp.status_code in (401, 403):
                    return build_error_response(
                        id=rpc_req.id,
                        code=JSONRPCErrorCode.INTERNAL_ERROR,
                        message=f"HTTP transport auth failed ({resp.status_code}): {resp.text}",
                    )
                if resp.status_code == 413:
                    return build_error_response(
                        id=rpc_req.id,
                        code=JSONRPCErrorCode.INVALID_REQUEST,
                        message="HTTP transport payload rejected: 413 Payload Too Large",
                    )
                return build_error_response(
                    id=rpc_req.id,
                    code=JSONRPCErrorCode.INTERNAL_ERROR,
                    message=f"HTTP transport error {resp.status_code}: {resp.text}",
                )
            except (httpx.ConnectError, httpx.TimeoutException, httpx.NetworkError) as exc:
                last_error = exc
                if attempt < self.max_retries - 1:
                    await asyncio.sleep(self.backoff_factor * (2**attempt))
            except Exception as exc:
                return build_error_response(
                    id=rpc_req.id,
                    code=JSONRPCErrorCode.INTERNAL_ERROR,
                    message=f"HTTP transport delivery failure: {exc}",
                )

        return build_error_response(
            id=rpc_req.id,
            code=JSONRPCErrorCode.INTERNAL_ERROR,
            message=(
                f"HTTP transport connection failed after {self.max_retries} attempts: {last_error}"
            ),
        )
