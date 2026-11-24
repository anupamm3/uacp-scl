"""FastAPI application for the UACP HTTP server."""

from __future__ import annotations

import asyncio
import contextlib
import hmac
import json
import os
import time
from typing import Any

from fastapi import FastAPI, HTTPException, Request, Response, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from uacp.memory.whiteboard.base import WhiteboardBackend
from uacp.memory.whiteboard.inmem_backend import InMemoryWhiteboardBackend
from uacp.memory.whiteboard.sqlite_backend import SqliteWhiteboardBackend
from uacp.protocol.capabilities import AgentCard, CapabilityRegistry
from uacp.protocol.jsonrpc import (
    JSONRPCError,
    JSONRPCErrorCode,
    JSONRPCRequest,
    build_error_response,
    parse_request,
)
from uacp.protocol.router import Router
from uacp.protocol.transport.inproc import InProcessTransport


class HaltRequest(BaseModel):
    """Payload model for manual session halt requests."""

    reason: str = "manual halt"


def create_app(
    router: Router | None = None,
    whiteboard: WhiteboardBackend | None = None,
    registry: CapabilityRegistry | None = None,
    auth_secret: str | None = None,
    max_body_bytes: int = 1_048_576,
) -> FastAPI:
    """Create and configure the FastAPI application for the UACP HTTP server.

    Args:
        router: Router instance for routing UACP messages. If None, one is created.
        whiteboard: Whiteboard backend for session state. If None, SqliteWhiteboardBackend is used.
        registry: CapabilityRegistry for agent discovery. If None, a new one is created.
        auth_secret: Optional shared secret for authentication. Defaults to UACP_SERVER_SECRET env.
        max_body_bytes: Maximum allowed request body size in bytes (default: 1MB).

    Returns:
        Configured FastAPI application instance.
    """
    app = FastAPI(title="UACP HTTP Server", version="0.1.0")

    # Wire default dependencies if not supplied
    wb: WhiteboardBackend = whiteboard or SqliteWhiteboardBackend("uacp_whiteboard.db")
    reg: CapabilityRegistry = registry or CapabilityRegistry()
    rtr: Router = router or Router(
        whiteboard=wb,
        transport=InProcessTransport(),
        registry=reg,
    )

    effective_secret = (
        auth_secret if auth_secret is not None else os.environ.get("UACP_SERVER_SECRET")
    )

    # ── Request Body Size Limit Middleware ────────────────────────────────────
    @app.middleware("http")
    async def body_size_limit_middleware(request: Request, call_next: Any) -> Response:
        content_length = request.headers.get("content-length")
        if content_length:
            try:
                if int(content_length) > max_body_bytes:
                    return Response(
                        status_code=413,
                        content=json.dumps({"detail": "Payload Too Large"}),
                        media_type="application/json",
                    )
            except ValueError:
                pass
        res: Response = await call_next(request)
        return res

    # ── Shared-Secret Auth Middleware ─────────────────────────────────────────
    @app.middleware("http")
    async def auth_middleware(request: Request, call_next: Any) -> Response:
        if effective_secret:
            path = request.url.path
            if path not in ("/healthz", "/docs", "/openapi.json"):
                secret = request.headers.get("x-uacp-secret")
                if not secret:
                    auth_h = request.headers.get("authorization", "")
                    if auth_h.startswith("Bearer "):
                        secret = auth_h[7:].strip()
                if not secret:
                    return Response(
                        status_code=401,
                        content=json.dumps({"detail": "Authentication required"}),
                        media_type="application/json",
                    )
                if not hmac.compare_digest(secret, effective_secret):
                    return Response(
                        status_code=403,
                        content=json.dumps({"detail": "Invalid authentication credentials"}),
                        media_type="application/json",
                    )
        res: Response = await call_next(request)
        return res

    # ── Endpoints ─────────────────────────────────────────────────────────────

    @app.get("/healthz")
    async def healthz() -> dict[str, str]:
        """Liveness health check endpoint."""
        return {"status": "ok"}

    @app.post("/rpc")
    async def rpc_endpoint(request: Request) -> Response:
        """Process incoming JSON-RPC 2.0 requests wrapping UACP messages."""
        body = await request.body()
        if len(body) > max_body_bytes:
            return Response(
                status_code=413,
                content=json.dumps({"detail": "Payload Too Large"}),
                media_type="application/json",
            )

        try:
            raw_json = json.loads(body.decode("utf-8"))
        except Exception as exc:
            err_resp = build_error_response(
                id=None,
                code=JSONRPCErrorCode.PARSE_ERROR,
                message=f"Parse error: invalid JSON ({exc})",
            )
            return JSONResponse(status_code=200, content=err_resp.model_dump(mode="json"))

        try:
            parsed_req = parse_request(raw_json)
        except JSONRPCError as err:
            return JSONResponse(status_code=200, content=err.to_response().model_dump(mode="json"))
        except Exception as exc:
            req_id = raw_json.get("id") if isinstance(raw_json, dict) else None
            err_resp = build_error_response(
                id=req_id,
                code=JSONRPCErrorCode.INVALID_REQUEST,
                message=f"Invalid Request: {exc}",
            )
            return JSONResponse(status_code=200, content=err_resp.model_dump(mode="json"))

        if not isinstance(parsed_req, JSONRPCRequest):
            return JSONResponse(status_code=200, content=parsed_req.model_dump(mode="json"))

        try:
            recipient_id = parsed_req.params.recipient.agent_id
            if (
                parsed_req.params.circuit_breaker.current_hop > 0
                and rtr.transport.has_agent(recipient_id)
            ):
                rpc_response = await rtr.transport.deliver(parsed_req)
            else:
                rpc_response = await rtr.send(parsed_req)
            return JSONResponse(status_code=200, content=rpc_response.model_dump(mode="json"))
        except Exception as exc:
            err_resp = build_error_response(
                id=parsed_req.id,
                code=JSONRPCErrorCode.INTERNAL_ERROR,
                message=f"Internal router error: {exc}",
            )
            return JSONResponse(status_code=200, content=err_resp.model_dump(mode="json"))

    @app.get("/agents")
    async def list_agents() -> list[dict[str, Any]]:
        """Return registered AgentCards currently in the CapabilityRegistry."""
        return [card.model_dump(mode="json") for card in reg._cards.values()]

    @app.post("/agents")
    async def register_agent(card: AgentCard) -> dict[str, Any]:
        """Register or update an AgentCard in the CapabilityRegistry."""
        reg.register(card)
        return {"status": "registered", "agent_id": card.agent_id}

    @app.get("/sessions/{session_id}")
    async def get_session(session_id: str) -> dict[str, Any]:
        """Return Whiteboard state snapshot, authoritative hop count, and session status."""
        try:
            state = await wb.get_state(session_id)
        except Exception as exc:
            raise HTTPException(
                status_code=404,
                detail=f"Session '{session_id}' not found: {exc}",
            ) from exc

        current_hop = 0
        max_hops = 5
        expires_at = 0.0
        is_closed = False

        if isinstance(wb, SqliteWhiteboardBackend):
            with wb._lock:
                cur = wb._conn.execute(
                    "SELECT current_hop, max_hops, expires_at, is_closed "
                    "FROM sessions WHERE session_id = ?",
                    (session_id,),
                )
                row = cur.fetchone()
                if not row:
                    raise HTTPException(
                        status_code=404,
                        detail=f"Session '{session_id}' not found",
                    )
                current_hop, max_hops, expires_at, is_closed_val = row
                is_closed = bool(is_closed_val)
        elif isinstance(wb, InMemoryWhiteboardBackend):
            sess = wb._sessions.get(session_id)
            if sess is None:
                raise HTTPException(
                    status_code=404,
                    detail=f"Session '{session_id}' not found",
                )
            current_hop = sess.current_hop
            max_hops = sess.max_hops
            expires_at = sess.expires_at
            is_closed = sess.is_closed

        now = time.time()
        is_halted = is_closed or (session_id in getattr(rtr.breaker, "_halted_sessions", {}))
        if not is_halted:
            events = await wb.events(session_id)
            if any(e.event_type == "session.halted" for e in events):
                is_halted = True

        if is_halted:
            status = "halted"
        elif expires_at > 0 and now > expires_at:
            status = "expired"
        else:
            status = "active"

        return {
            "session_id": session_id,
            "state": state,
            "current_hop": current_hop,
            "max_hops": max_hops,
            "status": status,
            "expires_at": expires_at,
        }

    @app.get("/sessions/{session_id}/events")
    async def get_session_events(
        session_id: str,
        since: str | int | None = None,
    ) -> list[dict[str, Any]]:
        """Return Whiteboard event list for a session, optionally filtered by since."""
        try:
            events = await wb.events(session_id, since=since)
            return [e.model_dump(mode="json") for e in events]
        except Exception as exc:
            raise HTTPException(
                status_code=404,
                detail=f"Failed to retrieve events for session '{session_id}': {exc}",
            ) from exc

    @app.websocket("/sessions/{session_id}/stream")
    async def stream_session_events(
        websocket: WebSocket,
        session_id: str,
    ) -> None:
        """Stream Whiteboard events as JSON in real-time over WebSocket."""
        if effective_secret:
            secret = websocket.headers.get("x-uacp-secret")
            if not secret:
                auth_h = websocket.headers.get("authorization", "")
                if auth_h.startswith("Bearer "):
                    secret = auth_h[7:].strip()
            if not secret:
                secret = websocket.query_params.get("secret")
            if not secret or not hmac.compare_digest(secret, effective_secret):
                await websocket.close(code=1008)
                return

        await websocket.accept()
        last_event_id: int | str | None = None

        try:
            # Send initial existing events
            initial_events = await wb.events(session_id)
            for ev in initial_events:
                await websocket.send_text(ev.model_dump_json())
                last_event_id = ev.event_id

            while True:
                await asyncio.sleep(0.05)
                new_events = await wb.events(session_id, since=last_event_id)
                for ev in new_events:
                    await websocket.send_text(ev.model_dump_json())
                    last_event_id = ev.event_id
        except WebSocketDisconnect:
            pass
        except Exception:
            with contextlib.suppress(Exception):
                await websocket.close()

    @app.post("/sessions/{session_id}/halt")
    async def halt_session(
        session_id: str,
        req: HaltRequest | None = None,
    ) -> dict[str, Any]:
        """Manually halt a session and record a session.halted event."""
        actual_req = req or HaltRequest()
        err = await rtr.halt(session_id, actual_req.reason)
        return {
            "status": "halted",
            "session_id": session_id,
            "reason": actual_req.reason,
            "error": err.to_response().model_dump(mode="json"),
        }

    return app
