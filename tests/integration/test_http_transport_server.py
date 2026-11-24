"""Integration tests for HTTP transport, FastAPI server, and multi-process simulation."""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import httpx
import pytest
from examples.data.make_ledger import (
    find_anomalies_baseline,
    generate_clean_ledger,
)
from fastapi.testclient import TestClient

from uacp.adapters.fake_adapter import FakeAdapter
from uacp.agents.alpha_research import AlphaResearchAgent
from uacp.agents.beta_execution import BetaExecutionAgent
from uacp.memory.whiteboard.client import WhiteboardClient
from uacp.memory.whiteboard.sqlite_backend import SqliteWhiteboardBackend
from uacp.protocol.capabilities import AgentCard, Capability, CapabilityRegistry
from uacp.protocol.jsonrpc import JSONRPCErrorCode, JSONRPCRequest
from uacp.protocol.messages import Intent, Party, TaskDelegationPayload, UACPMessage
from uacp.protocol.router import Router
from uacp.protocol.transport.http import HTTPTransport
from uacp.protocol.transport.inproc import InProcessTransport
from uacp.server.app import create_app


@pytest.mark.asyncio
async def test_server_healthz() -> None:
    """Verify GET /healthz returns ok without auth."""
    app = create_app(auth_secret="secret123")
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=cast(Any, app)),
        base_url="http://testserver",
    ) as client:
        resp = await client.get("/healthz")
        assert resp.status_code == 200
        assert resp.json() == {"status": "ok"}


@pytest.mark.asyncio
async def test_server_agents_listing_and_registration() -> None:
    """Verify listing and remote registration of AgentCards via /agents."""
    registry = CapabilityRegistry()
    app = create_app(registry=registry)

    card = AgentCard(
        agent_id="test_worker",
        engine="default",
        role="Worker for testing",
        capabilities=[Capability(name="test_task", complexity_level=1)],
    )

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=cast(Any, app)),
        base_url="http://testserver",
    ) as client:
        # Initial list empty
        resp = await client.get("/agents")
        assert resp.status_code == 200
        assert resp.json() == []

        # Register agent card
        post_resp = await client.post("/agents", json=card.model_dump())
        assert post_resp.status_code == 200
        assert post_resp.json() == {"status": "registered", "agent_id": "test_worker"}

        # Verify listed
        list_resp = await client.get("/agents")
        assert list_resp.status_code == 200
        agents = list_resp.json()
        assert len(agents) == 1
        assert agents[0]["agent_id"] == "test_worker"


@pytest.mark.asyncio
async def test_server_session_lifecycle(tmp_path: Path) -> None:
    """Verify session inspection, event retrieval, and manual halt."""
    db_path = str(tmp_path / "wb.db")
    wb = SqliteWhiteboardBackend(db_path)
    router = Router(whiteboard=wb, transport=InProcessTransport())
    app = create_app(router=router, whiteboard=wb)

    session_id = await wb.create_session(ttl_s=3600.0, max_hops=5)
    await wb.put(session_id, "test_key", {"status": "init"}, author="test")

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=cast(Any, app)),
        base_url="http://testserver",
    ) as client:
        # Session info
        resp = await client.get(f"/sessions/{session_id}")
        assert resp.status_code == 200
        data = resp.json()
        assert data["session_id"] == session_id
        assert data["status"] == "active"
        assert "test_key" in data["state"]

        # Events list
        ev_resp = await client.get(f"/sessions/{session_id}/events")
        assert ev_resp.status_code == 200
        events = ev_resp.json()
        assert len(events) >= 1

        # Manual halt
        halt_resp = await client.post(
            f"/sessions/{session_id}/halt",
            json={"reason": "Operator request"},
        )
        assert halt_resp.status_code == 200
        halt_data = halt_resp.json()
        assert halt_data["status"] == "halted"

        # Check session status is now halted
        status_resp = await client.get(f"/sessions/{session_id}")
        assert status_resp.status_code == 200
        assert status_resp.json()["status"] == "halted"

    wb.close_db()


def test_server_websocket_event_streaming(tmp_path: Path) -> None:
    """Verify real-time event streaming over WebSocket."""
    db_path = str(tmp_path / "wb_ws.db")
    wb = SqliteWhiteboardBackend(db_path)
    app = create_app(whiteboard=wb)

    import asyncio

    session_id = asyncio.run(wb.create_session(ttl_s=3600.0))

    client = TestClient(app)
    with client.websocket_connect(f"/sessions/{session_id}/stream") as ws:
        # Append event to whiteboard
        asyncio.run(
            wb.append_event(
                session_id,
                {"event_type": "custom.stream_test", "data": {"seq": 42}},
            )
        )
        # Receive over WebSocket
        msg = ws.receive_json()
        assert msg["event_type"] == "custom.stream_test"
        assert msg["data"]["seq"] == 42

    wb.close_db()


@pytest.mark.asyncio
async def test_server_authentication() -> None:
    """Verify shared-secret authentication enforcement (401/403/200)."""
    app = create_app(auth_secret="correct_secret_99")

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=cast(Any, app)),
        base_url="http://testserver",
    ) as client:
        # Missing auth header -> 401
        r_no_auth = await client.get("/agents")
        assert r_no_auth.status_code == 401

        # Wrong auth secret -> 403
        r_bad_auth = await client.get("/agents", headers={"X-UACP-Secret": "wrong_secret"})
        assert r_bad_auth.status_code == 403

        # Correct via X-UACP-Secret -> 200
        r_ok_custom = await client.get(
            "/agents",
            headers={"X-UACP-Secret": "correct_secret_99"},
        )
        assert r_ok_custom.status_code == 200

        # Correct via Authorization: Bearer -> 200
        r_ok_bearer = await client.get(
            "/agents",
            headers={"Authorization": "Bearer correct_secret_99"},
        )
        assert r_ok_bearer.status_code == 200


@pytest.mark.asyncio
async def test_server_request_body_size_limit() -> None:
    """Verify 413 Payload Too Large is returned when request body exceeds max_body_bytes."""
    app = create_app(max_body_bytes=50)

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=cast(Any, app)),
        base_url="http://testserver",
    ) as client:
        large_payload = {"padding": "x" * 200}
        resp = await client.post("/rpc", json=large_payload)
        assert resp.status_code == 413


@pytest.mark.asyncio
async def test_server_malformed_json_and_invalid_jsonrpc() -> None:
    """Verify standard JSON-RPC 2.0 error codes for malformed JSON and invalid requests."""
    app = create_app()

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=cast(Any, app)),
        base_url="http://testserver",
    ) as client:
        # Malformed JSON -> -32700 Parse error
        resp_malformed = await client.post(
            "/rpc",
            content=b"{bad_json: invalid}",
            headers={"Content-Type": "application/json"},
        )
        assert resp_malformed.status_code == 200
        data = resp_malformed.json()
        assert data["error"]["code"] == JSONRPCErrorCode.PARSE_ERROR

        # Invalid JSON-RPC structure (missing jsonrpc) -> -32600 Invalid Request
        resp_invalid = await client.post(
            "/rpc",
            json={"id": "123"},
        )
        assert resp_invalid.status_code == 200
        data_invalid = resp_invalid.json()
        assert data_invalid["error"]["code"] == JSONRPCErrorCode.INVALID_REQUEST


@pytest.mark.asyncio
async def test_http_transport_local_and_remote_delivery() -> None:
    """Verify HTTPTransport delivers locally registered handlers and falls back to HTTP."""
    transport = HTTPTransport(endpoint="http://remote-server")

    # Local handler
    async def _mock_handler(req: JSONRPCRequest) -> dict[str, str]:
        return {"result": f"hello from {req.params.recipient.agent_id}"}

    transport.register_agent("local_agent", _mock_handler)
    assert transport.has_agent("local_agent") is True
    assert "local_agent" in transport.registered_agents()

    msg = UACPMessage(
        correlation_id="corr-test",
        trace_id="trace-test",
        sender=Party(agent_id="sender"),
        recipient=Party(agent_id="local_agent"),
        intent=Intent.task_delegation,
        payload={"instruction": "test task", "whiteboard_session_ref": "sess-test"},
    )
    res = await transport.deliver(msg)
    assert res.error is None
    assert res.result == {"result": "hello from local_agent"}

    # Deregister local handler
    assert transport.deregister_agent("local_agent") is True
    assert transport.has_agent("local_agent") is False


@pytest.mark.asyncio
async def test_multiprocess_simulation_alpha_beta_delegation(tmp_path: Path) -> None:
    """Simulate Alpha and Beta as two separate ASGI apps communicating via HTTPTransport.

    Both share a SQLite Whiteboard backend to coordinate state and complete
    a full cross-app forensic delegation round-trip.
    """
    shared_db = str(tmp_path / "shared_multi_process.db")
    wb_beta = SqliteWhiteboardBackend(shared_db)
    wb_alpha = SqliteWhiteboardBackend(shared_db)

    # 1. Alpha creates shared session on SQLite Whiteboard
    session_id = await wb_alpha.create_session(ttl_s=3600.0, max_hops=5)

    # ── Beta ASGI Server ──────────────────────────────────────────────────────
    mitigation_code = """
def mitigate(ledger: list[dict]) -> tuple[list[dict], list[dict]]:
    seen = set()
    corrected = []
    changes = []
    for tx in ledger:
        if tx["id"] in seen:
            changes.append({"action": "drop_duplicate", "id": tx["id"]})
        else:
            seen.add(tx["id"])
            corrected.append(dict(tx))
    return corrected, changes
"""
    expected_response = {
        "python_code": mitigation_code,
        "rationale": "Remediated ledger anomalies by deduplicating transactions.",
        "risks": ["None"],
    }
    remediation_script: list[str | dict[str, Any] | Exception] = [
        expected_response,
        expected_response,
    ]
    fake_adapter_beta = FakeAdapter(script=remediation_script)
    beta_agent = BetaExecutionAgent(
        agent_id="beta",
        adapter=fake_adapter_beta,
        whiteboard=WhiteboardClient(wb_beta, session_id=session_id, author="beta"),
    )

    transport_beta = InProcessTransport()
    transport_beta.register_agent("beta", beta_agent._on_transport_request)
    router_beta = Router(whiteboard=wb_beta, transport=transport_beta)
    app_beta = create_app(router=router_beta, whiteboard=wb_beta)

    # ── Alpha Client App ──────────────────────────────────────────────────────
    http_transport_to_beta = HTTPTransport(
        endpoint="http://beta-server",
        transport=httpx.ASGITransport(app=cast(Any, app_beta)),
    )
    router_alpha = Router(whiteboard=wb_alpha, transport=http_transport_to_beta)
    alpha_script: list[str | dict[str, Any] | Exception] = ["[]"]
    alpha_agent = AlphaResearchAgent(
        agent_id="alpha",
        adapter=FakeAdapter(script=alpha_script),
        whiteboard=WhiteboardClient(wb_alpha, session_id=session_id, author="alpha"),
        router=router_alpha,
    )

    # 2. Alpha writes synthetic ledger & findings into the shared Whiteboard
    clean = generate_clean_ledger(batches=1, entries_per_batch=4, seed=42)
    corrupted = [dict(tx) for tx in clean]
    corrupted.append(dict(clean[0]))

    initial_anomalies = find_anomalies_baseline(corrupted)
    assert len(initial_anomalies) > 0

    findings = [
        {
            "id": a.get("id", "anom-01"),
            "type": a.get("type", "duplicate_id"),
            "severity": "high",
            "description": a.get("description", "Duplicate transaction"),
        }
        for a in initial_anomalies
    ]
    await wb_alpha.put(session_id, "ledger.v1", corrupted, author="alpha")
    await wb_alpha.put(
        session_id,
        "findings.v1",
        {"anomalies": findings},
        author="alpha",
    )

    # 3. Alpha delegates task to Beta over HTTPTransport
    delegation_payload = TaskDelegationPayload(
        instruction="Remediate the duplicate transaction ID findings.",
        whiteboard_session_ref=session_id,
    )
    delegation_msg = UACPMessage(
        correlation_id=session_id,
        trace_id=f"trace-{session_id}",
        sender=Party(agent_id=alpha_agent.agent_id),
        recipient=Party(agent_id=beta_agent.agent_id),
        intent=Intent.task_delegation,
        payload=delegation_payload.model_dump(),
    )

    response = await router_alpha.send(delegation_msg, session_id=session_id)

    # 4. Verify HTTP JSON-RPC 2.0 response from Beta ASGI app
    assert response.error is None
    assert response.result is not None
    payload = response.result.get("payload", {})
    assert payload.get("status") == "success", f"Payload was: {payload}"

    # 5. Verify Beta wrote final report into the shared SQLite Whiteboard
    state = await wb_alpha.get_state(session_id)
    assert "final_report.v1" in state

    wb_alpha.close_db()
    wb_beta.close_db()
