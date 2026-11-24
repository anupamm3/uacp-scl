"""Unit tests for Agent Alpha (research and analysis agent).

Validates:
- Centralized capability declaration on AgentCard
- Chunking large ledgers by batch and merging anomalies
- De-duplication by (batch, id)
- Output persistence on Whiteboard under 'findings.v1'
- Outbound task_delegation message sent to Agent Beta via Router
- Automatic repair recovery on schema violations
- QuotaGuard / RateLimited backoff retry recovery
- Free-tier call limits with graceful partial 'incomplete' status
- Text-only tasks without ledger payload
- Compatibility with legacy list-of-strings output
- Self-assessed difficulty scoring (1 to 5)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from examples.data.make_ledger import generate_clean_ledger
from uacp.adapters.base import RateLimited
from uacp.adapters.fake_adapter import FakeAdapter
from uacp.agents.alpha_research import (
    ALPHA_CAPABILITIES,
    AlphaResearchAgent,
    Finding,
    FindingsReport,
    chunk_ledger,
)
from uacp.memory.whiteboard import InMemoryWhiteboardBackend, WhiteboardClient
from uacp.protocol.capabilities import CapabilityRegistry
from uacp.protocol.jsonrpc import JSONRPCRequest, JSONRPCResponse, build_response
from uacp.protocol.messages import (
    Intent,
    Party,
    TaskDelegationPayload,
    UACPMessage,
)
from uacp.protocol.router import Router
from uacp.protocol.transport.inproc import InProcessTransport

SCL_PATH = Path("examples/financial_auditor.scl")


@pytest.fixture
def whiteboard_backend() -> InMemoryWhiteboardBackend:
    return InMemoryWhiteboardBackend()


@pytest.fixture
def transport() -> InProcessTransport:
    return InProcessTransport()


@pytest.fixture
def registry() -> CapabilityRegistry:
    return CapabilityRegistry()


@pytest.fixture
def router(
    whiteboard_backend: InMemoryWhiteboardBackend,
    transport: InProcessTransport,
    registry: CapabilityRegistry,
) -> Router:
    return Router(whiteboard=whiteboard_backend, transport=transport, registry=registry)


# ── Test Cases ────────────────────────────────────────────────────────────────


def test_alpha_card_abilities(registry: CapabilityRegistry, router: Router) -> None:
    """Verify Alpha declares its abilities with strengths, complexity level 3, and cost hint."""
    adapter = FakeAdapter()
    agent = AlphaResearchAgent(
        agent_id="alpha_test",
        scl_path=SCL_PATH,
        adapter=adapter,
    )
    agent.register(registry, router)

    card = agent.card
    assert card is not None
    assert card.agent_id == "alpha_test"

    caps = {c.name: c for c in card.capabilities}
    assert "financial_audit" in caps
    audit_cap = caps["financial_audit"]
    assert "analysis" in audit_cap.strengths
    assert "pattern_detection" in audit_cap.strengths
    assert audit_cap.complexity_level == 3
    assert audit_cap.cost_hint == 0.2


@pytest.mark.asyncio
async def test_alpha_chunking_and_merging() -> None:
    """Verify ledger is chunked across batches and findings are merged."""
    clean_ledger = generate_clean_ledger(batches=3, entries_per_batch=20, seed=42)
    assert len(clean_ledger) == 60

    # Script FakeAdapter to return 1 distinct anomaly per chunk
    script = [
        {
            "audit_complete": True,
            "anomalies": [
                {
                    "id": "TX-B01-0001",
                    "batch": "BATCH-2026-Q1",
                    "delta": 10.0,
                    "type": "sign_flip",
                    "evidence": "Inverted debit/credit polarity",
                }
            ],
        },
        {
            "audit_complete": True,
            "anomalies": [
                {
                    "id": "TX-B02-0003",
                    "batch": "BATCH-2026-Q2",
                    "delta": 0.05,
                    "type": "rounding_drift",
                    "evidence": "Counterpart rounding mismatch",
                }
            ],
        },
        {
            "audit_complete": True,
            "anomalies": [
                {
                    "id": "TX-B03-0005",
                    "batch": "BATCH-2026-Q3",
                    "delta": 500.0,
                    "type": "missing_counterpart",
                    "evidence": "Orphaned transaction entry",
                }
            ],
        },
    ]

    adapter = FakeAdapter(script=script)
    agent = AlphaResearchAgent(
        agent_id="alpha_test",
        scl_path=SCL_PATH,
        adapter=adapter,
        entries_per_batch=20,
    )

    report, usage, repairs = await agent.audit_ledger(clean_ledger)

    assert usage.requests == 3
    assert report.audit_complete is True
    assert report.status == "success"
    assert report.total_anomalies == 3
    ids = [a.id for a in report.anomalies]
    assert ids == ["TX-B01-0001", "TX-B02-0003", "TX-B03-0005"]


@pytest.mark.asyncio
async def test_alpha_deduplication() -> None:
    """Verify findings are de-duplicated by (batch, id)."""
    ledger = [
        {"id": "TX-01", "batch": "BATCH-1", "debit": 100.0, "credit": 0.0},
        {"id": "TX-02", "batch": "BATCH-1", "debit": 0.0, "credit": 100.0},
    ]

    duplicate_anomaly = {
        "id": "TX-DUP-01",
        "batch": "BATCH-1",
        "delta": 50.0,
        "type": "duplicate_id",
        "evidence": "Duplicated entry",
    }

    # Two chunks return identical (batch, id)
    script = [
        {"audit_complete": True, "anomalies": [duplicate_anomaly]},
        {"audit_complete": True, "anomalies": [duplicate_anomaly]},
    ]

    adapter = FakeAdapter(script=script)
    agent = AlphaResearchAgent(
        agent_id="alpha_test",
        scl_path=SCL_PATH,
        adapter=adapter,
        entries_per_batch=1,  # Forces 2 chunks
    )

    report, usage, _ = await agent.audit_ledger(ledger)

    assert usage.requests == 2
    assert report.total_anomalies == 1
    assert report.anomalies[0].id == "TX-DUP-01"
    assert report.anomalies[0].batch == "BATCH-1"


@pytest.mark.asyncio
async def test_alpha_whiteboard_write_and_beta_message(
    whiteboard_backend: InMemoryWhiteboardBackend,
    transport: InProcessTransport,
    router: Router,
) -> None:
    """Verify findings are written to Whiteboard under 'findings.v1' and Beta is notified."""
    session_id = await whiteboard_backend.create_session(ttl_s=3600.0)
    wb_client = WhiteboardClient(backend=whiteboard_backend, session_id=session_id, author="alpha")

    # Register mock Beta agent to verify message reception
    beta_received: list[UACPMessage] = []

    async def beta_handler(req: JSONRPCRequest) -> JSONRPCResponse:
        msg: UACPMessage = req.params
        beta_received.append(msg)
        return build_response(id=req.id, result={"status": "acknowledged"})

    transport.register_agent("beta", beta_handler)

    script = [
        {
            "audit_complete": True,
            "anomalies": [
                {
                    "id": "TX-ANOM-1",
                    "batch": "BATCH-2026-Q1",
                    "delta": 250.0,
                    "type": "unauthorized_disbursement",
                    "evidence": "Suspicious vendor payment",
                }
            ],
        }
    ]

    adapter = FakeAdapter(script=script)
    agent = AlphaResearchAgent(
        agent_id="alpha",
        scl_path=SCL_PATH,
        adapter=adapter,
        whiteboard=wb_client,
        router=router,
    )
    transport.register_agent("alpha", agent._on_transport_request)

    ledger = [{"id": "TX-01", "batch": "BATCH-2026-Q1", "debit": 250.0, "credit": 0.0}]
    inbound_message = UACPMessage(
        correlation_id="corr-123",
        trace_id="trace-abc",
        sender=Party(agent_id="orchestrator"),
        recipient=Party(agent_id="alpha"),
        intent=Intent.task_delegation,
        payload={
            "instruction": "Audit transactions",
            "whiteboard_session_ref": session_id,
            "ledger_v1": ledger,
        },
    )

    response = await agent.handle(inbound_message)

    # 1. Check response summary
    assert response.intent == Intent.task_result
    assert response.payload["status"] == "success"
    assert response.payload["counts"]["anomalies"] == 1
    assert response.payload["result_ref"] == "findings.v1"

    # 2. Check Whiteboard contains 'findings.v1'
    stored_report = await wb_client.read_model("findings.v1", FindingsReport)
    assert stored_report.total_anomalies == 1
    assert stored_report.anomalies[0].id == "TX-ANOM-1"

    # 3. Check message was delivered to Beta
    assert len(beta_received) == 1
    beta_msg = beta_received[0]
    assert beta_msg.recipient.agent_id == "beta"
    assert beta_msg.intent == Intent.task_delegation
    assert beta_msg.payload["whiteboard_session_ref"] == session_id
    assert "Mitigate" in beta_msg.payload["instruction"]


@pytest.mark.asyncio
async def test_alpha_chunk_repair_behaviour() -> None:
    """Verify Alpha recovers when a chunk returns invalid JSON on first attempt."""
    # First response is invalid schema (missing required fields 'audit_complete', 'anomalies')
    bad_reply = '{"status": "broken", "details": "not matching SCL schema"}'
    good_reply = json.dumps(
        {
            "audit_complete": True,
            "anomalies": [
                {
                    "id": "TX-REPAIRED-01",
                    "batch": "BATCH-1",
                    "delta": 10.0,
                    "type": "sign_flip",
                    "evidence": "Corrected after repair prompt",
                }
            ],
        }
    )

    adapter = FakeAdapter(script=[bad_reply, good_reply])
    agent = AlphaResearchAgent(
        agent_id="alpha",
        scl_path=SCL_PATH,
        adapter=adapter,
    )

    ledger = [{"id": "TX-01", "batch": "BATCH-1", "debit": 10.0, "credit": 0.0}]
    report, usage, repairs = await agent.audit_ledger(ledger)

    assert repairs == 1
    assert usage.requests == 2
    assert report.total_anomalies == 1
    assert report.anomalies[0].id == "TX-REPAIRED-01"


@pytest.mark.asyncio
async def test_alpha_rate_limited_retry_success() -> None:
    """Verify Alpha catches RateLimited, backs off, retries, and succeeds."""
    good_reply = {
        "audit_complete": True,
        "anomalies": [
            {
                "id": "TX-OK-01",
                "batch": "BATCH-1",
                "delta": 0.0,
                "type": "duplicate_id",
                "evidence": "Detected after 429 retry",
            }
        ],
    }

    # First attempt raises RateLimited, second succeeds
    adapter = FakeAdapter(script=[RateLimited("429 Too Many Requests", retry_after=0.01), good_reply])
    agent = AlphaResearchAgent(
        agent_id="alpha",
        scl_path=SCL_PATH,
        adapter=adapter,
        rate_limit_backoff=0.01,
    )

    ledger = [{"id": "TX-01", "batch": "BATCH-1", "debit": 10.0, "credit": 0.0}]
    report, usage, repairs = await agent.audit_ledger(ledger)

    assert report.status == "success"
    assert report.total_anomalies == 1
    assert report.anomalies[0].id == "TX-OK-01"


@pytest.mark.asyncio
async def test_alpha_free_tier_call_limit_incomplete() -> None:
    """Verify Alpha stops gracefully with status 'incomplete' when max_llm_calls is reached."""
    # 3 batches, entries_per_batch=20 -> 3 chunks
    clean_ledger = generate_clean_ledger(batches=3, entries_per_batch=20, seed=42)

    script = [
        {
            "audit_complete": True,
            "anomalies": [
                {
                    "id": "TX-B01-0001",
                    "batch": "BATCH-2026-Q1",
                    "delta": 1.0,
                    "type": "sign_flip",
                    "evidence": "Finding 1",
                }
            ],
        },
        {
            "audit_complete": True,
            "anomalies": [
                {
                    "id": "TX-B02-0003",
                    "batch": "BATCH-2026-Q2",
                    "delta": 2.0,
                    "type": "sign_flip",
                    "evidence": "Finding 2",
                }
            ],
        },
    ]

    adapter = FakeAdapter(script=script)
    agent = AlphaResearchAgent(
        agent_id="alpha",
        scl_path=SCL_PATH,
        adapter=adapter,
        entries_per_batch=20,
        max_llm_calls=2,  # Limit to 2 calls
    )

    report, usage, _ = await agent.audit_ledger(clean_ledger)

    assert usage.requests == 2
    assert report.status == "incomplete"
    assert report.audit_complete is False
    assert report.total_anomalies == 2
    assert report.metadata["is_incomplete"] is True
    assert report.metadata["total_chunks"] == 3
    assert report.metadata["processed_chunks"] == 2


@pytest.mark.asyncio
async def test_alpha_plain_text_without_ledger() -> None:
    """Verify Alpha executes SCL directly on plain text instruction when no ledger is provided."""
    reply = {
        "audit_complete": True,
        "anomalies": [
            {
                "id": "TX-MANUAL-1",
                "batch": "BATCH-DEFAULT",
                "delta": 100.0,
                "type": "unauthorized_disbursement",
                "evidence": "Text finding",
            }
        ],
    }

    adapter = FakeAdapter(script=[reply])
    agent = AlphaResearchAgent(
        agent_id="alpha",
        scl_path=SCL_PATH,
        adapter=adapter,
    )

    msg = UACPMessage(
        correlation_id="corr-text",
        trace_id="trace-text",
        sender=Party(agent_id="user"),
        recipient=Party(agent_id="alpha"),
        intent=Intent.task_delegation,
        payload={"instruction": "Perform ad-hoc review of vendor policy"},
    )

    response = await agent.handle(msg)

    assert response.intent == Intent.task_result
    assert response.payload["status"] == "success"
    assert response.payload["counts"]["anomalies"] == 1


def test_alpha_string_list_compatibility() -> None:
    """Verify FindingsReport coerces legacy simple list-of-strings into Finding objects."""
    raw_strings = [
        "TX-B01-0001: Rounding drift discrepancy of 0.03",
        "Suspicious manual override without approval ID",
    ]

    report = FindingsReport(audit_complete=True, anomalies=raw_strings)
    assert report.total_anomalies == 2
    assert report.anomalies[0].id == "TX-B01-0001"
    assert "Rounding drift" in report.anomalies[0].evidence
    assert report.anomalies[1].id.startswith("STR-")
    assert "Suspicious manual" in report.anomalies[1].evidence


def test_alpha_difficulty_assessment() -> None:
    """Verify optional self-assessed difficulty (1 to 5) per finding and aggregate report."""
    f1 = Finding(id="TX-1", type="duplicate_id", difficulty=2)
    f2 = Finding(id="TX-2", type="rounding_drift", difficulty=4)

    report = FindingsReport(anomalies=[f1, f2])
    assert report.difficulty == 4

    # When individual difficulties are omitted, heuristic sets difficulty based on type
    f3 = Finding(id="TX-3", type="rounding_drift")
    report2 = FindingsReport(anomalies=[f3])
    assert report2.difficulty == 4


@pytest.mark.asyncio
async def test_alpha_direct_analyze_call() -> None:
    """Verify direct call to agent.analyze without UACP envelope."""
    reply = {
        "audit_complete": True,
        "anomalies": [
            {
                "id": "TX-DIRECT-01",
                "batch": "BATCH-1",
                "delta": 30.0,
                "type": "sign_flip",
                "evidence": "Direct analysis",
            }
        ],
    }
    adapter = FakeAdapter(script=[reply])
    agent = AlphaResearchAgent(
        agent_id="alpha",
        scl_path=SCL_PATH,
        adapter=adapter,
    )

    ledger = [{"id": "TX-01", "batch": "BATCH-1", "debit": 30.0, "credit": 0.0}]
    report = await agent.analyze(payload={"ledger_v1": ledger})
    assert report.total_anomalies == 1
    assert report.anomalies[0].id == "TX-DIRECT-01"

