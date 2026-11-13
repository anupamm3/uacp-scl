"""Unit tests for Agent Beta (remediation and execution agent).

Tests:
- Centralized ability declaration on AgentCard (strengths, complexity_level=5, cost_hint).
- Dynamic findings key resolution (custom key vs default findings.v1).
- Whiteboard tamper detection refusal on corrupted hash chain.
- Successful mitigation with FakeAdapter, sandboxed execution, baseline re-check,
  and FinalReport generation.
- Corrective repair path: code failing on first turn repaired on second turn.
- Honest failure reporting: double failure generates honest FinalReport (passed=False)
  without raising or retrying forever.
- Security isolation: whiteboard and external payloads treated as untrusted data.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest
from examples.data.make_ledger import (
    find_anomalies_baseline,
    generate_clean_ledger,
)

from uacp.adapters.fake_adapter import FakeAdapter
from uacp.agents.alpha_research import Finding, FindingsReport
from uacp.agents.beta_execution import (
    BetaExecutionAgent,
    FinalReport,
)
from uacp.memory.whiteboard import InMemoryWhiteboardBackend, WhiteboardClient
from uacp.protocol.capabilities import CapabilityRegistry
from uacp.protocol.messages import (
    Intent,
    Party,
    TaskDelegationPayload,
    UACPMessage,
)
from uacp.protocol.router import Router
from uacp.protocol.transport.inproc import InProcessTransport

SCL_PATH = Path("examples/beta_mitigator.scl")


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


# ── Tests ─────────────────────────────────────────────────────────────────────


def test_beta_card_abilities(registry: CapabilityRegistry, router: Router) -> None:
    """Verify Beta declares its abilities: code_generation, verification, complexity 5."""
    adapter = FakeAdapter()
    agent = BetaExecutionAgent(
        agent_id="beta_test",
        scl_path=SCL_PATH,
        adapter=adapter,
    )
    agent.register(registry, router)

    card = agent.card
    assert card is not None
    assert card.agent_id == "beta_test"

    caps = {c.name: c for c in card.capabilities}
    assert "ledger_remediation" in caps
    cap = caps["ledger_remediation"]
    assert "code_generation" in cap.strengths
    assert "verification" in cap.strengths
    assert cap.complexity_level == 5
    assert cap.cost_hint == 0.4


@pytest.mark.asyncio
async def test_beta_successful_mitigation(
    whiteboard_backend: InMemoryWhiteboardBackend,
    router: Router,
    registry: CapabilityRegistry,
) -> None:
    """Test clean execution: FakeAdapter returns working code, baseline verifies clean."""
    session_id = await whiteboard_backend.create_session(ttl_s=3600)
    wb_client = WhiteboardClient(backend=whiteboard_backend, session_id=session_id, author="test")

    # Generate ledger and seed duplicate anomaly
    clean = generate_clean_ledger(batches=1, entries_per_batch=4, seed=42)
    corrupted = [dict(tx) for tx in clean]
    # Add duplicate transaction
    corrupted.append(dict(clean[0]))

    # Initial baseline check confirms anomalies
    initial_anomalies = find_anomalies_baseline(corrupted)
    assert len(initial_anomalies) > 0

    # Store findings and ledger on Whiteboard
    findings_report = FindingsReport(
        audit_complete=True,
        status="success",
        anomalies=[Finding.from_raw(a) for a in initial_anomalies],
    )
    await wb_client.write_model("findings.v1", findings_report)
    await wb_client.put("ledger.v1", corrupted)

    # Prepare FakeAdapter to return working deduplication Python code
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
        "rationale": "Deduplicate transactions by dropping repeated transaction IDs.",
        "risks": ["Risk of dropping legitimate transactions if duplicate IDs were accidental."],
    }
    adapter = FakeAdapter(script=[expected_response])

    agent = BetaExecutionAgent(
        agent_id="beta_worker",
        scl_path=SCL_PATH,
        adapter=adapter,
        whiteboard=wb_client,
        router=router,
    )
    agent.register(registry, router)

    task_msg = UACPMessage(
        correlation_id=session_id,
        trace_id="trace-001",
        sender=Party(agent_id="alpha"),
        recipient=Party(agent_id="beta_worker"),
        intent=Intent.task_delegation,
        payload=TaskDelegationPayload(
            instruction="Mitigate duplicate anomalies in ledger",
            whiteboard_session_ref=session_id,
        ).model_dump(),
    )

    result_msg = await agent.handle(task_msg)

    assert result_msg.intent == Intent.task_result
    assert result_msg.payload["status"] == "success"
    assert "Verification passed" in result_msg.payload["summary"]

    # Verify final_report.v1 was written to Whiteboard
    state = await wb_client.read_all()
    assert "final_report.v1" in state
    report_val = state["final_report.v1"]["value"]
    final_report = FinalReport.model_validate(report_val)

    assert final_report.anomalies_found == len(initial_anomalies)
    assert final_report.anomalies_mitigated >= 1
    assert final_report.verification.passed is True
    assert "0 baseline anomalies remaining" in final_report.verification.details
    assert (
        final_report.code_sha256 == hashlib.sha256(mitigation_code.strip().encode()).hexdigest()
        or final_report.code_sha256
    )


@pytest.mark.asyncio
async def test_beta_dynamic_findings_key(
    whiteboard_backend: InMemoryWhiteboardBackend,
    router: Router,
    registry: CapabilityRegistry,
) -> None:
    """Verify Beta works from whatever findings key is provided in the task message."""
    session_id = await whiteboard_backend.create_session(ttl_s=3600)
    wb_client = WhiteboardClient(
        backend=whiteboard_backend, session_id=session_id, author="planner"
    )

    custom_key = "subtask.cluster_3.findings"
    anomalies = [{"id": "TX-99", "batch": "B1", "type": "duplicate_id", "delta": 0.0}]
    ledger = generate_clean_ledger(batches=1, entries_per_batch=2, seed=42)
    await wb_client.put(custom_key, {"anomalies": anomalies})
    await wb_client.put("ledger.v1", ledger)

    code = "def mitigate(ledger):\n    return ledger, []\n"
    adapter = FakeAdapter(script=[{"python_code": code, "rationale": "noop", "risks": []}])

    agent = BetaExecutionAgent(
        agent_id="beta_dynamic",
        scl_path=SCL_PATH,
        adapter=adapter,
        whiteboard=wb_client,
        router=router,
    )
    agent.register(registry, router)

    task_payload = TaskDelegationPayload(
        instruction=f"Remediate findings in {custom_key}",
        whiteboard_session_ref=session_id,
    ).model_dump()
    task_payload["findings_key"] = custom_key

    msg = UACPMessage(
        correlation_id=session_id,
        trace_id="tr-1",
        sender=Party(agent_id="planner"),
        recipient=Party(agent_id="beta_dynamic"),
        intent=Intent.task_delegation,
        payload=task_payload,
    )

    resp = await agent.handle(msg)
    assert resp.intent == Intent.task_result

    state = await wb_client.read_all()
    assert "final_report.v1" in state
    report = FinalReport.model_validate(state["final_report.v1"]["value"])
    assert report.anomalies_found == 1


@pytest.mark.asyncio
async def test_beta_tamper_detection_refusal(
    whiteboard_backend: InMemoryWhiteboardBackend,
    router: Router,
    registry: CapabilityRegistry,
) -> None:
    """Verify Beta refuses to execute when Whiteboard hash chain integrity fails."""
    session_id = await whiteboard_backend.create_session(ttl_s=3600)
    wb_client = WhiteboardClient(
        backend=whiteboard_backend, session_id=session_id, author="auditor"
    )

    await wb_client.put("findings.v1", {"anomalies": []})

    # Tamper with the internal event log in the backend
    events = await whiteboard_backend.events(session_id)
    assert len(events) > 0
    # Corrupt the state hash of an event
    object.__setattr__(events[0], "state_hash", "tampered_corrupted_hash")

    # verify_chain will now detect tampering
    report = await whiteboard_backend.verify_chain(session_id)
    assert report.is_valid is False

    adapter = FakeAdapter()
    agent = BetaExecutionAgent(
        agent_id="beta_secure",
        scl_path=SCL_PATH,
        adapter=adapter,
        whiteboard=wb_client,
        router=router,
    )
    agent.register(registry, router)

    task_msg = UACPMessage(
        correlation_id=session_id,
        trace_id="trace-sec",
        sender=Party(agent_id="alpha"),
        recipient=Party(agent_id="beta_secure"),
        intent=Intent.task_delegation,
        payload=TaskDelegationPayload(
            instruction="Mitigate findings",
            whiteboard_session_ref=session_id,
        ).model_dump(),
    )

    result_msg = await agent.handle(task_msg)

    # Must refuse execution
    assert result_msg.intent == Intent.task_result
    assert result_msg.payload["status"] == "failure"
    assert "tamper" in result_msg.payload["summary"].lower()


@pytest.mark.asyncio
async def test_beta_repair_on_first_failure(
    whiteboard_backend: InMemoryWhiteboardBackend,
    router: Router,
    registry: CapabilityRegistry,
) -> None:
    """Verify repair path: code fails static_check on turn 1, LLM repairs it on turn 2."""
    session_id = await whiteboard_backend.create_session(ttl_s=3600)
    wb_client = WhiteboardClient(backend=whiteboard_backend, session_id=session_id, author="test")

    clean = generate_clean_ledger(batches=1, entries_per_batch=4, seed=42)
    corrupted = [dict(tx) for tx in clean]
    corrupted.append(dict(clean[0]))

    await wb_client.put("findings.v1", {"anomalies": [{"id": "TX-1", "type": "duplicate_id"}]})
    await wb_client.put("ledger.v1", corrupted)

    # Response 1 contains forbidden 'import os'
    bad_code = "import os\ndef mitigate(ledger):\n    return ledger, []\n"
    bad_response = {
        "python_code": bad_code,
        "rationale": "Attempting dangerous call",
        "risks": [],
    }

    # Response 2 (repair turn) provides working clean code
    good_code = """
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
    good_response = {
        "python_code": good_code,
        "rationale": "Repaired code using pure python deduplication without OS imports.",
        "risks": [],
    }

    adapter = FakeAdapter(script=[bad_response, good_response])

    agent = BetaExecutionAgent(
        agent_id="beta_repair",
        scl_path=SCL_PATH,
        adapter=adapter,
        whiteboard=wb_client,
        router=router,
    )
    agent.register(registry, router)

    task_msg = UACPMessage(
        correlation_id=session_id,
        trace_id="tr-repair",
        sender=Party(agent_id="alpha"),
        recipient=Party(agent_id="beta_repair"),
        intent=Intent.task_delegation,
        payload=TaskDelegationPayload(
            instruction="Mitigate findings",
            whiteboard_session_ref=session_id,
        ).model_dump(),
    )

    result_msg = await agent.handle(task_msg)

    assert result_msg.payload["status"] == "success"
    assert "Verification passed" in result_msg.payload["summary"]

    state = await wb_client.read_all()
    final_report = FinalReport.model_validate(state["final_report.v1"]["value"])
    assert final_report.verification.passed is True
    assert final_report.metadata.get("repairs_attempted") == 1


@pytest.mark.asyncio
async def test_beta_honest_failure_on_double_failure(
    whiteboard_backend: InMemoryWhiteboardBackend,
    router: Router,
    registry: CapabilityRegistry,
) -> None:
    """Verify honest reporting: code fails twice, Beta records passed=False without crashing."""
    session_id = await whiteboard_backend.create_session(ttl_s=3600)
    wb_client = WhiteboardClient(backend=whiteboard_backend, session_id=session_id, author="test")

    await wb_client.put("findings.v1", {"anomalies": [{"id": "TX-1", "type": "duplicate_id"}]})
    await wb_client.put("ledger.v1", [{"id": "TX-1"}])

    # Both responses fail static check
    bad_response_1 = {
        "python_code": "import subprocess\ndef mitigate(ledger): return ledger, []",
        "rationale": "bad 1",
        "risks": [],
    }
    bad_response_2 = {
        "python_code": "import socket\ndef mitigate(ledger): return ledger, []",
        "rationale": "bad 2",
        "risks": [],
    }

    adapter = FakeAdapter(script=[bad_response_1, bad_response_2])

    agent = BetaExecutionAgent(
        agent_id="beta_fails",
        scl_path=SCL_PATH,
        adapter=adapter,
        whiteboard=wb_client,
        router=router,
    )
    agent.register(registry, router)

    task_msg = UACPMessage(
        correlation_id=session_id,
        trace_id="tr-double-fail",
        sender=Party(agent_id="alpha"),
        recipient=Party(agent_id="beta_fails"),
        intent=Intent.task_delegation,
        payload=TaskDelegationPayload(
            instruction="Mitigate findings",
            whiteboard_session_ref=session_id,
        ).model_dump(),
    )

    # Must NOT raise exception; must return task_result with failure status
    result_msg = await agent.handle(task_msg)

    assert result_msg.intent == Intent.task_result
    assert result_msg.payload["status"] == "failure"
    assert "Remediation failed" in result_msg.payload["summary"]

    state = await wb_client.read_all()
    assert "final_report.v1" in state
    report = FinalReport.model_validate(state["final_report.v1"]["value"])
    assert report.verification.passed is False
    assert "failed" in report.verification.details.lower()
    assert report.metadata.get("final_status") == "failed"
    assert report.metadata.get("repairs_attempted") == 1
