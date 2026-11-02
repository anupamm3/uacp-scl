"""Unit tests for Agent base class, lifecycle, SCL run_llm, and capability routing."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from uacp.adapters.fake_adapter import FakeAdapter
from uacp.agents.base import Agent
from uacp.memory.whiteboard import InMemoryWhiteboardBackend
from uacp.memory.whiteboard.client import WhiteboardClient
from uacp.protocol.capabilities import Capability, CapabilityRegistry
from uacp.protocol.messages import (
    Intent,
    Party,
    TaskDelegationPayload,
    UACPMessage,
)
from uacp.protocol.router import Router
from uacp.protocol.transport.inproc import InProcessTransport

TEST_SCL = """<scl version="1">
  <identity profile="test_assistant" tone="neutral">
    You are a test assistant.
  </identity>
  <variable name="input_text" source="data.query" type="text">
    Query: {{input_text}}
  </variable>
  <output format="json_schema" strict="true">
    {
      "$schema": "http://json-schema.org/draft-07/schema#",
      "type": "object",
      "required": ["status", "answer"],
      "properties": {
        "status": { "type": "string" },
        "answer": { "type": "string" }
      }
    }
  </output>
</scl>"""


@pytest.fixture
def scl_file(tmp_path: Path) -> Path:
    """Fixture providing a temporary valid SCL contract."""
    file = tmp_path / "test_assistant.scl"
    file.write_text(TEST_SCL, encoding="utf-8")
    return file


@pytest.mark.asyncio
async def test_agent_run_llm_with_fake_adapter(scl_file: Path) -> None:
    """Verify Agent.run_llm parses SCL, binds data, invokes FakeAdapter, and logs to whiteboard."""
    backend = InMemoryWhiteboardBackend()
    session_id = await backend.create_session(ttl_s=300.0)
    wb_client = WhiteboardClient(backend=backend, session_id=session_id, author="test_agent")

    expected_output = {"status": "ok", "answer": "Synthesized test answer."}
    adapter = FakeAdapter(model_id="fake-v1", script=[json.dumps(expected_output)])

    agent = Agent(
        agent_id="test_agent",
        scl_path=scl_file,
        adapter=adapter,
        whiteboard=wb_client,
    )

    parsed, usage = await agent.run_llm(
        user_message="What is the test answer?",
        runtime_data={"data": {"query": "sample query"}},
        session_id=session_id,
        intent="test_intent",
    )

    assert parsed == expected_output
    assert usage.requests == 1
    assert usage.input_tokens > 0
    assert usage.output_tokens > 0

    # Verify event logged to Whiteboard
    events = await backend.events(session_id)
    llm_events = [e for e in events if e.event_type == "agent.llm_call"]
    assert len(llm_events) == 1
    data = llm_events[0].data
    assert data["agent_id"] == "test_agent"
    assert data["intent"] == "test_intent"
    assert data["repairs"] == 0
    assert data["tokens"]["input_tokens"] == usage.input_tokens


@pytest.mark.asyncio
async def test_agent_registration_and_load_updates(scl_file: Path) -> None:
    """Verify Agent.register builds card, registers with transport, and updates load."""
    backend = InMemoryWhiteboardBackend()
    transport = InProcessTransport()
    registry = CapabilityRegistry()
    router = Router(whiteboard=backend, transport=transport, registry=registry)

    abilities = [
        Capability(
            name="summarization",
            strengths=["speed", "brevity"],
            complexity_level=2,
            cost_hint=0.01,
        )
    ]

    agent = Agent(
        agent_id="summary_bot",
        scl_path=scl_file,
        capabilities=abilities,
        current_load=0,
    )

    agent.register(registry, router)

    # Verify registration in CapabilityRegistry and transport
    assert registry.is_alive("summary_bot")
    assert transport.has_agent("summary_bot")
    assert agent.card is not None
    assert agent.card.agent_id == "summary_bot"
    assert agent.card.role == "You are a test assistant."
    assert agent.card.has_capability("summarization")

    # Update load dynamically
    agent.update_load(3)
    assert agent.current_load == 3
    card_in_reg = registry.get("summary_bot")
    assert card_in_reg is not None
    assert card_in_reg.current_load == 3

    # Send heartbeat
    assert agent.heartbeat() is True


@pytest.mark.asyncio
async def test_agent_handle_message_task_delegation(scl_file: Path) -> None:
    """Verify Agent.handle dispatches task_delegation, runs LLM, and records trace."""
    backend = InMemoryWhiteboardBackend()
    session_id = await backend.create_session(ttl_s=300.0)
    wb_client = WhiteboardClient(backend=backend, session_id=session_id, author="test_agent")

    expected_output = {"status": "success", "answer": "Delegated answer."}
    adapter = FakeAdapter(model_id="fake-v1", script=[json.dumps(expected_output)])

    agent = Agent(
        agent_id="worker_agent",
        scl_path=scl_file,
        adapter=adapter,
        whiteboard=wb_client,
    )

    msg = UACPMessage(
        correlation_id="corr-101",
        trace_id="trace-101",
        sender=Party(agent_id="master_agent"),
        recipient=Party(agent_id="worker_agent"),
        intent=Intent.task_delegation,
        payload=TaskDelegationPayload(
            instruction="Execute task",
            whiteboard_session_ref=session_id,
        ).model_dump()
        | {"data": {"query": "payload data"}},
    )

    response_msg = await agent.handle(msg)

    assert response_msg.intent == Intent.task_result
    assert response_msg.sender.agent_id == "worker_agent"
    assert response_msg.recipient.agent_id == "master_agent"
    assert response_msg.payload["status"] == "success"
    assert response_msg.payload["result"] == expected_output

    # Check whiteboard events
    events = await backend.events(session_id)
    handled_events = [e for e in events if e.event_type == "agent.message_handled"]
    assert len(handled_events) == 1
    assert handled_events[0].data["sender"] == "master_agent"
    assert handled_events[0].data["intent"] == Intent.task_delegation.value


@pytest.mark.asyncio
async def test_two_agents_capability_routing(scl_file: Path) -> None:
    """Verify router routes by capability name to the matching agent among two agents."""
    backend = InMemoryWhiteboardBackend()
    session_id = await backend.create_session(ttl_s=300.0)
    transport = InProcessTransport()
    registry = CapabilityRegistry()
    router = Router(whiteboard=backend, transport=transport, registry=registry)

    # Agent Alpha: fast lightweight analysis
    alpha_cap = Capability(
        name="quick_analysis",
        description="Quick high-level analysis",
        strengths=["fast", "concise"],
        complexity_level=1,
    )
    agent_alpha = Agent(
        agent_id="agent_alpha",
        scl_path=scl_file,
        adapter=FakeAdapter(
            model_id="fake-a",
            script=[json.dumps({"status": "ok", "answer": "Alpha result"})],
        ),
        capabilities=[alpha_cap],
    )
    agent_alpha.register(registry, router)

    # Agent Beta: forensic deep audit
    beta_cap = Capability(
        name="forensic_audit",
        description="In-depth forensic audit and fraud analysis",
        strengths=["deep_audit", "fraud_detection"],
        complexity_level=4,
    )
    agent_beta = Agent(
        agent_id="agent_beta",
        scl_path=scl_file,
        adapter=FakeAdapter(
            model_id="fake-b",
            script=[json.dumps({"status": "ok", "answer": "Beta result"})],
        ),
        capabilities=[beta_cap],
    )
    agent_beta.register(registry, router)

    # Route message requiring forensic_audit capability
    msg_for_beta = UACPMessage(
        correlation_id="corr-audit-01",
        trace_id="trace-01",
        sender=Party(agent_id="coordinator"),
        recipient=Party(agent_id="", capability="forensic_audit"),
        intent=Intent.task_delegation,
        payload=TaskDelegationPayload(
            instruction="Audit transactions for fraud",
            whiteboard_session_ref=session_id,
        ).model_dump()
        | {"data": {"query": "audit data"}},
    )

    res_beta = await router.send(msg_for_beta)
    assert res_beta.error is None
    assert res_beta.result is not None
    # Verify delivery to agent_beta
    assert res_beta.result["sender"]["agent_id"] == "agent_beta"
    assert res_beta.result["payload"]["result"]["answer"] == "Beta result"

    # Route message requiring quick_analysis capability in its own session
    session_id_alpha = await backend.create_session(ttl_s=300.0)
    msg_for_alpha = UACPMessage(
        correlation_id="corr-analysis-01",
        trace_id="trace-02",
        sender=Party(agent_id="coordinator"),
        recipient=Party(agent_id="", capability="quick_analysis"),
        intent=Intent.task_delegation,
        payload=TaskDelegationPayload(
            instruction="Summarize ledger overview",
            whiteboard_session_ref=session_id_alpha,
        ).model_dump()
        | {"data": {"query": "quick query"}},
    )

    res_alpha = await router.send(msg_for_alpha)
    assert res_alpha.error is None
    assert res_alpha.result is not None
    # Verify delivery to agent_alpha
    assert res_alpha.result["sender"]["agent_id"] == "agent_alpha"
    assert res_alpha.result["payload"]["result"]["answer"] == "Alpha result"
