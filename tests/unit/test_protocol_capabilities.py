"""Unit tests for UACP capability discovery, registry, ranking, and task splitting."""

from pathlib import Path

import pytest
from pydantic import BaseModel, ValidationError

from uacp.protocol.capabilities import (
    AgentCard,
    Capability,
    CapabilityRegistry,
    Subtask,
    build_agent_card,
    capability,
)
from uacp.protocol.jsonrpc import (
    JSONRPCError,
    JSONRPCErrorCode,
    JSONRPCRequest,
)
from uacp.protocol.messages import (
    CapabilityQueryPayload,
    CircuitBreakerState,
    Intent,
    Party,
    UACPMessage,
)


class InputSchema(BaseModel):
    query: str
    limit: int = 10


class OutputSchema(BaseModel):
    results: list[str]


# ── Model Tests ─────────────────────────────────────────────────────────────


def test_capability_model_defaults_and_validation() -> None:
    cap = Capability(name="text_summarisation")
    assert cap.name == "text_summarisation"
    assert cap.complexity_level == 1
    assert cap.strengths == []
    assert cap.cost_hint == 0.0
    assert cap.avg_latency_ms == 0.0
    assert cap.max_input_chars is None

    # Complexity must be between 1 and 5
    with pytest.raises(ValidationError):
        Capability(name="invalid", complexity_level=6)

    with pytest.raises(ValidationError):
        Capability(name="invalid", complexity_level=0)


def test_agent_card_model_and_methods() -> None:
    cap1 = Capability(name="analysis", complexity_level=3, strengths=["data"])
    cap2 = Capability(name="codegen", complexity_level=4, strengths=["python"])
    card = AgentCard(
        agent_id="agent_alpha",
        engine="alpha:gemini",
        role="Senior Analyst",
        current_load=1,
        capabilities=[cap1, cap2],
    )

    assert card.has_capability("analysis") is True
    assert card.has_capability("unknown") is False
    assert card.get_capability("analysis") == cap1
    assert card.get_capability("unknown") is None

    # Load update
    card.current_load = 3
    assert card.current_load == 3


# ── Registry Registration and Heartbeat Tests ───────────────────────────────


def test_registry_registration_and_deregistration() -> None:
    registry = CapabilityRegistry(default_ttl=30.0)
    card = AgentCard(
        agent_id="agent_alpha",
        engine="alpha:gemini",
        capabilities=[Capability(name="audit")],
    )

    registry.register(card)
    assert registry.get("agent_id_unknown") is None
    retrieved = registry.get("agent_alpha")
    assert retrieved is not None
    assert retrieved.agent_id == "agent_alpha"

    assert registry.deregister("agent_alpha") is True
    assert registry.deregister("agent_alpha") is False
    assert registry.get("agent_alpha") is None


def test_registry_heartbeat_and_expiry() -> None:
    now = 1000.0

    def clock() -> float:
        return now

    registry = CapabilityRegistry(default_ttl=10.0, clock=clock)
    card = AgentCard(
        agent_id="agent_beta",
        engine="beta:groq",
        capabilities=[Capability(name="compile", complexity_level=2)],
    )

    registry.register(card, ttl_seconds=10.0)
    assert registry.is_alive("agent_beta") is True

    # Advance time within TTL
    now = 1008.0
    assert registry.is_alive("agent_beta") is True
    assert len(registry.find("compile")) == 1

    # Heartbeat renews TTL
    assert registry.heartbeat("agent_beta") is True
    assert registry.heartbeat("unknown_agent") is False

    # Advance time past original TTL, but within renewed TTL
    now = 1016.0
    assert registry.is_alive("agent_beta") is True

    # Advance time past renewed TTL -> expired
    now = 1028.0
    assert registry.is_alive("agent_beta") is False
    assert registry.get("agent_beta") is None
    # Expired agent excluded from find
    assert len(registry.find("compile")) == 0


# ── Registry Ranking and Constraints Tests ──────────────────────────────────


def test_registry_ranking_criteria() -> None:
    registry = CapabilityRegistry()

    # Agent 1: 1 strength match, load 2, cost 0.05
    c1 = AgentCard(
        agent_id="agent_1",
        engine="engine_x",
        current_load=2,
        capabilities=[
            Capability(
                name="code_review",
                strengths=["security"],
                cost_hint=0.05,
                complexity_level=3,
            )
        ],
    )
    # Agent 2: 2 strengths match, load 2, cost 0.10
    c2 = AgentCard(
        agent_id="agent_2",
        engine="engine_x",
        current_load=2,
        capabilities=[
            Capability(
                name="code_review",
                strengths=["security", "ast_analysis"],
                cost_hint=0.10,
                complexity_level=3,
            )
        ],
    )
    # Agent 3: 2 strengths match, load 0, cost 0.10
    c3 = AgentCard(
        agent_id="agent_3",
        engine="engine_x",
        current_load=0,
        capabilities=[
            Capability(
                name="code_review",
                strengths=["security", "ast_analysis"],
                cost_hint=0.10,
                complexity_level=3,
            )
        ],
    )

    registry.register(c1)
    registry.register(c2)
    registry.register(c3)

    # Ranking by strengths first, then lowest current_load, then cost_hint
    results = registry.find(
        "code_review",
        required_strengths=["security", "ast_analysis"],
    )

    # Agent 3 and Agent 2 match 2 strengths. Agent 3 has load 0 vs Agent 2 load 2.
    # Agent 1 matches only 1 strength.
    assert [r.agent_id for r in results] == ["agent_3", "agent_2", "agent_1"]
    assert "strengths_matched=2/2" in results[0].reason
    assert "load=0" in results[0].reason


def test_registry_complexity_level_filtering() -> None:
    registry = CapabilityRegistry()
    junior = AgentCard(
        agent_id="junior",
        engine="engine_small",
        capabilities=[Capability(name="solve", complexity_level=2)],
    )
    senior = AgentCard(
        agent_id="senior",
        engine="engine_large",
        capabilities=[Capability(name="solve", complexity_level=5)],
    )
    registry.register(junior)
    registry.register(senior)

    # Subtask requires complexity level 4
    results = registry.find("solve", required_complexity=4)
    assert len(results) == 1
    assert results[0].agent_id == "senior"
    assert results[0].card.agent_id == "senior"


# ── Subtask Splitting Tests ─────────────────────────────────────────────────


def test_suggest_split_scenarios() -> None:
    now = 1000.0

    def clock() -> float:
        return now

    registry = CapabilityRegistry(default_ttl=20.0, clock=clock)

    # Agent Alpha: High complexity, strengths: "deep_reasoning", "code_generation"
    alpha = AgentCard(
        agent_id="agent_alpha",
        engine="gemini",
        current_load=0,
        capabilities=[
            Capability(
                name="task_solver",
                complexity_level=5,
                strengths=["deep_reasoning", "code_generation"],
                cost_hint=0.08,
            )
        ],
    )
    # Agent Beta: Low complexity, strengths: "formatting"
    beta = AgentCard(
        agent_id="agent_beta",
        engine="groq",
        current_load=0,
        capabilities=[
            Capability(
                name="task_solver",
                complexity_level=2,
                strengths=["formatting"],
                cost_hint=0.01,
            )
        ],
    )
    # Agent Gamma: Same qualifications as Beta but higher current load
    gamma = AgentCard(
        agent_id="agent_gamma",
        engine="groq",
        current_load=2,
        capabilities=[
            Capability(
                name="task_solver",
                complexity_level=2,
                strengths=["formatting"],
                cost_hint=0.01,
            )
        ],
    )

    registry.register(alpha)
    registry.register(beta)
    registry.register(gamma)

    subtasks = [
        # Subtask 1: Only Alpha can take (required complexity 4)
        Subtask(
            id="st-1",
            capability="task_solver",
            required_complexity=4,
            required_strengths=["code_generation"],
        ),
        # Subtask 2: Beta and Gamma qualify (complexity 2, formatting),
        # tie broken by lower load (Beta load 0 < Gamma load 2)
        Subtask(
            id="st-2",
            capability="task_solver",
            required_complexity=2,
            required_strengths=["formatting"],
        ),
        # Subtask 3: Nobody can handle (required complexity 6 > 5)
        Subtask(
            id="st-3",
            capability="task_solver",
            required_complexity=6,
        ),
        # Subtask 4: Nonexistent capability
        Subtask(
            id="st-4",
            capability="quantum_computing",
            required_complexity=1,
        ),
    ]

    assignments = registry.suggest_split(subtasks)
    assert len(assignments) == 4

    # Subtask 1 -> alpha
    assert assignments[0].subtask_id == "st-1"
    assert assignments[0].assigned is True
    assert assignments[0].assigned_agent_id == "agent_alpha"

    # Subtask 2 -> beta (tie broken by lower load: beta=0 vs gamma=2)
    assert assignments[1].subtask_id == "st-2"
    assert assignments[1].assigned is True
    assert assignments[1].assigned_agent_id == "agent_beta"

    # Subtask 3 -> unassigned (exceeds max complexity)
    assert assignments[2].subtask_id == "st-3"
    assert assignments[2].assigned is False
    assert "exceeds max available level 5" in assignments[2].reason

    # Subtask 4 -> unassigned (unknown capability)
    assert assignments[3].subtask_id == "st-4"
    assert assignments[3].assigned is False
    assert "No agent registered with capability 'quantum_computing'" in assignments[3].reason

    # Test expired agent is excluded:
    # Expire Agent Alpha by advancing clock past TTL
    now = 1030.0
    alpha_subtask = [Subtask(id="st-alpha-only", capability="task_solver", required_complexity=4)]
    expired_assignments = registry.suggest_split(alpha_subtask)
    assert expired_assignments[0].assigned is False
    assert (
        "exceeds max available level 2" in expired_assignments[0].reason
        or "expired" in expired_assignments[0].reason
    )


# ── Decorator and Card Generation Tests ─────────────────────────────────────


class SampleAgent:
    """Sample test agent class."""

    agent_id = "sample_agent"
    engine = "gemini-flash"

    @capability(
        name="calculate",
        input=InputSchema,
        output=OutputSchema,
        description="Calculate metrics",
        complexity_level=3,
        strengths=["math", "finance"],
        cost_hint=0.02,
    )
    def calculate(self, query: str) -> list[str]:
        return [query]


def test_decorator_and_build_agent_card() -> None:
    card = build_agent_card(SampleAgent)
    assert card.agent_id == "sample_agent"
    assert card.engine == "gemini-flash"
    assert len(card.capabilities) == 1

    cap = card.capabilities[0]
    assert cap.name == "calculate"
    assert cap.complexity_level == 3
    assert cap.strengths == ["math", "finance"]
    assert cap.input_schema is not None
    assert "query" in cap.input_schema["properties"]
    assert cap.output_schema is not None
    assert "results" in cap.output_schema["properties"]


def test_build_agent_card_from_scl(tmp_path: Path) -> None:
    scl_content = """<identity profile="lead_auditor">
Financial ledger auditor specializing in discrepancy identification.
</identity>

<variable name="ledger_id" source="batch.id">{{ledger_id}}</variable>

<output format="json_schema">
{
  "type": "object",
  "properties": {
    "discrepancies": { "type": "array", "items": { "type": "string" } }
  },
  "required": ["discrepancies"]
}
</output>
"""
    scl_file = tmp_path / "auditor.scl"
    scl_file.write_text(scl_content, encoding="utf-8")

    class SclAgent:
        pass

    card = build_agent_card(SclAgent, scl_path=scl_file, engine="gemini-2.5-flash")
    assert card.agent_id == "lead_auditor"
    assert card.engine == "gemini-2.5-flash"
    assert "auditor specializing" in card.role
    assert len(card.capabilities) == 1
    assert card.capabilities[0].name == "lead_auditor"
    assert card.capabilities[0].input_schema is not None
    assert "ledger_id" in card.capabilities[0].input_schema["properties"]
    assert card.capabilities[0].output_schema is not None
    assert "discrepancies" in card.capabilities[0].output_schema["properties"]


# ── Capability Query and Response JSON-RPC Tests ────────────────────────────


def test_handle_query_and_response() -> None:
    registry = CapabilityRegistry()
    card = AgentCard(
        agent_id="solver_agent",
        engine="gemini",
        capabilities=[Capability(name="translate", description="Translate languages")],
    )
    registry.register(card)

    # Query existing capability
    query_msg = UACPMessage(
        correlation_id="corr-1",
        trace_id="tr-1",
        sender=Party(agent_id="requester", engine="groq"),
        recipient=Party(agent_id="capability_registry", engine="uacp:system"),
        intent=Intent.capability_query,
        payload=CapabilityQueryPayload(query=["translate"]).model_dump(),
        circuit_breaker=CircuitBreakerState(max_hops=5, current_hop=0),
    )

    resp_msg = registry.handle_query(query_msg)
    assert resp_msg.intent == Intent.capability_response
    assert resp_msg.recipient.agent_id == "requester"
    assert "translate" in resp_msg.payload["capabilities"]

    # Ingest response via handle_response
    ingested = registry.handle_response(resp_msg)
    assert len(ingested) == 1
    assert ingested[0].agent_id == "solver_agent"


def test_handle_query_unknown_capability_raises_32002() -> None:
    registry = CapabilityRegistry()
    query_msg = UACPMessage(
        correlation_id="corr-1",
        trace_id="tr-1",
        sender=Party(agent_id="requester", engine="groq"),
        recipient=Party(agent_id="capability_registry", engine="uacp:system"),
        intent=Intent.capability_query,
        payload=CapabilityQueryPayload(query=["nonexistent_capability"]).model_dump(),
    )

    with pytest.raises(JSONRPCError) as exc_info:
        registry.handle_query(query_msg)

    assert exc_info.value.code == JSONRPCErrorCode.CAPABILITY_NOT_FOUND
    assert exc_info.value.code == -32002
    assert "nonexistent_capability" in exc_info.value.message


def test_handle_jsonrpc_request_unknown_capability_returns_32002_response() -> None:
    registry = CapabilityRegistry()
    query_msg = UACPMessage(
        correlation_id="corr-1",
        trace_id="tr-1",
        sender=Party(agent_id="requester", engine="groq"),
        recipient=Party(agent_id="capability_registry", engine="uacp:system"),
        intent=Intent.capability_query,
        payload=CapabilityQueryPayload(query=["quantum_solve"]).model_dump(),
    )
    request = JSONRPCRequest(
        id="req-cap-1",
        method="uacp.capability_query",
        params=query_msg,
    )

    response = registry.handle_request(request)
    assert response.id == "req-cap-1"
    assert response.result is None
    assert response.error is not None
    assert response.error.code == -32002
    assert "CapabilityNotFound" in response.error.message
