"""Unit tests for frontend helper functions (pure functions, no Streamlit runtime needed)."""

from __future__ import annotations

import pytest
from frontend.helpers import (
    compile_for_providers,
    detect_environment_mode,
    evaluate_circuit_breaker_status,
    format_metrics_summary,
    format_trace_timeline,
    format_whiteboard_state,
    parse_and_validate_scl,
)

SAMPLE_VALID_SCL = """<scl version="1">
  <identity profile="test_assistant">
    You are an auditor assistant.
  </identity>
  <variable name="manifest" source="data.records" type="json">
    Transactions:
    {{manifest}}
  </variable>
  <example pair="true">
    <input>ID: 100</input>
    <output>{"valid": true}</output>
  </example>
  <output format="json_schema">
    {
      "type": "object",
      "required": ["valid"],
      "properties": {"valid": {"type": "boolean"}}
    }
  </output>
</scl>"""


def test_detect_environment_mode(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify live vs replay mode detection without leaking credentials."""
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    assert detect_environment_mode() == "replay"

    monkeypatch.setenv("GEMINI_API_KEY", "dummy_key")
    assert detect_environment_mode() == "live"


def test_parse_and_validate_scl_valid() -> None:
    """Verify parsing and validation of a valid SCL string."""
    res = parse_and_validate_scl(SAMPLE_VALID_SCL)
    assert res["valid"] is True
    assert res["ast"] is not None
    assert len(res["diagnostics"]) == 0
    assert res["ast"]["identity"]["profile"] == "test_assistant"


def test_parse_and_validate_scl_invalid() -> None:
    """Verify diagnostics extraction for invalid SCL strings."""
    # Empty string
    empty_res = parse_and_validate_scl("")
    assert empty_res["valid"] is False
    assert len(empty_res["diagnostics"]) > 0

    # Broken XML / SCL
    broken_res = parse_and_validate_scl("<scl><identity>Unclosed tag")
    assert broken_res["valid"] is False
    assert len(broken_res["diagnostics"]) > 0
    assert broken_res["diagnostics"][0]["line"] >= 1


def test_compile_for_providers() -> None:
    """Verify multi-provider compilation side-by-side."""
    runtime_data = {"records": [{"id": 101, "amount": 500}]}
    compiled = compile_for_providers(SAMPLE_VALID_SCL, runtime_data)

    assert compiled["error"] is None
    assert compiled["gemini"] is not None
    assert "system_instruction" in compiled["gemini"]
    assert len(compiled["gemini"]["contents"]) >= 3

    assert compiled["openai_compat"] is not None
    assert "messages" in compiled["openai_compat"]
    assert any(m["role"] == "system" for m in compiled["openai_compat"]["messages"])

    assert compiled["ollama"] is not None
    assert compiled["ollama"]["format"] == "json"


def test_format_trace_timeline() -> None:
    """Verify transformation of trace events into timeline rows."""
    events = [
        {
            "event_type": "message.delivered",
            "data": {
                "hop": 1,
                "sender": "alpha",
                "recipient": "beta",
                "intent": "task_delegation",
            },
        },
        {
            "event_type": "message.delivered",
            "data": {
                "hop": 2,
                "sender": "beta",
                "recipient": "alpha",
                "intent": "task_result",
            },
        },
    ]

    timeline = format_trace_timeline(events)
    assert len(timeline) == 2
    assert timeline[0]["hop"] == 1
    assert timeline[0]["sender"] == "alpha"
    assert timeline[0]["recipient"] == "beta"
    assert timeline[1]["hop"] == 2
    assert timeline[1]["intent"] == "task_result"


def test_format_whiteboard_state() -> None:
    """Verify whiteboard state formatting with version and author metadata."""
    state = {
        "findings.v1": {
            "version": 2,
            "author": "agent_alpha",
            "updated_at": "2026-10-22T00:00:00Z",
            "value": [{"id": "TX-1", "issue": "duplicate"}],
        }
    }

    formatted = format_whiteboard_state(state)
    assert "findings.v1" in formatted
    assert formatted["findings.v1"]["version"] == 2
    assert formatted["findings.v1"]["author"] == "agent_alpha"
    assert len(formatted["findings.v1"]["value"]) == 1


def test_evaluate_circuit_breaker_status() -> None:
    """Verify circuit breaker evaluation for healthy and tripped sessions."""
    # Healthy
    healthy_events = [
        {"event_type": "message.delivered", "data": {"hop": 2}},
    ]
    st1 = evaluate_circuit_breaker_status(healthy_events, max_hops=5)
    assert st1["tripped"] is False
    assert st1["current_hop"] == 2
    assert st1["color"] == "green"

    # Tripped via session.halted
    tripped_events = [
        {"event_type": "message.delivered", "data": {"hop": 5}},
        {"event_type": "session.halted", "data": {"reason": "HopLimitExceeded at hop 5"}},
    ]
    st2 = evaluate_circuit_breaker_status(tripped_events, max_hops=5)
    assert st2["tripped"] is True
    assert st2["color"] == "red"
    assert "HopLimitExceeded" in str(st2["halt_reason"])


def test_format_metrics_summary() -> None:
    """Verify metrics summary normalization."""
    metrics_dict = {
        "alpha_precision": 1.0,
        "alpha_recall": 0.85,
        "anomalies_mitigated_fraction": 1.0,
        "total_repairs": 0,
        "total_hops": 3,
        "wall_time_seconds": 4.5,
        "passed_schema": True,
        "final_report_status": "success",
    }

    summary = format_metrics_summary(metrics_dict)
    assert summary["alpha_precision"] == 1.0
    assert summary["alpha_recall"] == 0.85
    assert summary["anomalies_mitigated_fraction"] == 1.0
    assert summary["passed_schema"] is True
