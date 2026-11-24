"""Pure helper functions for the Streamlit UI, formatting, SCL parsing, and metrics.

All logic is testable without a running Streamlit server. Never exposes API keys.
"""

from __future__ import annotations

import os
from typing import Any

from uacp.scl.context import RuntimeContext, bind
from uacp.scl.parser import parse
from uacp.scl.validate import validate_strict


def detect_environment_mode() -> str:
    """Detect whether live provider credentials are configured.

    Returns:
        'live' if GEMINI_API_KEY, GROQ_API_KEY, or OPENAI_API_KEY is non-empty,
        otherwise 'replay'. Never prints or logs the actual keys.
    """
    live_keys = ["GEMINI_API_KEY", "GROQ_API_KEY", "OPENAI_API_KEY"]
    for key_name in live_keys:
        val = os.environ.get(key_name, "").strip()
        if val:
            return "live"
    return "replay"


def parse_and_validate_scl(scl_content: str) -> dict[str, Any]:
    """Parse and strictly validate SCL contract source.

    Returns:
        Dictionary containing 'valid' bool, 'ast' dict or None, and 'diagnostics' list.
    """
    diagnostics: list[dict[str, Any]] = []

    if not scl_content.strip():
        return {
            "valid": False,
            "ast": None,
            "diagnostics": [{"line": 1, "message": "SCL document is empty"}],
        }

    try:
        doc = parse(scl_content)
        validate_strict(doc)
        return {
            "valid": True,
            "ast": doc.model_dump(mode="json"),
            "diagnostics": [],
        }
    except Exception as exc:
        line_num = getattr(exc, "line", 1) or 1
        diagnostics.append({"line": line_num, "message": str(exc)})
        return {
            "valid": False,
            "ast": None,
            "diagnostics": diagnostics,
        }


def compile_for_providers(
    scl_content: str,
    runtime_data: dict[str, Any],
) -> dict[str, Any]:
    """Bind runtime data into SCL AST and compile requests for multiple providers.

    Produces compiled representations for Gemini, OpenAI-compatible, and Ollama.
    """
    res: dict[str, Any] = {
        "gemini": None,
        "openai_compat": None,
        "ollama": None,
        "error": None,
    }

    try:
        doc = parse(scl_content)
        validate_strict(doc)

        data_dict = dict(runtime_data)
        if "data" not in data_dict:
            data_dict["data"] = dict(data_dict)
        if "runtime" not in data_dict:
            data_dict["runtime"] = {"payload": dict(data_dict)}

        ctx = RuntimeContext(data_dict)
        bound_doc = bind(doc, ctx)
        sub_doc = bound_doc.to_document()

        # 1. Gemini compiled representation
        system_inst = sub_doc.identity.body.strip() if sub_doc.identity else ""
        gemini_contents: list[dict[str, Any]] = []
        for ex in sub_doc.examples:
            gemini_contents.append({"role": "user", "parts": [ex.input.strip()]})
            gemini_contents.append({"role": "model", "parts": [ex.output.strip()]})

        user_content_parts = []
        for v in sub_doc.variables:
            user_content_parts.append(f"{v.name}:\n{v.template or ''}")
        gemini_contents.append({"role": "user", "parts": user_content_parts})

        res["gemini"] = {
            "system_instruction": system_inst,
            "contents": gemini_contents,
            "output_schema": sub_doc.output.schema if sub_doc.output else None,
        }

        # 2. OpenAI-compatible compiled representation
        openai_messages: list[dict[str, str]] = []
        if system_inst:
            openai_messages.append({"role": "system", "content": system_inst})
        for ex in sub_doc.examples:
            openai_messages.append({"role": "user", "content": ex.input.strip()})
            openai_messages.append({"role": "assistant", "content": ex.output.strip()})
        openai_messages.append({"role": "user", "content": "\n\n".join(user_content_parts)})

        res["openai_compat"] = {
            "messages": openai_messages,
            "response_format": (
                {"type": "json_object"}
                if sub_doc.output and sub_doc.output.format == "json_schema"
                else None
            ),
        }

        # 3. Ollama compiled representation
        has_schema = sub_doc.output and sub_doc.output.format == "json_schema"
        res["ollama"] = {
            "model": "llama3:latest",
            "messages": openai_messages,
            "stream": False,
            "format": "json" if has_schema else "",
        }

    except Exception as exc:
        res["error"] = str(exc)

    return res


def format_trace_timeline(events: list[dict[str, Any]] | list[Any]) -> list[dict[str, Any]]:
    """Format raw Whiteboard or Router events into a structured message timeline."""
    rows: list[dict[str, Any]] = []

    for ev in events:
        data = (
            ev.data
            if hasattr(ev, "data")
            else (ev.get("data", {}) if isinstance(ev, dict) else {})
        )
        event_type = (
            ev.event_type
            if hasattr(ev, "event_type")
            else (ev.get("event_type", "") if isinstance(ev, dict) else "")
        )

        if event_type == "message.delivered":
            rows.append(
                {
                    "hop": data.get("hop", len(rows) + 1),
                    "sender": data.get("sender", "unknown"),
                    "recipient": data.get("recipient", "unknown"),
                    "intent": data.get("intent", "task_delegation"),
                }
            )
        elif event_type == "agent.message_handled":
            rows.append(
                {
                    "hop": data.get("hop", len(rows) + 1),
                    "sender": data.get("sender", "unknown"),
                    "recipient": data.get("agent_id", "unknown"),
                    "intent": data.get("intent", "handled"),
                }
            )

    return rows


def format_whiteboard_state(state: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Format whiteboard state entries for expandable JSON inspection."""
    formatted: dict[str, Any] = {}
    for key, entry in state.items():
        if isinstance(entry, dict):
            formatted[key] = {
                "version": entry.get("version", 1),
                "author": entry.get("author", "system"),
                "updated_at": entry.get("updated_at", ""),
                "value": entry.get("value"),
            }
        else:
            formatted[key] = entry
    return formatted


def evaluate_circuit_breaker_status(
    events: list[dict[str, Any]] | list[Any],
    max_hops: int = 5,
) -> dict[str, Any]:
    """Evaluate whether circuit breaker halted the session and compute current hop."""
    tripped = False
    halt_reason: str | None = None
    max_recorded_hop = 0

    for ev in events:
        event_type = (
            ev.event_type
            if hasattr(ev, "event_type")
            else (ev.get("event_type", "") if isinstance(ev, dict) else "")
        )
        data = (
            ev.data
            if hasattr(ev, "data")
            else (ev.get("data", {}) if isinstance(ev, dict) else {})
        )

        if event_type == "session.halted":
            tripped = True
            halt_reason = data.get("reason", "Hop limit exceeded")
        elif event_type in ("message.delivered", "agent.message_handled"):
            hop_val = int(data.get("hop", 0))
            if hop_val > max_recorded_hop:
                max_recorded_hop = hop_val

    if max_recorded_hop >= max_hops and not tripped:
        tripped = True
        halt_reason = f"HopLimitExceeded: reached {max_recorded_hop} hops (max: {max_hops})"

    return {
        "tripped": tripped,
        "current_hop": max_recorded_hop,
        "max_hops": max_hops,
        "halt_reason": halt_reason,
        "color": "red" if tripped else "green",
    }


def format_metrics_summary(metrics: Any) -> dict[str, Any]:
    """Format ValidationMetrics or dictionary into a clean metrics mapping."""
    if hasattr(metrics, "model_dump"):
        raw = metrics.model_dump()
    elif isinstance(metrics, dict):
        raw = dict(metrics)
    else:
        return {}

    return {
        "alpha_precision": raw.get("alpha_precision", 0.0),
        "alpha_recall": raw.get("alpha_recall", 0.0),
        "anomalies_mitigated_fraction": raw.get("anomalies_mitigated_fraction", 0.0),
        "total_repairs": raw.get("total_repairs", 0),
        "total_hops": raw.get("total_hops", 0),
        "wall_time_seconds": raw.get("wall_time_seconds", 0.0),
        "passed_schema": raw.get("passed_schema", False),
        "final_report_status": raw.get("final_report_status", "unknown"),
    }
