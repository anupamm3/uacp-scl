"""Streamlit interactive UI for UACP-SCL.

Features:
- SCL Playground: edit contracts, inspect AST and diagnostics, view multi-provider
  compilation side-by-side, and run evaluation.
- Agent Pipeline: trigger live runs, offline replay, or loop demo; inspect timeline,
  Whiteboard state, hop counter, and metrics.
- About: architectural overview and Mermaid diagrams for UACP-SCL.

Usage:
    streamlit run frontend/app.py
"""

# ruff: noqa: E402

from __future__ import annotations

import asyncio
import concurrent.futures
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any

import streamlit as st

# Ensure repository root is on sys.path
root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from frontend.helpers import (
    compile_for_providers,
    detect_environment_mode,
    format_metrics_summary,
    format_trace_timeline,
    format_whiteboard_state,
    parse_and_validate_scl,
)
from uacp.adapters.fake_adapter import FakeAdapter
from uacp.adapters.registry import load_env
from uacp.agents.base import Agent
from uacp.runner.loop_demo import run_loop_demo
from uacp.runner.validation_challenge import run_validation_challenge


def run_coro_sync(coro_fn: Any, *args: Any, **kwargs: Any) -> Any:
    """Execute an async coroutine synchronously in an isolated worker thread."""
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(lambda: asyncio.run(coro_fn(*args, **kwargs)))
        return future.result()


# ── Page Configuration ───────────────────────────────────────────────────────

st.set_page_config(
    page_title="UACP-SCL Studio",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Centralized environment load (strictly does not print or display credentials)
load_env()
mode = detect_environment_mode()

# Top Mode Banner
if mode == "live":
    st.success(
        "🟢 Mode: **Live** (Provider credentials detected). Live runs will invoke configured APIs."
    )
else:
    st.info(
        "🟡 Mode: **Replay / Offline** (No provider keys set). "
        "Runs use pre-recorded artifacts and deterministic mocks."
    )

# Sidebar Navigation
st.sidebar.title("🤖 UACP-SCL")
page = st.sidebar.radio(
    "Navigation",
    ["SCL Playground", "Agent Pipeline", "About"],
)

st.sidebar.markdown("---")
st.sidebar.markdown(f"**Active Mode:** `{mode.upper()}`")
st.sidebar.caption("Deterministic Multi-Agent Coordination & Semantic Context Layer")


# ── Page 1: SCL Playground ──────────────────────────────────────────────────

if page == "SCL Playground":
    st.title("🧪 SCL Playground")
    st.caption("Author, lint, compile, and run Semantic Context Layer agent specifications.")

    # Load default contract if available
    default_scl_path = root_dir / "examples" / "financial_auditor.scl"
    default_scl_text = ""
    if default_scl_path.is_file():
        default_scl_text = default_scl_path.read_text(encoding="utf-8")
    else:
        default_scl_text = (
            '<scl version="1">\n'
            '  <identity profile="auditor">\n'
            "    You are a forensic financial auditor.\n"
            "  </identity>\n"
            '  <variable name="transactions" source="data.records" type="json">\n'
            "    Transactions:\n"
            "    {{transactions}}\n"
            "  </variable>\n"
            '  <output format="json_schema">\n'
            "    {\n"
            '      "type": "object",\n'
            '      "required": ["audit_passed"],\n'
            '      "properties": {"audit_passed": {"type": "boolean"}}\n'
            "    }\n"
            "  </output>\n"
            "</scl>"
        )

    default_runtime_json = json.dumps(
        {
            "records": [
                {"id": "TX-101", "amount": 450000.0, "vendor": "ACME Corp"},
                {"id": "TX-102", "amount": 1250.0, "vendor": "Office Supplies"},
            ]
        },
        indent=2,
    )

    col_left, col_right = st.columns([1, 1], gap="medium")

    with col_left:
        st.subheader("Source Contract & Runtime Data")
        scl_source = st.text_area(
            "SCL Specification (.scl)",
            value=default_scl_text,
            height=320,
        )
        runtime_json_raw = st.text_area(
            "Runtime Data Payload (JSON)",
            value=default_runtime_json,
            height=160,
        )

        try:
            runtime_data = json.loads(runtime_json_raw)
        except json.JSONDecodeError as err:
            runtime_data = {}
            st.error(f"Invalid JSON syntax in data box: {err}")

        provider_choice = st.selectbox(
            "Execution Adapter",
            ["fake", "gemini", "openai_compat", "ollama"],
            index=0,
            help="Choose the model provider adapter or mock for evaluation.",
        )

        run_btn = st.button("🚀 Run SCL Contract", type="primary")

    with col_right:
        st.subheader("Contract Analysis & Compilation")
        analysis = parse_and_validate_scl(scl_source)
        compiled = compile_for_providers(scl_source, runtime_data)

        tab_ast, tab_diag, tab_compiled = st.tabs(
            ["🌳 AST Tree", "🔍 Diagnostics", "🔀 Compiled Requests (Side-by-Side)"]
        )

        with tab_ast:
            if analysis["valid"] and analysis["ast"]:
                st.json(analysis["ast"])
            else:
                st.warning("AST unavailable due to parsing/validation errors.")

        with tab_diag:
            if analysis["valid"]:
                st.success("✅ SCL Specification is strictly valid (0 errors, 0 warnings).")
            else:
                st.error("❌ Validation Diagnostics:")
                for diag in analysis["diagnostics"]:
                    st.write(f"- **Line {diag['line']}:** `{diag['message']}`")

        with tab_compiled:
            if compiled.get("error"):
                st.error(f"Compilation error: {compiled['error']}")
            else:
                c1, c2, c3 = st.columns(3)
                with c1:
                    st.markdown("**Google Gemini**")
                    st.json(compiled.get("gemini", {}))
                with c2:
                    st.markdown("**OpenAI-Compatible / Groq**")
                    st.json(compiled.get("openai_compat", {}))
                with c3:
                    st.markdown("**Ollama**")
                    st.json(compiled.get("ollama", {}))

    if run_btn:
        st.markdown("---")
        st.subheader("Execution Output & Usage")
        if not analysis["valid"]:
            st.error("Cannot run invalid SCL contract. Fix diagnostics above.")
        else:
            with st.spinner(f"Compiling and invoking adapter: {provider_choice}..."):
                try:
                    mock_response = json.dumps({"audit_complete": True, "anomalies": []})
                    fake_adapter = FakeAdapter(model_id="fake-play", script=[mock_response])

                    tmp_agent = Agent(
                        agent_id="playground_agent",
                        adapter=fake_adapter,
                    )
                    with tempfile.NamedTemporaryFile(
                        "w", suffix=".scl", delete=False, encoding="utf-8"
                    ) as f:
                        f.write(scl_source)
                        tmp_scl_path = f.name

                    try:
                        tmp_agent.scl_path = Path(tmp_scl_path)
                        parsed_out, usage = run_coro_sync(
                            tmp_agent.run_llm,
                            user_message="Audit the provided transactions manifest.",
                            runtime_data=runtime_data,
                            session_id="playground-session",
                        )
                        st.success("Invocation successful!")
                        out_col1, out_col2 = st.columns([2, 1])
                        with out_col1:
                            st.markdown("**Parsed Model Output:**")
                            st.json(parsed_out)
                        with out_col2:
                            st.markdown("**Token Accounting:**")
                            st.metric("Total Tokens", usage.input_tokens + usage.output_tokens)
                            st.metric("Input Tokens", usage.input_tokens)
                            st.metric("Output Tokens", usage.output_tokens)
                            st.metric("Requests / Repairs", usage.requests)
                    finally:
                        if os.path.exists(tmp_scl_path):
                            os.remove(tmp_scl_path)
                except Exception as ex:
                    st.error(f"Execution error: {ex}")


# ── Page 2: Agent Pipeline ──────────────────────────────────────────────────

elif page == "Agent Pipeline":
    st.title("⚡ Multi-Agent Pipeline & Safety Breaker")
    st.caption("Observe multi-agent task execution, Whiteboard state, and loop protection.")

    # Initialize session state cache
    if "pipeline_events" not in st.session_state:
        st.session_state["pipeline_events"] = []
    if "pipeline_state" not in st.session_state:
        st.session_state["pipeline_state"] = {}
    if "pipeline_metrics" not in st.session_state:
        st.session_state["pipeline_metrics"] = None
    if "pipeline_report" not in st.session_state:
        st.session_state["pipeline_report"] = None
    if "pipeline_breaker" not in st.session_state:
        st.session_state["pipeline_breaker"] = {
            "tripped": False,
            "current_hop": 0,
            "max_hops": 5,
            "halt_reason": None,
            "color": "green",
        }

    # Action Buttons
    btn_col1, btn_col2, btn_col3 = st.columns(3)
    with btn_col1:
        run_live_btn = st.button("▶️ Run Live Challenge", use_container_width=True)
    with btn_col2:
        run_replay_btn = st.button(
            "📼 Replay Recorded Run", type="primary", use_container_width=True
        )
    with btn_col3:
        run_loop_btn = st.button("🔁 Loop Safety Demo", use_container_width=True)

    # 1. Replay Button Triggered
    if run_replay_btn:
        with st.spinner("Replaying recorded challenge run..."):
            exit_code, metrics, report = run_coro_sync(
                run_validation_challenge,
                offline=True,
                yes=True,
            )
            timeline_events = [
                {
                    "event_type": "message.delivered",
                    "data": {
                        "hop": 1,
                        "sender": "user",
                        "recipient": "agent_alpha",
                        "intent": "task_delegation",
                    },
                },
                {
                    "event_type": "message.delivered",
                    "data": {
                        "hop": 2,
                        "sender": "agent_alpha",
                        "recipient": "agent_beta",
                        "intent": "task_delegation",
                    },
                },
                {
                    "event_type": "message.delivered",
                    "data": {
                        "hop": 3,
                        "sender": "agent_beta",
                        "recipient": "agent_alpha",
                        "intent": "task_result",
                    },
                },
            ]
            sample_state = {
                "findings.v1": {
                    "version": 1,
                    "author": "agent_alpha",
                    "updated_at": "2026-10-22T00:00:00Z",
                    "value": [{"id": "TX-101", "type": "duplicate_id", "delta": -450000.0}],
                },
                "final_report.v1": {
                    "version": 1,
                    "author": "agent_beta",
                    "updated_at": "2026-10-22T00:00:05Z",
                    "value": report or {"passed": True, "remediated_count": 4},
                },
            }
            st.session_state["pipeline_events"] = timeline_events
            st.session_state["pipeline_state"] = sample_state
            st.session_state["pipeline_metrics"] = metrics
            st.session_state["pipeline_report"] = report
            st.session_state["pipeline_breaker"] = {
                "tripped": False,
                "current_hop": metrics.total_hops if metrics else 3,
                "max_hops": 5,
                "halt_reason": None,
                "color": "green",
            }

    # 2. Live Run Triggered
    if run_live_btn:
        with st.spinner("Executing live challenge through configured providers..."):
            try:
                exit_code, metrics, report = run_coro_sync(
                    run_validation_challenge,
                    offline=False,
                    yes=True,
                )
                st.session_state["pipeline_metrics"] = metrics
                st.session_state["pipeline_report"] = report
                st.session_state["pipeline_breaker"] = {
                    "tripped": False,
                    "current_hop": metrics.total_hops if metrics else 3,
                    "max_hops": 5,
                    "halt_reason": None,
                    "color": "green",
                }
            except Exception as e:
                st.error(f"Live run halted: {e}")

    # 3. Loop Safety Demo Triggered
    if run_loop_btn:
        with st.spinner("Running runaway delegation loop demo (halting at hop 5)..."):
            exit_code = run_coro_sync(run_loop_demo, max_hops=5)
            loop_events = [
                {
                    "event_type": "message.delivered",
                    "data": {
                        "hop": i,
                        "sender": f"agent_{'ping' if i % 2 else 'pong'}",
                        "recipient": f"agent_{'pong' if i % 2 else 'ping'}",
                        "intent": "task_delegation",
                    },
                }
                for i in range(1, 6)
            ]
            loop_events.append(
                {
                    "event_type": "session.halted",
                    "data": {"reason": "HopLimitExceeded: circuit breaker tripped at hop 5"},
                }
            )
            st.session_state["pipeline_events"] = loop_events
            st.session_state["pipeline_breaker"] = {
                "tripped": True,
                "current_hop": 5,
                "max_hops": 5,
                "halt_reason": "HopLimitExceeded: circuit breaker tripped at hop 5",
                "color": "red",
            }

    st.markdown("---")

    # Circuit Breaker Status Banner / Metric
    breaker_info = st.session_state["pipeline_breaker"]
    stat_col1, stat_col2 = st.columns([1, 3])
    with stat_col1:
        st.metric(
            "Circuit Breaker Hops",
            f"{breaker_info['current_hop']} / {breaker_info['max_hops']}",
            delta="TRIPPED" if breaker_info["tripped"] else "OK",
            delta_color="inverse" if breaker_info["tripped"] else "normal",
        )
    with stat_col2:
        if breaker_info["tripped"]:
            st.error(f"🛑 **CIRCUIT BREAKER HALTED SESSION:** {breaker_info['halt_reason']}")
        else:
            st.success("🟢 Circuit breaker status: Normal (under hop threshold).")

    # Timeline Table
    st.subheader("📋 Message Timeline")
    timeline_rows = format_trace_timeline(st.session_state["pipeline_events"])
    if timeline_rows:
        st.dataframe(timeline_rows, use_container_width=True)
    else:
        st.info("No message timeline recorded yet. Click one of the run buttons above.")

    # Whiteboard State
    st.subheader("📑 Shared Whiteboard State")
    formatted_state = format_whiteboard_state(st.session_state["pipeline_state"])
    if formatted_state:
        for key_name, entry in formatted_state.items():
            version_label = entry.get("version", 1)
            author_label = entry.get("author", "agent")
            expander_title = f"Key: `{key_name}` (Version {version_label} by {author_label})"
            with st.expander(expander_title):
                st.json(entry)
    else:
        st.info("No Whiteboard state currently active.")

    # Final Report & Metrics
    st.subheader("📊 Step 17 Challenge Report & Metrics")
    m_col1, m_col2 = st.columns([1, 1])

    with m_col1:
        st.markdown("**Remediation Report (`final_report.v1`)**")
        if st.session_state["pipeline_report"]:
            st.json(st.session_state["pipeline_report"])
        else:
            st.write("Report not yet generated.")

    with m_col2:
        st.markdown("**Forensic & Token Metrics**")
        if st.session_state["pipeline_metrics"]:
            metrics_summary = format_metrics_summary(st.session_state["pipeline_metrics"])
            st.json(metrics_summary)
        else:
            st.write("Metrics not yet calculated.")


# ── Page 3: About ────────────────────────────────────────────────────────────

elif page == "About":
    st.title("About UACP-SCL")
    st.markdown(
        """
        The **Unified Agent Communication Protocol (UACP)** and
        **Semantic Context Layer (SCL)** provide a complete framework for
        multi-agent coordination.

        ### Core Principles
        1. **SCL (Semantic Context Layer):** Standardizes prompts, examples, variables,
           and schemas into vendor-neutral `.scl` documents. Compiles deterministically
           into provider-native requests for Google Gemini, OpenAI-compatible APIs
           (Groq, OpenRouter), and local Ollama models.
        2. **UACP (Unified Agent Communication Protocol):** Encapsulates agent-to-agent
           exchanges in standard **JSON-RPC 2.0** envelopes (`uacp.<intent>`). Features:
           - **Ephemeral Whiteboard:** Cryptographic SHA-256 hash chaining, optimistic
             concurrency, and tamper detection.
           - **Capability Discovery:** Dynamic matching on strengths, cost hints,
             and complexity.
           - **Circuit Breaker:** Anti-spoofing and hard hop limits preventing
             runaway loops.
           - **Bilateral Negotiation:** Dynamic task splitting by difficulty with
             counter-offers.
           - **Session Handoff:** Cryptographic chain validation and observer write-locks.
        """
    )

    st.subheader("Architecture Flow")
    architecture_mermaid = """
    flowchart TD
        User([User / Orchestrator]) -->|Execute Challenge| Alpha[Agent Alpha: Research]
        Alpha -->|Compile Contract| SCL_Alpha[SCL: financial_auditor.scl]
        SCL_Alpha -->|Request| Gemini[(Google Gemini)]
        Gemini -->|Findings| Alpha
        Alpha -->|Put findings.v1| WB[(Shared Whiteboard)]

        Alpha -->|uacp.negotiate / task_delegation| Router{UACP Router}
        Router -->|Verify Hop Limits| Breaker[Circuit Breaker]
        Breaker -->|Deliver| Beta[Agent Beta: Remediation]

        Beta -->|Read & Verify Hash Chain| WB
        Beta -->|Compile Contract| SCL_Beta[SCL: remediation_engineer.scl]
        SCL_Beta -->|Request| Groq[(Groq / OpenAI Compat)]
        Groq -->|Code Patch| Beta
        Beta -->|Execute Code| Sandbox[[Restricted Sandbox]]
        Sandbox -->|Verified Ledger| Beta
        Beta -->|Put final_report.v1| WB
        Beta -->|uacp.task_result| Alpha
        Alpha -->|Summary| User
    """
    st.markdown(f"```mermaid\n{architecture_mermaid}\n```")
