"""End-to-end runner for the UACP forensic audit and remediation challenge."""

from __future__ import annotations

import json
import sys
import time
import uuid
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.panel import Panel
from rich.prompt import Confirm

from uacp.adapters.base import Adapter
from uacp.adapters.registry import get_adapter
from uacp.agents.alpha_research import AlphaResearchAgent
from uacp.agents.beta_execution import BetaExecutionAgent
from uacp.memory.whiteboard.base import WhiteboardBackend
from uacp.memory.whiteboard.client import WhiteboardClient
from uacp.memory.whiteboard.inmem_backend import InMemoryWhiteboardBackend
from uacp.memory.whiteboard.sqlite_backend import SqliteWhiteboardBackend
from uacp.protocol.capabilities import CapabilityRegistry
from uacp.protocol.circuit_breaker import CircuitBreaker
from uacp.protocol.messages import (
    CircuitBreakerState,
    Intent,
    Party,
    TaskDelegationPayload,
    UACPMessage,
)
from uacp.protocol.transport.inproc import InProcessTransport
from uacp.runner.metrics import (
    ValidationMetrics,
    aggregate_token_metrics,
    calculate_alpha_metrics,
    calculate_mitigation_fraction,
    render_metrics_table,
    save_metrics_json,
    validate_final_report_schema,
)
from uacp.runner.recorder import RecordManager
from uacp.runner.trace import RunnerTracer, TracingRouter, TracingWhiteboardBackend


def _find_path(rel_path: str) -> Path:
    """Find a path relative to current working directory or repository root."""
    candidate = Path.cwd() / rel_path
    if candidate.exists():
        return candidate
    current = Path(__file__).resolve().parent
    for _ in range(4):
        candidate = current / rel_path
        if candidate.exists():
            return candidate
        current = current.parent
    return Path.cwd() / rel_path


async def run_validation_challenge(
    alpha_alias: str = "alpha_default",
    beta_alias: str = "beta_default",
    backend_type: str = "inmem",
    replay_file: str | Path | None = None,
    record_file: str | Path | None = None,
    offline: bool = False,
    yes: bool = False,
    ledger_path: str | Path | None = None,
    ground_truth_path: str | Path | None = None,
    metrics_out: str | Path | None = None,
    alpha_adapter_override: Adapter | None = None,
    beta_adapter_override: Adapter | None = None,
    console: Console | None = None,
) -> tuple[int, ValidationMetrics | None, dict[str, Any] | None]:
    """Execute the end-to-end multi-agent forensic validation challenge.

    Returns:
        Tuple of (exit_code, metrics, final_report_dict).
        exit_code is 0 if final_report.v1 satisfies schema validation, 1 otherwise.
    """
    c = console or Console()
    tracer = RunnerTracer(console=c)
    tracer.print_section("UACP Validation Challenge — Multi-Agent Forensic Pipeline")

    resolved_ledger_path = (
        Path(ledger_path)
        if ledger_path
        else _find_path("examples/data/ledger_with_anomalies.json")
    )
    resolved_gt_path = (
        Path(ground_truth_path)
        if ground_truth_path
        else _find_path("examples/data/ground_truth.json")
    )

    # 1. Load data datasets
    if not resolved_ledger_path.is_file():
        c.print(f"[bold red]Error:[/bold red] Ledger dataset not found: {resolved_ledger_path}")
        return 1, None, None
    with open(resolved_ledger_path, encoding="utf-8") as f:
        ledger_data = json.load(f)

    ground_truth: list[dict[str, Any]] = []
    if resolved_gt_path.is_file():
        with open(resolved_gt_path, encoding="utf-8") as f:
            ground_truth = json.load(f)

    # 2. Quota Safety Check
    is_live_run = not offline and replay_file is None and alpha_adapter_override is None
    if is_live_run:
        c.print(
            Panel(
                f"[bold yellow]QUOTA SAFETY NOTICE:[/bold yellow]\n"
                f"You are about to run a LIVE execution using external LLM providers.\n"
                f"• Agent Alpha: [bold cyan]{alpha_alias}[/bold cyan] (Gemini API)\n"
                f"• Agent Beta:  [bold cyan]{beta_alias}[/bold cyan] (Groq API)\n"
                f"• Planned Max Calls: [bold white]~8[/bold white] (Alpha ~6, Beta ~2)\n"
                f"• Rate-limiting: Enforced by local QuotaGuard from config/models.yaml",
                border_style="yellow",
                title="Quota Guard",
            )
        )
        if not yes:
            if not sys.stdin.isatty():
                c.print(
                    "[bold red]Non-interactive environment detected without --yes. "
                    "Aborting live run.[/bold red]"
                )
                return 1, None, None
            confirmed = Confirm.ask("Proceed with planned live LLM calls?", console=c)
            if not confirmed:
                c.print("[yellow]Execution cancelled by user.[/yellow]")
                return 1, None, None

    # 3. Resolve Adapters
    record_mgr: RecordManager | None = None
    if record_file:
        record_mgr = RecordManager()

    offline_fixture = _find_path("examples/recorded/validation_challenge_offline.json")

    alpha_adapter: Adapter
    beta_adapter: Adapter

    if alpha_adapter_override is not None:
        alpha_adapter = alpha_adapter_override
    elif offline:
        tracer.on_info(f"Running in --offline mode using fixture: {offline_fixture.name}")
        alpha_adapter = RecordManager.create_replay_adapter("alpha_default", offline_fixture)
    elif replay_file:
        tracer.on_info(f"Running in --replay mode using file: {replay_file}")
        alpha_adapter = RecordManager.create_replay_adapter(alpha_alias, replay_file)
    else:
        alpha_adapter = get_adapter(alpha_alias)

    if beta_adapter_override is not None:
        beta_adapter = beta_adapter_override
    elif offline:
        beta_adapter = RecordManager.create_replay_adapter("beta_default", offline_fixture)
    elif replay_file:
        beta_adapter = RecordManager.create_replay_adapter(beta_alias, replay_file)
    else:
        beta_adapter = get_adapter(beta_alias)

    # Wrap with recorder if recording is requested
    if record_mgr is not None:
        alpha_adapter = record_mgr.wrap_adapter(alpha_alias, alpha_adapter)
        beta_adapter = record_mgr.wrap_adapter(beta_alias, beta_adapter)

    # 4. Whiteboard Setup
    raw_wb: WhiteboardBackend
    if backend_type == "sqlite":
        raw_wb = SqliteWhiteboardBackend(":memory:")
    else:
        raw_wb = InMemoryWhiteboardBackend()

    whiteboard = TracingWhiteboardBackend(raw_wb, tracer=tracer)
    session_id = await whiteboard.create_session(ttl_s=3600.0, max_hops=10)
    wb_client = WhiteboardClient(backend=whiteboard, session_id=session_id, author="orchestrator")

    # Store anomalous ledger on whiteboard under standard keys
    await wb_client.put("ledger.v1", ledger_data)
    await wb_client.put("ledger", ledger_data)

    tracer.on_info(
        f"Initialized Whiteboard session '{session_id}' "
        f"({backend_type} backend, {len(ledger_data)} entries)."
    )

    # 5. Router and Transport Setup
    transport = InProcessTransport()
    registry = CapabilityRegistry()
    circuit_breaker = CircuitBreaker(whiteboard=whiteboard, max_hops=10)
    router = TracingRouter(
        tracer=tracer,
        whiteboard=whiteboard,
        transport=transport,
        registry=registry,
        circuit_breaker=circuit_breaker,
        max_hops=10,
    )

    # 6. Instantiate and Register Agents
    alpha_scl = _find_path("examples/financial_auditor.scl")
    beta_scl = _find_path("examples/beta_mitigator.scl")

    wb_alpha = WhiteboardClient(backend=whiteboard, session_id=session_id, author="alpha")
    wb_beta = WhiteboardClient(backend=whiteboard, session_id=session_id, author="beta")

    alpha_agent = AlphaResearchAgent(
        agent_id="alpha",
        scl_path=alpha_scl,
        adapter=alpha_adapter,
        whiteboard=wb_alpha,
        router=router,
        alias=alpha_alias,
    )
    beta_agent = BetaExecutionAgent(
        agent_id="beta",
        scl_path=beta_scl,
        adapter=beta_adapter,
        whiteboard=wb_beta,
        router=router,
        alias=beta_alias,
    )

    alpha_agent.register(registry, router)
    beta_agent.register(registry, router)
    tracer.on_info("Registered Alpha and Beta with CapabilityRegistry and Router.")

    # 7. Execute Challenge Pipeline
    tracer.print_section("Pipeline Execution Trace")
    start_time = time.perf_counter()

    init_payload = TaskDelegationPayload(
        instruction="Perform forensic audit of ledger_v1 transactions and detect anomalies.",
        whiteboard_session_ref=session_id,
    ).model_dump()
    init_payload["ledger_v1"] = ledger_data
    init_payload["data"] = {"ledger_v1": ledger_data}

    init_message = UACPMessage(
        correlation_id=f"challenge-{uuid.uuid4().hex[:8]}",
        trace_id=f"trace-{uuid.uuid4().hex[:8]}",
        sender=Party(agent_id="orchestrator"),
        recipient=Party(agent_id="alpha"),
        intent=Intent.task_delegation,
        payload=init_payload,
        circuit_breaker=CircuitBreakerState(max_hops=10, current_hop=1),
    )

    # Dispatch to Alpha
    await router.send(init_message, recipient_agent_id="alpha", session_id=session_id)
    wall_time = time.perf_counter() - start_time

    # 8. Save Record File if requested
    if record_mgr is not None and record_file:
        record_mgr.save(record_file)
        tracer.on_info(
            f"Saved {len(record_mgr.records)} recorded LLM interactions to {record_file}"
        )

    # 9. Harvest Results from Whiteboard
    state = await whiteboard.get_state(session_id)
    events_raw = await whiteboard.events(session_id)
    events_list = [ev.model_dump() for ev in events_raw]

    findings_raw = state.get("findings.v1", {}).get("value")
    final_report_raw = state.get("final_report.v1", {}).get("value")

    # Schema Validation
    passed_schema, report_status = validate_final_report_schema(final_report_raw)

    # Forensic Accuracy Calculation
    findings_list: list[Any] = []
    if isinstance(findings_raw, dict):
        findings_list = findings_raw.get("anomalies") or []
    elif isinstance(findings_raw, list):
        findings_list = findings_raw

    alpha_prec, alpha_rec = calculate_alpha_metrics(findings_list, ground_truth)

    # Remediation Mitigation Calculation
    anomalies_found = 0
    anomalies_mitigated = 0
    if isinstance(final_report_raw, dict):
        anomalies_found = int(final_report_raw.get("anomalies_found", 0))
        anomalies_mitigated = int(final_report_raw.get("anomalies_mitigated", 0))

    if anomalies_found > 0:
        mitigated_frac = round(float(anomalies_mitigated) / anomalies_found, 4)
        rem_count = max(0, anomalies_found - anomalies_mitigated)
    else:
        mitigated_frac, anomalies_found, rem_count = calculate_mitigation_fraction(
            ledger_data, None
        )

    # Token and Repair Accounting
    by_prov, tot_tok, total_repairs = aggregate_token_metrics(
        events_list, alpha_alias=alpha_alias, beta_alias=beta_alias
    )

    total_hops = whiteboard.get_hop_count(session_id)

    metrics = ValidationMetrics(
        alpha_precision=alpha_prec,
        alpha_recall=alpha_rec,
        anomalies_mitigated_fraction=mitigated_frac,
        initial_anomalies_count=anomalies_found,
        remaining_anomalies_count=rem_count,
        tokens_by_provider=by_prov,
        total_tokens=tot_tok,
        total_repairs=total_repairs,
        total_hops=total_hops,
        wall_time_seconds=round(wall_time, 2),
        passed_schema=passed_schema,
        final_report_status=report_status,
    )

    # 10. Print Outputs
    tracer.print_section("Final Remediation Report (JSON)")
    if final_report_raw:
        c.print_json(json.dumps(final_report_raw, indent=2))
    else:
        c.print("[bold red]No final_report.v1 entry recorded on Whiteboard.[/bold red]")

    tracer.print_section("Evaluation Metrics")
    render_metrics_table(metrics, console=c)

    if metrics_out:
        save_metrics_json(metrics, metrics_out)
        tracer.on_info(f"Saved run metrics to {metrics_out}")

    exit_code = 0 if passed_schema else 1
    return exit_code, metrics, final_report_raw
