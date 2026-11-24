"""Performance, forensic accuracy, and token accounting metrics for UACP challenge runs."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

try:
    from examples.data.make_ledger import find_anomalies_baseline
except ModuleNotFoundError:
    root_dir = Path(__file__).resolve().parent.parent.parent.parent
    if str(root_dir) not in sys.path:
        sys.path.insert(0, str(root_dir))
    from examples.data.make_ledger import find_anomalies_baseline
from pydantic import BaseModel, ConfigDict, Field
from rich.console import Console
from rich.table import Table

from uacp.agents.alpha_research import Finding
from uacp.agents.beta_execution import FinalReport


class ProviderTokens(BaseModel):
    """Token consumption and invocation count for a provider or alias."""

    model_config = ConfigDict(frozen=True)

    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    llm_calls: int = 0


class ValidationMetrics(BaseModel):
    """Comprehensive performance and forensic evaluation metrics."""

    model_config = ConfigDict(frozen=True)

    alpha_precision: float = Field(..., description="Precision of Alpha anomaly detection.")
    alpha_recall: float = Field(..., description="Recall of Alpha anomaly detection.")
    anomalies_mitigated_fraction: float = Field(
        ..., description="Fraction of detected baseline anomalies remediated by Beta."
    )
    initial_anomalies_count: int = Field(default=0)
    remaining_anomalies_count: int = Field(default=0)
    tokens_by_provider: dict[str, ProviderTokens] = Field(default_factory=dict)
    total_tokens: ProviderTokens = Field(default_factory=ProviderTokens)
    total_repairs: int = 0
    total_hops: int = 0
    wall_time_seconds: float = 0.0
    passed_schema: bool = False
    final_report_status: str = "unknown"


def calculate_alpha_metrics(
    findings: list[Finding | dict[str, Any]],
    ground_truth: list[dict[str, Any]],
) -> tuple[float, float]:
    """Calculate forensic precision and recall of Alpha findings against ground truth.

    Args:
        findings: Findings reported by Agent Alpha.
        ground_truth: Authoritative benchmark anomalies list.

    Returns:
        Tuple of (precision, recall) rounded to 4 decimals.
    """
    if not ground_truth:
        return 1.0, 1.0

    gt_ids = {str(item.get("id")) for item in ground_truth}

    detected_ids: set[str] = set()
    for f in findings:
        if isinstance(f, Finding):
            detected_ids.add(f.id)
        elif isinstance(f, dict) and "id" in f:
            detected_ids.add(str(f["id"]))

    tp = len(detected_ids.intersection(gt_ids))
    fp = len(detected_ids - gt_ids)
    fn = len(gt_ids - detected_ids)

    precision = float(tp) / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = float(tp) / (tp + fn) if (tp + fn) > 0 else 0.0

    return round(precision, 4), round(recall, 4)


def calculate_mitigation_fraction(
    initial_ledger: list[dict[str, Any]],
    corrected_ledger: list[dict[str, Any]] | None,
) -> tuple[float, int, int]:
    """Calculate the fraction of baseline anomalies eliminated by Beta's remediation.

    Returns:
        Tuple of (fraction_mitigated, initial_count, remaining_count).
    """
    try:
        initial_anomalies = find_anomalies_baseline(initial_ledger)
    except Exception:
        initial_anomalies = []

    init_count = len(initial_anomalies)
    if init_count == 0:
        return 1.0, 0, 0

    if not corrected_ledger:
        return 0.0, init_count, init_count

    try:
        remaining_anomalies = find_anomalies_baseline(corrected_ledger)
    except Exception:
        remaining_anomalies = initial_anomalies

    rem_count = len(remaining_anomalies)
    mitigated = max(0, init_count - rem_count)
    fraction = round(float(mitigated) / init_count, 4)
    return fraction, init_count, rem_count


def aggregate_token_metrics(
    whiteboard_events: list[dict[str, Any]],
    alpha_alias: str,
    beta_alias: str,
) -> tuple[dict[str, ProviderTokens], ProviderTokens, int]:
    """Aggregate token usage and call counts per provider from Whiteboard event log."""
    by_provider: dict[str, dict[str, int]] = {}
    tot_input = 0
    tot_output = 0
    tot_calls = 0
    tot_repairs = 0

    alias_map = {
        "alpha": alpha_alias,
        "beta": beta_alias,
    }

    for ev in whiteboard_events:
        if ev.get("event_type") == "agent.llm_call":
            data = ev.get("data", {})
            agent_id = str(data.get("agent_id", "unknown"))
            provider_key = alias_map.get(agent_id, agent_id)

            tokens = data.get("tokens", {})
            in_tok = int(tokens.get("input_tokens", 0))
            out_tok = int(tokens.get("output_tokens", 0))
            repairs = int(data.get("repairs", 0))

            if provider_key not in by_provider:
                by_provider[provider_key] = {
                    "input_tokens": 0,
                    "output_tokens": 0,
                    "total_tokens": 0,
                    "llm_calls": 0,
                }

            by_provider[provider_key]["input_tokens"] += in_tok
            by_provider[provider_key]["output_tokens"] += out_tok
            by_provider[provider_key]["total_tokens"] += in_tok + out_tok
            by_provider[provider_key]["llm_calls"] += 1

            tot_input += in_tok
            tot_output += out_tok
            tot_calls += 1
            tot_repairs += repairs

    providers_result = {
        k: ProviderTokens(
            input_tokens=v["input_tokens"],
            output_tokens=v["output_tokens"],
            total_tokens=v["total_tokens"],
            llm_calls=v["llm_calls"],
        )
        for k, v in by_provider.items()
    }

    total_tokens = ProviderTokens(
        input_tokens=tot_input,
        output_tokens=tot_output,
        total_tokens=tot_input + tot_output,
        llm_calls=tot_calls,
    )

    return providers_result, total_tokens, tot_repairs


def validate_final_report_schema(raw_report: Any) -> tuple[bool, str]:
    """Check whether the final report satisfies the FinalReport Pydantic schema."""
    if not isinstance(raw_report, dict):
        return False, "invalid_type"
    try:
        parsed = FinalReport.model_validate(raw_report)
        status = "success" if parsed.verification.passed else "remediation_failed"
        return True, status
    except Exception:
        return False, "schema_violation"


def render_metrics_table(metrics: ValidationMetrics, console: Console | None = None) -> None:
    """Print an attractive summary table of run metrics using Rich."""
    c = console or Console()

    table = Table(
        title="UACP Validation Challenge — Run Metrics",
        header_style="bold cyan",
        border_style="dim",
    )
    table.add_column("Metric", style="bold white", width=32)
    table.add_column("Value", justify="right", width=24)

    # Forensic Accuracy
    prec_str = f"{metrics.alpha_precision * 100:.1f}%"
    rec_str = f"{metrics.alpha_recall * 100:.1f}%"
    mitigated_count = metrics.initial_anomalies_count - metrics.remaining_anomalies_count
    mit_str = (
        f"{metrics.anomalies_mitigated_fraction * 100:.1f}% "
        f"({mitigated_count}/{metrics.initial_anomalies_count})"
    )

    table.add_row("Alpha Precision (vs GT)", prec_str)
    table.add_row("Alpha Recall (vs GT)", rec_str)
    table.add_row("Beta Mitigation Fraction", mit_str)

    # Provider Tokens
    table.add_section()
    for prov, tok in metrics.tokens_by_provider.items():
        prov_details = (
            f"{tok.total_tokens:,} ({tok.llm_calls} calls, "
            f"{tok.input_tokens}in/{tok.output_tokens}out)"
        )
        table.add_row(f"Provider '{prov}' Tokens", prov_details)
    table.add_row(
        "Total Tokens Consumed",
        f"{metrics.total_tokens.total_tokens:,} ({metrics.total_tokens.llm_calls} calls)",
    )

    # Run Mechanics
    table.add_section()
    table.add_row("Total Protocol Hops", str(metrics.total_hops))
    table.add_row("Total Repair Attempts", str(metrics.total_repairs))
    table.add_row("Wall Clock Time", f"{metrics.wall_time_seconds:.2f}s")

    # Schema Validation
    table.add_section()
    schema_status = (
        "[bold green]PASSED[/bold green]"
        if metrics.passed_schema
        else "[bold red]FAILED[/bold red]"
    )
    table.add_row("Final Report Schema", schema_status)
    table.add_row("Final Report Status", metrics.final_report_status)

    c.print(table)


def save_metrics_json(metrics: ValidationMetrics, path: str | Path) -> None:
    """Save metrics to a JSON file."""
    file_path = Path(path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(metrics.model_dump(), f, indent=2)
