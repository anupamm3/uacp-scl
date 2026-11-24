"""UACP runner package for multi-agent workflows, tracing, metrics, and offline replay."""

from __future__ import annotations

from uacp.runner.loop_demo import run_loop_demo
from uacp.runner.metrics import (
    ProviderTokens,
    ValidationMetrics,
    aggregate_token_metrics,
    calculate_alpha_metrics,
    calculate_mitigation_fraction,
    render_metrics_table,
    save_metrics_json,
    validate_final_report_schema,
)
from uacp.runner.recorder import (
    RecordingAdapter,
    RecordManager,
    ReplayAdapter,
    compute_request_hash,
)
from uacp.runner.trace import RunnerTracer, TracingRouter, TracingWhiteboardBackend
from uacp.runner.validation_challenge import run_validation_challenge

__all__ = [
    "ProviderTokens",
    "RecordManager",
    "RecordingAdapter",
    "ReplayAdapter",
    "RunnerTracer",
    "TracingRouter",
    "TracingWhiteboardBackend",
    "ValidationMetrics",
    "aggregate_token_metrics",
    "calculate_alpha_metrics",
    "calculate_mitigation_fraction",
    "compute_request_hash",
    "render_metrics_table",
    "run_loop_demo",
    "run_validation_challenge",
    "save_metrics_json",
    "validate_final_report_schema",
]
