"""Agent Alpha — research / analysis agent.

Forensic financial analysis agent adhering to UACP-SCL specifications.
Integrates batch chunking, Whiteboard persistence under 'findings.v1',
rate-limit resilience, and downstream delegation to Agent Beta.
"""

from __future__ import annotations

import asyncio
import contextlib
import re
from pathlib import Path
from typing import Any, ClassVar

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from uacp.adapters.base import Adapter, RateLimited, Usage
from uacp.adapters.registry import get_adapter
from uacp.agents.base import Agent
from uacp.memory.whiteboard.client import WhiteboardClient
from uacp.protocol.capabilities import Capability
from uacp.protocol.messages import (
    Intent,
    Party,
    TaskDelegationPayload,
    TaskResultPayload,
    UACPMessage,
)
from uacp.protocol.router import Router

# ── Centralized Ability Configuration ─────────────────────────────────────────

ALPHA_CAPABILITIES: list[Capability] = [
    Capability(
        name="financial_audit",
        description="Forensic financial transaction auditing and anomaly detection.",
        strengths=["analysis", "pattern_detection"],
        complexity_level=3,
        cost_hint=0.2,
    )
]


def _find_default_scl_path() -> Path:
    """Locate examples/financial_auditor.scl relative to repo root or package."""
    cwd_candidate = Path.cwd() / "examples" / "financial_auditor.scl"
    if cwd_candidate.is_file():
        return cwd_candidate
    current = Path(__file__).resolve().parent
    for _ in range(4):
        cand = current / "examples" / "financial_auditor.scl"
        if cand.is_file():
            return cand
        current = current.parent
    return cwd_candidate


def _estimate_difficulty(anomalies: list[Finding]) -> int:
    """Self-assess task difficulty (1 to 5) based on anomaly types."""
    if not anomalies:
        return 1
    types = {a.type.lower() for a in anomalies}
    if "rounding_drift" in types:
        return 4
    if "sign_flip" in types or "missing_counterpart" in types:
        return 3
    if "duplicate_id" in types:
        return 2
    return 2


# ── Pydantic Models for Findings and Reports ──────────────────────────────────


class Finding(BaseModel):
    """Structured forensic anomaly finding."""

    model_config = ConfigDict(extra="ignore")

    id: str = Field(..., description="Transaction identifier.")
    batch: str = Field(default="BATCH-DEFAULT", description="Batch identifier.")
    delta: float = Field(default=0.0, description="Numerical discrepancy or delta.")
    type: str = Field(default="anomaly", description="Classification of anomaly.")
    evidence: str = Field(default="", description="Forensic evidence or explanatory notes.")
    difficulty: int | None = Field(
        default=None,
        ge=1,
        le=5,
        description="Optional self-assessed difficulty (1 to 5) for downstream planning.",
    )

    @classmethod
    def from_raw(cls, item: Any, default_batch: str = "BATCH-DEFAULT") -> Finding:
        """Convert a raw dictionary or legacy string into a Finding instance."""
        if isinstance(item, Finding):
            return item
        if isinstance(item, dict):
            batch_val = str(item.get("batch") or default_batch)
            delta_val = float(item.get("delta", 0.0) or 0.0)
            id_val = str(item.get("id") or "UNKNOWN")
            type_val = str(item.get("type") or "anomaly")
            evidence_val = str(item.get("evidence") or "")
            diff_val = item.get("difficulty")
            difficulty = int(diff_val) if diff_val is not None else None
            return cls(
                id=id_val,
                batch=batch_val,
                delta=delta_val,
                type=type_val,
                evidence=evidence_val,
                difficulty=difficulty,
            )
        if isinstance(item, str):
            match = re.search(r"\b(TX-[\w-]+)\b", item)
            tx_id = match.group(1) if match else f"STR-{abs(hash(item)) % 100000:05d}"
            return cls(
                id=tx_id,
                batch=default_batch,
                delta=0.0,
                type="text_finding",
                evidence=item,
            )
        raise ValueError(f"Cannot convert item of type {type(item)} to Finding: {item!r}")


class FindingsReport(BaseModel):
    """Aggregated audit findings report stored on Whiteboard under 'findings.v1'."""

    model_config = ConfigDict(extra="ignore")

    audit_complete: bool = Field(default=True, description="Whether audit completed fully.")
    status: str = Field(default="success", description="Audit status ('success' or 'incomplete').")
    anomalies: list[Finding] = Field(default_factory=list, description="Detected findings.")
    total_anomalies: int = Field(default=0, description="Total count of detected anomalies.")
    difficulty: int | None = Field(
        default=None,
        ge=1,
        le=5,
        description="Self-assessed aggregate difficulty for the group of findings.",
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Execution metadata including chunk counts and incomplete status.",
    )

    @field_validator("anomalies", mode="before")
    @classmethod
    def _coerce_anomalies(cls, v: Any) -> list[Any]:
        if not isinstance(v, list):
            return []
        coerced: list[Any] = []
        for item in v:
            if isinstance(item, (str, dict)):
                coerced.append(Finding.from_raw(item))
            else:
                coerced.append(item)
        return coerced

    @model_validator(mode="after")
    def _sync_totals_and_difficulty(self) -> FindingsReport:
        self.total_anomalies = len(self.anomalies)
        if self.difficulty is None and self.anomalies:
            difficulties = [f.difficulty for f in self.anomalies if f.difficulty is not None]
            if difficulties:
                self.difficulty = max(difficulties)
            else:
                self.difficulty = _estimate_difficulty(self.anomalies)
        return self


# ── Helper Functions ─────────────────────────────────────────────────────────


def chunk_ledger(
    ledger: list[dict[str, Any]],
    chunk_size: int = 20,
) -> list[tuple[str, list[dict[str, Any]]]]:
    """Group transactions by batch and partition into chunks of maximum chunk_size.

    Returns:
        List of (batch_name, chunk_transactions) tuples.
    """
    batches: dict[str, list[dict[str, Any]]] = {}
    for entry in ledger:
        b_name = str(entry.get("batch") or "BATCH-DEFAULT")
        batches.setdefault(b_name, []).append(entry)

    chunks: list[tuple[str, list[dict[str, Any]]]] = []
    for b_name, entries in batches.items():
        if chunk_size <= 0 or len(entries) <= chunk_size:
            chunks.append((b_name, entries))
        else:
            for i in range(0, len(entries), chunk_size):
                chunks.append((b_name, entries[i : i + chunk_size]))
    return chunks


def _extract_ledger(payload: dict[str, Any]) -> list[dict[str, Any]] | None:
    """Extract a list of transaction dictionaries from a payload dictionary, if present."""
    candidates = [
        payload.get("ledger_v1"),
        payload.get("ledger"),
        payload.get("transactions"),
        payload.get("data", {}).get("ledger_v1")
        if isinstance(payload.get("data"), dict)
        else None,
        payload.get("data", {}).get("ledger") if isinstance(payload.get("data"), dict) else None,
        payload.get("runtime", {}).get("payload", {}).get("ledger_v1")
        if isinstance(payload.get("runtime"), dict)
        and isinstance(payload.get("runtime", {}).get("payload"), dict)
        else None,
    ]
    for candidate in candidates:
        if isinstance(candidate, list) and candidate and isinstance(candidate[0], dict):
            return candidate
    return None


def _extract_anomalies_from_result(
    result: dict[str, Any] | list[Any] | str,
    default_batch: str = "BATCH-DEFAULT",
) -> list[Finding]:
    """Parse raw LLM output into a list of Finding objects."""
    if isinstance(result, dict):
        raw_list = result.get("anomalies") or []
        if isinstance(raw_list, list):
            return [Finding.from_raw(item, default_batch=default_batch) for item in raw_list]
    elif isinstance(result, list):
        return [Finding.from_raw(item, default_batch=default_batch) for item in result]
    elif isinstance(result, str):
        with contextlib.suppress(Exception):
            import json

            parsed = json.loads(result)
            return _extract_anomalies_from_result(parsed, default_batch=default_batch)
        return [Finding.from_raw(result, default_batch=default_batch)]
    return []


# ── Agent Alpha Implementation ───────────────────────────────────────────────


class AlphaResearchAgent(Agent):
    """Forensic financial research and analysis agent."""

    abilities: ClassVar[list[Capability]] = ALPHA_CAPABILITIES

    def __init__(
        self,
        agent_id: str = "alpha",
        scl_path: str | Path | None = None,
        adapter: Adapter | None = None,
        alias: str = "alpha_default",
        whiteboard: WhiteboardClient | None = None,
        router: Router | None = None,
        entries_per_batch: int = 20,
        max_llm_calls: int = 10,
        rate_limit_backoff: float = 1.0,
        **kwargs: Any,
    ) -> None:
        """Initialize Agent Alpha.

        Args:
            agent_id: Unique identifier for Agent Alpha (default: 'alpha').
            scl_path: Optional path to SCL contract (default: examples/financial_auditor.scl).
            adapter: Optional Adapter instance. If None, resolved via alias.
            alias: Model alias from config/models.yaml (default: 'alpha_default').
            whiteboard: Optional WhiteboardClient for state storage and audit logging.
            router: Optional Router for inter-agent communication.
            entries_per_batch: Maximum transaction entries per LLM call chunk (default: 20).
            max_llm_calls: Hard ceiling on LLM calls per task to protect free tier (default: 10).
            rate_limit_backoff: Base backoff time in seconds when RateLimited occurs.
            **kwargs: Extra arguments passed to base Agent.
        """
        resolved_scl = scl_path if scl_path is not None else _find_default_scl_path()
        resolved_adapter = adapter
        if resolved_adapter is None:
            with contextlib.suppress(Exception):
                resolved_adapter = get_adapter(alias)

        super().__init__(
            agent_id=agent_id,
            scl_path=resolved_scl,
            adapter=resolved_adapter,
            whiteboard=whiteboard,
            router=router,
            **kwargs,
        )
        self.entries_per_batch = entries_per_batch
        self.max_llm_calls = max_llm_calls
        self.rate_limit_backoff = rate_limit_backoff

    async def _run_chunk_with_retry(
        self,
        instruction: str,
        runtime_data: dict[str, Any],
        session_id: str | None,
        max_retries: int = 3,
    ) -> tuple[dict[str, Any] | list[Any] | str, Usage, int]:
        """Invoke LLM on a single chunk, with retry on RateLimited exceptions."""
        for attempt in range(max_retries + 1):
            try:
                result, usage = await self.run_llm(
                    user_message=instruction,
                    runtime_data=runtime_data,
                    session_id=session_id,
                    intent="financial_audit",
                )
                repairs = max(0, usage.requests - 1)
                return result, usage, repairs
            except RateLimited as exc:
                if attempt >= max_retries:
                    raise
                delay = (
                    exc.retry_after
                    if exc.retry_after is not None and exc.retry_after > 0
                    else self.rate_limit_backoff
                )
                await asyncio.sleep(delay)
        raise RuntimeError("Unreachable rate limit loop exit")

    async def audit_ledger(
        self,
        ledger: list[dict[str, Any]],
        instruction: str = "Analyze the transactions and identify any anomalies or compliance violations.",
        session_id: str | None = None,
        entries_per_batch: int | None = None,
        max_llm_calls: int | None = None,
    ) -> tuple[FindingsReport, Usage, int]:
        """Audit a ledger by chunking, merging anomalies, and de-duplicating by (batch, id).

        Args:
            ledger: List of transaction entries.
            instruction: Task instructions.
            session_id: Optional Whiteboard session identifier.
            entries_per_batch: Optional override for entries per chunk.
            max_llm_calls: Optional override for maximum LLM calls.

        Returns:
            Tuple of (FindingsReport, combined Usage, total repairs).
        """
        chunk_size = entries_per_batch if entries_per_batch is not None else self.entries_per_batch
        max_calls = max_llm_calls if max_llm_calls is not None else self.max_llm_calls

        chunks = chunk_ledger(ledger, chunk_size=chunk_size)
        total_calls = 0
        total_chunks_processed = 0
        total_input_tokens = 0
        total_output_tokens = 0
        total_repairs = 0
        unique_anomalies: list[Finding] = []
        seen_keys: set[tuple[str, str]] = set()
        is_incomplete = False

        for batch_name, chunk in chunks:
            if total_calls >= max_calls:
                is_incomplete = True
                break

            runtime_data = {
                "ledger_v1": chunk,
                "runtime": {"payload": {"ledger_v1": chunk}},
                "data": {"ledger_v1": chunk},
            }

            parsed_chunk, usage, repairs = await self._run_chunk_with_retry(
                instruction=instruction,
                runtime_data=runtime_data,
                session_id=session_id,
            )
            total_calls += usage.requests
            total_chunks_processed += 1
            total_input_tokens += usage.input_tokens
            total_output_tokens += usage.output_tokens
            total_repairs += repairs

            chunk_anomalies = _extract_anomalies_from_result(
                parsed_chunk, default_batch=batch_name
            )
            for f in chunk_anomalies:
                key = (f.batch, f.id)
                if key not in seen_keys:
                    seen_keys.add(key)
                    unique_anomalies.append(f)

        status = "incomplete" if is_incomplete else "success"
        report = FindingsReport(
            audit_complete=not is_incomplete,
            status=status,
            anomalies=unique_anomalies,
            metadata={
                "total_chunks": len(chunks),
                "processed_chunks": total_chunks_processed,
                "is_incomplete": is_incomplete,
            },
        )
        combined_usage = Usage(
            input_tokens=total_input_tokens,
            output_tokens=total_output_tokens,
            requests=total_calls,
        )
        return report, combined_usage, total_repairs

    async def analyze(
        self,
        instruction: str = "Analyze transactions for anomalies",
        payload: dict[str, Any] | None = None,
        session_id: str | None = None,
    ) -> FindingsReport:
        """Direct call method analyzing ledger or text payload and returning FindingsReport."""
        data = dict(payload or {})
        ledger = _extract_ledger(data)
        if ledger is not None:
            report, _, _ = await self.audit_ledger(
                ledger=ledger,
                instruction=instruction,
                session_id=session_id,
            )
            return report

        if "ledger_v1" not in data:
            data["ledger_v1"] = []
        if "runtime" not in data:
            data["runtime"] = {"payload": {"ledger_v1": []}}
        elif isinstance(data["runtime"], dict) and "payload" not in data["runtime"]:
            data["runtime"]["payload"] = {"ledger_v1": []}
        elif isinstance(data["runtime"].get("payload"), dict):
            data["runtime"]["payload"].setdefault("ledger_v1", [])

        parsed_result, _ = await self.run_llm(
            user_message=instruction,
            runtime_data=data,
            session_id=session_id,
            intent="financial_audit",
        )
        anomalies = _extract_anomalies_from_result(parsed_result, default_batch="DEFAULT")
        return FindingsReport(audit_complete=True, status="success", anomalies=anomalies)

    async def handle_task_delegation(self, message: UACPMessage) -> UACPMessage:
        """Handle task delegation message from upstream coordinator or peer."""
        payload_dict = dict(message.payload)
        session_id = (
            payload_dict.get("whiteboard_session_ref")
            or message.correlation_id
            or (self.whiteboard.session_id if self.whiteboard else "default-session")
        )
        payload_dict["whiteboard_session_ref"] = session_id
        delegation = TaskDelegationPayload.model_validate(payload_dict)
        instruction = delegation.instruction

        ledger = _extract_ledger(message.payload)
        report: FindingsReport
        usage: Usage
        repairs: int

        if ledger is not None:
            report, usage, repairs = await self.audit_ledger(
                ledger=ledger,
                instruction=instruction,
                session_id=session_id,
            )
        else:
            runtime_data = dict(message.payload.get("data") or message.payload.get("runtime_data", {}) or {})
            if "ledger_v1" not in runtime_data:
                runtime_data["ledger_v1"] = []
            if "runtime" not in runtime_data:
                runtime_data["runtime"] = {"payload": {"ledger_v1": []}}
            elif isinstance(runtime_data["runtime"], dict) and "payload" not in runtime_data["runtime"]:
                runtime_data["runtime"]["payload"] = {"ledger_v1": []}
            elif isinstance(runtime_data["runtime"].get("payload"), dict):
                runtime_data["runtime"]["payload"].setdefault("ledger_v1", [])

            parsed_result, usage = await self.run_llm(
                user_message=instruction,
                runtime_data=runtime_data,
                session_id=session_id,
                intent=Intent.task_delegation.value,
            )
            repairs = max(0, usage.requests - 1)
            anomalies = _extract_anomalies_from_result(parsed_result, default_batch="DEFAULT")
            report = FindingsReport(
                audit_complete=True,
                status="success",
                anomalies=anomalies,
            )

        # 2. Write findings to Whiteboard under 'findings.v1'
        wb = self.whiteboard
        if wb is None and self.router is not None and self.router.whiteboard is not None:
            wb = WhiteboardClient(
                backend=self.router.whiteboard, session_id=session_id, author=self.agent_id
            )

        if wb is not None:
            if wb.session_id != session_id:
                wb = WhiteboardClient(
                    backend=wb.backend, session_id=session_id, author=self.agent_id
                )
            with contextlib.suppress(Exception):
                await wb.write_model("findings.v1", report)

        # 3. Send task_delegation message to Agent Beta via Router
        if self.router is not None:
            beta_msg = UACPMessage(
                correlation_id=message.correlation_id,
                trace_id=message.trace_id,
                sender=Party(
                    agent_id=self.agent_id,
                    engine=self.adapter.name if self.adapter else "default",
                ),
                recipient=Party(agent_id="beta"),
                intent=Intent.task_delegation,
                payload=TaskDelegationPayload(
                    instruction="Mitigate the detected financial anomalies in findings.v1",
                    whiteboard_session_ref=session_id,
                ).model_dump(),
            )
            with contextlib.suppress(Exception):
                await self.router.send(beta_msg, recipient_agent_id="beta", session_id=session_id)

        # 4. Return task_result summary
        task_result = TaskResultPayload(
            status=report.status,
            summary=(
                f"Audit by {self.agent_id}: {report.total_anomalies} anomalies detected "
                f"({report.status}, calls={usage.requests}, repairs={repairs})"
            ),
            result_ref="findings.v1",
        )
        payload_result = task_result.model_dump()
        payload_result["findings"] = report.model_dump(mode="json")
        payload_result["counts"] = {
            "anomalies": report.total_anomalies,
            "llm_calls": usage.requests,
        }
        payload_result["tokens"] = {
            "input_tokens": usage.input_tokens,
            "output_tokens": usage.output_tokens,
            "total_tokens": usage.input_tokens + usage.output_tokens,
        }
        payload_result["repairs"] = repairs
        payload_result["status"] = report.status

        engine = (
            self.card.engine
            if self.card
            else (self.adapter.name if self.adapter else "default")
        )
        return UACPMessage(
            correlation_id=message.correlation_id,
            trace_id=message.trace_id,
            sender=Party(agent_id=self.agent_id, engine=engine),
            recipient=message.sender,
            intent=Intent.task_result,
            payload=payload_result,
        )


AlphaAgent = AlphaResearchAgent
