"""Agent Beta — code generation and remediation execution agent.

Adheres strictly to UACP-SCL specifications:
- Declares abilities (code_gen, verification, complexity_level=5, cost_hint) in one config place.
- Reads findings dynamically from the key specified in the task message (default 'findings.v1').
- Validates Whiteboard cryptographic integrity via verify_chain and refuses on tampering.
- Treats all Whiteboard artifacts and external inputs as untrusted data inputs.
- Validates code with AST static_check before sandboxing.
- Executes remediation code inside an isolated subprocess sandbox.
- Re-runs find_anomalies_baseline to verify complete resolution.
- Enforces an automated repair loop (1 retry) on failure, reporting honest failure.
- Persists validated FinalReport under 'final_report.v1' on the Whiteboard.
"""

from __future__ import annotations

import contextlib
import hashlib
import re
from pathlib import Path
from typing import Any, ClassVar

from examples.data.make_ledger import find_anomalies_baseline
from pydantic import BaseModel, ConfigDict, Field

from uacp.adapters.base import Adapter
from uacp.adapters.registry import get_adapter
from uacp.agents.alpha_research import Finding
from uacp.agents.base import Agent
from uacp.agents.sandbox import (
    SandboxOutputLimitError,
    SandboxResult,
    SandboxSecurityError,
    SandboxTimeoutError,
    run_in_sandbox,
    static_check,
)
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

BETA_CAPABILITIES: list[Capability] = [
    Capability(
        name="ledger_remediation",
        description="Generates, sandboxes, and verifies Python ledger remediation code.",
        strengths=["code_generation", "verification"],
        complexity_level=5,
        cost_hint=0.4,
    )
]


def _find_default_scl_path() -> Path:
    """Locate examples/beta_mitigator.scl relative to repo root or package."""
    cwd_candidate = Path.cwd() / "examples" / "beta_mitigator.scl"
    if cwd_candidate.is_file():
        return cwd_candidate
    current = Path(__file__).resolve().parent
    for _ in range(4):
        cand = current / "examples" / "beta_mitigator.scl"
        if cand.is_file():
            return cand
        current = current.parent
    return cwd_candidate


# ── Strict Report and Verification Schemas ─────────────────────────────────────


class VerificationDetails(BaseModel):
    """Post-remediation ledger verification results."""

    model_config = ConfigDict(extra="ignore")

    passed: bool = Field(
        ...,
        description="Whether baseline anomaly check confirmed all anomalies were resolved.",
    )
    details: str = Field(..., description="Forensic verification details or reason for failure.")


class FinalReport(BaseModel):
    """Final remediation report written to Whiteboard under 'final_report.v1'."""

    model_config = ConfigDict(extra="ignore")

    session_id: str = Field(..., description="Whiteboard session identifier.")
    anomalies_found: int = Field(
        default=0,
        ge=0,
        description="Total anomalies detected prior to remediation.",
    )
    anomalies_mitigated: int = Field(
        default=0,
        ge=0,
        description="Total anomalies successfully resolved.",
    )
    code_sha256: str = Field(
        default="",
        description="SHA-256 digest of executed Python remediation code.",
    )
    verification: VerificationDetails = Field(
        ...,
        description="Verification outcome from re-running baseline detector.",
    )
    residual_risks: list[str] = Field(
        default_factory=list,
        description="Documented operational or financial residual risks.",
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Execution metadata including repair attempts and timings.",
    )


# ── Agent Beta Implementation ──────────────────────────────────────────────────


class BetaExecutionAgent(Agent):
    """Agent Beta — code generation and sandboxed remediation agent."""

    abilities: ClassVar[list[Capability]] = BETA_CAPABILITIES

    def __init__(
        self,
        agent_id: str = "beta",
        scl_path: str | Path | None = None,
        adapter: Adapter | None = None,
        whiteboard: WhiteboardClient | None = None,
        router: Router | None = None,
        alias: str = "beta_default",
        timeout_s: float = 10.0,
        **kwargs: Any,
    ) -> None:
        """Initialize Agent Beta.

        Args:
            agent_id: Unique agent identifier (default 'beta').
            scl_path: Optional path to examples/beta_mitigator.scl.
            adapter: Optional Adapter instance. If None, resolves from alias 'beta_default'.
            whiteboard: Optional WhiteboardClient.
            router: Optional Router for communication.
            alias: Model alias in config/models.yaml (default 'beta_default' backing Groq).
            timeout_s: Subprocess execution timeout in seconds.
            **kwargs: Extra parameters passed to base Agent.
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
        self.timeout_s = timeout_s

    def _extract_findings_key(self, message: UACPMessage) -> str:
        """Resolve the Whiteboard key holding findings from the task message."""
        payload = message.payload
        if "findings_key" in payload and isinstance(payload["findings_key"], str):
            return payload["findings_key"]
        if "result_ref" in payload and isinstance(payload["result_ref"], str):
            return payload["result_ref"]

        # Check instruction text for pattern like "findings.v1" or similar
        instruction = str(payload.get("instruction", ""))
        match = re.search(r"\b(findings\.[a-zA-Z0-9_-]+)\b", instruction)
        if match:
            return match.group(1)

        return "findings.v1"

    def _extract_ledger(
        self,
        payload: dict[str, Any],
        wb_state: dict[str, dict[str, Any]] | None = None,
    ) -> list[dict[str, Any]]:
        """Extract transactions from payload or whiteboard state."""
        # 1. Payload direct candidates
        candidates = [
            payload.get("ledger_v1"),
            payload.get("ledger"),
            payload.get("transactions"),
            payload.get("data", {}).get("ledger_v1")
            if isinstance(payload.get("data"), dict)
            else None,
            payload.get("data", {}).get("ledger")
            if isinstance(payload.get("data"), dict)
            else None,
            payload.get("runtime", {}).get("payload", {}).get("ledger_v1")
            if isinstance(payload.get("runtime"), dict)
            and isinstance(payload.get("runtime", {}).get("payload"), dict)
            else None,
        ]
        for c in candidates:
            if isinstance(c, list) and c and isinstance(c[0], dict):
                return c

        # 2. Whiteboard state candidates
        if wb_state is not None:
            for k in ("ledger.v1", "ledger", "transactions.v1"):
                if k in wb_state:
                    val = wb_state[k].get("value")
                    if isinstance(val, list) and val and isinstance(val[0], dict):
                        return val
                    if (
                        isinstance(val, dict)
                        and "ledger" in val
                        and isinstance(val["ledger"], list)
                    ):
                        return val["ledger"]

        return []

    def _parse_llm_output(self, parsed_result: Any) -> tuple[str, str, list[str]]:
        """Extract python_code, rationale, and risks from SCL LLM output."""
        code = ""
        rationale = ""
        risks: list[str] = []

        if isinstance(parsed_result, dict):
            code = str(parsed_result.get("python_code") or "")
            rationale = str(parsed_result.get("rationale") or "")
            raw_risks = parsed_result.get("risks") or []
            if isinstance(raw_risks, list):
                risks = [str(r) for r in raw_risks]
            elif isinstance(raw_risks, str):
                risks = [raw_risks]
        elif isinstance(parsed_result, str):
            # Attempt to extract code blocks
            code_match = re.search(r"```python(.*?)```", parsed_result, re.DOTALL)
            code = code_match.group(1).strip() if code_match else parsed_result.strip()
            rationale = "Extracted from raw text model output."
            risks = []

        return code, rationale, risks

    def _try_execute(
        self, code: str, ledger: list[dict[str, Any]]
    ) -> tuple[bool, SandboxResult | None, str | None]:
        """Perform static check and sandboxed execution.

        Returns:
            Tuple of (success, result, error_details).
        """
        try:
            static_check(code)
        except SandboxSecurityError as sec_err:
            return False, None, f"Static security check rejected code: {sec_err}"
        except Exception as exc:
            return False, None, f"Static check exception: {exc}"

        try:
            res = run_in_sandbox(code, ledger, timeout_s=self.timeout_s)
            if res.exit_code != 0:
                err_msg = res.error or f"Process exited with non-zero code {res.exit_code}"
                return False, res, f"Sandbox execution failed: {err_msg}"
            return True, res, None
        except SandboxTimeoutError as t_err:
            return False, None, f"Sandbox execution timed out: {t_err}"
        except SandboxOutputLimitError as o_err:
            return False, None, f"Sandbox output limit exceeded: {o_err}"
        except Exception as exc:
            return False, None, f"Sandbox error: {exc}"

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

        # 1. Resolve Whiteboard client
        wb = self.whiteboard
        if wb is None and self.router is not None and self.router.whiteboard is not None:
            wb = WhiteboardClient(
                backend=self.router.whiteboard, session_id=session_id, author=self.agent_id
            )
        elif wb is not None and wb.session_id != session_id:
            wb = WhiteboardClient(backend=wb.backend, session_id=session_id, author=self.agent_id)

        # 2. Whiteboard verification & tamper check
        wb_state: dict[str, dict[str, Any]] = {}
        if wb is not None:
            with contextlib.suppress(Exception):
                wb_state = await wb.read_all()

            # Cryptographic chain verification
            try:
                report = await wb.verify_chain()
                if not report.is_valid:
                    # Refuse on tampering
                    task_result = TaskResultPayload(
                        status="failure",
                        summary=(
                            f"Refused execution: Whiteboard tamper detected for session "
                            f"'{session_id}'. {report.details}"
                        ),
                        result_ref=None,
                    )
                    engine = self.card.engine if self.card else "default"
                    return UACPMessage(
                        correlation_id=message.correlation_id,
                        trace_id=message.trace_id,
                        sender=Party(agent_id=self.agent_id, engine=engine),
                        recipient=message.sender,
                        intent=Intent.task_result,
                        payload=task_result.model_dump(),
                    )
            except Exception as e:
                # If verify_chain raises TamperError or similar, refuse
                task_result = TaskResultPayload(
                    status="failure",
                    summary=f"Refused execution: Whiteboard verification exception ({e})",
                    result_ref=None,
                )
                engine = self.card.engine if self.card else "default"
                return UACPMessage(
                    correlation_id=message.correlation_id,
                    trace_id=message.trace_id,
                    sender=Party(agent_id=self.agent_id, engine=engine),
                    recipient=message.sender,
                    intent=Intent.task_result,
                    payload=task_result.model_dump(),
                )

        # 3. Read findings from whiteboard key dynamically
        findings_key = self._extract_findings_key(message)
        anomalies_list: list[dict[str, Any]] = []

        if wb is not None and findings_key in wb_state:
            raw_findings = wb_state[findings_key]["value"]
            if isinstance(raw_findings, dict) and "anomalies" in raw_findings:
                anomalies_list = [
                    Finding.from_raw(a).model_dump() for a in raw_findings.get("anomalies", [])
                ]
            elif isinstance(raw_findings, list):
                anomalies_list = [Finding.from_raw(a).model_dump() for a in raw_findings]
        elif "findings" in message.payload:
            p_f = message.payload["findings"]
            if isinstance(p_f, list):
                anomalies_list = [Finding.from_raw(a).model_dump() for a in p_f]
            elif isinstance(p_f, dict) and "anomalies" in p_f:
                anomalies_list = [Finding.from_raw(a).model_dump() for a in p_f["anomalies"]]

        total_anomalies = len(anomalies_list)

        # 4. Extract ledger
        ledger = self._extract_ledger(message.payload, wb_state)

        # 5. Build runtime data (treating external inputs as untrusted data)
        runtime_data: dict[str, Any] = {
            "findings": anomalies_list,
            "ledger_v1": ledger,
        }

        # 6. First LLM generation pass
        parsed_result, _usage = await self.run_llm(
            user_message=instruction,
            runtime_data=runtime_data,
            session_id=session_id,
            intent="ledger_remediation",
        )

        code, rationale, risks = self._parse_llm_output(parsed_result)
        code_sha256 = hashlib.sha256(code.encode("utf-8")).hexdigest() if code else ""

        # 7. Static check and sandboxed execution
        exec_ok, sandbox_res, exec_error = self._try_execute(code, ledger)
        repairs_attempted = 0

        # 8. Single corrective repair path if failed
        if not exec_ok:
            repairs_attempted = 1
            repair_prompt = (
                f"The generated remediation code failed validation/execution:\n"
                f"{exec_error}\n\n"
                f"Please repair the code. The output must strictly define "
                f"`def mitigate(ledger: list[dict]) -> tuple[list[dict], list[dict]]`, "
                f"use ONLY allowlisted standard modules, "
                f"avoid calls to prohibited functions (eval, exec, open, compile), "
                f"avoid dunder attributes, and fix all anomalies."
            )

            try:
                repair_result, _repair_usage = await self.run_llm(
                    user_message=repair_prompt,
                    runtime_data=runtime_data,
                    session_id=session_id,
                    intent="ledger_remediation_repair",
                )
                repaired_code, repair_rationale, repair_risks = self._parse_llm_output(
                    repair_result
                )
                if repaired_code:
                    code = repaired_code
                    code_sha256 = hashlib.sha256(code.encode("utf-8")).hexdigest()
                    rationale = repair_rationale or rationale
                    risks = repair_risks or risks
                exec_ok, sandbox_res, exec_error = self._try_execute(code, ledger)
            except Exception as repair_exc:
                exec_ok = False
                exec_error = f"Repair invocation failed: {repair_exc}"

        # 9. Handle double failure honestly
        if not exec_ok or sandbox_res is None or sandbox_res.exit_code != 0:
            final_report = FinalReport(
                session_id=session_id,
                anomalies_found=total_anomalies,
                anomalies_mitigated=0,
                code_sha256=code_sha256,
                verification=VerificationDetails(
                    passed=False,
                    details=f"Remediation code failed execution: {exec_error}",
                ),
                residual_risks=risks or ["Remediation execution failed; manual review required."],
                metadata={
                    "repairs_attempted": repairs_attempted,
                    "final_status": "failed",
                    "error": exec_error,
                },
            )

            if wb is not None:
                with contextlib.suppress(Exception):
                    await wb.write_model("final_report.v1", final_report)

            task_result = TaskResultPayload(
                status="failure",
                summary=(
                    f"Remediation failed by {self.agent_id}: code execution error ({exec_error})"
                ),
                result_ref="final_report.v1",
            )
            payload_out = task_result.model_dump()
            payload_out["final_report"] = final_report.model_dump()
            engine = self.card.engine if self.card else "default"
            return UACPMessage(
                correlation_id=message.correlation_id,
                trace_id=message.trace_id,
                sender=Party(agent_id=self.agent_id, engine=engine),
                recipient=message.sender,
                intent=Intent.task_result,
                payload=payload_out,
            )

        # 10. Re-run baseline anomaly detector on corrected ledger
        try:
            remaining_anomalies = find_anomalies_baseline(sandbox_res.corrected_ledger)
        except Exception:
            remaining_anomalies = [
                {"id": "UNKNOWN", "batch": "UNKNOWN", "type": "baseline_check_error", "delta": 0.0}
            ]

        anomalies_mitigated = max(0, total_anomalies - len(remaining_anomalies))
        verification_passed = len(remaining_anomalies) == 0

        if verification_passed:
            verif_details = (
                "Verification passed: 0 baseline anomalies remaining in corrected ledger."
            )
        else:
            verif_details = (
                f"Verification incomplete: {len(remaining_anomalies)} baseline anomalies remain."
            )

        final_report = FinalReport(
            session_id=session_id,
            anomalies_found=total_anomalies,
            anomalies_mitigated=anomalies_mitigated,
            code_sha256=code_sha256,
            verification=VerificationDetails(
                passed=verification_passed,
                details=verif_details,
            ),
            residual_risks=risks,
            metadata={
                "repairs_attempted": repairs_attempted,
                "duration_s": sandbox_res.duration,
                "change_log_entries": len(sandbox_res.change_log),
                "remaining_anomalies": len(remaining_anomalies),
                "rationale": rationale,
            },
        )

        # 11. Write final_report.v1 to Whiteboard
        if wb is not None:
            with contextlib.suppress(Exception):
                await wb.write_model("final_report.v1", final_report)

        # 12. Return task_result
        task_status = "success" if verification_passed else "failure"
        task_result = TaskResultPayload(
            status=task_status,
            summary=(
                f"Remediation by {self.agent_id}: {anomalies_mitigated}/{total_anomalies} "
                f"anomalies mitigated. Verification passed={verification_passed}."
            ),
            result_ref="final_report.v1",
        )
        payload_out = task_result.model_dump()
        payload_out["final_report"] = final_report.model_dump()

        engine = self.card.engine if self.card else "default"
        return UACPMessage(
            correlation_id=message.correlation_id,
            trace_id=message.trace_id,
            sender=Party(agent_id=self.agent_id, engine=engine),
            recipient=message.sender,
            intent=Intent.task_result,
            payload=payload_out,
        )
