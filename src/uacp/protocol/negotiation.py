"""Bilateral task difficulty negotiation and protocol models."""

from __future__ import annotations

import contextlib
import uuid
from enum import StrEnum
from typing import Any, Protocol

from pydantic import BaseModel, ConfigDict, Field

from uacp.memory.whiteboard.base import WhiteboardBackend
from uacp.protocol.circuit_breaker import CircuitBreaker
from uacp.protocol.jsonrpc import JSONRPCErrorCode
from uacp.protocol.messages import CircuitBreakerState, Intent, Party, UACPMessage
from uacp.protocol.router import Router


def _default_subtask_id() -> str:
    """Generate a random unique subtask identifier."""
    return f"subtask-{uuid.uuid4().hex[:8]}"


class NegotiationAction(StrEnum):
    """Action outcome for a negotiation round."""

    accept = "accept"
    counter = "counter"
    reject = "reject"
    no_agreement = "no_agreement"


class NegotiationProposal(BaseModel):
    """Proposal specification carrying task constraints and difficulty score."""

    model_config = ConfigDict(extra="ignore")

    subtask_id: str = Field(default_factory=_default_subtask_id, description="Subtask ID.")
    subtask_text: str = Field(..., description="Actionable instruction or subtask description.")
    difficulty: int = Field(..., ge=1, le=5, description="Difficulty score from 1 to 5.")
    required_strengths: list[str] = Field(
        default_factory=list,
        description="Required capability strength tags.",
    )
    deadline_s: float = Field(..., gt=0, description="Execution deadline in seconds.")
    token_budget: int = Field(..., gt=0, description="Allocated token budget.")


class NegotiationDecision(BaseModel):
    """Decision answering a negotiation proposal or counter-offer."""

    model_config = ConfigDict(extra="ignore")

    action: NegotiationAction = Field(..., description="Chosen action (accept, counter, reject).")
    proposal: NegotiationProposal | None = Field(
        default=None,
        description="Revised proposal when action is counter.",
    )
    modified_constraints: dict[str, Any] | None = Field(
        default=None,
        description="Revised constraints when action is counter.",
    )
    reason_code: str = Field(..., description="Machine-readable operational reason code.")
    reason: str = Field(default="", description="Human-readable explanation.")
    round: int = Field(default=1, ge=1, description="Current negotiation round index.")


class AgentNegotiationContext(BaseModel):
    """Contextual metadata about an agent used by negotiation policies."""

    model_config = ConfigDict(frozen=True)

    agent_id: str
    complexity_level: int = Field(default=1, ge=1, le=5)
    strengths: list[str] = Field(default_factory=list)
    current_load: int = Field(default=0, ge=0)
    capabilities: list[str] = Field(default_factory=list)
    card: Any = Field(default=None)


class NegotiationPolicy(Protocol):
    """Replaceable agent policy evaluated per negotiation proposal."""

    def evaluate(
        self,
        proposal: Any,
        context: AgentNegotiationContext,
        round_num: int = 1,
    ) -> NegotiationDecision:
        """Evaluate a proposal and return an accept, counter, or reject decision."""
        ...



class DefaultNegotiationPolicy:
    """Default rule-based negotiation policy balancing load, complexity, and strengths.

    Rules:
    1. Accept when receiver's complexity_level >= proposed difficulty, required strengths
       overlap with receiver's strengths (or none required), and current_load < max_load.
    2. Counter when task is feasible with different constraints (longer deadline, bigger
       token budget, or difficulty score adjusted by at most 1 level per round with reason).
    3. Reject when receiver cannot perform the task at all, returning an explicit reason code.
    4. Maximum rounds: 2 rounds. Exceeding ends with 'no_agreement'.
    """

    def __init__(
        self,
        max_load: int = 5,
        max_rounds: int = 2,
        counter_deadline_multiplier: float = 1.5,
        counter_token_multiplier: float = 1.5,
    ) -> None:
        self.max_load = max_load
        self.max_rounds = max_rounds
        self.counter_deadline_multiplier = counter_deadline_multiplier
        self.counter_token_multiplier = counter_token_multiplier

    def evaluate(
        self,
        proposal: NegotiationProposal | Any,
        context: AgentNegotiationContext,
        round_num: int = 1,
    ) -> NegotiationDecision:
        """Evaluate proposal according to default policy rules."""
        if round_num > self.max_rounds:
            return NegotiationDecision(
                action=NegotiationAction.no_agreement,
                reason_code="MAX_ROUNDS_REACHED",
                reason=f"Exceeded maximum negotiation rounds ({self.max_rounds})",
                round=round_num,
            )

        # 1. Handle NegotiationProposal (legacy/difficulty-based)
        if isinstance(proposal, NegotiationProposal):
            if proposal.required_strengths:
                overlap = set(proposal.required_strengths) & set(context.strengths)
                if not overlap:
                    return NegotiationDecision(
                        action=NegotiationAction.reject,
                        reason_code="STRENGTH_MISMATCH",
                        reason=(
                            f"Agent '{context.agent_id}' lacks required strengths: "
                            f"{proposal.required_strengths}"
                        ),
                        round=round_num,
                    )

            if context.complexity_level < proposal.difficulty:
                diff_gap = proposal.difficulty - context.complexity_level
                if diff_gap > 1:
                    return NegotiationDecision(
                        action=NegotiationAction.reject,
                        reason_code="COMPLEXITY_EXCEEDED",
                        reason=(
                            f"Agent complexity {context.complexity_level} cannot handle "
                            f"difficulty {proposal.difficulty} (gap {diff_gap} > 1)"
                        ),
                        round=round_num,
                    )

                new_difficulty = max(1, proposal.difficulty - 1)
                counter_proposal = proposal.model_copy(update={"difficulty": new_difficulty})
                return NegotiationDecision(
                    action=NegotiationAction.counter,
                    proposal=counter_proposal,
                    modified_constraints={"difficulty": new_difficulty},
                    reason_code="DIFFICULTY_ADJUSTED",
                    reason=(
                        f"Reduced difficulty from {proposal.difficulty} to {new_difficulty} "
                        f"to match agent capacity"
                    ),
                    round=round_num,
                )

            if context.current_load >= self.max_load:
                new_deadline = round(proposal.deadline_s * self.counter_deadline_multiplier, 1)
                new_budget = int(proposal.token_budget * self.counter_token_multiplier)
                counter_proposal = proposal.model_copy(
                    update={"deadline_s": new_deadline, "token_budget": new_budget}
                )
                return NegotiationDecision(
                    action=NegotiationAction.counter,
                    proposal=counter_proposal,
                    modified_constraints={
                        "deadline_s": new_deadline,
                        "token_budget": new_budget,
                    },
                    reason_code="CONSTRAINTS_ADJUSTED",
                    reason=(
                        f"Agent load {context.current_load} >= max {self.max_load}; "
                        f"requested extended deadline {new_deadline}s and budget {new_budget}"
                    ),
                    round=round_num,
                )

            return NegotiationDecision(
                action=NegotiationAction.accept,
                reason_code="ACCEPTED",
                reason=f"Task accepted by {context.agent_id}",
                round=round_num,
            )

        # 2. Handle NegotiatePayload or dictionary proposal with AgentCard capability inspection
        task = getattr(proposal, "task", None) or (
            proposal.get("task") if isinstance(proposal, dict) else str(proposal)
        )
        raw_constraints = getattr(proposal, "constraints", None) or (
            proposal.get("constraints") if isinstance(proposal, dict) else None
        )
        constraints = dict(raw_constraints) if isinstance(raw_constraints, dict) else {}

        card = context.card
        if task:
            has_cap = False
            matching_cap = None
            if card is not None and hasattr(card, "capabilities"):
                for cap in card.capabilities:
                    if cap.name == task:
                        has_cap = True
                        matching_cap = cap
                        break
            elif context.capabilities:
                has_cap = task in context.capabilities

            if not has_cap and (card is not None or context.capabilities):
                return NegotiationDecision(
                    action=NegotiationAction.reject,
                    reason_code="CAPABILITY_NOT_FOUND",
                    reason=(
                        f"Agent '{context.agent_id}' does not support requested capability "
                        f"'{task}'"
                    ),
                    round=round_num,
                )

            if matching_cap is not None:
                # Latency / deadline constraint check
                req_deadline = constraints.get("deadline_s")
                if req_deadline is not None:
                    min_latency_s = (
                        matching_cap.avg_latency_ms / 1000.0
                        if matching_cap.avg_latency_ms > 0
                        else 0.0
                    )
                    if min_latency_s > 0 and float(req_deadline) < min_latency_s:
                        counter_dl = round(min_latency_s * self.counter_deadline_multiplier, 2)
                        return NegotiationDecision(
                            action=NegotiationAction.counter,
                            modified_constraints={**constraints, "deadline_s": counter_dl},
                            reason_code="DEADLINE_TOO_SHORT",
                            reason=(
                                f"Requested deadline {req_deadline}s is below average execution "
                                f"latency {min_latency_s:.2f}s; countered with {counter_dl}s"
                            ),
                            round=round_num,
                        )

                # Token / input chars constraint check
                req_tokens = constraints.get("max_tokens")
                if (
                    req_tokens is not None
                    and matching_cap.max_input_chars is not None
                    and int(req_tokens) > matching_cap.max_input_chars
                ):
                    counter_tokens = matching_cap.max_input_chars
                    return NegotiationDecision(
                            action=NegotiationAction.counter,
                            modified_constraints={**constraints, "max_tokens": counter_tokens},
                            reason_code="MAX_TOKENS_EXCEEDED",
                            reason=(
                                f"Requested max_tokens {req_tokens} exceeds agent limit "
                                f"{matching_cap.max_input_chars}; countered with {counter_tokens}"
                            ),
                            round=round_num,
                        )

                # Complexity constraint check
                req_comp = constraints.get("complexity")
                if req_comp is not None and int(req_comp) > matching_cap.complexity_level:
                    gap = int(req_comp) - matching_cap.complexity_level
                    if gap > 1:
                        return NegotiationDecision(
                            action=NegotiationAction.reject,
                            reason_code="COMPLEXITY_EXCEEDED",
                            reason=(
                                f"Requested complexity {req_comp} exceeds capability limit "
                                f"{matching_cap.complexity_level} (gap {gap} > 1)"
                            ),
                            round=round_num,
                        )
                    counter_comp = matching_cap.complexity_level
                    return NegotiationDecision(
                        action=NegotiationAction.counter,
                        modified_constraints={**constraints, "complexity": counter_comp},
                        reason_code="COMPLEXITY_ADJUSTED",
                        reason=f"Countered complexity adjusted to {counter_comp}",
                        round=round_num,
                    )

        # 3. Agent Load check
        if context.current_load >= self.max_load:
            orig_dl = float(constraints.get("deadline_s", 5.0))
            new_dl = round(orig_dl * self.counter_deadline_multiplier, 1)
            return NegotiationDecision(
                action=NegotiationAction.counter,
                modified_constraints={**constraints, "deadline_s": new_dl},
                reason_code="CONSTRAINTS_ADJUSTED",
                reason=(
                    f"Agent overloaded (load {context.current_load} >= {self.max_load}); "
                    f"countered with {new_dl}s"
                ),
                round=round_num,
            )

        return NegotiationDecision(
            action=NegotiationAction.accept,
            reason_code="ACCEPTED",
            reason=f"Task accepted by {context.agent_id}",
            round=round_num,
        )



def _get_current_hop(whiteboard: Any, session_id: str) -> int:
    """Read current session hop count from whiteboard backend if accessible."""
    backend = (
        getattr(whiteboard, "inner", None)
        or getattr(whiteboard, "_backend", None)
        or whiteboard
    )
    if hasattr(backend, "_sessions"):
        sessions_dict = backend._sessions
        if isinstance(sessions_dict, dict):
            session = sessions_dict.get(session_id)
            if session is not None and hasattr(session, "current_hop"):
                return int(session.current_hop)
    if hasattr(backend, "get_hop_count"):
        try:
            return int(backend.get_hop_count(session_id))
        except Exception:
            pass
    if hasattr(backend, "_conn") and hasattr(backend, "_lock"):
        try:
            with backend._lock:
                cursor = backend._conn.cursor()
                cursor.execute(
                    "SELECT current_hop FROM sessions WHERE session_id = ?",
                    (session_id,),
                )
                row = cursor.fetchone()
                if row:
                    return int(row[0])
        except Exception:
            pass
    return 0


class NegotiationOutcome(BaseModel):
    """Final recorded outcome of a bilateral negotiation."""

    model_config = ConfigDict(extra="ignore")

    subtask_id: str
    initiator: str
    responder: str
    initial_proposal: NegotiationProposal
    final_decision: NegotiationDecision
    history: list[dict[str, Any]] = Field(default_factory=list)
    agreed: bool = False
    rounds_completed: int = 1


async def record_negotiation_outcome(
    whiteboard: WhiteboardBackend,
    session_id: str,
    outcome: NegotiationOutcome,
    author: str,
) -> int:
    """Persist negotiation outcome to Whiteboard under negotiation.<subtask_id>."""
    key = f"negotiation.{outcome.subtask_id}"
    return await whiteboard.put(
        session_id=session_id,
        key=key,
        value=outcome.model_dump(mode="json"),
        author=author,
    )


async def execute_negotiation(
    router: Router,
    session_id: str,
    initiator_agent_id: str,
    responder_agent_id: str,
    proposal: NegotiationProposal,
    max_rounds: int = 2,
    circuit_breaker: CircuitBreaker | None = None,
) -> NegotiationOutcome:
    """Execute bilateral negotiation through Router, respecting circuit breaker hops.

    Returns NegotiationOutcome. If the circuit breaker trips or the round limit is
    exceeded, returns an explicit 'no_agreement' result rather than raising an exception.
    """
    history: list[dict[str, Any]] = []
    current_proposal = proposal
    round_num = 1
    breaker = circuit_breaker or router.breaker

    while round_num <= max_rounds:
        # Check if circuit breaker is already halted before dispatching
        if breaker.is_session_halted(session_id):
            halt_err = breaker.get_halt_error(session_id)
            err_msg = halt_err.message if halt_err else "Circuit breaker halted"
            outcome = NegotiationOutcome(
                subtask_id=proposal.subtask_id,
                initiator=initiator_agent_id,
                responder=responder_agent_id,
                initial_proposal=proposal,
                final_decision=NegotiationDecision(
                    action=NegotiationAction.no_agreement,
                    reason_code="CIRCUIT_BREAKER_TRIPPED",
                    reason=err_msg,
                    round=round_num,
                ),
                history=history,
                agreed=False,
                rounds_completed=round_num,
            )
            with contextlib.suppress(Exception):
                await record_negotiation_outcome(
                    router.whiteboard, session_id, outcome, author=initiator_agent_id
                )
            return outcome

        history.append(
            {
                "round": round_num,
                "step": "proposal",
                "from": initiator_agent_id,
                "to": responder_agent_id,
                "proposal": current_proposal.model_dump(),
            }
        )

        cur_hop = _get_current_hop(router.whiteboard, session_id)
        max_hops_limit = breaker.get_max_hops(session_id)

        msg = UACPMessage(
            correlation_id=session_id,
            trace_id=f"neg-{proposal.subtask_id}-r{round_num}",
            sender=Party(agent_id=initiator_agent_id),
            recipient=Party(agent_id=responder_agent_id),
            intent=Intent.negotiate,
            payload={
                "proposal": current_proposal.model_dump(),
                "round": round_num,
                "whiteboard_session_ref": session_id,
            },
            circuit_breaker=CircuitBreakerState(
                max_hops=max_hops_limit,
                current_hop=cur_hop + 1,
            ),
        )

        response = await router.send(msg, session_id=session_id)

        # Handle circuit breaker trip or delivery failure
        if response.error is not None:
            code = response.error.code
            is_trip = code in (
                JSONRPCErrorCode.HOP_LIMIT_EXCEEDED,
                JSONRPCErrorCode.QUOTA_EXCEEDED,
                JSONRPCErrorCode.SESSION_EXPIRED,
            )
            reason_code = "CIRCUIT_BREAKER_TRIPPED" if is_trip else "DELIVERY_ERROR"
            outcome = NegotiationOutcome(
                subtask_id=proposal.subtask_id,
                initiator=initiator_agent_id,
                responder=responder_agent_id,
                initial_proposal=proposal,
                final_decision=NegotiationDecision(
                    action=NegotiationAction.no_agreement,
                    reason_code=reason_code,
                    reason=response.error.message,
                    round=round_num,
                ),
                history=history,
                agreed=False,
                rounds_completed=round_num,
            )
            with contextlib.suppress(Exception):
                await record_negotiation_outcome(
                    router.whiteboard, session_id, outcome, author=initiator_agent_id
                )
            return outcome

        # Parse responder decision
        res_payload = (
            response.result.get("payload", {}) if isinstance(response.result, dict) else {}
        )
        raw_decision = res_payload.get("decision")
        if not raw_decision and isinstance(res_payload.get("proposal"), dict):
            raw_decision = res_payload.get("proposal")

        if not raw_decision:
            outcome = NegotiationOutcome(
                subtask_id=proposal.subtask_id,
                initiator=initiator_agent_id,
                responder=responder_agent_id,
                initial_proposal=proposal,
                final_decision=NegotiationDecision(
                    action=NegotiationAction.no_agreement,
                    reason_code="INVALID_RESPONSE",
                    reason="Missing decision in response",
                    round=round_num,
                ),
                history=history,
                agreed=False,
                rounds_completed=round_num,
            )
            with contextlib.suppress(Exception):
                await record_negotiation_outcome(
                    router.whiteboard, session_id, outcome, author=initiator_agent_id
                )
            return outcome

        decision = NegotiationDecision.model_validate(raw_decision)
        history.append(
            {
                "round": round_num,
                "step": "decision",
                "from": responder_agent_id,
                "to": initiator_agent_id,
                "decision": decision.model_dump(),
            }
        )

        if decision.action == NegotiationAction.accept:
            outcome = NegotiationOutcome(
                subtask_id=proposal.subtask_id,
                initiator=initiator_agent_id,
                responder=responder_agent_id,
                initial_proposal=proposal,
                final_decision=decision,
                history=history,
                agreed=True,
                rounds_completed=round_num,
            )
            with contextlib.suppress(Exception):
                await record_negotiation_outcome(
                    router.whiteboard, session_id, outcome, author=initiator_agent_id
                )
            return outcome

        if decision.action == NegotiationAction.reject:
            outcome = NegotiationOutcome(
                subtask_id=proposal.subtask_id,
                initiator=initiator_agent_id,
                responder=responder_agent_id,
                initial_proposal=proposal,
                final_decision=decision,
                history=history,
                agreed=False,
                rounds_completed=round_num,
            )
            with contextlib.suppress(Exception):
                await record_negotiation_outcome(
                    router.whiteboard, session_id, outcome, author=initiator_agent_id
                )
            return outcome

        if decision.action == NegotiationAction.counter:
            counter_prop = decision.proposal
            if counter_prop is None:
                outcome = NegotiationOutcome(
                    subtask_id=proposal.subtask_id,
                    initiator=initiator_agent_id,
                    responder=responder_agent_id,
                    initial_proposal=proposal,
                    final_decision=NegotiationDecision(
                        action=NegotiationAction.no_agreement,
                        reason_code="MALFORMED_COUNTER",
                        reason="Counter offer missing proposed constraints",
                        round=round_num,
                    ),
                    history=history,
                    agreed=False,
                    rounds_completed=round_num,
                )
                with contextlib.suppress(Exception):
                    await record_negotiation_outcome(
                        router.whiteboard, session_id, outcome, author=initiator_agent_id
                    )
                return outcome

            # Verify difficulty score change does not exceed 1 level per round
            diff_delta = abs(counter_prop.difficulty - current_proposal.difficulty)
            if diff_delta > 1:
                outcome = NegotiationOutcome(
                    subtask_id=proposal.subtask_id,
                    initiator=initiator_agent_id,
                    responder=responder_agent_id,
                    initial_proposal=proposal,
                    final_decision=NegotiationDecision(
                        action=NegotiationAction.reject,
                        reason_code="INVALID_COUNTER_DELTA",
                        reason=f"Difficulty change ({diff_delta}) exceeded 1 level limit",
                        round=round_num,
                    ),
                    history=history,
                    agreed=False,
                    rounds_completed=round_num,
                )
                with contextlib.suppress(Exception):
                    await record_negotiation_outcome(
                        router.whiteboard, session_id, outcome, author=initiator_agent_id
                    )
                return outcome

            current_proposal = counter_prop
            round_num += 1

    # Max rounds reached without accept/reject
    outcome = NegotiationOutcome(
        subtask_id=proposal.subtask_id,
        initiator=initiator_agent_id,
        responder=responder_agent_id,
        initial_proposal=proposal,
        final_decision=NegotiationDecision(
            action=NegotiationAction.no_agreement,
            reason_code="MAX_ROUNDS_REACHED",
            reason=f"Negotiation ended without agreement after {max_rounds} rounds",
            round=max_rounds,
        ),
        history=history,
        agreed=False,
        rounds_completed=max_rounds,
    )
    with contextlib.suppress(Exception):
        await record_negotiation_outcome(
            router.whiteboard, session_id, outcome, author=initiator_agent_id
        )
    return outcome
