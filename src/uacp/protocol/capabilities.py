"""Agent capability discovery, registration, negotiation, and task splitting."""

from __future__ import annotations

import inspect
import time
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, TypeVar

from pydantic import BaseModel, ConfigDict, Field

from uacp.protocol.jsonrpc import (
    JSONRPCError,
    JSONRPCErrorCode,
    JSONRPCRequest,
    JSONRPCResponse,
    build_error_response,
    build_response,
)
from uacp.protocol.messages import (
    CapabilityQueryPayload,
    CapabilityResponsePayload,
    Intent,
    Party,
    UACPMessage,
)
from uacp.scl.parser import parse_file

F = TypeVar("F", bound=Callable[..., Any])


# ── Core Capability and Agent Models ────────────────────────────────────────


class Capability(BaseModel):
    """Represents a discrete functional capability advertised by an agent."""

    model_config = ConfigDict(frozen=True)

    name: str = Field(..., description="Unique capability identifier.")
    description: str = Field(default="", description="Human-readable capability description.")
    input_schema: dict[str, Any] | None = Field(
        default=None, description="JSON Schema for capability input parameters."
    )
    output_schema: dict[str, Any] | None = Field(
        default=None, description="JSON Schema for capability output."
    )
    cost_hint: float = Field(
        default=0.0, ge=0.0, description="Estimated relative cost or token cost hint."
    )
    avg_latency_ms: float = Field(
        default=0.0, ge=0.0, description="Average execution latency in milliseconds."
    )
    complexity_level: int = Field(
        default=1,
        ge=1,
        le=5,
        description="Hardest task difficulty this agent handles well (1 to 5).",
    )
    strengths: list[str] = Field(
        default_factory=list,
        description="Tag list of functional strengths (e.g., 'analysis', 'code_generation').",
    )
    max_input_chars: int | None = Field(
        default=None,
        ge=1,
        description="Maximum input characters handled without truncation, if bounded.",
    )


class AgentCard(BaseModel):
    """Advertised specification and capability manifest for an agent."""

    model_config = ConfigDict(extra="ignore", validate_assignment=True)

    agent_id: str = Field(..., description="Unique agent identifier.")
    engine: str = Field(..., description="LLM provider or engine moniker backing the agent.")
    role: str = Field(default="", description="Short human-readable role description.")
    current_load: int = Field(
        default=0,
        ge=0,
        description="Current task load on the agent.",
    )
    capabilities: list[Capability] = Field(
        default_factory=list,
        description="List of capabilities advertised by the agent.",
    )
    endpoint: str | None = Field(
        default=None,
        description="Network or transport endpoint address, if applicable.",
    )
    version: str = Field(
        default="1.0",
        description="Agent or manifest specification version.",
    )

    def has_capability(self, name: str) -> bool:
        """Return True if this card offers the named capability."""
        return any(cap.name == name for cap in self.capabilities)

    def get_capability(self, name: str) -> Capability | None:
        """Return the named Capability if present on this card, else None."""
        for cap in self.capabilities:
            if cap.name == name:
                return cap
        return None


class RankedAgent(BaseModel):
    """Ranked search result linking an agent card, matching capability, and rationale."""

    model_config = ConfigDict(frozen=True)

    card: AgentCard
    matching_capability: Capability
    reason: str = Field(..., description="Human/planner-readable rationale for ranking.")
    score: float = Field(default=0.0, description="Composite numeric ranking score.")

    @property
    def ranking_reason(self) -> str:
        """Alias for reason."""
        return self.reason

    @property
    def agent_id(self) -> str:
        """Convenience property for agent identifier."""
        return self.card.agent_id

    @property
    def engine(self) -> str:
        """Convenience property for backing engine."""
        return self.card.engine

    @property
    def current_load(self) -> int:
        """Convenience property for current load."""
        return self.card.current_load

    @property
    def role(self) -> str:
        """Convenience property for role description."""
        return self.card.role


class Subtask(BaseModel):
    """Specification of a subtask to be assigned to an agent."""

    model_config = ConfigDict(extra="ignore")

    id: str = Field(..., description="Unique subtask identifier.")
    capability: str = Field(..., description="Required capability identifier.")
    required_complexity: int = Field(
        default=1,
        ge=1,
        description="Required minimum complexity handling level.",
    )
    required_strengths: list[str] = Field(
        default_factory=list,
        description="Desired strength tags (e.g. ['analysis', 'code_generation']).",
    )
    description: str = Field(default="", description="Optional subtask instruction or details.")


class SubtaskAssignment(BaseModel):
    """Outcome of attempting to assign a subtask to an agent."""

    model_config = ConfigDict(frozen=True)

    subtask_id: str
    assigned: bool
    assigned_agent_id: str | None = None
    assigned_card: AgentCard | None = None
    reason: str


# ── Capability Decorator & Card Helper ──────────────────────────────────────


def capability(
    name: str,
    *,
    input: type[BaseModel] | None = None,
    output: type[BaseModel] | None = None,
    description: str = "",
    cost_hint: float = 0.0,
    avg_latency_ms: float = 0.0,
    complexity_level: int = 1,
    strengths: list[str] | None = None,
    max_input_chars: int | None = None,
) -> Callable[[F], F]:
    """Decorator attaching capability metadata to an agent method.

    Extracts JSON Schemas from the provided Pydantic input and output models.
    """

    def decorator(fn: F) -> F:
        doc = (getattr(fn, "__doc__", "") or "").strip()
        cap = Capability(
            name=name,
            description=description or doc,
            input_schema=input.model_json_schema() if input is not None else None,
            output_schema=output.model_json_schema() if output is not None else None,
            cost_hint=cost_hint,
            avg_latency_ms=avg_latency_ms,
            complexity_level=complexity_level,
            strengths=strengths or [],
            max_input_chars=max_input_chars,
        )
        setattr(fn, "__capability__", cap)  # noqa: B010
        return fn

    return decorator


def build_agent_card(
    agent_cls: type,
    scl_path: str | Path | None = None,
    agent_id: str | None = None,
    engine: str | None = None,
    role: str | None = None,
    endpoint: str | None = None,
    version: str = "1.0",
    current_load: int = 0,
) -> AgentCard:
    """Build an AgentCard automatically from a decorated agent class and optional .scl file.

    Args:
        agent_cls: Agent class with methods decorated by @capability.
        scl_path: Optional path to an SCL file defining agent identity and contracts.
        agent_id: Optional explicit agent ID override.
        engine: Optional explicit engine moniker override.
        role: Optional human-readable role override.
        endpoint: Optional transport endpoint URI.
        version: Manifest version (default '1.0').
        current_load: Initial task load integer (default 0).

    Returns:
        A validated AgentCard instance.
    """
    discovered_capabilities: list[Capability] = []

    for _, member in inspect.getmembers(agent_cls):
        cap = getattr(member, "__capability__", None)
        if isinstance(cap, Capability) and cap not in discovered_capabilities:
            discovered_capabilities.append(cap)

    resolved_agent_id: str | None = agent_id or getattr(agent_cls, "agent_id", None)
    resolved_engine: str = engine or getattr(agent_cls, "engine", "default") or "default"
    resolved_role: str | None = role or getattr(agent_cls, "role", None)
    resolved_version = version

    if scl_path is not None:
        doc = parse_file(str(scl_path))
        if resolved_agent_id is None:
            resolved_agent_id = doc.identity.profile.lower().replace(" ", "_")
        if resolved_role is None:
            resolved_role = doc.identity.body.strip()
        if doc.version and resolved_version == "1.0":
            resolved_version = doc.version

        # If no methods were decorated with @capability, build a default one from SCL
        if not discovered_capabilities:
            var_props = {
                var.name: {"type": "string" if var.type == "text" else "object"}
                for var in doc.variables
            }
            input_schema = (
                {"type": "object", "properties": var_props, "required": list(var_props.keys())}
                if var_props
                else None
            )
            output_schema = doc.output.schema if doc.output else None
            discovered_capabilities.append(
                Capability(
                    name=resolved_agent_id,
                    description=doc.identity.profile,
                    input_schema=input_schema,
                    output_schema=output_schema,
                )
            )

    final_agent_id = (
        resolved_agent_id if resolved_agent_id is not None else agent_cls.__name__.lower()
    )
    final_role = resolved_role if resolved_role is not None else (agent_cls.__doc__ or "").strip()

    return AgentCard(
        agent_id=final_agent_id,
        engine=resolved_engine,
        role=final_role,
        current_load=current_load,
        capabilities=discovered_capabilities,
        endpoint=endpoint,
        version=resolved_version,
    )


# ── Capability Registry ─────────────────────────────────────────────────────


class CapabilityRegistry:
    """In-memory registry managing agent cards, TTL liveness, queries, and task splits."""

    def __init__(
        self,
        default_ttl: float = 30.0,
        clock: Callable[[], float] | None = None,
    ) -> None:
        self.default_ttl = default_ttl
        self._clock = clock or time.time
        self._cards: dict[str, AgentCard] = {}
        self._last_heartbeat: dict[str, float] = {}
        self._ttls: dict[str, float] = {}

    def _now(self) -> float:
        return self._clock()

    def register(self, card: AgentCard, ttl_seconds: float | None = None) -> None:
        """Register or update an agent card with an active heartbeat and TTL."""
        self._cards[card.agent_id] = card
        self._last_heartbeat[card.agent_id] = self._now()
        self._ttls[card.agent_id] = ttl_seconds if ttl_seconds is not None else self.default_ttl

    def deregister(self, agent_id: str) -> bool:
        """Deregister an agent from the registry.

        Returns True if the agent existed and was removed, False otherwise.
        """
        existed = agent_id in self._cards
        self._cards.pop(agent_id, None)
        self._last_heartbeat.pop(agent_id, None)
        self._ttls.pop(agent_id, None)
        return existed

    def heartbeat(self, agent_id: str, timestamp: datetime | None = None) -> bool:
        """Renew the liveness lease for an agent.

        Returns True if the agent was found and renewed, False if not registered.
        """
        if agent_id not in self._cards:
            return False
        hb_ts = timestamp.timestamp() if timestamp is not None else self._now()
        self._last_heartbeat[agent_id] = hb_ts
        return True

    def is_alive(self, agent_id: str) -> bool:
        """Return True if the agent is registered and its heartbeat TTL has not expired."""
        if agent_id not in self._cards:
            return False
        elapsed = self._now() - self._last_heartbeat.get(agent_id, 0.0)
        ttl = self._ttls.get(agent_id, self.default_ttl)
        return elapsed <= ttl

    def update_load(self, agent_id: str, current_load: int) -> bool:
        """Update current task load on an agent.

        Returns True if agent exists and load was updated, False otherwise.
        """
        card = self._cards.get(agent_id)
        if card is None:
            return False
        card.current_load = current_load
        return True

    def get(self, agent_id: str) -> AgentCard | None:
        """Return the AgentCard for agent_id if registered and alive, else None."""
        if not self.is_alive(agent_id):
            return None
        return self._cards.get(agent_id)

    def find(
        self,
        capability_name: str,
        constraints: dict[str, Any] | None = None,
        *,
        required_complexity: int | None = None,
        required_strengths: list[str] | None = None,
        load_overrides: dict[str, int] | None = None,
    ) -> list[RankedAgent]:
        """Find active agents offering a capability, filtered and ranked by constraints.

        Ranking criteria:
        1. Best strength match first (highest count of matching required strengths).
        2. Lowest current load (using load_overrides if provided, else card.current_load).
        3. Lowest cost hint.
        4. Preferred engine match (bonus).
        5. Lowest average latency.
        6. Lexicographical agent_id tie-breaker.

        Args:
            capability_name: Name of capability queried.
            constraints: Optional dictionary with filtering/ranking keys (preferred_engine,
                         max_latency_ms, max_cost_hint, required_complexity, required_strengths).
            required_complexity: Minimum complexity level (1 to 5) handled by the agent.
            required_strengths: List of desired functional strengths.
            load_overrides: Optional mapping of agent_id to effective load (for split planning).

        Returns:
            A list of RankedAgent items sorted from best to worst candidate.
        """
        c_dict = constraints or {}
        req_comp = (
            required_complexity
            if required_complexity is not None
            else c_dict.get("required_complexity")
        )
        req_strengths = (
            required_strengths
            if required_strengths is not None
            else c_dict.get("required_strengths", [])
        )
        pref_engine = c_dict.get("preferred_engine")
        max_latency = c_dict.get("max_latency_ms")
        max_cost = c_dict.get("max_cost_hint")

        candidates: list[RankedAgent] = []

        for card in self._cards.values():
            if not self.is_alive(card.agent_id):
                continue

            matching_cap = card.get_capability(capability_name)
            if matching_cap is None:
                continue

            # Complexity filter: agent must handle at least the required complexity
            if req_comp is not None and matching_cap.complexity_level < req_comp:
                continue

            # Latency and cost threshold filters
            if max_latency is not None and matching_cap.avg_latency_ms > max_latency:
                continue
            if max_cost is not None and matching_cap.cost_hint > max_cost:
                continue

            effective_load = (
                load_overrides.get(card.agent_id, card.current_load)
                if load_overrides is not None
                else card.current_load
            )

            # Strength overlap calculation
            if req_strengths:
                matched_strengths = [s for s in req_strengths if s in matching_cap.strengths]
                strength_count = len(matched_strengths)
            else:
                matched_strengths = []
                strength_count = 0

            engine_match = (card.engine == pref_engine) if pref_engine else False

            # Build human-readable ranking rationale
            reason_parts: list[str] = []
            if req_strengths:
                s_str = ", ".join(matched_strengths) if matched_strengths else "none"
                reason_parts.append(
                    f"strengths_matched={strength_count}/{len(req_strengths)} ({s_str})"
                )
            reason_parts.append(f"load={effective_load}")
            reason_parts.append(f"cost={matching_cap.cost_hint}")
            reason_parts.append(f"complexity={matching_cap.complexity_level}>={req_comp or 1}")
            if pref_engine:
                reason_parts.append(f"engine_match={engine_match}")
            reason_parts.append(f"latency={matching_cap.avg_latency_ms}ms")

            score = float(
                strength_count * 100
                - effective_load * 10
                - matching_cap.cost_hint
                + (20.0 if engine_match else 0.0)
            )

            candidates.append(
                RankedAgent(
                    card=card,
                    matching_capability=matching_cap,
                    reason="; ".join(reason_parts),
                    score=score,
                )
            )

        # Sort: best strength match (-count), lowest load, lowest cost, engine match, latency, id
        def _sort_key(item: RankedAgent) -> tuple[int, int, float, int, float, str]:
            cap = item.matching_capability
            eff_load = (
                load_overrides.get(item.card.agent_id, item.card.current_load)
                if load_overrides is not None
                else item.card.current_load
            )
            s_count = len([s for s in req_strengths if s in cap.strengths]) if req_strengths else 0
            eng_match = (item.card.engine == pref_engine) if pref_engine else False
            return (
                -s_count,
                eff_load,
                cap.cost_hint,
                0 if eng_match else 1,
                cap.avg_latency_ms,
                item.card.agent_id,
            )

        candidates.sort(key=_sort_key)
        return candidates

    def suggest_split(
        self,
        subtasks: list[Subtask | dict[str, Any]],
    ) -> list[SubtaskAssignment]:
        """Suggest deterministic assignment of subtasks to registered agents.

        Balances loads dynamically across subtasks during the assignment plan.
        If a subtask cannot be handled, marks it unassigned with an explicit explanation.

        Args:
            subtasks: List of Subtask models or dictionaries containing id, capability,
                      required_complexity, and required_strengths.

        Returns:
            List of SubtaskAssignment results.
        """
        assignments: list[SubtaskAssignment] = []
        simulated_loads: dict[str, int] = {
            agent_id: card.current_load for agent_id, card in self._cards.items()
        }

        for item in subtasks:
            sub = item if isinstance(item, Subtask) else Subtask.model_validate(item)

            candidates = self.find(
                sub.capability,
                required_complexity=sub.required_complexity,
                required_strengths=sub.required_strengths,
                load_overrides=simulated_loads,
            )

            if not candidates:
                # Diagnose reason for assignment failure
                agents_with_cap = [
                    c
                    for c in self._cards.values()
                    if any(cap.name == sub.capability for cap in c.capabilities)
                ]
                if not agents_with_cap:
                    reason = f"No agent registered with capability '{sub.capability}'"
                else:
                    alive_with_cap = [c for c in agents_with_cap if self.is_alive(c.agent_id)]
                    if not alive_with_cap:
                        reason = (
                            f"All agents offering capability '{sub.capability}' "
                            "are expired (no heartbeat)"
                        )
                    else:
                        max_comp = max(
                            max(
                                cap.complexity_level
                                for cap in c.capabilities
                                if cap.name == sub.capability
                            )
                            for c in alive_with_cap
                        )
                        if max_comp < sub.required_complexity:
                            reason = (
                                f"Required complexity {sub.required_complexity} "
                                f"exceeds max available level {max_comp}"
                            )
                        else:
                            reason = (
                                "No active agent satisfied the required strengths "
                                "or latency/cost constraints"
                            )

                assignments.append(
                    SubtaskAssignment(
                        subtask_id=sub.id,
                        assigned=False,
                        assigned_agent_id=None,
                        assigned_card=None,
                        reason=reason,
                    )
                )
            else:
                chosen = candidates[0]
                simulated_loads[chosen.card.agent_id] = (
                    simulated_loads.get(chosen.card.agent_id, 0) + 1
                )
                assignments.append(
                    SubtaskAssignment(
                        subtask_id=sub.id,
                        assigned=True,
                        assigned_agent_id=chosen.card.agent_id,
                        assigned_card=chosen.card,
                        reason=f"Assigned to {chosen.card.agent_id} ({chosen.reason})",
                    )
                )

        return assignments

    def handle_query(self, message: UACPMessage) -> UACPMessage:
        """Handle incoming capability_query intent through the registry.

        Returns a capability_response UACPMessage, or raises JSONRPCError with code -32002
        if a queried capability is unknown or unavailable.
        """
        payload = CapabilityQueryPayload.model_validate(message.payload)
        queried = payload.query

        if not queried:
            # Return summary of all active capabilities across all alive agents
            active_cards = [
                card.model_dump() for card in self._cards.values() if self.is_alive(card.agent_id)
            ]
            response_payload = CapabilityResponsePayload(capabilities={"agents": active_cards})
        else:
            result_map: dict[str, Any] = {}
            for cap_name in queried:
                ranked = self.find(cap_name)
                if not ranked:
                    raise JSONRPCError(
                        code=JSONRPCErrorCode.CAPABILITY_NOT_FOUND,
                        message=(
                            f"CapabilityNotFound: requested agent capability "
                            f"'{cap_name}' missing or unavailable"
                        ),
                    )
                result_map[cap_name] = [r.card.model_dump() for r in ranked]

            response_payload = CapabilityResponsePayload(capabilities=result_map)

        return UACPMessage(
            correlation_id=message.correlation_id,
            trace_id=message.trace_id,
            sender=Party(agent_id="capability_registry", engine="uacp:system"),
            recipient=message.sender,
            intent=Intent.capability_response,
            payload=response_payload.model_dump(),
            circuit_breaker=message.circuit_breaker.next_hop(),
            created_at=datetime.now(UTC),
        )

    def handle_request(self, request: JSONRPCRequest) -> JSONRPCResponse:
        """Handle JSON-RPC 2.0 request envelope for capability discovery.

        Maps unknown capability errors to JSON-RPC error code -32002.
        """
        try:
            resp_msg = self.handle_query(request.params)
            return build_response(request.id, result=resp_msg.model_dump())
        except JSONRPCError as err:
            return build_error_response(
                request.id, code=err.code, message=err.message, data=err.data
            )

    def handle_response(self, message: UACPMessage) -> list[AgentCard]:
        """Process incoming capability_response intent and return discovered cards."""
        if message.intent != Intent.capability_response:
            return []

        payload = CapabilityResponsePayload.model_validate(message.payload)
        cards: list[AgentCard] = []

        if isinstance(payload.capabilities, dict):
            # Check for direct 'agents' list
            if "agents" in payload.capabilities and isinstance(
                payload.capabilities["agents"], list
            ):
                for item in payload.capabilities["agents"]:
                    if isinstance(item, dict):
                        cards.append(AgentCard.model_validate(item))
            else:
                for agent_list in payload.capabilities.values():
                    if isinstance(agent_list, list):
                        for item in agent_list:
                            if isinstance(item, dict):
                                cards.append(AgentCard.model_validate(item))

        return cards
