"""Contradiction resolver for the persistent knowledge store.

SECURITY NOTICE — UNTRUSTED CONTENT:
Fact statements and evidence evaluated by the resolver originate from untrusted
agent actions and whiteboard records. When using an LLM judge, prompt inputs
are bounded and outputs are strictly validated against JSON schema contracts.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, TypeAlias

import numpy as np
import yaml  # type: ignore[import-untyped]
from pydantic import BaseModel, ConfigDict, Field

from uacp.adapters.base import Adapter, CompiledRequest, Message
from uacp.adapters.schema_tools import extract_json
from uacp.memory.knowledge.store import (
    ConflictCandidate,
    ConflictRecord,
    Fact,
    KnowledgeStore,
)

ClockFn: TypeAlias = Callable[[], datetime]


class ResolverConfig(BaseModel):
    """Configuration weights, half-life, thresholds, and priors for contradiction resolution."""

    model_config = ConfigDict(frozen=True)

    w_recency: float = Field(default=0.4, ge=0.0, le=1.0)
    w_certainty: float = Field(default=0.5, ge=0.0, le=1.0)
    w_source: float = Field(default=0.1, ge=0.0, le=1.0)
    half_life_days: float = Field(default=30.0, gt=0.0)
    margin: float = Field(default=0.15, ge=0.0)
    similarity_threshold: float = Field(default=0.85, ge=0.0, le=1.0)
    default_source_prior: float = Field(default=0.5, ge=0.0, le=1.0)
    source_priors: dict[str, float] = Field(default_factory=dict)
    use_llm_judge: bool = Field(default=False)

    @classmethod
    def load_from_yaml(cls, config_path: str | Path | None = None) -> ResolverConfig:
        """Load resolver configuration from config/models.yaml with sensible fallbacks."""
        import os

        candidates: list[Path] = []
        if config_path:
            candidates.append(Path(config_path))
        if env_path := os.environ.get("UACP_MODELS_CONFIG"):
            candidates.append(Path(env_path))
        candidates.append(Path("config/models.yaml"))
        candidates.append(Path(__file__).resolve().parents[4] / "config" / "models.yaml")

        for p in candidates:
            if p.is_file():
                try:
                    with open(p, encoding="utf-8") as f:
                        data = yaml.safe_load(f) or {}
                    if "resolver" in data and isinstance(data["resolver"], dict):
                        return cls(**data["resolver"])
                except Exception:
                    continue

        return cls()


class ConflictResolution(BaseModel):
    """Result of resolving a conflict candidate pair."""

    model_config = ConfigDict(frozen=True)

    conflict_id: str = Field(..., description="Conflict record ID.")
    fact_a: Fact = Field(..., description="First fact.")
    fact_b: Fact = Field(..., description="Second fact.")
    score_a: float = Field(..., description="Calculated resolution score for fact_a.")
    score_b: float = Field(..., description="Calculated resolution score for fact_b.")
    winner_id: str | None = Field(default=None, description="Winning fact ID if decided.")
    loser_id: str | None = Field(default=None, description="Superseded fact ID if decided.")
    outcome: str = Field(
        ...,
        description="'active_superseded', 'disputed', or 'not_contradictory'.",
    )
    method: str = Field(..., description="'scoring', 'scoring+llm_judge', 'llm_judge', 'manual'.")
    resolved_at: str | None = Field(default=None)
    details: str = Field(default="")


class ContradictionResolver:
    """Evaluates fact contradiction pairs via scoring formulas and optional LLM arbitration."""

    def __init__(
        self,
        store: KnowledgeStore,
        config: ResolverConfig | None = None,
        adapter: Adapter | None = None,
        now_fn: Callable[[], datetime] | None = None,
    ) -> None:
        self.store = store
        self.config = config or ResolverConfig.load_from_yaml()
        self.adapter = adapter
        self.now_fn: Callable[[], datetime] = now_fn or (lambda: datetime.now(UTC))

    def compute_recency(self, fact: Fact) -> float:
        """Compute exponential decay recency factor based on half-life in days."""
        fact_dt = datetime.fromisoformat(fact.created_at)
        if fact_dt.tzinfo is None:
            fact_dt = fact_dt.replace(tzinfo=UTC)
        now = self.now_fn()
        if now.tzinfo is None:
            now = now.replace(tzinfo=UTC)

        age_seconds = max(0.0, (now - fact_dt).total_seconds())
        age_days = age_seconds / 86400.0
        return float(0.5 ** (age_days / self.config.half_life_days))

    async def compute_certainty(self, fact: Fact) -> float:
        """Compute certainty score: confidence * mean evidence cosine similarity."""
        if not fact.evidence:
            return float(fact.confidence)

        snippets = [ev.snippet for ev in fact.evidence if ev.snippet.strip()]
        if not snippets:
            return float(fact.confidence)

        fact_vec = await self.store.get_fact_embedding(fact.id)
        if fact_vec is None:
            raw = await asyncio.to_thread(self.store.embeddings.embed, [fact.text])
            fact_vec = raw[0]

        ev_vecs = await asyncio.to_thread(self.store.embeddings.embed, snippets)
        sims = [float(np.dot(fact_vec, ev_v)) for ev_v in ev_vecs]
        mean_sim = float(np.mean(sims))
        mean_sim = max(0.0, min(1.0, mean_sim))
        return float(fact.confidence * mean_sim)

    def compute_source_reliability(self, fact: Fact) -> float:
        """Retrieve source reliability prior for the fact's authoring agent."""
        return float(
            self.config.source_priors.get(
                fact.source_agent,
                self.config.default_source_prior,
            )
        )

    async def compute_score(self, fact: Fact) -> float:
        """Calculate aggregate reliability score for a fact."""
        recency = self.compute_recency(fact)
        certainty = await self.compute_certainty(fact)
        source = self.compute_source_reliability(fact)
        score = (
            self.config.w_recency * recency
            + self.config.w_certainty * certainty
            + self.config.w_source * source
        )
        return float(score)

    async def check_llm_judge(self, fact_a: Fact, fact_b: Fact) -> tuple[bool, str]:
        """Optional LLM arbitration returning (contradicts: bool, reason: str)."""
        if not self.config.use_llm_judge or self.adapter is None:
            return True, "llm_judge_disabled"

        prompt = (
            "You are a factual contradiction judge. Determine whether Statement A and "
            "Statement B contradict each other.\n"
            f"Statement A: {fact_a.text}\n"
            f"Statement B: {fact_b.text}\n\n"
            "Respond ONLY with a JSON object adhering to this schema:\n"
            '{"contradicts": boolean, "reason": "brief string"}'
        )
        req = CompiledRequest(
            provider=self.adapter.name,
            model="judge",
            messages=[Message(role="user", content=prompt)],
            response_schema={
                "type": "object",
                "properties": {
                    "contradicts": {"type": "boolean"},
                    "reason": {"type": "string"},
                },
                "required": ["contradicts", "reason"],
            },
            output_mode="json_mode",
        )
        res = await self.adapter.invoke(req)
        try:
            parsed: dict[str, Any] = extract_json(res.text)
            contradicts = bool(parsed.get("contradicts", True))
            reason = str(parsed.get("reason", ""))
            return contradicts, reason
        except Exception as exc:
            # Fall back safely to proceeding with scoring resolution
            return True, f"Failed to parse LLM judge reply: {exc}"

    async def resolve_pair(self, fact_a_id: str, fact_b_id: str) -> ConflictResolution:
        """Resolve a conflict candidate pair using scoring and state transitions."""
        fact_a = await self.store.get_fact(fact_a_id)
        fact_b = await self.store.get_fact(fact_b_id)
        if fact_a is None or fact_b is None:
            raise ValueError(f"Facts '{fact_a_id}' and/or '{fact_b_id}' not found in store.")

        # Check LLM judge if enabled
        contradicts, judge_reason = await self.check_llm_judge(fact_a, fact_b)
        now_iso = self.now_fn().isoformat()

        if not contradicts:
            cnf = await self.store.record_conflict(
                fact_a=fact_a.id,
                fact_b=fact_b.id,
                score_a=0.0,
                score_b=0.0,
                method="llm_judge",
                resolution="not_contradictory",
                resolved_at=now_iso,
            )
            return ConflictResolution(
                conflict_id=cnf.id,
                fact_a=fact_a,
                fact_b=fact_b,
                score_a=0.0,
                score_b=0.0,
                winner_id=None,
                loser_id=None,
                outcome="not_contradictory",
                method="llm_judge",
                resolved_at=now_iso,
                details=f"LLM judge determined statements do not contradict: {judge_reason}",
            )

        score_a = await self.compute_score(fact_a)
        score_b = await self.compute_score(fact_b)
        diff = score_a - score_b
        method = "scoring+llm_judge" if (self.config.use_llm_judge and self.adapter) else "scoring"

        if abs(diff) >= self.config.margin:
            if score_a > score_b:
                winner, loser = fact_a, fact_b
            else:
                winner, loser = fact_b, fact_a

            await self.store.set_fact_status(winner.id, "active", None)
            await self.store.set_fact_status(loser.id, "superseded", now_iso)
            resolution_str = f"winner:{winner.id};loser:{loser.id}"

            cnf = await self.store.record_conflict(
                fact_a=fact_a.id,
                fact_b=fact_b.id,
                score_a=score_a,
                score_b=score_b,
                method=method,
                resolution=resolution_str,
                resolved_at=now_iso,
            )
            return ConflictResolution(
                conflict_id=cnf.id,
                fact_a=fact_a,
                fact_b=fact_b,
                score_a=score_a,
                score_b=score_b,
                winner_id=winner.id,
                loser_id=loser.id,
                outcome="active_superseded",
                method=method,
                resolved_at=now_iso,
                details=(
                    f"Fact {winner.id} (score {max(score_a, score_b):.3f}) superseded "
                    f"{loser.id} (score {min(score_a, score_b):.3f}) with margin "
                    f"{abs(diff):.3f} >= {self.config.margin}"
                ),
            )
        else:
            await self.store.set_fact_status(fact_a.id, "disputed", None)
            await self.store.set_fact_status(fact_b.id, "disputed", None)
            resolution_str = "disputed"

            cnf = await self.store.record_conflict(
                fact_a=fact_a.id,
                fact_b=fact_b.id,
                score_a=score_a,
                score_b=score_b,
                method=method,
                resolution=resolution_str,
                resolved_at=None,
            )
            return ConflictResolution(
                conflict_id=cnf.id,
                fact_a=fact_a,
                fact_b=fact_b,
                score_a=score_a,
                score_b=score_b,
                winner_id=None,
                loser_id=None,
                outcome="disputed",
                method=method,
                resolved_at=None,
                details=(
                    f"Disputed: score difference {abs(diff):.3f} "
                    f"is within margin {self.config.margin}"
                ),
            )

    async def resolve_all(
        self, candidates: Sequence[ConflictCandidate] | None = None
    ) -> list[ConflictResolution]:
        """Batch-resolve conflict candidates or discover candidate pairs among active facts."""
        results: list[ConflictResolution] = []
        if candidates is not None:
            for cand in candidates:
                res = await self.resolve_pair(cand.fact_id, cand.existing_fact.id)
                results.append(res)
            return results

        active_facts = await self.store.active_view()
        resolved_pairs: set[tuple[str, str]] = set()

        for i, f1 in enumerate(active_facts):
            for f2 in active_facts[i + 1 :]:
                pair_key = (min(f1.id, f2.id), max(f1.id, f2.id))
                if pair_key in resolved_pairs:
                    continue

                same_sp = f1.subject == f2.subject and f1.predicate == f2.predicate
                diff_obj = f1.object != f2.object
                if same_sp and diff_obj:
                    resolved_pairs.add(pair_key)
                    results.append(await self.resolve_pair(f1.id, f2.id))
                    continue

                f1_vec = await self.store.get_fact_embedding(f1.id)
                f2_vec = await self.store.get_fact_embedding(f2.id)
                if f1_vec is not None and f2_vec is not None and diff_obj:
                    sim = float(np.dot(f1_vec, f2_vec))
                    if sim >= self.config.similarity_threshold:
                        resolved_pairs.add(pair_key)
                        results.append(await self.resolve_pair(f1.id, f2.id))

        return results

    async def review_queue(self) -> list[ConflictRecord]:
        """Return all unresolved or disputed conflict records."""
        return await self.store.get_conflicts(unresolved_only=True)

    async def manual_resolve(self, conflict_id: str, winner_fact_id: str) -> ConflictResolution:
        """Manually adjudicate a conflict, setting winner active and loser superseded."""
        cnf = await self.store.get_conflict(conflict_id)
        if cnf is None:
            raise KeyError(f"Conflict '{conflict_id}' not found.")
        if winner_fact_id not in (cnf.fact_a, cnf.fact_b):
            raise ValueError(
                f"Winner '{winner_fact_id}' does not match either conflict fact "
                f"('{cnf.fact_a}', '{cnf.fact_b}')."
            )

        loser_fact_id = cnf.fact_b if winner_fact_id == cnf.fact_a else cnf.fact_a
        now_iso = self.now_fn().isoformat()

        await self.store.set_fact_status(winner_fact_id, "active", None)
        await self.store.set_fact_status(loser_fact_id, "superseded", now_iso)
        resolution_str = f"winner:{winner_fact_id};loser:{loser_fact_id}"

        await self.store.update_conflict(
            conflict_id=conflict_id,
            resolution=resolution_str,
            method="manual",
            resolved_at=now_iso,
        )

        fact_a = await self.store.get_fact(cnf.fact_a)
        fact_b = await self.store.get_fact(cnf.fact_b)
        assert fact_a is not None and fact_b is not None

        return ConflictResolution(
            conflict_id=conflict_id,
            fact_a=fact_a,
            fact_b=fact_b,
            score_a=cnf.score_a,
            score_b=cnf.score_b,
            winner_id=winner_fact_id,
            loser_id=loser_fact_id,
            outcome="active_superseded",
            method="manual",
            resolved_at=now_iso,
            details=f"Manually resolved: {winner_fact_id} declared winner over {loser_fact_id}",
        )
