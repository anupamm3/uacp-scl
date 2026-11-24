"""Unit tests for contradiction resolution, scoring, state transitions, and review queue."""

from __future__ import annotations

import json
from collections.abc import Sequence
from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import pytest
from numpy.typing import NDArray

from uacp.adapters.fake_adapter import FakeAdapter
from uacp.memory.knowledge.embeddings import HashEmbeddings
from uacp.memory.knowledge.resolver import (
    ContradictionResolver,
    ResolverConfig,
)
from uacp.memory.knowledge.store import Evidence, KnowledgeStore


@pytest.fixture
def mock_clock() -> tuple[datetime, datetime]:
    """Provide fixed timestamps for testing."""
    now = datetime(2026, 11, 24, 12, 0, 0, tzinfo=UTC)
    ten_days_ago = now - timedelta(days=10)
    return now, ten_days_ago


@pytest.mark.asyncio
async def test_newer_weak_vs_older_strong(
    tmp_path: Path, mock_clock: tuple[datetime, datetime]
) -> None:
    """Verify an older fact with strong evidence and reliability beats a weak newer fact."""
    now, ten_days_ago = mock_clock
    db_file = tmp_path / "test_strong_weak.db"
    embedder = HashEmbeddings(dim=32)

    config = ResolverConfig(
        w_recency=0.2,
        w_certainty=0.5,
        w_source=0.3,
        half_life_days=30.0,
        margin=0.15,
        default_source_prior=0.5,
        source_priors={"TrustedAgent": 0.95, "UntrustedAgent": 0.1},
    )

    async with KnowledgeStore(db_path=db_file, embeddings=embedder) as store:
        resolver = ContradictionResolver(store=store, config=config, now_fn=lambda: now)

        # Older fact: Trusted source, high confidence, strong evidence
        ev = Evidence(kind="report", ref="doc-1", snippet="Acme Corp verified revenue 100M")
        f_old, _ = await store.add_fact(
            subject="Acme",
            predicate="revenue",
            obj="100M",
            text="Acme Corp verified revenue 100M",
            confidence=0.99,
            source_agent="TrustedAgent",
            source_session="s-1",
            evidence=[ev],
            valid_from=ten_days_ago.isoformat(),
        )
        # Update created_at to ten_days_ago for recency decay testing
        with store._lock, store._conn:
            store._conn.execute(
                "UPDATE facts SET created_at = ? WHERE id = ?",
                (ten_days_ago.isoformat(), f_old.id),
            )

        # Newer fact: Untrusted source, low confidence, no evidence, created now
        f_new, _ = await store.add_fact(
            subject="Acme",
            predicate="revenue",
            obj="999M",
            text="Acme Corp rumors of revenue 999M",
            confidence=0.2,
            source_agent="UntrustedAgent",
            source_session="s-2",
            valid_from=now.isoformat(),
        )

        res = await resolver.resolve_pair(f_old.id, f_new.id)
        assert res.outcome == "active_superseded"
        assert res.winner_id == f_old.id
        assert res.loser_id == f_new.id
        assert res.score_a > res.score_b

        # Check DB states
        f_old_updated = await store.get_fact(f_old.id)
        f_new_updated = await store.get_fact(f_new.id)
        assert f_old_updated is not None and f_old_updated.status == "active"
        assert f_new_updated is not None and f_new_updated.status == "superseded"
        assert f_new_updated.valid_to is not None


@pytest.mark.asyncio
async def test_tie_within_margin_goes_to_disputed(
    tmp_path: Path, mock_clock: tuple[datetime, datetime]
) -> None:
    """Verify close scores within margin transition both facts to disputed in review queue."""
    now, _ = mock_clock
    db_file = tmp_path / "test_tie.db"
    embedder = HashEmbeddings(dim=32)

    config = ResolverConfig(
        w_recency=0.4,
        w_certainty=0.4,
        w_source=0.2,
        margin=0.15,
        default_source_prior=0.5,
    )

    async with KnowledgeStore(db_path=db_file, embeddings=embedder) as store:
        resolver = ContradictionResolver(store=store, config=config, now_fn=lambda: now)

        f1, _ = await store.add_fact(
            subject="ProjectX",
            predicate="deadline",
            obj="2027-01-01",
            text="ProjectX deadline is 2027-01-01",
            confidence=0.85,
            source_agent="Agent1",
            source_session="s-1",
        )
        f2, _ = await store.add_fact(
            subject="ProjectX",
            predicate="deadline",
            obj="2027-06-01",
            text="ProjectX deadline is 2027-06-01",
            confidence=0.86,
            source_agent="Agent2",
            source_session="s-2",
        )

        res = await resolver.resolve_pair(f1.id, f2.id)
        assert res.outcome == "disputed"
        assert res.winner_id is None
        assert res.loser_id is None

        # Both facts are disputed
        f1_u = await store.get_fact(f1.id)
        f2_u = await store.get_fact(f2.id)
        assert f1_u is not None and f1_u.status == "disputed"
        assert f2_u is not None and f2_u.status == "disputed"

        # Neither in active_view
        active = await store.active_view()
        assert len(active) == 0

        # In review queue
        queue = await resolver.review_queue()
        assert len(queue) == 1
        assert queue[0].resolution == "disputed"


@pytest.mark.asyncio
async def test_same_source_update_newer_supersedes_older(
    tmp_path: Path, mock_clock: tuple[datetime, datetime]
) -> None:
    """Verify an agent updating its own assertion allows newer fact to supersede older."""
    now, _ = mock_clock
    sixty_days_ago = now - timedelta(days=60)
    db_file = tmp_path / "test_same_source.db"
    embedder = HashEmbeddings(dim=32)

    config = ResolverConfig(
        w_recency=0.6,
        w_certainty=0.3,
        w_source=0.1,
        half_life_days=30.0,
        margin=0.15,
    )

    async with KnowledgeStore(db_path=db_file, embeddings=embedder) as store:
        resolver = ContradictionResolver(store=store, config=config, now_fn=lambda: now)

        f_old, _ = await store.add_fact(
            subject="Server",
            predicate="ip",
            obj="10.0.0.1",
            text="Server IP is 10.0.0.1",
            confidence=0.9,
            source_agent="SysAgent",
            source_session="s-old",
        )
        with store._lock, store._conn:
            store._conn.execute(
                "UPDATE facts SET created_at = ? WHERE id = ?",
                (sixty_days_ago.isoformat(), f_old.id),
            )

        f_new, _ = await store.add_fact(
            subject="Server",
            predicate="ip",
            obj="10.0.0.2",
            text="Server IP is 10.0.0.2",
            confidence=0.9,
            source_agent="SysAgent",
            source_session="s-new",
        )

        res = await resolver.resolve_pair(f_old.id, f_new.id)
        assert res.outcome == "active_superseded"
        assert res.winner_id == f_new.id
        assert res.loser_id == f_old.id


@pytest.mark.asyncio
async def test_facts_with_no_evidence_fallback(
    tmp_path: Path, mock_clock: tuple[datetime, datetime]
) -> None:
    """Verify facts with no evidence fallback to confidence for certainty calculation."""
    now, _ = mock_clock
    db_file = tmp_path / "test_no_evidence.db"
    embedder = HashEmbeddings(dim=32)

    async with KnowledgeStore(db_path=db_file, embeddings=embedder) as store:
        resolver = ContradictionResolver(store=store, now_fn=lambda: now)

        f, _ = await store.add_fact(
            subject="Fact1",
            predicate="value",
            obj="100",
            text="Fact1 value is 100",
            confidence=0.77,
            source_agent="A",
            source_session="s",
        )
        certainty = await resolver.compute_certainty(f)
        assert np.isclose(certainty, 0.77)


@pytest.mark.asyncio
async def test_evidence_backed_older_fact_beats_unsupported_newer_fact(
    tmp_path: Path, mock_clock: tuple[datetime, datetime]
) -> None:
    """Verify evidence similarity lifts certainty allowing older fact to win."""
    now, _ = mock_clock
    db_file = tmp_path / "test_evidence_win.db"
    embedder = HashEmbeddings(dim=32)

    config = ResolverConfig(
        w_recency=0.2,
        w_certainty=0.7,
        w_source=0.1,
        margin=0.10,
    )

    async with KnowledgeStore(db_path=db_file, embeddings=embedder) as store:
        resolver = ContradictionResolver(store=store, config=config, now_fn=lambda: now)

        ev = Evidence(kind="doc", ref="r1", snippet="Quantum speedup achieved 100x")
        f1, _ = await store.add_fact(
            subject="Quantum",
            predicate="speedup",
            obj="100x",
            text="Quantum speedup achieved 100x",
            confidence=1.0,
            evidence=[ev],
        )

        f2, _ = await store.add_fact(
            subject="Quantum",
            predicate="speedup",
            obj="10x",
            text="Quantum speedup estimated 10x",
            confidence=0.3,
            evidence=[],
        )

        res = await resolver.resolve_pair(f1.id, f2.id)
        assert res.winner_id == f1.id
        assert res.loser_id == f2.id


@pytest.mark.asyncio
async def test_config_overrides_flip_winner(
    tmp_path: Path, mock_clock: tuple[datetime, datetime]
) -> None:
    """Verify changing weights and source priors flips the winning fact."""
    now, _ = mock_clock
    ten_days_ago = now - timedelta(days=10)
    db_file = tmp_path / "test_config_flip.db"
    embedder = HashEmbeddings(dim=32)

    async with KnowledgeStore(db_path=db_file, embeddings=embedder) as store:
        # Fact A: Trusted source, created 10 days ago, confidence 0.7
        f_a, _ = await store.add_fact(
            subject="Target",
            predicate="state",
            obj="A",
            text="Target is in state A",
            confidence=0.7,
            source_agent="ExpertAgent",
        )
        with store._lock, store._conn:
            store._conn.execute(
                "UPDATE facts SET created_at = ? WHERE id = ?",
                (ten_days_ago.isoformat(), f_a.id),
            )

        # Fact B: Unknown source, created today, confidence 0.8
        f_b, _ = await store.add_fact(
            subject="Target",
            predicate="state",
            obj="B",
            text="Target is in state B",
            confidence=0.8,
            source_agent="UnknownAgent",
        )

        # Config 1: Recency heavy -> Fact B wins
        config_recency = ResolverConfig(
            w_recency=0.8,
            w_certainty=0.1,
            w_source=0.1,
            margin=0.05,
            default_source_prior=0.5,
            source_priors={"ExpertAgent": 0.9, "UnknownAgent": 0.2},
        )
        resolver_recency = ContradictionResolver(
            store=store, config=config_recency, now_fn=lambda: now
        )
        res1 = await resolver_recency.resolve_pair(f_a.id, f_b.id)
        assert res1.winner_id == f_b.id

        # Config 2: Source heavy -> Fact A wins
        config_source = ResolverConfig(
            w_recency=0.1,
            w_certainty=0.1,
            w_source=0.8,
            margin=0.05,
            default_source_prior=0.5,
            source_priors={"ExpertAgent": 0.95, "UnknownAgent": 0.1},
        )
        resolver_source = ContradictionResolver(
            store=store, config=config_source, now_fn=lambda: now
        )
        res2 = await resolver_source.resolve_pair(f_a.id, f_b.id)
        assert res2.winner_id == f_a.id


@pytest.mark.asyncio
async def test_manual_resolve_flow(tmp_path: Path, mock_clock: tuple[datetime, datetime]) -> None:
    """Verify manual resolution promotes winner, supersedes loser, and clears review queue."""
    now, _ = mock_clock
    db_file = tmp_path / "test_manual.db"
    embedder = HashEmbeddings(dim=32)

    config = ResolverConfig(margin=0.5)  # High margin forces dispute

    async with KnowledgeStore(db_path=db_file, embeddings=embedder) as store:
        resolver = ContradictionResolver(store=store, config=config, now_fn=lambda: now)

        f1, _ = await store.add_fact("City", "mayor", "Alice", "Alice is mayor")
        f2, _ = await store.add_fact("City", "mayor", "Bob", "Bob is mayor")

        res_dispute = await resolver.resolve_pair(f1.id, f2.id)
        assert res_dispute.outcome == "disputed"

        queue_before = await resolver.review_queue()
        assert len(queue_before) == 1
        conflict_id = queue_before[0].id

        # Manually declare f2 (Bob) as winner
        manual_res = await resolver.manual_resolve(conflict_id, f2.id)
        assert manual_res.winner_id == f2.id
        assert manual_res.loser_id == f1.id
        assert manual_res.method == "manual"

        # Check DB fact statuses
        f1_u = await store.get_fact(f1.id)
        f2_u = await store.get_fact(f2.id)
        assert f1_u is not None and f1_u.status == "superseded"
        assert f2_u is not None and f2_u.status == "active"

        # Review queue is now empty
        queue_after = await resolver.review_queue()
        assert len(queue_after) == 0


@pytest.mark.asyncio
async def test_llm_judge_contradicts_false(
    tmp_path: Path, mock_clock: tuple[datetime, datetime]
) -> None:
    """Verify LLM judge returning contradicts=False preserves both facts as active."""
    now, _ = mock_clock
    db_file = tmp_path / "test_llm_false.db"
    embedder = HashEmbeddings(dim=32)

    judge_reply = json.dumps(
        {"contradicts": False, "reason": "Statements describe different aspects"}
    )
    fake_adapter = FakeAdapter(script=[judge_reply])
    config = ResolverConfig(use_llm_judge=True)

    async with KnowledgeStore(db_path=db_file, embeddings=embedder) as store:
        resolver = ContradictionResolver(
            store=store, config=config, adapter=fake_adapter, now_fn=lambda: now
        )

        f1, _ = await store.add_fact(
            "Service", "port", "8080", "Service listens on HTTP port 8080"
        )
        f2, _ = await store.add_fact(
            "Service", "port", "8443", "Service listens on HTTPS port 8443"
        )

        res = await resolver.resolve_pair(f1.id, f2.id)
        assert res.outcome == "not_contradictory"
        assert res.method == "llm_judge"

        # Both remain active
        f1_u = await store.get_fact(f1.id)
        f2_u = await store.get_fact(f2.id)
        assert f1_u is not None and f1_u.status == "active"
        assert f2_u is not None and f2_u.status == "active"


@pytest.mark.asyncio
async def test_llm_judge_contradicts_true(
    tmp_path: Path, mock_clock: tuple[datetime, datetime]
) -> None:
    """Verify LLM judge confirming contradiction triggers scoring+llm_judge resolution."""
    now, _ = mock_clock
    db_file = tmp_path / "test_llm_true.db"
    embedder = HashEmbeddings(dim=32)

    fake_adapter = FakeAdapter(
        script=[json.dumps({"contradicts": True, "reason": "Mutual exclusion on primary key"})]
    )
    config = ResolverConfig(use_llm_judge=True, margin=0.1)

    async with KnowledgeStore(db_path=db_file, embeddings=embedder) as store:
        resolver = ContradictionResolver(
            store=store, config=config, adapter=fake_adapter, now_fn=lambda: now
        )

        f1, _ = await store.add_fact(
            "Device", "status", "online", "Device is online", confidence=0.99
        )
        f2, _ = await store.add_fact(
            "Device", "status", "offline", "Device is offline", confidence=0.1
        )

        res = await resolver.resolve_pair(f1.id, f2.id)
        assert res.outcome == "active_superseded"
        assert res.method == "scoring+llm_judge"
        assert res.winner_id == f1.id
        assert res.loser_id == f2.id


@pytest.mark.asyncio
async def test_high_similarity_candidate_detection(tmp_path: Path) -> None:
    """Verify resolve_all detects and resolves conflict on high semantic similarity."""
    db_file = tmp_path / "test_high_sim.db"

    class HighSimEmbeddings:
        """Deterministic embeddings returning unit vectors with controlled dot product."""

        def embed(self, texts: Sequence[str]) -> NDArray[np.float32]:
            # Return same vector regardless of text so similarity is 1.0 >= 0.85
            vec = np.array([[1.0, 0.0]], dtype=np.float32)
            return np.repeat(vec, len(texts), axis=0)

    embedder = HighSimEmbeddings()
    config = ResolverConfig(similarity_threshold=0.85, margin=0.1)

    async with KnowledgeStore(db_path=db_file, embeddings=embedder) as store:
        resolver = ContradictionResolver(store=store, config=config)

        f1, _ = await store.add_fact(
            subject="Co",
            predicate="headcount",
            obj="500",
            text="Co headcount is 500",
            confidence=0.95,
        )
        f2, _ = await store.add_fact(
            subject="Co",
            predicate="employee_count",
            obj="1200",
            text="Co total employees numbered 1200",
            confidence=0.2,
        )

        resolutions = await resolver.resolve_all()
        assert len(resolutions) >= 1
        res = resolutions[0]
        assert res.outcome == "active_superseded"
        assert res.winner_id == f1.id
        assert res.loser_id == f2.id


@pytest.mark.asyncio
async def test_history_preservation(tmp_path: Path, mock_clock: tuple[datetime, datetime]) -> None:
    """Verify superseded facts are never deleted from SQLite and retain valid_to timestamp."""
    now, _ = mock_clock
    db_file = tmp_path / "test_history.db"
    embedder = HashEmbeddings(dim=32)

    config = ResolverConfig(margin=0.05)

    async with KnowledgeStore(db_path=db_file, embeddings=embedder) as store:
        resolver = ContradictionResolver(store=store, config=config, now_fn=lambda: now)

        f_win, _ = await store.add_fact(
            "Account", "balance", "$5000", "Balance is $5000", confidence=0.95
        )
        f_lose, _ = await store.add_fact(
            "Account", "balance", "$100", "Balance is $100", confidence=0.10
        )

        await resolver.resolve_pair(f_win.id, f_lose.id)

        # active_view excludes f_lose
        active = await store.active_view()
        assert len(active) == 1
        assert active[0].id == f_win.id

        # Raw SQLite check confirms f_lose exists with status='superseded' and valid_to
        with store._lock:
            cur = store._conn.cursor()
            cur.execute("SELECT status, valid_to FROM facts WHERE id = ?", (f_lose.id,))
            row = cur.fetchone()
            assert row is not None
            assert row[0] == "superseded"
            assert row[1] == now.isoformat()
