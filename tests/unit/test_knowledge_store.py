"""Unit tests for SQLite persistent KnowledgeStore and semantic queries."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pytest
from numpy.typing import NDArray

from uacp.memory.knowledge.embeddings import HashEmbeddings
from uacp.memory.knowledge.store import (
    Evidence,
    KnowledgeStore,
)
from uacp.memory.whiteboard.inmem_backend import InMemoryWhiteboardBackend


@pytest.mark.asyncio
async def test_insert_and_top_k_query(tmp_path: Path) -> None:
    """Verify inserting facts and querying with cosine top-k similarity."""
    db_file = tmp_path / "test_knowledge.db"
    embedder = HashEmbeddings(dim=32)

    async with KnowledgeStore(db_path=db_file, embeddings=embedder) as store:
        ev1 = Evidence(kind="report", ref="doc-1", snippet="Jupiter gas composition")
        fact1, conflicts1 = await store.add_fact(
            subject="Jupiter",
            predicate="is_a",
            obj="gas giant",
            text="Jupiter is the largest gas giant planet in our solar system",
            confidence=0.98,
            source_agent="researcher",
            source_session="session-1",
            evidence=[ev1],
        )
        assert conflicts1 == []
        assert fact1.subject == "jupiter"
        assert fact1.predicate == "is_a"
        assert len(fact1.evidence) == 1

        await store.add_fact(
            subject="Mars",
            predicate="is_a",
            obj="rocky planet",
            text="Mars is a dusty rocky planet with thin atmosphere",
            confidence=0.95,
            source_agent="researcher",
            source_session="session-1",
        )

        # Query exact match for fact 1 text
        results = await store.query(
            "Jupiter is the largest gas giant planet in our solar system", k=2
        )
        assert len(results) == 2
        # Exact text should have similarity ~1.0
        assert results[0].fact.subject == "jupiter"
        assert np.isclose(results[0].score, 1.0, atol=1e-4)
        assert len(results[0].fact.evidence) == 1
        assert results[0].fact.evidence[0].ref == "doc-1"


@pytest.mark.asyncio
async def test_active_view_filtering_and_history_rule(tmp_path: Path) -> None:
    """Verify active_view excludes superseded and disputed facts, but records remain in DB."""
    db_file = tmp_path / "test_history.db"
    embedder = HashEmbeddings(dim=16)

    async with KnowledgeStore(db_path=db_file, embeddings=embedder) as store:
        f1, _ = await store.add_fact("Earth", "orbits", "Sun", "Earth orbits the Sun")
        f2, _ = await store.add_fact("Pluto", "status", "planet", "Pluto is a ninth planet")
        f3, _ = await store.add_fact("Earth", "shape", "flat", "Earth is flat")

        # Supersede Pluto fact
        await store.supersede_fact(f2.id)
        # Dispute Flat Earth fact
        await store.dispute_fact(f3.id)

        # active_view should only return f1
        active = await store.active_view()
        assert len(active) == 1
        assert active[0].id == f1.id
        assert active[0].status == "active"

        # Filter by subject
        active_earth = await store.active_view(subject="Earth")
        assert len(active_earth) == 1
        assert active_earth[0].id == f1.id

        active_mars = await store.active_view(subject="Mars")
        assert len(active_mars) == 0

        # Verify history rule: raw database still retains all 3 facts
        with store._lock:
            cur = store._conn.cursor()
            cur.execute("SELECT count(*) FROM facts")
            total_rows = cur.fetchone()[0]
            assert total_rows == 3

            cur.execute("SELECT status, valid_to FROM facts WHERE id = ?", (f2.id,))
            st, v_to = cur.fetchone()
            assert st == "superseded"
            assert v_to is not None

            cur.execute("SELECT status FROM facts WHERE id = ?", (f3.id,))
            assert cur.fetchone()[0] == "disputed"


@pytest.mark.asyncio
async def test_conflict_candidate_detection(tmp_path: Path) -> None:
    """Verify conflict candidate detection on exact triple mismatch and semantic similarity."""
    db_file = tmp_path / "test_conflicts.db"

    class FixedSimilarityEmbeddings:
        """Embeddings returning controlled vectors to test semantic similarity."""

        def __init__(self) -> None:
            self.v1 = np.array([[1.0, 0.0]], dtype=np.float32)
            self.v2 = np.array([[0.95, 0.31225]], dtype=np.float32)  # dot product ~0.95

        def embed(self, texts: Sequence[str]) -> NDArray[np.float32]:
            if "alpha" in texts[0].lower():
                return np.repeat(self.v1, len(texts), axis=0)
            return np.repeat(self.v2, len(texts), axis=0)

    embedder = FixedSimilarityEmbeddings()

    async with KnowledgeStore(
        db_path=db_file, embeddings=embedder, conflict_threshold=0.85
    ) as store:
        # Add fact 1
        f1, c1 = await store.add_fact(
            subject="Acme",
            predicate="revenue",
            obj="100M",
            text="Acme alpha revenue is 100M",
        )
        assert c1 == []

        # Exact subject + predicate mismatch (different object)
        f2, c2 = await store.add_fact(
            subject="Acme",
            predicate="revenue",
            obj="200M",
            text="Acme alpha revenue is 200M",
        )
        assert len(c2) == 1
        assert c2[0].conflict_type == "subject_predicate_mismatch"
        assert c2[0].existing_fact.id == f1.id
        assert c2[0].existing_fact.object == "100M"

        # Semantic similarity conflict: different predicate/object, but high similarity
        _f3, c3 = await store.add_fact(
            subject="Acme",
            predicate="annual_turnover",
            obj="300M",
            text="Acme turnover beta is 300M",
        )
        # Cosine similarity between v1 and v2 is ~0.95 >= 0.85
        sem_candidates = [c for c in c3 if c.conflict_type == "semantic_similarity"]
        assert len(sem_candidates) >= 1
        assert sem_candidates[0].similarity >= 0.85

        # Identical object should not trigger conflict
        _, c4 = await store.add_fact(
            subject="Acme",
            predicate="revenue",
            obj="200M",
            text="Acme alpha revenue confirmation is 200M",
        )
        # Existing fact f2 has object 200M, so f2 should not be in conflict
        conflicts_with_f2 = [c for c in c4 if c.existing_fact.id == f2.id]
        assert len(conflicts_with_f2) == 0


@pytest.mark.asyncio
async def test_neighbors_graph(tmp_path: Path) -> None:
    """Verify graph view navigation with active facts."""
    db_file = tmp_path / "test_graph.db"
    embedder = HashEmbeddings(dim=16)

    async with KnowledgeStore(db_path=db_file, embeddings=embedder) as store:
        await store.add_fact("AgentAlpha", "delegates_to", "AgentBeta", "Alpha delegates to Beta")
        await store.add_fact("AgentBeta", "delegates_to", "AgentGamma", "Beta delegates to Gamma")
        await store.add_fact(
            "AgentBeta", "audits", "LedgerService", "Beta audits the LedgerService"
        )

        graph_info = await store.neighbors("AgentBeta")
        assert graph_info["subject"] == "agentbeta"
        assert "agentalpha" in [p.lower() for p in graph_info["predecessors"]]
        assert "agentgamma" in [s.lower() for s in graph_info["successors"]]
        assert "ledgerservice" in [s.lower() for s in graph_info["successors"]]
        assert len(graph_info["triples"]) == 3

        # Non-existent node
        empty_info = await store.neighbors("UnknownAgent")
        assert empty_info["neighbors"] == []
        assert empty_info["triples"] == []


@pytest.mark.asyncio
async def test_promote_from_whiteboard(tmp_path: Path) -> None:
    """Verify promote reads reviewed whiteboard keys and stores facts with provenance."""
    db_file = tmp_path / "test_promote.db"
    wb = InMemoryWhiteboardBackend()
    session_id = await wb.create_session(ttl_s=3600)

    # Write two reviewed keys to whiteboard
    await wb.put(
        session_id=session_id,
        key="finding.duplicate_id",
        value={
            "subject": "TX-1001",
            "predicate": "has_flaw",
            "object": "duplicate_id",
            "text": "Transaction TX-1001 exhibits a duplicate ID anomaly",
            "confidence": 0.99,
        },
        author="AgentAlpha",
    )
    await wb.add_review(
        session_id=session_id,
        key="finding.duplicate_id",
        reviewer="AgentBeta",
        verdict="agree",
        notes="Confirmed duplicate entry in ledger batch 2",
    )

    embedder = HashEmbeddings(dim=16)
    async with KnowledgeStore(db_path=db_file, embeddings=embedder) as store:
        promoted = await store.promote(wb, session_id, ["finding.duplicate_id"])
        assert len(promoted) == 1
        p_fact = promoted[0]
        assert p_fact.subject == "tx-1001"
        assert p_fact.predicate == "has_flaw"
        assert p_fact.object == "duplicate_id"
        assert p_fact.source_agent == "AgentAlpha"
        assert p_fact.source_session == session_id
        assert len(p_fact.evidence) == 1
        assert p_fact.evidence[0].kind == "whiteboard"
        assert f"{session_id}/finding.duplicate_id/v1" in p_fact.evidence[0].ref


@pytest.mark.asyncio
async def test_db_persistence_across_reopen(tmp_path: Path) -> None:
    """Verify facts and embeddings persist correctly after closing and reopening the DB."""
    db_file = tmp_path / "persistent_store.db"
    embedder = HashEmbeddings(dim=24)

    # First session: write fact
    store1 = KnowledgeStore(db_path=db_file, embeddings=embedder)
    ev = Evidence(kind="audit", ref="audit-ref-1", snippet="Verified ledger")
    await store1.add_fact(
        subject="VendorA",
        predicate="status",
        obj="approved",
        text="VendorA is fully approved for operations",
        evidence=[ev],
    )
    store1.close()

    # Second session: reopen database and verify persistence
    store2 = KnowledgeStore(db_path=db_file, embeddings=embedder)
    try:
        active = await store2.active_view()
        assert len(active) == 1
        assert active[0].subject == "vendora"
        assert active[0].predicate == "status"
        assert active[0].object == "approved"
        assert len(active[0].evidence) == 1
        assert active[0].evidence[0].ref == "audit-ref-1"

        # Semantic query over reopened store
        results = await store2.query("VendorA is fully approved for operations", k=1)
        assert len(results) == 1
        assert results[0].fact.id == active[0].id
        assert np.isclose(results[0].score, 1.0, atol=1e-4)
    finally:
        store2.close()


@pytest.mark.asyncio
async def test_dimension_validation(tmp_path: Path) -> None:
    """Verify error is raised if stored embedding dimension does not match provider."""
    db_file = tmp_path / "test_dim.db"
    embedder32 = HashEmbeddings(dim=32)
    embedder64 = HashEmbeddings(dim=64)

    store1 = KnowledgeStore(db_path=db_file, embeddings=embedder32)
    await store1.add_fact("Subj", "pred", "obj", "Some fact statement")
    store1.close()

    # Open with 64-dim provider against 32-dim stored vector
    store2 = KnowledgeStore(db_path=db_file, embeddings=embedder64)
    try:
        with pytest.raises(ValueError, match="Stored embedding dimension 32 != query 64"):
            await store2.query("Some query text")
    finally:
        store2.close()
