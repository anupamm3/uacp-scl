"""Unit tests for the UACP Ephemeral Whiteboard system.

Tests cover both InMemoryWhiteboardBackend and SqliteWhiteboardBackend via
parameterization, validating:
- Basic put/get and state versions
- Optimistic concurrency and VersionConflict
- Concurrent writers with asyncio.gather
- Instant TTL expiry using an injectable clock
- Atomic hop counter under concurrent increments
- Cryptographic hash chain verification and tamper detection
- Multi-agent review verdicts and drift reports
- Typed Pydantic v2 WhiteboardClient helpers
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import TYPE_CHECKING

import pytest
from pydantic import BaseModel, Field

from uacp.memory.whiteboard import (
    DriftReport,
    InMemoryWhiteboardBackend,
    ReviewVerdict,
    SessionExpired,
    SqliteWhiteboardBackend,
    VerificationReport,
    VersionConflict,
    WhiteboardBackend,
    WhiteboardClient,
)

if TYPE_CHECKING:
    from collections.abc import AsyncIterator


class MockClock:
    """Controllable clock for fast, deterministic time testing."""

    def __init__(self, initial_time: float = 1000.0) -> None:
        self.current_time = initial_time

    def __call__(self) -> float:
        return self.current_time

    def advance(self, seconds: float) -> None:
        self.current_time += seconds


class FindingArtifact(BaseModel):
    """Sample typed Pydantic artifact for agent testing."""

    anomaly_code: str = Field(..., description="Anomaly classification code.")
    severity: int = Field(..., ge=1, le=5)
    description: str


@pytest.fixture(params=["inmem", "sqlite"])
async def backend(
    request: pytest.FixtureRequest,
    tmp_path: Path,
) -> AsyncIterator[tuple[WhiteboardBackend, MockClock]]:
    """Yield a fresh whiteboard backend instance with an injectable clock."""
    clock = MockClock(1000.0)
    be: WhiteboardBackend
    sqlite_be: SqliteWhiteboardBackend | None = None

    if request.param == "inmem":
        be = InMemoryWhiteboardBackend(clock=clock)
    else:
        db_file = tmp_path / "test_wb.sqlite"
        sqlite_be = SqliteWhiteboardBackend(db_path=db_file, clock=clock)
        be = sqlite_be

    yield be, clock

    if sqlite_be is not None:
        sqlite_be.close_db()


# ── Basic Put / Get & State ──────────────────────────────────────────────


@pytest.mark.asyncio
async def test_basic_put_get(backend: tuple[WhiteboardBackend, MockClock]) -> None:
    wb, _ = backend
    session_id = await wb.create_session(ttl_s=60.0)

    # Initial put
    v1 = await wb.put(session_id, "summary", {"status": "ok"}, author="agent_alpha")
    assert v1 == 1

    # Overwrite put
    v2 = await wb.put(session_id, "summary", {"status": "resolved"}, author="agent_beta")
    assert v2 == 2

    state = await wb.get_state(session_id)
    assert "summary" in state
    entry = state["summary"]
    assert entry["version"] == 2
    assert entry["value"] == {"status": "resolved"}
    assert entry["author"] == "agent_beta"
    assert "updated_at" in entry


# ── Optimistic Concurrency & Version Conflict ───────────────────────────


@pytest.mark.asyncio
async def test_version_conflict(backend: tuple[WhiteboardBackend, MockClock]) -> None:
    wb, _ = backend
    session_id = await wb.create_session(ttl_s=60.0)

    # Create initial key (version 1)
    await wb.put(session_id, "doc", "v1 content", author="alice")

    # Update with expected version 1 -> success (version 2)
    v2 = await wb.put(session_id, "doc", "v2 content", author="bob", expected_version=1)
    assert v2 == 2

    # Attempt update with stale expected version 1 -> raises VersionConflict
    with pytest.raises(VersionConflict) as exc_info:
        await wb.put(session_id, "doc", "v3 stale", author="charlie", expected_version=1)
    assert "Version conflict on key 'doc'" in str(exc_info.value)

    # Attempt creation of non-existent key with non-zero expected version
    with pytest.raises(VersionConflict):
        await wb.put(session_id, "non_existent", "val", author="alice", expected_version=5)


# ── Concurrent Writers ──────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_concurrent_writers_distinct_keys(
    backend: tuple[WhiteboardBackend, MockClock],
) -> None:
    wb, _ = backend
    session_id = await wb.create_session(ttl_s=60.0)

    async def write_item(idx: int) -> int:
        return await wb.put(session_id, f"item_{idx}", f"payload_{idx}", author=f"agent_{idx}")

    results = await asyncio.gather(*(write_item(i) for i in range(15)))
    assert len(results) == 15
    assert all(r == 1 for r in results)

    state = await wb.get_state(session_id)
    assert len(state) == 15
    for i in range(15):
        assert state[f"item_{i}"]["value"] == f"payload_{i}"


# ── TTL Expiry & Session Close ──────────────────────────────────────────


@pytest.mark.asyncio
async def test_ttl_expiry_injectable_clock(
    backend: tuple[WhiteboardBackend, MockClock],
) -> None:
    wb, clock = backend
    session_id = await wb.create_session(ttl_s=10.0)

    await wb.put(session_id, "test_key", "active", author="alice")
    state = await wb.get_state(session_id)
    assert state["test_key"]["value"] == "active"

    # Advance clock past TTL without sleeping
    clock.advance(15.0)

    with pytest.raises(SessionExpired):
        await wb.get_state(session_id)

    with pytest.raises(SessionExpired):
        await wb.put(session_id, "test_key", "new", author="bob")

    with pytest.raises(SessionExpired):
        await wb.incr_hop(session_id)

    with pytest.raises(SessionExpired):
        await wb.add_review(session_id, "test_key", "rev", ReviewVerdict.agree)


@pytest.mark.asyncio
async def test_explicit_session_close(
    backend: tuple[WhiteboardBackend, MockClock],
) -> None:
    wb, _ = backend
    session_id = await wb.create_session(ttl_s=100.0)
    await wb.put(session_id, "k", "v", author="alice")

    await wb.close(session_id)

    with pytest.raises(SessionExpired):
        await wb.get_state(session_id)


# ── Atomic Hop Counter ──────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_atomic_hop_counter(backend: tuple[WhiteboardBackend, MockClock]) -> None:
    wb, _ = backend
    session_id = await wb.create_session(ttl_s=60.0, max_hops=100)

    hops = await asyncio.gather(*(wb.incr_hop(session_id) for _ in range(25)))
    assert len(hops) == 25
    assert sorted(hops) == list(range(1, 26))


# ── Hash Chain Verification & Tamper Detection ──────────────────────────


@pytest.mark.asyncio
async def test_hash_chain_verify_valid(backend: tuple[WhiteboardBackend, MockClock]) -> None:
    wb, _ = backend
    session_id = await wb.create_session(ttl_s=60.0)

    await wb.put(session_id, "step1", {"data": 1}, author="alpha")
    await wb.put(session_id, "step2", {"data": 2}, author="beta")
    await wb.put(session_id, "step1", {"data": 3}, author="gamma")

    report: VerificationReport = await wb.verify_chain(session_id)
    assert report.is_valid is True
    assert report.events_verified == 3
    assert report.corrupted_event_id is None


@pytest.mark.asyncio
async def test_hash_chain_detect_event_tampering(
    backend: tuple[WhiteboardBackend, MockClock],
) -> None:
    wb, _ = backend
    session_id = await wb.create_session(ttl_s=60.0)

    await wb.put(session_id, "k1", "original_1", author="alpha")
    await wb.put(session_id, "k2", "original_2", author="beta")

    # Manually tamper with the hash of the first event out of band
    if isinstance(wb, InMemoryWhiteboardBackend):
        session = wb._get_session(session_id)
        # Corrupt the hash of the first event
        tampered_event = session.events[0].model_copy(update={"hash": "deadbeef" * 8})
        session.events[0] = tampered_event
    elif isinstance(wb, SqliteWhiteboardBackend):
        with wb._lock, wb._conn:
            wb._conn.execute(
                "UPDATE events SET hash = ? WHERE session_id = ? AND event_id = 1",
                ("deadbeef" * 8, session_id),
            )

    report = await wb.verify_chain(session_id)
    assert report.is_valid is False
    assert report.corrupted_event_id is not None


@pytest.mark.asyncio
async def test_hash_chain_detect_state_tampering(
    backend: tuple[WhiteboardBackend, MockClock],
) -> None:
    wb, _ = backend
    session_id = await wb.create_session(ttl_s=60.0)

    await wb.put(session_id, "k1", "original_1", author="alpha")

    # Tamper with current state out of band without generating an event
    if isinstance(wb, InMemoryWhiteboardBackend):
        session = wb._get_session(session_id)
        session.state["k1"] = session.state["k1"].model_copy(update={"value": "tampered_value"})
    elif isinstance(wb, SqliteWhiteboardBackend):
        with wb._lock, wb._conn:
            wb._conn.execute(
                "UPDATE state SET value = ? WHERE session_id = ? AND key = 'k1'",
                ('"tampered_value"', session_id),
            )

    report = await wb.verify_chain(session_id)
    assert report.is_valid is False
    assert "state tampering detected" in report.details


# ── Drift Report & Reviews ──────────────────────────────────────────────


@pytest.mark.asyncio
async def test_drift_report(backend: tuple[WhiteboardBackend, MockClock]) -> None:
    wb, _ = backend
    session_id = await wb.create_session(ttl_s=60.0)

    await wb.put(session_id, "finding_1", "anomalous entry", author="auditor")

    await wb.add_review(
        session_id, "finding_1", "agent_a", ReviewVerdict.agree, notes="Looks solid"
    )
    await wb.add_review(
        session_id, "finding_1", "agent_b", ReviewVerdict.disagree, notes="False positive"
    )
    await wb.add_review(
        session_id, "finding_1", "agent_c", ReviewVerdict.needs_info, notes="Need batch id"
    )

    report: DriftReport = await wb.drift_report(session_id)
    assert report.session_id == session_id
    assert report.total_reviews == 3
    assert report.has_drift is True
    assert "finding_1" in report.keys

    summary = report.keys["finding_1"]
    assert summary.agree_count == 1
    assert summary.disagree_count == 1
    assert summary.needs_info_count == 1
    assert summary.has_drift is True
    assert "Looks solid" in summary.notes
    assert "False positive" in summary.notes


# ── WhiteboardClient Typed Helpers ──────────────────────────────────────


@pytest.mark.asyncio
async def test_whiteboard_client_typed_models(
    backend: tuple[WhiteboardBackend, MockClock],
) -> None:
    wb, _ = backend
    session_id = await wb.create_session(ttl_s=60.0)
    client = WhiteboardClient(backend=wb, session_id=session_id, author="agent_alpha")

    model_in = FindingArtifact(
        anomaly_code="ERR-402",
        severity=4,
        description="Duplicate invoice detected across batch 3 and 4.",
    )

    ver = await client.write_model("findings/402", model_in)
    assert ver == 1

    # Read back typed model
    model_out = await client.read_model("findings/402", FindingArtifact)
    assert model_out == model_in
    assert model_out.anomaly_code == "ERR-402"
    assert model_out.severity == 4

    # Read all state
    all_state = await client.read_all()
    assert "findings/402" in all_state
    assert all_state["findings/402"]["value"]["severity"] == 4

    # Key not found raises KeyError
    with pytest.raises(KeyError):
        await client.read_model("findings/missing", FindingArtifact)


# ── Chronological Event Retrieval ───────────────────────────────────────


@pytest.mark.asyncio
async def test_events_retrieval(backend: tuple[WhiteboardBackend, MockClock]) -> None:
    wb, _ = backend
    session_id = await wb.create_session(ttl_s=60.0)

    await wb.put(session_id, "k1", 100, author="a")
    await wb.put(session_id, "k2", 200, author="b")

    events = await wb.events(session_id)
    assert len(events) == 2
    first_id = int(events[0].event_id)

    filtered = await wb.events(session_id, since=first_id)
    assert len(filtered) == 1
    assert int(filtered[0].event_id) > first_id
