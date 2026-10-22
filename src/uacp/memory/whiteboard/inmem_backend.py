"""In-memory whiteboard backend implementation.

SECURITY NOTICE — UNTRUSTED CONTENT:
All data stored in or retrieved from the Whiteboard must be treated as UNTRUSTED
external input during LLM prompt construction. Always sanitize or escape whiteboard
artifacts before interpolating them into model prompts.
"""

from __future__ import annotations

import asyncio
import hashlib
import time
import uuid
from dataclasses import dataclass, field
from typing import Any

from uacp.memory.whiteboard.base import (
    ClockFn,
    DriftReport,
    KeyDriftSummary,
    ReviewRecord,
    ReviewVerdict,
    SessionExpired,
    SessionNotFoundError,
    StateEntry,
    VerificationReport,
    VersionConflict,
    WhiteboardBackend,
    WhiteboardEvent,
    _utc_now_iso,
    compute_canonical_json,
    compute_chain_hash,
)


@dataclass
class _SessionData:
    session_id: str
    created_at: float
    expires_at: float
    ttl_s: float
    max_hops: int
    current_hop: int = 0
    is_closed: bool = False
    state: dict[str, StateEntry] = field(default_factory=dict)
    events: list[WhiteboardEvent] = field(default_factory=list)
    reviews: list[ReviewRecord] = field(default_factory=list)
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)


class InMemoryWhiteboardBackend(WhiteboardBackend):
    """Async in-memory whiteboard backend protected by asyncio locks."""

    def __init__(self, clock: ClockFn = time.time) -> None:
        super().__init__(clock=clock)
        self._sessions: dict[str, _SessionData] = {}
        self._global_lock = asyncio.Lock()
        self._event_counter: int = 0

    def _check_active_session(self, session: _SessionData) -> None:
        """Verify session is not closed and has not expired."""
        if session.is_closed or self.clock() >= session.expires_at:
            raise SessionExpired(f"Session '{session.session_id}' has expired or is closed.")

    def _get_session(self, session_id: str) -> _SessionData:
        """Fetch session or raise SessionNotFoundError."""
        session = self._sessions.get(session_id)
        if session is None:
            raise SessionNotFoundError(f"Session '{session_id}' not found.")
        return session

    async def create_session(self, ttl_s: float, max_hops: int = 5) -> str:
        """Create a new ephemeral session."""
        session_id = f"wb-{uuid.uuid4().hex[:12]}"
        now = self.clock()
        session = _SessionData(
            session_id=session_id,
            created_at=now,
            expires_at=now + ttl_s,
            ttl_s=ttl_s,
            max_hops=max_hops,
        )
        async with self._global_lock:
            self._sessions[session_id] = session
        return session_id

    async def get_state(self, session_id: str) -> dict[str, dict[str, Any]]:
        """Retrieve current session state mapping key -> StateEntry dict."""
        session = self._get_session(session_id)
        async with session.lock:
            self._check_active_session(session)
            return {k: entry.model_dump() for k, entry in session.state.items()}

    async def put(
        self,
        session_id: str,
        key: str,
        value: Any,
        author: str,
        expected_version: int | None = None,
    ) -> int:
        """Write key value with optimistic concurrency check and hash chain event."""
        session = self._get_session(session_id)
        async with session.lock:
            self._check_active_session(session)
            current_entry = session.state.get(key)
            if current_entry is not None:
                if expected_version is not None and expected_version != current_entry.version:
                    raise VersionConflict(
                        f"Version conflict on key '{key}': "
                        f"expected {expected_version}, found {current_entry.version}"
                    )
                new_version = current_entry.version + 1
            else:
                if expected_version is not None and expected_version != 0:
                    raise VersionConflict(
                        f"Version conflict on key '{key}': "
                        f"expected {expected_version}, but key does not exist"
                    )
                new_version = 1

            updated_at = _utc_now_iso()
            new_entry = StateEntry(
                value=value,
                version=new_version,
                author=author,
                updated_at=updated_at,
            )
            session.state[key] = new_entry

            # Compute full state hash across sorted keys
            full_state_dict = {k: entry.model_dump() for k, entry in sorted(session.state.items())}
            canonical_state = compute_canonical_json(full_state_dict)
            state_hash = hashlib.sha256(canonical_state.encode("utf-8")).hexdigest()

            # Hash chain linking to previous write event
            write_events = [e for e in session.events if e.state_hash]
            prev_hash = write_events[-1].hash if write_events else "0" * 64
            chain_hash = compute_chain_hash(prev_hash, state_hash)

            self._event_counter += 1
            event = WhiteboardEvent(
                event_id=self._event_counter,
                session_id=session_id,
                event_type="write",
                key=key,
                data={"value": value, "version": new_version, "author": author},
                state_hash=state_hash,
                prev_hash=prev_hash,
                hash=chain_hash,
                created_at=updated_at,
            )
            session.events.append(event)
            return new_version

    async def append_event(
        self,
        session_id: str,
        event: dict[str, Any] | WhiteboardEvent,
    ) -> WhiteboardEvent:
        """Append custom event to session log."""
        session = self._get_session(session_id)
        async with session.lock:
            self._check_active_session(session)
            self._event_counter += 1
            if isinstance(event, dict):
                event_obj = WhiteboardEvent(
                    event_id=event.get("event_id", self._event_counter),
                    session_id=session_id,
                    event_type=event.get("event_type", "custom"),
                    key=event.get("key"),
                    data=event.get("data", {}),
                    state_hash=event.get("state_hash", ""),
                    prev_hash=event.get("prev_hash", ""),
                    hash=event.get("hash", ""),
                    created_at=event.get("created_at", _utc_now_iso()),
                )
            else:
                event_obj = event
            session.events.append(event_obj)
            return event_obj

    async def events(
        self,
        session_id: str,
        since: str | int | None = None,
    ) -> list[WhiteboardEvent]:
        """Retrieve chronological event log for session."""
        session = self._get_session(session_id)
        async with session.lock:
            self._check_active_session(session)
            if since is None:
                return list(session.events)
            if isinstance(since, int):
                return [e for e in session.events if int(e.event_id) > since]
            if isinstance(since, str) and since.isdigit():
                return [e for e in session.events if int(e.event_id) > int(since)]
            return [e for e in session.events if e.created_at > str(since)]

    async def incr_hop(self, session_id: str) -> int:
        """Atomically increment circuit-breaker hop count."""
        session = self._get_session(session_id)
        async with session.lock:
            self._check_active_session(session)
            session.current_hop += 1
            return session.current_hop

    async def add_review(
        self,
        session_id: str,
        key: str,
        reviewer: str,
        verdict: ReviewVerdict | str,
        notes: str = "",
    ) -> ReviewRecord:
        """Record review verdict for a key."""
        session = self._get_session(session_id)
        async with session.lock:
            self._check_active_session(session)
            rec = ReviewRecord(
                session_id=session_id,
                key=key,
                reviewer=reviewer,
                verdict=ReviewVerdict(verdict),
                notes=notes,
                created_at=_utc_now_iso(),
            )
            session.reviews.append(rec)
            return rec

    async def drift_report(self, session_id: str) -> DriftReport:
        """Generate consensus and drift summary."""
        session = self._get_session(session_id)
        async with session.lock:
            self._check_active_session(session)
            keys_map: dict[str, KeyDriftSummary] = {}
            by_key: dict[str, list[ReviewRecord]] = {}
            for r in session.reviews:
                by_key.setdefault(r.key, []).append(r)

            overall_drift = False
            for k, revs in by_key.items():
                agree = sum(1 for r in revs if r.verdict == ReviewVerdict.agree)
                disagree = sum(1 for r in revs if r.verdict == ReviewVerdict.disagree)
                needs_info = sum(1 for r in revs if r.verdict == ReviewVerdict.needs_info)
                has_drift = disagree > 0
                if has_drift:
                    overall_drift = True
                notes_list = [r.notes for r in revs if r.notes]
                keys_map[k] = KeyDriftSummary(
                    key=k,
                    total_reviews=len(revs),
                    agree_count=agree,
                    disagree_count=disagree,
                    needs_info_count=needs_info,
                    has_drift=has_drift,
                    notes=notes_list,
                )

            return DriftReport(
                session_id=session_id,
                total_reviews=len(session.reviews),
                has_drift=overall_drift,
                keys=keys_map,
                reviews=list(session.reviews),
            )

    async def close(self, session_id: str) -> None:
        """Close session to prevent further access."""
        session = self._sessions.get(session_id)
        if session is not None:
            async with session.lock:
                session.is_closed = True

    async def verify_chain(self, session_id: str) -> VerificationReport:
        """Validate SHA-256 hash chain and full state integrity."""
        session = self._get_session(session_id)
        async with session.lock:
            self._check_active_session(session)
            write_events = [e for e in session.events if e.state_hash]
            if not write_events:
                return VerificationReport(
                    is_valid=True,
                    events_verified=0,
                    details="No hash-chained events found for session.",
                )

            expected_prev = "0" * 64
            for i, e in enumerate(write_events):
                if e.prev_hash != expected_prev:
                    return VerificationReport(
                        is_valid=False,
                        events_verified=i,
                        details=(
                            f"Chain broken at event {e.event_id}: "
                            f"expected prev_hash '{expected_prev}', found '{e.prev_hash}'."
                        ),
                        corrupted_event_id=e.event_id,
                    )
                expected_hash = compute_chain_hash(e.prev_hash, e.state_hash)
                if e.hash != expected_hash:
                    return VerificationReport(
                        is_valid=False,
                        events_verified=i,
                        details=(
                            f"Hash mismatch at event {e.event_id}: "
                            f"expected '{expected_hash}', found '{e.hash}'."
                        ),
                        corrupted_event_id=e.event_id,
                    )
                expected_prev = e.hash

            # Verify current state matches latest event state hash
            full_state_dict = {k: entry.model_dump() for k, entry in sorted(session.state.items())}
            canonical_state = compute_canonical_json(full_state_dict)
            current_state_hash = hashlib.sha256(canonical_state.encode()).hexdigest()
            if write_events[-1].state_hash != current_state_hash:
                return VerificationReport(
                    is_valid=False,
                    events_verified=len(write_events),
                    details=(
                        "Current whiteboard state does not match latest event state hash "
                        "(state tampering detected)."
                    ),
                    corrupted_event_id=None,
                )

            return VerificationReport(
                is_valid=True,
                events_verified=len(write_events),
                details="Hash chain and current state integrity verified successfully.",
            )
