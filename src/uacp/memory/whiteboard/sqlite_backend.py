"""SQLite-backed whiteboard backend implementation.

SECURITY NOTICE — UNTRUSTED CONTENT:
All data stored in or retrieved from the Whiteboard must be treated as UNTRUSTED
external input during LLM prompt construction. Always sanitize or escape whiteboard
artifacts before interpolating them into model prompts.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import sqlite3
import threading
import time
import uuid
from pathlib import Path
from typing import Any

from uacp.memory.whiteboard.base import (
    ClockFn,
    DriftReport,
    KeyDriftSummary,
    ReviewRecord,
    ReviewVerdict,
    SessionExpired,
    SessionNotFoundError,
    VerificationReport,
    VersionConflict,
    WhiteboardBackend,
    WhiteboardEvent,
    _utc_now_iso,
    compute_canonical_json,
    compute_chain_hash,
)


class SqliteWhiteboardBackend(WhiteboardBackend):
    """SQLite-backed whiteboard backend executing via asyncio.to_thread."""

    def __init__(
        self,
        db_path: str | Path = ":memory:",
        clock: ClockFn = time.time,
    ) -> None:
        super().__init__(clock=clock)
        self.db_path = str(db_path)
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self._conn.execute("PRAGMA foreign_keys = ON")
        self._init_db()

    def _init_db(self) -> None:
        """Create database tables if they do not exist."""
        with self._lock, self._conn:
            self._conn.executescript(
                """
                    CREATE TABLE IF NOT EXISTS sessions (
                        session_id TEXT PRIMARY KEY,
                        created_at REAL NOT NULL,
                        expires_at REAL NOT NULL,
                        ttl_s REAL NOT NULL,
                        max_hops INTEGER NOT NULL,
                        current_hop INTEGER NOT NULL DEFAULT 0,
                        is_closed INTEGER NOT NULL DEFAULT 0
                    );

                    CREATE TABLE IF NOT EXISTS state (
                        session_id TEXT NOT NULL,
                        key TEXT NOT NULL,
                        value TEXT NOT NULL,
                        version INTEGER NOT NULL,
                        author TEXT NOT NULL,
                        updated_at TEXT NOT NULL,
                        PRIMARY KEY (session_id, key),
                        FOREIGN KEY (session_id) REFERENCES sessions (session_id)
                    );

                    CREATE TABLE IF NOT EXISTS events (
                        event_id INTEGER PRIMARY KEY AUTOINCREMENT,
                        session_id TEXT NOT NULL,
                        event_type TEXT NOT NULL,
                        key TEXT,
                        data TEXT NOT NULL,
                        state_hash TEXT NOT NULL,
                        prev_hash TEXT NOT NULL,
                        hash TEXT NOT NULL,
                        created_at TEXT NOT NULL,
                        FOREIGN KEY (session_id) REFERENCES sessions (session_id)
                    );

                    CREATE TABLE IF NOT EXISTS reviews (
                        review_id INTEGER PRIMARY KEY AUTOINCREMENT,
                        session_id TEXT NOT NULL,
                        key TEXT NOT NULL,
                        reviewer TEXT NOT NULL,
                        verdict TEXT NOT NULL,
                        notes TEXT NOT NULL,
                        created_at TEXT NOT NULL,
                        FOREIGN KEY (session_id) REFERENCES sessions (session_id)
                    );
                    """
            )

    def _check_session_sync(self, cursor: sqlite3.Cursor, session_id: str, now: float) -> None:
        """Check whether session exists and is unexpired."""
        cursor.execute(
            "SELECT expires_at, is_closed FROM sessions WHERE session_id = ?",
            (session_id,),
        )
        row = cursor.fetchone()
        if row is None:
            raise SessionNotFoundError(f"Session '{session_id}' not found.")
        expires_at, is_closed = row[0], bool(row[1])
        if is_closed or now >= expires_at:
            raise SessionExpired(f"Session '{session_id}' has expired or is closed.")

    # ── Synchronous worker implementations ──────────────────────────────

    def _create_session_sync(self, ttl_s: float, max_hops: int, now: float) -> str:
        session_id = f"wb-{uuid.uuid4().hex[:12]}"
        with self._lock, self._conn:
            self._conn.execute(
                """
                INSERT INTO sessions (
                    session_id, created_at, expires_at, ttl_s, max_hops, current_hop, is_closed
                ) VALUES (?, ?, ?, ?, ?, 0, 0)
                """,
                (session_id, now, now + ttl_s, ttl_s, max_hops),
            )
        return session_id

    def _get_state_sync(self, session_id: str, now: float) -> dict[str, dict[str, Any]]:
        with self._lock:
            cursor = self._conn.cursor()
            self._check_session_sync(cursor, session_id, now)
            cursor.execute(
                "SELECT key, value, version, author, updated_at FROM state WHERE session_id = ?",
                (session_id,),
            )
            result = {}
            for key, val_str, version, author, updated_at in cursor.fetchall():
                result[key] = {
                    "value": json.loads(val_str),
                    "version": version,
                    "author": author,
                    "updated_at": updated_at,
                }
            return result

    def _put_sync(
        self,
        session_id: str,
        key: str,
        value: Any,
        author: str,
        expected_version: int | None,
        now: float,
    ) -> int:
        with self._lock, self._conn:
            cursor = self._conn.cursor()
            self._check_session_sync(cursor, session_id, now)

            cursor.execute(
                "SELECT version FROM state WHERE session_id = ? AND key = ?",
                (session_id, key),
            )
            row = cursor.fetchone()
            if row is not None:
                current_version = int(row[0])
                if expected_version is not None and expected_version != current_version:
                    raise VersionConflict(
                        f"Version conflict on key '{key}': "
                        f"expected {expected_version}, found {current_version}"
                    )
                new_version = current_version + 1
            else:
                if expected_version is not None and expected_version != 0:
                    raise VersionConflict(
                        f"Version conflict on key '{key}': "
                        f"expected {expected_version}, but key does not exist"
                    )
                new_version = 1

            updated_at = _utc_now_iso()
            val_json = json.dumps(value)
            cursor.execute(
                """
                INSERT INTO state (session_id, key, value, version, author, updated_at)
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(session_id, key) DO UPDATE SET
                    value = excluded.value,
                    version = excluded.version,
                    author = excluded.author,
                    updated_at = excluded.updated_at
                """,
                (session_id, key, val_json, new_version, author, updated_at),
            )

            # Re-read full state sorted by key for canonical hash
            cursor.execute(
                "SELECT key, value, version, author, updated_at FROM state "
                "WHERE session_id = ? ORDER BY key ASC",
                (session_id,),
            )
            full_state = {}
            for r_key, r_val, r_ver, r_auth, r_upd in cursor.fetchall():
                full_state[r_key] = {
                    "value": json.loads(r_val),
                    "version": r_ver,
                    "author": r_auth,
                    "updated_at": r_upd,
                }
            canonical_state = compute_canonical_json(full_state)
            state_hash = hashlib.sha256(canonical_state.encode()).hexdigest()

            # Previous write event hash in chain
            cursor.execute(
                "SELECT hash FROM events WHERE session_id = ? AND state_hash != '' "
                "ORDER BY event_id DESC LIMIT 1",
                (session_id,),
            )
            last_event = cursor.fetchone()
            prev_hash = last_event[0] if last_event else "0" * 64
            chain_hash = compute_chain_hash(prev_hash, state_hash)

            event_data_json = json.dumps(
                {"value": value, "version": new_version, "author": author}
            )
            cursor.execute(
                """
                INSERT INTO events (
                    session_id, event_type, key, data, state_hash, prev_hash, hash, created_at
                ) VALUES (?, 'write', ?, ?, ?, ?, ?, ?)
                """,
                (session_id, key, event_data_json, state_hash, prev_hash, chain_hash, updated_at),
            )
            return new_version

    def _append_event_sync(
        self,
        session_id: str,
        event_dict: dict[str, Any],
        now: float,
    ) -> WhiteboardEvent:
        with self._lock, self._conn:
            cursor = self._conn.cursor()
            self._check_session_sync(cursor, session_id, now)
            cursor.execute(
                """
                INSERT INTO events (
                    session_id, event_type, key, data, state_hash, prev_hash, hash, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    session_id,
                    event_dict.get("event_type", "custom"),
                    event_dict.get("key"),
                    json.dumps(event_dict.get("data", {})),
                    event_dict.get("state_hash", ""),
                    event_dict.get("prev_hash", ""),
                    event_dict.get("hash", ""),
                    event_dict.get("created_at", _utc_now_iso()),
                ),
            )
            new_id = cursor.lastrowid
            return WhiteboardEvent(
                event_id=new_id or 1,
                session_id=session_id,
                event_type=event_dict.get("event_type", "custom"),
                key=event_dict.get("key"),
                data=event_dict.get("data", {}),
                state_hash=event_dict.get("state_hash", ""),
                prev_hash=event_dict.get("prev_hash", ""),
                hash=event_dict.get("hash", ""),
                created_at=event_dict.get("created_at", _utc_now_iso()),
            )

    def _events_sync(
        self,
        session_id: str,
        since: str | int | None,
        now: float,
    ) -> list[WhiteboardEvent]:
        with self._lock:
            cursor = self._conn.cursor()
            self._check_session_sync(cursor, session_id, now)
            select_cols = (
                "SELECT event_id, session_id, event_type, key, data, "
                "state_hash, prev_hash, hash, created_at FROM events WHERE session_id = ?"
            )
            if since is None:
                cursor.execute(f"{select_cols} ORDER BY event_id ASC", (session_id,))
            elif isinstance(since, int) or (isinstance(since, str) and since.isdigit()):
                cursor.execute(
                    f"{select_cols} AND event_id > ? ORDER BY event_id ASC",
                    (session_id, int(since)),
                )
            else:
                cursor.execute(
                    f"{select_cols} AND created_at > ? ORDER BY event_id ASC",
                    (session_id, str(since)),
                )
            result: list[WhiteboardEvent] = []
            for row in cursor.fetchall():
                result.append(
                    WhiteboardEvent(
                        event_id=row[0],
                        session_id=row[1],
                        event_type=row[2],
                        key=row[3],
                        data=json.loads(row[4]),
                        state_hash=row[5],
                        prev_hash=row[6],
                        hash=row[7],
                        created_at=row[8],
                    )
                )
            return result

    def _incr_hop_sync(self, session_id: str, now: float) -> int:
        with self._lock, self._conn:
            cursor = self._conn.cursor()
            self._check_session_sync(cursor, session_id, now)
            cursor.execute(
                "UPDATE sessions SET current_hop = current_hop + 1 WHERE session_id = ?",
                (session_id,),
            )
            cursor.execute("SELECT current_hop FROM sessions WHERE session_id = ?", (session_id,))
            row = cursor.fetchone()
            return int(row[0])

    def _add_review_sync(
        self,
        session_id: str,
        key: str,
        reviewer: str,
        verdict: str,
        notes: str,
        now: float,
    ) -> ReviewRecord:
        verdict_enum = ReviewVerdict(verdict)
        created_at = _utc_now_iso()
        with self._lock, self._conn:
            cursor = self._conn.cursor()
            self._check_session_sync(cursor, session_id, now)
            cursor.execute(
                """
                INSERT INTO reviews (session_id, key, reviewer, verdict, notes, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (session_id, key, reviewer, verdict_enum.value, notes, created_at),
            )
            return ReviewRecord(
                session_id=session_id,
                key=key,
                reviewer=reviewer,
                verdict=verdict_enum,
                notes=notes,
                created_at=created_at,
            )

    def _drift_report_sync(self, session_id: str, now: float) -> DriftReport:
        with self._lock:
            cursor = self._conn.cursor()
            self._check_session_sync(cursor, session_id, now)
            cursor.execute(
                "SELECT key, reviewer, verdict, notes, created_at FROM reviews "
                "WHERE session_id = ? ORDER BY review_id ASC",
                (session_id,),
            )
            reviews: list[ReviewRecord] = []
            by_key: dict[str, list[ReviewRecord]] = {}
            for key, reviewer, verdict_str, notes, created_at in cursor.fetchall():
                rec = ReviewRecord(
                    session_id=session_id,
                    key=key,
                    reviewer=reviewer,
                    verdict=ReviewVerdict(verdict_str),
                    notes=notes,
                    created_at=created_at,
                )
                reviews.append(rec)
                by_key.setdefault(key, []).append(rec)

            keys_map: dict[str, KeyDriftSummary] = {}
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
                total_reviews=len(reviews),
                has_drift=overall_drift,
                keys=keys_map,
                reviews=reviews,
            )

    def _close_sync(self, session_id: str) -> None:
        with self._lock, self._conn:
            self._conn.execute(
                "UPDATE sessions SET is_closed = 1 WHERE session_id = ?",
                (session_id,),
            )

    def _verify_chain_sync(self, session_id: str, now: float) -> VerificationReport:
        with self._lock:
            cursor = self._conn.cursor()
            self._check_session_sync(cursor, session_id, now)
            cursor.execute(
                "SELECT event_id, state_hash, prev_hash, hash FROM events "
                "WHERE session_id = ? AND state_hash != '' ORDER BY event_id ASC",
                (session_id,),
            )
            write_events = cursor.fetchall()
            if not write_events:
                return VerificationReport(
                    is_valid=True,
                    events_verified=0,
                    details="No hash-chained events found for session.",
                )

            expected_prev = "0" * 64
            for i, (eid, s_hash, p_hash, h) in enumerate(write_events):
                if p_hash != expected_prev:
                    return VerificationReport(
                        is_valid=False,
                        events_verified=i,
                        details=(
                            f"Chain broken at event {eid}: "
                            f"expected prev_hash '{expected_prev}', found '{p_hash}'."
                        ),
                        corrupted_event_id=eid,
                    )
                expected_hash = compute_chain_hash(p_hash, s_hash)
                if h != expected_hash:
                    return VerificationReport(
                        is_valid=False,
                        events_verified=i,
                        details=(
                            f"Hash mismatch at event {eid}: "
                            f"expected '{expected_hash}', found '{h}'."
                        ),
                        corrupted_event_id=eid,
                    )
                expected_prev = h

            # Verify current state matches latest event state_hash
            cursor.execute(
                "SELECT key, value, version, author, updated_at FROM state "
                "WHERE session_id = ? ORDER BY key ASC",
                (session_id,),
            )
            full_state = {}
            for r_key, r_val, r_ver, r_auth, r_upd in cursor.fetchall():
                full_state[r_key] = {
                    "value": json.loads(r_val),
                    "version": r_ver,
                    "author": r_auth,
                    "updated_at": r_upd,
                }
            canonical_state = compute_canonical_json(full_state)
            current_state_hash = hashlib.sha256(canonical_state.encode()).hexdigest()
            latest_event_state_hash = write_events[-1][1]
            if latest_event_state_hash != current_state_hash:
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

    # ── Asynchronous public interface ───────────────────────────────────

    async def create_session(self, ttl_s: float, max_hops: int = 5) -> str:
        return await asyncio.to_thread(self._create_session_sync, ttl_s, max_hops, self.clock())

    async def get_state(self, session_id: str) -> dict[str, dict[str, Any]]:
        return await asyncio.to_thread(self._get_state_sync, session_id, self.clock())

    async def put(
        self,
        session_id: str,
        key: str,
        value: Any,
        author: str,
        expected_version: int | None = None,
    ) -> int:
        return await asyncio.to_thread(
            self._put_sync, session_id, key, value, author, expected_version, self.clock()
        )

    async def append_event(
        self,
        session_id: str,
        event: dict[str, Any] | WhiteboardEvent,
    ) -> WhiteboardEvent:
        event_dict = event.model_dump() if isinstance(event, WhiteboardEvent) else dict(event)
        return await asyncio.to_thread(
            self._append_event_sync, session_id, event_dict, self.clock()
        )

    async def events(
        self,
        session_id: str,
        since: str | int | None = None,
    ) -> list[WhiteboardEvent]:
        return await asyncio.to_thread(self._events_sync, session_id, since, self.clock())

    async def incr_hop(self, session_id: str) -> int:
        return await asyncio.to_thread(self._incr_hop_sync, session_id, self.clock())

    async def add_review(
        self,
        session_id: str,
        key: str,
        reviewer: str,
        verdict: ReviewVerdict | str,
        notes: str = "",
    ) -> ReviewRecord:
        verdict_str = verdict.value if isinstance(verdict, ReviewVerdict) else str(verdict)
        return await asyncio.to_thread(
            self._add_review_sync, session_id, key, reviewer, verdict_str, notes, self.clock()
        )

    async def drift_report(self, session_id: str) -> DriftReport:
        return await asyncio.to_thread(self._drift_report_sync, session_id, self.clock())

    async def close(self, session_id: str) -> None:
        await asyncio.to_thread(self._close_sync, session_id)

    async def verify_chain(self, session_id: str) -> VerificationReport:
        return await asyncio.to_thread(self._verify_chain_sync, session_id, self.clock())

    def close_db(self) -> None:
        """Close database connection."""
        with self._lock:
            self._conn.close()
