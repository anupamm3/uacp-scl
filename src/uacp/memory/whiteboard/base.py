"""Abstract whiteboard backend interface and core models.

SECURITY NOTICE — UNTRUSTED CONTENT:
All data stored in or retrieved from the Whiteboard must be treated as UNTRUSTED
external input during LLM prompt construction. Peer agents or intermediate actors
could inject malicious instructions or adversarial context into whiteboard keys.
Always sanitize, escape, or validate whiteboard artifacts (e.g. using SCL variable
binding or strict output schemas) before interpolating them into model prompts.
"""

from __future__ import annotations

import abc
import hashlib
import json
import time
from collections.abc import Callable
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any, TypeAlias

from pydantic import BaseModel, ConfigDict, Field

ClockFn: TypeAlias = Callable[[], float]


def _utc_now_iso() -> str:
    """Return the current ISO-8601 UTC timestamp."""
    return datetime.now(UTC).isoformat()


def compute_canonical_json(data: Any) -> str:
    """Produce deterministic canonical JSON for hashing."""
    return json.dumps(data, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def compute_chain_hash(prev_hash: str, state_hash: str) -> str:
    """Compute sha256 of previous chain hash and current state hash."""
    return hashlib.sha256(f"{prev_hash}:{state_hash}".encode()).hexdigest()


# ── Exceptions ───────────────────────────────────────────────────────────


class WhiteboardError(Exception):
    """Base exception for all Whiteboard errors."""


class SessionExpired(WhiteboardError):  # noqa: N818
    """Raised when accessing an expired or closed whiteboard session."""


class SessionNotFoundError(SessionExpired):
    """Raised when a session does not exist (inherits SessionExpired)."""


class VersionConflict(WhiteboardError):  # noqa: N818
    """Raised when a put operation fails optimistic concurrency version check."""


class TamperError(WhiteboardError):
    """Raised when hash chain or state tampering is detected."""


# ── Models ───────────────────────────────────────────────────────────────


class ReviewVerdict(StrEnum):
    """Verdict for agent review of a whiteboard entry."""

    agree = "agree"
    disagree = "disagree"
    needs_info = "needs_info"


class StateEntry(BaseModel):
    """Entry stored under a key in the whiteboard state."""

    model_config = ConfigDict(frozen=True)

    value: Any = Field(..., description="Arbitrary JSON-serializable value.")
    version: int = Field(..., ge=1, description="Monotonically increasing version number.")
    author: str = Field(..., description="Agent or entity that wrote this version.")
    updated_at: str = Field(..., description="ISO-8601 UTC timestamp of last update.")


class WhiteboardEvent(BaseModel):
    """Chronological event logged during whiteboard operations."""

    model_config = ConfigDict(frozen=True)

    event_id: int | str = Field(..., description="Unique event identifier.")
    session_id: str = Field(..., description="Associated session identifier.")
    event_type: str = Field(..., description="Event type (e.g. 'write', 'review', 'custom').")
    key: str | None = Field(default=None, description="Affected key if applicable.")
    data: dict[str, Any] = Field(default_factory=dict, description="Event payload data.")
    state_hash: str = Field(
        default="", description="SHA-256 of canonical full state at event time."
    )
    prev_hash: str = Field(default="", description="SHA-256 hash of previous event in chain.")
    hash: str = Field(default="", description="SHA-256 hash chaining prev_hash and state_hash.")
    created_at: str = Field(default_factory=_utc_now_iso, description="ISO-8601 UTC timestamp.")


class ReviewRecord(BaseModel):
    """Record of an agent's review on a specific key."""

    model_config = ConfigDict(frozen=True)

    session_id: str = Field(..., description="Whiteboard session identifier.")
    key: str = Field(..., description="Reviewed whiteboard key.")
    reviewer: str = Field(..., description="Identifier of the reviewer agent.")
    verdict: ReviewVerdict = Field(..., description="Review outcome verdict.")
    notes: str = Field(default="", description="Optional review comments or critique.")
    created_at: str = Field(default_factory=_utc_now_iso, description="ISO-8601 UTC timestamp.")


class KeyDriftSummary(BaseModel):
    """Summary of drift and review consensus for a single whiteboard key."""

    model_config = ConfigDict(frozen=True)

    key: str = Field(..., description="Whiteboard key.")
    total_reviews: int = Field(default=0, ge=0)
    agree_count: int = Field(default=0, ge=0)
    disagree_count: int = Field(default=0, ge=0)
    needs_info_count: int = Field(default=0, ge=0)
    has_drift: bool = Field(default=False, description="True if any disagreements were recorded.")
    notes: list[str] = Field(default_factory=list, description="Aggregated review notes.")


class DriftReport(BaseModel):
    """Comprehensive drift report for all reviewed keys in a session."""

    model_config = ConfigDict(frozen=True)

    session_id: str = Field(..., description="Whiteboard session identifier.")
    total_reviews: int = Field(default=0, ge=0)
    has_drift: bool = Field(default=False, description="True if any key exhibits drift.")
    keys: dict[str, KeyDriftSummary] = Field(default_factory=dict)
    reviews: list[ReviewRecord] = Field(default_factory=list)


class VerificationReport(BaseModel):
    """Report detailing the outcome of hash chain and state integrity verification."""

    model_config = ConfigDict(frozen=True)

    is_valid: bool = Field(..., description="True if chain is intact and untampered.")
    events_verified: int = Field(default=0, ge=0, description="Number of events validated.")
    details: str = Field(..., description="Human-readable verification outcome description.")
    corrupted_event_id: int | str | None = Field(
        default=None,
        description="ID of the first event that failed integrity check.",
    )


# ── Backend Abstract Class ───────────────────────────────────────────────


class WhiteboardBackend(abc.ABC):
    """Abstract asynchronous whiteboard storage backend."""

    def __init__(self, clock: ClockFn = time.time) -> None:
        self.clock: ClockFn = clock

    @abc.abstractmethod
    async def create_session(self, ttl_s: float, max_hops: int = 5) -> str:
        """Create a new ephemeral session with a TTL and circuit breaker hop limit.

        Args:
            ttl_s: Session lifetime in seconds.
            max_hops: Maximum allowed delegation hops (circuit breaker).

        Returns:
            The created session_id.
        """

    @abc.abstractmethod
    async def get_state(self, session_id: str) -> dict[str, dict[str, Any]]:
        """Retrieve current session state mapping key -> {value, version, author, updated_at}.

        Args:
            session_id: Target session ID.

        Returns:
            Dictionary mapping each key to its StateEntry dictionary.

        Raises:
            SessionExpired: If the session has expired, is closed, or does not exist.
        """

    @abc.abstractmethod
    async def put(
        self,
        session_id: str,
        key: str,
        value: Any,
        author: str,
        expected_version: int | None = None,
    ) -> int:
        """Write a value to a key, updating version and recording a hash-chained event.

        Args:
            session_id: Target session ID.
            key: State entry key.
            value: JSON-serializable value.
            author: Agent or entity modifying the state.
            expected_version: Optional version expected for optimistic concurrency.

        Returns:
            The new version integer for the key.

        Raises:
            VersionConflict: If expected_version does not match current key version.
            SessionExpired: If the session has expired, is closed, or does not exist.
        """

    @abc.abstractmethod
    async def append_event(
        self,
        session_id: str,
        event: dict[str, Any] | WhiteboardEvent,
    ) -> WhiteboardEvent:
        """Append an event to the session event log.

        Args:
            session_id: Target session ID.
            event: Event model or dictionary payload.

        Returns:
            The persisted WhiteboardEvent.

        Raises:
            SessionExpired: If the session has expired, is closed, or does not exist.
        """

    @abc.abstractmethod
    async def events(
        self,
        session_id: str,
        since: str | int | None = None,
    ) -> list[WhiteboardEvent]:
        """Retrieve chronological event log for a session, optionally filtered by timestamp or ID.

        Args:
            session_id: Target session ID.
            since: Optional ISO-8601 UTC timestamp or event_id lower bound (exclusive).

        Returns:
            List of WhiteboardEvent instances.

        Raises:
            SessionExpired: If the session has expired, is closed, or does not exist.
        """

    @abc.abstractmethod
    async def incr_hop(self, session_id: str) -> int:
        """Atomically increment and return the circuit-breaker hop counter for a session.

        Args:
            session_id: Target session ID.

        Returns:
            The incremented hop count.

        Raises:
            SessionExpired: If the session has expired, is closed, or does not exist.
        """

    @abc.abstractmethod
    async def add_review(
        self,
        session_id: str,
        key: str,
        reviewer: str,
        verdict: ReviewVerdict | str,
        notes: str = "",
    ) -> ReviewRecord:
        """Record an agent review and verdict for a specific whiteboard key.

        Args:
            session_id: Target session ID.
            key: Reviewed key.
            reviewer: Agent performing the review.
            verdict: Verdict enum or string ('agree', 'disagree', 'needs_info').
            notes: Optional explanatory notes.

        Returns:
            The stored ReviewRecord.

        Raises:
            SessionExpired: If the session has expired, is closed, or does not exist.
        """

    @abc.abstractmethod
    async def drift_report(self, session_id: str) -> DriftReport:
        """Generate a consensus and drift report based on recorded reviews for the session.

        Args:
            session_id: Target session ID.

        Returns:
            DriftReport containing key-level summaries and drift flags.

        Raises:
            SessionExpired: If the session has expired, is closed, or does not exist.
        """

    @abc.abstractmethod
    async def close(self, session_id: str) -> None:
        """Close a whiteboard session, invalidating subsequent access.

        Args:
            session_id: Target session ID.
        """

    @abc.abstractmethod
    async def verify_chain(self, session_id: str) -> VerificationReport:
        """Verify the integrity of the cryptographic hash chain and state for the session.

        Recomputes hashes across the write event chain and checks against the current state
        to detect any out-of-band tampering or log corruption.

        Args:
            session_id: Target session ID.

        Returns:
            VerificationReport indicating whether chain is valid and noting any corruption.

        Raises:
            SessionExpired: If the session has expired, is closed, or does not exist.
        """
