"""Circuit breaker: loop prevention, anti-spoofing, cycle detection, and session safety."""

from __future__ import annotations

import contextlib
import hashlib
import inspect
import os
from collections import defaultdict
from collections.abc import Awaitable, Callable
from typing import Any, TypeAlias

from uacp.memory.whiteboard.base import (
    SessionExpired,
    WhiteboardBackend,
    compute_canonical_json,
)
from uacp.protocol.jsonrpc import (
    JSONRPCError,
    JSONRPCErrorCode,
)
from uacp.protocol.messages import UACPMessage

QuotaHook: TypeAlias = Callable[[str, UACPMessage], Awaitable[bool | None] | bool | None]


class CircuitBreaker:
    """Monitors multi-agent delegation chains to prevent loops, cycles, and quota exhaustion."""

    def __init__(
        self,
        whiteboard: WhiteboardBackend,
        max_hops: int | None = None,
        quota_hook: QuotaHook | None = None,
    ) -> None:
        """Initialize the circuit breaker.

        Args:
            whiteboard: Ephemeral Whiteboard backend holding authoritative session state.
            max_hops: Maximum permitted delegation hops. Read from UACP_MAX_HOPS env (default 5).
            quota_hook: Optional hook to check provider quota for a session/message.
        """
        self.whiteboard = whiteboard
        if max_hops is not None:
            self.max_hops = max_hops
        else:
            env_val = os.environ.get("UACP_MAX_HOPS", "5")
            try:
                self.max_hops = int(env_val)
            except ValueError:
                self.max_hops = 5

        self.quota_hook = quota_hook
        self._halted_sessions: dict[str, JSONRPCError] = {}
        self._seen_signatures: dict[str, dict[tuple[str, str, str], int]] = defaultdict(
            lambda: defaultdict(int)
        )
        self._session_max_hops: dict[str, int] = {}

    def set_session_max_hops(self, session_id: str, max_hops: int) -> None:
        """Configure a custom max_hops limit for a specific session."""
        self._session_max_hops[session_id] = max_hops

    def get_max_hops(self, session_id: str) -> int:
        """Return the effective max_hops limit for a given session."""
        return self._session_max_hops.get(session_id, self.max_hops)

    def is_session_halted(self, session_id: str) -> bool:
        """Return True if session has been halted by the circuit breaker or manual stop."""
        return session_id in self._halted_sessions

    def get_halt_error(self, session_id: str) -> JSONRPCError | None:
        """Return the halt error descriptor if session is halted, else None."""
        return self._halted_sessions.get(session_id)

    async def halt(
        self,
        session_id: str,
        reason: str,
        code: int = JSONRPCErrorCode.HOP_LIMIT_EXCEEDED,
    ) -> JSONRPCError:
        """Halt a session, recording a 'session.halted' event and refusing further deliveries."""
        err = JSONRPCError(code=code, message=reason)
        self._halted_sessions[session_id] = err

        with contextlib.suppress(Exception):
            await self.whiteboard.append_event(
                session_id,
                {
                    "event_type": "session.halted",
                    "data": {
                        "reason": reason,
                        "code": code,
                    },
                },
            )

        return err

    def _compute_payload_sha(self, payload: dict[str, Any]) -> str:
        """Compute SHA-256 hash of canonical JSON-serialized payload."""
        canonical_str = compute_canonical_json(payload)
        return hashlib.sha256(canonical_str.encode("utf-8")).hexdigest()

    async def check(
        self,
        session_id: str,
        message: UACPMessage,
        *,
        raise_error: bool = False,
    ) -> JSONRPCError | None:
        """Check safety invariants for an incoming message in the given session.

        Trips if:
        1. Session was previously halted (refuses further deliveries).
        2. Session has expired.
        3. Provider quota is exhausted.
        4. Authoritative hop count from Whiteboard exceeds max_hops.
        5. Advisory hop count inside message differs from authoritative count (anti-spoofing).
        6. Cycle detected: same (sender, recipient, payload hash) appears 2 or more times.

        Args:
            session_id: Target Whiteboard session identifier.
            message: Incoming UACPMessage envelope.
            raise_error: If True, raises JSONRPCError instead of returning it.

        Returns:
            JSONRPCError if breaker tripped, or None if message passes all checks.

        Raises:
            JSONRPCError: If breaker tripped and raise_error is True.
        """
        # 1. Check if session was already halted
        if session_id in self._halted_sessions:
            halt_err = self._halted_sessions[session_id]
            if raise_error:
                raise halt_err
            return halt_err

        # 2. Check provider quota hook
        if self.quota_hook is not None:
            try:
                res = self.quota_hook(session_id, message)
                if inspect.isawaitable(res):
                    res = await res
                if res is False:
                    err = await self.halt(
                        session_id,
                        reason="QuotaExceeded: provider quota exhausted for session",
                        code=JSONRPCErrorCode.QUOTA_EXCEEDED,
                    )
                    if raise_error:
                        raise err
                    return err
            except Exception as exc:
                err = await self.halt(
                    session_id,
                    reason=f"QuotaExceeded: {exc}",
                    code=JSONRPCErrorCode.QUOTA_EXCEEDED,
                )
                if raise_error:
                    raise err from exc
                return err

        # 3. Increment authoritative hop count on Whiteboard
        try:
            auth_hop = await self.whiteboard.incr_hop(session_id)
        except SessionExpired as exc:
            err = await self.halt(
                session_id,
                reason=f"SessionExpired: {exc}",
                code=JSONRPCErrorCode.SESSION_EXPIRED,
            )
            if raise_error:
                raise err from exc
            return err

        # 4. Anti-spoofing check: message advisory hop count must match authoritative count
        advisory_hop = message.circuit_breaker.current_hop
        if advisory_hop != auth_hop:
            err = await self.halt(
                session_id,
                reason=(
                    f"AntiSpoofingViolation: message advisory hop {advisory_hop} "
                    f"differs from authoritative Whiteboard hop {auth_hop}"
                ),
                code=JSONRPCErrorCode.HOP_LIMIT_EXCEEDED,
            )
            if raise_error:
                raise err
            return err

        # 5. Check authoritative hop count against max_hops
        limit = self.get_max_hops(session_id)
        if auth_hop >= limit:
            err = await self.halt(
                session_id,
                reason=(
                    f"HopLimitExceeded: authoritative hop {auth_hop} reached or exceeded "
                    f"max_hops {limit}"
                ),
                code=JSONRPCErrorCode.HOP_LIMIT_EXCEEDED,
            )
            if raise_error:
                raise err
            return err

        # 6. Cycle detection: (sender, recipient, sha256_canonical_payload) >= 2 appearances
        payload_sha = self._compute_payload_sha(message.payload)
        sig = (message.sender.agent_id, message.recipient.agent_id, payload_sha)

        seen_count = self._seen_signatures[session_id][sig] + 1
        self._seen_signatures[session_id][sig] = seen_count

        if seen_count >= 2:
            err = await self.halt(
                session_id,
                reason=(
                    f"CycleDetected: identical transmission ({message.sender.agent_id} -> "
                    f"{message.recipient.agent_id}, payload_sha={payload_sha[:12]}) "
                    f"appeared {seen_count} times in session"
                ),
                code=JSONRPCErrorCode.HOP_LIMIT_EXCEEDED,
            )
            if raise_error:
                raise err
            return err

        return None
