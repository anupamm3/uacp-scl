"""High-level typed WhiteboardClient for agent interactions.

SECURITY NOTICE — UNTRUSTED CONTENT:
All data stored in or retrieved from the Whiteboard must be treated as UNTRUSTED
external input during LLM prompt construction. Peer agents or intermediate actors
could inject malicious instructions or adversarial context into whiteboard keys.
Always sanitize, escape, or validate whiteboard artifacts (e.g. using SCL variable
binding or strict output schemas) before interpolating them into model prompts.
"""

from __future__ import annotations

from typing import Any, TypeVar

from pydantic import BaseModel

from uacp.memory.whiteboard.base import (
    DriftReport,
    ReviewRecord,
    ReviewVerdict,
    VerificationReport,
    WhiteboardBackend,
)

T = TypeVar("T", bound=BaseModel)


class WhiteboardClient:
    """Agent client providing typed helpers and ergonomic whiteboard access."""

    def __init__(
        self,
        backend: WhiteboardBackend,
        session_id: str,
        author: str = "agent",
    ) -> None:
        self.backend = backend
        self.session_id = session_id
        self.author = author

    async def write_model(
        self,
        key: str,
        model: BaseModel,
        expected_version: int | None = None,
    ) -> int:
        """Serialize a Pydantic model and write it to the whiteboard.

        Args:
            key: Target whiteboard key.
            model: Pydantic BaseModel instance to serialize.
            expected_version: Optional version expected for optimistic locking.

        Returns:
            The new version number of the key.
        """
        payload = model.model_dump(mode="json")
        return await self.backend.put(
            session_id=self.session_id,
            key=key,
            value=payload,
            author=self.author,
            expected_version=expected_version,
        )

    async def read_model(self, key: str, model_cls: type[T]) -> T:
        """Retrieve and parse a key's value into a specified Pydantic model.

        Args:
            key: Whiteboard key to read.
            model_cls: Pydantic model class to validate and construct.

        Returns:
            Validated instance of model_cls.

        Raises:
            KeyError: If key does not exist on the whiteboard.
            pydantic.ValidationError: If stored value fails schema validation.
        """
        state = await self.backend.get_state(self.session_id)
        if key not in state:
            raise KeyError(f"Key '{key}' not found in whiteboard session '{self.session_id}'.")
        raw_val = state[key]["value"]
        return model_cls.model_validate(raw_val)

    async def read_all(self) -> dict[str, dict[str, Any]]:
        """Retrieve all state entries in the current session."""
        return await self.backend.get_state(self.session_id)

    async def put(
        self,
        key: str,
        value: Any,
        expected_version: int | None = None,
    ) -> int:
        """Write raw JSON-serializable value to key."""
        return await self.backend.put(
            session_id=self.session_id,
            key=key,
            value=value,
            author=self.author,
            expected_version=expected_version,
        )

    async def get(self, key: str) -> Any:
        """Retrieve raw value stored under key."""
        state = await self.backend.get_state(self.session_id)
        if key not in state:
            raise KeyError(f"Key '{key}' not found in whiteboard session '{self.session_id}'.")
        return state[key]["value"]

    async def incr_hop(self) -> int:
        """Atomically increment the session circuit-breaker hop count."""
        return await self.backend.incr_hop(self.session_id)

    async def add_review(
        self,
        key: str,
        verdict: ReviewVerdict | str,
        notes: str = "",
    ) -> ReviewRecord:
        """Record a review verdict for key."""
        return await self.backend.add_review(
            session_id=self.session_id,
            key=key,
            reviewer=self.author,
            verdict=verdict,
            notes=notes,
        )

    async def drift_report(self) -> DriftReport:
        """Generate consensus and drift report for current session."""
        return await self.backend.drift_report(self.session_id)

    async def verify_chain(self) -> VerificationReport:
        """Verify hash chain and state integrity for current session."""
        return await self.backend.verify_chain(self.session_id)

    async def close(self) -> None:
        """Close current session."""
        await self.backend.close(self.session_id)

