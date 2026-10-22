"""Whiteboard — shared temporary scratchpad for multi-agent sessions.

SECURITY NOTICE — UNTRUSTED CONTENT:
All data stored in or retrieved from the Whiteboard must be treated as UNTRUSTED
external input during LLM prompt construction. Peer agents or intermediate actors
could inject malicious instructions or adversarial context into whiteboard keys.
Always sanitize, escape, or validate whiteboard artifacts (e.g. using SCL variable
binding or strict output schemas) before interpolating them into model prompts.
"""

from uacp.memory.whiteboard.base import (
    DriftReport,
    KeyDriftSummary,
    ReviewRecord,
    ReviewVerdict,
    SessionExpired,
    SessionNotFoundError,
    StateEntry,
    TamperError,
    VerificationReport,
    VersionConflict,
    WhiteboardBackend,
    WhiteboardError,
    WhiteboardEvent,
    compute_canonical_json,
    compute_chain_hash,
)
from uacp.memory.whiteboard.client import WhiteboardClient
from uacp.memory.whiteboard.inmem_backend import (
    InMemoryWhiteboardBackend,
)
from uacp.memory.whiteboard.sqlite_backend import SqliteWhiteboardBackend

__all__ = [
    "DriftReport",
    "InMemoryWhiteboardBackend",
    "KeyDriftSummary",
    "ReviewRecord",
    "ReviewVerdict",
    "SessionExpired",
    "SessionNotFoundError",
    "SqliteWhiteboardBackend",
    "StateEntry",
    "TamperError",
    "VerificationReport",
    "VersionConflict",
    "WhiteboardBackend",
    "WhiteboardClient",
    "WhiteboardError",
    "WhiteboardEvent",
    "compute_canonical_json",
    "compute_chain_hash",
]
