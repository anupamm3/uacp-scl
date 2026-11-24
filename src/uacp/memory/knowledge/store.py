"""Persistent knowledge store with SQLite backing and semantic embeddings.

SECURITY NOTICE — UNTRUSTED CONTENT:
Facts stored in the knowledge store may originate from untrusted external sources,
intermediate agents, or whiteboard state. Always treat stored knowledge as untrusted
input when constructing LLM prompts.
"""

from __future__ import annotations

import asyncio
import json
import sqlite3
import threading
import time
import uuid
from collections.abc import Callable, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, TypeAlias

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict, Field

from uacp.memory.knowledge.embeddings import (
    EmbeddingProvider,
    HashEmbeddings,
)
from uacp.memory.whiteboard.base import WhiteboardBackend
from uacp.memory.whiteboard.client import WhiteboardClient

ClockFn: TypeAlias = Callable[[], float]


def _utc_now_iso() -> str:
    """Return the current ISO-8601 UTC timestamp."""
    return datetime.now(UTC).isoformat()


# ── Models ───────────────────────────────────────────────────────────────


class Evidence(BaseModel):
    """Evidence supporting or referencing a fact."""

    model_config = ConfigDict(frozen=True)

    id: str = Field(default_factory=lambda: f"ev-{uuid.uuid4().hex[:12]}")
    fact_id: str = Field(default="", description="Identifier of the associated fact.")
    kind: str = Field(..., description="Evidence type (e.g. 'whiteboard', 'document', 'quote').")
    ref: str = Field(..., description="Reference pointer or URI (e.g. session/key/version).")
    snippet: str = Field(..., description="Excerpt or description of evidence.")


class Fact(BaseModel):
    """A statement of fact stored in the knowledge store."""

    model_config = ConfigDict(frozen=True)

    id: str = Field(default_factory=lambda: f"fact-{uuid.uuid4().hex[:12]}")
    subject: str = Field(..., description="Normalised subject of the fact triple.")
    predicate: str = Field(..., description="Normalised predicate of the fact triple.")
    object: str = Field(..., description="Object value of the fact triple.")
    text: str = Field(..., description="Natural language description of the fact.")
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    source_agent: str = Field(..., description="Authoring agent identifier.")
    source_session: str = Field(..., description="Originating session identifier.")
    created_at: str = Field(default_factory=_utc_now_iso)
    valid_from: str = Field(default_factory=_utc_now_iso)
    valid_to: str | None = Field(default=None)
    status: str = Field(
        default="active",
        description="Fact status: 'active', 'superseded', or 'disputed'.",
    )
    evidence: list[Evidence] = Field(default_factory=list)


class ConflictCandidate(BaseModel):
    """Potential contradiction detected upon adding a fact."""

    model_config = ConfigDict(frozen=True)

    fact_id: str = Field(..., description="ID of the newly added fact.")
    existing_fact: Fact = Field(..., description="Existing active fact in conflict.")
    conflict_type: str = Field(
        ...,
        description="'subject_predicate_mismatch' or 'semantic_similarity'.",
    )
    similarity: float = Field(..., description="Cosine similarity between facts.")
    score: float | None = Field(default=None, description="Optional conflict priority score.")


class QueryResult(BaseModel):
    """Result of a semantic top-k query."""

    model_config = ConfigDict(frozen=True)

    fact: Fact = Field(..., description="Matching fact.")
    score: float = Field(..., description="Cosine similarity score.")


# ── Store Implementation ─────────────────────────────────────────────────


class KnowledgeStore:
    """Async persistent knowledge store backed by stdlib sqlite3 and vector embeddings."""

    def __init__(
        self,
        db_path: str | Path = ":memory:",
        embeddings: EmbeddingProvider | None = None,
        conflict_threshold: float = 0.85,
        clock: ClockFn = time.time,
    ) -> None:
        self.db_path = str(db_path)
        self.embeddings = embeddings or HashEmbeddings()
        self.conflict_threshold = conflict_threshold
        self.clock = clock
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self._conn.execute("PRAGMA foreign_keys = ON")
        self._init_db()

    def _init_db(self) -> None:
        with self._lock, self._conn:
            self._conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS facts (
                    id TEXT PRIMARY KEY,
                    subject TEXT NOT NULL,
                    predicate TEXT NOT NULL,
                    object TEXT NOT NULL,
                    text TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    source_agent TEXT NOT NULL,
                    source_session TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    valid_from TEXT NOT NULL,
                    valid_to TEXT,
                    status TEXT NOT NULL,
                    embedding BLOB NOT NULL
                );

                CREATE TABLE IF NOT EXISTS evidence (
                    id TEXT PRIMARY KEY,
                    fact_id TEXT NOT NULL,
                    kind TEXT NOT NULL,
                    ref TEXT NOT NULL,
                    snippet TEXT NOT NULL,
                    FOREIGN KEY (fact_id) REFERENCES facts (id)
                );

                CREATE TABLE IF NOT EXISTS conflicts (
                    id TEXT PRIMARY KEY,
                    fact_a TEXT NOT NULL,
                    fact_b TEXT NOT NULL,
                    score_a REAL NOT NULL,
                    score_b REAL NOT NULL,
                    method TEXT NOT NULL,
                    resolved_at TEXT,
                    resolution TEXT,
                    FOREIGN KEY (fact_a) REFERENCES facts (id),
                    FOREIGN KEY (fact_b) REFERENCES facts (id)
                );

                CREATE INDEX IF NOT EXISTS idx_facts_status ON facts(status);
                CREATE INDEX IF NOT EXISTS idx_facts_sp ON facts(subject, predicate);
                CREATE INDEX IF NOT EXISTS idx_evidence_fact ON evidence(fact_id);
                """
            )

    # ── Synchronous worker helpers ──────────────────────────────────────

    def _load_evidence_for_facts(
        self, cursor: sqlite3.Cursor, fact_ids: list[str]
    ) -> dict[str, list[Evidence]]:
        if not fact_ids:
            return {}
        placeholders = ",".join("?" for _ in fact_ids)
        cursor.execute(
            f"SELECT id, fact_id, kind, ref, snippet FROM evidence "
            f"WHERE fact_id IN ({placeholders})",
            fact_ids,
        )
        evidence_map: dict[str, list[Evidence]] = {fid: [] for fid in fact_ids}
        for eid, fid, kind, ref, snippet in cursor.fetchall():
            evidence_map[fid].append(
                Evidence(id=eid, fact_id=fid, kind=kind, ref=ref, snippet=snippet)
            )
        return evidence_map

    def _row_to_fact(self, row: tuple[Any, ...], evidence: list[Evidence] | None = None) -> Fact:
        (
            fid,
            subject,
            predicate,
            obj,
            text,
            confidence,
            source_agent,
            source_session,
            created_at,
            valid_from,
            valid_to,
            status,
        ) = row[:12]
        return Fact(
            id=fid,
            subject=subject,
            predicate=predicate,
            object=obj,
            text=text,
            confidence=confidence,
            source_agent=source_agent,
            source_session=source_session,
            created_at=created_at,
            valid_from=valid_from,
            valid_to=valid_to,
            status=status,
            evidence=evidence or [],
        )

    def _add_fact_sync(
        self,
        subject: str,
        predicate: str,
        obj: str,
        text: str,
        confidence: float,
        source_agent: str,
        source_session: str,
        evidence: Sequence[Evidence] | None,
        valid_from: str | None,
        embedding: NDArray[np.float32],
    ) -> tuple[Fact, list[ConflictCandidate]]:
        norm_subject = subject.strip().lower()
        norm_predicate = predicate.strip().lower()
        norm_object = obj.strip()
        vec = embedding.astype(np.float32)
        blob = vec.tobytes()

        fact_id = f"fact-{uuid.uuid4().hex[:12]}"
        now_iso = _utc_now_iso()
        v_from = valid_from or now_iso

        with self._lock, self._conn:
            cursor = self._conn.cursor()

            # Retrieve active facts for conflict detection
            cursor.execute(
                """
                SELECT id, subject, predicate, object, text, confidence, source_agent,
                       source_session, created_at, valid_from, valid_to, status, embedding
                FROM facts WHERE status = 'active'
                """
            )
            active_rows = cursor.fetchall()
            active_facts: list[Fact] = []
            active_vecs: list[NDArray[np.float32]] = []
            fact_ids = [r[0] for r in active_rows]
            ev_map = self._load_evidence_for_facts(cursor, fact_ids)

            for r in active_rows:
                active_facts.append(self._row_to_fact(r, ev_map.get(r[0], [])))
                stored_vec = np.frombuffer(r[12], dtype=np.float32)
                if len(stored_vec) != len(vec):
                    raise ValueError(
                        f"Stored embedding dimension {len(stored_vec)} != query {len(vec)}"
                    )
                active_vecs.append(stored_vec)

            # Detect conflict candidates
            conflict_candidates: list[ConflictCandidate] = []
            for existing_fact, existing_vec in zip(active_facts, active_vecs, strict=True):
                # Check 1: Exact triple mismatch (same subject and predicate, different object)
                same_sp = (
                    existing_fact.subject == norm_subject
                    and existing_fact.predicate == norm_predicate
                )
                diff_obj = existing_fact.object != norm_object
                sim = float(np.dot(vec, existing_vec))

                if same_sp and diff_obj:
                    conflict_candidates.append(
                        ConflictCandidate(
                            fact_id=fact_id,
                            existing_fact=existing_fact,
                            conflict_type="subject_predicate_mismatch",
                            similarity=sim,
                        )
                    )
                elif diff_obj and sim >= self.conflict_threshold:
                    conflict_candidates.append(
                        ConflictCandidate(
                            fact_id=fact_id,
                            existing_fact=existing_fact,
                            conflict_type="semantic_similarity",
                            similarity=sim,
                        )
                    )

            # Insert fact
            cursor.execute(
                """
                INSERT INTO facts (
                    id, subject, predicate, object, text, confidence, source_agent,
                    source_session, created_at, valid_from, valid_to, status, embedding
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'active', ?)
                """,
                (
                    fact_id,
                    norm_subject,
                    norm_predicate,
                    norm_object,
                    text,
                    confidence,
                    source_agent,
                    source_session,
                    now_iso,
                    v_from,
                    None,
                    blob,
                ),
            )

            # Insert evidence
            stored_evidence: list[Evidence] = []
            if evidence:
                for ev in evidence:
                    ev_id = ev.id or f"ev-{uuid.uuid4().hex[:12]}"
                    cursor.execute(
                        """
                        INSERT INTO evidence (id, fact_id, kind, ref, snippet)
                        VALUES (?, ?, ?, ?, ?)
                        """,
                        (ev_id, fact_id, ev.kind, ev.ref, ev.snippet),
                    )
                    stored_evidence.append(
                        Evidence(
                            id=ev_id,
                            fact_id=fact_id,
                            kind=ev.kind,
                            ref=ev.ref,
                            snippet=ev.snippet,
                        )
                    )

            new_fact = Fact(
                id=fact_id,
                subject=norm_subject,
                predicate=norm_predicate,
                object=norm_object,
                text=text,
                confidence=confidence,
                source_agent=source_agent,
                source_session=source_session,
                created_at=now_iso,
                valid_from=v_from,
                valid_to=None,
                status="active",
                evidence=stored_evidence,
            )
            return new_fact, conflict_candidates

    def _query_sync(self, query_vec: NDArray[np.float32], k: int) -> list[QueryResult]:
        vec = query_vec.astype(np.float32)
        with self._lock:
            cursor = self._conn.cursor()
            cursor.execute(
                """
                SELECT id, subject, predicate, object, text, confidence, source_agent,
                       source_session, created_at, valid_from, valid_to, status, embedding
                FROM facts WHERE status = 'active'
                """
            )
            rows = cursor.fetchall()
            if not rows:
                return []

            scored: list[tuple[float, tuple[Any, ...]]] = []
            for r in rows:
                fact_vec = np.frombuffer(r[12], dtype=np.float32)
                if len(fact_vec) != len(vec):
                    raise ValueError(
                        f"Stored embedding dimension {len(fact_vec)} != query {len(vec)}"
                    )
                score = float(np.dot(vec, fact_vec))
                scored.append((score, r))

            scored.sort(key=lambda x: x[0], reverse=True)
            top_k = scored[:k]

            fact_ids = [r[0] for _, r in top_k]
            ev_map = self._load_evidence_for_facts(cursor, fact_ids)

            results: list[QueryResult] = []
            for score, r in top_k:
                fact = self._row_to_fact(r, ev_map.get(r[0], []))
                results.append(QueryResult(fact=fact, score=score))

            return results

    def _active_view_sync(self, subject: str | None) -> list[Fact]:
        with self._lock:
            cursor = self._conn.cursor()
            if subject is not None:
                norm_subject = subject.strip().lower()
                cursor.execute(
                    """
                    SELECT id, subject, predicate, object, text, confidence, source_agent,
                           source_session, created_at, valid_from, valid_to, status
                    FROM facts WHERE status = 'active' AND subject = ?
                    ORDER BY created_at ASC
                    """,
                    (norm_subject,),
                )
            else:
                cursor.execute(
                    """
                    SELECT id, subject, predicate, object, text, confidence, source_agent,
                           source_session, created_at, valid_from, valid_to, status
                    FROM facts WHERE status = 'active'
                    ORDER BY created_at ASC
                    """
                )
            rows = cursor.fetchall()
            fact_ids = [r[0] for r in rows]
            ev_map = self._load_evidence_for_facts(cursor, fact_ids)
            return [self._row_to_fact(r, ev_map.get(r[0], [])) for r in rows]

    def _neighbors_sync(self, subject: str) -> dict[str, Any]:
        norm_subject = subject.strip().lower()
        with self._lock:
            cursor = self._conn.cursor()
            cursor.execute(
                """
                SELECT id, subject, predicate, object, text, confidence, source_agent,
                       source_session, created_at, valid_from, valid_to, status
                FROM facts WHERE status = 'active'
                """
            )
            rows = cursor.fetchall()
            fact_ids = [r[0] for r in rows]
            ev_map = self._load_evidence_for_facts(cursor, fact_ids)
            active_facts = [self._row_to_fact(r, ev_map.get(r[0], [])) for r in rows]

        try:
            import networkx as nx  # type: ignore[import-untyped]

            g = nx.MultiDiGraph()
            for fact in active_facts:
                s_node = fact.subject.strip().lower()
                o_node = fact.object.strip().lower()
                g.add_node(s_node)
                g.add_node(o_node)
                g.add_edge(
                    s_node,
                    o_node,
                    key=fact.predicate,
                    predicate=fact.predicate,
                    fact_id=fact.id,
                    confidence=fact.confidence,
                )

            if norm_subject in g:
                successors = list(g.successors(norm_subject))
                predecessors = list(g.predecessors(norm_subject))
                nodes = list(dict.fromkeys([norm_subject, *successors, *predecessors]))
                subgraph = g.subgraph(nodes)
                triples = []
                for u, v, k, data in g.edges(nodes, keys=True, data=True):
                    if u == norm_subject or v == norm_subject:
                        triples.append(
                            {
                                "subject": u,
                                "predicate": data.get("predicate", k),
                                "object": v,
                                "fact_id": data.get("fact_id"),
                            }
                        )
                return {
                    "subject": norm_subject,
                    "neighbors": [n for n in dict.fromkeys(successors + predecessors)],
                    "successors": successors,
                    "predecessors": predecessors,
                    "triples": triples,
                    "graph": subgraph,
                }
            return {
                "subject": norm_subject,
                "neighbors": [],
                "successors": [],
                "predecessors": [],
                "triples": [],
                "graph": g.subgraph([]),
            }
        except ImportError:
            # Graceful fallback without networkx
            successors = []
            predecessors = []
            triples = []
            for fact in active_facts:
                s_node = fact.subject.strip().lower()
                o_node = fact.object.strip().lower()
                if s_node == norm_subject:
                    successors.append(o_node)
                    triples.append(
                        {
                            "subject": fact.subject,
                            "predicate": fact.predicate,
                            "object": fact.object,
                            "fact_id": fact.id,
                        }
                    )
                if o_node == norm_subject:
                    predecessors.append(s_node)
                    triples.append(
                        {
                            "subject": fact.subject,
                            "predicate": fact.predicate,
                            "object": fact.object,
                            "fact_id": fact.id,
                        }
                    )
            return {
                "subject": norm_subject,
                "neighbors": list(dict.fromkeys(successors + predecessors)),
                "successors": successors,
                "predecessors": predecessors,
                "triples": triples,
                "graph": None,
            }

    def _supersede_sync(self, fact_id: str, valid_to: str) -> None:
        with self._lock, self._conn:
            self._conn.execute(
                "UPDATE facts SET status = 'superseded', valid_to = ? WHERE id = ?",
                (valid_to, fact_id),
            )

    def _dispute_sync(self, fact_id: str) -> None:
        with self._lock, self._conn:
            self._conn.execute(
                "UPDATE facts SET status = 'disputed' WHERE id = ?",
                (fact_id,),
            )

    # ── Asynchronous public interface ───────────────────────────────────

    async def add_fact(
        self,
        subject: str,
        predicate: str,
        obj: str,
        text: str,
        confidence: float = 1.0,
        source_agent: str = "agent",
        source_session: str = "session",
        evidence: Sequence[Evidence] | None = None,
        valid_from: str | None = None,
    ) -> tuple[Fact, list[ConflictCandidate]]:
        """Add a new fact to the knowledge store and detect conflict candidates.

        Args:
            subject: Subject entity.
            predicate: Relation or predicate.
            obj: Object or value.
            text: Human-readable natural language statement.
            confidence: Fact confidence score (0.0 to 1.0).
            source_agent: Authoring agent ID.
            source_session: Originating session ID.
            evidence: Supporting evidence items.
            valid_from: Optional starting validity ISO timestamp.

        Returns:
            Tuple of (Fact, list of ConflictCandidate).
        """
        raw_emb = await asyncio.to_thread(self.embeddings.embed, [text])
        vec = raw_emb[0]
        return await asyncio.to_thread(
            self._add_fact_sync,
            subject,
            predicate,
            obj,
            text,
            confidence,
            source_agent,
            source_session,
            evidence,
            valid_from,
            vec,
        )

    async def query(self, text: str, k: int = 5) -> list[QueryResult]:
        """Perform cosine similarity top-k search over active facts."""
        raw_emb = await asyncio.to_thread(self.embeddings.embed, [text])
        vec = raw_emb[0]
        return await asyncio.to_thread(self._query_sync, vec, k)

    async def active_view(self, subject: str | None = None) -> list[Fact]:
        """Return non-superseded (status='active') facts, optionally filtered by subject."""
        return await asyncio.to_thread(self._active_view_sync, subject)

    async def neighbors(self, subject: str) -> dict[str, Any]:
        """Return neighbouring triples and nodes for subject using networkx graph view."""
        return await asyncio.to_thread(self._neighbors_sync, subject)

    async def supersede_fact(self, fact_id: str, valid_to: str | None = None) -> None:
        """Mark a fact as superseded with valid_to timestamp (never deleted)."""
        v_to = valid_to or _utc_now_iso()
        await asyncio.to_thread(self._supersede_sync, fact_id, v_to)

    async def dispute_fact(self, fact_id: str) -> None:
        """Mark a fact as disputed (never deleted)."""
        await asyncio.to_thread(self._dispute_sync, fact_id)

    async def promote(
        self,
        whiteboard: WhiteboardClient | WhiteboardBackend,
        session_id: str,
        keys: Sequence[str],
    ) -> list[Fact]:
        """Promote reviewed whiteboard entries into persistent facts with provenance."""
        backend: WhiteboardBackend
        if isinstance(whiteboard, WhiteboardClient):
            backend = whiteboard.backend
            session_id = whiteboard.session_id or session_id
        else:
            backend = whiteboard

        state = await backend.get_state(session_id)
        promoted_facts: list[Fact] = []

        for key in keys:
            if key not in state:
                continue
            entry = state[key]
            val = entry["value"]
            version = entry["version"]
            author = entry["author"]

            ev = Evidence(
                fact_id="",
                kind="whiteboard",
                ref=f"{session_id}/{key}/v{version}",
                snippet=f"Promoted from whiteboard key '{key}' (v{version}) by '{author}'",
            )

            if isinstance(val, dict):
                s = str(val.get("subject", key))
                p = str(val.get("predicate", "states"))
                o = str(val.get("object", json.dumps(val)))
                t = str(val.get("text", f"{s} {p} {o}"))
                c = float(val.get("confidence", 1.0))
            else:
                s = str(key)
                p = "states"
                o = str(val)
                t = f"{s} {p} {o}"
                c = 1.0

            fact, _ = await self.add_fact(
                subject=s,
                predicate=p,
                obj=o,
                text=t,
                confidence=c,
                source_agent=author,
                source_session=session_id,
                evidence=[ev],
            )
            promoted_facts.append(fact)

        return promoted_facts

    def close(self) -> None:
        """Close SQLite database connection."""
        with self._lock:
            self._conn.close()

    async def __aenter__(self) -> KnowledgeStore:
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.close()
