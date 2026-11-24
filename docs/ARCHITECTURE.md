# Architecture Overview

This document provides architectural blueprints and subsystem specifications for the UACP-SCL project.

---

## 1. Subsystems Summary

- **SCL (Semantic Context Layer):** AST parsing, variable binding, output schema normalization, and provider-agnostic request compilation.
- **UACP (Unified Agent Communication Protocol):** JSON-RPC 2.0 message envelope, circuit breaker runaway protection, capability discovery, and task negotiation.
- **Adapters (`uacp.adapters`):** Isolation layer for LLM providers (Gemini, OpenAI-compatible Groq/OpenRouter/Ollama) with schema repair and quota guarding.
- **Memory Subsystems (`uacp.memory`):**
  - **Whiteboard (`whiteboard`):** Ephemeral collaborative workspace with optimistic concurrency, hash-chain auditing, and drift detection.
  - **Knowledge Store (`knowledge`):** Long-term persistent triple store backed by SQLite and vector embeddings with automated contradiction resolution.

---

## 2. Persistent Knowledge Store & Contradiction Resolution

The Persistent Knowledge Store (`src/uacp/memory/knowledge/store.py`) and Contradiction Resolver (`src/uacp/memory/knowledge/resolver.py`) provide persistent, evidence-backed fact storage and contradiction handling across multi-agent workflows.

### 2.1 Contradiction Detection Rules

When a fact is added via `store.add_fact()`, conflict candidates are identified against active facts in SQLite using two complementary rules:
1. **Exact Triple Mismatch:** Active facts sharing the exact normalized $(subject, predicate)$ pair but having a different $object$ value.
2. **High Semantic Similarity:** Active facts with different $object$ values whose embedding cosine similarity exceeds `similarity_threshold` (default `0.85`).
3. **Optional LLM Arbitration:** When `use_llm_judge=True` and an adapter is configured, candidates can be pre-screened by an LLM returning `{"contradicts": bool, "reason": str}`. If the judge evaluates that no contradiction exists, resolution is marked `not_contradictory` and both facts remain `active`.

### 2.2 Scoring Formula

When a contradiction is confirmed, each competing fact ($A$ and $B$) is assigned a composite reliability score:

$$\text{score} = w_{\text{recency}} \times \text{recency} + w_{\text{certainty}} \times \text{certainty} + w_{\text{source}} \times \text{source\_reliability}$$

- **Recency Decay:**
  $$\text{recency} = 0.5^{(\text{age\_days} / \text{half\_life\_days})}$$
  where $\text{age\_days} = \max(0, (\text{now} - \text{created\_at}) / 86400)$ computed via an injectable clock.
- **Certainty:**
  $$\text{certainty} = \text{confidence} \times \text{mean\_cosine\_similarity}(\text{fact\_embedding}, \text{evidence\_embeddings})$$
  If a fact has no supporting evidence items, certainty defaults directly to `confidence`.
- **Source Reliability:**
  $$\text{source\_reliability} = \text{source\_priors.get}(\text{source\_agent}, \text{default\_source\_prior})$$
  Configurable per agent identifier.

#### Default Configuration Weights (`config/models.yaml`)
- Recency weight ($w_{\text{recency}}$): `0.4`
- Certainty weight ($w_{\text{certainty}}$): `0.5`
- Source reliability weight ($w_{\text{source}}$): `0.1`
- Half-life ($\text{half\_life\_days}$): `30.0` days
- Decision margin ($\text{margin}$): `0.15`
- Semantic similarity threshold: `0.85`
- Default source prior: `0.5`

### 2.3 State Transitions & Decision Rules

Resolution applies deterministic thresholding to the score differential:

$$\Delta = |\text{score}_a - \text{score}_b|$$

1. **Clear Winner ($\Delta \ge \text{margin}$):**
   - The higher-scoring fact remains/becomes status `'active'`.
   - The lower-scoring fact transitions to status `'superseded'` with `valid_to` stamped to the resolution time.
   - The decision is logged in the `conflicts` table as `resolution = "winner:<id>;loser:<id>"`.
2. **Ambiguous / Tie ($\Delta < \text{margin}$):**
   - Both competing facts transition to status `'disputed'`.
   - The pair is logged in `conflicts` with `resolution = "disputed"` and `resolved_at = NULL`.
   - Disputed facts are queued for manual inspection via `resolver.review_queue()`.
3. **Manual Adjudication:**
   - Operators can invoke `manual_resolve(conflict_id, winner_fact_id)`.
   - The selected winner transitions to `'active'`, the opposing fact transitions to `'superseded'` with `valid_to`, and the conflict record is updated with `method = "manual"`, removing it from the active review queue.
4. **History Preservation Rule:**
   - Facts in the knowledge store are strictly append-only and never deleted.
   - Status updates (`active` $\rightarrow$ `superseded` / `disputed`) update metadata without purging provenance or historical snapshots.
