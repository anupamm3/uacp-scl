# Unified Agent Communication Protocol (UACP) Specification

> **Version:** 1.0  
> **Status:** Active  
> **Transport Envelope:** JSON-RPC 2.0  

---

## 1. Overview

The **Unified Agent Communication Protocol (UACP)** provides a vendor-neutral, deterministic messaging standard for collaborative multi-agent systems. While the **Semantic Context Layer (SCL)** standardizes how individual agent prompts and schemas compile across LLM providers, UACP standardizes how these heterogeneous agents communicate, exchange structured data, coordinate shared state, and prevent runaway execution loops.

All agent-to-agent exchanges are encapsulated inside a standard **JSON-RPC 2.0** envelope, where the RPC `method` corresponds to `uacp.<intent>` and `params` contains the validated `UACPMessage`.

---

## 2. Core Envelope & Message Models

### 2.1 JSON-RPC 2.0 Envelopes

UACP adheres strictly to JSON-RPC 2.0 specifications:

#### Request Envelope (`JSONRPCRequest`)
| Field | Type | Required | Description |
|---|---|---|---|
| `jsonrpc` | `string` | Yes | Must be exactly `"2.0"`. |
| `id` | `string \| integer \| null` | Yes | Correlation identifier for request/response pairing. |
| `method` | `string` | Yes | Protocol method string formatted as `uacp.<intent>` (e.g. `uacp.task_delegation`). |
| `params` | `UACPMessage` | Yes | Validated UACP message payload. |

#### Response Envelope (`JSONRPCResponse`)
| Field | Type | Required | Description |
|---|---|---|---|
| `jsonrpc` | `string` | Yes | Must be exactly `"2.0"`. |
| `id` | `string \| integer \| null` | Yes | Matches the `id` from the corresponding request. |
| `result` | `any` | Optional | Result object on success (mutually exclusive with `error`). |
| `error` | `JSONRPCErrorData` | Optional | Error descriptor on failure (mutually exclusive with `result`). |

---

### 2.2 UACP Message (`UACPMessage`)

Every message transmitted over UACP has a standardized envelope with metadata for tracing, routing, and circuit-breaker safety:

| Field | Type | Description |
|---|---|---|
| `uacp_version` | `string` | Protocol version; default is `"1.0"`. |
| `message_id` | `string` | Monotonically sortable unique identifier (`<timestamp_ms_14digits>-<rand_hex>`). |
| `correlation_id` | `string` | Conversation or delegation chain identifier shared across multi-agent turns. |
| `trace_id` | `string` | Distributed tracing ID for observability. |
| `sender` | `Party` | Originating agent metadata (`agent_id`, `engine`). |
| `recipient` | `Party` | Intended destination agent metadata (`agent_id`, `engine`). |
| `intent` | `Intent` | Primary intent enum specifying the semantics of the message. |
| `payload` | `object` | Intent-specific payload dictionary validated against the intent schema. |
| `circuit_breaker` | `CircuitBreakerState` | Current loop-prevention state (`max_hops=5`, `current_hop=0`). |
| `created_at` | `string` (ISO-8601) | Timezone-aware UTC timestamp of message creation. |

#### Supporting Structures

- **`Party`**:
  - `agent_id` (`string`): Unique name/ID of the agent (e.g., `"agent_alpha"`).
  - `engine` (`string`): LLM engine or provider alias powering the agent (e.g., `"alpha_default:gemini"`).
- **`CircuitBreakerState`**:
  - `max_hops` (`integer`): Maximum permissible hops in the delegation chain (default `5`).
  - `current_hop` (`integer`): Hop index reached by the message (starts at `0`).

---

## 3. Intents & Typed Payloads

UACP defines eight standard intents. Each intent pairs with a typed payload model:

| Intent | Payload Model | Purpose |
|---|---|---|
| `task_delegation` | `TaskDelegationPayload` | Delegate an analytical, generation, or execution task to a peer agent. |
| `task_result` | `TaskResultPayload` | Return the outcome, summary, or reference to results for a delegated task. |
| `handoff` | `HandoffPayload` | Transfer conversation ownership or primary session control to another agent. |
| `negotiate` | `NegotiatePayload` | Bilateral negotiation of task constraints, parameters, or schemas. |
| `capability_query` | `CapabilityQueryPayload` | Query peer agent for supported capabilities, intents, or tools. |
| `capability_response`| `CapabilityResponsePayload`| Return advertised agent capabilities, modalities, and constraints. |
| `error` | `ErrorPayload` | Protocol or application-level error notification. |
| `heartbeat` | `HeartbeatPayload` | Agent liveness probe and health status signal. |

### Payload Model Schemas

1. **`TaskDelegationPayload`**:
   - `instruction` (`string`): Actionable instruction for the recipient.
   - `whiteboard_session_ref` (`string`): Shared whiteboard session containing contextual artifacts.
   - `expected_output_schema` (`object | null`): Optional JSON Schema governing expected output.
   - `deadline_s` (`float | null`): Maximum execution budget in seconds.

2. **`TaskResultPayload`**:
   - `status` (`string`): Execution outcome status (`"success"`, `"failure"`, `"partial"`).
   - `summary` (`string`): Human/agent readable summary of execution or findings.
   - `result_ref` (`string | null`): Whiteboard reference key or URI for structured result data.

3. **`NegotiatePayload`**:
   - `proposal` (`object | string`): Proposed parameters, batch size, or schema constraints.
   - `constraints` (`object | list | null`): Active constraints limiting feasible configurations.
   - `round` (`integer`): Negotiation iteration number (defaults to 1).

4. **`ErrorPayload`**:
   - `code` (`integer`): Error code.
   - `message` (`string`): Descriptive error message.
   - `data` (`any`): Optional contextual diagnostic metadata.

5. **`HandoffPayload`**:
   - `target_agent` (`string`): Destination agent ID.
   - `reason` (`string`): Operational explanation for handoff.
   - `context_ref` (`string | null`): Session state or memory reference.

6. **`CapabilityQueryPayload`**:
   - `query` (`list[string]`): List of capability keys queried.

7. **`CapabilityResponsePayload`**:
   - `capabilities` (`list[string] | object`): Supported capability listing or detailed metadata map.

8. **`HeartbeatPayload`**:
   - `status` (`string`): Health indicator (e.g. `"ok"`, `"degraded"`).
   - `timestamp` (`string`): Timezone-aware UTC timestamp.

---

## 4. Error Code Taxonomy

UACP integrates standard JSON-RPC 2.0 error codes and allocates reserved protocol error codes in the `-32001` to `-32005` range.

| Code | Identifier | Classification | Description |
|---|---|---|---|
| `-32700` | `PARSE_ERROR` | Standard JSON-RPC | Invalid JSON received by the server. |
| `-32600` | `INVALID_REQUEST` | Standard JSON-RPC | The JSON sent is not a valid JSON-RPC 2.0 Request object. |
| `-32601` | `METHOD_NOT_FOUND` | Standard JSON-RPC | The method does not exist or does not match `uacp.<intent>`. |
| `-32602` | `INVALID_PARAMS` | Standard JSON-RPC | Invalid method parameter (fails `UACPMessage` validation or intent mismatch). |
| `-32603` | `INTERNAL_ERROR` | Standard JSON-RPC | Internal JSON-RPC processing error. |
| `-32001` | `HOP_LIMIT_EXCEEDED` | UACP Reserved | Circuit breaker tripped; delegation chain exceeded `max_hops`. |
| `-32002` | `CAPABILITY_NOT_FOUND`| UACP Reserved | Requested agent or capability is unavailable or unsupported. |
| `-32003` | `SCHEMA_VIOLATION` | UACP Reserved | Message payload fails contract validation for the given intent. |
| `-32004` | `QUOTA_EXCEEDED` | UACP Reserved | Provider rate limit or token quota exhausted for the agent. |
| `-32005` | `SESSION_EXPIRED` | UACP Reserved | Whiteboard session has expired or been terminated. |

---

## 5. Concrete Message Example

The following example shows **Agent Alpha** (financial auditor on Google Gemini) delegating a ledger anomaly mitigation task to **Agent Beta** (mitigation engineer on Groq):

```json
{
  "jsonrpc": "2.0",
  "id": "req-alpha-001",
  "method": "uacp.task_delegation",
  "params": {
    "uacp_version": "1.0",
    "message_id": "01729555200000-8f921ab39c4d",
    "correlation_id": "corr-audit-session-202610",
    "trace_id": "trace-7729-381a",
    "sender": {
      "agent_id": "agent_alpha",
      "engine": "alpha_default:gemini-2.5-flash"
    },
    "recipient": {
      "agent_id": "agent_beta",
      "engine": "beta_default:llama-3.3-70b-versatile"
    },
    "intent": "task_delegation",
    "payload": {
      "instruction": "Mitigate duplicate transaction anomalies identified in ledger batch 4.",
      "whiteboard_session_ref": "wb-session-alpha-4421",
      "expected_output_schema": {
        "type": "object",
        "properties": {
          "code": { "type": "string" },
          "rationale": { "type": "string" }
        },
        "required": ["code", "rationale"]
      },
      "deadline_s": 30.0
    },
    "circuit_breaker": {
      "max_hops": 5,
      "current_hop": 1
    },
    "created_at": "2026-10-22T00:50:00Z"
  }
}
```

Agent Beta returns the following JSON-RPC response:

```json
{
  "jsonrpc": "2.0",
  "id": "req-alpha-001",
  "result": {
    "uacp_version": "1.0",
    "message_id": "01729555205000-11fa89cb7e22",
    "correlation_id": "corr-audit-session-202610",
    "trace_id": "trace-7729-381a",
    "sender": {
      "agent_id": "agent_beta",
      "engine": "beta_default:llama-3.3-70b-versatile"
    },
    "recipient": {
      "agent_id": "agent_alpha",
      "engine": "alpha_default:gemini-2.5-flash"
    },
    "intent": "task_result",
    "payload": {
      "status": "success",
      "summary": "Mitigation script generated and validated against sandbox tests.",
      "result_ref": "wb-session-alpha-4421/mitigations/v1"
    },
    "circuit_breaker": {
      "max_hops": 5,
      "current_hop": 2
    },
    "created_at": "2026-10-22T00:50:05Z"
  }
}
```

---

## 6. Delegation Sequence Diagram

The following Mermaid diagram illustrates the execution flow and circuit breaker validation during task delegation between Agent Alpha and Agent Beta:

```mermaid
sequenceDiagram
    autonumber
    actor User as User / Client
    participant Alpha as Agent Alpha (Gemini)
    participant WB as Shared Whiteboard
    participant Router as UACP Router / Breaker
    participant Beta as Agent Beta (Groq)

    User->>Alpha: Execute financial ledger audit
    Alpha->>WB: Write ledger findings ("findings.v1")
    WB-->>Alpha: Findings hash recorded

    Alpha->>Router: send(JSONRPCRequest: uacp.task_delegation)
    Note over Router: Check hop count (current_hop < max_hops)<br/>Check anti-spoofing against Whiteboard<br/>Hop count becomes 1
    
    alt Hop count exceeds limit (current_hop >= max_hops)
        Router-->>Alpha: JSONRPCResponse Error -32001 (HopLimitExceeded)
        Router->>WB: Record "session.halted" event
    else Circuit breaker healthy
        Router->>Beta: Deliver uacp.task_delegation
        Beta->>WB: Read "findings.v1" & verify hash chain
        WB-->>Beta: Validated findings data
        Beta->>Beta: Generate & sandbox mitigation code
        Beta->>Router: send(JSONRPCRequest: uacp.task_result)
        Router->>Alpha: Deliver task result
        Alpha-->>User: Final audit & mitigation report
    end
```

---

## 7. Agent Capability Discovery & Task Splitting

### 7.1 Manifest & Capability Models

Agent capabilities are described via two primary data structures:
- **`Capability`**: Represents a discrete callable capability.
  - `name` (`string`): Unique capability key.
  - `description` (`string`): Human-readable summary.
  - `input_schema` (`object | null`): JSON Schema for invocation inputs.
  - `output_schema` (`object | null`): JSON Schema for results.
  - `cost_hint` (`float`): Relative invocation or token cost estimate.
  - `avg_latency_ms` (`float`): Typical execution latency in milliseconds.
  - `complexity_level` (`int`, 1..5): Highest task difficulty handled well by this agent.
  - `strengths` (`list[string]`): Tags describing core strengths (e.g., `["analysis", "code_generation"]`).
  - `max_input_chars` (`int | null`): Input size budget, if constrained.
- **`AgentCard`**: Advertised manifest of an agent.
  - `agent_id` (`string`): Unique agent moniker.
  - `engine` (`string`): Underlying LLM provider or engine alias.
  - `role` (`string`): Agent persona or functional specialization.
  - `current_load` (`int`): Active concurrent task count (updated dynamically).
  - `capabilities` (`list[Capability]`): List of supported capabilities.
  - `endpoint` (`string | null`): Optional transport network address.
  - `version` (`string`): Agent contract version (default `"1.0"`).

### 7.2 Capability Registry & Ranking

The `CapabilityRegistry` maintains registered agent cards with a liveness lease governed by `HeartbeatPayload` (`default_ttl=30s`). Queries (`uacp.capability_query`) return ranked candidates evaluated across:
1. **Strengths Match**: Agents matching the highest count of `required_strengths` rank first.
2. **Current Load**: Agents with lower `current_load` are preferred to balance system workload.
3. **Cost Hint**: Lowest `cost_hint` breaks load ties.
4. **Preferred Engine**: Engine matches provide bonus priority.
5. **Latency**: Lowest `avg_latency_ms` provides secondary optimization.
6. **Tie-Breaker**: Lexicographical `agent_id`.

Each search result includes a `reason` explaining the ranking choices for downstream planner inspection.

### 7.3 Complexity-Based Task Splitting

When decomposing a complex objective, `suggest_split(subtasks)` dynamically assigns subtasks based on `required_complexity` and `required_strengths`. Agents with `complexity_level < required_complexity` or expired heartbeats are excluded. Subtasks that cannot be fulfilled are marked unassigned with an explicit operational reason.

### 7.4 Protocol Error Semantics

If a queried capability cannot be resolved among active (non-expired) agents during `uacp.capability_query`, the registry returns JSON-RPC error code `-32002` (`CAPABILITY_NOT_FOUND`).

---

## 8. Bilateral Task Negotiation & Session Handoff

### 8.1 Task Difficulty Negotiation (`uacp.negotiate`)

Heterogeneous agents dynamically negotiate subtask assignments by evaluating operational feasibility and difficulty:
- **`NegotiationProposal`**:
  - `subtask_id` (`string`): Unique identifier for the subtask.
  - `subtask_text` (`string`): Human/agent actionable task description.
  - `difficulty` (`int`, 1..5): Task complexity score.
  - `required_strengths` (`list[string]`): Desired functional capabilities.
  - `deadline_s` (`float`): Execution budget in seconds.
  - `token_budget` (`int`): Token allowance for LLM reasoning and repair.
- **Actions & Decisions**: Responding agents evaluate proposals and return `accept`, `counter`, or `reject` paired with explicit reason codes (`NegotiationDecision`).
- **Negotiation Policy**:
  - Each agent implements a pluggable `NegotiationPolicy` class.
  - **Default Policy (`DefaultNegotiationPolicy`)**:
    - **Accept**: Receiver's `complexity_level >= proposed difficulty`, required strengths overlap with its own, and `current_load < max_load`.
    - **Counter**: Task is feasible with relaxed constraints (extended deadline, higher token budget, or difficulty adjusted by at most $\pm 1$ level per round with documented reason). Maximum 2 rounds.
    - **Reject**: Agent cannot perform the task (e.g. zero strength overlap or complexity gap $> 1$), returning an explicit reason code.
- **Circuit Breaker Integration**: Every exchange increments authoritative session hops. If the breaker trips or round limit is exceeded, negotiation terminates gracefully with an explicit `"no_agreement"` result (no uncaught exception).
- **Whiteboard Audit Trail**: All proposals, counter-offers, reason codes, and final outcomes are persisted to the Whiteboard under `negotiation.<subtask_id>`.

#### Negotiation Sequence Diagrams

##### Successful Negotiation (Propose -> Counter -> Accept)
```mermaid
sequenceDiagram
    autonumber
    actor Initiator as Agent Alpha (Initiator)
    actor Responder as Agent Beta (Responder)
    participant WB as Whiteboard
    participant CB as CircuitBreaker

    Initiator->>CB: incr_hop(session_id) [Hop 1]
    Initiator->>Responder: uacp.negotiate (Round 1: Propose task, constraints)
    Responder->>Responder: Evaluate AgentCard limits (e.g. overloaded)
    Responder->>CB: incr_hop(session_id) [Hop 2]
    Responder-->>Initiator: uacp.negotiate (Round 1: Counter with modified_constraints)
    Initiator->>Initiator: Evaluate counter-offer (within 2-round limit)
    Initiator->>CB: incr_hop(session_id) [Hop 3]
    Initiator->>Responder: uacp.negotiate (Round 2: Accept modified constraints)
    Responder-->>Initiator: uacp.negotiate (Round 2: Acknowledged accepted)
    Initiator->>WB: put("negotiation.<subtask_id>", outcome=agreed)
```

##### Rejected Negotiation or Round-Limit Halt
```mermaid
sequenceDiagram
    autonumber
    actor Initiator as Agent Alpha
    actor Responder as Agent Beta
    participant WB as Whiteboard
    participant CB as CircuitBreaker

    alt Missing Capability or Infeasible Constraints
        Initiator->>CB: incr_hop(session_id) [Hop 1]
        Initiator->>Responder: uacp.negotiate (Round 1: Propose unknown task)
        Responder->>Responder: Inspect AgentCard (Capability missing / complexity gap > 1)
        Responder-->>Initiator: uacp.negotiate (Round 1: Reject, reason="CAPABILITY_NOT_FOUND")
        Initiator->>WB: put("negotiation.<subtask_id>", outcome=rejected)
    else Exceeding Maximum Rounds (Max 2 Rounds)
        Initiator->>Responder: uacp.negotiate (Round 2: Counter offer)
        Responder->>Responder: Check round_num > max_rounds (2)
        Responder-->>Initiator: uacp.negotiate (Action="no_agreement", reason="MAX_ROUNDS_REACHED")
        Initiator->>WB: put("negotiation.<subtask_id>", outcome=no_agreement)
    end
```

### 8.2 Session Handoff & Observer Invariants (`uacp.handoff`)

When transferring primary session control between agents:
1. **Cryptographic Integrity Verification**: The recipient verifies the session hash chain (`verify_chain`) AND confirms the latest chain tip matches `last_hash`. If state or event tampering is detected or hash mismatches, the handoff is rejected (`TamperDetected` or `rejected`).
2. **Ownership Transfer Event**: Upon successful verification, the recipient records a `"session.handoff"` event on the Whiteboard documenting previous owner, new owner, and metadata.
3. **Observer Write Restriction**: Upon transferring ownership, the originating agent transitions to observer mode (`is_observer=True` and registered in session `observers`). Both the agent client and backend enforce read-only access; any subsequent write attempts (`put`, `write_model`, `run_llm`) raise `ObserverWriteError` / `PermissionError`.

#### Task Handoff Sequence Diagram
```mermaid
sequenceDiagram
    autonumber
    actor PrevOwner as Previous Owner (Agent Alpha)
    actor NewOwner as New Owner (Agent Beta)
    participant WB as Whiteboard Backend

    PrevOwner->>WB: get_latest_hash(session_id)
    WB-->>PrevOwner: return last_hash
    PrevOwner->>NewOwner: uacp.handoff (session_ref, last_hash, summary)
    PrevOwner->>PrevOwner: set_observer(True)
    
    NewOwner->>WB: verify_chain(session_id)
    WB-->>NewOwner: VerificationReport(is_valid)
    NewOwner->>WB: get_latest_hash(session_id)
    WB-->>NewOwner: current_tip

    alt Valid Chain & Tip Matches last_hash
        NewOwner->>WB: append_event("session.handoff", prev=Alpha, new=Beta)
        NewOwner->>WB: add_observer(session_id, "agent_alpha")
        NewOwner-->>PrevOwner: task_result(status="success", summary="Handoff accepted")
        Note over PrevOwner,WB: Observer Write Protection Active
        PrevOwner-xWB: put(session_id, key, val) -> BLOCKED (ObserverWriteError)
    else Tampered Chain or Tip Mismatch
        NewOwner->>WB: append_event("session.handoff.rejected", reason)
        NewOwner-->>PrevOwner: task_result(status="rejected", is_valid=False)
        Note over PrevOwner,NewOwner: Ownership NOT transferred
    end
```



