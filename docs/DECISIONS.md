# Design Decisions

| ID  | Decision |
|-----|----------|
| D-1 | `<variable>` gets a required `name` attribute. `source` is a dotted path into runtime data. The body is an optional template wrapping `{{name}}`. |
| D-2 | A document has an implicit root. An optional `<scl version="1">…</scl>` wrapper is allowed. Mismatched tags raise a clear error with line and column. The blueprint's stray `</excel>` is a typo. |
| D-3 | MCP is a tool protocol, not a provider API. It is out of scope; it can be a future extension. |
| D-4 | Model IDs live only in `config/models.yaml` and environment variables, never in code. |
| D-5 | The UACP message from the blueprint is wrapped in a JSON-RPC 2.0 envelope as `params`. |
| D-6 | Generated code from Agent Beta runs only inside a sandbox (restricted subprocess with timeout). |
| D-7 | Contradiction resolution uses a documented scoring formula (Step 20) with tunable weights. |
| D-8 | Only free providers are used by default. The OpenAI-compatible adapter makes paid providers a config change. |
| D-9 | Whiteboard defaults to in-memory with an SQLite option. Redis is an optional backend. |
