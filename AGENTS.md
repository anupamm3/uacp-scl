# UACP-SCL — Master Context

## WHAT IT IS

1. **SCL (Semantic Context Layer):** a parser + AST + compiler for a small XML-like language
   (`<identity>`, `<variable>`, `<example>`, `<output>`) that compiles one agent definition into
   provider-native requests for different LLM providers.
2. **UACP (Unified Agent Communication Protocol):** JSON-RPC 2.0 based messaging between agents,
   with capability discovery, a shared temporary Whiteboard, a persistent knowledge store with
   contradiction resolution, and a circuit breaker (default max 5 hops) preventing infinite loops.

## HARD RULES

- Python 3.11+, full type hints, Pydantic v2 models, pytest, ruff and mypy clean.
- **FREE APIS ONLY** by default: Google Gemini API (free key), Groq (OpenAI-compatible endpoint),
  optional OpenRouter free models, optional Ollama (local). Never require a paid service.
- No provider SDK imports outside `src/uacp/adapters/`.
- Never hardcode API keys or model names. Keys come from environment variables (`.env`, git-ignored).
  Model names come from `config/models.yaml`.
- Tests must never call live APIs. Use `FakeAdapter`, httpx `MockTransport` or recorded fixtures.
- Small modules, no global mutable state, docstrings on public functions.
- If you are unsure about a provider's current parameter names, model names or limits, check the
  official documentation and tell me exactly what you verified. Do not guess.
- Work on **ONE step at a time**. Do not modify modules outside the step's scope unless I approve.

## HOW TO WORK

1. Read `AGENTS.md` and `docs/DECISIONS.md` first.
2. Restate the acceptance criteria and list files you will create or change. Wait for my OK.
3. Implement, with tests written alongside the code.
4. Run ruff, mypy and pytest. Fix failures.
5. Finish with: summary of changes, commands to run the tests, open questions, and any new entries
   for `docs/DECISIONS.md`.
