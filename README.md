# UACP-SCL

**Unified Agent Communication Protocol & Semantic Context Layer**

> 🚧 Pre-alpha — under active development.

## Overview

- **SCL** — a small XML-like language for defining LLM agent personas, compiled to provider-native
  requests.
- **UACP** — JSON-RPC 2.0 messaging between agents with capability discovery, a shared Whiteboard,
  a persistent Knowledge Store, and a circuit breaker.

## Quick Start

```bash
# Clone and install (editable, with dev tools)
pip install -e ".[dev]"

# Run checks
ruff check src/ tests/
mypy src/uacp/
pytest
```

## Project Layout

See [AGENTS.md](AGENTS.md) for the master context and [docs/DECISIONS.md](docs/DECISIONS.md) for
design decisions.

