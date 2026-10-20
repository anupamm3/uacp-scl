"""Unit tests for GeminiAdapter (offline, mocked network calls only)."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest
from google.genai import errors, types

from uacp.adapters.base import (
    AuthError,
    CompiledRequest,
    Message,
    ModelRefusal,
    ProviderError,
    RateLimited,
    TruncatedOutput,
)
from uacp.adapters.gemini import GeminiAdapter
from uacp.adapters.registry import QuotaGuard, get_adapter
from uacp.scl.context import RuntimeContext, bind
from uacp.scl.parser import parse_file


def test_compile_financial_auditor_snapshot(snapshot: object) -> None:
    """Test compiling examples/financial_auditor.scl produces matching snapshot."""
    scl_path = Path(__file__).resolve().parents[2] / "examples" / "financial_auditor.scl"
    doc = parse_file(scl_path)

    ctx = RuntimeContext(
        {
            "runtime": {
                "payload": {
                    "ledger_v1": json.dumps(
                        [
                            {
                                "id": "TX-100",
                                "batch": "B1",
                                "delta": -500.0,
                                "type": "transfer",
                                "evidence": "Valid",
                            }
                        ]
                    )
                }
            }
        }
    )
    bound_doc = bind(doc, ctx)

    cfg_path = Path(__file__).resolve().parents[2] / "config" / "models.yaml"
    adapter = GeminiAdapter(config_path=cfg_path)
    compiled = adapter.compile(bound_doc, "Analyze ledger transactions for discrepancies.")

    assert compiled.provider == "gemini"
    assert compiled.model == "gemini-2.0-flash"
    assert compiled.output_mode == "native_schema"
    assert compiled.system is not None
    assert "Senior Forensic Financial Auditor" in compiled.system
    assert "Profile: financial_auditor" in compiled.system

    # Verify examples and data block in messages
    assert len(compiled.messages) == 3
    assert compiled.messages[0].role == "user"
    assert "TX-1092" in compiled.messages[0].content
    assert compiled.messages[1].role == "assistant"
    assert "unauthorized_disbursement" in compiled.messages[1].content
    assert compiled.messages[2].role == "user"
    assert '<variable name="transaction_manifest">' in compiled.messages[2].content
    assert "Analyze ledger transactions for discrepancies." in compiled.messages[2].content

    # Verify schema normalized for Gemini ($schema removed)
    assert compiled.response_schema is not None
    assert "$schema" not in compiled.response_schema
    assert "audit_complete" in compiled.response_schema["properties"]

    # Match syrupy snapshot
    assert compiled.model_dump() == snapshot


@pytest.mark.asyncio
async def test_invoke_rate_limited_mapping() -> None:
    """Ensure HTTP 429 errors from Google GenAI SDK map to RateLimited."""
    mock_client = MagicMock()
    mock_generate = AsyncMock()
    mock_client.aio.models.generate_content = mock_generate

    mock_generate.side_effect = errors.ClientError(
        429, {"error": {"message": "RESOURCE_EXHAUSTED: quota exceeded"}}
    )

    adapter = GeminiAdapter(client=mock_client)
    request = CompiledRequest(
        provider="gemini",
        model="gemini-2.0-flash",
        messages=[Message(role="user", content="Ping")],
    )

    with pytest.raises(RateLimited) as exc_info:
        await adapter.invoke(request)

    assert exc_info.value.provider == "gemini"
    assert "rate limit exceeded" in str(exc_info.value)


@pytest.mark.asyncio
async def test_invoke_auth_error_mapping() -> None:
    """Ensure HTTP 401/403 errors map to AuthError."""
    mock_client = MagicMock()
    mock_generate = AsyncMock()
    mock_client.aio.models.generate_content = mock_generate

    # Test 401
    mock_generate.side_effect = errors.ClientError(
        401, {"error": {"message": "API key not valid"}}
    )

    adapter = GeminiAdapter(client=mock_client)
    request = CompiledRequest(
        provider="gemini",
        model="gemini-2.0-flash",
        messages=[Message(role="user", content="Ping")],
    )

    with pytest.raises(AuthError) as exc_info_401:
        await adapter.invoke(request)
    assert exc_info_401.value.provider == "gemini"

    # Test 403
    mock_generate.side_effect = errors.ClientError(
        403, {"error": {"message": "PERMISSION_DENIED"}}
    )
    with pytest.raises(AuthError) as exc_info_403:
        await adapter.invoke(request)
    assert exc_info_403.value.provider == "gemini"


@pytest.mark.asyncio
async def test_invoke_safety_block_mapping() -> None:
    """Ensure safety blocks and finish reasons map to ModelRefusal."""
    mock_client = MagicMock()
    mock_generate = AsyncMock()
    mock_client.aio.models.generate_content = mock_generate

    # 1. Candidate finish_reason = SAFETY
    mock_generate.return_value = types.GenerateContentResponse(
        candidates=[
            types.Candidate(
                finish_reason=types.FinishReason.SAFETY,
                finish_message="Candidate violated safety policy",
            )
        ]
    )

    adapter = GeminiAdapter(client=mock_client)
    request = CompiledRequest(
        provider="gemini",
        model="gemini-2.0-flash",
        messages=[Message(role="user", content="Dangerous query")],
    )

    with pytest.raises(ModelRefusal) as exc_info:
        await adapter.invoke(request)
    assert "safety policy" in str(exc_info.value)

    # 2. Prompt blocked
    mock_generate.return_value = types.GenerateContentResponse(
        prompt_feedback=types.GenerateContentResponsePromptFeedback(
            block_reason=types.BlockedReason.SAFETY
        )
    )
    with pytest.raises(ModelRefusal) as exc_info_prompt:
        await adapter.invoke(request)
    assert "Prompt blocked" in str(exc_info_prompt.value)

    # 3. Empty candidates
    mock_generate.return_value = types.GenerateContentResponse(candidates=[])
    with pytest.raises(ModelRefusal) as exc_info_empty:
        await adapter.invoke(request)
    assert "No candidate responses" in str(exc_info_empty.value)


@pytest.mark.asyncio
async def test_invoke_truncation_mapping() -> None:
    """Ensure finish_reason=MAX_TOKENS maps to TruncatedOutput."""
    mock_client = MagicMock()
    mock_generate = AsyncMock()
    mock_client.aio.models.generate_content = mock_generate

    mock_generate.return_value = types.GenerateContentResponse(
        candidates=[types.Candidate(finish_reason=types.FinishReason.MAX_TOKENS)]
    )

    adapter = GeminiAdapter(client=mock_client)
    request = CompiledRequest(
        provider="gemini",
        model="gemini-2.0-flash",
        messages=[Message(role="user", content="Long essay")],
    )

    with pytest.raises(TruncatedOutput) as exc_info:
        await adapter.invoke(request)
    assert "max token limit" in str(exc_info.value)


@pytest.mark.asyncio
async def test_invoke_timeout_mapping() -> None:
    """Ensure timeout raises ProviderError."""
    mock_client = MagicMock()
    mock_generate = AsyncMock(side_effect=TimeoutError())
    mock_client.aio.models.generate_content = mock_generate

    adapter = GeminiAdapter(client=mock_client, timeout=0.01)
    request = CompiledRequest(
        provider="gemini",
        model="gemini-2.0-flash",
        messages=[Message(role="user", content="Hello")],
    )

    with pytest.raises(ProviderError) as exc_info:
        await adapter.invoke(request)
    assert "timed out" in str(exc_info.value)


@pytest.mark.asyncio
async def test_invoke_success_with_parsed_and_usage() -> None:
    """Ensure successful invocation returns valid LLMResult with usage and parsed JSON."""
    mock_client = MagicMock()
    mock_generate = AsyncMock()
    mock_client.aio.models.generate_content = mock_generate

    mock_generate.return_value = types.GenerateContentResponse(
        candidates=[
            types.Candidate(
                content=types.Content(
                    role="model",
                    parts=[types.Part.from_text(text='{"audit_complete": true, "anomalies": []}')],
                ),
                finish_reason=types.FinishReason.STOP,
            )
        ],
        usage_metadata=types.GenerateContentResponseUsageMetadata(
            prompt_token_count=120,
            candidates_token_count=35,
            total_token_count=155,
        ),
    )

    adapter = GeminiAdapter(client=mock_client)
    request = CompiledRequest(
        provider="gemini",
        model="gemini-2.0-flash",
        messages=[Message(role="user", content="Audit payload")],
        output_mode="native_schema",
    )

    result = await adapter.invoke(request)
    assert result.text == '{"audit_complete": true, "anomalies": []}'
    assert result.parsed == {"audit_complete": True, "anomalies": []}
    assert result.usage.input_tokens == 120
    assert result.usage.output_tokens == 35
    assert result.usage.requests == 1
    assert result.raw_meta["finish_reason"] == "FinishReason.STOP"


@pytest.mark.asyncio
async def test_invoke_schema_rejection_fallback() -> None:
    """Ensure 400 rejection of native schema falls back to json_mode call."""
    mock_client = MagicMock()
    mock_generate = AsyncMock()
    mock_client.aio.models.generate_content = mock_generate

    # 1st call fails due to invalid response schema
    # 2nd call succeeds with json_mode
    mock_generate.side_effect = [
        errors.ClientError(400, {"error": {"message": "Invalid schema for response_schema"}}),
        types.GenerateContentResponse(
            candidates=[
                types.Candidate(
                    content=types.Content(
                        role="model",
                        parts=[types.Part.from_text(text='{"recovered": true}')],
                    ),
                    finish_reason=types.FinishReason.STOP,
                )
            ],
            usage_metadata=types.GenerateContentResponseUsageMetadata(
                prompt_token_count=10, candidates_token_count=5
            ),
        ),
    ]

    adapter = GeminiAdapter(client=mock_client)
    request = CompiledRequest(
        provider="gemini",
        model="gemini-2.0-flash",
        messages=[Message(role="user", content="Ping")],
        output_mode="native_schema",
        response_schema={"type": "object", "properties": {"a": {"type": "string"}}},
    )

    result = await adapter.invoke(request)
    assert result.parsed == {"recovered": True}
    assert mock_generate.await_count == 2


@pytest.mark.asyncio
async def test_quota_guard_integration() -> None:
    """Ensure QuotaGuard limits invocations and raises RateLimited when limit reached."""
    mock_client = MagicMock()
    mock_generate = AsyncMock()
    mock_client.aio.models.generate_content = mock_generate
    mock_generate.return_value = types.GenerateContentResponse(
        candidates=[
            types.Candidate(
                content=types.Content(
                    role="model",
                    parts=[types.Part.from_text(text="ok")],
                ),
                finish_reason=types.FinishReason.STOP,
            )
        ]
    )

    guard = QuotaGuard(requests_per_minute=2, raise_on_limit=True)
    adapter = GeminiAdapter(client=mock_client, quota_guard=guard)
    request = CompiledRequest(
        provider="gemini",
        model="gemini-2.0-flash",
        messages=[Message(role="user", content="Hi")],
    )

    # Invocations 1 and 2 succeed
    await adapter.invoke(request)
    await adapter.invoke(request)

    # Invocation 3 hits minute quota limit
    with pytest.raises(RateLimited) as exc_info:
        await adapter.invoke(request)

    assert "Minute quota exceeded" in str(exc_info.value)


def test_registry_integration() -> None:
    """Ensure get_adapter('alpha_default') returns GeminiAdapter instance."""
    adapter = get_adapter("alpha_default")
    assert isinstance(adapter, GeminiAdapter)
    assert adapter.name == "gemini"
    assert adapter.model_id == "gemini-2.0-flash"
    assert adapter.quota_guard.requests_per_minute == 15


def test_missing_api_key_raises_auth_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """Ensure accessing client without GEMINI_API_KEY set raises AuthError."""
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    adapter = GeminiAdapter()
    with pytest.raises(AuthError) as exc_info:
        _ = adapter.client
    assert "GEMINI_API_KEY is not set" in str(exc_info.value)
