"""Unit tests for OpenAICompatAdapter (offline, mocked network calls only)."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest
from google.genai import types as gemini_types

from uacp.adapters.base import (
    AdapterCapabilities,
    AuthError,
    CompiledRequest,
    Message,
    ModelRefusal,
    TruncatedOutput,
)
from uacp.adapters.enforcer import enforce, invoke_with_repair
from uacp.adapters.fake_adapter import FakeAdapter
from uacp.adapters.gemini import GeminiAdapter
from uacp.adapters.openai_compat_adapter import OpenAICompatAdapter
from uacp.adapters.registry import QuotaGuard, get_adapter
from uacp.scl.context import RuntimeContext, bind
from uacp.scl.parser import parse

TEST_SCL_STRICT = """<scl version="1">
  <identity profile="analyst">
    You analyze input and respond strictly in JSON.
  </identity>
  <example>
    <input>alpha</input>
    <output>{"metric": "alpha", "score": 10}</output>
  </example>
  <variable name="runtime_val" source="payload.val">val: {{runtime_val}}</variable>
  <output format="json_schema" strict="true">
    {
      "type": "object",
      "required": ["metric", "score"],
      "properties": {
        "metric": {"type": "string"},
        "score": {"type": "integer"}
      }
    }
  </output>
</scl>"""

TEST_SCL_LOOSE = """<scl version="1">
  <identity profile="loose_bot">
    Loose JSON bot.
  </identity>
  <output format="json_schema" strict="false">
    {
      "type": "object",
      "properties": {
        "notes": {"type": "string"}
      }
    }
  </output>
</scl>"""


def test_compile_native_schema_mode() -> None:
    """When strict=True and native schema is supported, output_mode is native_schema."""
    doc = parse(TEST_SCL_STRICT)
    bound_doc = bind(doc, RuntimeContext({"payload": {"val": "test123"}}))

    adapter = OpenAICompatAdapter(
        name="groq",
        model_id="test-model",
        capabilities=AdapterCapabilities(supports_native_schema=True),
    )
    req = adapter.compile(bound_doc, "Analyze payload")

    assert req.output_mode == "native_schema"
    assert req.response_schema is not None
    # Verify openai_strict normalization applied
    assert req.response_schema.get("additionalProperties") is False
    assert set(req.response_schema.get("required", [])) == {"metric", "score"}
    assert req.system is not None
    assert "Profile: analyst" in req.system
    assert "You analyze input and respond strictly in JSON." in req.system
    # Examples
    assert len(req.messages) == 3
    assert req.messages[0].role == "user"
    assert req.messages[0].content == "alpha"
    assert req.messages[1].role == "assistant"
    assert req.messages[1].content == '{"metric": "alpha", "score": 10}'
    # Final user message contains data block + user prompt
    assert '<variable name="runtime_val">' in req.messages[2].content
    assert "Analyze payload" in req.messages[2].content


def test_compile_json_mode() -> None:
    """When native schema is unsupported or strict=False, output_mode is json_mode."""
    doc = parse(TEST_SCL_LOOSE)
    bound_doc = bind(doc, RuntimeContext({}))

    adapter = OpenAICompatAdapter(
        name="groq",
        model_id="test-model",
        capabilities=AdapterCapabilities(supports_native_schema=True),
    )
    req = adapter.compile(bound_doc, "Generate notes")

    assert req.output_mode == "json_mode"
    assert req.system is not None
    assert "Loose JSON bot." in req.system
    assert "You must respond in valid JSON" in req.system


def test_compile_prompt_only_mode() -> None:
    """When format is not json_schema, output_mode is prompt_only."""
    scl_text = (
        '<scl><identity profile="writer">Writer</identity>'
        '<output format="markdown"># Notes</output></scl>'
    )
    doc = parse(scl_text)
    bound_doc = bind(doc, RuntimeContext({}))

    adapter = OpenAICompatAdapter(name="groq", model_id="test-model")
    req = adapter.compile(bound_doc, "Write markdown")

    assert req.output_mode == "prompt_only"
    assert req.response_schema is None


@pytest.mark.asyncio
async def test_invoke_native_schema_mock_transport() -> None:
    """Test invoking in native_schema mode using httpx.MockTransport."""
    captured_requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        captured_requests.append(request)
        body = json.loads(request.read())
        assert body["response_format"]["type"] == "json_schema"
        assert body["response_format"]["json_schema"]["strict"] is True

        payload = {
            "id": "chatcmpl-test",
            "object": "chat.completion",
            "created": 1234567890,
            "model": "test-model",
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": json.dumps({"metric": "accuracy", "score": 98}),
                    },
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 20, "completion_tokens": 15, "total_tokens": 35},
        }
        return httpx.Response(200, json=payload)

    http_client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    adapter = OpenAICompatAdapter(
        name="groq",
        model_id="test-model",
        api_key="mock-key",
        base_url="https://api.groq.com/openai/v1",
        http_client=http_client,
    )

    doc = parse(TEST_SCL_STRICT)
    bound_doc = bind(doc, RuntimeContext({"payload": {"val": "1"}}))
    compiled = adapter.compile(bound_doc, "Run eval")

    result = await adapter.invoke(compiled)
    assert result.parsed == {"metric": "accuracy", "score": 98}
    assert result.usage.input_tokens == 20
    assert result.usage.output_tokens == 15
    assert len(captured_requests) == 1


@pytest.mark.asyncio
async def test_invoke_json_mode_mock_transport() -> None:
    """Test invoking in json_mode using httpx.MockTransport."""

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.read())
        assert body["response_format"]["type"] == "json_object"

        payload = {
            "choices": [
                {
                    "message": {"role": "assistant", "content": '{"notes": "all good"}'},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5},
        }
        return httpx.Response(200, json=payload)

    http_client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    adapter = OpenAICompatAdapter(
        name="groq",
        model_id="test-model",
        api_key="mock-key",
        http_client=http_client,
        capabilities=AdapterCapabilities(supports_native_schema=False),
    )

    doc = parse(TEST_SCL_LOOSE)
    bound_doc = bind(doc, RuntimeContext({}))
    compiled = adapter.compile(bound_doc, "Notes please")

    result = await adapter.invoke(compiled)
    assert result.parsed == {"notes": "all good"}


@pytest.mark.asyncio
async def test_invoke_rate_limited_retry_after() -> None:
    """Test 429 rate limit retry honoring Retry-After header."""
    attempts = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            return httpx.Response(
                429,
                headers={"retry-after": "0.01"},
                json={
                    "error": {
                        "message": "Rate limit exceeded. Please wait.",
                        "type": "tokens",
                        "code": "rate_limit_exceeded",
                    }
                },
            )
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {"role": "assistant", "content": '{"status": "recovered"}'},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"prompt_tokens": 5, "completion_tokens": 5},
            },
        )

    http_client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    adapter = OpenAICompatAdapter(
        name="groq",
        model_id="test-model",
        api_key="mock-key",
        http_client=http_client,
    )

    compiled = CompiledRequest(
        provider="groq",
        model="test-model",
        messages=[Message(role="user", content="Ping")],
    )

    result = await adapter.invoke(compiled)
    assert attempts == 2
    assert result.parsed == {"status": "recovered"}


@pytest.mark.asyncio
async def test_invoke_truncation_mapping() -> None:
    """Ensure finish_reason='length' raises TruncatedOutput."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {"role": "assistant", "content": '{"incomplete":'},
                        "finish_reason": "length",
                    }
                ]
            },
        )

    http_client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    adapter = OpenAICompatAdapter(
        name="groq",
        model_id="test-model",
        api_key="mock-key",
        http_client=http_client,
    )

    compiled = CompiledRequest(
        provider="groq",
        model="test-model",
        messages=[Message(role="user", content="Too long")],
    )

    with pytest.raises(TruncatedOutput) as exc_info:
        await adapter.invoke(compiled)
    assert "max token limit" in str(exc_info.value)


@pytest.mark.asyncio
async def test_invoke_refusal_mapping() -> None:
    """Ensure content_filter or refusal raises ModelRefusal."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": None,
                            "refusal": "I cannot fulfill this request.",
                        },
                        "finish_reason": "content_filter",
                    }
                ]
            },
        )

    http_client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    adapter = OpenAICompatAdapter(
        name="groq",
        model_id="test-model",
        api_key="mock-key",
        http_client=http_client,
    )

    compiled = CompiledRequest(
        provider="groq",
        model="test-model",
        messages=[Message(role="user", content="Harmful prompt")],
    )

    with pytest.raises(ModelRefusal) as exc_info:
        await adapter.invoke(compiled)
    assert "cannot fulfill" in str(exc_info.value)


@pytest.mark.asyncio
async def test_invoke_auth_error_mapping() -> None:
    """Ensure 401 Unauthorized raises AuthError."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            401,
            json={"error": {"message": "Invalid API key", "type": "invalid_request_error"}},
        )

    http_client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    adapter = OpenAICompatAdapter(
        name="groq",
        model_id="test-model",
        api_key="bad-key",
        http_client=http_client,
    )

    compiled = CompiledRequest(
        provider="groq",
        model="test-model",
        messages=[Message(role="user", content="Test")],
    )

    with pytest.raises(AuthError) as exc_info:
        await adapter.invoke(compiled)
    assert "authentication failed" in str(exc_info.value)


@pytest.mark.asyncio
async def test_bad_json_repaired_by_repair_loop() -> None:
    """Test that malformed JSON returned on first attempt is corrected by invoke_with_repair."""
    attempts = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            # First attempt returns invalid JSON violating schema
            content = "Here is the result: {metric: missing_quotes}"
        else:
            # Repaired attempt returns valid JSON adhering to schema
            content = '{"metric": "repaired_metric", "score": 100}'

        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {"role": "assistant", "content": content},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"prompt_tokens": 10, "completion_tokens": 10},
            },
        )

    http_client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    adapter = OpenAICompatAdapter(
        name="groq",
        model_id="test-model",
        api_key="mock-key",
        http_client=http_client,
    )

    doc = parse(TEST_SCL_STRICT)
    bound_doc = bind(doc, RuntimeContext({"payload": {"val": "1"}}))
    compiled = adapter.compile(bound_doc, "Evaluate")

    assert doc.output is not None
    result = await invoke_with_repair(adapter, compiled, doc.output, max_repairs=2)
    assert attempts == 2
    assert result == {"metric": "repaired_metric", "score": 100}


@pytest.mark.asyncio
async def test_cross_provider_parity() -> None:
    """Parity test: Fake, Gemini, and OpenAICompat produce identical parsed objects."""
    target_json = json.dumps({"metric": "parity_check", "score": 42})
    expected_parsed = {"metric": "parity_check", "score": 42}

    doc = parse(TEST_SCL_STRICT)
    bound_doc = bind(doc, RuntimeContext({"payload": {"val": "parity"}}))

    # 1. FakeAdapter
    fake_adapter = FakeAdapter(script=[target_json])
    fake_compiled = fake_adapter.compile(bound_doc, "Run parity")
    fake_res = await fake_adapter.invoke(fake_compiled)
    assert doc.output is not None
    fake_enforced = enforce(doc.output, fake_res.text)

    # 2. GeminiAdapter (mocked)
    gemini_client = MagicMock()
    gemini_generate = AsyncMock()
    gemini_client.aio.models.generate_content = gemini_generate
    gemini_generate.return_value = gemini_types.GenerateContentResponse(
        candidates=[
            gemini_types.Candidate(
                finish_reason=gemini_types.FinishReason.STOP,
                content=gemini_types.Content(
                    parts=[gemini_types.Part.from_text(text=target_json)]
                ),
            )
        ]
    )
    gemini_adapter = GeminiAdapter(
        client=gemini_client,
        quota_guard=QuotaGuard(),
    )
    gemini_compiled = gemini_adapter.compile(bound_doc, "Run parity")
    gemini_res = await gemini_adapter.invoke(gemini_compiled)
    gemini_enforced = enforce(doc.output, gemini_res.text)

    # 3. OpenAICompatAdapter (mocked)
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {"role": "assistant", "content": target_json},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"prompt_tokens": 10, "completion_tokens": 10},
            },
        )

    openai_client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    openai_adapter = OpenAICompatAdapter(
        name="groq",
        model_id="test-model",
        api_key="mock-key",
        http_client=openai_client,
        quota_guard=QuotaGuard(),
    )
    openai_compiled = openai_adapter.compile(bound_doc, "Run parity")
    openai_res = await openai_adapter.invoke(openai_compiled)
    openai_enforced = enforce(doc.output, openai_res.text)

    # Cross-provider parity verification
    assert fake_enforced == expected_parsed
    assert gemini_enforced == expected_parsed
    assert openai_enforced == expected_parsed
    assert fake_enforced == gemini_enforced == openai_enforced


def test_registry_get_adapter_beta_default() -> None:
    """Ensure get_adapter('beta_default') instantiates OpenAICompatAdapter with groq provider."""
    adapter = get_adapter("beta_default")
    assert isinstance(adapter, OpenAICompatAdapter)
    assert adapter.name == "groq"
    assert adapter.base_url == "https://api.groq.com/openai/v1"
    assert adapter.api_key_env == "GROQ_API_KEY"
    assert adapter.capabilities.supports_native_schema is True
