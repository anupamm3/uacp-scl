"""Unit and snapshot tests for the adapter base, FakeAdapter, retry helper, and QuotaGuard."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from uacp.adapters import (
    AdapterCapabilities,
    AuthError,
    CompiledRequest,
    FakeAdapter,
    Message,
    ModelRefusal,
    ProviderError,
    QuotaGuard,
    RateLimited,
    get_adapter,
    load_models_config,
    render_examples,
    with_retry,
)
from uacp.scl.context import RuntimeContext, bind
from uacp.scl.parser import parse

# ── 1. Compile Snapshot and Structure Tests ─────────────────────────


def test_fake_adapter_compile_snapshot(snapshot: Any) -> None:
    scl_text = (
        '<scl version="1">\n'
        '  <identity profile="analyst" tone="concise">\n'
        "    You evaluate financial statements.\n"
        "  </identity>\n"
        '  <variable name="ledger" source="runtime.data.ledger" type="json">\n'
        "    Ledger payload:\n"
        "    {{ledger}}\n"
        "  </variable>\n"
        '  <example pair="true">\n'
        "    <input>Analyze Q1</input>\n"
        "    <output>Q1 was profitable.</output>\n"
        "  </example>\n"
        '  <output format="json_schema" strict="true">\n'
        '    {"type": "object", "properties": {"healthy": {"type": "boolean"}}, '
        '"required": ["healthy"]}\n'
        "  </output>\n"
        "</scl>"
    )
    doc = parse(scl_text)
    payload = {"runtime": {"data": {"ledger": {"revenue": 500_000, "expenses": 320_000}}}}
    ctx = RuntimeContext(payload)
    bound = bind(doc, ctx)

    adapter = FakeAdapter(model_id="fake-echo-v1")
    req = adapter.compile(bound, "Please evaluate the current health status.")

    assert req.model_dump() == snapshot


def test_fake_adapter_compile_system_role_disabled() -> None:
    scl_text = (
        '<scl version="1">\n'
        '  <identity profile="bot" tone="helpful">Assistant persona.</identity>\n'
        "</scl>"
    )
    doc = parse(scl_text)
    bound = bind(doc, RuntimeContext())

    adapter = FakeAdapter(
        capabilities=AdapterCapabilities(supports_system_role=False, supports_native_schema=False)
    )
    req = adapter.compile(bound, "Hello")

    assert req.system is None
    assert req.messages[0].role == "user"
    assert "Assistant persona." in req.messages[0].content
    assert req.messages[1].role == "assistant"
    assert req.messages[2].role == "user"
    assert req.messages[2].content == "Hello"


def test_render_examples_helper() -> None:
    scl_text = (
        "<scl>\n"
        '  <identity profile="tester">Persona</identity>\n'
        '  <example pair="true"><input>A</input><output>B</output></example>\n'
        '  <example pair="false"><input>C</input><output>D</output></example>\n'
        "</scl>"
    )
    doc = parse(scl_text)
    pairs = render_examples(doc)
    assert pairs == [("A", "B"), ("C", "D")]


# ── 2. FakeAdapter Invocation and Replay ─────────────────────────────


@pytest.mark.asyncio
async def test_fake_adapter_scripted_replies_and_token_usage() -> None:
    script: list[str | dict[str, Any] | Exception] = [
        "First reply text",
        {"healthy": True, "score": 95},
        RateLimited("Temporarily unavailable", retry_after=2.0),
    ]
    adapter = FakeAdapter(script=script)

    req1 = CompiledRequest(
        provider="fake",
        model="fake-echo-v1",
        messages=[Message(role="user", content="Test 1")],
    )
    res1 = await adapter.invoke(req1)
    assert res1.text == "First reply text"
    assert res1.content == "First reply text"
    assert res1.parsed is None
    assert res1.usage.input_tokens >= 1
    assert res1.usage.output_tokens >= 1
    assert len(adapter.recorded_requests) == 1

    req2 = CompiledRequest(
        provider="fake",
        model="fake-echo-v1",
        messages=[Message(role="user", content="Test 2")],
    )
    res2 = await adapter.invoke(req2)
    assert res2.parsed == {"healthy": True, "score": 95}
    assert json.loads(res2.text) == {"healthy": True, "score": 95}

    req3 = CompiledRequest(
        provider="fake",
        model="fake-echo-v1",
        messages=[Message(role="user", content="Test 3")],
    )
    with pytest.raises(RateLimited) as exc:
        await adapter.invoke(req3)
    assert exc.value.retry_after == 2.0

    # Exhausted script raises IndexError
    with pytest.raises(IndexError, match="No scripted replies remaining"):
        await adapter.invoke(req3)


@pytest.mark.asyncio
async def test_fake_adapter_from_file(tmp_path: Path) -> None:
    json_file = tmp_path / "replay.json"
    json_file.write_text(
        json.dumps(["replay turn 1", {"action": "confirm"}]),
        encoding="utf-8",
    )

    adapter = FakeAdapter.from_file(json_file)
    req = CompiledRequest(
        provider="fake",
        model="fake-echo-v1",
        messages=[Message(role="user", content="Hi")],
    )

    res1 = await adapter.invoke(req)
    assert res1.text == "replay turn 1"

    res2 = await adapter.invoke(req)
    assert res2.parsed == {"action": "confirm"}


# ── 3. Retry Helper with Mock Clock and Sleep ───────────────────────


@pytest.mark.asyncio
async def test_with_retry_honours_retry_after() -> None:
    attempts = 0
    slept_durations: list[float] = []

    async def mock_sleep(seconds: float) -> None:
        slept_durations.append(seconds)

    async def flaky_call() -> str:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise RateLimited("Too Many Requests", retry_after=4.5)
        return "success"

    result = await with_retry(
        flaky_call,
        jitter=False,
        sleep=mock_sleep,
    )

    assert result == "success"
    assert attempts == 2
    assert slept_durations == [4.5]


@pytest.mark.asyncio
async def test_with_retry_exponential_backoff() -> None:
    attempts = 0
    slept_durations: list[float] = []

    async def mock_sleep(seconds: float) -> None:
        slept_durations.append(seconds)

    async def error_call() -> str:
        nonlocal attempts
        attempts += 1
        if attempts < 3:
            raise ProviderError(f"Temporary server error {attempts}")
        return "recovered"

    result = await with_retry(
        error_call,
        initial_delay=1.0,
        backoff_factor=2.0,
        jitter=False,
        sleep=mock_sleep,
    )

    assert result == "recovered"
    assert attempts == 3
    assert slept_durations == [1.0, 2.0]


@pytest.mark.asyncio
async def test_with_retry_non_retryable_aborts_immediately() -> None:
    attempts = 0

    async def auth_failed() -> None:
        nonlocal attempts
        attempts += 1
        raise AuthError("Invalid API key")

    with pytest.raises(AuthError):
        await with_retry(auth_failed, max_retries=5)

    assert attempts == 1


@pytest.mark.asyncio
async def test_with_retry_model_refusal_aborts_immediately() -> None:
    attempts = 0

    async def refused() -> None:
        nonlocal attempts
        attempts += 1
        raise ModelRefusal("Content policy violation")

    with pytest.raises(ModelRefusal):
        await with_retry(refused, max_retries=3)

    assert attempts == 1


# ── 4. QuotaGuard Rate Limiting ─────────────────────────────────────


@pytest.mark.asyncio
async def test_quota_guard_minute_sliding_window_raises() -> None:
    current_time = 100.0

    def mock_clock() -> float:
        return current_time

    guard = QuotaGuard(
        requests_per_minute=2,
        clock=mock_clock,
        raise_on_limit=True,
    )

    # 1st and 2nd requests pass
    await guard.acquire()
    current_time += 10.0
    await guard.acquire()

    # 3rd request within the same minute raises RateLimited
    current_time += 10.0  # now at t=120.0, first was at t=100.0 (elapsed 20s < 60s)
    with pytest.raises(RateLimited) as exc:
        await guard.acquire()

    assert exc.value.retry_after == pytest.approx(40.0)

    # Advance beyond 60s window from 1st request (100.0 + 60.0 = 160.0)
    current_time = 161.0
    await guard.acquire()  # Now succeeds because 1st request dropped off


@pytest.mark.asyncio
async def test_quota_guard_sleeps_when_not_raising() -> None:
    current_time = 0.0
    sleeps: list[float] = []

    def mock_clock() -> float:
        return current_time

    async def mock_sleep(seconds: float) -> None:
        nonlocal current_time
        sleeps.append(seconds)
        current_time += seconds

    guard = QuotaGuard(
        requests_per_minute=1,
        clock=mock_clock,
        sleep=mock_sleep,
        raise_on_limit=False,
    )

    await guard.acquire()  # recorded at t=0.0

    # Next acquire needs to sleep 60.0s
    await guard.acquire()
    assert sleeps == [60.0]
    assert current_time == 60.0


@pytest.mark.asyncio
async def test_quota_guard_daily_limit_raises() -> None:
    current_time = 1_000.0

    def mock_clock() -> float:
        return current_time

    guard = QuotaGuard(
        daily_requests=2,
        clock=mock_clock,
        raise_on_limit=True,
    )

    await guard.acquire()
    current_time += 100.0
    await guard.acquire()

    # 3rd request on same day raises
    with pytest.raises(RateLimited) as exc:
        await guard.acquire()

    assert exc.value.retry_after == pytest.approx(86400.0 - 100.0)


# ── 5. Config Loader and get_adapter Factory ────────────────────────


def test_load_models_config_and_env_overrides(monkeypatch: pytest.MonkeyPatch) -> None:
    configs = load_models_config()
    assert "fake" in configs
    assert "alpha_default" in configs
    assert configs["fake"].provider == "fake"
    assert configs["fake"].limits.requests_per_minute == 999

    # Test override
    monkeypatch.setenv("UACP_MODEL_FAKE_RPM", "42")
    monkeypatch.setenv("UACP_MODEL_FAKE_MODEL_ID", "custom-fake-model")
    reloaded = load_models_config()
    assert reloaded["fake"].limits.requests_per_minute == 42
    assert reloaded["fake"].model_id == "custom-fake-model"


def test_get_adapter_factory() -> None:
    adapter = get_adapter("fake")
    assert isinstance(adapter, FakeAdapter)

    with pytest.raises(KeyError, match="not found"):
        get_adapter("non_existent_alias")

    with pytest.raises(NotImplementedError, match="is not implemented yet"):
        get_adapter("local_default")
