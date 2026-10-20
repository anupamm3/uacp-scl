"""Base adapter interface, data models, exceptions, and retry helpers."""

from __future__ import annotations

import asyncio
import random
import time
from abc import ABC, abstractmethod
from collections.abc import Callable, Coroutine
from typing import Any, Literal, TypeVar

from pydantic import BaseModel, ConfigDict, Field, model_validator

from uacp.scl.ast import Document
from uacp.scl.context import BoundDocument


class Message(BaseModel):
    """A single chat message with a role and string content."""

    role: Literal["system", "user", "assistant"]
    content: str


class CompiledRequest(BaseModel):
    """Provider-agnostic compiled request ready for adapter invocation."""

    provider: str
    model: str
    messages: list[Message]
    system: str | None = None
    response_schema: dict[str, Any] | None = None
    output_mode: Literal["native_schema", "json_mode", "prompt_only"] = "prompt_only"
    params: dict[str, Any] = Field(default_factory=dict)


class Usage(BaseModel):
    """Token consumption and request accounting metadata."""

    input_tokens: int = 0
    output_tokens: int = 0
    requests: int = 1


class LLMResult(BaseModel):
    """Result of an LLM invocation containing text, parsed data, and usage."""

    text: str = ""
    content: str = ""
    parsed: dict[str, Any] | list[Any] | None = None
    usage: Usage = Field(default_factory=Usage)
    raw_meta: dict[str, Any] = Field(default_factory=dict)

    model_config = ConfigDict(populate_by_name=True)

    @model_validator(mode="before")
    @classmethod
    def _compat_content(cls, data: Any) -> Any:
        if isinstance(data, dict):
            if "content" in data and "text" not in data:
                data["text"] = data["content"]
            elif "text" in data and "content" not in data:
                data["content"] = data["text"]
        return data

    @model_validator(mode="after")
    def _sync_content_text(self) -> LLMResult:
        if not self.text and self.content:
            self.text = self.content
        elif not self.content and self.text:
            self.content = self.text
        return self


class AdapterCapabilities(BaseModel):
    """Capabilities supported by a specific LLM adapter/model."""

    supports_native_schema: bool = False
    supports_system_role: bool = True
    max_context: int = 8192


class Adapter(ABC):
    """Abstract base class for all provider-specific LLM adapters."""

    name: str
    capabilities: AdapterCapabilities

    @abstractmethod
    def compile(self, bound_doc: BoundDocument, user_message: str | Message) -> CompiledRequest:
        """Compile a bound SCL document and user prompt into a CompiledRequest."""
        ...

    @abstractmethod
    async def invoke(self, request: CompiledRequest) -> LLMResult:
        """Execute a compiled request asynchronously against the model provider."""
        ...


def render_examples(doc: BoundDocument | Document) -> list[tuple[str, str]]:
    """Return an ordered list of (input, output) example pairs from an SCL document."""
    examples = doc.rendered_examples if isinstance(doc, BoundDocument) else doc.examples
    return [(ex.input, ex.output) for ex in examples]


def build_identity_system(bound_doc: BoundDocument) -> str:
    """Build identity and system instructions from <identity>, including profile and tone."""
    parts: list[str] = []
    identity = bound_doc.document.identity

    if identity.profile:
        parts.append(f"Profile: {identity.profile}")
    if identity.tone:
        parts.append(f"Tone: {identity.tone}")
    if bound_doc.rendered_identity.strip():
        parts.append(bound_doc.rendered_identity.strip())

    return "\n\n".join(parts)


def build_data_blocks(bound_doc: BoundDocument) -> str:
    """Build data blocks from bound variables for injection into the prompt."""
    blocks: list[str] = []
    for name, var in bound_doc.variables.items():
        blocks.append(f'<variable name="{name}">\n{var.rendered_value}\n</variable>')
    return "\n\n".join(blocks)


# ── Exception Hierarchy ──────────────────────────────────────────────


class ProviderError(Exception):
    """Base exception for LLM provider errors."""

    def __init__(self, message: str, provider: str | None = None) -> None:
        self.provider = provider
        super().__init__(message)


class RateLimited(ProviderError):  # noqa: N818
    """Raised when provider rate limits are exceeded (HTTP 429)."""

    def __init__(
        self,
        message: str = "Provider rate limit exceeded",
        retry_after: float | None = None,
        provider: str | None = None,
    ) -> None:
        self.retry_after = retry_after
        super().__init__(message, provider=provider)


class ModelRefusal(ProviderError):  # noqa: N818
    """Raised when a model refuses to generate an answer due to safety or policy."""


class TruncatedOutput(ProviderError):  # noqa: N818
    """Raised when model output is cut off due to context length or max token limits."""


class AuthError(ProviderError):
    """Raised when authentication with provider fails (HTTP 401/403)."""


# ── Retry Helper ─────────────────────────────────────────────────────

T = TypeVar("T")


async def with_retry(
    coro_fn: Callable[[], Coroutine[Any, Any, T]],
    *,
    max_retries: int = 3,
    initial_delay: float = 1.0,
    max_delay: float = 60.0,
    backoff_factor: float = 2.0,
    jitter: bool = True,
    retryable_errors: tuple[type[Exception], ...] = (RateLimited, ProviderError),
    non_retryable_errors: tuple[type[Exception], ...] = (AuthError, ModelRefusal),
    clock: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], Coroutine[Any, Any, None]] = asyncio.sleep,
) -> T:
    """Execute an async operation with exponential backoff and jitter, respecting retry_after.

    Args:
        coro_fn: Nullary callable returning a coroutine to execute.
        max_retries: Maximum number of retry attempts before propagating exception.
        initial_delay: Base delay in seconds for exponential backoff.
        max_delay: Maximum permissible delay between attempts.
        backoff_factor: Multiplier applied per subsequent retry.
        jitter: Whether to add random jitter to delays.
        retryable_errors: Exception types that trigger retry.
        non_retryable_errors: Exception types that abort retry immediately.
        clock: Monotonic clock provider function.
        sleep: Async sleep function.

    Returns:
        Result of successful coroutine execution.

    Raises:
        The last encountered exception when retries are exhausted or non-retryable error occurs.
    """
    attempt = 0
    while True:
        try:
            return await coro_fn()
        except non_retryable_errors:
            raise
        except retryable_errors as exc:
            attempt += 1
            if attempt > max_retries:
                raise exc

            # Calculate delay
            delay: float
            if isinstance(exc, RateLimited) and exc.retry_after is not None:
                delay = exc.retry_after
            else:
                delay = initial_delay * (backoff_factor ** (attempt - 1))

            if jitter and delay > 0:
                delay += random.uniform(0.0, 0.25 * delay)

            delay = min(delay, max_delay)
            await sleep(delay)
