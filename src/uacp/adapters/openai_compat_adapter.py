"""OpenAI-compatible adapter implementation for Groq, OpenRouter, and compatible endpoints."""

from __future__ import annotations

import asyncio
import contextlib
import json
import os
from pathlib import Path
from typing import Any, Literal

import httpx
import openai

from uacp.adapters.base import (
    Adapter,
    AdapterCapabilities,
    AuthError,
    CompiledRequest,
    LLMResult,
    Message,
    ModelRefusal,
    ProviderError,
    RateLimited,
    TruncatedOutput,
    Usage,
    build_data_blocks,
    build_identity_system,
    render_examples,
    with_retry,
)
from uacp.adapters.registry import QuotaGuard, load_env, load_models_config
from uacp.adapters.schema_tools import extract_json, normalize_schema_for
from uacp.scl.context import BoundDocument


class OpenAICompatAdapter(Adapter):
    """Adapter for OpenAI-compatible providers (Groq, OpenRouter, Ollama, etc.)."""

    def __init__(
        self,
        *,
        name: str = "openai_compat",
        model_id: str | None = None,
        base_url: str | None = None,
        api_key_env: str | None = None,
        api_key: str | None = None,
        client: openai.AsyncOpenAI | None = None,
        http_client: httpx.AsyncClient | None = None,
        quota_guard: QuotaGuard | None = None,
        timeout: float = 60.0,
        temperature: float = 0.0,
        capabilities: AdapterCapabilities | None = None,
        config_path: str | Path | None = None,
    ) -> None:
        """Initialize OpenAICompatAdapter.

        Args:
            name: Provider name identifier (e.g. 'groq', 'openrouter', 'openai_compat').
            model_id: Target model identifier (defaults to models.yaml for provider/alias).
            base_url: Base endpoint URL (e.g. 'https://api.groq.com/openai/v1').
            api_key_env: Environment variable name holding API key (e.g. 'GROQ_API_KEY').
            api_key: Optional explicit API key (defaults to os.environ[api_key_env]).
            client: Optional pre-configured AsyncOpenAI client (for testing/mocking).
            http_client: Optional custom httpx.AsyncClient (for MockTransport testing).
            quota_guard: Rate limiting QuotaGuard instance.
            timeout: Per-request timeout in seconds.
            temperature: Sampling temperature (defaults to 0.0).
            capabilities: Provider/model capabilities override.
            config_path: Custom models.yaml configuration path.
        """
        load_env()
        self.name = name
        self.timeout = timeout
        self.temperature = temperature

        models_cfg = load_models_config(config_path)
        matching_cfg = None
        for alias_cfg in models_cfg.values():
            if alias_cfg.provider == name:
                matching_cfg = alias_cfg
                break

        if model_id is not None:
            self.model_id = model_id
        elif matching_cfg is not None:
            self.model_id = matching_cfg.model_id
        else:
            self.model_id = ""

        self.base_url: str | None
        if base_url is not None:
            self.base_url = base_url
        elif matching_cfg is not None and matching_cfg.base_url:
            self.base_url = matching_cfg.base_url
        else:
            self.base_url = None

        self.api_key_env: str | None
        if api_key_env is not None:
            self.api_key_env = api_key_env
        elif matching_cfg is not None and matching_cfg.api_key_env:
            self.api_key_env = matching_cfg.api_key_env
        else:
            self.api_key_env = f"{name.upper()}_API_KEY"

        if capabilities is not None:
            self.capabilities = capabilities
        elif matching_cfg is not None:
            self.capabilities = matching_cfg.capabilities
        else:
            self.capabilities = AdapterCapabilities(
                supports_native_schema=True,
                supports_system_role=True,
                max_context=131072,
            )

        if quota_guard is not None:
            self.quota_guard = quota_guard
        elif matching_cfg is not None:
            self.quota_guard = QuotaGuard.from_limits(matching_cfg.limits)
        else:
            self.quota_guard = QuotaGuard()

        self._api_key = api_key
        self._client = client
        self._http_client = http_client

    @property
    def client(self) -> openai.AsyncOpenAI:
        """Lazily initialize and return the official AsyncOpenAI client."""
        if self._client is None:
            key = self._api_key or (
                os.environ.get(self.api_key_env) if self.api_key_env else None
            )
            # Some local endpoints (like local Ollama) might not require an API key
            if not key and self.base_url and "localhost" in self.base_url:
                key = "ollama-local"

            if not key:
                env_var = self.api_key_env or "API_KEY"
                raise AuthError(
                    f"{env_var} is not set. Set {env_var} environment variable.",
                    provider=self.name,
                )
            self._client = openai.AsyncOpenAI(
                api_key=key,
                base_url=self.base_url,
                timeout=self.timeout,
                http_client=self._http_client,  # type: ignore[arg-type]
            )
        return self._client

    def compile(self, bound_doc: BoundDocument, user_message: str | Message) -> CompiledRequest:
        """Compile a bound SCL document and user prompt into a canonical CompiledRequest.

        Rules:
        - identity -> system message.
        - examples -> alternating user and assistant turns.
        - data blocks and user text -> final user message.
        - Output modes decided from capabilities:
            - native_schema: response_format json_schema with
              normalize_schema_for("openai_strict").
            - json_mode: response_format json_object + instruction containing the schema.
            - prompt_only: instruction + schema, relying on repair.
        """
        user_content = (
            user_message.content if isinstance(user_message, Message) else str(user_message)
        )

        sys_text = build_identity_system(bound_doc)
        system = sys_text if sys_text else None

        messages: list[Message] = []
        for inp, outp in render_examples(bound_doc):
            messages.append(Message(role="user", content=inp))
            messages.append(Message(role="assistant", content=outp))

        data_blocks = build_data_blocks(bound_doc)
        if data_blocks and user_content:
            final_user = f"{data_blocks}\n\n{user_content}"
        elif data_blocks:
            final_user = data_blocks
        else:
            final_user = user_content

        if final_user:
            messages.append(Message(role="user", content=final_user))

        schema: dict[str, Any] | None = None
        output_mode: Literal["native_schema", "json_mode", "prompt_only"] = "prompt_only"

        output_node = bound_doc.document.output
        if output_node is not None:
            raw_schema = output_node.schema
            if raw_schema is None and output_node.body.strip():
                with contextlib.suppress(Exception):
                    raw_schema = json.loads(output_node.body)

            if output_node.format == "json_schema":
                if output_node.strict and raw_schema is not None:
                    if self.capabilities.supports_native_schema:
                        try:
                            schema = normalize_schema_for("openai_strict", raw_schema)
                            output_mode = "native_schema"
                        except Exception:
                            schema = raw_schema
                            output_mode = "json_mode"
                    else:
                        schema = raw_schema
                        output_mode = "json_mode"
                elif raw_schema is not None:
                    schema = raw_schema
                    output_mode = "json_mode"
                else:
                    output_mode = "prompt_only"
            else:
                output_mode = "prompt_only"

        # In json_mode or prompt_only with schema, add instruction containing the schema
        if output_mode == "json_mode" and schema is not None:
            schema_inst = (
                f"You must respond in valid JSON conforming to the following JSON schema:\n"
                f"{json.dumps(schema, indent=2)}"
            )
            system = f"{system}\n\n{schema_inst}".strip() if system else schema_inst
        elif (
            output_mode == "prompt_only"
            and output_node is not None
            and output_node.format == "json_schema"
        ):
            target_schema = schema or output_node.schema
            if target_schema is not None:
                schema_inst = (
                    f"Please output valid JSON matching this schema:\n"
                    f"{json.dumps(target_schema, indent=2)}"
                )
                system = f"{system}\n\n{schema_inst}".strip() if system else schema_inst

        params: dict[str, Any] = {"temperature": self.temperature}

        return CompiledRequest(
            provider=self.name,
            model=self.model_id,
            messages=messages,
            system=system,
            response_schema=schema,
            output_mode=output_mode,
            params=params,
        )

    async def invoke(self, request: CompiledRequest) -> LLMResult:
        """Execute a compiled request asynchronously against OpenAI-compatible API.

        Wraps API call with QuotaGuard rate limiting, exponential backoff retry respecting
        Retry-After, request timeout, and maps responses/exceptions to canonical UACP models.
        """
        await self.quota_guard.acquire()

        openai_messages: list[dict[str, str]] = []
        if request.system:
            openai_messages.append({"role": "system", "content": request.system})
        for msg in request.messages:
            role_val = msg.get("role") if isinstance(msg, dict) else msg.role
            content_val = msg.get("content") if isinstance(msg, dict) else msg.content
            openai_messages.append({"role": str(role_val), "content": str(content_val)})

        if not openai_messages:
            openai_messages = [{"role": "user", "content": ""}]

        extra_kwargs: dict[str, Any] = {}
        if request.output_mode == "native_schema" and request.response_schema is not None:
            extra_kwargs["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": "response",
                    "strict": True,
                    "schema": request.response_schema,
                },
            }
        elif request.output_mode == "json_mode":
            extra_kwargs["response_format"] = {"type": "json_object"}

        async def _call_api() -> Any:
            try:
                return await asyncio.wait_for(
                    self.client.chat.completions.create(
                        model=request.model or self.model_id,
                        messages=openai_messages,  # type: ignore[arg-type]
                        temperature=request.params.get("temperature", self.temperature),
                        **extra_kwargs,
                    ),
                    timeout=self.timeout,
                )
            except TimeoutError as err:
                raise ProviderError(
                    f"{self.name} API request timed out after {self.timeout}s",
                    provider=self.name,
                ) from err
            except openai.RateLimitError as err:
                msg = getattr(err, "message", None) or str(err)
                retry_after: float | None = None
                if hasattr(err, "response") and err.response is not None:
                    raw_header = err.response.headers.get("retry-after")
                    if raw_header:
                        with contextlib.suppress(ValueError):
                            retry_after = float(raw_header)
                raise RateLimited(
                    f"{self.name} rate limit exceeded: {msg}",
                    retry_after=retry_after,
                    provider=self.name,
                ) from err
            except (openai.AuthenticationError, openai.PermissionDeniedError) as err:
                msg = getattr(err, "message", None) or str(err)
                raise AuthError(
                    f"{self.name} authentication failed: {msg}",
                    provider=self.name,
                ) from err
            except openai.BadRequestError as err:
                msg = getattr(err, "message", None) or str(err)
                # Fallback to json_object if native_schema rejected with 400 schema error
                if (
                    "response_format" in extra_kwargs
                    and isinstance(extra_kwargs["response_format"], dict)
                    and extra_kwargs["response_format"].get("type") == "json_schema"
                    and "schema" in msg.lower()
                ):
                    extra_kwargs["response_format"] = {"type": "json_object"}
                    return await self.client.chat.completions.create(
                        model=request.model or self.model_id,
                        messages=openai_messages,  # type: ignore[arg-type]
                        temperature=request.params.get("temperature", self.temperature),
                        **extra_kwargs,
                    )
                raise ProviderError(
                    f"{self.name} bad request error (400): {msg}",
                    provider=self.name,
                ) from err
            except openai.APIStatusError as err:
                msg = getattr(err, "message", None) or str(err)
                raise ProviderError(
                    f"{self.name} API error ({err.status_code}): {msg}",
                    provider=self.name,
                ) from err
            except openai.APIError as err:
                msg = getattr(err, "message", None) or str(err)
                raise ProviderError(
                    f"{self.name} API error: {msg}",
                    provider=self.name,
                ) from err

        response = await with_retry(
            _call_api,
            retryable_errors=(RateLimited, ProviderError),
            non_retryable_errors=(AuthError, ModelRefusal, TruncatedOutput),
        )

        if not response.choices:
            raise ModelRefusal(f"No choices returned by {self.name}", provider=self.name)

        choice = response.choices[0]
        finish_reason = getattr(choice, "finish_reason", None)

        if finish_reason == "length":
            raise TruncatedOutput("Generation stopped due to max token limit", provider=self.name)

        refusal = getattr(choice.message, "refusal", None)
        if finish_reason == "content_filter" or refusal:
            refusal_msg = refusal or f"Generation refused due to safety policy: {finish_reason}"
            raise ModelRefusal(refusal_msg, provider=self.name)

        text = getattr(choice.message, "content", None) or ""
        parsed: dict[str, Any] | list[Any] | None = None

        if request.output_mode in ("native_schema", "json_mode") or text.strip().startswith(
            ("{", "[")
        ):
            with contextlib.suppress(ValueError):
                parsed = extract_json(text)

        input_tokens = 0
        output_tokens = 0
        if getattr(response, "usage", None) is not None:
            input_tokens = getattr(response.usage, "prompt_tokens", 0) or 0
            output_tokens = getattr(response.usage, "completion_tokens", 0) or 0

        usage = Usage(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            requests=1,
        )

        return LLMResult(
            text=text,
            content=text,
            parsed=parsed if isinstance(parsed, (dict, list)) else None,
            usage=usage,
            raw_meta={
                "model": getattr(response, "model", None) or request.model or self.model_id,
                "finish_reason": str(finish_reason) if finish_reason else None,
            },
        )
