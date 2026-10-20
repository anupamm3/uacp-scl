"""Google Gemini adapter implementation using the official google-genai SDK."""

from __future__ import annotations

import asyncio
import contextlib
import json
import os
from pathlib import Path
from typing import Any, Literal

from google import genai
from google.genai import errors, types

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
from uacp.adapters.registry import QuotaGuard, load_models_config
from uacp.adapters.schema_tools import extract_json, normalize_schema_for
from uacp.scl.context import BoundDocument


class GeminiAdapter(Adapter):
    """Adapter for Google Gemini models using the official google-genai SDK."""

    name: str = "gemini"

    def __init__(
        self,
        *,
        model_id: str | None = None,
        api_key: str | None = None,
        client: genai.Client | None = None,
        quota_guard: QuotaGuard | None = None,
        timeout: float = 60.0,
        temperature: float = 0.0,
        capabilities: AdapterCapabilities | None = None,
        config_path: str | Path | None = None,
    ) -> None:
        """Initialize GeminiAdapter.

        Args:
            model_id: Model identifier (defaults to config/models.yaml alias alpha_default).
            api_key: Optional Gemini API key (defaults to GEMINI_API_KEY environment variable).
            client: Optional pre-configured genai.Client (for testing/mocking).
            quota_guard: Rate limiting QuotaGuard instance.
            timeout: Per-request timeout in seconds.
            temperature: Sampling temperature (defaults to 0.0).
            capabilities: Provider capabilities override.
            config_path: Custom models.yaml configuration path.
        """
        self.timeout = timeout
        self.temperature = temperature
        self.capabilities = capabilities or AdapterCapabilities(
            supports_native_schema=True,
            supports_system_role=True,
            max_context=1_048_576,
        )

        models_cfg = load_models_config(config_path)
        gemini_cfg = None
        for alias_cfg in models_cfg.values():
            if alias_cfg.provider == "gemini":
                gemini_cfg = alias_cfg
                break

        if model_id is not None:
            self.model_id = model_id
        elif gemini_cfg is not None:
            self.model_id = gemini_cfg.model_id
        else:
            raise KeyError("No Gemini configuration found in models.yaml")

        if quota_guard is not None:
            self.quota_guard = quota_guard
        elif gemini_cfg is not None:
            self.quota_guard = QuotaGuard.from_limits(gemini_cfg.limits)
        else:
            self.quota_guard = QuotaGuard()

        self._api_key = api_key or os.environ.get("GEMINI_API_KEY")
        self._client = client

    @property
    def client(self) -> genai.Client:
        """Lazily initialize and return the official google-genai Client."""
        if self._client is None:
            key = self._api_key or os.environ.get("GEMINI_API_KEY")
            if not key:
                raise AuthError(
                    "GEMINI_API_KEY is not set. Set GEMINI_API_KEY environment variable.",
                    provider=self.name,
                )
            self._client = genai.Client(api_key=key)
        return self._client

    def compile(self, bound_doc: BoundDocument, user_message: str | Message) -> CompiledRequest:
        """Compile a bound SCL document and user prompt into a canonical CompiledRequest.

        Rules:
        - identity -> system instruction.
        - examples -> alternating user and model turns (role='assistant' in Message).
        - data blocks and user message -> final user turn.
        - json_schema output with strict=True -> native_schema with normalized schema.
        - Fallback to json_mode or prompt_only if native schema is not applicable.
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
                    try:
                        normalized = normalize_schema_for("gemini", raw_schema)
                        schema = normalized
                        output_mode = "native_schema"
                    except Exception:
                        schema = raw_schema
                        output_mode = "json_mode"
                elif raw_schema is not None:
                    schema = raw_schema
                    output_mode = "json_mode"
                else:
                    output_mode = "prompt_only"
            else:
                output_mode = "prompt_only"

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
        """Execute a compiled request asynchronously against the Google Gemini API.

        Wraps the SDK call with QuotaGuard rate limiting, exponential backoff retry,
        and request timeout. Maps provider errors and finish reasons to canonical exceptions.
        """
        await self.quota_guard.acquire()

        contents: list[types.Content] = []
        for msg in request.messages:
            role = "model" if msg.role == "assistant" else "user"
            contents.append(
                types.Content(
                    role=role,
                    parts=[types.Part.from_text(text=msg.content)],
                )
            )

        if not contents:
            contents = [types.Content(role="user", parts=[types.Part.from_text(text="")])]

        config_args: dict[str, Any] = {}
        if request.system:
            config_args["system_instruction"] = request.system

        temp = request.params.get("temperature", self.temperature)
        config_args["temperature"] = temp

        if request.output_mode == "native_schema" and request.response_schema is not None:
            config_args["response_mime_type"] = "application/json"
            config_args["response_schema"] = request.response_schema
        elif request.output_mode == "json_mode":
            config_args["response_mime_type"] = "application/json"

        async def _call_api(cfg: types.GenerateContentConfig) -> types.GenerateContentResponse:
            try:
                return await asyncio.wait_for(
                    self.client.aio.models.generate_content(
                        model=request.model or self.model_id,
                        contents=contents,
                        config=cfg,
                    ),
                    timeout=self.timeout,
                )
            except TimeoutError as err:
                raise ProviderError(
                    f"Gemini API request timed out after {self.timeout}s",
                    provider=self.name,
                ) from err
            except (errors.ClientError, errors.APIError) as err:
                code = getattr(err, "code", None)
                msg = getattr(err, "message", None) or str(err)

                if code == 429 or "RESOURCE_EXHAUSTED" in msg:
                    raise RateLimited(
                        f"Gemini rate limit exceeded: {msg}", provider=self.name
                    ) from err
                if code in (401, 403) or "UNAUTHENTICATED" in msg or "PERMISSION_DENIED" in msg:
                    raise AuthError(
                        f"Gemini authentication failed: {msg}",
                        provider=self.name,
                    ) from err

                # If native schema was rejected with 400 Bad Request, fall back
                if (
                    code == 400
                    and "response_schema" in config_args
                    and ("schema" in msg.lower() or "json_schema" in msg.lower())
                ):
                    config_args.pop("response_schema", None)
                    config_args["response_mime_type"] = "application/json"
                    fallback_cfg = types.GenerateContentConfig(**config_args)
                    return await self.client.aio.models.generate_content(
                        model=request.model or self.model_id,
                        contents=contents,
                        config=fallback_cfg,
                    )

                raise ProviderError(
                    f"Gemini API error ({code}): {msg}",
                    provider=self.name,
                ) from err

        current_config = types.GenerateContentConfig(**config_args)
        response = await with_retry(
            lambda: _call_api(current_config),
            retryable_errors=(RateLimited, ProviderError),
            non_retryable_errors=(AuthError, ModelRefusal, TruncatedOutput),
        )

        if response.prompt_feedback and response.prompt_feedback.block_reason:
            raise ModelRefusal(
                f"Prompt blocked by safety filter: {response.prompt_feedback.block_reason}",
                provider=self.name,
            )

        if not response.candidates:
            raise ModelRefusal("No candidate responses returned by Gemini", provider=self.name)

        candidate = response.candidates[0]
        finish_reason = candidate.finish_reason

        if finish_reason == types.FinishReason.MAX_TOKENS or str(finish_reason) == "MAX_TOKENS":
            raise TruncatedOutput("Generation stopped due to max token limit", provider=self.name)

        refusal_reasons = {
            types.FinishReason.SAFETY,
            types.FinishReason.RECITATION,
            types.FinishReason.BLOCKLIST,
            types.FinishReason.PROHIBITED_CONTENT,
            types.FinishReason.SPII,
            types.FinishReason.IMAGE_SAFETY,
        }
        if finish_reason in refusal_reasons or any(
            r in str(finish_reason) for r in ("SAFETY", "RECITATION", "BLOCKLIST", "PROHIBITED")
        ):
            refusal_msg = (
                candidate.finish_message
                or f"Generation refused due to safety policy: {finish_reason}"
            )
            raise ModelRefusal(refusal_msg, provider=self.name)

        text = response.text or ""
        parsed: dict[str, Any] | list[Any] | None = None

        is_json_target = request.output_mode in ("native_schema", "json_mode")
        if is_json_target or text.strip().startswith(("{", "[")):
            with contextlib.suppress(ValueError):
                parsed = extract_json(text)

        input_tokens = 0
        output_tokens = 0
        if response.usage_metadata is not None:
            input_tokens = response.usage_metadata.prompt_token_count or 0
            output_tokens = response.usage_metadata.candidates_token_count or 0

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
                "model": request.model or self.model_id,
                "finish_reason": str(finish_reason) if finish_reason else None,
            },
        )
