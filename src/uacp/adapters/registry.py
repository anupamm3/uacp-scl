"""Model configuration loader, QuotaGuard rate limiter, and adapter registry."""

from __future__ import annotations

import asyncio
import os
import time
from collections import deque
from collections.abc import Callable, Coroutine
from pathlib import Path
from typing import Any

import dotenv
import yaml  # type: ignore[import-untyped]
from pydantic import BaseModel, ConfigDict, Field

from uacp.adapters.base import Adapter, AdapterCapabilities, RateLimited
from uacp.adapters.fake_adapter import FakeAdapter

_DOTENV_LOADED = False


def load_env(env_path: str | Path | None = None) -> None:
    """Load environment variables from .env file in one shared place.

    Searches cwd and parent directories up to 4 levels unless an explicit path is provided.
    Guarded to avoid redundant disk I/O when called repeatedly.
    """
    global _DOTENV_LOADED
    if env_path is not None:
        dotenv.load_dotenv(Path(env_path))
        return

    if not _DOTENV_LOADED:
        cwd_env = Path.cwd() / ".env"
        if cwd_env.is_file():
            dotenv.load_dotenv(cwd_env)
        else:
            current = Path(__file__).resolve().parent
            for _ in range(4):
                candidate = current / ".env"
                if candidate.is_file():
                    dotenv.load_dotenv(candidate)
                    break
                current = current.parent
        _DOTENV_LOADED = True


class LimitConfig(BaseModel):
    """Rate limit configuration for an LLM alias."""

    requests_per_minute: int | None = None
    tokens_per_minute: int | None = None
    daily_requests: int | None = None


class ModelAliasConfig(BaseModel):
    """Configuration mapping an alias to a provider, model ID, capabilities, and limits."""

    provider: str
    model_id: str
    base_url: str | None = None
    api_key_env: str | None = None
    capabilities: AdapterCapabilities = Field(default_factory=AdapterCapabilities)
    limits: LimitConfig = Field(default_factory=LimitConfig)

    model_config = ConfigDict(protected_namespaces=())


def _find_default_config_path() -> Path:
    """Find config/models.yaml relative to current working directory or package root."""
    cwd_path = Path.cwd() / "config" / "models.yaml"
    if cwd_path.is_file():
        return cwd_path

    # Check parent directories up to 3 levels
    current = Path(__file__).resolve().parent
    for _ in range(4):
        candidate = current / "config" / "models.yaml"
        if candidate.is_file():
            return candidate
        current = current.parent

    return cwd_path


def load_models_config(path: str | Path | None = None) -> dict[str, ModelAliasConfig]:
    """Load model configuration from YAML with environment variable overrides.

    Args:
        path: Optional path to models.yaml. Defaults to finding config/models.yaml.

    Returns:
        Mapping from alias name to ModelAliasConfig.
    """
    load_env()
    config_file = Path(path) if path is not None else _find_default_config_path()
    if not config_file.is_file():
        raise FileNotFoundError(f"Configuration file not found: {config_file}")

    with open(config_file, encoding="utf-8") as f:
        raw_data = yaml.safe_load(f) or {}

    providers_raw = raw_data.get("providers", {})
    aliases_raw = raw_data.get("aliases", {})
    configs: dict[str, ModelAliasConfig] = {}

    for alias, entry in aliases_raw.items():
        provider = entry.get("provider", "")
        model_id = entry.get("model_id", "")
        limits_data = entry.get("limits") or {}

        # Provider defaults
        prov_data = providers_raw.get(provider, {})
        base_url = entry.get("base_url", prov_data.get("base_url"))
        api_key_env = entry.get("api_key_env", prov_data.get("api_key_env"))
        caps_data = entry.get("capabilities", prov_data.get("capabilities") or {})
        capabilities = AdapterCapabilities(**caps_data)

        # Environment variable overrides
        prefix = f"UACP_MODEL_{alias.upper()}_"
        provider = os.environ.get(f"{prefix}PROVIDER", provider)
        model_id = os.environ.get(f"{prefix}MODEL_ID", model_id)
        base_url = os.environ.get(f"{prefix}BASE_URL", base_url)
        api_key_env = os.environ.get(f"{prefix}API_KEY_ENV", api_key_env)

        rpm_env = os.environ.get(f"{prefix}RPM")
        requests_per_minute = (
            int(rpm_env) if rpm_env is not None else limits_data.get("requests_per_minute")
        )

        tpm_env = os.environ.get(f"{prefix}TPM")
        tokens_per_minute = (
            int(tpm_env) if tpm_env is not None else limits_data.get("tokens_per_minute")
        )

        daily_env = os.environ.get(f"{prefix}DAILY")
        daily_requests = (
            int(daily_env) if daily_env is not None else limits_data.get("daily_requests")
        )

        configs[alias] = ModelAliasConfig(
            provider=provider,
            model_id=model_id,
            base_url=base_url,
            api_key_env=api_key_env,
            capabilities=capabilities,
            limits=LimitConfig(
                requests_per_minute=requests_per_minute,
                tokens_per_minute=tokens_per_minute,
                daily_requests=daily_requests,
            ),
        )

    return configs


class QuotaGuard:
    """Sliding-window in-process rate limiter for free-tier quotas."""

    def __init__(
        self,
        requests_per_minute: int | None = None,
        daily_requests: int | None = None,
        *,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], Coroutine[Any, Any, None]] = asyncio.sleep,
        raise_on_limit: bool = False,
    ) -> None:
        self.requests_per_minute = requests_per_minute
        self.daily_requests = daily_requests
        self.clock = clock
        self.sleep = sleep
        self.raise_on_limit = raise_on_limit

        self._minute_window: deque[float] = deque()
        self._daily_window: deque[float] = deque()

    @classmethod
    def from_limits(
        cls,
        limits: LimitConfig,
        *,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], Coroutine[Any, Any, None]] = asyncio.sleep,
        raise_on_limit: bool = False,
    ) -> QuotaGuard:
        """Create a QuotaGuard directly from a LimitConfig."""
        return cls(
            requests_per_minute=limits.requests_per_minute,
            daily_requests=limits.daily_requests,
            clock=clock,
            sleep=sleep,
            raise_on_limit=raise_on_limit,
        )

    async def acquire(self) -> None:
        """Acquire quota slot, either sleeping or raising RateLimited if limits exceeded."""
        while True:
            now = self.clock()

            # Evict timestamps outside 60-second window
            while self._minute_window and (now - self._minute_window[0] >= 60.0):
                self._minute_window.popleft()

            # Evict timestamps outside 24-hour (86400s) window
            while self._daily_window and (now - self._daily_window[0] >= 86400.0):
                self._daily_window.popleft()

            # Check minute limit
            if (
                self.requests_per_minute is not None
                and len(self._minute_window) >= self.requests_per_minute
            ):
                wait_time = 60.0 - (now - self._minute_window[0])
                if wait_time > 0:
                    if self.raise_on_limit:
                        raise RateLimited(
                            f"Minute quota exceeded ({self.requests_per_minute} req/min)",
                            retry_after=wait_time,
                        )
                    await self.sleep(wait_time)
                    continue

            # Check daily limit
            if self.daily_requests is not None and len(self._daily_window) >= self.daily_requests:
                wait_time = 86400.0 - (now - self._daily_window[0])
                if wait_time > 0:
                    if self.raise_on_limit:
                        raise RateLimited(
                            f"Daily quota exceeded ({self.daily_requests} req/day)",
                            retry_after=wait_time,
                        )
                    await self.sleep(wait_time)
                    continue

            # Within limits: register invocation timestamp
            self._minute_window.append(now)
            self._daily_window.append(now)
            return


_ADAPTER_REGISTRY: dict[str, type[Adapter]] = {"fake": FakeAdapter}


def register_adapter(provider: str, adapter_cls: type[Adapter]) -> None:
    """Register an adapter class for a given provider name."""
    _ADAPTER_REGISTRY[provider] = adapter_cls


def _resolve_adapter_cls(provider: str) -> type[Adapter] | None:
    if provider in _ADAPTER_REGISTRY:
        return _ADAPTER_REGISTRY[provider]
    if provider == "gemini":
        from uacp.adapters.gemini import GeminiAdapter

        return GeminiAdapter
    if provider in ("groq", "openrouter"):
        from uacp.adapters.openai_compat_adapter import OpenAICompatAdapter

        return OpenAICompatAdapter
    return None


def get_adapter(alias: str, config_path: str | Path | None = None) -> Adapter:
    """Instantiate and return an Adapter configured for the requested alias.

    Args:
        alias: Alias key defined in models.yaml (e.g. 'fake', 'alpha_default').
        config_path: Optional explicit configuration file path.

    Returns:
        Instantiated Adapter.

    Raises:
        KeyError: If alias is not defined.
        NotImplementedError: If provider implementation is not yet registered.
    """
    configs = load_models_config(config_path)
    if alias not in configs:
        raise KeyError(f"Model alias '{alias}' not found in configuration")

    model_config = configs[alias]
    provider = model_config.provider
    adapter_cls = _resolve_adapter_cls(provider)

    if adapter_cls is None:
        raise NotImplementedError(
            f"Provider '{provider}' for alias '{alias}' is not implemented yet"
        )

    if adapter_cls is FakeAdapter:
        return FakeAdapter(model_id=model_config.model_id)

    from uacp.adapters.gemini import GeminiAdapter

    if adapter_cls is GeminiAdapter:
        return GeminiAdapter(
            model_id=model_config.model_id,
            quota_guard=QuotaGuard.from_limits(model_config.limits),
            config_path=config_path,
        )

    from uacp.adapters.openai_compat_adapter import OpenAICompatAdapter

    if issubclass(adapter_cls, OpenAICompatAdapter) or adapter_cls is OpenAICompatAdapter:
        return OpenAICompatAdapter(
            name=model_config.provider,
            model_id=model_config.model_id,
            base_url=model_config.base_url,
            api_key_env=model_config.api_key_env,
            capabilities=model_config.capabilities,
            quota_guard=QuotaGuard.from_limits(model_config.limits),
            config_path=config_path,
        )

    return adapter_cls()
