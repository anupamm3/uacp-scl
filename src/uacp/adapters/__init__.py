"""LLM provider adapters and contract enforcement tooling."""

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
from uacp.adapters.enforcer import (
    LLMAdapter,
    OutputViolation,
    enforce,
    invoke_with_repair,
)
from uacp.adapters.fake_adapter import FakeAdapter
from uacp.adapters.openai_compat_adapter import OpenAICompatAdapter
from uacp.adapters.registry import (
    LimitConfig,
    ModelAliasConfig,
    QuotaGuard,
    get_adapter,
    load_env,
    load_models_config,
    register_adapter,
)
from uacp.adapters.schema_tools import (
    extract_json,
    extract_xml_block,
    normalize_schema_for,
)

__all__ = [
    "Adapter",
    "AdapterCapabilities",
    "AuthError",
    "CompiledRequest",
    "FakeAdapter",
    "LLMAdapter",
    "LLMResult",
    "LimitConfig",
    "Message",
    "ModelAliasConfig",
    "ModelRefusal",
    "OpenAICompatAdapter",
    "OutputViolation",
    "ProviderError",
    "QuotaGuard",
    "RateLimited",
    "TruncatedOutput",
    "Usage",
    "build_data_blocks",
    "build_identity_system",
    "enforce",
    "extract_json",
    "extract_xml_block",
    "get_adapter",
    "invoke_with_repair",
    "load_env",
    "load_models_config",
    "normalize_schema_for",
    "register_adapter",
    "render_examples",
    "with_retry",
]
