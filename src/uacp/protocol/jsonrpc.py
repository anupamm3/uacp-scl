"""JSON-RPC 2.0 envelope handling for UACP messages."""

from __future__ import annotations

import json
from enum import IntEnum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from uacp.protocol.messages import Intent, UACPMessage


class JSONRPCErrorCode(IntEnum):
    """Standard JSON-RPC 2.0 and UACP reserved error codes."""

    # Standard JSON-RPC 2.0 codes (-32700 to -32600)
    PARSE_ERROR = -32700
    INVALID_REQUEST = -32600
    METHOD_NOT_FOUND = -32601
    INVALID_PARAMS = -32602
    INTERNAL_ERROR = -32603

    # UACP Reserved Protocol codes (-32001 to -32005)
    HOP_LIMIT_EXCEEDED = -32001
    CAPABILITY_NOT_FOUND = -32002
    SCHEMA_VIOLATION = -32003
    QUOTA_EXCEEDED = -32004
    SESSION_EXPIRED = -32005


DEFAULT_ERROR_MESSAGES: dict[int, str] = {
    JSONRPCErrorCode.PARSE_ERROR: "Parse error",
    JSONRPCErrorCode.INVALID_REQUEST: "Invalid Request",
    JSONRPCErrorCode.METHOD_NOT_FOUND: "Method not found",
    JSONRPCErrorCode.INVALID_PARAMS: "Invalid params",
    JSONRPCErrorCode.INTERNAL_ERROR: "Internal error",
    JSONRPCErrorCode.HOP_LIMIT_EXCEEDED: ("HopLimitExceeded: circuit breaker limit reached"),
    JSONRPCErrorCode.CAPABILITY_NOT_FOUND: (
        "CapabilityNotFound: requested agent capability missing"
    ),
    JSONRPCErrorCode.SCHEMA_VIOLATION: (
        "SchemaViolation: message payload violates contract schema"
    ),
    JSONRPCErrorCode.QUOTA_EXCEEDED: "QuotaExceeded: rate or token quota limit exhausted",
    JSONRPCErrorCode.SESSION_EXPIRED: "SessionExpired: whiteboard session has expired",
}


class JSONRPCErrorData(BaseModel):
    """Structured error object inside a JSON-RPC 2.0 response."""

    model_config = ConfigDict(frozen=True)

    code: int = Field(..., description="Numeric error code.")
    message: str = Field(..., description="Short textual summary of the error.")
    data: Any = Field(default=None, description="Optional diagnostic details or payload trace.")


class JSONRPCRequest(BaseModel):
    """JSON-RPC 2.0 request envelope wrapping a UACP message."""

    model_config = ConfigDict(extra="ignore")

    jsonrpc: Literal["2.0"] = Field(default="2.0", description="JSON-RPC protocol version.")
    id: str | int | None = Field(
        default=None,
        description="Correlation identifier for request/response pairing.",
    )
    method: str = Field(..., description="Method name in 'uacp.<intent>' format.")
    params: UACPMessage = Field(..., description="UACPMessage object payload.")

    @model_validator(mode="after")
    def _validate_method_matches_intent(self) -> JSONRPCRequest:
        expected_method = f"uacp.{self.params.intent.value}"
        if self.method != expected_method:
            msg = (
                f"Method '{self.method}' does not match message "
                f"intent '{self.params.intent.value}'."
            )
            raise ValueError(msg)
        return self


class JSONRPCResponse(BaseModel):
    """JSON-RPC 2.0 response envelope returning either result or error."""

    model_config = ConfigDict(extra="ignore")

    jsonrpc: Literal["2.0"] = Field(default="2.0", description="JSON-RPC protocol version.")
    id: str | int | None = Field(default=None, description="Request ID this response answers.")
    result: Any = Field(default=None, description="Success result payload, if successful.")
    error: JSONRPCErrorData | None = Field(
        default=None,
        description="Error descriptor if request failed.",
    )

    @model_validator(mode="after")
    def _validate_result_xor_error(self) -> JSONRPCResponse:
        if self.error is not None and self.result is not None:
            raise ValueError("A JSON-RPC response must not contain both 'result' and 'error'.")
        return self


class JSONRPCError(Exception):
    """Exception representing a JSON-RPC 2.0 protocol error."""

    def __init__(
        self,
        code: int,
        message: str | None = None,
        data: Any = None,
        id: str | int | None = None,
    ) -> None:
        resolved_msg = message or DEFAULT_ERROR_MESSAGES.get(code, "Unknown error")
        super().__init__(f"[{code}] {resolved_msg}")
        self.code = code
        self.message = resolved_msg
        self.data = data
        self.id = id

    def to_response(self) -> JSONRPCResponse:
        """Convert this error into a JSON-RPC 2.0 error response object."""
        return build_error_response(self.id, self.code, self.message, self.data)


# Alias for backward/forward compatibility
JSONRPCException = JSONRPCError


def build_response(
    id: str | int | None,
    result: Any = None,
    error: JSONRPCErrorData | None = None,
) -> JSONRPCResponse:
    """Build a standard JSON-RPC 2.0 response."""
    return JSONRPCResponse(id=id, result=result, error=error)


def build_error_response(
    id: str | int | None,
    code: int,
    message: str | None = None,
    data: Any = None,
) -> JSONRPCResponse:
    """Build a standard JSON-RPC 2.0 error response."""
    msg = message if message is not None else DEFAULT_ERROR_MESSAGES.get(code, "Unknown error")
    return JSONRPCResponse(
        id=id,
        result=None,
        error=JSONRPCErrorData(code=code, message=msg, data=data),
    )


def parse_request(
    raw: str | bytes | dict[str, Any],
    raise_errors: bool = False,
) -> JSONRPCRequest | JSONRPCResponse:
    """Parse raw JSON or a dictionary into a validated JSONRPCRequest.

    If parsing fails and `raise_errors` is False (default), returns a `JSONRPCResponse`
    containing the appropriate JSON-RPC error code and message. If `raise_errors` is
    True, raises `JSONRPCError`.

    Args:
        raw: Raw JSON string, UTF-8 bytes, or pre-parsed dictionary.
        raise_errors: Whether to raise `JSONRPCError` instead of returning `JSONRPCResponse`.

    Returns:
        A validated `JSONRPCRequest` on success, or a `JSONRPCResponse` with an error.

    Raises:
        JSONRPCError: If `raise_errors` is True and the request is malformed.
    """
    req_id: str | int | None = None

    def _fail(code: int, message: str | None = None, data: Any = None) -> JSONRPCResponse:
        err = JSONRPCError(code=code, message=message, data=data, id=req_id)
        if raise_errors:
            raise err
        return err.to_response()

    # 1. Deserialize raw input if string or bytes
    payload: Any
    if isinstance(raw, (str, bytes)):
        try:
            payload = json.loads(raw)
        except (json.JSONDecodeError, UnicodeDecodeError) as e:
            return _fail(JSONRPCErrorCode.PARSE_ERROR, data=str(e))
    elif isinstance(raw, dict):
        payload = raw
    else:
        return _fail(JSONRPCErrorCode.INVALID_REQUEST, data="Expected JSON object")

    if not isinstance(payload, dict):
        return _fail(JSONRPCErrorCode.INVALID_REQUEST, data="Expected JSON object mapping")

    # Extract ID if present (must be str, int, or None)
    raw_id = payload.get("id")
    if isinstance(raw_id, (str, int)) or raw_id is None:
        req_id = raw_id
    else:
        return _fail(
            JSONRPCErrorCode.INVALID_REQUEST,
            data="Request 'id' must be string, integer, or null",
        )

    # 2. Check JSON-RPC version
    if payload.get("jsonrpc") != "2.0":
        return _fail(
            JSONRPCErrorCode.INVALID_REQUEST,
            data="Field 'jsonrpc' must be exactly '2.0'",
        )

    # 3. Check method
    method = payload.get("method")
    if not isinstance(method, str) or not method.startswith("uacp."):
        return _fail(
            JSONRPCErrorCode.METHOD_NOT_FOUND,
            data=f"Method '{method}' is invalid; must start with 'uacp.'",
        )

    intent_candidate = method[5:]
    valid_intents = {intent.value for intent in Intent}
    if intent_candidate not in valid_intents:
        return _fail(
            JSONRPCErrorCode.METHOD_NOT_FOUND,
            data=f"Unknown UACP intent method '{method}'",
        )

    # 4. Check params & build model
    params_data = payload.get("params")
    if not isinstance(params_data, dict):
        return _fail(
            JSONRPCErrorCode.INVALID_PARAMS,
            data="Field 'params' must be a valid UACPMessage object",
        )

    try:
        message = UACPMessage.model_validate(params_data)
    except ValidationError as e:
        return _fail(JSONRPCErrorCode.INVALID_PARAMS, data=e.errors())

    # 5. Check method matches message intent
    if method != f"uacp.{message.intent.value}":
        return _fail(
            JSONRPCErrorCode.INVALID_PARAMS,
            data=(f"Method '{method}' does not match message intent '{message.intent.value}'"),
        )

    return JSONRPCRequest(id=req_id, method=method, params=message)
