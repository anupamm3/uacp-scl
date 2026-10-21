"""Unit tests for JSON-RPC 2.0 envelopes and error code mapping."""

from __future__ import annotations

import json
from typing import Any

import pytest
from pydantic import ValidationError

from uacp.protocol.jsonrpc import (
    JSONRPCError,
    JSONRPCErrorCode,
    JSONRPCRequest,
    JSONRPCResponse,
    build_error_response,
    build_response,
    parse_request,
)
from uacp.protocol.messages import Intent, Party, UACPMessage


@pytest.fixture
def valid_uacp_message_dict() -> dict[str, Any]:
    """Return a valid dictionary representation of a UACPMessage."""
    return {
        "uacp_version": "1.0",
        "message_id": "01729555200000-8f921ab39c4d",
        "correlation_id": "corr-123",
        "trace_id": "trace-456",
        "sender": {"agent_id": "alpha", "engine": "gemini"},
        "recipient": {"agent_id": "beta", "engine": "groq"},
        "intent": "task_delegation",
        "payload": {
            "instruction": "Audit anomalies",
            "whiteboard_session_ref": "wb-session-1",
        },
        "circuit_breaker": {"max_hops": 5, "current_hop": 0},
        "created_at": "2026-10-22T00:50:00Z",
    }


def test_parse_valid_request(valid_uacp_message_dict: dict[str, Any]) -> None:
    """Test parsing valid JSON-RPC requests across dict, string, and bytes representations."""
    raw_req = {
        "jsonrpc": "2.0",
        "id": "req-1",
        "method": "uacp.task_delegation",
        "params": valid_uacp_message_dict,
    }

    # Dict
    parsed_dict = parse_request(raw_req)
    assert isinstance(parsed_dict, JSONRPCRequest)
    assert parsed_dict.id == "req-1"
    assert parsed_dict.method == "uacp.task_delegation"
    assert parsed_dict.params.intent == Intent.task_delegation

    # JSON String
    raw_str = json.dumps(raw_req)
    parsed_str = parse_request(raw_str)
    assert isinstance(parsed_str, JSONRPCRequest)
    assert parsed_str.id == "req-1"

    # UTF-8 Bytes
    raw_bytes = raw_str.encode("utf-8")
    parsed_bytes = parse_request(raw_bytes)
    assert isinstance(parsed_bytes, JSONRPCRequest)
    assert parsed_bytes.id == "req-1"


def test_error_parse_json_malformed() -> None:
    """Malformed JSON string must return -32700 Parse error."""
    res = parse_request("{malformed: json")
    assert isinstance(res, JSONRPCResponse)
    assert res.error is not None
    assert res.error.code == JSONRPCErrorCode.PARSE_ERROR
    assert res.id is None


def test_error_invalid_request_structure() -> None:
    """Non-dict JSON or invalid jsonrpc version must return -32600 Invalid Request."""
    # List instead of dict
    res_list = parse_request("[1, 2, 3]")
    assert isinstance(res_list, JSONRPCResponse)
    assert res_list.error is not None
    assert res_list.error.code == JSONRPCErrorCode.INVALID_REQUEST

    # Missing jsonrpc: "2.0"
    res_no_ver = parse_request({"id": 1, "method": "uacp.heartbeat", "params": {}})
    assert isinstance(res_no_ver, JSONRPCResponse)
    assert res_no_ver.error is not None
    assert res_no_ver.error.code == JSONRPCErrorCode.INVALID_REQUEST
    assert res_no_ver.id == 1

    # Invalid ID type (e.g. float or dict)
    res_bad_id = parse_request(
        {"jsonrpc": "2.0", "id": {"nested": "id"}, "method": "uacp.heartbeat", "params": {}}
    )
    assert isinstance(res_bad_id, JSONRPCResponse)
    assert res_bad_id.error is not None
    assert res_bad_id.error.code == JSONRPCErrorCode.INVALID_REQUEST


def test_error_method_not_found() -> None:
    """Unknown method or non-uacp prefix must return -32601 Method not found."""
    # Prefix not uacp.
    res_prefix = parse_request({"jsonrpc": "2.0", "id": 10, "method": "custom.task", "params": {}})
    assert isinstance(res_prefix, JSONRPCResponse)
    assert res_prefix.error is not None
    assert res_prefix.error.code == JSONRPCErrorCode.METHOD_NOT_FOUND
    assert res_prefix.id == 10

    # Unknown intent
    res_unknown = parse_request(
        {"jsonrpc": "2.0", "id": 11, "method": "uacp.unknown_intent", "params": {}}
    )
    assert isinstance(res_unknown, JSONRPCResponse)
    assert res_unknown.error is not None
    assert res_unknown.error.code == JSONRPCErrorCode.METHOD_NOT_FOUND
    assert res_unknown.id == 11


def test_error_invalid_params(valid_uacp_message_dict: dict[str, Any]) -> None:
    """Malformed UACPMessage or method/intent mismatch must return -32602 Invalid params."""
    # Missing required message fields
    res_bad_params = parse_request(
        {
            "jsonrpc": "2.0",
            "id": 20,
            "method": "uacp.task_delegation",
            "params": {"uacp_version": "1.0"},  # Missing sender, recipient, etc.
        }
    )
    assert isinstance(res_bad_params, JSONRPCResponse)
    assert res_bad_params.error is not None
    assert res_bad_params.error.code == JSONRPCErrorCode.INVALID_PARAMS
    assert res_bad_params.id == 20

    # Method and intent mismatch
    res_mismatch = parse_request(
        {
            "jsonrpc": "2.0",
            "id": 21,
            "method": "uacp.heartbeat",  # Method says heartbeat
            "params": valid_uacp_message_dict,  # params intent is task_delegation
        }
    )
    assert isinstance(res_mismatch, JSONRPCResponse)
    assert res_mismatch.error is not None
    assert res_mismatch.error.code == JSONRPCErrorCode.INVALID_PARAMS
    assert res_mismatch.id == 21


def test_raise_errors_option() -> None:
    """When raise_errors=True, JSONRPCError is raised with matching code."""
    with pytest.raises(JSONRPCError) as exc_info:
        parse_request("{invalid_json", raise_errors=True)
    assert exc_info.value.code == JSONRPCErrorCode.PARSE_ERROR
    err_resp = exc_info.value.to_response()
    assert err_resp.error is not None
    assert err_resp.error.code == -32700


def test_reserved_uacp_error_codes() -> None:
    """Verify definitions and building responses with reserved UACP error codes."""
    assert JSONRPCErrorCode.HOP_LIMIT_EXCEEDED.value == -32001
    assert JSONRPCErrorCode.CAPABILITY_NOT_FOUND.value == -32002
    assert JSONRPCErrorCode.SCHEMA_VIOLATION.value == -32003
    assert JSONRPCErrorCode.QUOTA_EXCEEDED.value == -32004
    assert JSONRPCErrorCode.SESSION_EXPIRED.value == -32005

    err_resp = build_error_response("req-99", JSONRPCErrorCode.HOP_LIMIT_EXCEEDED.value)
    assert err_resp.id == "req-99"
    assert err_resp.error is not None
    assert err_resp.error.code == -32001
    assert "HopLimitExceeded" in err_resp.error.message


def test_build_response_helpers() -> None:
    """Verify build_response and mutual exclusivity of result and error."""
    resp = build_response(id="req-1", result={"status": "ok"})
    assert resp.id == "req-1"
    assert resp.result == {"status": "ok"}
    assert resp.error is None

    # Mutual exclusivity
    with pytest.raises(ValidationError):
        JSONRPCResponse.model_validate(
            {
                "jsonrpc": "2.0",
                "id": "req-1",
                "result": {"status": "ok"},
                "error": {"code": -32600, "message": "error"},
            }
        )


def test_request_model_direct_validation() -> None:
    """Direct instantiation of JSONRPCRequest validates method matches params intent."""
    msg = UACPMessage(
        correlation_id="c",
        trace_id="t",
        sender=Party(agent_id="a", engine="e"),
        recipient=Party(agent_id="b", engine="e"),
        intent=Intent.heartbeat,
        payload={},
    )
    # Valid
    req = JSONRPCRequest(id=1, method="uacp.heartbeat", params=msg)
    assert req.method == "uacp.heartbeat"

    # Mismatched method
    with pytest.raises(ValidationError):
        JSONRPCRequest(id=1, method="uacp.task_delegation", params=msg)
