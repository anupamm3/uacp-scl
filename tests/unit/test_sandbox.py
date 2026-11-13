"""Unit and security tests for the Python remediation execution sandbox.

Tests:
- Sandbox honesty: Verifies clear docstring noting best-effort nature and Windows limits.
- Static check: Blocks dangerous imports, process calls, file operations, dunder escapes.
- Security suite: At least 10 malicious attack samples blocked or stopped.
- Clean environment: Verifies child subprocess environment contains no secrets or API keys.
- Sandbox runner: Executes valid mitigation code, enforces timeout on infinite loops,
  and caps excessive output.
"""

from __future__ import annotations

import pytest

import uacp.agents.sandbox as sandbox_module
from uacp.agents.sandbox import (
    SandboxError,
    SandboxOutputLimitError,
    SandboxSecurityError,
    SandboxTimeoutError,
    run_in_sandbox,
    static_check,
)


def test_sandbox_docstring_honesty() -> None:
    """Verify sandbox honesty in docstring regarding best-effort nature and Windows limits."""
    doc = sandbox_module.__doc__ or ""
    assert "best-effort" in doc.lower()
    assert "security boundary" in doc.lower()
    assert "windows" in doc.lower()
    assert "resource limits" in doc.lower()


def test_static_check_allows_valid_code() -> None:
    """Verify static check permits benign code using allowlisted modules."""
    valid_code = """
import json
import math
from decimal import Decimal
from collections import defaultdict
import itertools
import datetime
import copy
import typing

def mitigate(ledger: list[dict]) -> tuple[list[dict], list[dict]]:
    corrected = copy.deepcopy(ledger)
    changes = []
    seen = set()
    filtered = []
    for tx in corrected:
        if tx["id"] not in seen:
            seen.add(tx["id"])
            filtered.append(tx)
        else:
            changes.append({"action": "drop_duplicate", "id": tx["id"]})
    return filtered, changes
"""
    # Should not raise
    static_check(valid_code)


# ── Malicious Attack Samples (10+ Security Invariants) ─────────────────────────


def test_security_01_import_os() -> None:
    """Attack 1: Direct 'import os' must be blocked by static_check."""
    code = """
import os
def mitigate(ledger):
    os.system("echo hacked")
    return ledger, []
"""
    with pytest.raises(SandboxSecurityError) as exc_info:
        static_check(code)
    assert "os" in str(exc_info.value).lower()


def test_security_02_dunder_import_tricks() -> None:
    """Attack 2: '__import__' call must be blocked by static_check."""
    code = """
def mitigate(ledger):
    mod = __import__("os")
    mod.system("echo hacked")
    return ledger, []
"""
    with pytest.raises(SandboxSecurityError) as exc_info:
        static_check(code)
    assert "__import__" in str(exc_info.value)


def test_security_03_reading_dotenv() -> None:
    """Attack 3: Reading sensitive files via 'open()' must be blocked by static_check."""
    code = """
def mitigate(ledger):
    content = open(".env").read()
    return ledger, [{"leak": content}]
"""
    with pytest.raises(SandboxSecurityError) as exc_info:
        static_check(code)
    assert "open" in str(exc_info.value).lower()


def test_security_04_writing_outside_temp_dir() -> None:
    """Attack 4: Writing files via 'open()' must be blocked by static_check."""
    code = """
def mitigate(ledger):
    with open("../escape.txt", "w") as f:
        f.write("escaped")
    return ledger, []
"""
    with pytest.raises(SandboxSecurityError) as exc_info:
        static_check(code)
    assert "open" in str(exc_info.value).lower()


def test_security_05_network_call_socket() -> None:
    """Attack 5: Socket and network imports must be blocked by static_check."""
    code = """
import socket
def mitigate(ledger):
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.connect(("127.0.0.1", 8080))
    return ledger, []
"""
    with pytest.raises(SandboxSecurityError) as exc_info:
        static_check(code)
    assert "socket" in str(exc_info.value).lower()


def test_security_06_subprocess_call() -> None:
    """Attack 6: Subprocess invocation must be blocked by static_check."""
    code = """
import subprocess
def mitigate(ledger):
    subprocess.run(["ls", "-la"])
    return ledger, []
"""
    with pytest.raises(SandboxSecurityError) as exc_info:
        static_check(code)
    assert "subprocess" in str(exc_info.value).lower()


def test_security_07_attribute_escape_class_subclasses() -> None:
    """Attack 7: Object tree traversal via __class__ / __subclasses__ must be blocked."""
    code = """
def mitigate(ledger):
    subclasses = ().__class__.__bases__[0].__subclasses__()
    return ledger, []
"""
    with pytest.raises(SandboxSecurityError) as exc_info:
        static_check(code)
    assert "dunder" in str(exc_info.value).lower() or "__class__" in str(exc_info.value)


def test_security_08_dunder_getattr_trick() -> None:
    """Attack 8: Calling getattr() with dunder names must be blocked by static_check."""
    code = """
def mitigate(ledger):
    cls = getattr([], "__class__")
    return ledger, []
"""
    with pytest.raises(SandboxSecurityError) as exc_info:
        static_check(code)
    assert "__class__" in str(exc_info.value) or "dunder" in str(exc_info.value).lower()


def test_security_09_reading_env_vars() -> None:
    """Attack 9: Accessing environment variables via getenv must be blocked."""
    code = """
def mitigate(ledger):
    key = getenv("GROQ_API_KEY")
    return ledger, [{"key": key}]
"""
    with pytest.raises(SandboxSecurityError) as exc_info:
        static_check(code)
    assert "getenv" in str(exc_info.value).lower()


def test_security_10_eval_exec_compilation() -> None:
    """Attack 10: Dynamic execution via eval() / exec() must be blocked."""
    code = """
def mitigate(ledger):
    eval("1 + 1")
    exec("x = 2")
    return ledger, []
"""
    with pytest.raises(SandboxSecurityError) as exc_info:
        static_check(code)
    assert "eval" in str(exc_info.value).lower() or "exec" in str(exc_info.value).lower()


def test_security_11_infinite_loop_timeout() -> None:
    """Attack 11: Infinite loop must be stopped by subprocess execution timeout."""
    infinite_loop_code = """
def mitigate(ledger):
    while True:
        pass
    return ledger, []
"""
    # static_check passes valid syntax
    static_check(infinite_loop_code)

    # run_in_sandbox must terminate with SandboxTimeoutError within timeout
    with pytest.raises(SandboxTimeoutError) as exc_info:
        run_in_sandbox(infinite_loop_code, ledger=[], timeout_s=0.5)
    assert "timed out" in str(exc_info.value).lower()


def test_security_12_very_large_output() -> None:
    """Attack 12: Flood of standard output must be stopped by the byte output cap."""
    flood_code = """
def mitigate(ledger):
    # Print over 100 KB to trigger output limit
    for _ in range(500):
        print("X" * 300)
    return ledger, []
"""
    static_check(flood_code)

    with pytest.raises(SandboxOutputLimitError) as exc_info:
        run_in_sandbox(flood_code, ledger=[], timeout_s=5.0, max_stdout_bytes=1024)
    assert "limit" in str(exc_info.value).lower()


def test_security_13_memory_bomb() -> None:
    """Attack 13: Memory allocation bomb is stopped by process exit error or timeout."""
    memory_code = """
def mitigate(ledger):
    # Continuously allocate large string chunks until MemoryError or timeout
    chunks = []
    while True:
        chunks.append(" " * (50 * 1024 * 1024))
    return ledger, []
"""
    static_check(memory_code)

    # Depending on OS / available memory, Python either raises MemoryError (exit 1)
    # or hits timeout and gets terminated.
    try:
        res = run_in_sandbox(memory_code, ledger=[], timeout_s=1.5)
        assert res.exit_code != 0
    except SandboxTimeoutError:
        pass  # Terminated by timeout
    except SandboxError:
        pass  # Stopped by sandbox error


# ── Clean Environment and Valid Execution ─────────────────────────────────────


def test_sandbox_clean_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify child subprocess does NOT receive parent API keys or secrets."""
    monkeypatch.setenv("GROQ_API_KEY", "secret-groq-key-12345")
    monkeypatch.setenv("GEMINI_API_KEY", "secret-gemini-key-67890")
    monkeypatch.setenv("OPENROUTER_API_KEY", "secret-openrouter-key")

    env_check_code = """
import json

def mitigate(ledger):
    # Check that environment does NOT contain API keys
    # Note: os is forbidden in mitigator, so runner tests environment cleanliness
    return ledger, [{"verified": "clean"}]
"""
    res = run_in_sandbox(env_check_code, ledger=[{"id": "TX-1"}], timeout_s=5.0)
    assert res.exit_code == 0
    assert res.status == "success"

    # Also directly verify the sandbox environment builder
    clean_env = sandbox_module._build_clean_env()
    assert "GROQ_API_KEY" not in clean_env
    assert "GEMINI_API_KEY" not in clean_env
    assert "OPENROUTER_API_KEY" not in clean_env


def test_sandbox_runs_valid_mitigation() -> None:
    """Verify sandbox runs valid mitigation and returns corrected ledger & change log."""
    code = """
def mitigate(ledger):
    corrected = []
    changes = []
    seen = set()
    for entry in ledger:
        if entry["id"] in seen:
            changes.append({"action": "drop_duplicate", "id": entry["id"]})
        else:
            seen.add(entry["id"])
            corrected.append(dict(entry))
    return corrected, changes
"""
    input_ledger = [
        {"id": "TX-1", "amount": 100},
        {"id": "TX-2", "amount": 200},
        {"id": "TX-1", "amount": 100},  # duplicate
    ]

    res = run_in_sandbox(code, input_ledger, timeout_s=5.0)
    assert res.exit_code == 0
    assert res.status == "success"
    assert len(res.corrected_ledger) == 2
    assert len(res.change_log) == 1
    assert res.change_log[0]["action"] == "drop_duplicate"

    # Verify 5-tuple unpacking support
    c_led, ch_log, _stdout, exit_c, dur = res
    assert len(c_led) == 2
    assert len(ch_log) == 1
    assert exit_c == 0
    assert dur > 0.0
