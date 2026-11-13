"""Sandboxed execution environment for generated remediation code.

SECURITY NOTICE & BEST-EFFORT SANDBOX DISCLAIMER:
--------------------------------------------------
This module provides a BEST-EFFORT application sandbox combining static AST inspection
and a subprocess executed in Python isolated mode ('python -I') with execution timeouts
and sanitized environment variables.

THIS IS NOT A FORMAL OS-LEVEL OR HARDWARE-LEVEL SECURITY BOUNDARY (such as gVisor,
Firecracker microVMs, or Docker/seccomp containers). It is designed to prevent accidental
breakouts and common adversarial injection vectors during automated remediation workflows.

PLATFORM-SPECIFIC LIMITS:
- On Linux and macOS, resource limits (CPU time and memory address space via 'resource.setrlimit')
  are applied to the child subprocess when supported.
- On Windows, POSIX resource limits ('resource' module) are unavailable; execution safety
  relies strictly on the AST static check, process timeouts, output byte caps, and isolated
  subprocess execution.
"""

from __future__ import annotations

import ast
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from typing import Any

# Allowlisted standard modules permitted for remediation logic
ALLOWED_MODULES: frozenset[str] = frozenset(
    {
        "json",
        "math",
        "decimal",
        "collections",
        "itertools",
        "datetime",
        "copy",
        "typing",
    }
)

# Forbidden built-in call / identifier names
FORBIDDEN_CALLS: frozenset[str] = frozenset(
    {
        "eval",
        "exec",
        "compile",
        "open",
        "__import__",
        "globals",
        "locals",
        "breakpoint",
        "input",
        "help",
        "getenv",
    }
)

# Forbidden attribute names that allow sandbox escapes, OS access, or introspection
FORBIDDEN_ATTRIBUTES: frozenset[str] = frozenset(
    {
        "__class__",
        "__subclasses__",
        "__bases__",
        "__base__",
        "__mro__",
        "__globals__",
        "__code__",
        "__builtins__",
        "__reduce__",
        "__reduce_ex__",
        "__loader__",
        "__spec__",
        "__package__",
        "__init_subclass__",
        "__subclasshook__",
        "environ",
        "system",
        "popen",
        "spawn",
        "execv",
        "execve",
    }
)

# Denied module root names
FORBIDDEN_MODULE_ROOTS: frozenset[str] = frozenset(
    {
        "os",
        "sys",
        "subprocess",
        "socket",
        "urllib",
        "http",
        "requests",
        "ftplib",
        "multiprocessing",
        "ctypes",
        "threading",
        "shutil",
        "pathlib",
        "asyncio",
        "posix",
        "nt",
    }
)

MAX_STDOUT_BYTES: int = 64 * 1024  # 64 KB maximum standard output cap


# ── Exceptions ────────────────────────────────────────────────────────────────


class SandboxError(Exception):
    """Base exception for all sandbox operations."""


class SandboxSecurityError(SandboxError, ValueError):
    """Raised when code violates static AST security policies."""


class SandboxTimeoutError(SandboxError, TimeoutError):
    """Raised when sandbox execution exceeds the allotted timeout."""


class SandboxOutputLimitError(SandboxError, RuntimeError):
    """Raised when sandbox process stdout exceeds the maximum allowed byte cap."""


class SandboxExecutionError(SandboxError, RuntimeError):
    """Raised when sandbox runner fails or mitigator crashes inside the subprocess."""


# ── Execution Result ──────────────────────────────────────────────────────────


class SandboxResult:
    """Result of sandboxed execution, supporting 5-tuple unpacking and named attributes."""

    def __init__(
        self,
        corrected_ledger: list[dict[str, Any]],
        change_log: list[dict[str, Any]],
        stdout: str,
        exit_code: int,
        duration: float,
        status: str = "success",
        error: str | None = None,
    ) -> None:
        self.corrected_ledger = corrected_ledger
        self.change_log = change_log
        self.stdout = stdout
        self.exit_code = exit_code
        self.duration = duration
        self.status = status
        self.error = error

    def __iter__(self) -> Any:
        yield self.corrected_ledger
        yield self.change_log
        yield self.stdout
        yield self.exit_code
        yield self.duration

    def __getitem__(self, index: int) -> Any:
        return [
            self.corrected_ledger,
            self.change_log,
            self.stdout,
            self.exit_code,
            self.duration,
        ][index]

    def __repr__(self) -> str:
        return (
            f"SandboxResult(corrected_ledger={len(self.corrected_ledger)} items, "
            f"change_log={len(self.change_log)} items, exit_code={self.exit_code}, "
            f"duration={self.duration:.3f}s, status='{self.status}')"
        )


# ── Static AST Inspection ─────────────────────────────────────────────────────


class _SecurityASTVisitor(ast.NodeVisitor):
    """AST visitor enforcing allowlisted imports and blocking escape vectors."""

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            root_module = alias.name.split(".")[0]
            if root_module in FORBIDDEN_MODULE_ROOTS:
                raise SandboxSecurityError(
                    f"Forbidden module import '{alias.name}'. OS/network modules prohibited."
                )
            if root_module not in ALLOWED_MODULES:
                raise SandboxSecurityError(f"Import of module '{alias.name}' is not in allowlist.")
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        if node.level is not None and node.level > 0:
            raise SandboxSecurityError("Relative imports are prohibited in sandbox.")
        if not node.module:
            raise SandboxSecurityError("Empty module in import statement is prohibited.")

        root_module = node.module.split(".")[0]
        if root_module in FORBIDDEN_MODULE_ROOTS:
            raise SandboxSecurityError(
                f"Forbidden module import from '{node.module}'. OS/network modules prohibited."
            )
        if root_module not in ALLOWED_MODULES:
            raise SandboxSecurityError(f"Import from module '{node.module}' is not in allowlist.")
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        # Check direct calls like eval(...), open(...)
        if isinstance(node.func, ast.Name):
            func_name = node.func.id
            if func_name in FORBIDDEN_CALLS:
                raise SandboxSecurityError(
                    f"Calling built-in '{func_name}' is prohibited inside the sandbox."
                )
            if func_name in ("getattr", "setattr", "hasattr", "delattr") and len(node.args) >= 2:
                second_arg = node.args[1]
                if isinstance(second_arg, ast.Constant) and isinstance(second_arg.value, str):
                    val = second_arg.value
                    if val.startswith("__") and val.endswith("__"):
                        raise SandboxSecurityError(
                            f"Calling '{func_name}' with dunder attribute '{val}' is prohibited."
                        )
                    if val in FORBIDDEN_ATTRIBUTES:
                        raise SandboxSecurityError(
                            f"Calling '{func_name}' with forbidden attribute '{val}' is forbidden."
                        )

        # Check method calls like os.system(...)
        elif isinstance(node.func, ast.Attribute):
            attr_name = node.func.attr
            if attr_name.startswith("__") and attr_name.endswith("__"):
                raise SandboxSecurityError(
                    f"Calling prohibited dunder attribute or method '{attr_name}' is not allowed."
                )
            if attr_name in FORBIDDEN_CALLS or attr_name in FORBIDDEN_ATTRIBUTES:
                raise SandboxSecurityError(
                    f"Calling prohibited attribute or method '{attr_name}' is not allowed."
                )

        self.generic_visit(node)

    def visit_Attribute(self, node: ast.Attribute) -> None:
        attr_name = node.attr

        # Deny all dunder attributes (e.g. __class__, __subclasses__, __globals__, etc.)
        if attr_name.startswith("__") and attr_name.endswith("__"):
            raise SandboxSecurityError(
                f"Attribute access to dunder name '{attr_name}' is prohibited."
            )

        if attr_name in FORBIDDEN_ATTRIBUTES:
            raise SandboxSecurityError(
                f"Attribute access to '{attr_name}' is prohibited in the sandbox."
            )

        # Deny accessing attributes on names like os, sys, subprocess, socket
        if isinstance(node.value, ast.Name) and node.value.id in FORBIDDEN_MODULE_ROOTS:
            raise SandboxSecurityError(
                f"Attribute access on forbidden module '{node.value.id}.{attr_name}' prohibited."
            )

        self.generic_visit(node)

    def visit_Name(self, node: ast.Name) -> None:
        if node.id in FORBIDDEN_CALLS and isinstance(node.ctx, (ast.Load, ast.Store)):
            raise SandboxSecurityError(f"Use of restricted identifier '{node.id}' is prohibited.")
        if node.id in FORBIDDEN_MODULE_ROOTS and isinstance(node.ctx, (ast.Load, ast.Store)):
            raise SandboxSecurityError(
                f"Use of forbidden module reference '{node.id}' is prohibited."
            )
        self.generic_visit(node)


def static_check(code: str) -> None:
    """Validate source code against strict security policies using Python AST.

    Args:
        code: Plain Python source code string.

    Raises:
        SandboxSecurityError: If code contains forbidden imports, built-ins, or attribute escapes.
    """
    try:
        tree = ast.parse(code)
    except SyntaxError as err:
        raise SandboxSecurityError(f"Syntax error in code: {err}") from err

    visitor = _SecurityASTVisitor()
    visitor.visit(tree)


# ── Subprocess Sandbox Runner ─────────────────────────────────────────────────


def _build_clean_env() -> dict[str, str]:
    """Build a minimal, clean environment dictionary containing no user secrets or API keys."""
    clean: dict[str, str] = {}

    # Inherit essential OS system paths needed for Python runtime execution
    if sys.platform == "win32":
        for var in ("SYSTEMROOT", "WINDIR", "PATH", "PATHEXT", "TEMP", "TMP", "COMSPEC"):
            if var in os.environ:
                clean[var] = os.environ[var]
    else:
        for var in ("PATH", "TMPDIR", "TEMP", "TMP", "LANG", "LC_ALL"):
            if var in os.environ:
                clean[var] = os.environ[var]

    clean["PYTHONUNBUFFERED"] = "1"
    clean["PYTHONDONTWRITEBYTECODE"] = "1"
    return clean


_RUNNER_TEMPLATE = '''"""Internal runner harness executing mitigate() in isolated child process."""
import json
import os
import sys

def main() -> None:
    try:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        if script_dir not in sys.path:
            sys.path.insert(0, script_dir)

        with open("input.json", "r", encoding="utf-8") as f:
            ledger = json.load(f)

        import mitigator

        if not hasattr(mitigator, "mitigate"):
            sys.stderr.write("RUNNER_ERROR: 'mitigate' function not defined in mitigator\\n")
            sys.exit(2)

        result = mitigator.mitigate(ledger)

        if not isinstance(result, (tuple, list)) or len(result) != 2:
            sys.stderr.write("RUNNER_ERROR: mitigate must return tuple[list, list]\\n")
            sys.exit(3)

        corrected, change_log = result[0], result[1]
        if not isinstance(corrected, list) or not isinstance(change_log, list):
            sys.stderr.write("RUNNER_ERROR: corrected and change_log must both be lists\\n")
            sys.exit(4)

        with open("output.json", "w", encoding="utf-8") as f:
            json.dump({"corrected": corrected, "change_log": change_log}, f)

        sys.exit(0)
    except Exception as exc:
        sys.stderr.write(f"RUNNER_EXCEPTION: {type(exc).__name__}: {exc}\\n")
        sys.exit(1)

if __name__ == "__main__":
    main()
'''


def run_in_sandbox(
    code: str,
    ledger: list[dict[str, Any]],
    timeout_s: float = 10.0,
    max_stdout_bytes: int = MAX_STDOUT_BYTES,
) -> SandboxResult:
    """Execute remediation code against a ledger inside a restricted, isolated subprocess.

    Args:
        code: Python code defining `def mitigate(ledger) -> tuple[list, list]`.
        ledger: Anomaly ledger transaction records.
        timeout_s: Timeout in seconds for the subprocess execution.
        max_stdout_bytes: Cap on standard output size in bytes.

    Returns:
        SandboxResult containing (corrected_ledger, change_log, stdout, exit_code, duration).

    Raises:
        SandboxSecurityError: If static_check rejects the code.
        SandboxTimeoutError: If execution exceeds timeout_s.
        SandboxOutputLimitError: If process generates output exceeding max_stdout_bytes.
    """
    # 1. Run static AST security check first
    static_check(code)

    # 2. Setup temporary directory as execution jail
    temp_dir = tempfile.mkdtemp(prefix="uacp_sandbox_")
    start_time = time.perf_counter()

    try:
        mitigator_path = os.path.join(temp_dir, "mitigator.py")
        input_path = os.path.join(temp_dir, "input.json")
        runner_path = os.path.join(temp_dir, "runner.py")
        output_path = os.path.join(temp_dir, "output.json")

        with open(mitigator_path, "w", encoding="utf-8") as f:
            f.write(code)

        with open(input_path, "w", encoding="utf-8") as f:
            json.dump(ledger, f)

        with open(runner_path, "w", encoding="utf-8") as f:
            f.write(_RUNNER_TEMPLATE)

        # Build clean environment with zero secrets
        env = _build_clean_env()

        # POSIX resource limits (CPU time and memory)
        preexec_fn = None
        if sys.platform != "win32":

            def _apply_posix_limits() -> None:
                try:
                    import resource

                    cpu_limit = max(1, int(timeout_s) + 2)
                    resource.setrlimit(resource.RLIMIT_CPU, (cpu_limit, cpu_limit + 1))
                    # Memory limit: 512 MB
                    mem_bytes = 512 * 1024 * 1024
                    resource.setrlimit(resource.RLIMIT_AS, (mem_bytes, mem_bytes))
                except (ImportError, AttributeError, ValueError, OSError):
                    pass

            preexec_fn = _apply_posix_limits

        # Launch Python in isolated mode (-I)
        proc = subprocess.Popen(
            [sys.executable, "-I", "runner.py"],
            cwd=temp_dir,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            preexec_fn=preexec_fn,
        )

        try:
            stdout_bytes, stderr_bytes = proc.communicate(timeout=timeout_s)
        except subprocess.TimeoutExpired as exc:
            proc.kill()
            proc.communicate()
            duration = time.perf_counter() - start_time
            raise SandboxTimeoutError(
                f"Sandbox execution timed out after {timeout_s:.1f}s."
            ) from exc

        duration = time.perf_counter() - start_time

        # Check output limits
        if len(stdout_bytes) > max_stdout_bytes or len(stderr_bytes) > max_stdout_bytes:
            raise SandboxOutputLimitError(
                f"Sandbox output exceeded limit of {max_stdout_bytes} bytes "
                f"(stdout={len(stdout_bytes)}, stderr={len(stderr_bytes)})."
            )

        stdout_text = stdout_bytes.decode("utf-8", errors="replace")
        stderr_text = stderr_bytes.decode("utf-8", errors="replace")
        combined_stdout = stdout_text + ("\n" + stderr_text if stderr_text else "")

        if proc.returncode != 0:
            return SandboxResult(
                corrected_ledger=[],
                change_log=[],
                stdout=combined_stdout.strip(),
                exit_code=proc.returncode,
                duration=duration,
                status="error",
                error=stderr_text.strip() or f"Process exited with code {proc.returncode}",
            )

        # Read back outputs
        if not os.path.isfile(output_path):
            return SandboxResult(
                corrected_ledger=[],
                change_log=[],
                stdout=combined_stdout.strip(),
                exit_code=1,
                duration=duration,
                status="error",
                error="Runner did not produce output.json",
            )

        with open(output_path, encoding="utf-8") as f:
            data = json.load(f)

        corrected_ledger = data.get("corrected", [])
        change_log = data.get("change_log", [])

        return SandboxResult(
            corrected_ledger=corrected_ledger,
            change_log=change_log,
            stdout=combined_stdout.strip(),
            exit_code=0,
            duration=duration,
            status="success",
            error=None,
        )

    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)
