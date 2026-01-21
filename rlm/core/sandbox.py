"""Sandboxed code execution using RestrictedPython"""

import logging
from dataclasses import dataclass
from typing import Any, Optional

logger = logging.getLogger(__name__)

# Safe builtins for sandboxed execution
SAFE_BUILTINS = {
    'abs': abs,
    'all': all,
    'any': any,
    'bool': bool,
    'chr': chr,
    'dict': dict,
    'enumerate': enumerate,
    'filter': filter,
    'float': float,
    'format': format,
    'frozenset': frozenset,
    'getattr': getattr,
    'hasattr': hasattr,
    'hash': hash,
    'hex': hex,
    'int': int,
    'isinstance': isinstance,
    'issubclass': issubclass,
    'iter': iter,
    'len': len,
    'list': list,
    'map': map,
    'max': max,
    'min': min,
    'next': next,
    'oct': oct,
    'ord': ord,
    'pow': pow,
    'print': print,
    'range': range,
    'repr': repr,
    'reversed': reversed,
    'round': round,
    'set': set,
    'slice': slice,
    'sorted': sorted,
    'str': str,
    'sum': sum,
    'tuple': tuple,
    'type': type,
    'zip': zip,
    'True': True,
    'False': False,
    'None': None,
}

# Blocked names that should never be accessible
BLOCKED_NAMES = {
    '__import__',
    'eval',
    'exec',
    'compile',
    'open',
    'file',
    'input',
    'raw_input',
    '__builtins__',
    '__loader__',
    '__spec__',
    'globals',
    'locals',
    'vars',
    'dir',
    'delattr',
    'setattr',
    'breakpoint',
    'exit',
    'quit',
}


@dataclass
class SandboxResult:
    """Result of sandboxed code execution"""
    success: bool
    stdout: str
    stderr: str
    error: Optional[str] = None
    security_violation: bool = False


class SandboxError(Exception):
    """Raised when sandboxed code violates security policies"""
    pass


class Sandbox:
    """
    Sandboxed Python code execution environment.

    Uses RestrictedPython when available, falls back to a simpler
    execution model with blocked builtins.
    """

    def __init__(self, use_restricted_python: bool = True):
        self.use_restricted_python = use_restricted_python
        self._restricted_available = False

        if use_restricted_python:
            try:
                from RestrictedPython import compile_restricted, safe_builtins
                from RestrictedPython.Guards import safe_builtins as rp_safe_builtins
                from RestrictedPython.Eval import default_guarded_getiter, default_guarded_getitem
                self._restricted_available = True
                self._compile_restricted = compile_restricted
                self._rp_safe_builtins = rp_safe_builtins
                self._guarded_getiter = default_guarded_getiter
                self._guarded_getitem = default_guarded_getitem
                logger.info("RestrictedPython sandbox enabled")
            except ImportError:
                logger.warning("RestrictedPython not available, using basic sandbox")
                self._restricted_available = False

    def _check_for_violations(self, code: str) -> Optional[str]:
        """Check code for obvious security violations before execution"""
        for blocked in BLOCKED_NAMES:
            if blocked in code:
                return f"Security violation: '{blocked}' is not allowed"

        # Check for import statements
        if 'import ' in code or 'from ' in code:
            # Allow safe imports like 're', 'json', 'math'
            safe_modules = {'re', 'json', 'math', 'datetime', 'collections', 'itertools', 'functools'}
            import re as re_module
            imports = re_module.findall(r'(?:import|from)\s+(\w+)', code)
            for imp in imports:
                if imp not in safe_modules:
                    return f"Security violation: importing '{imp}' is not allowed"

        return None

    def execute(
        self,
        code: str,
        globals_dict: dict,
        timeout: float = 30.0
    ) -> SandboxResult:
        """
        Execute code in a sandboxed environment.

        Args:
            code: Python code to execute
            globals_dict: Global variables available to the code
            timeout: Maximum execution time in seconds

        Returns:
            SandboxResult with execution outcome
        """
        import io
        import sys
        from contextlib import redirect_stdout, redirect_stderr

        # Pre-execution security check
        violation = self._check_for_violations(code)
        if violation:
            return SandboxResult(
                success=False,
                stdout="",
                stderr="",
                error=violation,
                security_violation=True
            )

        # Prepare safe globals
        safe_globals = dict(globals_dict)
        safe_globals['__builtins__'] = SAFE_BUILTINS.copy()

        # Add safe modules
        import re
        import json
        import math
        safe_globals['re'] = re
        safe_globals['json'] = json
        safe_globals['math'] = math

        stdout_capture = io.StringIO()
        stderr_capture = io.StringIO()

        try:
            if self._restricted_available:
                # Use RestrictedPython for compilation
                byte_code = self._compile_restricted(
                    code,
                    filename='<sandboxed>',
                    mode='exec'
                )

                if byte_code.errors:
                    return SandboxResult(
                        success=False,
                        stdout="",
                        stderr="",
                        error=f"Compilation errors: {byte_code.errors}",
                        security_violation=True
                    )

                # Add RestrictedPython guards
                safe_globals['_getiter_'] = self._guarded_getiter
                safe_globals['_getitem_'] = self._guarded_getitem
                safe_globals['_write_'] = lambda x: x
                safe_globals['_print_'] = print

                with redirect_stdout(stdout_capture), redirect_stderr(stderr_capture):
                    exec(byte_code.code, safe_globals)
            else:
                # Basic sandbox without RestrictedPython
                compiled = compile(code, '<sandboxed>', 'exec')

                with redirect_stdout(stdout_capture), redirect_stderr(stderr_capture):
                    exec(compiled, safe_globals)

            # Update original globals with new values (excluding builtins)
            for key, value in safe_globals.items():
                if key not in ('__builtins__', '_getiter_', '_getitem_', '_write_', '_print_'):
                    globals_dict[key] = value

            return SandboxResult(
                success=True,
                stdout=stdout_capture.getvalue(),
                stderr=stderr_capture.getvalue()
            )

        except Exception as e:
            return SandboxResult(
                success=False,
                stdout=stdout_capture.getvalue(),
                stderr=stderr_capture.getvalue(),
                error=str(e)
            )


def execute_sandboxed(
    code: str,
    globals_dict: dict,
    timeout: float = 30.0,
    use_restricted_python: bool = True
) -> SandboxResult:
    """
    Convenience function for sandboxed execution.

    Args:
        code: Python code to execute
        globals_dict: Global variables available to the code
        timeout: Maximum execution time in seconds
        use_restricted_python: Whether to use RestrictedPython if available

    Returns:
        SandboxResult with execution outcome
    """
    sandbox = Sandbox(use_restricted_python=use_restricted_python)
    return sandbox.execute(code, globals_dict, timeout)
