"""Python REPL Executor for RLM"""

import io
import sys
import signal
import traceback
from dataclasses import dataclass
from contextlib import redirect_stdout, redirect_stderr
from typing import Optional


@dataclass
class ExecutionResult:
    """Result of code execution"""
    stdout: str
    stderr: str
    success: bool
    error: Optional[str] = None


class TimeoutError(Exception):
    """Raised when code execution times out"""
    pass


def _timeout_handler(signum, frame):
    raise TimeoutError("Code execution timed out")


def execute_code(
    code: str,
    globals_dict: dict,
    timeout: int = 30
) -> ExecutionResult:
    """
    Execute Python code in a REPL-like environment.

    Args:
        code: Python code to execute
        globals_dict: Global variables dict (will be modified with new variables)
        timeout: Maximum execution time in seconds

    Returns:
        ExecutionResult with stdout, stderr, success flag, and any error message

    Note:
        This is a simple exec() based executor for MVP.
        Production should use RestrictedPython or Docker sandbox.
    """
    stdout_capture = io.StringIO()
    stderr_capture = io.StringIO()

    # Set up timeout handler (Unix only)
    old_handler = None
    try:
        old_handler = signal.signal(signal.SIGALRM, _timeout_handler)
        signal.alarm(timeout)
    except (AttributeError, ValueError):
        # signal.SIGALRM not available on Windows
        pass

    try:
        with redirect_stdout(stdout_capture), redirect_stderr(stderr_capture):
            exec(code, globals_dict)

        return ExecutionResult(
            stdout=stdout_capture.getvalue(),
            stderr=stderr_capture.getvalue(),
            success=True,
            error=None
        )

    except TimeoutError as e:
        return ExecutionResult(
            stdout=stdout_capture.getvalue(),
            stderr=stderr_capture.getvalue(),
            success=False,
            error=f"Timeout: {str(e)}"
        )

    except Exception as e:
        # Get the traceback but exclude the exec() call itself
        tb_lines = traceback.format_exception(type(e), e, e.__traceback__)
        # Filter out internal frames
        filtered_tb = []
        skip_next = False
        for line in tb_lines:
            if 'exec(code, globals_dict)' in line:
                skip_next = True
                continue
            if skip_next and line.startswith('  File "<string>"'):
                skip_next = False
            if not skip_next:
                filtered_tb.append(line)

        error_msg = ''.join(filtered_tb) if filtered_tb else f"{type(e).__name__}: {str(e)}"

        return ExecutionResult(
            stdout=stdout_capture.getvalue(),
            stderr=stderr_capture.getvalue(),
            success=False,
            error=error_msg
        )

    finally:
        # Reset alarm and signal handler
        try:
            signal.alarm(0)
            if old_handler is not None:
                signal.signal(signal.SIGALRM, old_handler)
        except (AttributeError, ValueError):
            pass
