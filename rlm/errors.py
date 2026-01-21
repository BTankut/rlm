"""RLM Error Classes"""


class RLMError(Exception):
    """Base exception for RLM errors"""
    pass


class MaxIterationsError(RLMError):
    """Raised when maximum iterations are exceeded without finding an answer"""

    def __init__(self, iterations: int, message: str = None):
        self.iterations = iterations
        self.message = message or f"Maximum iterations ({iterations}) reached without finding answer"
        super().__init__(self.message)


class ExecutionError(RLMError):
    """Raised when code execution fails"""

    def __init__(self, code: str, error: str):
        self.code = code
        self.error = error
        super().__init__(f"Code execution failed: {error}")


class LLMError(RLMError):
    """Raised when LLM call fails"""

    def __init__(self, message: str, model: str = None):
        self.model = model
        super().__init__(message)


class SubCallLimitError(RLMError):
    """Raised when sub-call limit is exceeded"""

    def __init__(self, limit: int, message: str = None):
        self.limit = limit
        self.message = message or f"Sub-call limit ({limit}) exceeded"
        super().__init__(self.message)


class SandboxError(RLMError):
    """Raised when sandboxed code violates security policies"""

    def __init__(self, violation: str):
        self.violation = violation
        super().__init__(f"Security violation: {violation}")


class ContextTooLargeError(RLMError):
    """Raised when context exceeds maximum allowed size"""

    def __init__(self, size: int, max_size: int):
        self.size = size
        self.max_size = max_size
        super().__init__(f"Context size ({size}) exceeds maximum ({max_size})")


class ParseError(RLMError):
    """Raised when parsing LLM response fails"""

    def __init__(self, response: str, message: str):
        self.response = response
        super().__init__(message)
