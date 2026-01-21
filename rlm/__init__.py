"""RLM - Recursive Language Models Implementation (Paper-aligned minimal)"""

__version__ = "0.2.0"

from .rlm import RLM, RLMResult
from .config import Config
from .errors import (
    RLMError,
    MaxIterationsError,
    ExecutionError,
    LLMError,
    SubCallLimitError,
)

__all__ = [
    'RLM',
    'RLMResult',
    'Config',
    'RLMError',
    'MaxIterationsError',
    'ExecutionError',
    'LLMError',
    'SubCallLimitError',
]
