"""REPL Environment State Management"""

from dataclasses import dataclass, field
from typing import Any, Optional, Callable
from datetime import datetime

from .tracking import UsageTracker


@dataclass
class ExecutionRecord:
    """Record of a single code execution"""
    iteration: int
    code: str
    output: str
    success: bool
    error: Optional[str]
    timestamp: datetime = field(default_factory=datetime.now)


class REPLState:
    """
    Manages the state of the REPL environment.

    Maintains persistent variables across code executions
    and tracks execution history.
    """

    def __init__(self):
        self.globals: dict[str, Any] = {}
        self.history: list[ExecutionRecord] = []
        self.context_info: dict[str, Any] = {}

    def initialize_context(self, context: str, context_type: str = "string"):
        """
        Initialize the context variable in the REPL environment.

        Args:
            context: The context string to make available
            context_type: Type description (e.g., "string", "json", "code")
        """
        self.globals['context'] = context
        self.context_info = {
            'type': context_type,
            'total_length': len(context),
            'line_count': context.count('\n') + 1,
        }

    def get_context_info(self) -> dict:
        """
        Get context metadata for the system prompt.

        Returns:
            Dict with context_type, context_total_length, etc.
        """
        return {
            'context_type': self.context_info.get('type', 'string'),
            'context_total_length': self.context_info.get('total_length', 0),
            'context_line_count': self.context_info.get('line_count', 0),
        }

    def get_context_preview(self, max_chars: int = 500) -> str:
        """
        Get a preview of the context.

        Args:
            max_chars: Maximum characters to show

        Returns:
            Truncated context string
        """
        context = self.globals.get('context', '')
        if len(context) <= max_chars:
            return context
        return context[:max_chars] + f"\n... [{len(context) - max_chars} more characters]"

    def add_execution_record(
        self,
        iteration: int,
        code: str,
        output: str,
        success: bool,
        error: Optional[str] = None
    ):
        """Add a record of code execution to history"""
        record = ExecutionRecord(
            iteration=iteration,
            code=code,
            output=output,
            success=success,
            error=error
        )
        self.history.append(record)

    def get_variable(self, name: str) -> Any:
        """Get a variable from the REPL globals"""
        return self.globals.get(name)

    def set_variable(self, name: str, value: Any):
        """Set a variable in the REPL globals"""
        self.globals[name] = value

    def add_llm_query_function(self, llm_query_func):
        """
        Add the llm_query function to globals.

        Args:
            llm_query_func: The llm_query function to add
        """
        self.globals['llm_query'] = llm_query_func

    def get_execution_summary(self) -> dict:
        """Get a summary of all executions"""
        return {
            'total_executions': len(self.history),
            'successful': sum(1 for r in self.history if r.success),
            'failed': sum(1 for r in self.history if not r.success),
            'variables': list(self.globals.keys())
        }

    def reset(self):
        """Reset the state to initial"""
        self.globals = {}
        self.history = []
        self.context_info = {}


class SubCallLimitError(Exception):
    """Raised when sub-call limit is exceeded"""
    pass


def create_llm_query_function(
    llm_client,
    model: str,
    tracker: UsageTracker,
    max_chars: int = 500000,
    max_sub_calls: int = 100,
    warning_threshold: int = 50,
    rlm_logger=None
) -> Callable[[str], str]:
    """
    Create the llm_query function for use in REPL environment.

    This function allows the LLM to make recursive sub-calls to itself
    for processing chunks of context.

    Args:
        llm_client: The LLM client to use for calls
        model: Model name for sub-calls
        tracker: UsageTracker for logging calls
        max_chars: Maximum characters per query
        max_sub_calls: Maximum number of sub-calls allowed
        warning_threshold: Show warning after this many calls
        rlm_logger: Optional RLMLogger for structured logging

    Returns:
        The llm_query function for the REPL environment
    """
    call_count = [0]  # Use list for mutable closure

    def llm_query(prompt: str) -> str:
        """
        Query an LLM from within the REPL environment.

        Args:
            prompt: The prompt to send to the LLM

        Returns:
            The LLM's response as a string
        """
        nonlocal call_count

        # Check sub-call limit
        call_count[0] += 1
        if call_count[0] > max_sub_calls:
            raise SubCallLimitError(
                f"Sub-call limit ({max_sub_calls}) exceeded. "
                "Consider batching more information per call."
            )

        # Warning at threshold
        if call_count[0] == warning_threshold:
            print(f"[WARNING] {warning_threshold} sub-calls made. "
                  f"Limit is {max_sub_calls}. Consider batching.")

        # Truncate if needed
        original_len = len(prompt)
        if len(prompt) > max_chars:
            prompt = prompt[:max_chars] + "\n[TRUNCATED]"

        # Track input
        input_chars = len(prompt)

        # Make LLM call with token tracking
        llm_response = llm_client.call_with_usage(
            messages=[{"role": "user", "content": prompt}],
            model=model
        )

        # Track the call with real token counts
        tracker.log_call(
            call_type="sub",
            input_tokens=llm_response.input_tokens,
            output_tokens=llm_response.output_tokens,
            input_chars=input_chars,
            output_chars=len(llm_response.content),
            model=model
        )

        # Log sub-call if logger available
        if rlm_logger:
            rlm_logger.log_sub_call(
                prompt_length=input_chars,
                response_length=len(llm_response.content),
                model=model
            )

        return llm_response.content

    return llm_query


def create_llm_query_batch_function(
    async_llm_client,
    model: str,
    tracker: UsageTracker,
    max_chars: int = 500000,
    max_concurrent: int = 5,
    rlm_logger=None
) -> Callable:
    """
    Create an async llm_query_batch function for parallel sub-calls.

    Args:
        async_llm_client: AsyncLLMClient for async calls
        model: Model name for sub-calls
        tracker: UsageTracker for logging calls
        max_chars: Maximum characters per query
        max_concurrent: Maximum concurrent calls
        rlm_logger: Optional RLMLogger for structured logging

    Returns:
        Async llm_query_batch function for the REPL environment
    """
    import asyncio

    async def llm_query_batch(prompts: list[str]) -> list[str]:
        """
        Query LLM with multiple prompts in parallel.

        Args:
            prompts: List of prompts to process

        Returns:
            List of responses in the same order as prompts
        """
        # Truncate prompts if needed
        truncated_prompts = []
        for prompt in prompts:
            if len(prompt) > max_chars:
                truncated_prompts.append(prompt[:max_chars] + "\n[TRUNCATED]")
            else:
                truncated_prompts.append(prompt)

        # Create message batches
        message_batches = [
            [{"role": "user", "content": prompt}]
            for prompt in truncated_prompts
        ]

        # Make parallel calls
        responses = await async_llm_client.call_batch(
            message_batches,
            model=model,
            max_concurrent=max_concurrent
        )

        # Track and log each call
        results = []
        for i, response in enumerate(responses):
            tracker.log_call(
                call_type="sub",
                input_tokens=response.input_tokens,
                output_tokens=response.output_tokens,
                input_chars=len(truncated_prompts[i]),
                output_chars=len(response.content),
                model=model
            )

            if rlm_logger:
                rlm_logger.log_sub_call(
                    prompt_length=len(truncated_prompts[i]),
                    response_length=len(response.content),
                    model=model
                )

            results.append(response.content)

        return results

    def sync_wrapper(prompts: list[str]) -> list[str]:
        """Sync wrapper for async llm_query_batch"""
        return asyncio.run(llm_query_batch(prompts))

    return sync_wrapper
