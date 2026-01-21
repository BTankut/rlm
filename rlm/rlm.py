"""RLM - Recursive Language Model Orchestrator"""

import logging
import time
from dataclasses import dataclass, field
from typing import Optional, Tuple

from .config import Config, default_config
from .llm.client import LLMClient
from .llm.async_client import AsyncLLMClient
from .llm.prompts import get_system_prompt
from .core.state import REPLState, create_llm_query_function, create_llm_query_batch_function, SubCallLimitError
from .core.executor import execute_code
from .core.sandbox import execute_sandboxed, SandboxError
from .core.tracking import UsageTracker
from .core.cache import LLMCache, get_cache
from .core.logging import RLMLogger, get_logger as get_rlm_logger
from .core.parser import (
    extract_code_blocks,
    detect_final_answer,
    detect_final_var,
    has_code_block,
    has_final
)
from .errors import RLMError, MaxIterationsError, LLMError

logger = logging.getLogger(__name__)


@dataclass
class RLMResult:
    """Result of an RLM run"""
    answer: str
    iterations: int
    success: bool
    error: Optional[str] = None
    execution_history: list = None
    sub_calls: int = 0
    usage_stats: dict = field(default_factory=dict)


class RLM:
    """
    Main RLM orchestrator.

    Coordinates the iterative loop between LLM and REPL environment.
    """

    def __init__(self, config: Config = None, use_cache: bool = False, enable_async: bool = False):
        self.config = config or default_config
        self.llm_client = LLMClient(self.config)
        self.async_llm_client = AsyncLLMClient(self.config) if enable_async else None
        self.state: Optional[REPLState] = None
        self.tracker: Optional[UsageTracker] = None
        self.use_cache = use_cache
        self.cache = get_cache() if use_cache else None
        self.rlm_logger = get_rlm_logger("rlm") if self.config.debug else None

    def _call_llm_with_retry(self, messages: list, model: str) -> Tuple[str, int, int]:
        """
        Call LLM with retry and exponential backoff.

        Args:
            messages: Messages to send to LLM
            model: Model name

        Returns:
            Tuple of (response_content, input_tokens, output_tokens)

        Raises:
            LLMError: If all retries fail
        """
        last_error = None
        for attempt in range(self.config.max_llm_retries):
            try:
                llm_response = self.llm_client.call_with_usage(messages, model=model)
                return llm_response.content, llm_response.input_tokens, llm_response.output_tokens
            except Exception as e:
                last_error = e
                if attempt < self.config.max_llm_retries - 1:
                    # Exponential backoff
                    wait_time = self.config.retry_backoff_base ** attempt
                    if self.config.debug:
                        logger.warning(f"LLM call failed (attempt {attempt + 1}), retrying in {wait_time}s: {e}")
                    if self.rlm_logger:
                        self.rlm_logger.log_error(
                            "llm_retry",
                            f"Attempt {attempt + 1} failed: {e}",
                            {"wait_time": wait_time}
                        )
                    time.sleep(wait_time)

        # All retries failed
        raise LLMError(f"LLM call failed after {self.config.max_llm_retries} attempts: {last_error}", model)

    def run(self, query: str, context: str, context_type: str = "string") -> RLMResult:
        """
        Run the RLM loop to answer a query given context.

        Args:
            query: The question/task to answer
            context: The context string (can be very large)
            context_type: Type description for the prompt

        Returns:
            RLMResult with the answer and metadata
        """
        # Initialize state and tracker
        self.state = REPLState()
        self.state.initialize_context(context, context_type)
        self.tracker = UsageTracker()

        # Get context info for prompt
        context_info = self.state.get_context_info()

        # Determine if sub-calls are enabled
        use_subcalls = self.config.enable_sub_calls

        # Set up llm_query function if sub-calls are enabled
        if use_subcalls:
            llm_query_func = create_llm_query_function(
                llm_client=self.llm_client,
                model=self.config.sub_model,
                tracker=self.tracker,
                max_chars=self.config.max_sub_call_chars,
                max_sub_calls=self.config.max_sub_calls,
                warning_threshold=self.config.sub_call_warning_threshold,
                rlm_logger=self.rlm_logger
            )
            self.state.add_llm_query_function(llm_query_func)

            # Add llm_query_batch if async client is available
            if self.async_llm_client:
                llm_query_batch_func = create_llm_query_batch_function(
                    async_llm_client=self.async_llm_client,
                    model=self.config.sub_model,
                    tracker=self.tracker,
                    max_chars=self.config.max_sub_call_chars,
                    max_concurrent=5,
                    rlm_logger=self.rlm_logger
                )
                self.state.globals['llm_query_batch'] = llm_query_batch_func

        # Build system prompt
        system_prompt = get_system_prompt(
            context_type=context_info['context_type'],
            context_total_length=context_info['context_total_length'],
            with_subcalls=use_subcalls
        )

        # Initialize conversation
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": query}
        ]

        if self.config.debug:
            logger.info(f"Starting RLM run: query='{query[:100]}...', context_length={context_info['context_total_length']}")

        # Start structured logging
        if self.rlm_logger:
            self.rlm_logger.start_run(query, context_info['context_total_length'])

        # Track consecutive execution failures for error recovery
        consecutive_exec_failures = 0

        # Iterative loop
        for iteration in range(1, self.config.max_iterations + 1):
            if self.config.debug:
                logger.info(f"=== Iteration {iteration} ===")

            # Call LLM (with optional caching and retry)
            try:
                # Create cache key from messages
                cache_key = str(messages) if self.use_cache else None
                input_tokens = 0
                output_tokens = 0

                # Check cache first
                if self.cache and cache_key:
                    cached = self.cache.get(cache_key, self.config.root_model)
                    if cached:
                        response = cached
                        if self.config.debug:
                            logger.debug("Cache hit!")
                    else:
                        # Use retry-enabled call
                        response, input_tokens, output_tokens = self._call_llm_with_retry(
                            messages, self.config.root_model
                        )
                        self.cache.set(cache_key, self.config.root_model, response)
                else:
                    # Use retry-enabled call
                    response, input_tokens, output_tokens = self._call_llm_with_retry(
                        messages, self.config.root_model
                    )

                # Track root call with real token counts
                self.tracker.log_call(
                    call_type="root",
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    input_chars=sum(len(m.get("content", "")) for m in messages),
                    output_chars=len(response),
                    model=self.config.root_model
                )

            except LLMError as e:
                logger.error(f"LLM call failed at iteration {iteration}: {e}")
                if self.rlm_logger:
                    self.rlm_logger.log_error("llm_call_failed", str(e), {"iteration": iteration})
                return RLMResult(
                    answer="",
                    iterations=iteration,
                    success=False,
                    error=f"LLM call failed: {e}",
                    execution_history=self.state.history,
                    sub_calls=self.tracker.sub_calls,
                    usage_stats=self.tracker.get_summary()
                )

            if self.config.debug:
                logger.debug(f"LLM response ({len(response)} chars): {response[:500]}...")

            # Extract code blocks FIRST - we execute them before checking FINAL
            # This ensures code gets executed even when LLM outputs code + FINAL together
            code_blocks = extract_code_blocks(response)

            # Check for FINAL answer (but don't return yet - execute code first if present)
            is_final, final_answer = detect_final_answer(response)

            # Check for FINAL_VAR
            is_final_var, var_name = detect_final_var(response)

            if not code_blocks:
                # No code blocks - check if we can accept FINAL
                if is_final:
                    # FINAL without code blocks - require prior code execution
                    if len(self.state.history) > 0:
                        # Code was executed previously, accept FINAL
                        if self.config.debug:
                            logger.info(f"FINAL detected (no new code): {final_answer[:200]}...")
                        if self.rlm_logger:
                            self.rlm_logger.log_final(
                                answer=final_answer,
                                iterations=iteration,
                                sub_calls=self.tracker.sub_calls,
                                success=True
                            )
                        return RLMResult(
                            answer=final_answer,
                            iterations=iteration,
                            success=True,
                            execution_history=self.state.history,
                            sub_calls=self.tracker.sub_calls,
                            usage_stats=self.tracker.get_summary()
                        )
                    else:
                        # No code executed yet, reject FINAL
                        if self.config.debug:
                            logger.info("FINAL detected but no code executed yet - rejecting")
                        messages.append({"role": "assistant", "content": response})
                        messages.append({
                            "role": "user",
                            "content": "ERROR: You must run print(context) FIRST before using FINAL(). "
                                       "Execute code to read the actual data, then use FINAL() with the real answer."
                        })
                        continue

                if is_final_var:
                    var_value = self.state.get_variable(var_name)
                    if var_value is not None:
                        answer = str(var_value)
                        if self.config.debug:
                            logger.info(f"FINAL_VAR({var_name}) detected: {answer[:200]}...")
                        return RLMResult(
                            answer=answer,
                            iterations=iteration,
                            success=True,
                            execution_history=self.state.history,
                            sub_calls=self.tracker.sub_calls,
                            usage_stats=self.tracker.get_summary()
                        )
                    else:
                        # Variable not found, inform LLM
                        error_msg = f"Error: Variable '{var_name}' not found in REPL environment."
                        messages.append({"role": "assistant", "content": response})
                        messages.append({"role": "user", "content": error_msg})
                        continue

                # No code blocks, no FINAL, no FINAL_VAR - prompt LLM to continue
                if self.config.debug:
                    logger.debug("No code blocks found, prompting to continue...")
                messages.append({"role": "assistant", "content": response})
                messages.append({
                    "role": "user",
                    "content": "Please write code to explore the context or provide your FINAL() answer."
                })
                continue

            # Execute all code blocks and collect output
            all_output = []
            for i, code in enumerate(code_blocks):
                if self.config.debug:
                    logger.debug(f"Executing code block {i + 1}:\n{code[:300]}...")

                # Use sandbox if configured, otherwise use regular executor
                if self.config.use_sandbox:
                    sandbox_result = execute_sandboxed(
                        code=code,
                        globals_dict=self.state.globals,
                        timeout=self.config.execution_timeout,
                        use_restricted_python=self.config.sandbox_use_restricted_python
                    )
                    # Convert sandbox result to match executor result format
                    result_success = sandbox_result.success and not sandbox_result.security_violation
                    result_stdout = sandbox_result.stdout
                    result_stderr = sandbox_result.stderr
                    result_error = sandbox_result.error

                    # Report security violations
                    if sandbox_result.security_violation:
                        result_error = f"SECURITY VIOLATION: {sandbox_result.error}"
                        if self.rlm_logger:
                            self.rlm_logger.log_error(
                                "security_violation",
                                sandbox_result.error,
                                {"code_preview": code[:200], "iteration": iteration}
                            )
                else:
                    result = execute_code(
                        code=code,
                        globals_dict=self.state.globals,
                        timeout=self.config.execution_timeout
                    )
                    result_success = result.success
                    result_stdout = result.stdout
                    result_stderr = result.stderr
                    result_error = result.error

                # Record execution
                self.state.add_execution_record(
                    iteration=iteration,
                    code=code,
                    output=result_stdout if result_success else result_error,
                    success=result_success,
                    error=result_error
                )

                # Log iteration with RLMLogger
                if self.rlm_logger:
                    self.rlm_logger.log_iteration(
                        iteration=iteration,
                        code=code,
                        output=result_stdout if result_success else (result_error or ""),
                        success=result_success,
                        error=result_error
                    )

                if result_success:
                    output = result_stdout
                    if result_stderr:
                        output += f"\n[stderr]: {result_stderr}"
                    consecutive_exec_failures = 0  # Reset on success
                else:
                    output = f"Error: {result_error}"
                    consecutive_exec_failures += 1

                    # Check if we've hit the execution retry limit
                    if consecutive_exec_failures >= self.config.max_execution_retries:
                        if self.rlm_logger:
                            self.rlm_logger.log_error(
                                "max_execution_retries",
                                f"Hit {consecutive_exec_failures} consecutive execution failures",
                                {"last_error": result_error}
                            )
                        # Include helpful message for LLM to change approach
                        output += f"\n\n[SYSTEM: {consecutive_exec_failures} consecutive execution errors. Please try a different approach or simpler code.]"

                all_output.append(output)

            # Combine outputs and truncate
            combined_output = "\n---\n".join(all_output)
            truncated_output = self._smart_truncate(combined_output)

            if self.config.debug:
                logger.debug(f"Execution output ({len(combined_output)} chars, truncated to {len(truncated_output)}):\n{truncated_output[:500]}...")

            # If LLM output code + FINAL together, DON'T accept the FINAL yet
            # The FINAL was written before seeing code output, so it may be hallucinated
            # Instead, send the code output back and let LLM emit a new FINAL
            if is_final:
                if self.config.debug:
                    logger.info(f"FINAL detected with code - ignoring until LLM sees output")
                # Don't return - continue to send output back to LLM

            if is_final_var:
                if self.config.debug:
                    logger.info(f"FINAL_VAR detected with code - ignoring until LLM sees output")
                # Don't return - continue to send output back to LLM

            # Add to conversation
            messages.append({"role": "assistant", "content": response})
            messages.append({"role": "user", "content": f"REPL Output:\n{truncated_output}"})

        # Max iterations reached
        logger.warning(f"Max iterations ({self.config.max_iterations}) reached without FINAL answer")
        if self.rlm_logger:
            self.rlm_logger.log_final(
                answer="",
                iterations=self.config.max_iterations,
                sub_calls=self.tracker.sub_calls,
                success=False,
                error=f"Max iterations ({self.config.max_iterations}) reached"
            )
            self.rlm_logger.log_error(
                "max_iterations_reached",
                f"Max iterations ({self.config.max_iterations}) reached without finding answer",
                {"iterations": self.config.max_iterations}
            )
        return RLMResult(
            answer="",
            iterations=self.config.max_iterations,
            success=False,
            error=f"Max iterations ({self.config.max_iterations}) reached without finding answer",
            execution_history=self.state.history,
            sub_calls=self.tracker.sub_calls,
            usage_stats=self.tracker.get_summary()
        )

    def _smart_truncate(self, output: str, max_chars: int = None) -> str:
        """
        Smart truncation: show beginning and end of long outputs.

        Args:
            output: The output string to truncate
            max_chars: Maximum characters (defaults to config.max_output_chars)

        Returns:
            Truncated string
        """
        max_chars = max_chars or self.config.max_output_chars

        if len(output) <= max_chars:
            return output

        # Show first 40% and last 40%
        first_part = int(max_chars * 0.4)
        last_part = int(max_chars * 0.4)
        middle_chars = len(output) - first_part - last_part

        return (
            output[:first_part] +
            f"\n\n[... TRUNCATED {middle_chars} characters ...]\n\n" +
            output[-last_part:]
        )

    def get_usage_stats(self) -> dict:
        """Get LLM usage statistics"""
        return self.llm_client.get_usage_stats()
