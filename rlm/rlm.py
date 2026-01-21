"""RLM - Recursive Language Model Orchestrator (Paper-aligned minimal)"""

import logging
from dataclasses import dataclass, field
from typing import Optional, Union

from .config import Config, default_config
from .llm.client import LLMClient
from .llm.prompts import get_system_prompt
from .core.state import REPLState, create_llm_query_function
from .core.executor import execute_code
from .core.tracking import UsageTracker
from .core.parser import (
    extract_code_blocks,
    detect_final_answer,
    detect_final_var,
)

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
    Main RLM orchestrator (paper-aligned minimal).

    Coordinates the iterative loop between LLM and REPL environment.
    """

    def __init__(self, config: Config = None):
        self.config = config or default_config
        self.llm_client = LLMClient(self.config)
        self.state: Optional[REPLState] = None
        self.tracker: Optional[UsageTracker] = None

    def run(
        self,
        query: str,
        context: Union[str, list[str]],
        context_type: str = None
    ) -> RLMResult:
        """
        Run the RLM loop to answer a query given context.

        Args:
            query: The question/task to answer
            context: The context string or list of strings
            context_type: Type description for the prompt (auto-detected if None)

        Returns:
            RLMResult with the answer and metadata
        """
        # Initialize state and tracker
        self.state = REPLState()
        self.state.initialize_context(context)
        self.tracker = UsageTracker()

        # Compute context_lengths based on context type
        if isinstance(context, str):
            context_type = context_type or "string"
            context_lengths = [len(context)]
            context_total_length = len(context)
        else:
            context_type = context_type or "List[str]"
            context_lengths = [len(c) for c in context]
            context_total_length = sum(context_lengths)

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
                warning_threshold=self.config.sub_call_warning_threshold
            )
            self.state.add_llm_query_function(llm_query_func)

        # Build system prompt (paper-aligned)
        system_prompt = get_system_prompt(
            context_type=context_type,
            context_total_length=context_total_length,
            context_lengths=context_lengths,
            model_name=self.config.root_model,
            with_subcalls=use_subcalls
        )

        # Initialize conversation
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": query}
        ]

        if self.config.debug:
            logger.info(f"Starting RLM run: query='{query[:100]}...', context_length={context_total_length}")

        # Iterative loop
        for iteration in range(1, self.config.max_iterations + 1):
            if self.config.debug:
                logger.info(f"=== Iteration {iteration} ===")

            # Call LLM
            try:
                llm_response = self.llm_client.call_with_usage(messages, model=self.config.root_model)
                response = llm_response.content
                input_tokens = llm_response.input_tokens
                output_tokens = llm_response.output_tokens

                # Track root call
                self.tracker.log_call(
                    call_type="root",
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    input_chars=sum(len(m.get("content", "")) for m in messages),
                    output_chars=len(response),
                    model=self.config.root_model
                )
            except Exception as e:
                logger.error(f"LLM call failed at iteration {iteration}: {e}")
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

            # Check for FINAL answer
            is_final, final_answer = detect_final_answer(response)
            if is_final:
                if self.config.debug:
                    logger.info(f"FINAL detected: {final_answer[:200]}...")
                return RLMResult(
                    answer=final_answer,
                    iterations=iteration,
                    success=True,
                    execution_history=self.state.history,
                    sub_calls=self.tracker.sub_calls,
                    usage_stats=self.tracker.get_summary()
                )

            # Check for FINAL_VAR
            is_final_var, var_name = detect_final_var(response)
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

            # Extract and execute code blocks
            code_blocks = extract_code_blocks(response)

            if not code_blocks:
                # No code blocks, no FINAL - prompt LLM to continue
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

                result = execute_code(
                    code=code,
                    globals_dict=self.state.globals,
                    timeout=self.config.execution_timeout
                )

                # Record execution
                self.state.add_execution_record(
                    iteration=iteration,
                    code=code,
                    output=result.stdout if result.success else result.error,
                    success=result.success,
                    error=result.error
                )

                if result.success:
                    output = result.stdout
                    if result.stderr:
                        output += f"\n[stderr]: {result.stderr}"
                else:
                    output = f"Error: {result.error}"

                all_output.append(output)

            # Combine outputs and truncate
            combined_output = "\n---\n".join(all_output)
            truncated_output = self._truncate_output(combined_output)

            if self.config.debug:
                logger.debug(f"Execution output ({len(combined_output)} chars, truncated to {len(truncated_output)}):\n{truncated_output[:500]}...")

            # Add to conversation
            messages.append({"role": "assistant", "content": response})
            messages.append({"role": "user", "content": f"REPL Output:\n{truncated_output}"})

        # Max iterations reached
        logger.warning(f"Max iterations ({self.config.max_iterations}) reached without FINAL answer")
        return RLMResult(
            answer="",
            iterations=self.config.max_iterations,
            success=False,
            error=f"Max iterations ({self.config.max_iterations}) reached without finding answer",
            execution_history=self.state.history,
            sub_calls=self.tracker.sub_calls,
            usage_stats=self.tracker.get_summary()
        )

    def _truncate_output(self, output: str, max_chars: int = None) -> str:
        """
        Truncate output: show beginning and end of long outputs.

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
