"""Token and call tracking for RLM"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional


@dataclass
class CallRecord:
    """Record of a single LLM call"""
    call_type: str  # "root" or "sub"
    input_tokens: int
    output_tokens: int
    input_chars: int
    output_chars: int
    timestamp: datetime = field(default_factory=datetime.now)
    model: Optional[str] = None


class UsageTracker:
    """
    Tracks token usage and LLM calls for cost estimation and monitoring.
    """

    def __init__(self):
        self.root_calls = 0
        self.sub_calls = 0
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.total_input_chars = 0
        self.total_output_chars = 0
        self.call_history: list[CallRecord] = []

    def log_call(
        self,
        call_type: str,
        input_tokens: int = 0,
        output_tokens: int = 0,
        input_chars: int = 0,
        output_chars: int = 0,
        model: str = None
    ):
        """
        Log an LLM call.

        Args:
            call_type: "root" for main orchestrator calls, "sub" for llm_query calls
            input_tokens: Number of input tokens (if available)
            output_tokens: Number of output tokens (if available)
            input_chars: Number of input characters
            output_chars: Number of output characters
            model: Model name used
        """
        record = CallRecord(
            call_type=call_type,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            input_chars=input_chars,
            output_chars=output_chars,
            model=model
        )
        self.call_history.append(record)

        if call_type == "root":
            self.root_calls += 1
        else:
            self.sub_calls += 1

        self.total_input_tokens += input_tokens
        self.total_output_tokens += output_tokens
        self.total_input_chars += input_chars
        self.total_output_chars += output_chars

    def get_summary(self) -> dict:
        """Get summary statistics"""
        return {
            "root_calls": self.root_calls,
            "sub_calls": self.sub_calls,
            "total_calls": self.root_calls + self.sub_calls,
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
            "total_tokens": self.total_input_tokens + self.total_output_tokens,
            "total_input_chars": self.total_input_chars,
            "total_output_chars": self.total_output_chars,
        }

    def estimate_cost(
        self,
        price_per_1k_input: float = 0.0,
        price_per_1k_output: float = 0.0
    ) -> float:
        """
        Estimate cost based on token usage.

        Args:
            price_per_1k_input: Price per 1000 input tokens
            price_per_1k_output: Price per 1000 output tokens

        Returns:
            Estimated cost in the same currency as the prices
        """
        input_cost = (self.total_input_tokens / 1000) * price_per_1k_input
        output_cost = (self.total_output_tokens / 1000) * price_per_1k_output
        return input_cost + output_cost

    def reset(self):
        """Reset all tracking data"""
        self.root_calls = 0
        self.sub_calls = 0
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.total_input_chars = 0
        self.total_output_chars = 0
        self.call_history = []
