"""Structured logging for RLM"""

import logging
import json
from datetime import datetime
from typing import Optional
from pathlib import Path


class RLMLogger:
    """
    Structured logging for RLM runs.

    Provides JSON-formatted logs for analysis and debugging.
    """

    def __init__(self, name: str = "rlm", log_file: Optional[str] = None):
        self.logger = logging.getLogger(name)
        self.run_id: Optional[str] = None
        self.log_file = log_file

        # Set up file handler if specified
        if log_file:
            Path(log_file).parent.mkdir(parents=True, exist_ok=True)
            file_handler = logging.FileHandler(log_file)
            file_handler.setFormatter(logging.Formatter('%(message)s'))
            self.logger.addHandler(file_handler)

    def start_run(self, query: str, context_length: int, config: dict = None):
        """Log the start of an RLM run"""
        self.run_id = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        self.log("run_start", {
            "query_preview": query[:200] if len(query) > 200 else query,
            "query_length": len(query),
            "context_length": context_length,
            "config": config or {}
        })

    def log_iteration(
        self,
        iteration: int,
        code: str,
        output: str,
        success: bool,
        error: Optional[str] = None
    ):
        """Log a single iteration"""
        self.log("iteration", {
            "iteration": iteration,
            "code_length": len(code),
            "code_preview": code[:300] if len(code) > 300 else code,
            "output_length": len(output),
            "output_preview": output[:300] if len(output) > 300 else output,
            "success": success,
            "error": error
        })

    def log_sub_call(
        self,
        prompt_length: int,
        response_length: int,
        model: str = None
    ):
        """Log a sub-call (llm_query)"""
        self.log("sub_call", {
            "prompt_length": prompt_length,
            "response_length": response_length,
            "model": model
        })

    def log_final(
        self,
        answer: str,
        iterations: int,
        sub_calls: int,
        success: bool,
        error: Optional[str] = None
    ):
        """Log the final result"""
        self.log("run_end", {
            "answer_preview": answer[:200] if len(answer) > 200 else answer,
            "answer_length": len(answer),
            "total_iterations": iterations,
            "total_sub_calls": sub_calls,
            "success": success,
            "error": error
        })

    def log_error(self, error_type: str, message: str, details: dict = None):
        """Log an error"""
        self.log("error", {
            "error_type": error_type,
            "message": message,
            "details": details or {}
        })

    def log(self, event: str, data: dict):
        """Write a structured log entry"""
        entry = {
            "timestamp": datetime.now().isoformat(),
            "run_id": self.run_id,
            "event": event,
            **data
        }
        self.logger.info(json.dumps(entry))

    def get_run_id(self) -> Optional[str]:
        """Get current run ID"""
        return self.run_id


# Default logger instance
default_logger = RLMLogger("rlm")


def get_logger(name: str = "rlm", log_file: str = None) -> RLMLogger:
    """Get or create a logger instance"""
    return RLMLogger(name, log_file)
