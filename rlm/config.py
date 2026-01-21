"""RLM Configuration"""

import os
from dataclasses import dataclass, field
from dotenv import load_dotenv

load_dotenv()


@dataclass
class Config:
    """RLM Configuration settings"""

    # Ollama settings
    ollama_base_url: str = field(
        default_factory=lambda: os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1")
    )

    # Model settings
    root_model: str = field(
        default_factory=lambda: os.getenv("ROOT_MODEL", "qwen3:14b")
    )
    sub_model: str = field(
        default_factory=lambda: os.getenv("SUB_MODEL", None) or os.getenv("ROOT_MODEL", "qwen3:14b")
    )

    # Execution limits
    max_iterations: int = field(
        default_factory=lambda: int(os.getenv("MAX_ITERATIONS", "20"))
    )
    max_output_chars: int = field(
        default_factory=lambda: int(os.getenv("MAX_OUTPUT_CHARS", "10000"))
    )
    max_context_chars_display: int = field(
        default_factory=lambda: int(os.getenv("MAX_CONTEXT_CHARS_DISPLAY", "500"))
    )
    execution_timeout: int = field(
        default_factory=lambda: int(os.getenv("EXECUTION_TIMEOUT", "30"))
    )

    # Sub-call settings (Phase 2)
    max_sub_calls: int = field(
        default_factory=lambda: int(os.getenv("MAX_SUB_CALLS", "100"))
    )
    sub_call_warning_threshold: int = field(
        default_factory=lambda: int(os.getenv("SUB_CALL_WARNING_THRESHOLD", "50"))
    )
    max_sub_call_chars: int = field(
        default_factory=lambda: int(os.getenv("MAX_SUB_CALL_CHARS", "500000"))
    )

    # Enable sub-calls (llm_query function)
    enable_sub_calls: bool = field(
        default_factory=lambda: os.getenv("ENABLE_SUB_CALLS", "true").lower() == "true"
    )

    # LLM settings
    temperature: float = field(
        default_factory=lambda: float(os.getenv("TEMPERATURE", "0.7"))
    )

    # Debug
    debug: bool = field(
        default_factory=lambda: os.getenv("DEBUG", "false").lower() == "true"
    )

    # Sandbox settings (Phase 3)
    use_sandbox: bool = field(
        default_factory=lambda: os.getenv("USE_SANDBOX", "false").lower() == "true"
    )
    sandbox_use_restricted_python: bool = field(
        default_factory=lambda: os.getenv("SANDBOX_USE_RESTRICTED_PYTHON", "true").lower() == "true"
    )

    # Error recovery settings
    max_llm_retries: int = field(
        default_factory=lambda: int(os.getenv("MAX_LLM_RETRIES", "3"))
    )
    retry_backoff_base: float = field(
        default_factory=lambda: float(os.getenv("RETRY_BACKOFF_BASE", "2.0"))
    )
    max_execution_retries: int = field(
        default_factory=lambda: int(os.getenv("MAX_EXECUTION_RETRIES", "3"))
    )


# Default configuration instance
default_config = Config()
