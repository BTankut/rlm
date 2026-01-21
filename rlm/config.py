"""RLM Configuration (Paper-aligned minimal)"""

import os
from dataclasses import dataclass, field
from dotenv import load_dotenv

load_dotenv()


@dataclass
class Config:
    """RLM Configuration settings (paper-aligned minimal)"""

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
    execution_timeout: int = field(
        default_factory=lambda: int(os.getenv("EXECUTION_TIMEOUT", "30"))
    )

    # Sub-call settings
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


# Default configuration instance
default_config = Config()
