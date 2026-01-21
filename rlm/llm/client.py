"""Ollama-compatible LLM Client using OpenAI SDK"""

import logging
from dataclasses import dataclass
from typing import Optional
from openai import OpenAI

from ..config import Config, default_config

logger = logging.getLogger(__name__)


@dataclass
class LLMResponse:
    """Response from LLM including content and token usage"""
    content: str
    input_tokens: int = 0
    output_tokens: int = 0


class LLMClient:
    """Client for interacting with Ollama via OpenAI-compatible API"""

    def __init__(self, config: Config = None):
        self.config = config or default_config
        self.client = OpenAI(
            base_url=self.config.ollama_base_url,
            api_key="ollama"  # Ollama doesn't require a real API key
        )
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.call_count = 0

    def call_with_usage(
        self,
        messages: list[dict],
        model: str = None,
        temperature: float = None
    ) -> LLMResponse:
        """
        Call the LLM and return response with token usage.

        Args:
            messages: List of message dicts with 'role' and 'content'
            model: Model name (defaults to config.root_model)
            temperature: Sampling temperature (defaults to config.temperature)

        Returns:
            LLMResponse with content and token counts
        """
        model = model or self.config.root_model
        temperature = temperature if temperature is not None else self.config.temperature

        if self.config.debug:
            logger.debug(f"Calling LLM: model={model}, messages_count={len(messages)}")
            input_chars = sum(len(m.get("content", "")) for m in messages)
            logger.debug(f"Approximate input chars: {input_chars}")

        try:
            response = self.client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature
            )

            self.call_count += 1

            input_tokens = 0
            output_tokens = 0

            # Track token usage if available
            if response.usage:
                input_tokens = response.usage.prompt_tokens or 0
                output_tokens = response.usage.completion_tokens or 0
                self.total_input_tokens += input_tokens
                self.total_output_tokens += output_tokens
                if self.config.debug:
                    logger.debug(f"Tokens - input: {input_tokens}, output: {output_tokens}")

            content = response.choices[0].message.content
            return LLMResponse(
                content=content,
                input_tokens=input_tokens,
                output_tokens=output_tokens
            )

        except Exception as e:
            logger.error(f"LLM call failed: {e}")
            raise

    def call(
        self,
        messages: list[dict],
        model: str = None,
        temperature: float = None
    ) -> str:
        """
        Call the LLM with given messages (backwards compatible).

        Args:
            messages: List of message dicts with 'role' and 'content'
            model: Model name (defaults to config.root_model)
            temperature: Sampling temperature (defaults to config.temperature)

        Returns:
            The assistant's response text
        """
        response = self.call_with_usage(messages, model, temperature)
        return response.content

    def get_usage_stats(self) -> dict:
        """Get cumulative usage statistics"""
        return {
            "call_count": self.call_count,
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
            "total_tokens": self.total_input_tokens + self.total_output_tokens
        }

    def reset_stats(self):
        """Reset usage statistics"""
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.call_count = 0
