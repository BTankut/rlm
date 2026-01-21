"""Async Ollama-compatible LLM Client using OpenAI SDK"""

import asyncio
import logging
from dataclasses import dataclass
from typing import Optional

from openai import AsyncOpenAI

from ..config import Config, default_config

logger = logging.getLogger(__name__)


@dataclass
class AsyncLLMResponse:
    """Response from async LLM including content and token usage"""
    content: str
    input_tokens: int = 0
    output_tokens: int = 0


class AsyncLLMClient:
    """Async client for interacting with Ollama via OpenAI-compatible API"""

    def __init__(self, config: Config = None):
        self.config = config or default_config
        self.client = AsyncOpenAI(
            base_url=self.config.ollama_base_url,
            api_key="ollama"  # Ollama doesn't require a real API key
        )
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.call_count = 0
        self._lock = asyncio.Lock()

    async def call_with_usage(
        self,
        messages: list[dict],
        model: str = None,
        temperature: float = None
    ) -> AsyncLLMResponse:
        """
        Async call to the LLM and return response with token usage.

        Args:
            messages: List of message dicts with 'role' and 'content'
            model: Model name (defaults to config.root_model)
            temperature: Sampling temperature (defaults to config.temperature)

        Returns:
            AsyncLLMResponse with content and token counts
        """
        model = model or self.config.root_model
        temperature = temperature if temperature is not None else self.config.temperature

        if self.config.debug:
            logger.debug(f"Async calling LLM: model={model}, messages_count={len(messages)}")

        try:
            response = await self.client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature
            )

            async with self._lock:
                self.call_count += 1

                input_tokens = 0
                output_tokens = 0

                if response.usage:
                    input_tokens = response.usage.prompt_tokens or 0
                    output_tokens = response.usage.completion_tokens or 0
                    self.total_input_tokens += input_tokens
                    self.total_output_tokens += output_tokens

            content = response.choices[0].message.content
            return AsyncLLMResponse(
                content=content,
                input_tokens=input_tokens,
                output_tokens=output_tokens
            )

        except Exception as e:
            logger.error(f"Async LLM call failed: {e}")
            raise

    async def call(
        self,
        messages: list[dict],
        model: str = None,
        temperature: float = None
    ) -> str:
        """
        Async call the LLM with given messages (backwards compatible).

        Args:
            messages: List of message dicts with 'role' and 'content'
            model: Model name (defaults to config.root_model)
            temperature: Sampling temperature (defaults to config.temperature)

        Returns:
            The assistant's response text
        """
        response = await self.call_with_usage(messages, model, temperature)
        return response.content

    async def call_batch(
        self,
        batch: list[list[dict]],
        model: str = None,
        temperature: float = None,
        max_concurrent: int = 5
    ) -> list[AsyncLLMResponse]:
        """
        Make multiple LLM calls concurrently.

        Args:
            batch: List of message lists to process
            model: Model name (defaults to config.root_model)
            temperature: Sampling temperature
            max_concurrent: Maximum concurrent calls

        Returns:
            List of AsyncLLMResponse objects
        """
        semaphore = asyncio.Semaphore(max_concurrent)

        async def bounded_call(messages):
            async with semaphore:
                return await self.call_with_usage(messages, model, temperature)

        tasks = [bounded_call(messages) for messages in batch]
        return await asyncio.gather(*tasks)

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
