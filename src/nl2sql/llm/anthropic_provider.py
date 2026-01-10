"""Anthropic/Claude provider using the native Anthropic SDK."""

from __future__ import annotations

import logging
from typing import List, Optional

from .base import BaseLLMProvider, LLMMessage, LLMResponse
from .config import ProviderConfig

logger = logging.getLogger(__name__)


class AnthropicProvider(BaseLLMProvider):
    """
    Provider for Anthropic Claude API using native SDK.

    Handles Claude's different API format:
    - System prompt is a separate parameter, not a message
    - Response format differs from OpenAI
    """

    provider_type = "anthropic"

    def _setup_client(self) -> None:
        """Initialize the Anthropic client."""
        try:
            from anthropic import Anthropic
        except ImportError:
            raise ImportError(
                "anthropic package is required for AnthropicProvider. "
                "Install with: pip install anthropic"
            )

        client_kwargs = {
            "api_key": self.config.api_key,
            "timeout": self.config.timeout,
        }

        self.client = Anthropic(**client_kwargs)

    def generate(
        self,
        messages: List[LLMMessage],
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        stop: Optional[List[str]] = None,
        **kwargs,
    ) -> LLMResponse:
        """
        Generate a completion using Anthropic API.

        Parameters
        ----------
        messages : List[LLMMessage]
            List of messages in the conversation
        max_tokens : int, optional
            Maximum tokens to generate
        temperature : float, optional
            Sampling temperature
        stop : List[str], optional
            Stop sequences
        **kwargs
            Additional parameters

        Returns
        -------
        LLMResponse
            Standardized response
        """
        # Get parameters with defaults
        max_tokens = self._get_param(max_tokens, "max_tokens")
        temperature = self._get_param(temperature, "temperature")
        stop = self._get_param(stop, "stop")

        # Claude handles system prompt separately
        system_content = None
        anthropic_messages = []

        for m in messages:
            if m.role == "system":
                # Claude takes system as a separate parameter
                system_content = m.content
            else:
                anthropic_messages.append({"role": m.role, "content": m.content})

        # Build request kwargs
        request_kwargs = {
            "model": self.model,
            "messages": anthropic_messages,
            "max_tokens": max_tokens,
        }

        if system_content:
            request_kwargs["system"] = system_content

        # Only set temperature if non-zero (default 0 means deterministic)
        if temperature:
            request_kwargs["temperature"] = temperature

        if stop:
            request_kwargs["stop_sequences"] = stop

        # Add any additional kwargs
        top_p = kwargs.get("top_p") or self._get_param(None, "top_p")
        if top_p is not None and top_p < 1.0:
            request_kwargs["top_p"] = top_p

        # Make the API call with error handling
        try:
            response = self.client.messages.create(**request_kwargs)
        except Exception as e:
            logger.error(f"Anthropic API call failed: {e}")
            raise

        # Extract content (Claude returns a list of content blocks)
        content = ""
        if response.content:
            content = response.content[0].text

        # Build usage dict
        usage = None
        if response.usage:
            usage = {
                "prompt_tokens": response.usage.input_tokens,
                "completion_tokens": response.usage.output_tokens,
                "total_tokens": response.usage.input_tokens + response.usage.output_tokens,
            }

        return LLMResponse(
            content=content,
            usage=usage,
            model=response.model,
            finish_reason=response.stop_reason,
            raw_response=response,
        )
