"""OpenAI-compatible provider for vLLM, NVIDIA NIM, OpenRouter, and OpenAI."""

from __future__ import annotations

import logging
from typing import List, Optional

from openai import OpenAI

from .base import BaseLLMProvider, LLMMessage, LLMResponse
from .config import ProviderConfig

logger = logging.getLogger(__name__)


class OpenAICompatibleProvider(BaseLLMProvider):
    """
    Provider for OpenAI-compatible APIs.

    Supports:
    - OpenAI API
    - Local vLLM servers
    - NVIDIA NIM API
    - OpenRouter
    - Any other OpenAI-compatible endpoint
    """

    provider_type = "openai"

    def _setup_client(self) -> None:
        """Initialize the OpenAI client."""
        client_kwargs = {
            "api_key": self.config.api_key,
            "timeout": self.config.timeout,
        }

        if self.config.api_base:
            client_kwargs["base_url"] = self.config.api_base

        # Add extra headers if specified (e.g., for OpenRouter)
        if self.config.extra_headers:
            client_kwargs["default_headers"] = self.config.extra_headers

        self.client = OpenAI(**client_kwargs)

    def generate(
        self,
        messages: List[LLMMessage],
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        stop: Optional[List[str]] = None,
        **kwargs,
    ) -> LLMResponse:
        """
        Generate a completion using OpenAI-compatible API.

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
            Additional parameters (top_p, etc.)

        Returns
        -------
        LLMResponse
            Standardized response
        """
        # Get parameters with defaults
        max_tokens = self._get_param(max_tokens, "max_tokens")
        temperature = self._get_param(temperature, "temperature")
        stop = self._get_param(stop, "stop")

        # Convert messages to OpenAI format
        openai_messages = [
            {"role": m.role, "content": m.content} for m in messages
        ]

        # Build request kwargs
        request_kwargs = {
            "model": self.model,
            "messages": openai_messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }

        if stop:
            request_kwargs["stop"] = stop

        # Add any additional kwargs (top_p, etc.)
        top_p = kwargs.get("top_p") or self._get_param(None, "top_p")
        if top_p and top_p < 1.0:
            request_kwargs["top_p"] = top_p

        # Make the API call
        response = self.client.chat.completions.create(**request_kwargs)

        # Extract content
        content = response.choices[0].message.content or ""

        # Build usage dict
        usage = None
        if response.usage:
            usage = {
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens,
            }

        return LLMResponse(
            content=content,
            usage=usage,
            model=response.model,
            finish_reason=response.choices[0].finish_reason,
            raw_response=response,
        )
