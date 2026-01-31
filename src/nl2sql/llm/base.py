"""Abstract base class for LLM providers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

from pydantic import BaseModel

from .config import ProviderConfig


class LLMMessage(BaseModel):
    """Standardized message format for LLM interactions."""

    role: str  # "user", "assistant", "system"
    content: str


class LLMResponse(BaseModel):
    """Standardized response format from LLM providers."""

    content: str
    usage: Optional[Dict[str, int]] = None
    model: str
    finish_reason: Optional[str] = None
    raw_response: Optional[Any] = None  # Original response for debugging


class BaseLLMProvider(ABC):
    """
    Abstract base class for LLM providers.

    All provider implementations should inherit from this class
    and implement the required methods.
    """

    provider_type: str = ""

    def __init__(self, config: ProviderConfig, model: str):
        """
        Initialize the provider.

        Parameters
        ----------
        config : ProviderConfig
            Provider configuration
        model : str
            Model identifier to use
        """
        self.config = config
        self.model = model
        self._setup_client()

    @abstractmethod
    def _setup_client(self) -> None:
        """Initialize the underlying API client."""
        pass

    @abstractmethod
    def generate(
        self,
        messages: List[LLMMessage],
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        stop: Optional[List[str]] = None,
        **kwargs,
    ) -> LLMResponse:
        """
        Generate a completion from messages.

        Parameters
        ----------
        messages : List[LLMMessage]
            List of messages in the conversation
        max_tokens : int, optional
            Maximum tokens to generate. Uses config default if not specified.
        temperature : float, optional
            Sampling temperature. Uses config default if not specified.
        stop : List[str], optional
            Stop sequences. Uses config default if not specified.
        **kwargs
            Additional provider-specific parameters

        Returns
        -------
        LLMResponse
            Standardized response
        """
        pass

    def generate_text(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        stop: Optional[List[str]] = None,
        **kwargs,
    ) -> str:
        """
        Convenience method for simple text generation.

        Parameters
        ----------
        prompt : str
            User prompt
        system_prompt : str, optional
            System prompt to set context
        max_tokens : int, optional
            Maximum tokens to generate
        temperature : float, optional
            Sampling temperature
        stop : List[str], optional
            Stop sequences
        **kwargs
            Additional provider-specific parameters

        Returns
        -------
        str
            Generated text content
        """
        messages = []
        if system_prompt:
            messages.append(LLMMessage(role="system", content=system_prompt))
        messages.append(LLMMessage(role="user", content=prompt))

        response = self.generate(
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            stop=stop,
            **kwargs,
        )
        return response.content

    def _get_param(self, value: Optional[Any], param_name: str) -> Any:
        """
        Get parameter value, falling back to config default.

        Parameters
        ----------
        value : Any, optional
            Provided value (may be None)
        param_name : str
            Name of parameter in default_params

        Returns
        -------
        Any
            Value to use
        """
        if value is not None:
            return value
        return getattr(self.config.default_params, param_name, None)
