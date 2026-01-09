"""Provider factory and registry for the unified LLM system."""

from __future__ import annotations

import logging
from functools import wraps
from pathlib import Path
from typing import Callable, Dict, Optional, Type, TypeVar

from .base import BaseLLMProvider, LLMResponse
from .config import LLMConfig, ProviderConfig, load_llm_config
from .rate_limiter import RateLimiterRegistry, TokenBucketRateLimiter

logger = logging.getLogger(__name__)
T = TypeVar("T")


class ProviderRegistry:
    """Registry of available provider implementations."""

    _registry: Dict[str, Type[BaseLLMProvider]] = {}

    @classmethod
    def register(cls, provider_type: str, provider_class: Type[BaseLLMProvider]) -> None:
        """
        Register a provider implementation.

        Parameters
        ----------
        provider_type : str
            Type identifier (e.g., "openai", "anthropic")
        provider_class : Type[BaseLLMProvider]
            Provider class to register
        """
        cls._registry[provider_type] = provider_class

    @classmethod
    def get(cls, provider_type: str) -> Type[BaseLLMProvider]:
        """
        Get a provider class by type.

        Parameters
        ----------
        provider_type : str
            Type identifier

        Returns
        -------
        Type[BaseLLMProvider]
            Provider class

        Raises
        ------
        ValueError
            If provider type not found
        """
        if provider_type not in cls._registry:
            raise ValueError(
                f"Unknown provider type: {provider_type}. "
                f"Available: {list(cls._registry.keys())}"
            )
        return cls._registry[provider_type]

    @classmethod
    def list_types(cls) -> list[str]:
        """List all registered provider types."""
        return list(cls._registry.keys())


# Register built-in providers
def _register_providers():
    from .openai_compat import OpenAICompatibleProvider

    ProviderRegistry.register("openai", OpenAICompatibleProvider)

    # Anthropic is optional
    try:
        from .anthropic_provider import AnthropicProvider

        ProviderRegistry.register("anthropic", AnthropicProvider)
    except ImportError:
        logger.debug("Anthropic provider not available (anthropic package not installed)")


_register_providers()


class LLMFactory:
    """
    Factory for creating LLM provider instances.

    Handles:
    - Loading configuration
    - Creating provider instances
    - Applying rate limiting
    - Caching instances
    """

    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize the factory.

        Parameters
        ----------
        config_path : str, optional
            Path to providers.yaml. Uses default location if not specified.
        """
        self.config = load_llm_config(config_path)
        self._providers: Dict[str, BaseLLMProvider] = {}

    def get_provider(
        self,
        model_name: str,
        apply_rate_limit: bool = True,
    ) -> BaseLLMProvider:
        """
        Get a provider instance for a model.

        Parameters
        ----------
        model_name : str
            Name of the model from providers.yaml
        apply_rate_limit : bool
            Whether to apply rate limiting (default True)

        Returns
        -------
        BaseLLMProvider
            Provider instance with rate limiting applied
        """
        cache_key = f"{model_name}:{apply_rate_limit}"

        if cache_key in self._providers:
            return self._providers[cache_key]

        # Get config for this model
        provider_cfg, model_id = self.config.get_provider_for_model(model_name)

        # Get the provider class
        provider_class = ProviderRegistry.get(provider_cfg.type)

        # Create the provider instance
        provider = provider_class(provider_cfg, model_id)

        # Apply rate limiting if requested
        if apply_rate_limit:
            provider_name = self.config.models[model_name].provider
            rate_limiter = RateLimiterRegistry.get_or_create(
                provider_name, provider_cfg.rate_limit
            )
            provider = _wrap_with_rate_limit(provider, rate_limiter)

        self._providers[cache_key] = provider
        return provider

    def list_models(self) -> list[str]:
        """List all available model names."""
        return list(self.config.models.keys())

    def list_providers(self) -> list[str]:
        """List all available provider names."""
        return list(self.config.providers.keys())


def _wrap_with_rate_limit(
    provider: BaseLLMProvider,
    rate_limiter: TokenBucketRateLimiter,
) -> BaseLLMProvider:
    """
    Wrap a provider's generate method with rate limiting.

    Parameters
    ----------
    provider : BaseLLMProvider
        Provider to wrap
    rate_limiter : TokenBucketRateLimiter
        Rate limiter to apply

    Returns
    -------
    BaseLLMProvider
        Provider with rate-limited generate method
    """
    original_generate = provider.generate

    @wraps(original_generate)
    def rate_limited_generate(*args, **kwargs) -> LLMResponse:
        rate_limiter.acquire()
        return original_generate(*args, **kwargs)

    provider.generate = rate_limited_generate
    return provider


# Global factory instance for convenience
_default_factory: Optional[LLMFactory] = None


def get_llm(
    model_name: str,
    config_path: Optional[str] = None,
    apply_rate_limit: bool = True,
) -> BaseLLMProvider:
    """
    Get an LLM provider by model name.

    This is the main entry point for using the unified LLM system.

    Parameters
    ----------
    model_name : str
        Name of the model from providers.yaml
    config_path : str, optional
        Path to providers.yaml. Uses default if not specified.
    apply_rate_limit : bool
        Whether to apply rate limiting (default True)

    Returns
    -------
    BaseLLMProvider
        Provider instance ready to use

    Examples
    --------
    >>> llm = get_llm("codellama_7b")
    >>> response = llm.generate_text("Convert to SQL: show all users")

    >>> llm = get_llm("claude_sonnet")
    >>> response = llm.generate_text("Hello", system_prompt="You are a SQL expert")
    """
    global _default_factory

    if _default_factory is None or config_path is not None:
        _default_factory = LLMFactory(config_path)

    return _default_factory.get_provider(model_name, apply_rate_limit)


def reset_factory() -> None:
    """Reset the global factory. Useful for testing."""
    global _default_factory
    _default_factory = None
    RateLimiterRegistry.clear()
