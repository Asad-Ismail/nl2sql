"""Configuration schemas for the unified LLM provider system."""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

import yaml
from pydantic import BaseModel, Field, field_validator


def _resolve_env_vars(value: str) -> str:
    """Resolve ${ENV_VAR} references in config values."""
    if not isinstance(value, str):
        return value
    pattern = r"\$\{([^}]+)\}"
    matches = re.findall(pattern, value)
    for match in matches:
        env_value = os.getenv(match, "")
        value = value.replace(f"${{{match}}}", env_value)
    return value


class RateLimitConfig(BaseModel):
    """Rate limiting configuration per provider."""

    calls_per_minute: int = Field(60, ge=1)
    burst_limit: int = Field(10, ge=1)
    retry_on_rate_limit: bool = True
    max_retries: int = Field(3, ge=0)
    retry_delay_seconds: float = Field(1.0, ge=0.1)


class DefaultParams(BaseModel):
    """Default generation parameters."""

    temperature: float = Field(0.0, ge=0.0, le=2.0)
    max_tokens: int = Field(512, ge=1)
    top_p: float = Field(1.0, ge=0.0, le=1.0)
    stop: Optional[List[str]] = None


class ProviderConfig(BaseModel):
    """Configuration for a single LLM provider."""

    type: Literal["openai", "anthropic"] = "openai"
    api_base: Optional[str] = None
    api_key: str = "dummy"
    timeout: float = Field(120.0, ge=1.0)
    rate_limit: RateLimitConfig = Field(default_factory=RateLimitConfig)
    default_params: DefaultParams = Field(default_factory=DefaultParams)
    extra_headers: Optional[Dict[str, str]] = None

    @field_validator("api_key", "api_base", mode="before")
    @classmethod
    def resolve_env_vars(cls, v: str) -> str:
        if v is None:
            return v
        return _resolve_env_vars(v)


class ModelConfig(BaseModel):
    """Configuration for a specific model using a provider."""

    provider: str
    model: str
    override_params: Optional[DefaultParams] = None


class LLMConfig(BaseModel):
    """Complete LLM system configuration."""

    providers: Dict[str, ProviderConfig]
    models: Dict[str, ModelConfig]

    def get_provider_for_model(self, model_name: str) -> tuple[ProviderConfig, str]:
        """
        Get provider config and model ID for a model name.

        Parameters
        ----------
        model_name : str
            Name of the model from the models dict

        Returns
        -------
        tuple[ProviderConfig, str]
            (provider_config, model_id)

        Raises
        ------
        ValueError
            If model or provider not found
        """
        if model_name not in self.models:
            raise ValueError(
                f"Unknown model: {model_name}. Available: {list(self.models.keys())}"
            )

        model_cfg = self.models[model_name]
        if model_cfg.provider not in self.providers:
            raise ValueError(
                f"Unknown provider: {model_cfg.provider}. "
                f"Available: {list(self.providers.keys())}"
            )

        provider_cfg = self.providers[model_cfg.provider].model_copy(deep=True)

        # Apply model-specific param overrides
        if model_cfg.override_params:
            base_params = provider_cfg.default_params.model_dump()
            override = model_cfg.override_params.model_dump(exclude_none=True)
            base_params.update(override)
            provider_cfg.default_params = DefaultParams(**base_params)

        return provider_cfg, model_cfg.model


def load_llm_config(config_path: Optional[str] = None) -> LLMConfig:
    """
    Load LLM configuration from YAML file.

    Parameters
    ----------
    config_path : str, optional
        Path to providers.yaml. If None, uses default location.

    Returns
    -------
    LLMConfig
        Loaded and validated configuration
    """
    if config_path is None:
        # Default location
        config_path = (
            Path(__file__).parent.parent / "optim" / "configs" / "llm" / "providers.yaml"
        )
    else:
        config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(f"LLM config not found: {config_path}")

    with open(config_path) as f:
        data = yaml.safe_load(f)

    return LLMConfig(**data)
