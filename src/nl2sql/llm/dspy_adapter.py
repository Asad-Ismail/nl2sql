"""DSPy integration adapter for the unified LLM provider system."""

from __future__ import annotations

import logging
from typing import Optional, Tuple

import dspy

from .config import load_llm_config, LLMConfig
from .factory import LLMFactory

logger = logging.getLogger(__name__)


def configure_dspy_from_config(
    student_model: str,
    teacher_model: Optional[str] = None,
    config_path: Optional[str] = None,
) -> Tuple[dspy.LM, Optional[dspy.LM]]:
    """
    Configure DSPy with student (and optionally teacher) LMs from unified config.

    This function creates dspy.LM instances configured with the settings
    from providers.yaml, making it easy to switch between providers.

    Parameters
    ----------
    student_model : str
        Model name for student (from providers.yaml)
    teacher_model : str, optional
        Model name for teacher (from providers.yaml)
    config_path : str, optional
        Path to providers.yaml

    Returns
    -------
    Tuple[dspy.LM, Optional[dspy.LM]]
        (student_lm, teacher_lm) where teacher_lm may be None

    Examples
    --------
    >>> student_lm, teacher_lm = configure_dspy_from_config(
    ...     student_model="codellama_7b",
    ...     teacher_model="llama_70b_nvidia"
    ... )
    >>> dspy.configure(lm=student_lm)
    """
    config = load_llm_config(config_path)

    # Create student LM
    student_lm = _create_dspy_lm(config, student_model)
    dspy.configure(lm=student_lm)

    # Create teacher LM if specified
    teacher_lm = None
    if teacher_model:
        teacher_lm = _create_dspy_lm(config, teacher_model)

    return student_lm, teacher_lm


def _create_dspy_lm(config: LLMConfig, model_name: str) -> dspy.LM:
    """
    Create a dspy.LM instance from unified config.

    Parameters
    ----------
    config : LLMConfig
        Loaded LLM configuration
    model_name : str
        Name of the model from providers.yaml

    Returns
    -------
    dspy.LM
        Configured DSPy LM instance
    """
    provider_cfg, model_id = config.get_provider_for_model(model_name)

    # DSPy expects model format like "openai/model-name" for custom endpoints
    if provider_cfg.type == "openai":
        if provider_cfg.api_base:
            # Custom endpoint (vLLM, NVIDIA, OpenRouter)
            model_str = f"openai/{model_id}"
        else:
            # Direct OpenAI
            model_str = model_id
    elif provider_cfg.type == "anthropic":
        model_str = f"anthropic/{model_id}"
    else:
        model_str = model_id

    # Build LM kwargs
    lm_kwargs = {
        "model": model_str,
        "max_tokens": provider_cfg.default_params.max_tokens,
        "temperature": provider_cfg.default_params.temperature,
    }

    if provider_cfg.api_base:
        lm_kwargs["api_base"] = provider_cfg.api_base

    if provider_cfg.api_key:
        lm_kwargs["api_key"] = provider_cfg.api_key

    return dspy.LM(**lm_kwargs)


def get_dspy_lm(
    model_name: str,
    config_path: Optional[str] = None,
) -> dspy.LM:
    """
    Get a single dspy.LM instance for a model.

    Parameters
    ----------
    model_name : str
        Model name from providers.yaml
    config_path : str, optional
        Path to providers.yaml

    Returns
    -------
    dspy.LM
        Configured DSPy LM instance
    """
    config = load_llm_config(config_path)
    return _create_dspy_lm(config, model_name)
