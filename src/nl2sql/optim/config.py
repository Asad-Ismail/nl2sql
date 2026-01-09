"""Configuration schema and YAML loader for DSPy optimization."""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

import yaml
from pydantic import BaseModel, Field, field_validator, model_validator


class ExperimentConfig(BaseModel):
    """Experiment metadata configuration."""

    name: str = "dspy-nl2sql"
    description: str = ""
    seed: int = 42
    output_dir: str = "results/dspy_optimized"


class DatasetConfig(BaseModel):
    """Dataset source configuration."""

    name: str = "AsadIsmail/nl2sql-deduplicated"
    train_file: str = "spider_clean.jsonl"
    dev_file: str = "spider_dev_clean.jsonl"


class SplitsConfig(BaseModel):
    """Data split sizes."""

    train_size: int = Field(3000, ge=1)
    val_size: int = Field(2000, ge=1)


class PreprocessingConfig(BaseModel):
    """Data preprocessing options."""

    shuffle: bool = True
    seed: int = 42


class DataConfig(BaseModel):
    """Complete data configuration."""

    dataset: DatasetConfig = Field(default_factory=DatasetConfig)
    splits: SplitsConfig = Field(default_factory=SplitsConfig)
    preprocessing: PreprocessingConfig = Field(default_factory=PreprocessingConfig)


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


class ModelConfig(BaseModel):
    """LLM configuration."""

    name: str
    api_base: str
    api_key: str = "dummy"
    max_tokens: int = Field(512, ge=1)
    temperature: float = Field(0.0, ge=0.0, le=2.0)

    @field_validator("api_key", "api_base", mode="before")
    @classmethod
    def resolve_env_vars(cls, v: str) -> str:
        return _resolve_env_vars(v)


class TeacherConfig(BaseModel):
    """Teacher model configuration with enabled flag."""

    enabled: bool = False
    name: str = "moonshotai/kimi-k2-thinking"
    api_base: str = "https://integrate.api.nvidia.com/v1"
    api_key: str = "${NVIDIA_API_KEY}"
    temperature: float = Field(0.7, ge=0.0, le=2.0)

    @field_validator("api_key", "api_base", mode="before")
    @classmethod
    def resolve_env_vars(cls, v: str) -> str:
        return _resolve_env_vars(v)


class ModelsConfig(BaseModel):
    """All model configurations."""

    student: ModelConfig
    teacher: TeacherConfig = Field(default_factory=TeacherConfig)


class ModuleConfig(BaseModel):
    """DSPy module configuration."""

    use_cot: bool = True


class OptimizerConfig(BaseModel):
    """Optimizer configuration."""

    name: Literal[
        "BootstrapFewShot",
        "BootstrapFewShotWithRandomSearch",
        "MIPRO",
        "LabeledFewShot",
        "KNNFewShot",
        "COPRO",
    ] = "BootstrapFewShotWithRandomSearch"
    params: Dict[str, Any] = Field(
        default_factory=lambda: {
            "max_bootstrapped_demos": 2,
            "max_labeled_demos": 2,
        }
    )


class EvaluationConfig(BaseModel):
    """Evaluation configuration."""

    run_baseline: bool = True
    save_intermediate: bool = False


class LoggingConfig(BaseModel):
    """Logging configuration."""

    level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    enable_llm_logger: bool = True


class DSPyOptimizerConfig(BaseModel):
    """Root configuration model."""

    experiment: ExperimentConfig = Field(default_factory=ExperimentConfig)
    data: DataConfig = Field(default_factory=DataConfig)
    models: ModelsConfig
    module: ModuleConfig = Field(default_factory=ModuleConfig)
    optimizer: OptimizerConfig = Field(default_factory=OptimizerConfig)
    evaluation: EvaluationConfig = Field(default_factory=EvaluationConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)

    @model_validator(mode="after")
    def validate_teacher_if_enabled(self) -> "DSPyOptimizerConfig":
        """Warn if teacher is enabled but API key is missing."""
        if self.models.teacher.enabled:
            if not self.models.teacher.api_key:
                raise ValueError(
                    "Teacher is enabled but API key is not set. "
                    "Set NVIDIA_API_KEY environment variable."
                )
        return self


def _load_yaml(path: Path) -> Dict[str, Any]:
    """Load a YAML file and return as dictionary."""
    with open(path) as f:
        return yaml.safe_load(f) or {}


def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """Deep merge two dictionaries, with override taking precedence."""
    result = base.copy()
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def load_config(
    config_path: Optional[str] = None,
    base_config_path: Optional[str] = None,
    cli_overrides: Optional[Dict[str, Any]] = None,
) -> DSPyOptimizerConfig:
    """
    Load and validate DSPy optimizer configuration.

    Priority (highest to lowest):
    1. CLI overrides
    2. User config file
    3. Base/default config

    Parameters
    ----------
    config_path : str, optional
        Path to user config YAML
    base_config_path : str, optional
        Path to base config YAML (defaults to configs/default.yaml)
    cli_overrides : dict, optional
        Dictionary of CLI argument overrides

    Returns
    -------
    DSPyOptimizerConfig
        Validated configuration instance
    """
    config_dir = Path(__file__).parent / "configs"

    # Load base config
    if base_config_path:
        base_path = Path(base_config_path)
    else:
        base_path = config_dir / "default.yaml"

    config_dict = _load_yaml(base_path) if base_path.exists() else {}

    # Merge user config
    if config_path:
        user_path = Path(config_path)
        if user_path.exists():
            user_config = _load_yaml(user_path)
            config_dict = _deep_merge(config_dict, user_config)
        else:
            raise FileNotFoundError(f"Config file not found: {config_path}")

    # Apply CLI overrides
    if cli_overrides:
        config_dict = _deep_merge(config_dict, cli_overrides)

    return DSPyOptimizerConfig(**config_dict)


def cli_args_to_config_overrides(args) -> Dict[str, Any]:
    """
    Convert argparse namespace to config override dictionary.

    Only includes non-None values to avoid overriding config file values.
    """
    overrides: Dict[str, Any] = {}

    # Map CLI args to config paths
    mappings = {
        "student_model": ["models", "student", "name"],
        "api_base": ["models", "student", "api_base"],
        "api_key": ["models", "student", "api_key"],
        "max_tokens": ["models", "student", "max_tokens"],
        "optimizer": ["optimizer", "name"],
        "use_cot": ["module", "use_cot"],
        "use_teacher": ["models", "teacher", "enabled"],
        "teacher_model": ["models", "teacher", "name"],
        "output_dir": ["experiment", "output_dir"],
        "train_size": ["data", "splits", "train_size"],
        "val_size": ["data", "splits", "val_size"],
    }

    for arg_name, config_path in mappings.items():
        value = getattr(args, arg_name, None)
        if value is not None:
            current = overrides
            for key in config_path[:-1]:
                current = current.setdefault(key, {})
            current[config_path[-1]] = value

    return overrides
