"""Unified configuration for all optimizers."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import yaml
from pydantic import BaseModel, Field
from dotenv import load_dotenv


class OptimizerDataConfig(BaseModel):
    """Data configuration shared by all optimizers."""

    dataset_name: str = "AsadIsmail/nl2sql-deduplicated"
    train_file: str = "spider_clean.jsonl"
    dev_file: str = "spider_dev_clean.jsonl"
    train_size: int = 400
    val_size: int = 500
    shuffle_seed: int = 42


class OptimizerModelConfig(BaseModel):
    """Model configuration."""

    student_model: str = "codellama_7b"
    teacher_model: Optional[str] = None


class OptimizerExperimentConfig(BaseModel):
    """Experiment configuration."""

    output_dir: str
    name: str
    seed: int = 42


class OptimizerTrainingConfig(BaseModel):
    """Training configuration (for TextGrad)."""

    epochs: int = 1
    batch_size: int = 3


class OptimizerEvaluationConfig(BaseModel):
    """Evaluation configuration."""

    fitness_samples: int = 40


class TextGradOptimizerConfig(BaseModel):
    """Configuration for TextGrad optimizer."""

    experiment: OptimizerExperimentConfig
    data: OptimizerDataConfig = Field(default_factory=OptimizerDataConfig)
    models: OptimizerModelConfig = Field(default_factory=OptimizerModelConfig)
    training: OptimizerTrainingConfig = Field(default_factory=OptimizerTrainingConfig)
    evaluation: OptimizerEvaluationConfig = Field(default_factory=OptimizerEvaluationConfig)


def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """Deep merge two dictionaries, with override taking precedence."""

    result = base.copy()
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def load_optimizer_config(
    config_path: str,
    cli_overrides: Dict[str, Any] = None,
) -> TextGradOptimizerConfig:
    """
    Load optimizer configuration from YAML with CLI overrides.

    Priority:
    1. CLI overrides (highest)
    2. User config file
    3. Default values

    Parameters
    ----------
    config_path : str
        Path to YAML config file
    cli_overrides : dict, optional
        Dictionary of CLI argument overrides

    Returns
    -------
    TextGradOptimizerConfig
        Validated configuration instance
    """
    # Load .env from current dir and project root
    load_dotenv()  # Current directory
    load_dotenv(Path(__file__).parent.parent / ".env")  # Project root

    # Load YAML
    config_file = Path(config_path)
    if not config_file.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_file) as f:
        config_dict = yaml.safe_load(f)

    if config_dict is None:
        config_dict = {}

    # Apply CLI overrides (deep merge)
    if cli_overrides:
        config_dict = _deep_merge(config_dict, cli_overrides)

    return TextGradOptimizerConfig(**config_dict)


def cli_args_to_dict(args) -> Dict[str, Any]:
    """
    Convert argparse namespace to nested dict for config overrides.

    Only includes non-None values to avoid overriding config file values.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed command-line arguments

    Returns
    -------
    dict
        Nested dictionary for deep merge
    """
    overrides: Dict[str, Any] = {}

    # Map CLI args to config paths
    mappings = {
        "data_train_size": ["data", "train_size"],
        "data_val_size": ["data", "val_size"],
        "data_shuffle_seed": ["data", "shuffle_seed"],
        "student_model": ["models", "student_model"],
        "teacher_model": ["models", "teacher_model"],
        "train_epochs": ["training", "epochs"],
        "train_batch_size": ["training", "batch_size"],
        "eval_fitness_samples": ["evaluation", "fitness_samples"],
        "output_dir": ["experiment", "output_dir"],
        "exp_name": ["experiment", "name"],
        "exp_seed": ["experiment", "seed"],
    }

    for arg_name, config_path in mappings.items():
        value = getattr(args, arg_name, None)
        if value is not None:
            current = overrides
            for key in config_path[:-1]:
                current = current.setdefault(key, {})
            current[config_path[-1]] = value

    return overrides
