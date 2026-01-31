"""Shared configuration utilities for NL2SQL."""

from __future__ import annotations

import os
import re


def resolve_env_vars(value: str) -> str:
    """
    Resolve ${ENV_VAR} references in configuration values.

    Parameters
    ----------
    value : str
        String potentially containing environment variable references

    Returns
    -------
    str
        String with environment variables replaced by their values
    """
    if not isinstance(value, str):
        return value
    pattern = r"\$\{([^}]+)\}"
    matches = re.findall(pattern, value)
    for match in matches:
        env_value = os.getenv(match, "")
        value = value.replace(f"${{{match}}}", env_value)
    return value
