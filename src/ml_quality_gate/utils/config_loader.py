"""Configuration loader — reads YAML configs and exposes typed settings."""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml
from loguru import logger


CONFIG_DIR = Path(__file__).parents[3] / "config"


def _resolve_env_vars(obj: Any) -> Any:
    """Recursively resolve ${ENV_VAR} placeholders in config values."""
    if isinstance(obj, str):
        if obj.startswith("${") and obj.endswith("}"):
            env_key = obj[2:-1]
            value = os.getenv(env_key)
            if value is None:
                logger.warning(f"Env var '{env_key}' not set — using placeholder.")
            return value or obj
        return obj
    if isinstance(obj, dict):
        return {k: _resolve_env_vars(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_resolve_env_vars(i) for i in obj]
    return obj


@lru_cache(maxsize=4)
def load_config(filename: str) -> dict[str, Any]:
    """Load and cache a YAML config file."""
    path = CONFIG_DIR / filename
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with open(path) as f:
        raw = yaml.safe_load(f)
    resolved = _resolve_env_vars(raw)
    logger.debug(f"Loaded config: {filename}")
    return resolved


def get_thresholds() -> dict[str, Any]:
    return load_config("thresholds.yaml")


def get_model_config() -> dict[str, Any]:
    return load_config("model_config.yaml")


def get_threshold(section: str, key: str, sub_key: str = "minimum") -> float:
    """Convenience accessor for threshold values."""
    thresholds = get_thresholds()
    try:
        value = thresholds[section][key]
        if isinstance(value, dict):
            return float(value[sub_key])
        return float(value)
    except KeyError as e:
        raise KeyError(f"Threshold not found: {section}.{key}.{sub_key}") from e
