"""
Configuration loader — robust for local, CI and packaged environments.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml
from loguru import logger


def _resolve_env_vars(obj: Any) -> Any:
    """Resolve ${ENV_VAR} placeholders."""
    if isinstance(obj, str):
        if obj.startswith("${") and obj.endswith("}"):
            env_key = obj[2:-1]
            value = os.getenv(env_key)

            if value is None:
                logger.warning(f"Env var '{env_key}' not set")

            return value or obj

        return obj

    if isinstance(obj, dict):
        return {k: _resolve_env_vars(v) for k, v in obj.items()}

    if isinstance(obj, list):
        return [_resolve_env_vars(i) for i in obj]

    return obj


def _find_config_path(filename: str) -> Path:
    """
    Resolve config file path across:
    - local dev
    - monorepo (ml-quality-gate/)
    - CI
    - installed package
    - env override
    """

    candidates = [
        # 1. root/config
        Path.cwd() / "config" / filename,

        # 2. monorepo: ml-quality-gate/config  ✅ SEU CASO
        Path.cwd() / "ml-quality-gate" / "config" / filename,

        # 3. package install (src/ml_quality_gate/config)
        Path(__file__).resolve().parent.parent / "config" / filename,

        # 4. fallback mais profundo
        Path(__file__).resolve().parents[2] / "config" / filename,
    ]

    # 5. env override (prioridade alta)
    env_path = os.getenv("ML_CONFIG_PATH")
    if env_path:
        candidates.insert(0, Path(env_path) / filename)

    for path in candidates:
        if path.exists():
            logger.debug(f"✅ Config found: {path}")
            return path

    raise FileNotFoundError(
        f"❌ Config file not found: {filename}\nTried:\n"
        + "\n".join(str(p) for p in candidates)
    )


@lru_cache(maxsize=4)
def load_config(filename: str) -> dict[str, Any]:
    """Load YAML config with fallback resolution."""
    path = _find_config_path(filename)

    with open(path, encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    resolved = _resolve_env_vars(raw)

    logger.debug(f"Loaded config: {path}")

    return resolved


def get_thresholds() -> dict[str, Any]:
    return load_config("thresholds.yaml")


def get_model_config() -> dict[str, Any]:
    return load_config("model_config.yaml")


def get_threshold(
    section: str,
    key: str,
    sub_key: str = "minimum",
) -> float:
    """Typed accessor for threshold values."""

    thresholds = get_thresholds()

    try:
        value = thresholds[section][key]

        if isinstance(value, dict):
            return float(value[sub_key])

        return float(value)

    except KeyError as e:
        raise KeyError(
            f"Threshold not found: {section}.{key}.{sub_key}"
        ) from e