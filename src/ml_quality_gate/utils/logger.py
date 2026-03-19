"""Centralized logging configuration using loguru."""

from __future__ import annotations

import sys
from pathlib import Path

from loguru import logger


def setup_logger(level: str = "INFO", log_file: str | None = None) -> None:
    """Configure loguru with structured output for QA reporting."""
    logger.remove()

    # Console — human-readable
    logger.add(
        sys.stderr,
        level=level,
        format=(
            "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
            "<level>{message}</level>"
        ),
        colorize=True,
    )

    # File — JSON structured for ingestion (Datadog, Splunk, etc.)
    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        logger.add(
            log_file,
            level="DEBUG",
            format="{time} | {level} | {name} | {message}",
            rotation="10 MB",
            retention="7 days",
            serialize=True,  # JSON output
        )


setup_logger(log_file="reports/logs/ml_quality_gate.log")
