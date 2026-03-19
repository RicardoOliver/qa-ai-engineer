"""
Model Validator
Validates classification and regression model metrics against configured thresholds.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np
from loguru import logger
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)

from ml_quality_gate.utils.config_loader import get_thresholds


# ─────────────────────────────────────────────
# Result Models
# ─────────────────────────────────────────────

@dataclass
class MetricResult:
    name: str
    value: float
    threshold: float
    passed: bool
    critical: bool = False
    message: str = ""

    def __post_init__(self) -> None:
        direction = ">=" if self.name not in ("mae", "rmse") else "<="
        status = (
            "✅ PASS"
            if self.passed
            else "🚨 CRITICAL"
            if self.critical
            else "❌ FAIL"
        )

        self.message = (
            f"{status} | {self.name}: {self.value:.4f} "
            f"(threshold {direction} {self.threshold:.4f})"
        )


@dataclass
class ValidationReport:
    model_name: str
    task: Literal["classification", "regression"]
    results: list[MetricResult] = field(default_factory=list)
    passed: bool = True
    has_critical_failure: bool = False

    def add_result(self, result: MetricResult) -> None:
        self.results.append(result)

        if not result.passed:
            self.passed = False

        if result.critical:
            self.has_critical_failure = True

    def summary(self) -> dict[str, Any]:
        return {
            "model_name": self.model_name,
            "task": self.task,
            "passed": self.passed,
            "has_critical_failure": self.has_critical_failure,
            "total_checks": len(self.results),
            "passed_checks": sum(1 for r in self.results if r.passed),
            "failed_checks": sum(1 for r in self.results if not r.passed),
            "metrics": {
                r.name: {"value": r.value, "passed": r.passed}
                for r in self.results
            },
        }

    def print_report(self) -> None:
        logger.info(f"\n{'=' * 60}")
        logger.info(f"MODEL VALIDATION REPORT — {self.model_name.upper()}")
        logger.info(f"{'=' * 60}")

        for result in self.results:
            if result.passed:
                logger.success(result.message)
            else:
                logger.error(result.message)

        overall = "PASSED" if self.passed else "FAILED"
        logger.info(f"\nOverall: {overall}")
        logger.info(f"{'=' * 60}\n")


# ─────────────────────────────────────────────
# Validator
# ─────────────────────────────────────────────

class ModelValidator:
    def __init__(self, model_name: str = "model") -> None:
        self.model_name = model_name
        self._thresholds: dict[str, Any] = get_thresholds()["model"]

    def _check_metric(
        self,
        name: str,
        value: float,
        threshold_key: str,
        higher_is_better: bool = True,
    ) -> MetricResult:
        cfg = self._thresholds.get(threshold_key, {})

        if isinstance(cfg, dict):
            minimum = float(cfg.get("minimum", 0.0))
            critical_val = float(cfg.get("critical", minimum * 0.95))
            maximum = float(cfg.get("maximum", float("inf")))
        else:
            minimum = float(cfg)
            critical_val = minimum * 0.95
            maximum = float("inf")

        if higher_is_better:
            passed = value >= minimum
            critical = not passed and value < critical_val
            threshold = minimum
        else:
            passed = value <= maximum
            critical = not passed and value > maximum * 1.2
            threshold = maximum

        return MetricResult(
            name=name,
            value=value,
            threshold=threshold,
            passed=passed,
            critical=critical,
        )

    # ─────────────────────────────────────────────
    # Classification
    # ─────────────────────────────────────────────

    def validate_classification(
        self,
        y_true: np.ndarray | list[Any],
        y_pred: np.ndarray | list[Any],
        y_prob: np.ndarray | list[Any] | None = None,
        average: str = "binary",
    ) -> ValidationReport:

        report = ValidationReport(
            model_name=self.model_name,
            task="classification",
        )

        y_true_np = np.asarray(y_true)
        y_pred_np = np.asarray(y_pred)

        metrics: list[tuple[str, float, str]] = [
            ("accuracy", accuracy_score(y_true_np, y_pred_np), "accuracy"),
            (
                "precision",
                precision_score(y_true_np, y_pred_np, average=average, zero_division=0),
                "precision",
            ),
            (
                "recall",
                recall_score(y_true_np, y_pred_np, average=average, zero_division=0),
                "recall",
            ),
            (
                "f1_score",
                f1_score(y_true_np, y_pred_np, average=average, zero_division=0),
                "f1_score",
            ),
        ]

        for name, value, key in metrics:
            report.add_result(self._check_metric(name, value, key))

        if y_prob is not None:
            y_prob_np = np.asarray(y_prob)
            try:
                auc = roc_auc_score(y_true_np, y_prob_np)
                report.add_result(self._check_metric("roc_auc", auc, "roc_auc"))
            except ValueError as e:
                logger.warning(f"ROC-AUC skipped: {e}")

        report.print_report()
        return report

    # ─────────────────────────────────────────────
    # Regression
    # ─────────────────────────────────────────────

    def validate_regression(
        self,
        y_true: np.ndarray | list[Any],
        y_pred: np.ndarray | list[Any],
    ) -> ValidationReport:

        report = ValidationReport(
            model_name=self.model_name,
            task="regression",
        )

        y_true_np = np.asarray(y_true)
        y_pred_np = np.asarray(y_pred)

        r2 = r2_score(y_true_np, y_pred_np)
        mae = mean_absolute_error(y_true_np, y_pred_np)
        rmse = float(np.sqrt(mean_squared_error(y_true_np, y_pred_np)))

        report.add_result(self._check_metric("r2_score", r2, "r2_score"))
        report.add_result(self._check_metric("mae", mae, "mae", higher_is_better=False))
        report.add_result(self._check_metric("rmse", rmse, "rmse", higher_is_better=False))

        report.print_report()
        return report