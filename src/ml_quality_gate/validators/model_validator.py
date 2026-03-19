"""
Model Validator
Validates classification and regression model metrics against configured thresholds.
Supports: accuracy, precision, recall, F1, ROC-AUC, R2, MAE, RMSE.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np
import pandas as pd
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

from ml_quality_gate.utils.config_loader import get_threshold, get_thresholds


@dataclass
class MetricResult:
    name: str
    value: float
    threshold: float
    passed: bool
    critical: bool = False
    message: str = ""

    def __post_init__(self) -> None:
        direction = ">=" if not self.name.startswith(("mae", "rmse")) else "<="
        status = "✅ PASS" if self.passed else ("🚨 CRITICAL" if self.critical else "❌ FAIL")
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
            "metrics": {r.name: {"value": r.value, "passed": r.passed} for r in self.results},
        }

    def print_report(self) -> None:
        logger.info(f"\n{'='*60}")
        logger.info(f"  MODEL VALIDATION REPORT — {self.model_name.upper()}")
        logger.info(f"{'='*60}")
        for result in self.results:
            if result.passed:
                logger.success(result.message)
            else:
                logger.error(result.message)
        overall = "✅ PASSED" if self.passed else "❌ FAILED"
        logger.info(f"\nOverall: {overall}")
        logger.info(f"{'='*60}\n")


class ModelValidator:
    """
    Validates ML model metrics against configured thresholds.

    Usage:
        validator = ModelValidator(model_name="credit_risk_v2")
        report = validator.validate_classification(y_true, y_pred, y_prob)
        assert report.passed, f"Quality gate failed: {report.summary()}"
    """

    def __init__(self, model_name: str = "model") -> None:
        self.model_name = model_name
        self._thresholds = get_thresholds()["model"]

    def _check_metric(
        self,
        name: str,
        value: float,
        threshold_key: str,
        higher_is_better: bool = True,
    ) -> MetricResult:
        cfg = self._thresholds.get(threshold_key, {})
        minimum = float(cfg.get("minimum", cfg) if isinstance(cfg, dict) else cfg)
        critical_val = float(cfg.get("critical", minimum * 0.95)) if isinstance(cfg, dict) else minimum * 0.95
        maximum = float(cfg.get("maximum", float("inf"))) if isinstance(cfg, dict) else float("inf")

        if higher_is_better:
            passed = value >= minimum
            critical = not passed and value < critical_val
        else:
            passed = value <= maximum
            critical = not passed and value > maximum * 1.2

        return MetricResult(
            name=name,
            value=value,
            threshold=minimum if higher_is_better else maximum,
            passed=passed,
            critical=critical,
        )

    def validate_classification(
        self,
        y_true: np.ndarray | list,
        y_pred: np.ndarray | list,
        y_prob: np.ndarray | list | None = None,
        average: str = "binary",
    ) -> ValidationReport:
        """Run full classification validation suite."""
        report = ValidationReport(model_name=self.model_name, task="classification")

        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)

        metrics = [
            ("accuracy", accuracy_score(y_true, y_pred), "accuracy"),
            ("precision", precision_score(y_true, y_pred, average=average, zero_division=0), "precision"),
            ("recall", recall_score(y_true, y_pred, average=average, zero_division=0), "recall"),
            ("f1_score", f1_score(y_true, y_pred, average=average, zero_division=0), "f1_score"),
        ]

        for name, value, key in metrics:
            report.add_result(self._check_metric(name, value, key))
            logger.debug(f"Metric computed: {name}={value:.4f}")

        if y_prob is not None:
            y_prob = np.asarray(y_prob)
            try:
                auc = roc_auc_score(y_true, y_prob)
                report.add_result(self._check_metric("roc_auc", auc, "roc_auc"))
            except ValueError as e:
                logger.warning(f"Could not compute ROC-AUC: {e}")

        report.print_report()
        return report

    def validate_regression(
        self,
        y_true: np.ndarray | list,
        y_pred: np.ndarray | list,
    ) -> ValidationReport:
        """Run full regression validation suite."""
        report = ValidationReport(model_name=self.model_name, task="regression")

        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)

        r2 = r2_score(y_true, y_pred)
        mae = mean_absolute_error(y_true, y_pred)
        rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))

        report.add_result(self._check_metric("r2_score", r2, "r2_score"))
        report.add_result(self._check_metric("mae", mae, "mae", higher_is_better=False))
        report.add_result(self._check_metric("rmse", rmse, "rmse", higher_is_better=False))

        report.print_report()
        return report
