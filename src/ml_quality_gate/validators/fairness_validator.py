"""
Fairness Validator
Validates ML model fairness across demographic groups.
Metrics: Demographic Parity, Equalized Odds, Accuracy Disparity, Selection Rate.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from loguru import logger

from ml_quality_gate.utils.config_loader import get_thresholds


@dataclass
class FairnessMetric:
    metric_name: str
    sensitive_feature: str
    disparity: float
    threshold: float
    passed: bool
    group_values: dict[str, float] = field(default_factory=dict)

    def __str__(self) -> str:
        icon = "✅" if self.passed else "❌"
        groups_str = " | ".join(f"{g}={v:.3f}" for g, v in self.group_values.items())
        return (
            f"{icon} [{self.metric_name}] feature='{self.sensitive_feature}' | "
            f"disparity={self.disparity:.4f} (max: {self.threshold:.4f}) | {groups_str}"
        )


@dataclass
class FairnessReport:
    results: list[FairnessMetric] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return all(r.passed for r in self.results)

    @property
    def violations(self) -> list[FairnessMetric]:
        return [r for r in self.results if not r.passed]

    def summary(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "total_checks": len(self.results),
            "violations": len(self.violations),
            "violation_details": [
                {
                    "metric": v.metric_name,
                    "feature": v.sensitive_feature,
                    "disparity": v.disparity,
                    "threshold": v.threshold,
                }
                for v in self.violations
            ],
        }

    def print_report(self) -> None:
        logger.info(f"\n{'='*60}")
        logger.info("  FAIRNESS & BIAS REPORT")
        logger.info(f"{'='*60}")
        for result in self.results:
            msg = str(result)
            if result.passed:
                logger.success(msg)
            else:
                logger.error(msg)
        overall = "✅ PASSED" if self.passed else "❌ FAILED"
        logger.info(f"\nOverall Fairness: {overall} | Violations: {len(self.violations)}")
        logger.info(f"{'='*60}\n")


class FairnessValidator:
    """
    Evaluates model fairness across protected/sensitive demographic groups.

    Metrics computed per group:
    - Accuracy Disparity
    - Demographic Parity (selection rate diff)
    - Equalized Odds (TPR + FPR diff)

    Usage:
        validator = FairnessValidator()
        report = validator.validate(
            y_true, y_pred,
            sensitive_features=df[["gender", "age_group"]]
        )
        assert report.passed
    """

    def __init__(self) -> None:
        self._cfg = get_thresholds()["fairness"]

    def _compute_group_metric(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        groups: pd.Series,
        metric_fn: Any,
    ) -> dict[str, float]:
        results = {}
        for group_val in groups.unique():
            mask = groups == group_val
            if mask.sum() < 10:  # Skip groups with too few samples
                logger.warning(f"Group '{group_val}' has only {mask.sum()} samples — skipping.")
                continue
            try:
                results[str(group_val)] = float(metric_fn(y_true[mask], y_pred[mask]))
            except Exception as e:
                logger.warning(f"Could not compute metric for group '{group_val}': {e}")
        return results

    def _accuracy(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        return float((y_true == y_pred).mean())

    def _selection_rate(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        return float(y_pred.mean())

    def _tpr(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        pos_mask = y_true == 1
        if pos_mask.sum() == 0:
            return 0.0
        return float(y_pred[pos_mask].mean())

    def validate(
        self,
        y_true: np.ndarray | list,
        y_pred: np.ndarray | list,
        sensitive_features: pd.DataFrame | pd.Series,
    ) -> FairnessReport:
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)

        if isinstance(sensitive_features, pd.Series):
            sensitive_features = sensitive_features.to_frame()

        report = FairnessReport()

        max_acc_disp = self._cfg.get("max_accuracy_disparity", 0.05)
        max_dem_parity = self._cfg.get("max_demographic_parity_diff", 0.05)
        max_eq_odds = self._cfg.get("max_equalized_odds_diff", 0.05)

        for feature in sensitive_features.columns:
            groups = sensitive_features[feature]

            # 1. Accuracy Disparity
            acc_by_group = self._compute_group_metric(y_true, y_pred, groups, self._accuracy)
            if acc_by_group:
                disparity = max(acc_by_group.values()) - min(acc_by_group.values())
                report.results.append(FairnessMetric(
                    metric_name="accuracy_disparity",
                    sensitive_feature=feature,
                    disparity=disparity,
                    threshold=max_acc_disp,
                    passed=disparity <= max_acc_disp,
                    group_values=acc_by_group,
                ))

            # 2. Demographic Parity (Selection Rate)
            sel_by_group = self._compute_group_metric(y_true, y_pred, groups, self._selection_rate)
            if sel_by_group:
                disparity = max(sel_by_group.values()) - min(sel_by_group.values())
                report.results.append(FairnessMetric(
                    metric_name="demographic_parity",
                    sensitive_feature=feature,
                    disparity=disparity,
                    threshold=max_dem_parity,
                    passed=disparity <= max_dem_parity,
                    group_values=sel_by_group,
                ))

            # 3. Equalized Odds (TPR)
            tpr_by_group = self._compute_group_metric(y_true, y_pred, groups, self._tpr)
            if tpr_by_group:
                disparity = max(tpr_by_group.values()) - min(tpr_by_group.values())
                report.results.append(FairnessMetric(
                    metric_name="equalized_odds_tpr",
                    sensitive_feature=feature,
                    disparity=disparity,
                    threshold=max_eq_odds,
                    passed=disparity <= max_eq_odds,
                    group_values=tpr_by_group,
                ))

        report.print_report()
        return report
