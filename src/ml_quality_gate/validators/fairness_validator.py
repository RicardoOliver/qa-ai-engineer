"""
Fairness Validator
Validates ML model fairness across demographic groups.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np
import pandas as pd
from loguru import logger

from ml_quality_gate.utils.config_loader import get_thresholds


# ─────────────────────────────────────────────
# Data Models
# ─────────────────────────────────────────────

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
        groups_str = " | ".join(
            f"{g}={v:.3f}" for g, v in self.group_values.items()
        )

        return (
            f"{icon} [{self.metric_name}] "
            f"feature='{self.sensitive_feature}' | "
            f"disparity={self.disparity:.4f} "
            f"(max: {self.threshold:.4f}) | {groups_str}"
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
        logger.info(f"\n{'=' * 60}")
        logger.info("FAIRNESS & BIAS REPORT")
        logger.info(f"{'=' * 60}")

        for result in self.results:
            if result.passed:
                logger.success(str(result))
            else:
                logger.error(str(result))

        overall = "PASSED" if self.passed else "FAILED"

        logger.info(
            f"\nOverall Fairness: {overall} | Violations: {len(self.violations)}"
        )
        logger.info(f"{'=' * 60}\n")


# ─────────────────────────────────────────────
# Validator
# ─────────────────────────────────────────────

class FairnessValidator:
    def __init__(self) -> None:
        self._cfg: dict[str, Any] = get_thresholds()["fairness"]

    # 🔥 TIPAGEM CORRIGIDA (Callable)
    def _compute_group_metric(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        groups: pd.Series,
        metric_fn: Callable[[np.ndarray, np.ndarray], float],
    ) -> dict[str, float]:

        results: dict[str, float] = {}

        for group_val in groups.unique():
            mask = groups == group_val

            if int(mask.sum()) < 10:
                logger.warning(
                    f"Group '{group_val}' has only {mask.sum()} samples — skipping."
                )
                continue

            try:
                value = float(metric_fn(y_true[mask], y_pred[mask]))
                results[str(group_val)] = value
            except Exception as e:
                logger.warning(
                    f"Failed metric for group '{group_val}': {e}"
                )

        return results

    # ─────────────────────────────────────────────
    # Metrics
    # ─────────────────────────────────────────────

    def _accuracy(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        return float((y_true == y_pred).mean())

    def _selection_rate(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        return float(y_pred.mean())

    def _tpr(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        pos_mask = y_true == 1

        if int(pos_mask.sum()) == 0:
            return 0.0

        return float(y_pred[pos_mask].mean())

    # ─────────────────────────────────────────────
    # Main API
    # ─────────────────────────────────────────────

    def validate(
        self,
        y_true: np.ndarray | list[Any],
        y_pred: np.ndarray | list[Any],
        sensitive_features: pd.DataFrame | pd.Series,
    ) -> FairnessReport:

        y_true_np = np.asarray(y_true)
        y_pred_np = np.asarray(y_pred)

        if isinstance(sensitive_features, pd.Series):
            sensitive_features = sensitive_features.to_frame()

        report = FairnessReport()

        max_acc_disp = float(
            self._cfg.get("max_accuracy_disparity", 0.05)
        )
        max_dem_parity = float(
            self._cfg.get("max_demographic_parity_diff", 0.05)
        )
        max_eq_odds = float(
            self._cfg.get("max_equalized_odds_diff", 0.05)
        )

        for feature in sensitive_features.columns:
            groups = sensitive_features[feature]

            # ── Accuracy Disparity
            acc_by_group = self._compute_group_metric(
                y_true_np, y_pred_np, groups, self._accuracy
            )

            if acc_by_group:
                disparity = max(acc_by_group.values()) - min(acc_by_group.values())

                report.results.append(
                    FairnessMetric(
                        "accuracy_disparity",
                        feature,
                        disparity,
                        max_acc_disp,
                        disparity <= max_acc_disp,
                        acc_by_group,
                    )
                )

            # ── Demographic Parity
            sel_by_group = self._compute_group_metric(
                y_true_np, y_pred_np, groups, self._selection_rate
            )

            if sel_by_group:
                disparity = max(sel_by_group.values()) - min(sel_by_group.values())

                report.results.append(
                    FairnessMetric(
                        "demographic_parity",
                        feature,
                        disparity,
                        max_dem_parity,
                        disparity <= max_dem_parity,
                        sel_by_group,
                    )
                )

            # ── Equalized Odds (TPR)
            tpr_by_group = self._compute_group_metric(
                y_true_np, y_pred_np, groups, self._tpr
            )

            if tpr_by_group:
                disparity = max(tpr_by_group.values()) - min(tpr_by_group.values())

                report.results.append(
                    FairnessMetric(
                        "equalized_odds_tpr",
                        feature,
                        disparity,
                        max_eq_odds,
                        disparity <= max_eq_odds,
                        tpr_by_group,
                    )
                )

        report.print_report()
        return report