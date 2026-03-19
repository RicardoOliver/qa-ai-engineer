"""
Unit Tests — Model Metrics Validation
Validates that trained model meets all quality thresholds.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
from sklearn.ensemble import GradientBoostingClassifier

from ml_quality_gate.validators.model_validator import ModelValidator


@pytest.mark.unit
class TestModelMetrics:
    """Quality gate tests for model metrics. Fails CI/CD if thresholds not met."""

    @pytest.fixture(autouse=True)
    def setup(self, predictions: dict[str, Any]) -> None:
        self.y_true = predictions["y_true"]
        self.y_pred = predictions["y_pred"]
        self.y_prob = predictions["y_prob"]
        self.validator = ModelValidator(model_name="credit_risk_gbm")

    def test_full_classification_gate_passes(self) -> None:
        """Full classification quality gate must pass."""
        report = self.validator.validate_classification(
            y_true=self.y_true,
            y_pred=self.y_pred,
            y_prob=self.y_prob,
        )
        assert report.passed, (
            f"Quality gate FAILED. Failed metrics: "
            f"{[r.name for r in report.results if not r.passed]}"
        )

    def test_no_critical_failures(self) -> None:
        """No metric should be in critical failure zone."""
        report = self.validator.validate_classification(
            y_true=self.y_true,
            y_pred=self.y_pred,
            y_prob=self.y_prob,
        )
        assert not report.has_critical_failure, (
            f"CRITICAL metric failure detected: "
            f"{[r.name for r in report.results if r.critical]}"
        )

    def test_accuracy_meets_threshold(self) -> None:
        """Accuracy must meet minimum threshold."""
        from sklearn.metrics import accuracy_score
        from ml_quality_gate.utils.config_loader import get_threshold

        acc = accuracy_score(self.y_true, self.y_pred)
        threshold = get_threshold("model", "accuracy")
        assert acc >= threshold, f"Accuracy {acc:.4f} < threshold {threshold:.4f}"

    def test_precision_meets_threshold(self) -> None:
        from sklearn.metrics import precision_score
        from ml_quality_gate.utils.config_loader import get_threshold

        precision = precision_score(self.y_true, self.y_pred, zero_division=0)
        threshold = get_threshold("model", "precision")
        assert precision >= threshold, f"Precision {precision:.4f} < {threshold:.4f}"

    def test_recall_meets_threshold(self) -> None:
        from sklearn.metrics import recall_score
        from ml_quality_gate.utils.config_loader import get_threshold

        recall = recall_score(self.y_true, self.y_pred, zero_division=0)
        threshold = get_threshold("model", "recall")
        assert recall >= threshold, f"Recall {recall:.4f} < {threshold:.4f}"

    def test_f1_meets_threshold(self) -> None:
        from sklearn.metrics import f1_score
        from ml_quality_gate.utils.config_loader import get_threshold

        f1 = f1_score(self.y_true, self.y_pred, zero_division=0)
        threshold = get_threshold("model", "f1_score")
        assert f1 >= threshold, f"F1 {f1:.4f} < {threshold:.4f}"

    def test_roc_auc_meets_threshold(self) -> None:
        from sklearn.metrics import roc_auc_score
        from ml_quality_gate.utils.config_loader import get_threshold

        auc = roc_auc_score(self.y_true, self.y_prob)
        threshold = get_threshold("model", "roc_auc")
        assert auc >= threshold, f"ROC-AUC {auc:.4f} < {threshold:.4f}"

    def test_model_summary_has_all_keys(self) -> None:
        report = self.validator.validate_classification(
            y_true=self.y_true,
            y_pred=self.y_pred,
        )
        summary = report.summary()
        required_keys = {"model_name", "task", "passed", "total_checks", "metrics"}
        assert required_keys.issubset(summary.keys())

    def test_model_is_deterministic(self, trained_model: GradientBoostingClassifier, train_test_data: dict) -> None:
        """Same input must always produce same output."""
        X_test = train_test_data["X_test"]
        pred1 = trained_model.predict(X_test)
        pred2 = trained_model.predict(X_test)
        np.testing.assert_array_equal(pred1, pred2, err_msg="Model is non-deterministic!")

    def test_prediction_classes_are_binary(self) -> None:
        """Model must output only binary predictions [0, 1]."""
        unique_preds = set(self.y_pred.tolist())
        assert unique_preds.issubset({0, 1}), f"Non-binary predictions: {unique_preds}"

    def test_probability_scores_are_valid(self) -> None:
        """All probability scores must be in [0, 1]."""
        assert np.all(self.y_prob >= 0.0) and np.all(self.y_prob <= 1.0), (
            "Invalid probability scores detected outside [0, 1]"
        )

    def test_probability_calibration(self) -> None:
        """High-confidence predictions should be mostly correct (sanity check)."""
        high_conf_mask = self.y_prob > 0.9
        if high_conf_mask.sum() > 10:
            high_conf_preds = self.y_pred[high_conf_mask]
            high_conf_true = self.y_true[high_conf_mask]
            acc = (high_conf_preds == high_conf_true).mean()
            assert acc >= 0.75, f"High-confidence predictions are poorly calibrated: {acc:.2%} accuracy"
