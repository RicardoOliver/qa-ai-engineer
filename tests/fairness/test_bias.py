"""
Fairness & Bias Tests
Validates model treats all demographic groups equitably.
"""

from __future__ import annotations

from typing import Any

import pandas as pd
import pytest

from ml_quality_gate.validators.fairness_validator import FairnessValidator


@pytest.mark.fairness
class TestFairnessAndBias:
    """Bias and fairness tests. Blocks deployment on violations."""

    @pytest.fixture(autouse=True)
    def setup(self, predictions: dict[str, Any], sensitive_features: pd.DataFrame) -> None:
        self.y_true = predictions["y_true"]
        self.y_pred = predictions["y_pred"]
        # Align sensitive features with test set indices
        self.sensitive = sensitive_features.iloc[-len(self.y_true):].reset_index(drop=True)
        self.validator = FairnessValidator()

    def test_full_fairness_gate_passes(self) -> None:
        """Full fairness gate must pass for all sensitive features."""
        report = self.validator.validate(
            y_true=self.y_true,
            y_pred=self.y_pred,
            sensitive_features=self.sensitive,
        )
        assert report.passed, (
            f"Fairness gate FAILED. Violations:\n"
            + "\n".join(
                f"  • {v.metric_name} on '{v.sensitive_feature}': "
                f"disparity={v.disparity:.4f} > threshold={v.threshold:.4f}"
                for v in report.violations
            )
        )

    def test_no_gender_accuracy_disparity(self) -> None:
        """Accuracy gap between genders must not exceed threshold."""
        gender_only = self.sensitive[["gender"]]
        report = self.validator.validate(self.y_true, self.y_pred, gender_only)
        acc_checks = [r for r in report.results if r.metric_name == "accuracy_disparity"]
        assert all(c.passed for c in acc_checks), (
            f"Gender accuracy disparity detected: {[str(c) for c in acc_checks]}"
        )

    def test_no_gender_demographic_parity_violation(self) -> None:
        """Selection rate difference between genders must be within threshold."""
        gender_only = self.sensitive[["gender"]]
        report = self.validator.validate(self.y_true, self.y_pred, gender_only)
        parity_checks = [r for r in report.results if r.metric_name == "demographic_parity"]
        assert all(c.passed for c in parity_checks), (
            f"Demographic parity violation: {[str(c) for c in parity_checks]}"
        )

    def test_no_age_group_disparity(self) -> None:
        """Accuracy must be consistent across age groups."""
        age_only = self.sensitive[["age_group"]]
        report = self.validator.validate(self.y_true, self.y_pred, age_only)
        assert report.passed, (
            f"Age group fairness violation: {[str(v) for v in report.violations]}"
        )

    def test_fairness_report_structure(self) -> None:
        """Fairness report must have required structure."""
        report = self.validator.validate(self.y_true, self.y_pred, self.sensitive)
        summary = report.summary()
        assert "passed" in summary
        assert "total_checks" in summary
        assert "violations" in summary
        assert isinstance(summary["violation_details"], list)

    def test_no_violations_count(self) -> None:
        """Total fairness violations must be zero."""
        report = self.validator.validate(self.y_true, self.y_pred, self.sensitive)
        assert len(report.violations) == 0, (
            f"{len(report.violations)} fairness violation(s) detected. "
            f"Review model for bias before deployment."
        )
