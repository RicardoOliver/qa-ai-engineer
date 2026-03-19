"""
Advanced Unit Tests — Model Robustness & Edge Cases
Tests model behavior under adversarial and boundary conditions.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import GradientBoostingClassifier

from ml_quality_gate.validators.data_validator import DataValidator
from tests.plugin import MLAssert


@pytest.mark.unit
class TestModelRobustness:
    """Tests that go beyond metrics — testing model behavior and robustness."""

    @pytest.fixture(autouse=True)
    def setup(
        self,
        trained_model: GradientBoostingClassifier,
        train_test_data: dict[str, Any],
    ) -> None:
        self.model = trained_model
        self.x_test = train_test_data["X_test"]
        self.y_test = train_test_data["y_test"].values
        self.feature_cols = train_test_data["feature_cols"]

    def test_model_handles_all_employed_inputs(self) -> None:
        """Model must not crash or produce NaN on extreme but valid inputs."""
        extreme_df = pd.DataFrame(
            [
                {
                    "age": 99,
                    "income": 10_000_000.0,
                    "credit_score": 850,
                    "loan_amount": 1_000.0,
                    "loan_term_months": 360,
                    "employment_type": 0,
                    "marital_status": 1,
                }
            ]
        )

        pred = self.model.predict(extreme_df)
        prob = self.model.predict_proba(extreme_df)

        assert not np.isnan(pred).any(), "NaN in predictions for extreme input"
        assert not np.isnan(prob).any(), "NaN in probabilities for extreme input"

    def test_high_credit_score_biases_toward_no_default(self) -> None:
        """Applicants with top credit scores should mostly be no-default."""
        high_credit_rows = self.x_test[self.x_test["credit_score"] >= 800]

        if len(high_credit_rows) < 5:
            pytest.skip("Not enough high-credit-score samples.")

        preds = self.model.predict(high_credit_rows)
        no_default_rate = (preds == 0).mean()

        assert no_default_rate >= 0.70, (
            f"Only {no_default_rate:.0%} predicted as no-default"
        )

    def test_low_credit_score_biases_toward_default(self) -> None:
        """Applicants with low credit scores should mostly default."""
        low_credit_rows = self.x_test[self.x_test["credit_score"] <= 450]

        if len(low_credit_rows) < 5:
            pytest.skip("Not enough low-credit-score samples.")

        preds = self.model.predict(low_credit_rows)
        default_rate = (preds == 1).mean()

        assert default_rate >= 0.50, (
            f"Only {default_rate:.0%} predicted as default"
        )

    def test_monotonicity_credit_score(self) -> None:
        """Increasing credit score should reduce default probability."""
        base_row = self.x_test.iloc[0:1].copy()

        credit_scores = [350, 450, 550, 650, 700, 750, 800, 850]
        probs = []

        for score in credit_scores:
            row = base_row.copy()
            row["credit_score"] = score
            prob = self.model.predict_proba(row)[0, 1]
            probs.append(prob)

        trend_ok = all(
            probs[i] >= probs[i + 1] - 0.05
            for i in range(len(probs) - 1)
        )

        assert trend_ok, (
            "Monotonicity violated\n"
            f"Scores: {credit_scores}\n"
            f"Probs: {[f'{p:.3f}' for p in probs]}"
        )

    def test_confidence_calibration(self, predictions: dict[str, Any]) -> None:
        MLAssert.confidence_calibrated(
            y_prob=predictions["y_prob"],
            y_true=predictions["y_true"],
            y_pred=predictions["y_pred"],
            confidence_threshold=0.85,
            min_accuracy=0.75,
        )

    def test_prediction_batch_matches_individual(self) -> None:
        sample = self.x_test.iloc[:20]

        batch_preds = self.model.predict(sample)

        individual_preds = np.array(
            [
                self.model.predict(sample.iloc[i : i + 1])[0]
                for i in range(len(sample))
            ]
        )

        np.testing.assert_array_equal(batch_preds, individual_preds)

    def test_feature_importances_sum_to_one(self) -> None:
        total = self.model.feature_importances_.sum()
        assert abs(total - 1.0) < 1e-6

    def test_top_feature_is_credit_score(self) -> None:
        importances = dict(
            zip(self.feature_cols, self.model.feature_importances_)
        )
        top_feature = max(importances, key=importances.get)

        assert top_feature == "credit_score"

    def test_no_nan_in_predictions(self) -> None:
        preds = self.model.predict(self.x_test)
        assert not np.isnan(preds).any()

    def test_no_nan_in_probabilities(self) -> None:
        probs = self.model.predict_proba(self.x_test)
        assert not np.isnan(probs).any()

    def test_probability_rows_sum_to_one(self) -> None:
        probs = self.model.predict_proba(self.x_test)
        np.testing.assert_allclose(probs.sum(axis=1), 1.0, atol=1e-6)


@pytest.mark.unit
class TestDataValidatorEdgeCases:
    """Edge case tests for the DataValidator."""

    def test_empty_dataframe_fails_row_check(self) -> None:
        empty_df = pd.DataFrame({"age": [], "income": []})

        validator = DataValidator(empty_df, "empty")
        report = validator.check_minimum_rows()._report

        assert not report.passed

    def test_100pct_nulls_fails(self) -> None:
        df = pd.DataFrame(
            {
                "age": [np.nan] * 100,
                "income": [np.nan] * 100,
            }
        )

        validator = DataValidator(df, "all_nulls")
        report = validator.check_null_ratios()._report

        assert not report.passed

    def test_all_duplicates_fails(self) -> None:
        df = pd.DataFrame(
            {"age": [35] * 200, "income": [50000.0] * 200}
        )

        validator = DataValidator(df, "all_dupes")
        report = validator.check_duplicates()._report

        assert not report.passed

    def test_infinite_values_detected(self) -> None:
        df = pd.DataFrame(
            {
                "age": [35, np.inf, 40],
                "income": [50000, 60000, 70000],
            }
        )

        validator = DataValidator(df, "has_inf")
        report = validator.check_no_infinite_values()._report

        inf_checks = [
            r for r in report.results if r.check_name == "no_infinite"
        ]

        assert any(not r.passed for r in inf_checks)