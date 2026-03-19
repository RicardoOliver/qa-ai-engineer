"""
Unit Tests — Data Quality Validation
Validates training and inference data meets quality standards.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ml_quality_gate.validators.data_validator import DataValidator
from ml_quality_gate.validators.drift_detector import DriftDetector


@pytest.mark.unit
class TestDataQuality:
    """Tests that training data meets quality requirements."""

    def test_dataset_has_minimum_rows(self, raw_dataframe: pd.DataFrame) -> None:
        validator = DataValidator(raw_dataframe, dataset_name="training")
        report = validator.check_minimum_rows()._report
        row_check = next(r for r in report.results if r.check_name == "min_row_count")
        assert row_check.passed, row_check.detail

    def test_no_excessive_nulls(self, raw_dataframe: pd.DataFrame) -> None:
        validator = DataValidator(raw_dataframe, dataset_name="training")
        report = validator.check_null_ratios()._report
        errors = [r for r in report.results if not r.passed and r.severity == "error"]
        assert not errors, f"Null ratio errors: {[str(e) for e in errors]}"

    def test_no_excessive_duplicates(self, raw_dataframe: pd.DataFrame) -> None:
        validator = DataValidator(raw_dataframe, dataset_name="training")
        report = validator.check_duplicates()._report
        dup_check = next(r for r in report.results if r.check_name == "duplicate_ratio")
        assert dup_check.passed, dup_check.detail

    def test_age_within_valid_range(self, raw_dataframe: pd.DataFrame) -> None:
        validator = DataValidator(raw_dataframe, dataset_name="training")
        report = validator.check_feature_ranges()._report
        age_checks = [r for r in report.results if r.column == "age"]
        assert all(r.passed for r in age_checks), f"Age range violations: {age_checks}"

    def test_credit_score_within_valid_range(self, raw_dataframe: pd.DataFrame) -> None:
        assert raw_dataframe["credit_score"].between(300, 850).all(), (
            "Credit scores out of valid range [300, 850]"
        )

    def test_no_negative_income(self, raw_dataframe: pd.DataFrame) -> None:
        assert (raw_dataframe["income"] >= 0).all(), "Negative income values found"

    def test_employment_type_valid_categories(self, raw_dataframe: pd.DataFrame) -> None:
        valid = {"employed", "self_employed", "unemployed", "retired"}
        actual = set(raw_dataframe["employment_type"].unique())
        unexpected = actual - valid
        assert not unexpected, f"Unexpected employment_type values: {unexpected}"

    def test_marital_status_valid_categories(self, raw_dataframe: pd.DataFrame) -> None:
        valid = {"single", "married", "divorced", "widowed"}
        actual = set(raw_dataframe["marital_status"].unique())
        unexpected = actual - valid
        assert not unexpected, f"Unexpected marital_status values: {unexpected}"

    def test_no_infinite_values(self, raw_dataframe: pd.DataFrame) -> None:
        numeric = raw_dataframe.select_dtypes(include="number")
        assert not np.isinf(numeric.values).any(), "Infinite values found in dataset"

    def test_target_column_is_binary(self, raw_dataframe: pd.DataFrame) -> None:
        unique_vals = set(raw_dataframe["default"].unique())
        assert unique_vals.issubset({0, 1}), f"Non-binary target values: {unique_vals}"

    def test_full_data_quality_report_passes(self, raw_dataframe: pd.DataFrame) -> None:
        validator = DataValidator(raw_dataframe, dataset_name="full_training")
        report = validator.run_all_checks(target_col="default")
        assert report.passed, (
            f"Data quality gate FAILED. Errors: {[str(e) for e in report.errors]}"
        )


@pytest.mark.unit
class TestDataDrift:
    """Tests drift detection logic."""

    def test_no_drift_on_similar_data(
        self,
        reference_df: pd.DataFrame,
        current_df_no_drift: pd.DataFrame,
    ) -> None:
        numeric_features = ["age", "income", "credit_score", "loan_amount"]
        detector = DriftDetector(reference_df[numeric_features], current_df_no_drift[numeric_features])
        report = detector.detect_all()
        assert not report.dataset_drift, (
            f"False positive drift detected in: {report.drifted_features}"
        )

    def test_drift_detected_on_shifted_data(
        self,
        reference_df: pd.DataFrame,
        current_df_with_drift: pd.DataFrame,
    ) -> None:
        numeric_features = ["age", "income", "credit_score", "loan_amount"]
        detector = DriftDetector(reference_df[numeric_features], current_df_with_drift[numeric_features])
        report = detector.detect_all()
        assert report.dataset_drift, "Drift was NOT detected despite significant data shift"

    def test_drift_report_has_expected_structure(
        self,
        reference_df: pd.DataFrame,
        current_df_no_drift: pd.DataFrame,
    ) -> None:
        numeric_features = ["age", "income"]
        detector = DriftDetector(reference_df[numeric_features], current_df_no_drift[numeric_features])
        report = detector.detect_all()
        summary = report.summary()
        assert "dataset_drift" in summary
        assert "drifted_features" in summary
        assert "drift_share" in summary
        assert "total_features" in summary

    def test_drift_share_is_valid_probability(
        self,
        reference_df: pd.DataFrame,
        current_df_no_drift: pd.DataFrame,
    ) -> None:
        numeric_features = ["age", "income", "credit_score"]
        detector = DriftDetector(reference_df[numeric_features], current_df_no_drift[numeric_features])
        report = detector.detect_all()
        assert 0.0 <= report.drift_share <= 1.0
