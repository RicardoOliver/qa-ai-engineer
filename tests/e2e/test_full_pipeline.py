"""
End-to-End Tests — Full ML Pipeline Validation
Tests the complete flow: data → model → fairness → API → report.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import GradientBoostingClassifier

from ml_quality_gate.validators.model_validator import ModelValidator
from ml_quality_gate.validators.data_validator import DataValidator
from ml_quality_gate.validators.drift_detector import DriftDetector
from ml_quality_gate.validators.fairness_validator import FairnessValidator


@pytest.mark.e2e
@pytest.mark.slow
class TestFullMLPipeline:
    """
    End-to-end pipeline tests that validate the entire ML quality gate.
    These run last and act as the final deployment gate.
    """

    def test_complete_quality_gate_pipeline(
        self,
        raw_dataframe: pd.DataFrame,
        predictions: dict[str, Any],
        reference_df: pd.DataFrame,
        current_df_no_drift: pd.DataFrame,
        sensitive_features: pd.DataFrame,
    ) -> None:
        """
        Full pipeline: Data Quality → Model Metrics → Drift → Fairness.
        All gates must pass for deployment to proceed.
        """
        failures = []

        # ── Step 1: Data Quality ───────────────────────────────────────────────
        dq_report = DataValidator(raw_dataframe, "e2e_dataset").run_all_checks(target_col="default")
        if not dq_report.passed:
            failures.append(f"Data Quality: {len(dq_report.errors)} error(s)")

        # ── Step 2: Model Metrics ──────────────────────────────────────────────
        model_report = ModelValidator("credit_risk_gbm").validate_classification(
            y_true=predictions["y_true"],
            y_pred=predictions["y_pred"],
            y_prob=predictions["y_prob"],
        )
        if not model_report.passed:
            failures.append(f"Model Metrics: {[r.name for r in model_report.results if not r.passed]}")

        # ── Step 3: Drift Detection ────────────────────────────────────────────
        numeric_feats = ["age", "income", "credit_score", "loan_amount"]
        drift_report = DriftDetector(
            reference_df[numeric_feats],
            current_df_no_drift[numeric_feats],
        ).detect_all()
        if drift_report.dataset_drift:
            failures.append(f"Drift: detected in {drift_report.drifted_features}")

        # ── Step 4: Fairness ───────────────────────────────────────────────────
        aligned_sensitive = sensitive_features.iloc[-len(predictions["y_true"]):].reset_index(drop=True)
        fairness_report = FairnessValidator().validate(
            y_true=predictions["y_true"],
            y_pred=predictions["y_pred"],
            sensitive_features=aligned_sensitive,
        )
        if not fairness_report.passed:
            failures.append(f"Fairness: {len(fairness_report.violations)} violation(s)")

        # ── Final Gate ─────────────────────────────────────────────────────────
        assert not failures, (
            f"\n🚨 DEPLOYMENT BLOCKED — Quality Gate Failed:\n"
            + "\n".join(f"  ❌ {f}" for f in failures)
        )

    def test_model_retraining_regression_check(
        self,
        train_test_data: dict[str, Any],
        predictions: dict[str, Any],
    ) -> None:
        """
        Simulates a retrained model and checks it doesn't regress vs baseline.
        """
        from sklearn.metrics import f1_score

        baseline_f1 = f1_score(predictions["y_true"], predictions["y_pred"], zero_division=0)

        # Simulate a degraded model (fewer estimators = weaker)
        degraded_model = GradientBoostingClassifier(n_estimators=5, random_state=42)
        degraded_model.fit(train_test_data["X_train"], train_test_data["y_train"])
        degraded_preds = degraded_model.predict(train_test_data["X_test"])
        degraded_f1 = f1_score(train_test_data["y_test"], degraded_preds, zero_division=0)

        max_regression = 0.05  # Allow max 5% F1 regression
        regression = baseline_f1 - degraded_f1
        assert regression <= max_regression, (
            f"Model regression detected: baseline F1={baseline_f1:.4f}, "
            f"new F1={degraded_f1:.4f}, regression={regression:.4f} > {max_regression}"
        )

    def test_all_validators_produce_serializable_summaries(
        self,
        raw_dataframe: pd.DataFrame,
        predictions: dict[str, Any],
        reference_df: pd.DataFrame,
        current_df_no_drift: pd.DataFrame,
        sensitive_features: pd.DataFrame,
    ) -> None:
        """All report summaries must be JSON-serializable for downstream use."""
        import json

        numeric_feats = ["age", "income", "credit_score"]
        aligned_sensitive = sensitive_features.iloc[-len(predictions["y_true"]):].reset_index(drop=True)

        summaries = {
            "data": DataValidator(raw_dataframe).run_all_checks().summary(),
            "model": ModelValidator().validate_classification(
                predictions["y_true"], predictions["y_pred"]
            ).summary(),
            "drift": DriftDetector(
                reference_df[numeric_feats], current_df_no_drift[numeric_feats]
            ).detect_all().summary(),
            "fairness": FairnessValidator().validate(
                predictions["y_true"], predictions["y_pred"], aligned_sensitive
            ).summary(),
        }

        for name, summary in summaries.items():
            try:
                json.dumps(summary)
            except (TypeError, ValueError) as e:
                pytest.fail(f"Summary for '{name}' is not JSON-serializable: {e}")
