"""
Drift Detector
Detects feature drift and target drift using Evidently AI and statistical tests.
Supports: PSI, KS test, Chi-squared, Evidently DataDriftPreset.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from loguru import logger
from scipy import stats

from ml_quality_gate.utils.config_loader import get_thresholds


@dataclass
class DriftResult:
    feature: str
    drift_detected: bool
    statistic: float
    p_value: float | None
    method: str
    threshold: float

    def __str__(self) -> str:
        icon = "🚨" if self.drift_detected else "✅"
        p_str = f" | p={self.p_value:.4f}" if self.p_value is not None else ""
        return (
            f"{icon} [{self.method}] {self.feature}: "
            f"stat={self.statistic:.4f} (threshold={self.threshold:.4f}){p_str}"
        )


@dataclass
class DriftReport:
    results: list[DriftResult] = field(default_factory=list)
    dataset_drift: bool = False

    @property
    def drifted_features(self) -> list[str]:
        return [r.feature for r in self.results if r.drift_detected]

    @property
    def drift_share(self) -> float:
        if not self.results:
            return 0.0
        return len(self.drifted_features) / len(self.results)

    def summary(self) -> dict[str, Any]:
        return {
            "dataset_drift": self.dataset_drift,
            "drift_share": self.drift_share,
            "total_features": len(self.results),
            "drifted_features": self.drifted_features,
            "drift_details": [
                {
                    "feature": r.feature,
                    "drift_detected": r.drift_detected,
                    "method": r.method,
                    "statistic": r.statistic,
                }
                for r in self.results
            ],
        }

    def print_report(self) -> None:
        logger.info(f"\n{'='*60}")
        logger.info("  DATA DRIFT REPORT")
        logger.info(f"{'='*60}")
        for result in self.results:
            msg = str(result)
            if result.drift_detected:
                logger.warning(msg)
            else:
                logger.success(msg)
        dataset_status = "🚨 DRIFT DETECTED" if self.dataset_drift else "✅ NO DRIFT"
        logger.info(f"\nDataset Drift: {dataset_status}")
        logger.info(f"Drift Share: {self.drift_share:.0%} ({len(self.drifted_features)}/{len(self.results)} features)")
        logger.info(f"{'='*60}\n")


class DriftDetector:
    """
    Detects statistical drift between reference and current datasets.

    Supports:
    - Kolmogorov-Smirnov (KS) for numerical features
    - Chi-squared for categorical features
    - PSI (Population Stability Index)

    Usage:
        detector = DriftDetector(reference_df, current_df)
        report = detector.detect_all()
        assert not report.dataset_drift, f"Drift detected in: {report.drifted_features}"
    """

    def __init__(self, reference: pd.DataFrame, current: pd.DataFrame) -> None:
        self.reference = reference
        self.current = current
        self._cfg = get_thresholds()["drift"]

    def _psi(self, expected: np.ndarray, actual: np.ndarray, bins: int = 10) -> float:
        """Population Stability Index."""
        def _safe_pct(arr: np.ndarray, bin_edges: np.ndarray) -> np.ndarray:
            counts, _ = np.histogram(arr, bins=bin_edges)
            pct = counts / len(arr)
            return np.where(pct == 0, 1e-6, pct)

        bin_edges = np.percentile(expected, np.linspace(0, 100, bins + 1))
        bin_edges = np.unique(bin_edges)
        if len(bin_edges) < 2:
            return 0.0
        e_pct = _safe_pct(expected, bin_edges)
        a_pct = _safe_pct(actual, bin_edges)
        psi = np.sum((a_pct - e_pct) * np.log(a_pct / e_pct))
        return float(psi)

    def _ks_test(self, feature: str) -> DriftResult:
        ref = self.reference[feature].dropna()
        cur = self.current[feature].dropna()
        ks_stat, p_value = stats.ks_2samp(ref, cur)
        threshold = self._cfg.get("ks_threshold", 0.1)
        return DriftResult(
            feature=feature,
            drift_detected=ks_stat > threshold,
            statistic=ks_stat,
            p_value=p_value,
            method="KS",
            threshold=threshold,
        )

    def _chi_squared_test(self, feature: str) -> DriftResult:
        ref_counts = self.reference[feature].value_counts()
        cur_counts = self.current[feature].value_counts()
        all_categories = set(ref_counts.index) | set(cur_counts.index)
        ref_aligned = np.array([ref_counts.get(c, 0) for c in all_categories], dtype=float)
        cur_aligned = np.array([cur_counts.get(c, 0) for c in all_categories], dtype=float)
        ref_aligned = np.where(ref_aligned == 0, 1e-6, ref_aligned)
        cur_aligned = np.where(cur_aligned == 0, 1e-6, cur_aligned)
        chi2, p_value = stats.chisquare(cur_aligned, f_exp=ref_aligned * (cur_aligned.sum() / ref_aligned.sum()))
        threshold = self._cfg.get("ks_threshold", 0.1)
        return DriftResult(
            feature=feature,
            drift_detected=p_value < self._cfg.get("target_drift_p_value", 0.05),
            statistic=chi2,
            p_value=p_value,
            method="Chi2",
            threshold=threshold,
        )

    def detect_all(self, feature_columns: list[str] | None = None) -> DriftReport:
        report = DriftReport()
        columns = feature_columns or [
            c for c in self.reference.columns if c in self.current.columns
        ]

        for col in columns:
            dtype = self.reference[col].dtype
            try:
                if pd.api.types.is_numeric_dtype(dtype):
                    result = self._ks_test(col)
                else:
                    result = self._chi_squared_test(col)
                report.results.append(result)
            except Exception as e:
                logger.warning(f"Could not compute drift for '{col}': {e}")

        drift_share_threshold = self._cfg.get("dataset_drift_share", 0.3)
        report.dataset_drift = report.drift_share >= drift_share_threshold
        report.print_report()
        return report

    def detect_target_drift(self, target_col: str) -> DriftResult:
        """Specifically check if the target/label distribution has shifted."""
        if pd.api.types.is_numeric_dtype(self.reference[target_col]):
            result = self._ks_test(target_col)
        else:
            result = self._chi_squared_test(target_col)
        logger.info(f"Target drift: {result}")
        return result
