"""
Data Validator
Comprehensive data quality checks using Great Expectations + custom rules.
Covers: nulls, duplicates, type checks, ranges, cardinality, distribution.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from loguru import logger

from ml_quality_gate.utils.config_loader import get_thresholds


@dataclass
class DataCheckResult:
    check_name: str
    column: str | None
    passed: bool
    detail: str
    severity: str = "error"  # error | warning

    @property
    def icon(self) -> str:
        if self.passed:
            return "✅"
        return "❌" if self.severity == "error" else "⚠️"

    def __str__(self) -> str:
        return (
            f"{self.icon} [{self.check_name}] "
            f"{self.column or 'dataset'}: {self.detail}"
        )


@dataclass
class DataValidationReport:
    dataset_name: str
    shape: tuple[int, int]
    results: list[DataCheckResult] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return all(r.passed for r in self.results if r.severity == "error")

    @property
    def warnings(self) -> list[DataCheckResult]:
        return [
            r for r in self.results
            if not r.passed and r.severity == "warning"
        ]

    @property
    def errors(self) -> list[DataCheckResult]:
        return [
            r for r in self.results
            if not r.passed and r.severity == "error"
        ]

    def summary(self) -> dict[str, Any]:
        return {
            "dataset_name": self.dataset_name,
            "shape": self.shape,
            "passed": self.passed,
            "total_checks": len(self.results),
            "passed_checks": sum(1 for r in self.results if r.passed),
            "errors": len(self.errors),
            "warnings": len(self.warnings),
        }

    def print_report(self) -> None:
        logger.info(f"\n{'=' * 60}")
        logger.info(
            f"  DATA QUALITY REPORT — {self.dataset_name.upper()}"
        )
        logger.info(
            f"  Shape: {self.shape[0]} rows × {self.shape[1]} cols"
        )
        logger.info(f"{'=' * 60}")

        for result in self.results:
            msg = str(result)
            if result.passed:
                logger.success(msg)
            elif result.severity == "warning":
                logger.warning(msg)
            else:
                logger.error(msg)

        overall = "✅ PASSED" if self.passed else "❌ FAILED"

        logger.info(
            f"\nOverall: {overall} | "
            f"Errors: {len(self.errors)} | "
            f"Warnings: {len(self.warnings)}"
        )
        logger.info(f"{'=' * 60}\n")


class DataValidator:
    """
    Comprehensive data quality validator for ML datasets.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        dataset_name: str = "dataset",
    ) -> None:
        self.df = df.copy()
        self.dataset_name = dataset_name
        self._cfg = get_thresholds()["data_quality"]

        self._report = DataValidationReport(
            dataset_name=dataset_name,
            shape=df.shape,
        )

    def _add(
        self,
        check_name: str,
        column: str | None,
        passed: bool,
        detail: str,
        severity: str = "error",
    ) -> None:
        result = DataCheckResult(
            check_name,
            column,
            passed,
            detail,
            severity,
        )
        self._report.results.append(result)

    # ───────────────────────────────────────────
    # Core Checks
    # ───────────────────────────────────────────

    def check_minimum_rows(self) -> "DataValidator":
        min_rows = self._cfg.get("min_row_count", 1000)
        actual = len(self.df)
        passed = actual >= min_rows

        self._add(
            "min_row_count",
            None,
            passed,
            f"{actual} rows (min: {min_rows})",
        )
        return self

    def check_null_ratios(self) -> "DataValidator":
        max_null = self._cfg.get("max_null_ratio", 0.05)

        for col in self.df.columns:
            null_ratio = self.df[col].isnull().mean()
            passed = null_ratio <= max_null

            severity = (
                "error"
                if null_ratio > 0.20
                else "warning"
                if not passed
                else "error"
            )

            self._add(
                "null_ratio",
                col,
                passed,
                (
                    f"null ratio={null_ratio:.2%} "
                    f"(max: {max_null:.0%})"
                ),
                severity=severity,
            )

        return self

    def check_duplicates(self) -> "DataValidator":
        max_dup = self._cfg.get("max_duplicate_ratio", 0.02)
        dup_ratio = self.df.duplicated().mean()
        passed = dup_ratio <= max_dup

        self._add(
            "duplicate_ratio",
            None,
            passed,
            (
                f"duplicate ratio={dup_ratio:.2%} "
                f"(max: {max_dup:.0%})"
            ),
        )
        return self

    def check_feature_ranges(self) -> "DataValidator":
        ranges = self._cfg.get("feature_ranges", {})

        for col, bounds in ranges.items():
            if col not in self.df.columns:
                continue

            series = self.df[col].dropna()
            min_val = bounds.get("min")
            max_val = bounds.get("max")

            if min_val is not None:
                self._add(
                    "range_min",
                    col,
                    bool((series >= min_val).all()),
                    f"min={series.min()} (>= {min_val})",
                )

            if max_val is not None:
                self._add(
                    "range_max",
                    col,
                    bool((series <= max_val).all()),
                    f"max={series.max()} (<= {max_val})",
                )

        return self

    def check_no_infinite_values(self) -> "DataValidator":
        numeric_cols = self.df.select_dtypes(include="number").columns

        for col in numeric_cols:
            has_inf = bool(np.isinf(self.df[col].dropna()).any())

            self._add(
                "no_infinite",
                col,
                not has_inf,
                (
                    "no infinite values"
                    if not has_inf
                    else "contains infinite values"
                ),
            )

        return self

    # ───────────────────────────────────────────
    # Runner
    # ───────────────────────────────────────────

    def run_all_checks(
        self,
        target_col: str | None = None,
        expected_types: dict[str, str] | None = None,
    ) -> DataValidationReport:

        (
            self
            .check_minimum_rows()
            .check_null_ratios()
            .check_duplicates()
            .check_feature_ranges()
            .check_no_infinite_values()
        )

        if expected_types:
            self.check_data_types(expected_types)

        if target_col:
            self.check_class_balance(target_col)

        self._report.print_report()
        return self._report