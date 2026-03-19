"""
Data Validator
Comprehensive data quality checks for ML datasets.
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
    severity: str = "error"

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
        return [r for r in self.results if not r.passed and r.severity == "warning"]

    @property
    def errors(self) -> list[DataCheckResult]:
        return [r for r in self.results if not r.passed and r.severity == "error"]

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
        logger.info(f"DATA QUALITY REPORT — {self.dataset_name.upper()}")
        logger.info(f"Shape: {self.shape[0]} x {self.shape[1]}")
        logger.info(f"{'=' * 60}")

        for result in self.results:
            msg = str(result)
            if result.passed:
                logger.success(msg)
            elif result.severity == "warning":
                logger.warning(msg)
            else:
                logger.error(msg)

        overall = "PASSED" if self.passed else "FAILED"
        logger.info(
            f"\nOverall: {overall} | Errors: {len(self.errors)} | Warnings: {len(self.warnings)}"
        )


class DataValidator:
    def __init__(self, df: pd.DataFrame, dataset_name: str = "dataset") -> None:
        self.df = df.copy()
        self.dataset_name = dataset_name
        self._cfg = get_thresholds()["data_quality"]

        self._report = DataValidationReport(dataset_name, df.shape)

    def _add(
        self,
        check_name: str,
        column: str | None,
        passed: bool,
        detail: str,
        severity: str = "error",
    ) -> None:
        self._report.results.append(
            DataCheckResult(check_name, column, passed, detail, severity)
        )

    def check_minimum_rows(self) -> DataValidator:
        min_rows = self._cfg.get("min_row_count", 1000)
        actual = len(self.df)

        self._add(
            "min_row_count",
            None,
            actual >= min_rows,
            f"{actual} rows (min: {min_rows})",
        )
        return self

    def check_null_ratios(self) -> DataValidator:
        max_null = self._cfg.get("max_null_ratio", 0.05)

        for col in self.df.columns:
            ratio = self.df[col].isnull().mean()
            passed = ratio <= max_null

            severity = "error" if ratio > 0.2 else "warning"

            self._add(
                "null_ratio",
                col,
                passed,
                f"{ratio:.2%} nulls (max: {max_null:.0%})",
                severity,
            )

        return self

    def check_duplicates(self) -> DataValidator:
        max_dup = self._cfg.get("max_duplicate_ratio", 0.02)
        ratio = self.df.duplicated().mean()

        self._add(
            "duplicate_ratio",
            None,
            ratio <= max_dup,
            f"{ratio:.2%} duplicates (max: {max_dup:.0%})",
        )
        return self

    def check_no_infinite_values(self) -> DataValidator:
        for col in self.df.select_dtypes(include="number").columns:
            has_inf = np.isinf(self.df[col].dropna()).any()

            self._add(
                "no_infinite",
                col,
                not has_inf,
                "OK" if not has_inf else "contains inf",
            )

        return self

    # 🔥 MÉTODO 1
    def check_data_types(
        self,
        expected_types: dict[str, str] | None = None,
    ) -> DataValidator:
        if not expected_types:
            return self

        for col, expected in expected_types.items():
            if col not in self.df:
                self._add("dtype", col, False, "missing", "warning")
                continue

            actual = str(self.df[col].dtype)

            self._add(
                "dtype",
                col,
                actual == expected,
                f"{actual} vs {expected}",
            )

        return self

    # 🔥 MÉTODO 2
    def check_class_balance(
        self,
        target_col: str,
        max_imbalance_ratio: float = 10.0,
    ) -> DataValidator:
        if target_col not in self.df:
            return self

        counts = self.df[target_col].value_counts()

        if len(counts) < 2:
            return self

        ratio = counts.iloc[0] / max(counts.iloc[-1], 1)

        self._add(
            "class_balance",
            target_col,
            ratio <= max_imbalance_ratio,
            f"{ratio:.2f}x imbalance",
            "warning",
        )

        return self

    def run_all_checks(
        self,
        target_col: str | None = None,
        expected_types: dict[str, str] | None = None,
    ) -> DataValidationReport:
        self.check_minimum_rows().check_null_ratios().check_duplicates().check_no_infinite_values()

        if expected_types:
            self.check_data_types(expected_types)

        if target_col:
            self.check_class_balance(target_col)

        self._report.print_report()
        return self._report