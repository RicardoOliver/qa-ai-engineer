"""
pytest Plugin — ML Quality Gate
Adds custom assertions, hooks, and auto-reporting for ML test suites.

Auto-loaded via conftest.py — no manual registration needed.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import pytest
from rich.console import Console
from rich.table import Table
from rich import box

console = Console()

# ── Session-level metrics accumulator ─────────────────────────────────────────
_session_results: dict[str, Any] = {
    "start_time": None,
    "end_time": None,
    "suites": {},
    "total_passed": 0,
    "total_failed": 0,
    "total_skipped": 0,
}


# ── Hooks ──────────────────────────────────────────────────────────────────────

def pytest_sessionstart(session: pytest.Session) -> None:
    _session_results["start_time"] = time.time()
    console.print("\n[bold cyan]━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[/bold cyan]")
    console.print("[bold cyan]  🧪 ML Quality Gate — Test Session Starting[/bold cyan]")
    console.print("[bold cyan]━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[/bold cyan]\n")


def pytest_runtest_logreport(report: pytest.TestReport) -> None:
    if report.when != "call":
        return
    suite = report.nodeid.split("/")[1] if "/" in report.nodeid else "other"
    if suite not in _session_results["suites"]:
        _session_results["suites"][suite] = {"passed": 0, "failed": 0, "skipped": 0}
    if report.passed:
        _session_results["suites"][suite]["passed"] += 1
        _session_results["total_passed"] += 1
    elif report.failed:
        _session_results["suites"][suite]["failed"] += 1
        _session_results["total_failed"] += 1
    elif report.skipped:
        _session_results["suites"][suite]["skipped"] += 1
        _session_results["total_skipped"] += 1


def pytest_sessionfinish(session: pytest.Session, exitstatus: int) -> None:
    _session_results["end_time"] = time.time()
    duration = _session_results["end_time"] - _session_results["start_time"]

    passed  = _session_results["total_passed"]
    failed  = _session_results["total_failed"]
    skipped = _session_results["total_skipped"]
    total   = passed + failed + skipped

    # Rich summary table
    table = Table(
        title="ML Quality Gate — Session Summary",
        box=box.ROUNDED,
        border_style="cyan",
        show_footer=True,
    )
    table.add_column("Suite",   style="cyan",   footer="TOTAL")
    table.add_column("Passed",  style="green",  justify="right", footer=str(passed))
    table.add_column("Failed",  style="red",    justify="right", footer=str(failed))
    table.add_column("Skipped", style="yellow", justify="right", footer=str(skipped))
    table.add_column("Status",  justify="center")

    for suite, counts in _session_results["suites"].items():
        status = "[green]✅ PASS[/green]" if counts["failed"] == 0 else "[red]❌ FAIL[/red]"
        table.add_row(
            suite,
            str(counts["passed"]),
            str(counts["failed"]),
            str(counts["skipped"]),
            status,
        )

    console.print()
    console.print(table)

    gate_status = "[bold green]✅ QUALITY GATE PASSED[/bold green]" if failed == 0 else "[bold red]❌ QUALITY GATE FAILED[/bold red]"
    console.print(f"\n  {gate_status}")
    console.print(f"  Duration: [dim]{duration:.2f}s[/dim] | Tests: [dim]{total}[/dim]\n")

    # Persist JSON summary
    Path("reports/json").mkdir(parents=True, exist_ok=True)
    summary = {
        "gate_passed": failed == 0,
        "duration_seconds": round(duration, 2),
        "total": total,
        "passed": passed,
        "failed": failed,
        "skipped": skipped,
        "suites": _session_results["suites"],
    }
    Path("reports/json/session_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )


# ── Custom Assertions ──────────────────────────────────────────────────────────

class MLAssert:
    """
    Collection of domain-specific ML assertions for use in tests.

    Example:
        from tests.plugin import MLAssert
        MLAssert.metric_above("f1_score", f1, threshold=0.82)
        MLAssert.no_drift(drift_share, max_share=0.3)
        MLAssert.confidence_calibrated(y_prob, confidence_threshold=0.9, min_accuracy=0.75)
    """

    @staticmethod
    def metric_above(name: str, value: float, threshold: float) -> None:
        assert value >= threshold, (
            f"[MLAssert] '{name}' = {value:.4f} is below threshold {threshold:.4f}. "
            f"Gap: {threshold - value:.4f}"
        )

    @staticmethod
    def metric_below(name: str, value: float, maximum: float) -> None:
        assert value <= maximum, (
            f"[MLAssert] '{name}' = {value:.4f} exceeds maximum {maximum:.4f}. "
            f"Excess: {value - maximum:.4f}"
        )

    @staticmethod
    def no_drift(drift_share: float, max_share: float = 0.3) -> None:
        assert drift_share < max_share, (
            f"[MLAssert] Dataset drift detected: {drift_share:.0%} features drifted "
            f"(max allowed: {max_share:.0%})"
        )

    @staticmethod
    def no_bias(disparity: float, feature: str, metric: str, max_disparity: float = 0.05) -> None:
        assert disparity <= max_disparity, (
            f"[MLAssert] Fairness violation on '{feature}' ({metric}): "
            f"disparity={disparity:.4f} > max={max_disparity:.4f}"
        )

    @staticmethod
    def confidence_calibrated(
        y_prob: Any,
        y_true: Any,
        y_pred: Any,
        confidence_threshold: float = 0.9,
        min_accuracy: float = 0.75,
    ) -> None:
        import numpy as np
        y_prob = np.asarray(y_prob)
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)
        mask = y_prob > confidence_threshold
        if mask.sum() < 5:
            return  # Not enough high-confidence samples
        acc = (y_pred[mask] == y_true[mask]).mean()
        assert acc >= min_accuracy, (
            f"[MLAssert] High-confidence predictions (p>{confidence_threshold}) "
            f"have accuracy {acc:.2%} < {min_accuracy:.2%}. Model is poorly calibrated."
        )

    @staticmethod
    def response_time_ok(elapsed_ms: float, max_ms: float = 500) -> None:
        assert elapsed_ms < max_ms, (
            f"[MLAssert] Inference latency {elapsed_ms:.0f}ms exceeds SLO of {max_ms:.0f}ms"
        )

    @staticmethod
    def api_schema_valid(response_json: dict, required_fields: list[str]) -> None:
        missing = [f for f in required_fields if f not in response_json]
        assert not missing, (
            f"[MLAssert] API response missing required fields: {missing}"
        )
