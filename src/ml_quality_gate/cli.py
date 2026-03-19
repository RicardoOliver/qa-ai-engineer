"""
ML Quality Gate CLI
Run quality gates from the command line with rich output.

Usage:
    ml-quality-gate run --suite all
    ml-quality-gate run --suite model
    ml-quality-gate run --suite data
    ml-quality-gate run --suite fairness
    ml-quality-gate report
    ml-quality-gate check-drift
"""

from __future__ import annotations

import sys
import argparse
from pathlib import Path

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich import box

console = Console()


def print_banner() -> None:
    console.print(Panel.fit(
        "[bold cyan]🧪 ML Quality Gate Framework[/bold cyan]\n"
        "[dim]Enterprise QA for Machine Learning — Big Tech Standards[/dim]",
        border_style="cyan",
        padding=(1, 4),
    ))


def cmd_run(suite: str, verbose: bool) -> int:
    """Run quality gate suites."""
    import subprocess

    suite_map = {
        "all":        ["tests/unit/", "tests/fairness/", "tests/e2e/"],
        "model":      ["tests/unit/test_model_metrics.py"],
        "data":       ["tests/unit/test_data_quality.py"],
        "fairness":   ["tests/fairness/"],
        "integration":["tests/integration/"],
        "e2e":        ["tests/e2e/"],
    }

    if suite not in suite_map:
        console.print(f"[red]Unknown suite: {suite}. Choose from: {list(suite_map.keys())}[/red]")
        return 1

    targets = suite_map[suite]
    verbosity = "-v" if verbose else "-q"
    cmd = [sys.executable, "-m", "pytest"] + targets + [verbosity, "--tb=short", "--no-header"]

    console.print(f"\n[bold yellow]▶ Running suite:[/bold yellow] [cyan]{suite}[/cyan]\n")
    result = subprocess.run(cmd)
    return result.returncode


def cmd_report() -> int:
    """Generate HTML quality report."""
    console.print("\n[bold yellow]📊 Generating Quality Report...[/bold yellow]")
    try:
        sys.path.insert(0, str(Path(__file__).parents[1]))
        from scripts.generate_report import generate_report
        generate_report()
        console.print("[green]✅ Report generated at reports/html/quality_report.html[/green]")
        return 0
    except Exception as e:
        console.print(f"[red]❌ Report generation failed: {e}[/red]")
        return 1


def cmd_check_drift(reference: str, current: str) -> int:
    """Run drift detection between two datasets."""
    import pandas as pd
    sys.path.insert(0, str(Path(__file__).parents[2] / "src"))
    from ml_quality_gate.validators.drift_detector import DriftDetector

    console.print(f"\n[bold yellow]🔍 Checking drift...[/bold yellow]")
    console.print(f"  Reference: [cyan]{reference}[/cyan]")
    console.print(f"  Current:   [cyan]{current}[/cyan]\n")

    try:
        ref_df = pd.read_parquet(reference) if reference.endswith(".parquet") else pd.read_csv(reference)
        cur_df = pd.read_parquet(current) if current.endswith(".parquet") else pd.read_csv(current)

        numeric_cols = ref_df.select_dtypes(include="number").columns.tolist()
        detector = DriftDetector(ref_df[numeric_cols], cur_df[numeric_cols])
        report = detector.detect_all()

        table = Table(title="Drift Detection Results", box=box.ROUNDED, border_style="cyan")
        table.add_column("Feature", style="cyan")
        table.add_column("Method", style="dim")
        table.add_column("Statistic", justify="right")
        table.add_column("Drift", justify="center")

        for r in report.results:
            drift_icon = "[red]🚨 YES[/red]" if r.drift_detected else "[green]✅ NO[/green]"
            table.add_row(r.feature, r.method, f"{r.statistic:.4f}", drift_icon)

        console.print(table)

        if report.dataset_drift:
            console.print(f"\n[bold red]⚠️  Dataset drift detected! ({report.drift_share:.0%} features drifted)[/bold red]")
            return 1
        else:
            console.print(f"\n[bold green]✅ No significant dataset drift detected.[/bold green]")
            return 0

    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        return 1


def cmd_status() -> int:
    """Show current framework configuration."""
    sys.path.insert(0, str(Path(__file__).parents[2] / "src"))
    from ml_quality_gate.utils.config_loader import get_thresholds, get_model_config

    thresholds = get_thresholds()
    model_cfg = get_model_config()

    table = Table(title="Model Metric Thresholds", box=box.ROUNDED, border_style="cyan")
    table.add_column("Metric", style="cyan")
    table.add_column("Minimum", justify="right", style="green")
    table.add_column("Critical", justify="right", style="red")

    for metric, cfg in thresholds.get("model", {}).items():
        if isinstance(cfg, dict):
            table.add_row(metric, str(cfg.get("minimum", "-")), str(cfg.get("critical", "-")))

    console.print(f"\n[bold]Model:[/bold] {model_cfg['model']['name']} v{model_cfg['model']['version']}")
    console.print(table)
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(
        description="ML Quality Gate — Enterprise QA Framework CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command")

    # run
    run_p = subparsers.add_parser("run", help="Run a test suite")
    run_p.add_argument("--suite", default="all",
        choices=["all", "model", "data", "fairness", "integration", "e2e"],
        help="Which suite to run (default: all)")
    run_p.add_argument("--verbose", "-v", action="store_true")

    # report
    subparsers.add_parser("report", help="Generate HTML quality report")

    # check-drift
    drift_p = subparsers.add_parser("check-drift", help="Run drift detection between two datasets")
    drift_p.add_argument("--reference", required=True, help="Path to reference dataset (.csv or .parquet)")
    drift_p.add_argument("--current",   required=True, help="Path to current dataset (.csv or .parquet)")

    # status
    subparsers.add_parser("status", help="Show current thresholds and config")

    args = parser.parse_args()
    print_banner()

    if args.command == "run":
        sys.exit(cmd_run(args.suite, args.verbose))
    elif args.command == "report":
        sys.exit(cmd_report())
    elif args.command == "check-drift":
        sys.exit(cmd_check_drift(args.reference, args.current))
    elif args.command == "status":
        sys.exit(cmd_status())
    else:
        parser.print_help()
        sys.exit(0)


if __name__ == "__main__":
    main()
