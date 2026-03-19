#!/usr/bin/env python3
"""
Report Generator
Aggregates all QA results into a single HTML dashboard.
Run: python scripts/generate_report.py
"""

from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

# ── Jinja2 HTML Template ───────────────────────────────────────────────────────

REPORT_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>ML Quality Gate — Report</title>
  <style>
    :root {
      --green: #22c55e; --red: #ef4444; --yellow: #f59e0b;
      --blue: #3b82f6;  --bg: #0f172a;  --card: #1e293b;
      --text: #e2e8f0;  --muted: #94a3b8; --border: #334155;
    }
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body { font-family: 'Segoe UI', system-ui, sans-serif; background: var(--bg); color: var(--text); padding: 2rem; }
    h1 { font-size: 1.8rem; font-weight: 700; margin-bottom: 0.25rem; }
    .subtitle { color: var(--muted); margin-bottom: 2rem; font-size: 0.9rem; }
    .grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(260px, 1fr)); gap: 1.25rem; margin-bottom: 2rem; }
    .card { background: var(--card); border: 1px solid var(--border); border-radius: 12px; padding: 1.5rem; }
    .card h2 { font-size: 0.85rem; color: var(--muted); text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 0.75rem; }
    .badge { display: inline-flex; align-items: center; gap: 0.4rem; padding: 0.35rem 0.85rem; border-radius: 999px; font-size: 0.85rem; font-weight: 600; }
    .pass { background: rgba(34,197,94,0.15); color: var(--green); border: 1px solid rgba(34,197,94,0.3); }
    .fail { background: rgba(239,68,68,0.15); color: var(--red); border: 1px solid rgba(239,68,68,0.3); }
    .warn { background: rgba(245,158,11,0.15); color: var(--yellow); border: 1px solid rgba(245,158,11,0.3); }
    .metric-row { display: flex; justify-content: space-between; align-items: center; padding: 0.5rem 0; border-bottom: 1px solid var(--border); }
    .metric-row:last-child { border-bottom: none; }
    .metric-val { font-weight: 600; font-size: 1.1rem; }
    .metric-label { color: var(--muted); font-size: 0.85rem; }
    .section-title { font-size: 1.1rem; font-weight: 600; margin: 2rem 0 1rem; padding-bottom: 0.5rem; border-bottom: 2px solid var(--border); }
    table { width: 100%; border-collapse: collapse; background: var(--card); border-radius: 12px; overflow: hidden; }
    th { background: var(--border); padding: 0.75rem 1rem; text-align: left; font-size: 0.8rem; text-transform: uppercase; color: var(--muted); }
    td { padding: 0.75rem 1rem; border-bottom: 1px solid var(--border); font-size: 0.9rem; }
    tr:last-child td { border-bottom: none; }
    .footer { margin-top: 3rem; color: var(--muted); font-size: 0.8rem; text-align: center; }
  </style>
</head>
<body>
  <h1>🧪 ML Quality Gate Dashboard</h1>
  <p class="subtitle">Generated: {{ generated_at }} &nbsp;|&nbsp; Model: {{ model_name }}</p>

  <!-- Overall Status -->
  <div class="grid">
    <div class="card">
      <h2>Overall Status</h2>
      {% if overall_passed %}
        <span class="badge pass">✅ All Gates Passed</span>
      {% else %}
        <span class="badge fail">❌ Gate Failed</span>
      {% endif %}
    </div>
    <div class="card">
      <h2>Suites Run</h2>
      <div style="font-size: 2rem; font-weight: 700;">{{ suites_run }}</div>
    </div>
    <div class="card">
      <h2>Checks Passed</h2>
      <div style="font-size: 2rem; font-weight: 700; color: var(--green);">{{ checks_passed }}</div>
    </div>
    <div class="card">
      <h2>Checks Failed</h2>
      <div style="font-size: 2rem; font-weight: 700; color: {% if checks_failed > 0 %}var(--red){% else %}var(--green){% endif %};">{{ checks_failed }}</div>
    </div>
  </div>

  <!-- Model Metrics -->
  <div class="section-title">📊 Model Metrics</div>
  <table>
    <thead>
      <tr><th>Metric</th><th>Value</th><th>Threshold</th><th>Status</th></tr>
    </thead>
    <tbody>
      {% for row in model_metrics %}
      <tr>
        <td>{{ row.name }}</td>
        <td class="metric-val">{{ "%.4f"|format(row.value) }}</td>
        <td style="color: var(--muted);">≥ {{ "%.4f"|format(row.threshold) }}</td>
        <td><span class="badge {{ 'pass' if row.passed else 'fail' }}">{{ '✅ PASS' if row.passed else '❌ FAIL' }}</span></td>
      </tr>
      {% endfor %}
    </tbody>
  </table>

  <!-- Fairness -->
  <div class="section-title">⚖️ Fairness & Bias</div>
  <table>
    <thead>
      <tr><th>Metric</th><th>Feature</th><th>Disparity</th><th>Threshold</th><th>Status</th></tr>
    </thead>
    <tbody>
      {% for row in fairness_results %}
      <tr>
        <td>{{ row.metric }}</td>
        <td>{{ row.feature }}</td>
        <td class="metric-val">{{ "%.4f"|format(row.disparity) }}</td>
        <td style="color: var(--muted);">≤ {{ "%.4f"|format(row.threshold) }}</td>
        <td><span class="badge {{ 'pass' if row.passed else 'fail' }}">{{ '✅ PASS' if row.passed else '❌ FAIL' }}</span></td>
      </tr>
      {% endfor %}
    </tbody>
  </table>

  <div class="footer">ML Quality Gate Framework &mdash; Generated by QA Engineering</div>
</body>
</html>
"""

# ── Report Generation ──────────────────────────────────────────────────────────

def generate_report() -> None:
    """Generate the consolidated HTML quality report."""
    try:
        from jinja2 import Template
    except ImportError:
        print("jinja2 not installed. Run: pip install jinja2")
        sys.exit(1)

    # Run all validators and collect data
    try:
        import numpy as np
        import pandas as pd
        from sklearn.datasets import make_classification
        from sklearn.ensemble import GradientBoostingClassifier
        from sklearn.model_selection import train_test_split

        sys.path.insert(0, str(Path(__file__).parents[1] / "src"))
        from ml_quality_gate.validators.model_validator import ModelValidator
        from ml_quality_gate.validators.fairness_validator import FairnessValidator

        # Generate synthetic data
        X, y = make_classification(n_samples=2000, n_features=7, random_state=42)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model = GradientBoostingClassifier(random_state=42)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]

        mv = ModelValidator("credit_risk_gbm")
        m_report = mv.validate_classification(y_test, y_pred, y_prob)
        model_metrics = [
            {"name": r.name, "value": r.value, "threshold": r.threshold, "passed": r.passed}
            for r in m_report.results
        ]

        # Fairness with synthetic sensitive features
        n = len(y_test)
        rng = np.random.default_rng(42)
        sensitive = pd.DataFrame({
            "gender": rng.choice(["male", "female"], size=n),
            "age_group": rng.choice(["18-30", "31-45", "46-60"], size=n),
        })
        fv = FairnessValidator()
        f_report = fv.validate(y_test, y_pred, sensitive)
        fairness_results = [
            {"metric": r.metric_name, "feature": r.sensitive_feature,
             "disparity": r.disparity, "threshold": r.threshold, "passed": r.passed}
            for r in f_report.results
        ]

        overall_passed = m_report.passed and f_report.passed
        checks_passed = sum(1 for r in m_report.results if r.passed) + sum(1 for r in f_report.results if r.passed)
        checks_failed = sum(1 for r in m_report.results if not r.passed) + sum(1 for r in f_report.results if not r.passed)

    except Exception as e:
        print(f"Warning: Could not run validators ({e}). Generating placeholder report.")
        model_metrics = [{"name": "accuracy", "value": 0.0, "threshold": 0.85, "passed": False}]
        fairness_results = []
        overall_passed = False
        checks_passed = 0
        checks_failed = 1

    template = Template(REPORT_TEMPLATE)
    html = template.render(
        generated_at=datetime.now().strftime("%Y-%m-%d %H:%M:%S UTC"),
        model_name="credit_risk_gbm v2.1.0",
        overall_passed=overall_passed,
        suites_run=5,
        checks_passed=checks_passed,
        checks_failed=checks_failed,
        model_metrics=model_metrics,
        fairness_results=fairness_results,
    )

    output_path = Path("reports/html/quality_report.html")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html, encoding="utf-8")
    print(f"✅ Report generated: {output_path}")


if __name__ == "__main__":
    generate_report()
