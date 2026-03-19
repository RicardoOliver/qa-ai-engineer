"""
HTML Reporter
Generates professional standalone HTML quality report from validation results.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from loguru import logger


REPORT_CSS = """
<style>
  :root {
    --green:#22c55e;--red:#ef4444;--yellow:#f59e0b;--blue:#3b82f6;
    --bg:#0f172a;--card:#1e293b;--text:#e2e8f0;--muted:#94a3b8;--border:#334155;
  }
  *{box-sizing:border-box;margin:0;padding:0}
  body{font-family:'Segoe UI',system-ui,sans-serif;background:var(--bg);color:var(--text);padding:2rem;line-height:1.6}
  h1{font-size:1.8rem;font-weight:700;margin-bottom:.25rem}
  h2{font-size:1rem;font-weight:600;margin-bottom:1rem;color:var(--muted);text-transform:uppercase;letter-spacing:.05em}
  .subtitle{color:var(--muted);margin-bottom:2rem;font-size:.9rem}
  .grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(240px,1fr));gap:1.25rem;margin-bottom:2rem}
  .card{background:var(--card);border:1px solid var(--border);border-radius:12px;padding:1.5rem}
  .stat{font-size:2.5rem;font-weight:700;line-height:1}
  .stat.pass{color:var(--green)}.stat.fail{color:var(--red)}.stat.warn{color:var(--yellow)}
  .badge{display:inline-flex;align-items:center;gap:.4rem;padding:.3rem .8rem;border-radius:999px;font-size:.82rem;font-weight:600}
  .badge.pass{background:rgba(34,197,94,.15);color:var(--green);border:1px solid rgba(34,197,94,.3)}
  .badge.fail{background:rgba(239,68,68,.15);color:var(--red);border:1px solid rgba(239,68,68,.3)}
  .section{margin-bottom:2.5rem}
  .section-title{font-size:1.05rem;font-weight:600;padding-bottom:.5rem;border-bottom:2px solid var(--border);margin-bottom:1rem}
  table{width:100%;border-collapse:collapse;background:var(--card);border-radius:12px;overflow:hidden;font-size:.88rem}
  th{background:var(--border);padding:.7rem 1rem;text-align:left;font-size:.78rem;text-transform:uppercase;color:var(--muted);letter-spacing:.05em}
  td{padding:.7rem 1rem;border-bottom:1px solid var(--border)}
  tr:last-child td{border-bottom:none}
  .footer{margin-top:3rem;color:var(--muted);font-size:.78rem;text-align:center;padding-top:1rem;border-top:1px solid var(--border)}
  .pass-icon::before{content:"✅ "}
  .fail-icon::before{content:"❌ "}
</style>
"""


@dataclass
class HTMLReporter:
    """Generates a single-file HTML quality dashboard from validation results."""

    model_name: str = "model"
    output_path: str = "reports/html/quality_report.html"
    _sections: list[str] = field(default_factory=list)

    def _stat_card(self, title: str, value: str, css_class: str = "") -> str:
        return (
            f'<div class="card"><h2>{title}</h2>'
            f'<div class="stat {css_class}">{value}</div></div>'
        )

    def _table(self, headers: list[str], rows: list[list[str]]) -> str:
        ths = "".join(f"<th>{h}</th>" for h in headers)
        trs = "".join(
            "<tr>" + "".join(f"<td>{c}</td>" for c in row) + "</tr>"
            for row in rows
        )
        return f"<table><thead><tr>{ths}</tr></thead><tbody>{trs}</tbody></table>"

    def add_model_metrics(self, results: list[dict[str, Any]]) -> "HTMLReporter":
        rows = [
            [
                r["name"],
                f"{r['value']:.4f}",
                f"≥ {r['threshold']:.4f}",
                f'<span class="badge {"pass" if r["passed"] else "fail"}">'
                f'{"✅ PASS" if r["passed"] else "❌ FAIL"}</span>',
            ]
            for r in results
        ]
        table = self._table(["Metric", "Value", "Threshold", "Status"], rows)
        self._sections.append(
            f'<div class="section">'
            f'<div class="section-title">📊 Model Metrics</div>{table}</div>'
        )
        return self

    def add_fairness_results(self, results: list[dict[str, Any]]) -> "HTMLReporter":
        rows = [
            [
                r["metric_name"],
                r["sensitive_feature"],
                f"{r['disparity']:.4f}",
                f"≤ {r['threshold']:.4f}",
                f'<span class="badge {"pass" if r["passed"] else "fail"}">'
                f'{"✅ PASS" if r["passed"] else "❌ FAIL"}</span>',
            ]
            for r in results
        ]
        table = self._table(["Metric", "Feature", "Disparity", "Max Allowed", "Status"], rows)
        self._sections.append(
            f'<div class="section">'
            f'<div class="section-title">⚖️ Fairness & Bias</div>{table}</div>'
        )
        return self

    def add_drift_results(self, results: list[dict[str, Any]], dataset_drift: bool) -> "HTMLReporter":
        rows = [
            [
                r["feature"],
                r["method"],
                f"{r['statistic']:.4f}",
                f'<span class="badge {"fail" if r["drift_detected"] else "pass"}">'
                f'{"🚨 DRIFT" if r["drift_detected"] else "✅ STABLE"}</span>',
            ]
            for r in results
        ]
        table = self._table(["Feature", "Method", "Statistic", "Status"], rows)
        overall = (
            '<span class="badge fail">🚨 Dataset Drift Detected</span>'
            if dataset_drift
            else '<span class="badge pass">✅ No Dataset Drift</span>'
        )
        self._sections.append(
            f'<div class="section">'
            f'<div class="section-title">📉 Data Drift</div>'
            f"<p style='margin-bottom:1rem'>Overall: {overall}</p>"
            f"{table}</div>"
        )
        return self

    def render(
        self,
        passed: bool,
        total_checks: int,
        passed_checks: int,
        failed_checks: int,
    ) -> str:
        status_card = (
            '<span class="badge pass" style="font-size:1rem">✅ All Gates Passed</span>'
            if passed
            else '<span class="badge fail" style="font-size:1rem">❌ Quality Gate Failed</span>'
        )
        stats = f"""
        <div class="grid">
          <div class="card"><h2>Overall Status</h2>{status_card}</div>
          {self._stat_card("Total Checks", str(total_checks))}
          {self._stat_card("Passed", str(passed_checks), "pass")}
          {self._stat_card("Failed", str(failed_checks), "fail" if failed_checks > 0 else "pass")}
        </div>
        """
        body = stats + "\n".join(self._sections)
        return (
            "<!DOCTYPE html><html lang='en'><head>"
            "<meta charset='UTF-8'>"
            "<meta name='viewport' content='width=device-width,initial-scale=1'>"
            f"<title>ML Quality Gate — {self.model_name}</title>"
            f"{REPORT_CSS}"
            "</head><body>"
            f"<h1>🧪 ML Quality Gate Dashboard</h1>"
            f"<p class='subtitle'>Model: <strong>{self.model_name}</strong> &nbsp;|&nbsp; "
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>"
            f"{body}"
            "<div class='footer'>ML Quality Gate Framework — QA Engineering</div>"
            "</body></html>"
        )

    def save(
        self,
        passed: bool,
        total_checks: int,
        passed_checks: int,
        failed_checks: int,
    ) -> Path:
        html = self.render(passed, total_checks, passed_checks, failed_checks)
        path = Path(self.output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(html, encoding="utf-8")
        logger.success(f"HTML report saved: {path}")
        return path
