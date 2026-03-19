"""
Slack Notifier — Sends quality gate results to Slack.
Supports: pass/fail summaries, drift alerts, fairness violations.
"""

from __future__ import annotations

import os
from typing import Any

import requests
from loguru import logger


class SlackNotifier:
    """
    Posts QA results to a Slack channel via Incoming Webhooks.

    Usage:
        notifier = SlackNotifier()
        notifier.notify_quality_gate(report_summary, passed=True)
    """

    def __init__(self, webhook_url: str | None = None) -> None:
        self.webhook_url = webhook_url or os.getenv("SLACK_WEBHOOK_URL", "")
        if not self.webhook_url:
            logger.warning("SLACK_WEBHOOK_URL not set — notifications disabled.")

    def _post(self, payload: dict[str, Any]) -> bool:
        if not self.webhook_url:
            return False
        try:
            response = requests.post(self.webhook_url, json=payload, timeout=10)
            response.raise_for_status()
            return True
        except Exception as e:
            logger.error(f"Slack notification failed: {e}")
            return False

    def notify_quality_gate(
        self,
        model_name: str,
        summary: dict[str, Any],
        passed: bool,
        run_url: str = "",
    ) -> bool:
        color = "#36a64f" if passed else "#ff0000"
        status_emoji = "✅" if passed else "❌"
        status_text = "PASSED" if passed else "FAILED"

        fields = [
            {"title": "Model", "value": model_name, "short": True},
            {"title": "Status", "value": f"{status_emoji} {status_text}", "short": True},
        ]

        if "metrics" in summary:
            for metric, data in summary["metrics"].items():
                fields.append({
                    "title": metric.replace("_", " ").title(),
                    "value": f"{data['value']:.4f} {'✅' if data['passed'] else '❌'}",
                    "short": True,
                })

        payload = {
            "attachments": [
                {
                    "color": color,
                    "title": f"🧪 ML Quality Gate — {status_text}",
                    "title_link": run_url,
                    "fields": fields,
                    "footer": "ML Quality Gate Framework",
                }
            ]
        }
        return self._post(payload)

    def notify_drift_detected(self, drifted_features: list[str], drift_share: float) -> bool:
        payload = {
            "text": (
                f"🚨 *Data Drift Detected!*\n"
                f"• Drift share: {drift_share:.0%}\n"
                f"• Drifted features: `{', '.join(drifted_features)}`\n"
                f"_Consider retraining the model._"
            )
        }
        return self._post(payload)

    def notify_fairness_violation(self, violations: list[dict[str, Any]]) -> bool:
        lines = [f"• `{v['metric']}` on `{v['feature']}` — disparity={v['disparity']:.4f}" for v in violations]
        payload = {
            "text": (
                f"⚖️ *Fairness Violations Detected!*\n" + "\n".join(lines)
            )
        }
        return self._post(payload)
