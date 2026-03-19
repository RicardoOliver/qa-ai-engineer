"""
CloudWatch Reporter
Publica métricas do ML Quality Gate no Amazon CloudWatch.
Permite criar dashboards, alarmes e detectar regressões automaticamente.

Uso:
  from examples.aws.cloudwatch.cloudwatch_reporter import CloudWatchReporter

  reporter = CloudWatchReporter(model_name="credit-risk-v2")
  reporter.publish_model_metrics({"f1_score": 0.87, "accuracy": 0.91})
  reporter.publish_drift_metrics(drift_share=0.12, drifted_features=["income"])
  reporter.publish_fairness_metrics(violations=0, max_disparity=0.03)
  reporter.create_quality_dashboard()
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from typing import Any

import boto3
from loguru import logger


NAMESPACE  = os.getenv("CLOUDWATCH_NAMESPACE", "MLQualityGate")
REGION     = os.getenv("AWS_REGION", "us-east-1")


class CloudWatchReporter:
    """
    Publica métricas de qualidade do modelo no CloudWatch.
    Permite monitoramento contínuo, alertas e dashboards no console AWS.
    """

    def __init__(self, model_name: str, namespace: str = NAMESPACE) -> None:
        self.model_name = model_name
        self.namespace = namespace
        self._cw = boto3.client("cloudwatch", region_name=REGION)
        self._base_dims = [{"Name": "ModelName", "Value": model_name}]

    def _put(self, metric_data: list[dict[str, Any]]) -> None:
        """Envia métricas para o CloudWatch."""
        try:
            self._cw.put_metric_data(
                Namespace=self.namespace,
                MetricData=metric_data,
            )
            logger.success(f"✅ {len(metric_data)} métrica(s) publicadas no CloudWatch.")
        except Exception as e:
            logger.error(f"❌ Erro ao publicar no CloudWatch: {e}")
            raise

    def _metric(
        self,
        name: str,
        value: float,
        unit: str = "None",
        extra_dims: list[dict] | None = None,
    ) -> dict[str, Any]:
        dims = self._base_dims + (extra_dims or [])
        return {
            "MetricName": name,
            "Value": value,
            "Unit": unit,
            "Timestamp": datetime.now(tz=timezone.utc).isoformat(),
            "Dimensions": dims,
        }

    # ── Publicadores por categoria ─────────────────────────────────────────────

    def publish_model_metrics(self, metrics: dict[str, float]) -> None:
        """
        Publica métricas de qualidade do modelo (F1, Accuracy, ROC-AUC etc.).

        Args:
            metrics: dict com nome → valor. Ex: {"f1_score": 0.87, "roc_auc": 0.93}
        """
        metric_map = {
            "f1_score":  ("F1Score",  "None"),
            "accuracy":  ("Accuracy", "None"),
            "precision": ("Precision","None"),
            "recall":    ("Recall",   "None"),
            "roc_auc":   ("RocAuc",   "None"),
        }
        data = [
            self._metric(
                metric_map.get(k, (k, "None"))[0],
                v,
                metric_map.get(k, (k, "None"))[1],
            )
            for k, v in metrics.items()
        ]
        self._put(data)

    def publish_quality_gate_result(self, passed: bool, failed_checks: int = 0) -> None:
        """Publica resultado do quality gate (0 = falhou, 1 = passou)."""
        self._put([
            self._metric("QualityGatePassed", 1.0 if passed else 0.0),
            self._metric("FailedChecks", float(failed_checks), "Count"),
        ])

    def publish_drift_metrics(
        self,
        drift_share: float,
        drifted_features: list[str] | None = None,
        dataset_drift: bool = False,
    ) -> None:
        """
        Publica métricas de drift de dados.
        Útil para criar alarmes quando drift excede um threshold.
        """
        data = [
            self._metric("DriftShare", drift_share),
            self._metric("DriftedFeatureCount", float(len(drifted_features or [])), "Count"),
            self._metric("DatasetDriftDetected", 1.0 if dataset_drift else 0.0),
        ]

        # Drift por feature individual
        for feature in (drifted_features or []):
            data.append(self._metric(
                "FeatureDrift", 1.0,
                extra_dims=[{"Name": "Feature", "Value": feature}]
            ))

        self._put(data)

    def publish_fairness_metrics(
        self,
        violations: int,
        max_disparity: float,
        feature: str | None = None,
    ) -> None:
        """Publica métricas de fairness e viés."""
        extra = [{"Name": "SensitiveFeature", "Value": feature}] if feature else None
        self._put([
            self._metric("FairnessViolations", float(violations), "Count", extra),
            self._metric("MaxDisparity", max_disparity, "None", extra),
            self._metric("FairnessGatePassed", 0.0 if violations > 0 else 1.0),
        ])

    def publish_inference_latency(self, latency_p95_ms: float, latency_p99_ms: float) -> None:
        """Publica métricas de latência de inferência."""
        self._put([
            self._metric("InferenceLatencyP95", latency_p95_ms, "Milliseconds"),
            self._metric("InferenceLatencyP99", latency_p99_ms, "Milliseconds"),
        ])

    # ── Alarmes automáticos ────────────────────────────────────────────────────

    def create_quality_alarms(
        self,
        sns_topic_arn: str,
        f1_threshold: float = 0.82,
        drift_share_threshold: float = 0.30,
        latency_p95_ms: float = 300.0,
    ) -> None:
        """
        Cria alarmes no CloudWatch para métricas críticas.
        Notifica via SNS (pode ser integrado com Slack, email, PagerDuty etc.)
        """
        alarms = [
            {
                "AlarmName": f"{self.model_name}-f1-score-degraded",
                "AlarmDescription": f"F1-Score do modelo {self.model_name} caiu abaixo de {f1_threshold}",
                "MetricName": "F1Score",
                "Namespace": self.namespace,
                "Statistic": "Average",
                "Period": 3600,       # 1 hora
                "EvaluationPeriods": 1,
                "Threshold": f1_threshold,
                "ComparisonOperator": "LessThanThreshold",
                "TreatMissingData": "breaching",
                "AlarmActions": [sns_topic_arn],
                "Dimensions": self._base_dims,
            },
            {
                "AlarmName": f"{self.model_name}-drift-detected",
                "AlarmDescription": f"Data drift detectado para o modelo {self.model_name}",
                "MetricName": "DatasetDriftDetected",
                "Namespace": self.namespace,
                "Statistic": "Maximum",
                "Period": 86400,      # 24 horas
                "EvaluationPeriods": 1,
                "Threshold": 0.5,
                "ComparisonOperator": "GreaterThanOrEqualToThreshold",
                "TreatMissingData": "ignore",
                "AlarmActions": [sns_topic_arn],
                "Dimensions": self._base_dims,
            },
            {
                "AlarmName": f"{self.model_name}-latency-p95-slo",
                "AlarmDescription": f"Latência P95 excedeu {latency_p95_ms}ms para {self.model_name}",
                "MetricName": "InferenceLatencyP95",
                "Namespace": self.namespace,
                "Statistic": "Average",
                "Period": 300,        # 5 minutos
                "EvaluationPeriods": 3,
                "Threshold": latency_p95_ms,
                "ComparisonOperator": "GreaterThanThreshold",
                "TreatMissingData": "ignore",
                "AlarmActions": [sns_topic_arn],
                "Dimensions": self._base_dims,
            },
        ]

        for alarm in alarms:
            try:
                self._cw.put_metric_alarm(**alarm)
                logger.success(f"⏰ Alarme criado: {alarm['AlarmName']}")
            except Exception as e:
                logger.error(f"Erro ao criar alarme '{alarm['AlarmName']}': {e}")

    def create_quality_dashboard(self) -> str:
        """Cria um dashboard no CloudWatch com as métricas principais."""
        dashboard_name = f"MLQualityGate-{self.model_name}"
        dashboard_body = {
            "widgets": [
                {
                    "type": "metric",
                    "properties": {
                        "title": "Métricas do Modelo",
                        "metrics": [
                            [self.namespace, "F1Score",  "ModelName", self.model_name],
                            [self.namespace, "Accuracy", "ModelName", self.model_name],
                            [self.namespace, "RocAuc",   "ModelName", self.model_name],
                        ],
                        "period": 3600,
                        "stat": "Average",
                        "view": "timeSeries",
                    },
                },
                {
                    "type": "metric",
                    "properties": {
                        "title": "Data Drift",
                        "metrics": [
                            [self.namespace, "DriftShare",           "ModelName", self.model_name],
                            [self.namespace, "DatasetDriftDetected", "ModelName", self.model_name],
                        ],
                        "period": 86400,
                        "stat": "Maximum",
                        "view": "timeSeries",
                    },
                },
                {
                    "type": "metric",
                    "properties": {
                        "title": "Latência de Inferência",
                        "metrics": [
                            [self.namespace, "InferenceLatencyP95", "ModelName", self.model_name],
                            [self.namespace, "InferenceLatencyP99", "ModelName", self.model_name],
                        ],
                        "period": 300,
                        "stat": "Average",
                        "view": "timeSeries",
                    },
                },
                {
                    "type": "metric",
                    "properties": {
                        "title": "Quality Gate",
                        "metrics": [
                            [self.namespace, "QualityGatePassed", "ModelName", self.model_name],
                            [self.namespace, "FairnessViolations", "ModelName", self.model_name],
                        ],
                        "period": 86400,
                        "stat": "Sum",
                        "view": "singleValue",
                    },
                },
            ]
        }

        self._cw.put_dashboard(
            DashboardName=dashboard_name,
            DashboardBody=json.dumps(dashboard_body),
        )
        url = f"https://{REGION}.console.aws.amazon.com/cloudwatch/home?region={REGION}#dashboards:name={dashboard_name}"
        logger.success(f"📊 Dashboard criado: {url}")
        return url


# ── Uso standalone ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    reporter = CloudWatchReporter(model_name="credit-risk-v2")

    # Publicar métricas de exemplo
    reporter.publish_model_metrics({
        "f1_score": 0.871,
        "accuracy": 0.912,
        "roc_auc": 0.934,
    })
    reporter.publish_quality_gate_result(passed=True, failed_checks=0)
    reporter.publish_drift_metrics(drift_share=0.08, dataset_drift=False)
    reporter.publish_fairness_metrics(violations=0, max_disparity=0.032)
    reporter.publish_inference_latency(latency_p95_ms=187.4, latency_p99_ms=312.0)
    reporter.create_quality_dashboard()
