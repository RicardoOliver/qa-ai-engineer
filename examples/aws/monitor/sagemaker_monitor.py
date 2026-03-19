"""
SageMaker Model Monitor — Integração com ML Quality Gate
Combina o monitoramento nativo do SageMaker com os validators do projeto.

Funcionalidades:
  1. Configurar monitoramento automático no endpoint
  2. Analisar relatórios do Model Monitor
  3. Comparar resultados com o DriftDetector do projeto
  4. Criar schedule de monitoramento contínuo

Uso:
  python examples/aws/monitor/sagemaker_monitor.py \\
    --endpoint-name credit-risk-v2 \\
    --baseline-uri s3://meu-bucket/baselines/credit-risk/
"""

from __future__ import annotations

import argparse
import io
import json
import os
from typing import Any

import boto3
import pandas as pd
from loguru import logger

from examples.aws.s3.aws_data_loader import S3DataLoader


REGION     = os.getenv("AWS_REGION", "us-east-1")
ROLE_ARN   = os.getenv("SAGEMAKER_ROLE_ARN", "arn:aws:iam::123456789012:role/SageMakerRole")
S3_BUCKET  = os.getenv("S3_BUCKET", "meu-bucket-ml")


class SageMakerMonitorIntegration:
    """
    Integra o SageMaker Model Monitor com o ML Quality Gate.

    Workflow:
      1. Cria baseline de dados usando um job de processamento
      2. Configura schedule de monitoramento no endpoint
      3. Recupera e analisa os relatórios de violação
      4. Roda o DriftDetector local para comparação
    """

    def __init__(self, endpoint_name: str) -> None:
        self.endpoint_name = endpoint_name
        self.sm = boto3.client("sagemaker", region_name=REGION)
        self.s3 = boto3.client("s3", region_name=REGION)
        self.loader = S3DataLoader(bucket=S3_BUCKET, region=REGION)

    # ── Configuração do Baseline ───────────────────────────────────────────────

    def create_data_quality_baseline(
        self,
        baseline_data_uri: str,
        output_uri: str | None = None,
        instance_type: str = "ml.m5.large",
    ) -> str:
        """
        Cria o baseline de qualidade de dados para o Model Monitor.
        Deve ser executado UMA VEZ com os dados de treino/validação.

        Retorna o URI do S3 onde o baseline foi salvo.
        """
        output_uri = output_uri or f"s3://{S3_BUCKET}/model-monitor/baselines/{self.endpoint_name}/"

        logger.info(f"📊 Criando baseline de qualidade de dados...")
        logger.info(f"   Input:  {baseline_data_uri}")
        logger.info(f"   Output: {output_uri}")

        job_name = f"baseline-{self.endpoint_name}-{int(__import__('time').time())}"

        self.sm.create_processing_job(
            ProcessingJobName=job_name,
            ProcessingInputs=[{
                "InputName": "baseline_dataset_input",
                "AppManaged": False,
                "S3Input": {
                    "S3Uri": baseline_data_uri,
                    "LocalPath": "/opt/ml/processing/input/baseline",
                    "S3DataType": "S3Prefix",
                    "S3InputMode": "File",
                },
            }],
            ProcessingOutputConfig={
                "Outputs": [{
                    "OutputName": "baseline_statistics",
                    "S3Output": {
                        "S3Uri": output_uri,
                        "LocalPath": "/opt/ml/processing/output",
                        "S3UploadMode": "EndOfJob",
                    },
                }]
            },
            ProcessingResources={
                "ClusterConfig": {
                    "InstanceCount": 1,
                    "InstanceType": instance_type,
                    "VolumeSizeInGB": 20,
                }
            },
            AppSpecification={
                "ImageUri": f"156813124566.dkr.ecr.{REGION}.amazonaws.com/sagemaker-model-monitor-analyzer",
            },
            RoleArn=ROLE_ARN,
            Environment={"baseline_constraints": "suggest"},
        )

        logger.info(f"⏳ Job de baseline iniciado: {job_name}")
        waiter = self.sm.get_waiter("processing_job_completed_or_stopped")
        waiter.wait(ProcessingJobName=job_name)
        logger.success(f"✅ Baseline criado em: {output_uri}")
        return output_uri

    # ── Configuração do Schedule de Monitoramento ──────────────────────────────

    def setup_monitoring_schedule(
        self,
        baseline_uri: str,
        schedule_cron: str = "cron(0 * ? * * *)",  # A cada hora
        instance_type: str = "ml.m5.large",
    ) -> str:
        """
        Configura o schedule de monitoramento contínuo no endpoint.

        Args:
            baseline_uri: URI S3 do baseline criado anteriormente
            schedule_cron: Expressão cron para frequência do monitoramento
            instance_type: Tipo de instância para os jobs de análise

        Retorna o nome do schedule criado.
        """
        schedule_name = f"monitor-{self.endpoint_name}"
        output_uri = f"s3://{S3_BUCKET}/model-monitor/reports/{self.endpoint_name}/"

        logger.info(f"⏰ Configurando schedule de monitoramento: {schedule_name}")
        logger.info(f"   Frequência: {schedule_cron}")

        try:
            self.sm.delete_monitoring_schedule(MonitoringScheduleName=schedule_name)
            logger.info("Schedule anterior removido.")
        except self.sm.exceptions.ResourceNotFound:
            pass

        self.sm.create_monitoring_schedule(
            MonitoringScheduleName=schedule_name,
            MonitoringScheduleConfig={
                "ScheduleConfig": {"ScheduleExpression": schedule_cron},
                "MonitoringJobDefinition": {
                    "BaselineConfig": {
                        "ConstraintsResource": {"S3Uri": f"{baseline_uri}constraints.json"},
                        "StatisticsResource": {"S3Uri": f"{baseline_uri}statistics.json"},
                    },
                    "MonitoringInputs": [{
                        "EndpointInput": {
                            "EndpointName": self.endpoint_name,
                            "LocalPath": "/opt/ml/processing/input/endpoint",
                        }
                    }],
                    "MonitoringOutputConfig": {
                        "MonitoringOutputs": [{
                            "S3Output": {
                                "S3Uri": output_uri,
                                "LocalPath": "/opt/ml/processing/output",
                                "S3UploadMode": "EndOfJob",
                            }
                        }]
                    },
                    "MonitoringResources": {
                        "ClusterConfig": {
                            "InstanceCount": 1,
                            "InstanceType": instance_type,
                            "VolumeSizeInGB": 20,
                        }
                    },
                    "MonitoringAppSpecification": {
                        "ImageUri": f"156813124566.dkr.ecr.{REGION}.amazonaws.com/sagemaker-model-monitor-analyzer",
                    },
                    "RoleArn": ROLE_ARN,
                    "StoppingCondition": {"MaxRuntimeInSeconds": 1800},
                },
            },
        )

        logger.success(f"✅ Schedule '{schedule_name}' configurado com sucesso!")
        return schedule_name

    # ── Análise de Relatórios ──────────────────────────────────────────────────

    def get_latest_violations(self) -> list[dict[str, Any]]:
        """
        Recupera as violações do último job de monitoramento.
        Integra com o DriftDetector do ML Quality Gate.
        """
        schedule_name = f"monitor-{self.endpoint_name}"

        executions = self.sm.list_monitoring_executions(
            MonitoringScheduleName=schedule_name,
            SortBy="ScheduledTime",
            SortOrder="Descending",
            MaxResults=1,
        )

        if not executions["MonitoringExecutionSummaries"]:
            logger.warning("Nenhuma execução de monitoramento encontrada.")
            return []

        latest = executions["MonitoringExecutionSummaries"][0]
        status = latest["MonitoringExecutionStatus"]
        logger.info(f"Última execução: {status} em {latest['ScheduledTime']}")

        if status != "Completed":
            logger.warning(f"Job em estado: {status}")
            return []

        # Tentar ler relatório de violações do S3
        output_uri = latest.get("ProcessingJobArn", "")
        violations = self._read_violations_from_s3(schedule_name)
        return violations

    def _read_violations_from_s3(self, schedule_name: str) -> list[dict[str, Any]]:
        """Lê violações do relatório do Model Monitor no S3."""
        prefix = f"model-monitor/reports/{self.endpoint_name}/"
        violations_key = None

        paginator = self.s3.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=S3_BUCKET, Prefix=prefix):
            for obj in page.get("Contents", []):
                if obj["Key"].endswith("constraint_violations.json"):
                    violations_key = obj["Key"]

        if not violations_key:
            logger.info("Nenhum arquivo de violações encontrado — modelo dentro dos limites.")
            return []

        body = self.s3.get_object(Bucket=S3_BUCKET, Key=violations_key)["Body"].read()
        data = json.loads(body)
        violations = data.get("violations", [])

        if violations:
            logger.warning(f"⚠️  {len(violations)} violação(ões) detectada(s) pelo Model Monitor:")
            for v in violations:
                logger.warning(f"   • {v.get('feature_name')}: {v.get('description')}")
        else:
            logger.success("✅ Nenhuma violação detectada pelo Model Monitor.")

        return violations

    def run_local_drift_check(
        self,
        reference_uri: str,
        current_uri: str,
    ) -> dict[str, Any]:
        """
        Roda o DriftDetector local do ML Quality Gate
        sobre dados do S3 para comparação com o Model Monitor.
        """
        import sys
        sys.path.insert(0, "src")
        from ml_quality_gate.validators.drift_detector import DriftDetector

        logger.info("🔍 Rodando DriftDetector local para comparação...")
        ref_df = self.loader.load_parquet(reference_uri.replace(f"s3://{S3_BUCKET}/", ""))
        cur_df = self.loader.load_parquet(current_uri.replace(f"s3://{S3_BUCKET}/", ""))

        numeric_cols = ref_df.select_dtypes(include="number").columns.tolist()
        detector = DriftDetector(ref_df[numeric_cols], cur_df[numeric_cols])
        report = detector.detect_all()

        return report.summary()


def main():
    parser = argparse.ArgumentParser(description="Integração SageMaker Model Monitor")
    parser.add_argument("--endpoint-name", required=True)
    parser.add_argument("--baseline-uri",  required=True, help="S3 URI do baseline")
    parser.add_argument("--setup",  action="store_true", help="Configurar monitoring schedule")
    parser.add_argument("--check",  action="store_true", help="Verificar violações recentes")
    args = parser.parse_args()

    monitor = SageMakerMonitorIntegration(args.endpoint_name)

    if args.setup:
        monitor.setup_monitoring_schedule(baseline_uri=args.baseline_uri)

    if args.check:
        violations = monitor.get_latest_violations()
        if violations:
            logger.error(f"🚨 {len(violations)} violação(ões) detectada(s). Considere retreinar.")
            exit(1)
        logger.success("✅ Nenhuma violação. Modelo estável.")


if __name__ == "__main__":
    main()
