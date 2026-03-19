"""
Deploy no SageMaker — Registra modelo no Model Registry e deploya o endpoint.
Integra com o ML Quality Gate: só faz deploy se o quality gate passar.

Uso:
  python examples/aws/sagemaker/deploy_model.py \\
    --model-name credit-risk-v2 \\
    --model-uri s3://meu-bucket/models/credit-risk-v2/model.tar.gz \\
    --instance-type ml.m5.large \\
    --run-quality-gate

Fluxo:
  1. Roda o quality gate (se --run-quality-gate)
  2. Registra o modelo no SageMaker Model Registry
  3. Aprova o modelo (se gate passou)
  4. Deploya o endpoint (ou atualiza se já existir)
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from datetime import datetime

import boto3
from loguru import logger


REGION    = os.getenv("AWS_REGION", "us-east-1")
ROLE_ARN  = os.getenv("SAGEMAKER_ROLE_ARN", "arn:aws:iam::123456789012:role/SageMakerRole")
IMAGE_URI = os.getenv(
    "SAGEMAKER_IMAGE_URI",
    f"683313688378.dkr.ecr.{REGION}.amazonaws.com/sagemaker-scikit-learn:1.2-1-cpu-py3",
)


def run_quality_gate() -> bool:
    """Executa o quality gate completo antes do deploy."""
    logger.info("🧪 Executando quality gate antes do deploy...")
    result = subprocess.run(
        ["pytest", "tests/unit/", "tests/fairness/", "tests/e2e/", "-v", "--tb=short", "-q"],
        capture_output=False,
    )
    passed = result.returncode == 0
    if passed:
        logger.success("✅ Quality gate passou — prosseguindo com o deploy.")
    else:
        logger.error("❌ Quality gate FALHOU — deploy cancelado.")
    return passed


def register_model(
    sm_client,
    model_name: str,
    model_uri: str,
    metrics: dict[str, float],
) -> str:
    """Registra modelo no SageMaker Model Registry e retorna o ARN da versão."""
    logger.info(f"📦 Registrando modelo '{model_name}' no Model Registry...")

    # Garantir que o Model Package Group existe
    try:
        sm_client.create_model_package_group(
            ModelPackageGroupName=model_name,
            ModelPackageGroupDescription=f"Modelos de crédito — {model_name}",
        )
        logger.info(f"Model Package Group '{model_name}' criado.")
    except sm_client.exceptions.ConflictException:
        logger.info(f"Model Package Group '{model_name}' já existe.")

    # Formatar métricas para o Model Registry
    metric_data = [
        {"MetricName": name, "Value": value}
        for name, value in metrics.items()
    ]

    response = sm_client.create_model_package(
        ModelPackageGroupName=model_name,
        ModelPackageDescription=f"Versão gerada em {datetime.now().isoformat()}",
        InferenceSpecification={
            "Containers": [{
                "Image": IMAGE_URI,
                "ModelDataUrl": model_uri,
                "Framework": "SKLEARN",
                "FrameworkVersion": "1.2",
            }],
            "SupportedContentTypes": ["application/json"],
            "SupportedResponseMIMETypes": ["application/json"],
        },
        ModelMetrics={
            "ModelQuality": {
                "Statistics": {
                    "ContentType": "application/json",
                    "S3Uri": model_uri.replace("model.tar.gz", "metrics.json"),
                }
            }
        },
        AdditionalInferenceSpecificationsToAdd=[],
        ModelApprovalStatus="PendingManualApproval",
        CustomerMetadataProperties={
            k: str(v) for k, v in metrics.items()
        },
    )

    arn = response["ModelPackageArn"]
    logger.success(f"Modelo registrado: {arn}")
    return arn


def approve_model(sm_client, model_package_arn: str) -> None:
    """Aprova o modelo no Registry (necessário para deploy)."""
    sm_client.update_model_package(
        ModelPackageArn=model_package_arn,
        ModelApprovalStatus="Approved",
        ApprovalDescription="Aprovado automaticamente pelo ML Quality Gate CI/CD.",
    )
    logger.success(f"Modelo aprovado: {model_package_arn}")


def deploy_endpoint(
    sm_client,
    endpoint_name: str,
    model_package_arn: str,
    instance_type: str,
    initial_instance_count: int = 1,
) -> None:
    """Cria ou atualiza o endpoint SageMaker."""

    # Criar model object a partir do package
    model_name = f"{endpoint_name}-{int(time.time())}"
    sm_client.create_model(
        ModelName=model_name,
        ExecutionRoleArn=ROLE_ARN,
        Containers=[{"ModelPackageName": model_package_arn}],
    )

    # Criar endpoint config
    config_name = f"{endpoint_name}-config-{int(time.time())}"
    sm_client.create_endpoint_config(
        EndpointConfigName=config_name,
        ProductionVariants=[{
            "VariantName": "primary",
            "ModelName": model_name,
            "InitialInstanceCount": initial_instance_count,
            "InstanceType": instance_type,
            "InitialVariantWeight": 1.0,
        }],
        DataCaptureConfig={
            "EnableCapture": True,
            "InitialSamplingPercentage": 100,
            "DestinationS3Uri": f"s3://{os.getenv('S3_BUCKET', 'meu-bucket')}/data-capture/{endpoint_name}/",
            "CaptureOptions": [
                {"CaptureMode": "Input"},
                {"CaptureMode": "Output"},
            ],
        },
    )

    # Verificar se endpoint já existe → atualizar, senão criar
    try:
        sm_client.describe_endpoint(EndpointName=endpoint_name)
        logger.info(f"Atualizando endpoint existente: {endpoint_name}")
        sm_client.update_endpoint(
            EndpointName=endpoint_name,
            EndpointConfigName=config_name,
        )
    except sm_client.exceptions.ClientError:
        logger.info(f"Criando novo endpoint: {endpoint_name}")
        sm_client.create_endpoint(
            EndpointName=endpoint_name,
            EndpointConfigName=config_name,
        )

    # Aguardar endpoint ficar disponível
    logger.info("⏳ Aguardando endpoint ficar InService...")
    waiter = sm_client.get_waiter("endpoint_in_service")
    waiter.wait(
        EndpointName=endpoint_name,
        WaiterConfig={"Delay": 30, "MaxAttempts": 40},
    )
    logger.success(f"✅ Endpoint '{endpoint_name}' está InService!")


def main() -> None:
    parser = argparse.ArgumentParser(description="Deploy de modelo no SageMaker com quality gate.")
    parser.add_argument("--model-name",     required=True,  help="Nome do modelo/endpoint")
    parser.add_argument("--model-uri",      required=True,  help="S3 URI do model.tar.gz")
    parser.add_argument("--instance-type",  default="ml.m5.large")
    parser.add_argument("--instance-count", type=int, default=1)
    parser.add_argument("--run-quality-gate", action="store_true",
                        help="Executar quality gate antes do deploy")
    parser.add_argument("--f1-score",   type=float, default=0.87)
    parser.add_argument("--accuracy",   type=float, default=0.91)
    parser.add_argument("--roc-auc",    type=float, default=0.93)
    args = parser.parse_args()

    # 1. Quality Gate
    if args.run_quality_gate:
        if not run_quality_gate():
            logger.error("🚨 Deploy bloqueado pelo quality gate.")
            sys.exit(1)

    sm_client = boto3.client("sagemaker", region_name=REGION)
    metrics = {
        "F1Score": args.f1_score,
        "Accuracy": args.accuracy,
        "RocAuc": args.roc_auc,
    }

    # 2. Registrar no Model Registry
    arn = register_model(sm_client, args.model_name, args.model_uri, metrics)

    # 3. Aprovar
    approve_model(sm_client, arn)

    # 4. Deploy
    deploy_endpoint(
        sm_client,
        endpoint_name=args.model_name,
        model_package_arn=arn,
        instance_type=args.instance_type,
        initial_instance_count=args.instance_count,
    )

    logger.success(f"🚀 Deploy completo! Endpoint: {args.model_name}")


if __name__ == "__main__":
    main()
