"""
AWS Lambda — Quality Gate Trigger
Função Lambda que dispara o ML Quality Gate automaticamente quando:
  - Um novo modelo é registrado no SageMaker Model Registry
  - Um job de treinamento é concluído (via EventBridge)
  - Um schedule diário é atingido

Eventos suportados:
  - SageMaker Model Registry (ModelPackage state change)
  - EventBridge Scheduled Rules
  - S3 PUT (novo artefato de modelo)

Deploy:
  zip lambda.zip quality_gate_trigger.py
  aws lambda create-function \\
    --function-name ml-quality-gate-trigger \\
    --runtime python3.11 \\
    --handler quality_gate_trigger.handler \\
    --role arn:aws:iam::123456789012:role/LambdaMLQualityGateRole \\
    --zip-file fileb://lambda.zip
"""

from __future__ import annotations

import json
import os
from typing import Any

import boto3


ENDPOINT_NAME     = os.environ.get("SAGEMAKER_ENDPOINT_NAME", "credit-risk-v2")
S3_BUCKET         = os.environ.get("S3_BUCKET", "meu-bucket-ml")
SNS_TOPIC_ARN     = os.environ.get("SNS_TOPIC_ARN", "")
CODEBUILD_PROJECT = os.environ.get("CODEBUILD_PROJECT", "ml-quality-gate")
REGION            = os.environ.get("AWS_REGION", "us-east-1")


def _notify_slack(message: str, passed: bool) -> None:
    """Envia notificação via SNS (que pode estar integrado ao Slack)."""
    if not SNS_TOPIC_ARN:
        return
    sns = boto3.client("sns", region_name=REGION)
    sns.publish(
        TopicArn=SNS_TOPIC_ARN,
        Subject=f"ML Quality Gate {'✅ PASSOU' if passed else '❌ FALHOU'} — {ENDPOINT_NAME}",
        Message=message,
    )


def _trigger_codebuild(model_package_arn: str) -> dict[str, Any]:
    """Dispara um job CodeBuild para rodar o quality gate completo."""
    cb = boto3.client("codebuild", region_name=REGION)
    response = cb.start_build(
        projectName=CODEBUILD_PROJECT,
        environmentVariablesOverride=[
            {"name": "MODEL_PACKAGE_ARN", "value": model_package_arn, "type": "PLAINTEXT"},
            {"name": "SAGEMAKER_ENDPOINT_NAME", "value": ENDPOINT_NAME, "type": "PLAINTEXT"},
        ],
    )
    build_id = response["build"]["id"]
    return {"buildId": build_id, "status": "STARTED"}


def _run_inline_quality_checks(model_package_arn: str) -> dict[str, Any]:
    """
    Executa verificações rápidas inline (sem CodeBuild).
    Útil para gates síncronos antes de aprovação automática.
    """
    sm = boto3.client("sagemaker", region_name=REGION)
    cw = boto3.client("cloudwatch", region_name=REGION)

    results: dict[str, Any] = {"checks": [], "passed": True}

    # 1. Verificar metadados do model package
    try:
        pkg = sm.describe_model_package(ModelPackageName=model_package_arn)
        metrics = pkg.get("CustomerMetadataProperties", {})

        f1 = float(metrics.get("F1Score", 0))
        f1_threshold = float(os.environ.get("F1_THRESHOLD", "0.82"))
        f1_pass = f1 >= f1_threshold

        results["checks"].append({
            "name": "F1Score",
            "value": f1,
            "threshold": f1_threshold,
            "passed": f1_pass,
        })
        if not f1_pass:
            results["passed"] = False

    except Exception as e:
        results["checks"].append({"name": "MetadataCheck", "error": str(e), "passed": False})
        results["passed"] = False

    # 2. Verificar métricas recentes do CloudWatch
    try:
        response = cw.get_metric_statistics(
            Namespace="MLQualityGate",
            MetricName="DriftShare",
            Dimensions=[{"Name": "ModelName", "Value": ENDPOINT_NAME}],
            StartTime=__import__("datetime").datetime.utcnow() - __import__("datetime").timedelta(days=1),
            EndTime=__import__("datetime").datetime.utcnow(),
            Period=86400,
            Statistics=["Maximum"],
        )
        datapoints = response.get("Datapoints", [])
        if datapoints:
            drift_share = max(dp["Maximum"] for dp in datapoints)
            drift_threshold = float(os.environ.get("DRIFT_THRESHOLD", "0.30"))
            drift_pass = drift_share < drift_threshold
            results["checks"].append({
                "name": "DriftShare",
                "value": drift_share,
                "threshold": drift_threshold,
                "passed": drift_pass,
            })
            if not drift_pass:
                results["passed"] = False
    except Exception as e:
        results["checks"].append({"name": "DriftCheck", "warning": str(e), "passed": True})

    return results


def handler(event: dict[str, Any], context: Any) -> dict[str, Any]:
    """
    Handler principal da Lambda.
    Detecta o tipo de evento e executa o quality gate adequado.
    """
    print(f"Evento recebido: {json.dumps(event, default=str)}")

    # ── Evento: SageMaker Model Registry ──────────────────────────────────────
    if event.get("source") == "aws.sagemaker" and "ModelPackage" in event.get("detail-type", ""):
        detail = event.get("detail", {})
        model_package_arn = detail.get("ModelPackageArn", "")
        new_status = detail.get("ModelApprovalStatus", "")

        if new_status != "PendingManualApproval":
            return {"statusCode": 200, "body": "Evento ignorado — não é PendingManualApproval."}

        print(f"🆕 Novo modelo pendente de aprovação: {model_package_arn}")

        # Rodar quality gate inline
        results = _run_inline_quality_checks(model_package_arn)

        sm = boto3.client("sagemaker", region_name=REGION)
        if results["passed"]:
            # Aprovar automaticamente se gate passou
            sm.update_model_package(
                ModelPackageArn=model_package_arn,
                ModelApprovalStatus="Approved",
                ApprovalDescription="Aprovado automaticamente pelo ML Quality Gate Lambda.",
            )
            _notify_slack(
                f"✅ Modelo {model_package_arn} APROVADO automaticamente.\n"
                f"Checks: {results['checks']}",
                passed=True,
            )
            print("✅ Modelo aprovado automaticamente.")
        else:
            # Rejeitar e notificar
            sm.update_model_package(
                ModelPackageArn=model_package_arn,
                ModelApprovalStatus="Rejected",
                ApprovalDescription=f"Rejeitado pelo ML Quality Gate. Falhas: {results['checks']}",
            )
            _notify_slack(
                f"❌ Modelo {model_package_arn} REJEITADO pelo quality gate.\n"
                f"Falhas: {[c for c in results['checks'] if not c.get('passed')]}",
                passed=False,
            )
            print("❌ Modelo rejeitado — quality gate falhou.")

        return {"statusCode": 200, "body": json.dumps(results)}

    # ── Evento: Schedule diário (EventBridge) ─────────────────────────────────
    elif event.get("source") == "aws.events":
        print("⏰ Disparo agendado — iniciando quality gate via CodeBuild...")
        result = _trigger_codebuild(model_package_arn=ENDPOINT_NAME)
        return {"statusCode": 200, "body": json.dumps(result)}

    # ── Evento: S3 (novo artefato) ─────────────────────────────────────────────
    elif "Records" in event and event["Records"][0].get("eventSource") == "aws:s3":
        s3_key = event["Records"][0]["s3"]["object"]["key"]
        print(f"📦 Novo artefato detectado no S3: {s3_key}")
        result = _trigger_codebuild(model_package_arn=s3_key)
        return {"statusCode": 200, "body": json.dumps(result)}

    return {"statusCode": 400, "body": "Tipo de evento não reconhecido."}
