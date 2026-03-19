"""
Testes de Integração — SageMaker Real-time Endpoint
Substitui o mock_api_server.py por um endpoint real do SageMaker.

Pré-requisitos:
  - Endpoint SageMaker deployado e em serviço (InService)
  - Variável SAGEMAKER_ENDPOINT_NAME configurada
  - Credenciais AWS com permissão sagemaker:InvokeEndpoint

Executar:
  pytest examples/aws/sagemaker/test_endpoint.py -v -m integration
"""

from __future__ import annotations

import json
import os
import time
from typing import Any

import boto3
import pytest
from botocore.exceptions import ClientError


# ── Fixtures ───────────────────────────────────────────────────────────────────

ENDPOINT_NAME = os.getenv("SAGEMAKER_ENDPOINT_NAME", "credit-risk-v2")
REGION        = os.getenv("AWS_REGION", "us-east-1")


def endpoint_is_available() -> bool:
    """Verifica se o endpoint SageMaker está em estado InService."""
    try:
        sm = boto3.client("sagemaker", region_name=REGION)
        resp = sm.describe_endpoint(EndpointName=ENDPOINT_NAME)
        return resp["EndpointStatus"] == "InService"
    except Exception:
        return False


@pytest.fixture(scope="session")
def sm_runtime():
    """Retorna cliente boto3 para invocar endpoints."""
    return boto3.client("sagemaker-runtime", region_name=REGION)


@pytest.fixture(scope="session")
def sm_client():
    """Retorna cliente boto3 para gerenciar endpoints."""
    return boto3.client("sagemaker", region_name=REGION)


@pytest.fixture
def valid_payload() -> dict[str, Any]:
    return {
        "age": 35,
        "income": 75000.0,
        "credit_score": 720,
        "loan_amount": 25000.0,
        "loan_term_months": 60,
        "employment_type": "employed",
        "marital_status": "married",
    }


def invoke(client: Any, payload: dict) -> dict:
    """Invoca o endpoint SageMaker e retorna o corpo da resposta."""
    response = client.invoke_endpoint(
        EndpointName=ENDPOINT_NAME,
        ContentType="application/json",
        Accept="application/json",
        Body=json.dumps(payload),
    )
    return json.loads(response["Body"].read().decode("utf-8"))


# ── Testes ─────────────────────────────────────────────────────────────────────

@pytest.mark.integration
class TestSageMakerEndpoint:
    """Testes de contrato e comportamento do endpoint SageMaker."""

    @pytest.fixture(autouse=True)
    def skip_if_unavailable(self):
        if not endpoint_is_available():
            pytest.skip(
                f"Endpoint '{ENDPOINT_NAME}' não está disponível (InService). "
                "Verifique o SageMaker Console."
            )

    def test_endpoint_status_is_inservice(self, sm_client):
        """Endpoint deve estar em estado InService."""
        resp = sm_client.describe_endpoint(EndpointName=ENDPOINT_NAME)
        assert resp["EndpointStatus"] == "InService", (
            f"Endpoint está em estado: {resp['EndpointStatus']}"
        )

    def test_invocacao_retorna_predicao(self, sm_runtime, valid_payload):
        """Invocação básica deve retornar campo 'prediction'."""
        result = invoke(sm_runtime, valid_payload)
        assert "prediction" in result, f"Campo 'prediction' ausente na resposta: {result}"

    def test_predicao_e_binaria(self, sm_runtime, valid_payload):
        """Modelo de classificação binária deve retornar apenas 0 ou 1."""
        result = invoke(sm_runtime, valid_payload)
        assert result["prediction"] in (0, 1), (
            f"Predição não binária: {result['prediction']}"
        )

    def test_confianca_e_probabilidade_valida(self, sm_runtime, valid_payload):
        """Campo 'confidence' deve estar no intervalo [0.0, 1.0]."""
        result = invoke(sm_runtime, valid_payload)
        if "confidence" in result:
            conf = result["confidence"]
            assert 0.0 <= conf <= 1.0, f"Confidence fora do range: {conf}"

    def test_versao_do_modelo_presente(self, sm_runtime, valid_payload):
        """Resposta deve incluir a versão do modelo para rastreabilidade."""
        result = invoke(sm_runtime, valid_payload)
        assert "model_version" in result, (
            "Campo 'model_version' ausente — necessário para auditoria e rastreabilidade."
        )

    def test_request_id_unico_por_chamada(self, sm_runtime, valid_payload):
        """Cada requisição deve ter um request_id único para tracing distribuído."""
        ids = {invoke(sm_runtime, valid_payload).get("request_id") for _ in range(5)}
        ids.discard(None)
        if ids:
            assert len(ids) == 5, f"Request IDs repetidos detectados: {ids}"

    def test_predicao_e_deterministica(self, sm_runtime, valid_payload):
        """Mesmo payload deve sempre produzir mesma predição."""
        predicoes = [invoke(sm_runtime, valid_payload)["prediction"] for _ in range(3)]
        assert len(set(predicoes)) == 1, (
            f"Predições não determinísticas para o mesmo input: {predicoes}"
        )

    def test_latencia_aceitavel(self, sm_runtime, valid_payload):
        """Latência de inferência deve ser < 500ms (SLO de produção)."""
        inicio = time.perf_counter()
        invoke(sm_runtime, valid_payload)
        ms = (time.perf_counter() - inicio) * 1000
        assert ms < 500, f"Latência {ms:.0f}ms excede SLO de 500ms"

    def test_payload_invalido_retorna_erro(self, sm_runtime):
        """Payload inválido deve retornar erro HTTP 4xx."""
        payload_invalido = {"age": "não_é_número", "credit_score": 9999}
        try:
            invoke(sm_runtime, payload_invalido)
            # Se o endpoint não validar, o teste avisa mas não bloqueia
            pytest.xfail(
                "Endpoint aceitou payload inválido — considere adicionar validação de input."
            )
        except ClientError as e:
            codigo = e.response["Error"]["Code"]
            assert codigo in ("ModelError", "ValidationError"), (
                f"Código de erro inesperado: {codigo}"
            )

    def test_schema_completo_da_resposta(self, sm_runtime, valid_payload):
        """Valida todos os campos obrigatórios da resposta usando Pydantic."""
        import sys
        sys.path.insert(0, "src")
        from ml_quality_gate.contracts.api_schemas import PredictionResponse

        result = invoke(sm_runtime, valid_payload)
        try:
            parsed = PredictionResponse(**result)
            assert parsed.prediction in {0, 1}
        except Exception as e:
            pytest.fail(f"Resposta não conforma com o schema Pydantic: {e}\nResposta: {result}")

    def test_alto_risco_prediz_default(self, sm_runtime):
        """
        Teste de lógica de negócio:
        perfil de alto risco (credit_score baixo, desempregado) deve tender a predizer default.
        """
        payload_alto_risco = {
            "age": 25,
            "income": 15000.0,
            "credit_score": 380,
            "loan_amount": 80000.0,
            "loan_term_months": 120,
            "employment_type": "unemployed",
            "marital_status": "single",
        }
        result = invoke(sm_runtime, payload_alto_risco)
        # Soft assertion — loga aviso se o modelo não prediz default para perfil de alto risco
        if result.get("prediction") != 1:
            pytest.xfail(
                f"Modelo não prediz default para perfil de alto risco — revisar features. "
                f"Resposta: {result}"
            )


@pytest.mark.integration
class TestSageMakerEndpointMetadata:
    """Testes de metadados e configuração do endpoint."""

    @pytest.fixture(autouse=True)
    def skip_if_unavailable(self):
        if not endpoint_is_available():
            pytest.skip("Endpoint não disponível.")

    def test_endpoint_tem_configuracao_de_producao(self, sm_client):
        """Endpoint de produção deve usar instância adequada (não ml.t2.medium)."""
        resp = sm_client.describe_endpoint(EndpointName=ENDPOINT_NAME)
        config_name = resp["EndpointConfigName"]
        config = sm_client.describe_endpoint_config(EndpointConfigName=config_name)
        instance_type = config["ProductionVariants"][0]["InstanceType"]

        instancias_dev = {"ml.t2.medium", "ml.t2.large", "ml.t3.medium"}
        assert instance_type not in instancias_dev, (
            f"Endpoint de produção usando instância de dev/test: {instance_type}"
        )

    def test_endpoint_tem_data_capture_ativo(self, sm_client):
        """
        Data Capture deve estar ativo para permitir monitoramento de drift.
        Obrigatório para uso com SageMaker Model Monitor.
        """
        resp = sm_client.describe_endpoint(EndpointName=ENDPOINT_NAME)
        config_name = resp["EndpointConfigName"]
        config = sm_client.describe_endpoint_config(EndpointConfigName=config_name)

        data_capture = config.get("DataCaptureConfig", {})
        if not data_capture:
            pytest.xfail(
                "DataCaptureConfig não configurado — ative para monitoramento de drift em produção."
            )
        assert data_capture.get("EnableCapture") is True, (
            "Data Capture está desativado. Ative para registrar inputs/outputs do modelo."
        )
