"""
conftest_s3.py — Fixtures pytest usando Amazon S3
Substitui o conftest.py padrão quando executando contra dados reais da AWS.

Como usar:
  Copie este arquivo para o diretório raiz de testes como conftest.py,
  ou importe as fixtures no seu conftest.py existente.

Variáveis de ambiente necessárias:
  S3_BUCKET, S3_REFERENCE_DATA_KEY, S3_CURRENT_DATA_KEY,
  AWS_REGION, SAGEMAKER_ENDPOINT_NAME
"""

from __future__ import annotations

import os
from typing import Any

import pytest

from examples.aws.s3.aws_data_loader import S3DataLoader


S3_BUCKET = os.getenv("S3_BUCKET", "meu-bucket-ml")
REGION    = os.getenv("AWS_REGION", "us-east-1")


@pytest.fixture(scope="session")
def s3_loader() -> S3DataLoader:
    """Loader de dados do S3 com cache local."""
    return S3DataLoader(bucket=S3_BUCKET, region=REGION, cache_locally=True)


@pytest.fixture(scope="session")
def reference_df(s3_loader: S3DataLoader):
    """Dataset de referência (baseline) carregado do S3."""
    try:
        return s3_loader.load_reference_data()
    except Exception as e:
        pytest.skip(f"Não foi possível carregar dados de referência do S3: {e}")


@pytest.fixture(scope="session")
def current_df(s3_loader: S3DataLoader):
    """Dataset atual de produção carregado do S3."""
    try:
        return s3_loader.load_current_data()
    except Exception as e:
        pytest.skip(f"Não foi possível carregar dados atuais do S3: {e}")


@pytest.fixture(scope="session")
def trained_model(s3_loader: S3DataLoader):
    """Modelo de produção carregado do S3."""
    model_key = os.getenv("S3_MODEL_KEY", "models/credit-risk/model.pkl")
    try:
        return s3_loader.load_model(model_key)
    except Exception as e:
        pytest.skip(f"Não foi possível carregar modelo do S3: {e}")


@pytest.fixture(scope="session")
def api_base_url() -> str:
    """URL do endpoint SageMaker para testes de integração."""
    endpoint = os.getenv("SAGEMAKER_ENDPOINT_NAME", "")
    if endpoint:
        return f"https://runtime.sagemaker.{REGION}.amazonaws.com/endpoints/{endpoint}/invocations"
    return os.getenv("MODEL_API_URL", "http://localhost:8000")


@pytest.fixture
def valid_prediction_payload() -> dict[str, Any]:
    return {
        "age": 35,
        "income": 75000.0,
        "credit_score": 720,
        "loan_amount": 25000.0,
        "loan_term_months": 60,
        "employment_type": "employed",
        "marital_status": "married",
    }
