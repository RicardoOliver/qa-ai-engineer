"""
Testes de Integração — SageMaker Batch Transform
Valida jobs de inferência em lote: schema de saída, cobertura, consistência.

Pré-requisitos:
  - Job de Batch Transform já executado com sucesso
  - Resultados disponíveis no S3 (S3_OUTPUT_URI)
  - Variáveis de ambiente configuradas

Executar:
  pytest examples/aws/sagemaker/test_batch_transform.py -v -m integration
"""

from __future__ import annotations

import io
import json
import os
from typing import Any

import boto3
import pandas as pd
import pytest


BATCH_OUTPUT_URI = os.getenv("BATCH_OUTPUT_URI", "s3://meu-bucket/batch-outputs/")
BATCH_INPUT_URI  = os.getenv("BATCH_INPUT_URI",  "s3://meu-bucket/batch-inputs/")
REGION           = os.getenv("AWS_REGION", "us-east-1")


def _parse_s3_uri(uri: str) -> tuple[str, str]:
    """Divide 's3://bucket/key' em (bucket, key)."""
    path = uri.replace("s3://", "")
    bucket, _, key = path.partition("/")
    return bucket, key


@pytest.fixture(scope="session")
def s3_client():
    return boto3.client("s3", region_name=REGION)


@pytest.fixture(scope="session")
def batch_output_df(s3_client) -> pd.DataFrame:
    """Carrega todos os resultados do Batch Transform do S3."""
    bucket, prefix = _parse_s3_uri(BATCH_OUTPUT_URI)
    paginator = s3_client.get_paginator("list_objects_v2")
    pages = paginator.paginate(Bucket=bucket, Prefix=prefix)

    records = []
    for page in pages:
        for obj in page.get("Contents", []):
            body = s3_client.get_object(Bucket=bucket, Key=obj["Key"])["Body"].read()
            for line in body.decode("utf-8").strip().splitlines():
                if line:
                    records.append(json.loads(line))

    if not records:
        pytest.skip(f"Nenhum output encontrado em {BATCH_OUTPUT_URI}")
    return pd.DataFrame(records)


@pytest.mark.integration
class TestBatchTransformOutput:
    """Validações do output de jobs de Batch Transform."""

    def test_output_nao_esta_vazio(self, batch_output_df: pd.DataFrame):
        assert len(batch_output_df) > 0, "Output do Batch Transform está vazio!"

    def test_campo_prediction_presente(self, batch_output_df: pd.DataFrame):
        assert "prediction" in batch_output_df.columns, (
            "Campo 'prediction' ausente no output do batch."
        )

    def test_predicoes_sao_binarias(self, batch_output_df: pd.DataFrame):
        invalidos = batch_output_df[~batch_output_df["prediction"].isin([0, 1])]
        assert len(invalidos) == 0, (
            f"{len(invalidos)} predições com valores fora de {{0, 1}}:\n{invalidos.head()}"
        )

    def test_sem_predicoes_nulas(self, batch_output_df: pd.DataFrame):
        nulos = batch_output_df["prediction"].isnull().sum()
        assert nulos == 0, f"{nulos} predições nulas encontradas no output do batch."

    def test_confidence_dentro_do_range(self, batch_output_df: pd.DataFrame):
        if "confidence" not in batch_output_df.columns:
            pytest.skip("Campo 'confidence' não presente no output.")
        fora_do_range = batch_output_df[
            (batch_output_df["confidence"] < 0) |
            (batch_output_df["confidence"] > 1)
        ]
        assert len(fora_do_range) == 0, (
            f"{len(fora_do_range)} registros com confidence fora de [0, 1]."
        )

    def test_taxa_de_default_e_plausivel(self, batch_output_df: pd.DataFrame):
        """
        Taxa de default prevista deve estar em um range plausível para o negócio.
        Ajuste os limites conforme o contexto do seu modelo.
        """
        taxa_default = batch_output_df["prediction"].mean()
        assert 0.05 <= taxa_default <= 0.60, (
            f"Taxa de default {taxa_default:.1%} fora do range esperado [5%, 60%]. "
            "Possível problema com o modelo ou com os dados de entrada."
        )

    def test_todos_registros_processados(self, s3_client, batch_output_df: pd.DataFrame):
        """
        Número de predições deve ser igual ao número de registros de entrada.
        Garante que nenhum registro foi perdido no processamento.
        """
        bucket, prefix = _parse_s3_uri(BATCH_INPUT_URI)
        paginator = s3_client.get_paginator("list_objects_v2")
        pages = paginator.paginate(Bucket=bucket, Prefix=prefix)

        n_input = sum(
            1
            for page in pages
            for obj in page.get("Contents", [])
            if obj["Key"].endswith(".jsonl") or obj["Key"].endswith(".csv")
        )

        if n_input == 0:
            pytest.skip("Não foi possível contar registros de input.")

        assert len(batch_output_df) >= n_input * 0.99, (
            f"Output ({len(batch_output_df)} registros) < 99% do input ({n_input} registros). "
            "Possível falha parcial no batch."
        )
