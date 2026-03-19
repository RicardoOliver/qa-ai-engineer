"""
AWS S3 Data Loader
Carrega datasets de treino, referência e produção diretamente do Amazon S3.
Substitui os arquivos locais de 'artifacts/' quando em ambiente AWS.

Uso:
  from examples.aws.s3.aws_data_loader import S3DataLoader

  loader = S3DataLoader(bucket="meu-bucket-ml")
  reference_df = loader.load_reference_data()
  current_df   = loader.load_current_data()
  model        = loader.load_model()
"""

from __future__ import annotations

import io
import os
import pickle
from pathlib import Path
from typing import Any

import boto3
import pandas as pd
from loguru import logger


class S3DataLoader:
    """
    Carrega datasets e artefatos de modelo do Amazon S3.

    Parâmetros:
        bucket: Nome do bucket S3
        prefix: Prefixo base dos arquivos (ex: 'ml/credit-risk/')
        region: Região AWS (padrão: AWS_REGION env var)
        cache_locally: Se True, faz cache local dos arquivos para evitar re-downloads
    """

    def __init__(
        self,
        bucket: str | None = None,
        prefix: str = "",
        region: str | None = None,
        cache_locally: bool = True,
        cache_dir: str = ".s3_cache",
    ) -> None:
        self.bucket = bucket or os.environ["S3_BUCKET"]
        self.prefix = prefix.rstrip("/")
        self.region = region or os.getenv("AWS_REGION", "us-east-1")
        self.cache_locally = cache_locally
        self.cache_dir = Path(cache_dir)
        self._s3 = boto3.client("s3", region_name=self.region)

    def _key(self, relative_path: str) -> str:
        return f"{self.prefix}/{relative_path}".lstrip("/")

    def _download(self, key: str) -> bytes:
        """Faz download de um objeto S3 com cache local opcional."""
        if self.cache_locally:
            cache_path = self.cache_dir / key.replace("/", "_")
            if cache_path.exists():
                logger.debug(f"Cache hit: {cache_path}")
                return cache_path.read_bytes()

        logger.info(f"⬇️  Baixando s3://{self.bucket}/{key}")
        obj = self._s3.get_object(Bucket=self.bucket, Key=key)
        data = obj["Body"].read()

        if self.cache_locally:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            (self.cache_dir / key.replace("/", "_")).write_bytes(data)

        return data

    def load_parquet(self, relative_path: str) -> pd.DataFrame:
        """Carrega arquivo Parquet do S3."""
        data = self._download(self._key(relative_path))
        df = pd.read_parquet(io.BytesIO(data))
        logger.success(f"Dataset carregado: {len(df)} linhas × {df.shape[1]} colunas")
        return df

    def load_csv(self, relative_path: str, **kwargs) -> pd.DataFrame:
        """Carrega arquivo CSV do S3."""
        data = self._download(self._key(relative_path))
        df = pd.read_csv(io.BytesIO(data), **kwargs)
        logger.success(f"CSV carregado: {len(df)} linhas")
        return df

    def load_model(self, relative_path: str = "models/model.pkl") -> Any:
        """Carrega modelo serializado (pickle) do S3."""
        data = self._download(self._key(relative_path))
        model = pickle.loads(data)  # noqa: S301
        logger.success(f"Modelo carregado do S3: {type(model).__name__}")
        return model

    def load_reference_data(self) -> pd.DataFrame:
        """Atalho para carregar dados de referência (baseline)."""
        key = os.getenv("S3_REFERENCE_DATA_KEY", "data/reference/baseline.parquet")
        return self.load_parquet(key)

    def load_current_data(self) -> pd.DataFrame:
        """Atalho para carregar dados atuais de produção."""
        key = os.getenv("S3_CURRENT_DATA_KEY", "data/current/production.parquet")
        return self.load_parquet(key)

    def upload_report(self, local_path: str, s3_key: str) -> str:
        """Faz upload de relatório para o S3 e retorna a URL."""
        self._s3.upload_file(local_path, self.bucket, s3_key)
        url = f"https://{self.bucket}.s3.{self.region}.amazonaws.com/{s3_key}"
        logger.success(f"Relatório publicado em: {url}")
        return url

    def list_files(self, relative_prefix: str = "") -> list[str]:
        """Lista arquivos no bucket com o prefixo dado."""
        full_prefix = self._key(relative_prefix)
        paginator = self._s3.get_paginator("list_objects_v2")
        keys = []
        for page in paginator.paginate(Bucket=self.bucket, Prefix=full_prefix):
            for obj in page.get("Contents", []):
                keys.append(obj["Key"])
        return keys
