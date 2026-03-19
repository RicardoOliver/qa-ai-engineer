# ☁️ Exemplos de Integração com AWS

Este diretório contém exemplos prontos para integrar o **ML Quality Gate** com os principais serviços AWS usados em pipelines de MLOps.

---

## 📁 Estrutura

```
examples/aws/
├── sagemaker/
│   ├── test_endpoint.py          # Testes de integração com SageMaker Endpoint
│   ├── test_batch_transform.py   # Testes de Batch Transform
│   └── deploy_model.py           # Script para registrar e deployar modelo
├── s3/
│   ├── aws_data_loader.py        # Carrega datasets de treino/referência do S3
│   └── conftest_s3.py            # Fixtures do pytest usando S3
├── cloudwatch/
│   └── cloudwatch_reporter.py    # Publica métricas do quality gate no CloudWatch
├── monitor/
│   └── sagemaker_monitor.py      # Integração com SageMaker Model Monitor
├── lambda/
│   └── quality_gate_trigger.py   # Função Lambda que dispara o quality gate
├── iam/
│   └── policy.json               # Policy IAM mínima necessária
└── README.md                     # Este arquivo
```

---

## 🚀 Pré-requisitos

```bash
pip install boto3 awscli sagemaker
aws configure  # Configure suas credenciais AWS
```

### Variáveis de ambiente necessárias

```bash
export AWS_REGION=us-east-1
export AWS_ACCESS_KEY_ID=sua_access_key
export AWS_SECRET_ACCESS_KEY=sua_secret_key

# Específicas do projeto
export SAGEMAKER_ENDPOINT_NAME=credit-risk-v2
export SAGEMAKER_MODEL_NAME=credit-risk-gbm
export S3_BUCKET=meu-bucket-ml
export S3_REFERENCE_DATA_KEY=data/reference/baseline.parquet
export S3_CURRENT_DATA_KEY=data/current/production.parquet
export CLOUDWATCH_NAMESPACE=MLQualityGate
```

---

## ⚡ Como usar cada exemplo

### 1. Testar endpoint SageMaker
```bash
pytest examples/aws/sagemaker/test_endpoint.py -v -m integration
```

### 2. Verificar drift com dados do S3
```bash
python examples/aws/monitor/sagemaker_monitor.py
```

### 3. Publicar métricas no CloudWatch
```bash
python examples/aws/cloudwatch/cloudwatch_reporter.py
```

### 4. Deploy do modelo via script
```bash
python examples/aws/sagemaker/deploy_model.py \
  --model-name credit-risk-v2 \
  --instance-type ml.m5.large
```

---

## 🗺️ Mapa de serviços

| Componente local | Serviço AWS |
|---|---|
| `mock_api_server.py` | SageMaker Real-time Endpoint |
| `DriftDetector` | SageMaker Model Monitor (complementar) |
| `DataValidator` | AWS Glue Data Quality (complementar) |
| `HtmlReporter` | CloudWatch Dashboards |
| `SlackNotifier` | SNS + Lambda → Slack |
| `artifacts/` | Amazon S3 |
| `reports/` | S3 + Athena |
| CI/CD pipeline | GitHub Actions + IAM Role |
