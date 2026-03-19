# 🧪 ML Quality Gate Framework

> Framework de QA enterprise para sistemas de Machine Learning.

![Python](https://img.shields.io/badge/Python-3.11+-blue?logo=python)
![Pytest](https://img.shields.io/badge/Pytest-7.x-green?logo=pytest)
![K6](https://img.shields.io/badge/K6-Performance-purple?logo=k6)
![CI/CD](https://img.shields.io/badge/CI/CD-GitHub_Actions-black?logo=githubactions)
![License](https://img.shields.io/badge/License-MIT-yellow)

---

## 💡 O que é este projeto?

O **ML Quality Gate** é um framework de qualidade automatizado para pipelines de Machine Learning. Ele garante que nenhum modelo seja implantado em produção sem passar por validações rigorosas de métricas, qualidade de dados, detecção de drift, testes de viés e contrato de API — tudo integrado ao CI/CD.

Construído com foco em **padrões de engenharia de big tech**: código tipado, configuração externalizada, relatórios ricos e pipeline de qualidade totalmente automatizado.

---

## 🏗️ Estrutura do Projeto

```
ml-quality-gate/
├── src/ml_quality_gate/        # Core do framework
│   ├── validators/             # Validadores: modelo, dados, drift, fairness
│   ├── reporters/              # Relatórios: HTML, Slack, JSON
│   ├── contracts/              # Contratos de API (Pydantic v2)
│   ├── utils/                  # Config loader, logger estruturado
│   └── cli.py                  # Interface de linha de comando
├── tests/
│   ├── unit/                   # Métricas de modelo e qualidade de dados
│   ├── integration/            # Testes de contrato de API
│   ├── performance/            # Testes de carga com K6
│   ├── fairness/               # Validação de viés e equidade
│   ├── e2e/                    # Pipeline completo end-to-end
│   ├── conftest.py             # Fixtures compartilhadas (session scope)
│   └── plugin.py               # Plugin pytest com MLAssert e relatório rico
├── config/
│   ├── thresholds.yaml         # Thresholds de qualidade (editável sem código)
│   └── model_config.yaml       # Configuração do modelo e API
├── scripts/
│   ├── mock_api_server.py      # Servidor FastAPI mock para testes locais
│   └── generate_report.py      # Gerador de relatório HTML consolidado
├── docker/                     # Dockerfile e Docker Compose
├── .github/workflows/          # Pipeline CI/CD com GitHub Actions
└── docs/                       # ADRs e guia de contribuição
```

---

## 🚀 Como começar

### 1. Clonar e instalar

```bash
git clone https://github.com/seu-org/ml-quality-gate.git
cd ml-quality-gate

python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

pip install -e ".[dev]"
```

### 2. Configurar variáveis de ambiente

```bash
cp .env.example .env
# Edite .env com a URL da sua API e token de autenticação
```

### 3. Executar o quality gate completo

```bash
# Gate completo (lint + unit + fairness + e2e)
make quality-gate

# Suites individuais
make test-unit          # Métricas de modelo + qualidade de dados
make test-fairness      # Viés e equidade
make test-integration   # Contrato de API (requer API no ar)
make test-e2e           # Pipeline completo
make test-performance   # Testes de carga K6
```

### 4. Rodar o servidor mock localmente

```bash
uvicorn scripts.mock_api_server:app --reload --port 8000
# Documentação disponível em: http://localhost:8000/docs
```

### 5. Usar a CLI

```bash
ml-quality-gate run --suite model       # Executa testes de métricas
ml-quality-gate run --suite fairness    # Executa testes de fairness
ml-quality-gate check-drift \
  --reference data/ref.parquet \
  --current data/prod.parquet           # Detecta drift entre datasets
ml-quality-gate report                  # Gera relatório HTML
ml-quality-gate status                  # Exibe thresholds configurados
```

---

## 🧪 Suites de Teste

| Suite | Ferramenta | O que valida |
|---|---|---|
| **Validação de Modelo** | pytest + scikit-learn | Accuracy, Precision, Recall, F1, AUC-ROC |
| **Qualidade de Dados** | pytest + pandas | Nulos, duplicatas, ranges, categorias, infinitos |
| **Detecção de Drift** | KS-test, Chi-squared, PSI | Drift por feature e no dataset completo |
| **Contrato de API** | pytest + requests + Pydantic | Schema, status HTTP, consistência, latência |
| **Performance** | K6 | Latência p95/p99, throughput, taxa de erro |
| **Fairness & Viés** | pytest + fairlearn | Paridade demográfica, equalized odds |
| **Robustez** | pytest + numpy | Monotonicidade, edge cases, calibração |
| **E2E Pipeline** | pytest | Pipeline completo: dados → modelo → API → fairness |

---

## ⚙️ Thresholds Configuráveis

Todos os limites de qualidade ficam em `config/thresholds.yaml` — sem precisar alterar código:

```yaml
model:
  accuracy:
    minimum: 0.85       # Gate falha abaixo disso
    critical: 0.80      # Deploy bloqueado abaixo disso
  f1_score:
    minimum: 0.82

fairness:
  max_demographic_parity_diff: 0.05   # Máximo 5% de disparidade entre grupos

performance:
  latency:
    p95_ms: 300         # 95% das requisições < 300ms
    p99_ms: 500
```

---

## 📊 Relatórios Gerados

Após rodar os testes, os relatórios ficam disponíveis em:

| Caminho | Conteúdo |
|---|---|
| `reports/html/quality_report.html` | Dashboard HTML completo |
| `reports/html/unit_report.html` | Relatório de testes unitários |
| `reports/html/fairness_report.html` | Relatório de fairness |
| `reports/json/session_summary.json` | Sumário legível por máquina |
| `reports/json/k6_results.json` | Resultados brutos do K6 |
| `reports/coverage/` | Cobertura de código |

---

## 🔄 CI/CD — GitHub Actions

Cada Pull Request dispara automaticamente:

```
code-quality → data-quality → model-metrics ──┐
                                               ├──→ e2e → quality-gate-summary
                             fairness ─────────┘
                                               ↓ (somente scheduled/manual)
                             api-contracts → performance
```

O job `quality-gate-summary` **bloqueia o merge** se qualquer gate falhar e notifica o Slack automaticamente.

---

## 🐳 Docker

```bash
# Subir API mock + rodar todos os testes
make docker-test

# Apenas subir os serviços
make docker-up

# Parar tudo
make docker-down
```

---

## 🛠️ Stack Tecnológica

| Categoria | Ferramentas |
|---|---|
| **Testes** | pytest, pytest-cov, pytest-html, pytest-xdist |
| **ML/Dados** | scikit-learn, pandas, numpy, scipy |
| **Qualidade de Dados** | Great Expectations, Evidently AI |
| **Fairness** | Fairlearn |
| **API** | FastAPI (mock), requests, Pydantic v2 |
| **Performance** | K6 (Grafana) |
| **Relatórios** | Jinja2, Rich |
| **CI/CD** | GitHub Actions |
| **Containers** | Docker, Docker Compose |
| **Qualidade de Código** | Ruff, Black, mypy, pre-commit |

---

## 🤝 Como Contribuir

Veja [docs/CONTRIBUTING.md](docs/CONTRIBUTING.md) para o guia completo.

Alterações em thresholds (`config/thresholds.yaml`) exigem aprovação do QA Lead e do ML Engineer responsável.

---

## 📄 Licença

MIT © Sua Organização
