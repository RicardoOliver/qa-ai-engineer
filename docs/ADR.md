# Registros de Decisão de Arquitetura (ADR)

## ADR-001: pytest como principal executor de testes

**Status:** Aceito

**Contexto:** Necessidade de um executor de testes unificado para todas as suítes de QA (unitários, integração, fairness, e2e).

**Decisão:** Utilizar pytest com marcadores customizados para separar as suítes. Fixtures em `conftest.py` com escopo de sessão para evitar treinamento redundante de modelos.

**Consequências:** Todos os engenheiros utilizam o mesmo comando `pytest`, independentemente da suíte. Fixtures em cache no escopo de sessão mantêm as execuções de CI rápidas.

---

## ADR-002: Limiares externalizados em YAML

**Status:** Aceito

**Contexto:** Os limiares de métricas são decisões de negócio (acordadas entre Engenharia de ML + Produto) e devem ser revisáveis sem alterar o código de teste.

**Decisão:** Todos os limiares ficam em `config/thresholds.yaml`. Os testes os carregam em tempo de execução via `ConfigLoader`. Alterações nos limiares exigem um PR e aprovação de revisores.

**Consequências:** Alterações de limiares são rastreadas no histórico do git. Pessoas não engenheiras podem propor mudanças via PRs em YAML.

---

## ADR-003: K6 para testes de performance (não Locust/JMeter)

**Status:** Aceito

**Contexto:** Necessidade de uma ferramenta moderna e scriptável de performance que se integre com CI/CD.

**Decisão:** K6 (Grafana) foi escolhido por seu suporte a scripts em JavaScript, métricas nativas, baixo overhead e suporte nativo ao GitHub Actions. A equipe já utiliza a ferramenta.

**Consequências:** Testes de performance são escritos em JavaScript; é necessária uma instalação separada do K6. Limiares são aplicados nativamente sem necessidade de ferramentas externas.

---

## ADR-004: Fairlearn para métricas de equidade (fairness)

**Status:** Aceito

**Contexto:** Necessidade de uma forma padronizada para calcular paridade demográfica, equalized odds, entre outros.

**Decisão:** A biblioteca Microsoft Fairlearn fornece métricas de equidade padrão da indústria e integra-se facilmente com scikit-learn.

**Consequências:** Fairlearn adiciona uma dependência ao projeto. Um `FairnessValidator` customizado encapsula a biblioteca para aplicar nossa configuração específica de limiares.

---

## ADR-005: Sem dependência de MLflow na camada de QA

**Status:** Aceito

**Contexto:** MLflow (ou similares) poderia ser utilizado para rastreamento de experimentos e carregamento de modelos.

**Decisão:** A camada de QA é agnóstica ao MLflow. Os modelos são carregados a partir do diretório de artefatos ou fornecidos via fixtures. Isso mantém o framework de QA portável entre diferentes registries de modelos.

**Consequências:** Times que utilizam MLflow precisam criar um adaptador simples para carregar modelos no formato esperado pelas fixtures de teste. Um helper `scripts/load_from_mlflow.py` pode ser adicionado separadamente.
