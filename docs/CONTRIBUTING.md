# Contribuindo para o ML Quality Gate

## Primeiros Passos

1. Faça um fork do repositório e clone o seu fork
2. Crie um ambiente virtual: `python -m venv .venv && source .venv/bin/activate`
3. Instale as dependências de desenvolvimento: `pip install -e ".[dev]"`
4. Instale os hooks de pre-commit: `pre-commit install`

## Adicionando Novos Testes

- **Testes unitários** → `tests/unit/`
- **Testes de integração** → `tests/integration/`
- **Testes de fairness (equidade)** → `tests/fairness/`
- **Testes E2E** → `tests/e2e/`

Sempre marque os testes com o marcador apropriado do pytest (`@pytest.mark.unit`, etc.)

## Atualizando Limiares

Edite `config/thresholds.yaml`. Todas as alterações de limiares exigem aprovação do QA Lead e do Engenheiro de ML.

## Checklist de Pull Request

- [ ] Todos os novos testes passam localmente (`make quality-gate`)
- [ ] Código formatado (`make format`)
- [ ] Verificações de tipo aprovadas (`make typecheck`)
- [ ] Limiares documentados, se alterados
- [ ] Descrição do PR explica a mudança
