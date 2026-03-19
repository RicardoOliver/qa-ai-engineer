# Contributing to ML Quality Gate

## Getting Started

1. Fork the repo and clone your fork
2. Create a virtual env: `python -m venv .venv && source .venv/bin/activate`
3. Install dev deps: `pip install -e ".[dev]"`
4. Install pre-commit hooks: `pre-commit install`

## Adding New Tests

- **Unit tests** → `tests/unit/`
- **Integration tests** → `tests/integration/`
- **Fairness tests** → `tests/fairness/`
- **E2E tests** → `tests/e2e/`

Always mark tests with the appropriate pytest marker (`@pytest.mark.unit`, etc.)

## Updating Thresholds

Edit `config/thresholds.yaml`. All threshold changes require approval from the QA Lead and ML Engineer.

## Pull Request Checklist

- [ ] All new tests pass locally (`make quality-gate`)
- [ ] Code formatted (`make format`)
- [ ] Type checks pass (`make typecheck`)
- [ ] Thresholds documented if changed
- [ ] PR description explains the change
