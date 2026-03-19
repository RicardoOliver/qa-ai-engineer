# Architecture Decision Records (ADR)

## ADR-001: pytest as primary test runner

**Status:** Accepted

**Context:** Need a unified test runner for all QA suites (unit, integration, fairness, e2e).

**Decision:** Use pytest with custom markers to segregate suites. Fixtures in `conftest.py` at session scope to avoid redundant model training.

**Consequences:** All engineers use the same `pytest` command regardless of suite. Fixtures cached at session scope keep CI runs fast.

---

## ADR-002: Thresholds externalized to YAML

**Status:** Accepted

**Context:** Metric thresholds are business decisions (agreed by ML Engineering + Product) and should be reviewable without touching test code.

**Decision:** All thresholds live in `config/thresholds.yaml`. Tests load them at runtime via `ConfigLoader`. Changes to thresholds require a PR and reviewer approval.

**Consequences:** Threshold changes are tracked in git history. Non-engineers can propose threshold changes via YAML PRs.

---

## ADR-003: K6 for performance testing (not Locust/JMeter)

**Status:** Accepted

**Context:** Need a modern, scriptable performance tool that integrates with CI/CD.

**Decision:** K6 (Grafana) chosen for its JavaScript scripting, built-in metrics, low overhead, and native GitHub Actions support. Team already uses it.

**Consequences:** Performance tests are JavaScript; a separate K6 installation is required. Thresholds are enforced natively without external tooling.

---

## ADR-004: Fairlearn for fairness metrics

**Status:** Accepted

**Context:** Need a standardized way to compute demographic parity, equalized odds, etc.

**Decision:** Microsoft Fairlearn library provides industry-standard fairness metrics and integrates cleanly with scikit-learn.

**Consequences:** Fairlearn adds a dependency. Custom `FairnessValidator` wraps it to enforce our specific threshold config.

---

## ADR-005: No MLflow dependency in QA layer

**Status:** Accepted

**Context:** MLflow (or similar) could be used for experiment tracking and model loading.

**Decision:** QA layer is MLflow-agnostic. Models are loaded from artifacts directory or provided via fixtures. This keeps the QA framework portable across model registries.

**Consequences:** Teams using MLflow must write a thin adapter to load models into the test fixture format. A `scripts/load_from_mlflow.py` helper can be added separately.
