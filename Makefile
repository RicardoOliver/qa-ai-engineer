.PHONY: all install quality-gate test-unit test-integration test-fairness test-performance test-e2e lint format typecheck report clean docker-up docker-down

PYTHON = python3
PYTEST = pytest
K6 = k6
REPORTS_DIR = reports

# ─── Setup ────────────────────────────────────────────────────────────────────

install:
	pip install -e ".[dev]"
	pre-commit install

# ─── Quality Gate (Full Suite) ────────────────────────────────────────────────

quality-gate: lint test-unit test-integration test-fairness
	@echo "✅ Quality Gate PASSED"

# ─── Test Suites ──────────────────────────────────────────────────────────────

test-unit:
	@echo "🧪 Running Unit Tests (Model + Data Quality)..."
	$(PYTEST) tests/unit/ -v -m unit --html=$(REPORTS_DIR)/html/unit_report.html --self-contained-html

test-integration:
	@echo "🌐 Running Integration Tests (API Contract)..."
	$(PYTEST) tests/integration/ -v -m integration --html=$(REPORTS_DIR)/html/integration_report.html --self-contained-html

test-fairness:
	@echo "⚖️  Running Fairness & Bias Tests..."
	$(PYTEST) tests/fairness/ -v -m fairness --html=$(REPORTS_DIR)/html/fairness_report.html --self-contained-html

test-e2e:
	@echo "🔁 Running End-to-End Pipeline Tests..."
	$(PYTEST) tests/e2e/ -v -m e2e --html=$(REPORTS_DIR)/html/e2e_report.html --self-contained-html

test-performance:
	@echo "🚀 Running Performance Tests (K6)..."
	$(K6) run tests/performance/load_test_model.js --out json=$(REPORTS_DIR)/json/k6_results.json

test-all:
	$(PYTEST) tests/ -v --html=$(REPORTS_DIR)/html/full_report.html --self-contained-html

# ─── Code Quality ─────────────────────────────────────────────────────────────

lint:
	@echo "🔍 Linting..."
	ruff check src/ tests/
	black --check src/ tests/

format:
	black src/ tests/
	ruff check --fix src/ tests/

typecheck:
	mypy src/

# ─── Reports ──────────────────────────────────────────────────────────────────

report:
	$(PYTHON) scripts/generate_report.py
	@echo "📊 Report generated at $(REPORTS_DIR)/html/quality_report.html"

# ─── Docker ───────────────────────────────────────────────────────────────────

docker-up:
	docker compose -f docker/docker-compose.yml up -d
	@echo "⏳ Waiting for services to be healthy..."
	sleep 5

docker-down:
	docker compose -f docker/docker-compose.yml down

docker-test: docker-up test-all docker-down

# ─── Cleanup ──────────────────────────────────────────────────────────────────

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	rm -rf reports/html reports/json reports/coverage
	mkdir -p reports/html reports/json
