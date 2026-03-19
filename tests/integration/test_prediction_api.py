"""
Integration Tests — Prediction API
Tests API contract, schema validation, consistency, and error handling.
Requires MODEL_API_URL to be set and a running API server.
"""

from __future__ import annotations

import time
from typing import Any

import pytest
import requests

from ml_quality_gate.contracts.api_schemas import PredictionRequest, PredictionResponse


# Skip all tests if API is unreachable
def api_is_available(base_url: str) -> bool:
    try:
        r = requests.get(f"{base_url}/health", timeout=3)
        return r.status_code == 200
    except Exception:
        return False


@pytest.mark.integration
class TestPredictionAPIContract:
    """Contract tests for the ML model prediction API."""

    @pytest.fixture(autouse=True)
    def check_api_available(self, api_base_url: str) -> None:
        if not api_is_available(api_base_url):
            pytest.skip(f"API not available at {api_base_url} — skipping integration tests.")
        self.base_url = api_base_url

    # ── Happy Path ────────────────────────────────────────────────────────────

    def test_health_endpoint_returns_200(self) -> None:
        response = requests.get(f"{self.base_url}/health", timeout=5)
        assert response.status_code == 200, f"Health check failed: {response.text}"

    def test_health_response_has_status_ok(self) -> None:
        response = requests.get(f"{self.base_url}/health", timeout=5).json()
        assert response.get("status") in ("ok", "healthy")

    def test_prediction_returns_200(self, valid_prediction_payload: dict[str, Any]) -> None:
        response = requests.post(
            f"{self.base_url}/v1/predict",
            json=valid_prediction_payload,
            timeout=10,
        )
        assert response.status_code == 200, f"Expected 200, got {response.status_code}: {response.text}"

    def test_prediction_response_schema(self, valid_prediction_payload: dict[str, Any]) -> None:
        """Response must conform to PredictionResponse Pydantic schema."""
        response = requests.post(
            f"{self.base_url}/v1/predict",
            json=valid_prediction_payload,
            timeout=10,
        ).json()

        # Validate via Pydantic — raises ValidationError if schema is wrong
        parsed = PredictionResponse(**response)
        assert parsed.prediction in {0, 1}
        assert 0.0 <= parsed.confidence <= 1.0
        assert parsed.model_version
        assert parsed.request_id

    def test_prediction_is_binary(self, valid_prediction_payload: dict[str, Any]) -> None:
        response = requests.post(
            f"{self.base_url}/v1/predict",
            json=valid_prediction_payload,
            timeout=10,
        ).json()
        assert response["prediction"] in (0, 1), f"Non-binary prediction: {response['prediction']}"

    def test_confidence_is_valid_probability(self, valid_prediction_payload: dict[str, Any]) -> None:
        response = requests.post(
            f"{self.base_url}/v1/predict",
            json=valid_prediction_payload,
            timeout=10,
        ).json()
        confidence = response["confidence"]
        assert 0.0 <= confidence <= 1.0, f"Invalid confidence: {confidence}"

    def test_model_version_is_present(self, valid_prediction_payload: dict[str, Any]) -> None:
        response = requests.post(
            f"{self.base_url}/v1/predict",
            json=valid_prediction_payload,
            timeout=10,
        ).json()
        assert "model_version" in response and response["model_version"]

    def test_request_id_is_unique_per_request(self, valid_prediction_payload: dict[str, Any]) -> None:
        ids = set()
        for _ in range(5):
            r = requests.post(f"{self.base_url}/v1/predict", json=valid_prediction_payload, timeout=10)
            ids.add(r.json()["request_id"])
        assert len(ids) == 5, f"Request IDs are not unique: {ids}"

    # ── Consistency ───────────────────────────────────────────────────────────

    def test_prediction_is_deterministic(self, valid_prediction_payload: dict[str, Any]) -> None:
        """Same input must produce same prediction (model must be deterministic)."""
        results = [
            requests.post(f"{self.base_url}/v1/predict", json=valid_prediction_payload, timeout=10).json()
            for _ in range(3)
        ]
        predictions = [r["prediction"] for r in results]
        assert len(set(predictions)) == 1, f"Non-deterministic predictions: {predictions}"

    def test_response_time_is_acceptable(self, valid_prediction_payload: dict[str, Any]) -> None:
        """Single inference must respond within 500ms."""
        start = time.perf_counter()
        requests.post(f"{self.base_url}/v1/predict", json=valid_prediction_payload, timeout=10)
        elapsed_ms = (time.perf_counter() - start) * 1000
        assert elapsed_ms < 500, f"Response time {elapsed_ms:.0f}ms exceeds 500ms threshold"

    # ── Error Handling ────────────────────────────────────────────────────────

    def test_missing_required_field_returns_422(self) -> None:
        """Missing required fields must return 422 Unprocessable Entity."""
        payload = {"age": 35}  # Missing many required fields
        response = requests.post(f"{self.base_url}/v1/predict", json=payload, timeout=10)
        assert response.status_code == 422, f"Expected 422, got {response.status_code}"

    def test_invalid_age_returns_error(self) -> None:
        """Age below minimum (18) must be rejected."""
        payload = {
            "age": 10,  # Invalid
            "income": 50000.0,
            "credit_score": 700,
            "loan_amount": 10000.0,
            "loan_term_months": 36,
            "employment_type": "employed",
            "marital_status": "single",
        }
        response = requests.post(f"{self.base_url}/v1/predict", json=payload, timeout=10)
        assert response.status_code in (400, 422)

    def test_invalid_credit_score_returns_error(self) -> None:
        """Credit score out of [300, 850] range must be rejected."""
        payload = {
            "age": 35,
            "income": 50000.0,
            "credit_score": 999,  # Invalid
            "loan_amount": 10000.0,
            "loan_term_months": 36,
            "employment_type": "employed",
            "marital_status": "single",
        }
        response = requests.post(f"{self.base_url}/v1/predict", json=payload, timeout=10)
        assert response.status_code in (400, 422)

    def test_invalid_employment_type_returns_error(self) -> None:
        """Unknown employment_type must be rejected."""
        payload = {
            "age": 35,
            "income": 50000.0,
            "credit_score": 700,
            "loan_amount": 10000.0,
            "loan_term_months": 36,
            "employment_type": "astronaut",  # Invalid
            "marital_status": "single",
        }
        response = requests.post(f"{self.base_url}/v1/predict", json=payload, timeout=10)
        assert response.status_code in (400, 422)

    def test_empty_body_returns_error(self) -> None:
        response = requests.post(f"{self.base_url}/v1/predict", json={}, timeout=10)
        assert response.status_code in (400, 422)

    def test_content_type_is_json(self, valid_prediction_payload: dict[str, Any]) -> None:
        response = requests.post(
            f"{self.base_url}/v1/predict",
            json=valid_prediction_payload,
            timeout=10,
        )
        assert "application/json" in response.headers.get("Content-Type", "")
