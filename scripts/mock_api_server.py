"""
Mock ML Prediction API Server
FastAPI server that simulates a production ML model endpoint.
Used for local integration and performance testing without a real model.

Run: uvicorn scripts.mock_api_server:app --reload --port 8000
"""

from __future__ import annotations

import time
import uuid
from enum import Enum
from typing import Any

from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

app = FastAPI(
    title="Credit Risk Model API",
    description="Mock ML prediction API for QA testing",
    version="2.1.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

START_TIME = time.time()
REQUEST_COUNT = 0


# ── Schemas ────────────────────────────────────────────────────────────────────

class EmploymentType(str, Enum):
    employed = "employed"
    self_employed = "self_employed"
    unemployed = "unemployed"
    retired = "retired"


class MaritalStatus(str, Enum):
    single = "single"
    married = "married"
    divorced = "divorced"
    widowed = "widowed"


class PredictionRequest(BaseModel):
    age: int = Field(..., ge=18, le=100)
    income: float = Field(..., ge=0)
    credit_score: int = Field(..., ge=300, le=850)
    loan_amount: float = Field(..., ge=1_000, le=5_000_000)
    loan_term_months: int = Field(..., ge=1, le=360)
    employment_type: EmploymentType
    marital_status: MaritalStatus

    model_config = {
        "json_schema_extra": {
            "example": {
                "age": 35,
                "income": 75000,
                "credit_score": 720,
                "loan_amount": 25000,
                "loan_term_months": 60,
                "employment_type": "employed",
                "marital_status": "married",
            }
        }
    }


class PredictionResponse(BaseModel):
    prediction: int
    confidence: float
    model_version: str
    request_id: str
    feature_importance: dict[str, float]
    inference_time_ms: float


# ── Prediction Logic ───────────────────────────────────────────────────────────

def _compute_default_probability(req: PredictionRequest) -> float:
    """Deterministic rule-based scoring (mirrors real model logic for testing)."""
    # Normalize credit score [0..1], higher = better
    credit_factor = (req.credit_score - 300) / 550

    # Debt-to-income ratio
    annual_payment = (req.loan_amount / req.loan_term_months) * 12
    dti = annual_payment / max(req.income, 1)
    dti_factor = min(dti / 0.5, 1.0)  # Normalize to [0..1]

    # Employment adjustment
    employment_risk = {
        "employed": 0.0,
        "self_employed": 0.05,
        "retired": 0.08,
        "unemployed": 0.25,
    }[req.employment_type]

    # Compute base probability
    default_prob = (
        0.5
        - (credit_factor * 0.45)   # Credit score is most important
        + (dti_factor * 0.30)      # DTI second
        + employment_risk
    )

    # Age adjustment — very young applicants slightly higher risk
    if req.age < 25:
        default_prob += 0.05

    return float(max(0.02, min(0.98, default_prob)))


# ── Endpoints ──────────────────────────────────────────────────────────────────

@app.get("/health", tags=["Infrastructure"])
async def health_check() -> dict[str, Any]:
    """Health check endpoint."""
    return {
        "status": "ok",
        "model_version": "2.1.0",
        "uptime_seconds": round(time.time() - START_TIME, 2),
        "total_requests": REQUEST_COUNT,
        "checks": {
            "model_loaded": True,
            "database": True,
            "feature_store": True,
        },
    }


@app.get("/metrics", tags=["Infrastructure"])
async def prometheus_metrics() -> str:
    """Prometheus-compatible metrics endpoint."""
    return (
        f"# HELP ml_requests_total Total number of prediction requests\n"
        f"# TYPE ml_requests_total counter\n"
        f"ml_requests_total {REQUEST_COUNT}\n"
        f"# HELP ml_uptime_seconds Server uptime\n"
        f"# TYPE ml_uptime_seconds gauge\n"
        f"ml_uptime_seconds {time.time() - START_TIME:.2f}\n"
    )


@app.post(
    "/v1/predict",
    response_model=PredictionResponse,
    tags=["Prediction"],
    summary="Credit default risk prediction",
)
async def predict(req: PredictionRequest) -> PredictionResponse:
    """
    Predict credit default risk for a loan applicant.

    Returns:
    - **prediction**: 0 = no default, 1 = default
    - **confidence**: model confidence score [0, 1]
    - **model_version**: deployed model version
    - **request_id**: unique trace ID
    - **feature_importance**: SHAP-like per-feature contributions
    """
    global REQUEST_COUNT
    REQUEST_COUNT += 1

    t0 = time.perf_counter()

    default_prob = _compute_default_probability(req)
    prediction = 1 if default_prob >= 0.5 else 0
    confidence = abs(default_prob - 0.5) * 2  # Distance from decision boundary

    # Synthetic feature importances
    importance = {
        "credit_score":     round(0.42 * (1 - (req.credit_score - 300) / 550), 4),
        "income":           round(0.28 * min(1.0, 75_000 / max(req.income, 1)), 4),
        "loan_amount":      round(0.18 * min(1.0, req.loan_amount / 100_000), 4),
        "age":              round(0.08 * (1 - (req.age - 18) / 57), 4),
        "employment_type":  round(0.04, 4),
    }

    elapsed_ms = (time.perf_counter() - t0) * 1000

    return PredictionResponse(
        prediction=prediction,
        confidence=round(confidence, 6),
        model_version="2.1.0",
        request_id=str(uuid.uuid4()),
        feature_importance=importance,
        inference_time_ms=round(elapsed_ms, 3),
    )


@app.post("/v1/predict/batch", tags=["Prediction"])
async def predict_batch(requests: list[PredictionRequest]) -> list[PredictionResponse]:
    """Batch prediction endpoint — process multiple applicants at once."""
    if len(requests) > 100:
        raise HTTPException(status_code=400, detail="Batch size must be <= 100")
    return [await predict(req) for req in requests]


# ── Error Handlers ─────────────────────────────────────────────────────────────

@app.exception_handler(422)
async def validation_error_handler(request: Request, exc: Any) -> JSONResponse:
    return JSONResponse(
        status_code=422,
        content={
            "error": "Validation Error",
            "detail": str(exc),
            "request_id": str(uuid.uuid4()),
        },
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
