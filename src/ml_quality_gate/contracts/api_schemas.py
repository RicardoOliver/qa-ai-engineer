"""
API Contracts — Pydantic schemas for prediction API request/response validation.
"""

from __future__ import annotations

from enum import Enum
from typing import Any
from uuid import UUID

from pydantic import BaseModel, Field, field_validator


class EmploymentType(str, Enum):
    EMPLOYED = "employed"
    SELF_EMPLOYED = "self_employed"
    UNEMPLOYED = "unemployed"
    RETIRED = "retired"


class MaritalStatus(str, Enum):
    SINGLE = "single"
    MARRIED = "married"
    DIVORCED = "divorced"
    WIDOWED = "widowed"


# ── Request Schemas ───────────────────────────────────────────────────────────

class PredictionRequest(BaseModel):
    age: int = Field(..., ge=18, le=100, description="Applicant age")
    income: float = Field(..., ge=0, description="Annual income in USD")
    credit_score: int = Field(..., ge=300, le=850, description="Credit score")
    loan_amount: float = Field(..., ge=1_000, le=5_000_000)
    loan_term_months: int = Field(..., ge=1, le=360)
    employment_type: EmploymentType
    marital_status: MaritalStatus

    model_config = {"json_schema_extra": {
        "example": {
            "age": 35,
            "income": 75000.0,
            "credit_score": 720,
            "loan_amount": 25000.0,
            "loan_term_months": 60,
            "employment_type": "employed",
            "marital_status": "married",
        }
    }}


# ── Response Schemas ──────────────────────────────────────────────────────────

class PredictionResponse(BaseModel):
    prediction: int = Field(..., description="0=no default, 1=default")
    confidence: float = Field(..., ge=0.0, le=1.0)
    model_version: str = Field(..., description="Deployed model version")
    request_id: str = Field(..., description="Unique request identifier for tracing")
    feature_importance: dict[str, float] | None = None

    @field_validator("confidence")
    @classmethod
    def confidence_must_be_valid_probability(cls, v: float) -> float:
        if not (0.0 <= v <= 1.0):
            raise ValueError(f"confidence must be in [0, 1], got {v}")
        return v


class HealthResponse(BaseModel):
    status: str = Field(..., description="'ok' or 'degraded'")
    model_version: str
    uptime_seconds: float
    checks: dict[str, bool] = Field(default_factory=dict)


class ErrorResponse(BaseModel):
    error: str
    detail: str | None = None
    request_id: str | None = None
