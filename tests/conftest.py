"""
Pytest conftest.py — Shared fixtures for all test suites.
Generates synthetic but realistic ML data for testing.
"""

from __future__ import annotations

import os
from typing import Any, Generator

import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_classification
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


# ─── Seed for reproducibility ─────────────────────────────────────────────────
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)


# ─── Synthetic Dataset ────────────────────────────────────────────────────────

@pytest.fixture(scope="session")
def raw_dataframe() -> pd.DataFrame:
    """Realistic synthetic credit risk dataset."""
    n = 2_000
    rng = np.random.default_rng(RANDOM_SEED)

    df = pd.DataFrame({
        "age": rng.integers(18, 75, size=n),
        "income": rng.integers(20_000, 200_000, size=n).astype(float),
        "credit_score": rng.integers(300, 850, size=n),
        "loan_amount": rng.integers(1_000, 100_000, size=n).astype(float),
        "loan_term_months": rng.choice([12, 24, 36, 48, 60, 84, 120], size=n),
        "employment_type": rng.choice(["employed", "self_employed", "unemployed", "retired"], size=n),
        "marital_status": rng.choice(["single", "married", "divorced", "widowed"], size=n),
        "gender": rng.choice(["male", "female"], size=n),
        "age_group": pd.cut(
            rng.integers(18, 75, size=n),
            bins=[17, 30, 45, 60, 75],
            labels=["18-30", "31-45", "46-60", "61-75"],
        ).astype(str),
    })

    # Synthetic target: higher credit score → lower default
    default_prob = 1 - (df["credit_score"] - 300) / 550
    default_prob = default_prob.clip(0.05, 0.95)
    df["default"] = rng.binomial(1, default_prob, size=n)
    return df


@pytest.fixture(scope="session")
def clean_dataframe(raw_dataframe: pd.DataFrame) -> pd.DataFrame:
    """Encode categorical features for model training."""
    df = raw_dataframe.copy()
    le = LabelEncoder()
    for col in ["employment_type", "marital_status", "gender", "age_group"]:
        df[col] = le.fit_transform(df[col])
    return df


@pytest.fixture(scope="session")
def train_test_data(clean_dataframe: pd.DataFrame) -> dict[str, Any]:
    feature_cols = [
        "age", "income", "credit_score", "loan_amount",
        "loan_term_months", "employment_type", "marital_status",
    ]
    X = clean_dataframe[feature_cols]
    y = clean_dataframe["default"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_SEED, stratify=y
    )
    return {
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "feature_cols": feature_cols,
    }


@pytest.fixture(scope="session")
def trained_model(train_test_data: dict[str, Any]) -> GradientBoostingClassifier:
    """Train a real GBM model on synthetic data."""
    model = GradientBoostingClassifier(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=4,
        random_state=RANDOM_SEED,
    )
    model.fit(train_test_data["X_train"], train_test_data["y_train"])
    return model


@pytest.fixture(scope="session")
def predictions(trained_model: GradientBoostingClassifier, train_test_data: dict[str, Any]) -> dict[str, Any]:
    X_test = train_test_data["X_test"]
    y_test = train_test_data["y_test"]
    return {
        "y_pred": trained_model.predict(X_test),
        "y_prob": trained_model.predict_proba(X_test)[:, 1],
        "y_true": y_test.values,
    }


@pytest.fixture(scope="session")
def reference_df(raw_dataframe: pd.DataFrame) -> pd.DataFrame:
    return raw_dataframe.iloc[:1_000].copy()


@pytest.fixture(scope="session")
def current_df_no_drift(raw_dataframe: pd.DataFrame) -> pd.DataFrame:
    """Current data without drift — should pass drift detection."""
    return raw_dataframe.iloc[1_000:].copy()


@pytest.fixture(scope="session")
def current_df_with_drift(raw_dataframe: pd.DataFrame) -> pd.DataFrame:
    """Current data with injected drift — should trigger drift detection."""
    df = raw_dataframe.iloc[1_000:].copy()
    rng = np.random.default_rng(99)
    # Shift income distribution dramatically
    df["income"] = df["income"] * 0.3 + rng.integers(0, 10_000, size=len(df))
    # Shift credit score
    df["credit_score"] = (df["credit_score"] - 150).clip(300, 850)
    return df


@pytest.fixture(scope="session")
def sensitive_features(raw_dataframe: pd.DataFrame) -> pd.DataFrame:
    return raw_dataframe[["gender", "age_group"]].copy()


# ─── API Fixtures ─────────────────────────────────────────────────────────────

@pytest.fixture(scope="session")
def api_base_url() -> str:
    return os.getenv("MODEL_API_URL", "http://localhost:8000")


@pytest.fixture
def valid_prediction_payload() -> dict[str, Any]:
    return {
        "age": 35,
        "income": 75000.0,
        "credit_score": 720,
        "loan_amount": 25000.0,
        "loan_term_months": 60,
        "employment_type": "employed",
        "marital_status": "married",
    }
