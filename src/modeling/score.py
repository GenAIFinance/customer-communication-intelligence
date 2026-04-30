"""
score.py — Load trained model and score customer cases.

Supports:
  - Single customer dict → intervention score + flag
  - Batch DataFrame → scores for all rows
  - Model metadata retrieval

Used by the FastAPI /score-customer endpoint and Streamlit dashboard.

Run directly:
    python -m src.modeling.score
"""

import pandas as pd
import numpy as np
import joblib
import yaml
from pathlib import Path
from dataclasses import dataclass
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.features.build_features import build_features, get_feature_names, CATEGORICAL_FEATURES


# ── Config ─────────────────────────────────────────────────────────────────────

def _load_config() -> dict:
    config_path = Path(__file__).resolve().parents[2] / "config" / "config.yaml"
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


# ── Score result dataclass ─────────────────────────────────────────────────────

@dataclass
class ScoreResult:
    """Output from scoring a single customer case."""
    customer_id: str
    intervention_score: float   # probability of needing intervention (0-1)
    needs_intervention: int     # 1 = flag for intervention, 0 = no flag
    risk_band: str              # Low / Medium / High — human-readable


def _score_to_risk_band(score: float) -> str:
    """Convert a probability score to a business-readable risk band."""
    if score >= 0.70:
        return "High"
    elif score >= 0.45:
        return "Medium"
    else:
        return "Low"


# ── Model loader ───────────────────────────────────────────────────────────────

def load_model(model_path: str | None = None):
    """
    Load the trained model from disk.

    Args:
        model_path: Path to .joblib file. Defaults to config value.

    Returns:
        Trained sklearn model.

    Raises:
        FileNotFoundError: If model file does not exist.
    """
    cfg = _load_config()
    path = model_path or cfg["model"]["model_output_path"]

    if not Path(path).exists():
        raise FileNotFoundError(
            f"Model not found at '{path}'. "
            "Run 'python -m src.modeling.train_model' first."
        )

    return joblib.load(path)


# ── Feature alignment ──────────────────────────────────────────────────────────

def _align_features(df: pd.DataFrame, expected_columns: list[str]) -> pd.DataFrame:
    """
    Ensure the feature DataFrame has exactly the columns the model expects.

    Adds missing columns as 0 and drops unexpected columns.
    This handles cases where one-hot encoding produces different columns
    for a single-row input vs the full training set.

    Args:
        df:               Feature DataFrame.
        expected_columns: List of column names the model was trained on.

    Returns:
        Aligned DataFrame with exactly the expected columns in the right order.
    """
    for col in expected_columns:
        if col not in df.columns:
            df[col] = 0
    return df[expected_columns]


def _prepare_single(customer: dict, expected_columns: list[str]) -> pd.DataFrame:
    """
    Convert a raw customer dict into a model-ready feature row.

    Args:
        customer:         Dict with raw customer field values.
        expected_columns: Feature columns the model expects.

    Returns:
        Single-row DataFrame ready for model.predict_proba().
    """
    # Add a dummy target so build_features doesn't error
    customer_copy = {**customer, "needs_intervention": 0}

    # Add non-feature columns build_features expects to see and drop
    for col in ["customer_id", "campaign_id", "sent_date"]:
        if col not in customer_copy:
            customer_copy[col] = "UNKNOWN"

    df = pd.DataFrame([customer_copy])

    # Cast categoricals to string for get_dummies consistency
    for col in CATEGORICAL_FEATURES:
        if col in df.columns:
            df[col] = df[col].astype(str)

    X, _ = build_features(df)
    X = _align_features(X, expected_columns)
    return X


# ── Scoring functions ──────────────────────────────────────────────────────────

def score_customer(
    customer: dict,
    model=None,
    expected_columns: list[str] | None = None,
) -> ScoreResult:
    """
    Score a single customer case.

    Args:
        customer:         Dict of raw customer field values.
        model:            Trained model. Loaded from disk if None.
        expected_columns: Feature column names. Inferred if None.

    Returns:
        ScoreResult with score, flag, and risk band.
    """
    cfg = _load_config()
    threshold = cfg["model"]["threshold"]

    if model is None:
        model = load_model()

    if expected_columns is None:
        from src.utils.db import query_df
        df_sample = query_df("SELECT * FROM customer_communications LIMIT 100")
        expected_columns = get_feature_names(df_sample)

    X = _prepare_single(customer, expected_columns)
    proba = float(model.predict_proba(X)[0, 1])
    flag  = int(proba >= threshold)

    return ScoreResult(
        customer_id=customer.get("customer_id", "UNKNOWN"),
        intervention_score=round(proba, 4),
        needs_intervention=flag,
        risk_band=_score_to_risk_band(proba),
    )


def score_batch(
    df: pd.DataFrame,
    model=None,
    model_path: str | None = None,
) -> pd.DataFrame:
    """
    Score all rows in a DataFrame and append score columns.

    Args:
        df:         DataFrame with raw customer fields.
        model:      Trained model. Loaded from disk if None.
        model_path: Path to model file (used if model is None).

    Returns:
        Original DataFrame with added columns:
          - intervention_score
          - predicted_intervention
          - risk_band
    """
    cfg = _load_config()
    threshold = cfg["model"]["threshold"]

    if model is None:
        model = load_model(model_path)

    # Load feature names saved during training for reliable column alignment
    from src.modeling.train_model import load_feature_names
    expected_columns = load_feature_names(model_path)

    X, _ = build_features(df)
    X = _align_features(X, expected_columns)

    probas = model.predict_proba(X)[:, 1]
    flags  = (probas >= threshold).astype(int)

    result = df.copy()
    result["intervention_score"]      = probas.round(4)
    result["predicted_intervention"]  = flags
    result["risk_band"]               = [_score_to_risk_band(p) for p in probas]

    return result


# ── CLI entry point ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    from src.utils.db import query_df

    print("── Single customer scoring ───────────────────────────────")
    sample_customer = {
        "customer_id":             "CUST_TEST_001",
        "segment":                 "Premium",
        "product_type":            "Home",
        "channel":                 "Email",
        "campaign_id":             "CAMP_001",
        "sent_date":               "2026-01-01",
        "opened":                  0,
        "clicked":                 0,
        "response_flag":           0,
        "complaint_flag":          1,
        "escalation_flag":         0,
        "engagement_score":        0.12,
        "sentiment_text":          "negative",
        "premium_bucket":          "High",
        "tenure_months":           36,
        "days_since_last_contact": 75,
        "opt_out_flag":            0,
    }

    result = score_customer(sample_customer)
    print(f"  customer_id        : {result.customer_id}")
    print(f"  intervention_score : {result.intervention_score}")
    print(f"  needs_intervention : {result.needs_intervention}")
    print(f"  risk_band          : {result.risk_band}")

    print("\n── Batch scoring (first 10 rows from DB) ─────────────────")
    df_sample = query_df("SELECT * FROM customer_communications LIMIT 10")
    df_scored = score_batch(df_sample)
    print(df_scored[["customer_id", "intervention_score", "predicted_intervention", "risk_band"]])
    print(f"\n  Risk band distribution:")
    print(df_scored["risk_band"].value_counts())
