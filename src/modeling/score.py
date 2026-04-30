"""
score.py — Load trained model and score customer cases.

Supports:
  - Single customer dict → intervention score + flag
  - Batch DataFrame → scores for all rows (target column not required)
  - Model metadata retrieval

Used by the FastAPI /score-customer endpoint and Streamlit dashboard.

Codex fixes applied (v1.1):
  1. score_batch: target column injected internally — callers need not supply it
  2. score_batch: feature names loaded from artifact only when model loaded from disk
  3. score_customer: expected_columns defaults to persisted training schema, not DB sample

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

from src.features.build_features import build_features, CATEGORICAL_FEATURES, TARGET


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


# ── Model and feature name loaders ─────────────────────────────────────────────

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


def _load_feature_names_from_artifact(model_path: str | None = None) -> list[str]:
    """
    Load feature names saved alongside the model artifact at training time.

    This is the authoritative source of column order and names —
    more reliable than inferring from a DB sample which may miss categories.

    Args:
        model_path: Base model .joblib path. Feature file is co-located.

    Returns:
        List of feature column names in training order.

    Raises:
        FileNotFoundError: If feature names artifact does not exist.
    """
    cfg = _load_config()
    base_path = model_path or cfg["model"]["model_output_path"]
    names_path = str(base_path).replace(".joblib", "_features.joblib")

    if not Path(names_path).exists():
        raise FileNotFoundError(
            f"Feature names artifact not found at '{names_path}'. "
            "Re-run 'python -m src.modeling.train_model' to regenerate it."
        )

    return joblib.load(names_path)


# ── Feature preparation helpers ────────────────────────────────────────────────

def _align_features(df: pd.DataFrame, expected_columns: list[str]) -> pd.DataFrame:
    """
    Ensure the feature DataFrame has exactly the columns the model expects.

    Adds missing columns as 0 (handles unseen categories in small/single inputs)
    and drops unexpected columns. Returns columns in training order.

    Args:
        df:               Feature DataFrame (output of build_features).
        expected_columns: Ordered list of column names the model was trained on.

    Returns:
        Aligned DataFrame ready for model.predict_proba().
    """
    df = df.copy()
    for col in expected_columns:
        if col not in df.columns:
            df[col] = 0
    return df[expected_columns]


def _inject_target_if_missing(df: pd.DataFrame) -> pd.DataFrame:
    """
    Inject a dummy target column if not present in the input DataFrame.

    build_features() reads the target column to split X and y.
    In production/inference scenarios the label is unknown — injecting
    a placeholder 0 prevents KeyError without affecting feature engineering.

    Args:
        df: Raw input DataFrame, possibly without the target column.

    Returns:
        DataFrame guaranteed to contain the target column (placeholder value=0).
    """
    if TARGET not in df.columns:
        df = df.copy()
        df[TARGET] = 0
    return df


def _prepare_single(customer: dict, expected_columns: list[str]) -> pd.DataFrame:
    """
    Convert a raw customer dict into a model-ready single-row feature matrix.

    Handles:
    - Missing target column (injected as placeholder — fix #1 for single path)
    - Missing non-feature columns (customer_id, campaign_id, sent_date)
    - Categorical encoding alignment via _align_features

    Args:
        customer:         Dict with raw customer field values.
        expected_columns: Ordered feature columns the model expects.

    Returns:
        Single-row DataFrame ready for model.predict_proba().
    """
    customer_copy = dict(customer)

    # Inject placeholder target so build_features doesn't KeyError
    customer_copy.setdefault(TARGET, 0)

    # Inject placeholder non-feature columns that build_features will drop
    for col in ["customer_id", "campaign_id", "sent_date"]:
        customer_copy.setdefault(col, "UNKNOWN")

    df = pd.DataFrame([customer_copy])

    # Ensure categoricals are string type for consistent get_dummies behaviour
    for col in CATEGORICAL_FEATURES:
        if col in df.columns:
            df[col] = df[col].astype(str)

    X, _ = build_features(df)
    return _align_features(X, expected_columns)


# ── Scoring functions ──────────────────────────────────────────────────────────

def score_customer(
    customer: dict,
    model=None,
    model_path: str | None = None,
    expected_columns: list[str] | None = None,
) -> ScoreResult:
    """
    Score a single customer case.

    The target column (needs_intervention) is NOT required in the input dict.

    Args:
        customer:         Dict of raw customer field values.
        model:            Trained model object. Loaded from disk if None.
        model_path:       Path to model .joblib file.
        expected_columns: Feature column names in training order.
                          Loaded from persisted artifact if None (fix #3) —
                          never inferred from a DB sample.

    Returns:
        ScoreResult with intervention_score, needs_intervention flag, risk_band.
    """
    cfg = _load_config()
    threshold = cfg["model"]["threshold"]

    if model is None:
        model = load_model(model_path)

    # FIX #3: use persisted training schema — not a DB sample
    if expected_columns is None:
        expected_columns = _load_feature_names_from_artifact(model_path)

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
    expected_columns: list[str] | None = None,
) -> pd.DataFrame:
    """
    Score all rows in a DataFrame and append prediction columns.

    The target column (needs_intervention) is NOT required in the input —
    it is injected internally as a placeholder if absent (fix #1).

    Feature names are loaded from the persisted artifact only when the model
    is also loaded from disk — if a model object is passed in directly,
    the artifact load is skipped unless expected_columns is also None (fix #2).

    Args:
        df:               DataFrame with raw customer fields.
                          Target column is optional for inference use.
        model:            Trained model object. Loaded from disk if None.
        model_path:       Path to model .joblib file.
        expected_columns: Feature columns in training order.
                          If None, loaded from artifact at model_path.

    Returns:
        Original DataFrame with three added columns:
          - intervention_score      (float 0-1)
          - predicted_intervention  (int 0 or 1)
          - risk_band               (str: Low / Medium / High)
    """
    cfg = _load_config()
    threshold = cfg["model"]["threshold"]

    # FIX #2: only hit the artifact when model itself comes from disk
    if model is None:
        model = load_model(model_path)
        if expected_columns is None:
            expected_columns = _load_feature_names_from_artifact(model_path)
    else:
        if expected_columns is None:
            # Model provided in-memory but no columns given — try artifact
            expected_columns = _load_feature_names_from_artifact(model_path)

    # FIX #1: inject dummy target so build_features doesn't KeyError
    df_input = _inject_target_if_missing(df)

    X, _ = build_features(df_input)
    X = _align_features(X, expected_columns)

    probas = model.predict_proba(X)[:, 1]
    flags  = (probas >= threshold).astype(int)

    result = df.copy()   # return original df without injected target
    result["intervention_score"]     = probas.round(4)
    result["predicted_intervention"] = flags
    result["risk_band"]              = [_score_to_risk_band(p) for p in probas]

    return result


# ── CLI entry point ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    from src.utils.db import query_df

    print("── Single customer scoring (no target column) ────────────")
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
        # needs_intervention intentionally omitted — real inference scenario
    }

    result = score_customer(sample_customer)
    print(f"  customer_id        : {result.customer_id}")
    print(f"  intervention_score : {result.intervention_score}")
    print(f"  needs_intervention : {result.needs_intervention}")
    print(f"  risk_band          : {result.risk_band}")

    print("\n── Batch scoring WITH target column (labelled data) ──────")
    df_labelled = query_df("SELECT * FROM customer_communications LIMIT 10")
    df_scored = score_batch(df_labelled)
    print(df_scored[["customer_id", "intervention_score", "predicted_intervention", "risk_band"]])

    print("\n── Batch scoring WITHOUT target column (inference) ───────")
    df_inference = df_labelled.drop(columns=["needs_intervention"])
    df_scored2 = score_batch(df_inference)
    print(df_scored2[["customer_id", "intervention_score", "predicted_intervention", "risk_band"]])
    print(f"\n  Risk band distribution:")
    print(df_scored2["risk_band"].value_counts())
