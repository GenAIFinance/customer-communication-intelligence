"""
build_features.py — Feature engineering for the intervention model.

Takes the raw communications DataFrame and produces a model-ready
feature matrix. All transformations are explicit and documented so
they can be explained to a business audience.

Run directly:
    python -m src.features.build_features
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.utils.db import query_df, write_df

# ── Feature groups ─────────────────────────────────────────────────────────────

# Columns passed through as-is (already numeric)
PASSTHROUGH_FEATURES = [
    "opened",
    "clicked",
    "response_flag",
    "complaint_flag",
    "escalation_flag",
    "engagement_score",
    "tenure_months",
    "days_since_last_contact",
    "opt_out_flag",
]

# Categorical columns to one-hot encode
CATEGORICAL_FEATURES = [
    "segment",
    "product_type",
    "channel",
    "premium_bucket",
    "sentiment_text",
]

# Target column
TARGET = "needs_intervention"


# ── Feature engineering functions ─────────────────────────────────────────────

def encode_categoricals(df: pd.DataFrame, drop_first: bool = True) -> pd.DataFrame:
    """
    One-hot encode categorical columns.

    Args:
        df:         DataFrame containing categorical columns.
        drop_first: If True (default for training), drop first level per category
                    to avoid multicollinearity. Set False for single-row inference
                    so the one observed level is not silently dropped — _align_features
                    will remove any baseline columns not in the training schema.

    Returns:
        DataFrame with original categoricals replaced by dummy columns.
    """
    df = df.copy()
    dummies = pd.get_dummies(
        df[CATEGORICAL_FEATURES],
        drop_first=drop_first,
        dtype=int,
    )
    df = df.drop(columns=CATEGORICAL_FEATURES)
    df = pd.concat([df, dummies], axis=1)
    return df


def add_interaction_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add lightweight interaction features that capture business logic.

    Features added:
    - contact_but_no_response: contacted recently but zero response
    - high_risk_combo: complaint AND escalation together
    - engagement_x_tenure: engagement score weighted by tenure band
    - low_engagement_long_silence: low score + many days since contact

    Args:
        df: DataFrame with base numeric features.

    Returns:
        DataFrame with additional interaction columns.
    """
    df = df.copy()

    # Recently contacted but completely silent
    df["contact_but_no_response"] = (
        (df["days_since_last_contact"] < 30) & (df["response_flag"] == 0)
    ).astype(int)

    # Double risk flag — both complaint and escalation raised
    df["high_risk_combo"] = (
        (df["complaint_flag"] == 1) & (df["escalation_flag"] == 1)
    ).astype(int)

    # Engagement weighted by tenure (long-tenure low-engagers are higher risk)
    df["engagement_x_tenure"] = (
        df["engagement_score"] * np.log1p(df["tenure_months"])
    ).round(4)

    # Low engagement + long silence = dormancy signal
    df["low_engagement_long_silence"] = (
        (df["engagement_score"] < 0.35) & (df["days_since_last_contact"] > 60)
    ).astype(int)

    return df


def drop_non_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove columns that should not be used as model inputs.

    Drops:
    - Identifier columns (customer_id, campaign_id)
    - Date columns (not usable as-is without further engineering)
    - Target column (returned separately)

    Args:
        df: Full DataFrame.

    Returns:
        DataFrame with only model input columns.
    """
    drop_cols = ["customer_id", "campaign_id", "sent_date", TARGET]
    existing_drops = [c for c in drop_cols if c in df.columns]
    return df.drop(columns=existing_drops)


# ── Main pipeline ──────────────────────────────────────────────────────────────

def build_features(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """
    Full feature engineering pipeline.

    Steps:
    1. Add interaction features
    2. One-hot encode categoricals
    3. Drop non-feature columns
    4. Extract target series

    Args:
        df: Raw communications DataFrame (output of ingest pipeline).

    Returns:
        Tuple of (X: feature DataFrame, y: target Series).
    """
    # Extract target before transforming
    y = df[TARGET].copy()

    # Add interaction features on full df (needs categorical cols still present)
    df_engineered = add_interaction_features(df)

    # Encode categoricals
    df_encoded = encode_categoricals(df_engineered)

    # Drop non-feature columns
    X = drop_non_features(df_encoded)

    # Ensure all columns are numeric — safety check
    non_numeric = X.select_dtypes(exclude=[np.number]).columns.tolist()
    if non_numeric:
        raise ValueError(f"Non-numeric columns found after feature engineering: {non_numeric}")

    return X, y


def build_features_from_db(table_name: str = "customer_communications") -> tuple[pd.DataFrame, pd.Series]:
    """
    Load data from DuckDB and run the full feature pipeline.

    Args:
        table_name: DuckDB table to read from.

    Returns:
        Tuple of (X, y).
    """
    df = query_df(f"SELECT * FROM {table_name}")
    return build_features(df)


def get_feature_names(df: pd.DataFrame) -> list[str]:
    """
    Return the list of feature column names after engineering.

    Useful for model interpretation and API schema validation.

    Args:
        df: Raw DataFrame (used to infer encoded column names).

    Returns:
        List of feature column name strings.
    """
    X, _ = build_features(df)
    return X.columns.tolist()


# ── CLI entry point ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Loading data from DuckDB...")
    X, y = build_features_from_db()

    print(f"\nFeature matrix shape : {X.shape}")
    print(f"Target distribution  :")
    print(f"  needs_intervention=0 : {(y==0).sum():,} ({(y==0).mean():.1%})")
    print(f"  needs_intervention=1 : {(y==1).sum():,} ({(y==1).mean():.1%})")
    print(f"\nFeature columns ({len(X.columns)}):")
    for col in X.columns:
        print(f"  {col}")
    print(f"\nSample (3 rows):")
    print(X.head(3).T)
