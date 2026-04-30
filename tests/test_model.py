"""
test_model.py — Tests for feature engineering, model training, and scoring.

Run with:
    pytest tests/test_model.py -v
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.generate_data import generate_synthetic_data
from src.features.build_features import (
    build_features,
    encode_categoricals,
    add_interaction_features,
    CATEGORICAL_FEATURES,
    PASSTHROUGH_FEATURES,
    TARGET,
)
from src.modeling.score import score_customer, score_batch, _score_to_risk_band, ScoreResult


# ── Fixtures ───────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def raw_df():
    return generate_synthetic_data(n_rows=300, seed=42)


@pytest.fixture(scope="module")
def features(raw_df):
    X, y = build_features(raw_df)
    return X, y


@pytest.fixture(scope="module")
def trained_model(raw_df):
    """Train a quick model on small data for test speed."""
    from sklearn.ensemble import RandomForestClassifier
    X, y = build_features(raw_df)
    model = RandomForestClassifier(n_estimators=10, max_depth=4, random_state=42)
    model.fit(X, y)
    return model, X.columns.tolist()


@pytest.fixture
def sample_customer():
    return {
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


# ── Feature engineering tests ──────────────────────────────────────────────────

class TestBuildFeatures:

    def test_returns_tuple(self, raw_df):
        result = build_features(raw_df)
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_X_is_dataframe(self, features):
        X, y = features
        assert isinstance(X, pd.DataFrame)

    def test_y_is_series(self, features):
        X, y = features
        assert isinstance(y, pd.Series)

    def test_target_not_in_X(self, features):
        X, _ = features
        assert TARGET not in X.columns

    def test_customer_id_not_in_X(self, features):
        X, _ = features
        assert "customer_id" not in X.columns

    def test_all_columns_numeric(self, features):
        X, _ = features
        non_numeric = X.select_dtypes(exclude=[np.number]).columns.tolist()
        assert len(non_numeric) == 0, f"Non-numeric columns: {non_numeric}"

    def test_no_nulls_in_features(self, features):
        X, _ = features
        assert X.isna().sum().sum() == 0

    def test_row_count_preserved(self, raw_df, features):
        X, y = features
        assert len(X) == len(raw_df)
        assert len(y) == len(raw_df)

    def test_passthrough_features_present(self, features):
        X, _ = features
        for col in PASSTHROUGH_FEATURES:
            assert col in X.columns, f"Missing passthrough feature: {col}"

    def test_categorical_columns_removed(self, features):
        X, _ = features
        for col in CATEGORICAL_FEATURES:
            assert col not in X.columns, f"Categorical column not encoded: {col}"


class TestInteractionFeatures:

    def test_contact_but_no_response_is_binary(self, raw_df):
        df = add_interaction_features(raw_df)
        assert set(df["contact_but_no_response"].unique()).issubset({0, 1})

    def test_high_risk_combo_is_binary(self, raw_df):
        df = add_interaction_features(raw_df)
        assert set(df["high_risk_combo"].unique()).issubset({0, 1})

    def test_engagement_x_tenure_is_positive(self, raw_df):
        df = add_interaction_features(raw_df)
        assert (df["engagement_x_tenure"] >= 0).all()

    def test_low_engagement_long_silence_is_binary(self, raw_df):
        df = add_interaction_features(raw_df)
        assert set(df["low_engagement_long_silence"].unique()).issubset({0, 1})


# ── Scoring tests ──────────────────────────────────────────────────────────────

class TestScoring:

    def test_risk_band_high(self):
        assert _score_to_risk_band(0.80) == "High"

    def test_risk_band_medium(self):
        assert _score_to_risk_band(0.55) == "Medium"

    def test_risk_band_low(self):
        assert _score_to_risk_band(0.20) == "Low"

    def test_risk_band_boundary_high(self):
        assert _score_to_risk_band(0.70) == "High"

    def test_risk_band_boundary_medium(self):
        assert _score_to_risk_band(0.45) == "Medium"

    def test_score_customer_returns_score_result(self, sample_customer, trained_model):
        model, feature_names = trained_model
        result = score_customer(sample_customer, model=model, expected_columns=feature_names)
        assert isinstance(result, ScoreResult)

    def test_score_customer_score_in_range(self, sample_customer, trained_model):
        model, feature_names = trained_model
        result = score_customer(sample_customer, model=model, expected_columns=feature_names)
        assert 0.0 <= result.intervention_score <= 1.0

    def test_score_customer_flag_is_binary(self, sample_customer, trained_model):
        model, feature_names = trained_model
        result = score_customer(sample_customer, model=model, expected_columns=feature_names)
        assert result.needs_intervention in [0, 1]

    def test_score_customer_risk_band_valid(self, sample_customer, trained_model):
        model, feature_names = trained_model
        result = score_customer(sample_customer, model=model, expected_columns=feature_names)
        assert result.risk_band in ["Low", "Medium", "High"]

    def test_score_customer_id_preserved(self, sample_customer, trained_model):
        model, feature_names = trained_model
        result = score_customer(sample_customer, model=model, expected_columns=feature_names)
        assert result.customer_id == "CUST_TEST_001"

    def test_score_batch_adds_columns(self, raw_df, trained_model):
        model, feature_names = trained_model
        # Temporarily monkeypatch load_feature_names
        import src.modeling.score as score_mod
        original = score_mod.__dict__.get("load_feature_names")
        
        df_scored = score_batch(raw_df, model=model)
        assert "intervention_score" in df_scored.columns
        assert "predicted_intervention" in df_scored.columns
        assert "risk_band" in df_scored.columns

    def test_score_batch_preserves_row_count(self, raw_df, trained_model):
        model, _ = trained_model
        df_scored = score_batch(raw_df, model=model)
        assert len(df_scored) == len(raw_df)

    def test_high_risk_customer_scores_high(self, trained_model):
        """A customer with complaint + escalation + no engagement should score high."""
        model, feature_names = trained_model
        risky_customer = {
            "customer_id": "CUST_RISKY",
            "segment": "Basic",
            "product_type": "Auto",
            "channel": "Email",
            "campaign_id": "CAMP_001",
            "sent_date": "2026-01-01",
            "opened": 0,
            "clicked": 0,
            "response_flag": 0,
            "complaint_flag": 1,
            "escalation_flag": 1,
            "engagement_score": 0.05,
            "sentiment_text": "negative",
            "premium_bucket": "Low",
            "tenure_months": 6,
            "days_since_last_contact": 90,
            "opt_out_flag": 0,
        }
        result = score_customer(risky_customer, model=model, expected_columns=feature_names)
        assert result.intervention_score > 0.5, "High-risk customer should score above 0.5"

    def test_low_risk_customer_scores_low(self, trained_model):
        """A highly engaged customer with no flags should score low."""
        model, feature_names = trained_model
        safe_customer = {
            "customer_id": "CUST_SAFE",
            "segment": "Premium",
            "product_type": "Life",
            "channel": "Phone",
            "campaign_id": "CAMP_002",
            "sent_date": "2026-01-01",
            "opened": 1,
            "clicked": 1,
            "response_flag": 1,
            "complaint_flag": 0,
            "escalation_flag": 0,
            "engagement_score": 0.95,
            "sentiment_text": "positive",
            "premium_bucket": "High",
            "tenure_months": 84,
            "days_since_last_contact": 5,
            "opt_out_flag": 0,
        }
        result = score_customer(safe_customer, model=model, expected_columns=feature_names)
        assert result.intervention_score < 0.5, "Low-risk customer should score below 0.5"
