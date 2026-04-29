"""
test_data.py — Tests for data generation, validation, and ingestion.

Run with:
    pytest tests/test_data.py -v
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.generate_data import generate_synthetic_data, save_raw_csv
from src.data.validate import validate, ValidationReport, REQUIRED_COLUMNS
from src.data.ingest import clean


# ── Fixtures ───────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def sample_df():
    """Generate a small DataFrame for fast test runs."""
    return generate_synthetic_data(n_rows=500, seed=99)


# ── generate_data tests ────────────────────────────────────────────────────────

class TestGenerateData:

    def test_returns_dataframe(self, sample_df):
        assert isinstance(sample_df, pd.DataFrame)

    def test_correct_row_count(self, sample_df):
        assert len(sample_df) == 500

    def test_all_required_columns_present(self, sample_df):
        for col in REQUIRED_COLUMNS:
            assert col in sample_df.columns, f"Missing column: {col}"

    def test_no_nulls_in_critical_columns(self, sample_df):
        critical = ["customer_id", "segment", "engagement_score", "needs_intervention"]
        for col in critical:
            assert sample_df[col].isna().sum() == 0, f"Nulls found in {col}"

    def test_engagement_score_in_range(self, sample_df):
        assert sample_df["engagement_score"].between(0.0, 1.0).all()

    def test_binary_columns_are_binary(self, sample_df):
        binary_cols = ["opened", "clicked", "response_flag", "complaint_flag",
                       "escalation_flag", "opt_out_flag", "needs_intervention"]
        for col in binary_cols:
            unique_vals = set(sample_df[col].unique())
            assert unique_vals.issubset({0, 1}), f"{col} has non-binary values: {unique_vals}"

    def test_segment_values(self, sample_df):
        allowed = {"Premium", "Standard", "Basic"}
        assert set(sample_df["segment"].unique()).issubset(allowed)

    def test_channel_values(self, sample_df):
        allowed = {"Email", "SMS", "Phone", "Direct Mail"}
        assert set(sample_df["channel"].unique()).issubset(allowed)

    def test_customer_ids_are_unique(self, sample_df):
        assert sample_df["customer_id"].is_unique

    def test_intervention_rate_reasonable(self, sample_df):
        rate = sample_df["needs_intervention"].mean()
        assert 0.05 < rate < 0.80, f"Intervention rate {rate:.1%} outside expected range"

    def test_reproducibility_with_same_seed(self):
        df1 = generate_synthetic_data(n_rows=100, seed=7)
        df2 = generate_synthetic_data(n_rows=100, seed=7)
        pd.testing.assert_frame_equal(df1, df2)

    def test_different_seeds_produce_different_data(self):
        df1 = generate_synthetic_data(n_rows=100, seed=1)
        df2 = generate_synthetic_data(n_rows=100, seed=2)
        assert not df1["engagement_score"].equals(df2["engagement_score"])


# ── validate tests ─────────────────────────────────────────────────────────────

class TestValidate:

    def test_clean_data_passes(self, sample_df):
        report = validate(sample_df)
        assert report.passed is True
        assert len(report.errors) == 0

    def test_missing_column_fails(self, sample_df):
        df_bad = sample_df.drop(columns=["needs_intervention"])
        report = validate(df_bad)
        assert report.passed is False
        assert any("needs_intervention" in e for e in report.errors)

    def test_bad_binary_value_fails(self, sample_df):
        df_bad = sample_df.copy()
        df_bad.loc[0, "complaint_flag"] = 99
        report = validate(df_bad)
        assert report.passed is False
        assert any("complaint_flag" in e for e in report.errors)

    def test_out_of_range_engagement_fails(self, sample_df):
        df_bad = sample_df.copy()
        df_bad.loc[0, "engagement_score"] = 1.5
        report = validate(df_bad)
        assert report.passed is False

    def test_stats_populated(self, sample_df):
        report = validate(sample_df)
        assert "row_count" in report.stats
        assert "intervention_rate" in report.stats
        assert report.stats["row_count"] == 500

    def test_report_summary_string(self, sample_df):
        report = validate(sample_df)
        summary = report.summary()
        assert "PASSED" in summary or "FAILED" in summary


# ── clean (ingest) tests ───────────────────────────────────────────────────────

class TestClean:

    def test_sent_date_is_string(self, sample_df):
        cleaned = clean(sample_df)
        # Accept both "object" and pandas StringDtype
        assert pd.api.types.is_string_dtype(cleaned["sent_date"])

    def test_binary_cols_are_int(self, sample_df):
        cleaned = clean(sample_df)
        assert cleaned["needs_intervention"].dtype in [np.int32, np.int64, int]

    def test_no_nulls_after_clean(self, sample_df):
        cleaned = clean(sample_df)
        assert cleaned.isna().sum().sum() == 0

    def test_clean_does_not_mutate_input(self, sample_df):
        original_len = len(sample_df)
        _ = clean(sample_df)
        assert len(sample_df) == original_len
