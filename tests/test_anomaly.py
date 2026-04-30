"""
test_anomaly.py — Tests for all three anomaly detectors.

Run with:
    python -m pytest tests/test_anomaly.py -v
"""

import pytest
import pandas as pd
import numpy as np
from datetime import date, timedelta
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.modeling.anomaly import (
    detect_segment_engagement_drop,
    detect_complaint_spike,
    detect_campaign_underperformance,
    run_all_detectors,
    anomaly_summary_text,
    AnomalyResult,
)


# ── Fixtures ───────────────────────────────────────────────────────────────────

def _make_base_df(n: int = 200, seed: int = 42) -> pd.DataFrame:
    """Generate a clean base DataFrame with no anomalies."""
    from src.data.generate_data import generate_synthetic_data
    return generate_synthetic_data(n_rows=n, seed=seed)


@pytest.fixture(scope="module")
def base_df():
    return _make_base_df()


def _make_dated_df(
    n_days: int = 20,
    rows_per_day: int = 10,
    complaint_spike_day: int | None = None,
    engagement_drop_segment: str | None = None,
) -> pd.DataFrame:
    """
    Build a controlled DataFrame with known dates for deterministic tests.

    Args:
        n_days:                  Number of days of data.
        rows_per_day:            Rows per day.
        complaint_spike_day:     Day index (0-based) to inject a complaint spike.
        engagement_drop_segment: Segment to give low recent engagement.
    """
    today = date.today()
    rows = []
    for day_offset in range(n_days):
        d = today - timedelta(days=n_days - day_offset)
        for i in range(rows_per_day):
            segment = "Premium" if i % 3 == 0 else ("Standard" if i % 3 == 1 else "Basic")
            complaint = 0

            # Inject spike on specific day
            if complaint_spike_day is not None and day_offset == complaint_spike_day:
                complaint = 1

            # Low engagement for a segment in the last 7 days
            if (
                engagement_drop_segment
                and segment == engagement_drop_segment
                and day_offset >= n_days - 7
            ):
                eng = 0.10
            else:
                eng = 0.65

            rows.append({
                "customer_id":             f"CUST_{day_offset:03d}_{i:03d}",
                "segment":                 segment,
                "product_type":            "Auto",
                "channel":                 "Email",
                "campaign_id":             f"CAMP_{(i % 3) + 1:03d}",
                "sent_date":               str(d),
                "opened":                  1 if eng > 0.5 else 0,
                "clicked":                 1 if eng > 0.6 else 0,
                "response_flag":           1 if eng > 0.6 else 0,
                "complaint_flag":          complaint,
                "escalation_flag":         0,
                "engagement_score":        eng,
                "sentiment_text":          "positive" if eng > 0.5 else "negative",
                "premium_bucket":          "Mid",
                "tenure_months":           24,
                "days_since_last_contact": n_days - day_offset,
                "opt_out_flag":            0,
                "needs_intervention":      0,
            })

    return pd.DataFrame(rows)


# ── AnomalyResult tests ────────────────────────────────────────────────────────

class TestAnomalyResult:

    def test_default_not_flagged(self):
        r = AnomalyResult(detector="test")
        assert r.flagged is False
        assert r.anomalies == []

    def test_flagged_when_anomalies_set(self):
        r = AnomalyResult(detector="test", anomalies=[{"x": 1}], flagged=True)
        assert r.flagged is True
        assert len(r.anomalies) == 1


# ── Segment engagement drop tests ─────────────────────────────────────────────

class TestSegmentEngagementDrop:

    def test_returns_anomaly_result(self, base_df):
        result = detect_segment_engagement_drop(base_df)
        assert isinstance(result, AnomalyResult)
        assert result.detector == "segment_engagement_drop"

    def test_summary_is_string(self, base_df):
        result = detect_segment_engagement_drop(base_df)
        assert isinstance(result.summary, str)
        assert len(result.summary) > 0

    def test_detects_real_drop(self):
        """Inject a 50% engagement drop in Premium — must be flagged."""
        df = _make_dated_df(
            n_days=20,
            rows_per_day=15,
            engagement_drop_segment="Premium"
        )
        result = detect_segment_engagement_drop(df, drop_threshold=0.15)
        assert result.flagged is True
        flagged_segments = [a["segment"] for a in result.anomalies]
        assert "Premium" in flagged_segments

    def test_no_flag_when_stable(self):
        """Uniform engagement across weeks should not flag any segment."""
        df = _make_dated_df(n_days=20, rows_per_day=15)
        result = detect_segment_engagement_drop(df, drop_threshold=0.15)
        assert result.flagged is False

    def test_anomaly_has_required_keys(self):
        df = _make_dated_df(n_days=20, rows_per_day=15, engagement_drop_segment="Basic")
        result = detect_segment_engagement_drop(df, drop_threshold=0.10)
        if result.anomalies:
            required = {"segment", "current_week_avg", "previous_week_avg", "drop_pct", "drop_pct_label"}
            assert required.issubset(set(result.anomalies[0].keys()))

    def test_drop_pct_is_positive_when_flagged(self):
        df = _make_dated_df(n_days=20, rows_per_day=15, engagement_drop_segment="Standard")
        result = detect_segment_engagement_drop(df, drop_threshold=0.10)
        for a in result.anomalies:
            assert a["drop_pct"] > 0


# ── Complaint spike tests ──────────────────────────────────────────────────────

class TestComplaintSpike:

    def test_returns_anomaly_result(self, base_df):
        result = detect_complaint_spike(base_df)
        assert isinstance(result, AnomalyResult)
        assert result.detector == "complaint_spike"

    def test_detects_spike(self):
        """Day 5 has 100% complaint rate — must be flagged."""
        df = _make_dated_df(n_days=20, rows_per_day=10, complaint_spike_day=5)
        result = detect_complaint_spike(df, zscore_threshold=2.0)
        assert result.flagged is True

    def test_no_spike_when_uniform(self):
        """Zero complaints throughout — no spike possible."""
        df = _make_dated_df(n_days=20, rows_per_day=10)
        result = detect_complaint_spike(df, zscore_threshold=2.0)
        assert result.flagged is False

    def test_anomaly_has_required_keys(self):
        df = _make_dated_df(n_days=20, rows_per_day=10, complaint_spike_day=3)
        result = detect_complaint_spike(df, zscore_threshold=1.5)
        if result.anomalies:
            required = {"date", "complaints", "total_sent", "complaint_rate", "zscore"}
            assert required.issubset(set(result.anomalies[0].keys()))

    def test_zscore_above_threshold_when_flagged(self):
        df = _make_dated_df(n_days=20, rows_per_day=10, complaint_spike_day=5)
        result = detect_complaint_spike(df, zscore_threshold=2.0)
        for a in result.anomalies:
            assert a["zscore"] > 2.0

    def test_insufficient_data_handled(self):
        """Only 2 days of data — should return gracefully, not crash."""
        df = _make_dated_df(n_days=2, rows_per_day=5)
        result = detect_complaint_spike(df)
        assert isinstance(result, AnomalyResult)
        assert result.flagged is False

    def test_anomalies_sorted_by_zscore_desc(self):
        df = _make_dated_df(n_days=30, rows_per_day=10, complaint_spike_day=5)
        result = detect_complaint_spike(df, zscore_threshold=1.0)
        if len(result.anomalies) >= 2:
            scores = [a["zscore"] for a in result.anomalies]
            assert scores == sorted(scores, reverse=True)


# ── Campaign underperformance tests ───────────────────────────────────────────

class TestCampaignUnderperformance:

    def test_returns_anomaly_result(self, base_df):
        result = detect_campaign_underperformance(base_df)
        assert isinstance(result, AnomalyResult)
        assert result.detector == "campaign_underperformance"

    def test_detects_bad_campaign(self):
        """Inject one campaign with 0% open rate — must be flagged."""
        df = _make_dated_df(n_days=20, rows_per_day=30)
        # Override one campaign to never open
        df.loc[df["campaign_id"] == "CAMP_001", "opened"] = 0
        result = detect_campaign_underperformance(df, open_rate_ratio=0.5)
        assert result.flagged is True
        flagged_ids = [a["campaign_id"] for a in result.anomalies]
        assert "CAMP_001" in flagged_ids

    def test_anomaly_has_required_keys(self, base_df):
        result = detect_campaign_underperformance(base_df)
        if result.anomalies:
            required = {
                "campaign_id", "open_rate", "open_rate_label",
                "median_open_rate", "pct_of_median", "total_sent"
            }
            assert required.issubset(set(result.anomalies[0].keys()))

    def test_anomalies_sorted_worst_first(self):
        df = _make_dated_df(n_days=20, rows_per_day=30)
        df.loc[df["campaign_id"] == "CAMP_001", "opened"] = 0
        result = detect_campaign_underperformance(df, open_rate_ratio=0.5)
        if len(result.anomalies) >= 2:
            rates = [a["open_rate"] for a in result.anomalies]
            assert rates == sorted(rates)

    def test_metadata_contains_median(self, base_df):
        result = detect_campaign_underperformance(base_df)
        assert "median_open_rate" in result.metadata
        assert result.metadata["median_open_rate"] > 0


# ── run_all_detectors tests ────────────────────────────────────────────────────

class TestRunAllDetectors:

    def test_returns_dict_of_three(self, base_df):
        results = run_all_detectors(df=base_df)
        assert isinstance(results, dict)
        assert len(results) == 3

    def test_all_keys_present(self, base_df):
        results = run_all_detectors(df=base_df)
        assert "segment_engagement_drop"   in results
        assert "complaint_spike"           in results
        assert "campaign_underperformance" in results

    def test_all_values_are_anomaly_results(self, base_df):
        results = run_all_detectors(df=base_df)
        for v in results.values():
            assert isinstance(v, AnomalyResult)

    def test_summary_text_contains_all_detectors(self, base_df):
        results = run_all_detectors(df=base_df)
        summary = anomaly_summary_text(results)
        assert "segment_engagement_drop"   in summary
        assert "complaint_spike"           in summary
        assert "campaign_underperformance" in summary

    def test_summary_text_is_string(self, base_df):
        results = run_all_detectors(df=base_df)
        summary = anomaly_summary_text(results)
        assert isinstance(summary, str)
        assert len(summary) > 0
