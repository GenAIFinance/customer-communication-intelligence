"""
anomaly.py — Lightweight anomaly detection for customer communications data.

Three detectors — all pure pandas, no sklearn dependency:

  1. Segment Engagement Drop  — week-over-week engagement decline per segment
  2. Complaint Spike          — z-score on daily complaint counts
  3. Campaign Underperformance — campaign open rate vs overall median

Each detector returns a structured AnomalyResult so the API and dashboard
can consume a consistent format.

Run directly:
    python -m src.modeling.anomaly
"""

import pandas as pd
import numpy as np
import yaml
from pathlib import Path
from dataclasses import dataclass, field
from datetime import date, timedelta
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.utils.db import query_df


# ── Config ─────────────────────────────────────────────────────────────────────

def _load_config() -> dict:
    config_path = Path(__file__).resolve().parents[2] / "config" / "config.yaml"
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


# ── Result dataclass ───────────────────────────────────────────────────────────

@dataclass
class AnomalyResult:
    """
    Structured output from a single anomaly detector run.

    Attributes:
        detector:    Name of the detector that produced this result.
        anomalies:   List of flagged items with detail dicts.
        summary:     Human-readable summary sentence.
        flagged:     True if at least one anomaly was found.
        metadata:    Raw stats used in detection (for transparency/debugging).
    """
    detector: str
    anomalies: list[dict] = field(default_factory=list)
    summary: str = ""
    flagged: bool = False
    metadata: dict = field(default_factory=dict)


# ── Detector 1: Segment Engagement Drop ───────────────────────────────────────

def detect_segment_engagement_drop(
    df: pd.DataFrame,
    drop_threshold: float | None = None,
) -> AnomalyResult:
    """
    Detect week-over-week engagement score drops per segment.

    Method:
      - Compute mean engagement score per segment for the last 7 days (current week)
      - Compute mean engagement score per segment for the prior 7 days (previous week)
      - Flag segment if drop > threshold (default 15%)

    Args:
        df:             Communications DataFrame with sent_date + engagement_score.
        drop_threshold: Fractional drop to flag (e.g. 0.15 = 15%). Reads config if None.

    Returns:
        AnomalyResult with flagged segments and their drop percentages.
    """
    cfg = _load_config()
    threshold = drop_threshold or cfg["anomaly"]["engagement_drop_threshold"]

    df = df.copy()
    df["sent_date"] = pd.to_datetime(df["sent_date"])

    today = df["sent_date"].max()
    week_start     = today - timedelta(days=6)
    prev_week_start = today - timedelta(days=13)
    prev_week_end   = today - timedelta(days=7)

    current = df[df["sent_date"] >= week_start]
    previous = df[
        (df["sent_date"] >= prev_week_start) &
        (df["sent_date"] <= prev_week_end)
    ]

    current_avg  = current.groupby("segment")["engagement_score"].mean()
    previous_avg = previous.groupby("segment")["engagement_score"].mean()

    anomalies = []
    segments_compared = []

    for segment in current_avg.index:
        if segment not in previous_avg.index:
            continue

        curr_val = current_avg[segment]
        prev_val = previous_avg[segment]

        if prev_val == 0:
            continue

        drop_pct = (prev_val - curr_val) / prev_val
        segments_compared.append(segment)

        if drop_pct > threshold:
            anomalies.append({
                "segment":          segment,
                "current_week_avg": round(curr_val, 4),
                "previous_week_avg": round(prev_val, 4),
                "drop_pct":         round(drop_pct, 4),
                "drop_pct_label":   f"{drop_pct:.1%}",
            })

    flagged = len(anomalies) > 0

    if flagged:
        worst = max(anomalies, key=lambda x: x["drop_pct"])
        summary = (
            f"{len(anomalies)} segment(s) show engagement drops above "
            f"{threshold:.0%}. Worst: {worst['segment']} dropped "
            f"{worst['drop_pct_label']} week-over-week."
        )
    else:
        summary = (
            f"No segment engagement drops above {threshold:.0%} detected "
            f"across {len(segments_compared)} segment(s)."
        )

    return AnomalyResult(
        detector="segment_engagement_drop",
        anomalies=anomalies,
        summary=summary,
        flagged=flagged,
        metadata={
            "threshold":          threshold,
            "segments_compared":  segments_compared,
            "current_week_start": str(week_start.date()),
            "previous_week_start": str(prev_week_start.date()),
        },
    )


# ── Detector 2: Complaint Spike ────────────────────────────────────────────────

def detect_complaint_spike(
    df: pd.DataFrame,
    zscore_threshold: float | None = None,
) -> AnomalyResult:
    """
    Detect days with abnormally high complaint counts using z-score.

    Method:
      - Aggregate complaint_flag by sent_date
      - Compute z-score of daily complaint counts
      - Flag days where z-score > threshold (default 2.0)

    Args:
        df:                Communications DataFrame.
        zscore_threshold:  Z-score cutoff. Reads config if None.

    Returns:
        AnomalyResult with flagged dates and their z-scores.
    """
    cfg = _load_config()
    threshold = zscore_threshold or cfg["anomaly"]["complaint_zscore_threshold"]

    df = df.copy()
    df["sent_date"] = pd.to_datetime(df["sent_date"])

    daily = (
        df.groupby("sent_date")["complaint_flag"]
        .agg(["sum", "count"])
        .rename(columns={"sum": "complaints", "count": "total_sent"})
        .reset_index()
    )

    if len(daily) < 3:
        return AnomalyResult(
            detector="complaint_spike",
            summary="Insufficient data for complaint spike detection (need >= 3 days).",
            metadata={"threshold": threshold},
        )

    mean_complaints = daily["complaints"].mean()
    std_complaints  = daily["complaints"].std()

    if std_complaints == 0:
        return AnomalyResult(
            detector="complaint_spike",
            summary="Complaint counts are uniform — no spike detection possible.",
            metadata={"threshold": threshold, "mean": round(mean_complaints, 2)},
        )

    daily["zscore"] = (daily["complaints"] - mean_complaints) / std_complaints

    flagged_days = daily[daily["zscore"] > threshold].copy()

    anomalies = []
    for _, row in flagged_days.iterrows():
        anomalies.append({
            "date":         str(row["sent_date"].date()),
            "complaints":   int(row["complaints"]),
            "total_sent":   int(row["total_sent"]),
            "complaint_rate": round(row["complaints"] / row["total_sent"], 4),
            "zscore":       round(row["zscore"], 4),
        })

    # Sort by z-score descending
    anomalies = sorted(anomalies, key=lambda x: x["zscore"], reverse=True)
    flagged = len(anomalies) > 0

    if flagged:
        worst = anomalies[0]
        summary = (
            f"{len(anomalies)} day(s) with complaint spikes detected. "
            f"Highest: {worst['date']} with {worst['complaints']} complaints "
            f"(z-score: {worst['zscore']:.2f})."
        )
    else:
        summary = (
            f"No complaint spikes above z-score {threshold} detected. "
            f"Daily average: {mean_complaints:.1f} complaints."
        )

    return AnomalyResult(
        detector="complaint_spike",
        anomalies=anomalies,
        summary=summary,
        flagged=flagged,
        metadata={
            "threshold":        threshold,
            "mean_complaints":  round(mean_complaints, 2),
            "std_complaints":   round(std_complaints, 2),
            "days_analysed":    len(daily),
        },
    )


# ── Detector 3: Campaign Underperformance ──────────────────────────────────────

def detect_campaign_underperformance(
    df: pd.DataFrame,
    open_rate_ratio: float | None = None,
) -> AnomalyResult:
    """
    Detect campaigns whose open rate falls below a fraction of the median.

    Method:
      - Compute open rate per campaign (opened / total sent)
      - Compute median open rate across all campaigns
      - Flag campaigns with open rate < ratio * median (default: < 50% of median)

    Args:
        df:              Communications DataFrame.
        open_rate_ratio: Fraction of median below which a campaign is flagged.
                         Reads config if None.

    Returns:
        AnomalyResult with flagged campaigns and their open rates.
    """
    cfg = _load_config()
    ratio = open_rate_ratio or cfg["anomaly"]["campaign_open_rate_ratio"]

    campaign_stats = (
        df.groupby("campaign_id")
        .agg(
            total_sent=("opened", "count"),
            total_opened=("opened", "sum"),
            total_complaints=("complaint_flag", "sum"),
            avg_engagement=("engagement_score", "mean"),
        )
        .reset_index()
    )

    campaign_stats["open_rate"] = (
        campaign_stats["total_opened"] / campaign_stats["total_sent"]
    ).round(4)

    median_open_rate = campaign_stats["open_rate"].median()
    cutoff = median_open_rate * ratio

    underperforming = campaign_stats[
        campaign_stats["open_rate"] < cutoff
    ].copy()

    anomalies = []
    for _, row in underperforming.iterrows():
        anomalies.append({
            "campaign_id":      row["campaign_id"],
            "open_rate":        round(row["open_rate"], 4),
            "open_rate_label":  f"{row['open_rate']:.1%}",
            "median_open_rate": round(median_open_rate, 4),
            "pct_of_median":    round(row["open_rate"] / median_open_rate, 4) if median_open_rate > 0 else 0,
            "total_sent":       int(row["total_sent"]),
            "complaint_count":  int(row["total_complaints"]),
            "avg_engagement":   round(row["avg_engagement"], 4),
        })

    # Sort by open rate ascending (worst first)
    anomalies = sorted(anomalies, key=lambda x: x["open_rate"])
    flagged = len(anomalies) > 0

    if flagged:
        worst = anomalies[0]
        summary = (
            f"{len(anomalies)} campaign(s) underperforming. "
            f"Median open rate: {median_open_rate:.1%}. "
            f"Worst: {worst['campaign_id']} at {worst['open_rate_label']} "
            f"({worst['pct_of_median']:.0%} of median)."
        )
    else:
        summary = (
            f"All {len(campaign_stats)} campaigns performing above "
            f"{ratio:.0%} of median open rate ({median_open_rate:.1%})."
        )

    return AnomalyResult(
        detector="campaign_underperformance",
        anomalies=anomalies,
        summary=summary,
        flagged=flagged,
        metadata={
            "ratio_threshold":   ratio,
            "median_open_rate":  round(median_open_rate, 4),
            "cutoff_open_rate":  round(cutoff, 4),
            "campaigns_checked": len(campaign_stats),
        },
    )


# ── Run all detectors ──────────────────────────────────────────────────────────

def run_all_detectors(
    df: pd.DataFrame | None = None,
    table_name: str = "customer_communications",
) -> dict[str, AnomalyResult]:
    """
    Run all three anomaly detectors and return a dict of results.

    Args:
        df:         DataFrame to analyse. Loaded from DuckDB if None.
        table_name: DuckDB table to load from (used only if df is None).

    Returns:
        Dict mapping detector name to AnomalyResult.
    """
    if df is None:
        df = query_df(f"SELECT * FROM {table_name}")

    return {
        "segment_engagement_drop":    detect_segment_engagement_drop(df),
        "complaint_spike":            detect_complaint_spike(df),
        "campaign_underperformance":  detect_campaign_underperformance(df),
    }


def anomaly_summary_text(results: dict[str, AnomalyResult]) -> str:
    """
    Produce a short combined summary across all detectors.

    Used by the GenAI summariser as context and by the Streamlit
    Anomaly Monitor page header.

    Args:
        results: Output of run_all_detectors().

    Returns:
        Multi-line summary string.
    """
    lines = []
    for name, result in results.items():
        status = "⚠ FLAGGED" if result.flagged else "✓ OK"
        lines.append(f"  [{status}] {name}: {result.summary}")
    return "\n".join(lines)


# ── CLI entry point ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Loading data from DuckDB...")
    df = query_df("SELECT * FROM customer_communications")
    print(f"  Rows loaded: {len(df):,}\n")

    results = run_all_detectors(df)

    for name, result in results.items():
        print(f"── {name} {'─' * (50 - len(name))}")
        print(f"  Flagged : {result.flagged}")
        print(f"  Summary : {result.summary}")
        if result.anomalies:
            print(f"  Anomalies ({len(result.anomalies)}):")
            for a in result.anomalies[:3]:
                print(f"    {a}")
        print()

    print("── Combined summary ─────────────────────────────────────")
    print(anomaly_summary_text(results))
