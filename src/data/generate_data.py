"""
generate_data.py — Synthetic customer communication data generator.

Produces a realistic but entirely fictional dataset of ~5,000 rows.
All values are randomly generated using a fixed seed for reproducibility.
No real customer data is used or referenced.

Run directly:
    python -m src.data.generate_data
"""

import numpy as np
import pandas as pd
from pathlib import Path
from datetime import date, timedelta
import yaml
import os
import sys

# Allow running as a script from project root
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from dotenv import load_dotenv
load_dotenv()

# ── Config ─────────────────────────────────────────────────────────────────────

def _load_config() -> dict:
    config_path = Path(__file__).resolve().parents[2] / "config" / "config.yaml"
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


# ── Generators ─────────────────────────────────────────────────────────────────

def _make_campaign_ids(n_campaigns: int, prefix: str) -> list[str]:
    """Generate campaign ID labels like CAMP_001, CAMP_002 ..."""
    return [f"{prefix}_{str(i).zfill(3)}" for i in range(1, n_campaigns + 1)]


def _compute_engagement_score(
    opened: np.ndarray,
    clicked: np.ndarray,
    response_flag: np.ndarray,
    complaint_flag: np.ndarray,
    opt_out_flag: np.ndarray,
) -> np.ndarray:
    """
    Composite engagement score in [0.0, 1.0].

    Weights:
        opened        → 0.35
        clicked       → 0.30
        response_flag → 0.25
        complaint     → -0.15 penalty
        opt_out       → -0.10 penalty
    """
    score = (
        0.35 * opened
        + 0.30 * clicked
        + 0.25 * response_flag
        - 0.15 * complaint_flag
        - 0.10 * opt_out_flag
    )
    # Add small gaussian noise and clip to [0, 1]
    noise = np.zeros(len(opened))  # deterministic for reproducibility
    return np.clip(score + noise, 0.0, 1.0).round(4)


def _compute_needs_intervention(
    engagement_score: np.ndarray,
    opened: np.ndarray,
    complaint_flag: np.ndarray,
    escalation_flag: np.ndarray,
    days_since_last_contact: np.ndarray,
    response_flag: np.ndarray,
    opt_out_flag: np.ndarray,
    cfg: dict,
) -> np.ndarray:
    """
    Deterministic business rule for the intervention target.

    Flag = 1 when ANY of the following are true:
      1. Low engagement AND no open event
      2. Complaint received
      3. Escalation raised
      4. Dormant customer (long silence, no response, still opted in)
    """
    eng_threshold = cfg["target"]["engagement_threshold"]  # now 0.25 in config
    days_threshold = cfg["target"]["days_since_contact_threshold"]

    rule_low_engagement   = (engagement_score < eng_threshold) & (opened == 0) & (days_since_last_contact < 45)
    rule_complaint        = complaint_flag == 1
    rule_escalation       = escalation_flag == 1
    rule_dormant          = (
        (days_since_last_contact > days_threshold)
        & (response_flag == 0)
        & (opt_out_flag == 0)
        & (engagement_score < 0.5)
    )

    return (
        rule_low_engagement | rule_complaint | rule_escalation | rule_dormant
    ).astype(int)


def _random_dates(start: date, end: date, n: int, rng: np.random.Generator) -> list[date]:
    """Generate n random dates between start and end (inclusive)."""
    delta_days = (end - start).days
    offsets = rng.integers(0, delta_days + 1, size=n)
    return [start + timedelta(days=int(d)) for d in offsets]


def _sentiment_from_score(engagement_score: np.ndarray) -> list[str]:
    """Assign a sentiment label based on engagement score band."""
    labels = []
    for s in engagement_score:
        if s >= 0.70:
            labels.append("positive")
        elif s >= 0.45:
            labels.append("neutral")
        elif s >= 0.25:
            labels.append("mixed")
        else:
            labels.append("negative")
    return labels


# ── Main generator ─────────────────────────────────────────────────────────────

def generate_synthetic_data(n_rows: int | None = None, seed: int | None = None) -> pd.DataFrame:
    """
    Generate a synthetic customer communications DataFrame.

    Args:
        n_rows: Number of rows. Defaults to config value (5000).
        seed:   Random seed. Defaults to config value (42).

    Returns:
        pd.DataFrame with all schema columns including needs_intervention.
    """
    cfg = _load_config()
    n_rows = n_rows or cfg["data"]["synthetic_row_count"]
    seed   = seed   or cfg["data"]["random_seed"]

    rng = np.random.default_rng(seed)

    # ── Categorical fields ──
    segments       = rng.choice(cfg["segments"],       size=n_rows, p=[0.20, 0.50, 0.30])
    product_types  = rng.choice(cfg["product_types"],  size=n_rows)
    channels       = rng.choice(cfg["channels"],       size=n_rows, p=[0.45, 0.25, 0.20, 0.10])
    premium_bkts   = rng.choice(cfg["premium_buckets"], size=n_rows, p=[0.40, 0.40, 0.20])

    campaign_pool  = _make_campaign_ids(cfg["campaigns"]["count"], cfg["campaigns"]["prefix"])
    campaign_ids   = rng.choice(campaign_pool, size=n_rows)

    # ── Numeric fields ──
    tenure_months         = rng.integers(1, 121, size=n_rows)
    days_since_last_contact = rng.integers(0, 181, size=n_rows)

    # Behavioral flags — correlated slightly with channel and segment
    base_open_rate   = np.where(channels == "Email",  0.38,
                       np.where(channels == "SMS",    0.55,
                       np.where(channels == "Phone",  0.70, 0.30)))
    opened           = rng.binomial(1, base_open_rate)

    clicked          = rng.binomial(1, np.clip(base_open_rate * 0.45, 0, 1)) * opened
    response_flag    = rng.binomial(1, 0.25, size=n_rows)
    complaint_flag   = rng.binomial(1, 0.05, size=n_rows)   # ~5% complaint rate
    escalation_flag  = rng.binomial(1, 0.03, size=n_rows)   # ~3% escalation rate
    opt_out_flag     = rng.binomial(1, 0.08, size=n_rows)   # ~8% opt-out rate

    # Derived fields
    engagement_score = _compute_engagement_score(
        opened, clicked, response_flag, complaint_flag, opt_out_flag
    )
    sentiment_text = _sentiment_from_score(engagement_score)

    needs_intervention = _compute_needs_intervention(
        engagement_score, opened, complaint_flag, escalation_flag,
        days_since_last_contact, response_flag, opt_out_flag, cfg
    )

    # Dates: last 90 days from today
    today     = date.today()
    start_day = today - timedelta(days=90)
    sent_dates = _random_dates(start_day, today, n_rows, rng)

    # Customer IDs — unique zero-padded synthetic IDs (no duplicates)
    all_ids = rng.choice(range(1, 999999), size=n_rows, replace=False)
    customer_ids = [f"CUST_{str(i).zfill(6)}" for i in all_ids]

    df = pd.DataFrame({
        "customer_id":            customer_ids,
        "segment":                segments,
        "product_type":           product_types,
        "channel":                channels,
        "campaign_id":            campaign_ids,
        "sent_date":              sent_dates,
        "opened":                 opened.astype(int),
        "clicked":                clicked.astype(int),
        "response_flag":          response_flag.astype(int),
        "complaint_flag":         complaint_flag.astype(int),
        "escalation_flag":        escalation_flag.astype(int),
        "engagement_score":       engagement_score,
        "sentiment_text":         sentiment_text,
        "premium_bucket":         premium_bkts,
        "tenure_months":          tenure_months.astype(int),
        "days_since_last_contact": days_since_last_contact.astype(int),
        "opt_out_flag":           opt_out_flag.astype(int),
        "needs_intervention":     needs_intervention,
    })

    return df


def save_raw_csv(df: pd.DataFrame, path: str | None = None) -> str:
    """
    Save generated DataFrame as raw CSV.

    Args:
        df:   DataFrame to save.
        path: Output path. Defaults to config value.

    Returns:
        Absolute path string of saved file.
    """
    cfg = _load_config()
    path = path or cfg["data"]["raw_csv_path"]
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return str(Path(path).resolve())


# ── CLI entry point ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Generating synthetic data...")
    df = generate_synthetic_data()
    out_path = save_raw_csv(df)

    print(f"  Rows generated : {len(df):,}")
    print(f"  Intervention % : {df['needs_intervention'].mean():.1%}")
    print(f"  Saved to       : {out_path}")
    print("\nColumn dtypes:")
    print(df.dtypes)
    print("\nSample (5 rows):")
    print(df.head())
