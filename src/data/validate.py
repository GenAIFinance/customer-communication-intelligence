"""
validate.py — Data validation for the customer communications dataset.

Checks schema, types, value ranges, and business rules.
Returns a structured validation report rather than raising exceptions,
so callers can decide how to handle failures.

Run directly:
    python -m src.data.validate
"""

import pandas as pd
import numpy as np
from dataclasses import dataclass, field
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

# ── Schema definition ──────────────────────────────────────────────────────────

REQUIRED_COLUMNS = [
    "customer_id", "segment", "product_type", "channel", "campaign_id",
    "sent_date", "opened", "clicked", "response_flag", "complaint_flag",
    "escalation_flag", "engagement_score", "sentiment_text", "premium_bucket",
    "tenure_months", "days_since_last_contact", "opt_out_flag",
    "needs_intervention",
]

BINARY_COLUMNS = [
    "opened", "clicked", "response_flag", "complaint_flag",
    "escalation_flag", "opt_out_flag", "needs_intervention",
]

ALLOWED_VALUES = {
    "segment":         {"Premium", "Standard", "Basic"},
    "product_type":    {"Auto", "Home", "Life", "Health"},
    "channel":         {"Email", "SMS", "Phone", "Direct Mail"},
    "premium_bucket":  {"Low", "Mid", "High"},
    "sentiment_text":  {"positive", "neutral", "negative", "mixed"},
}


# ── Report dataclass ───────────────────────────────────────────────────────────

@dataclass
class ValidationReport:
    """Structured output from the validation run."""
    passed: bool = True
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    stats: dict = field(default_factory=dict)

    def add_error(self, msg: str) -> None:
        self.errors.append(msg)
        self.passed = False

    def add_warning(self, msg: str) -> None:
        self.warnings.append(msg)

    def summary(self) -> str:
        status = "PASSED" if self.passed else "FAILED"
        lines = [
            f"Validation {status}",
            f"  Errors   : {len(self.errors)}",
            f"  Warnings : {len(self.warnings)}",
        ]
        if self.errors:
            lines.append("  Error details:")
            for e in self.errors:
                lines.append(f"    ✗ {e}")
        if self.warnings:
            lines.append("  Warning details:")
            for w in self.warnings:
                lines.append(f"    ⚠ {w}")
        return "\n".join(lines)


# ── Validation checks ──────────────────────────────────────────────────────────

def check_required_columns(df: pd.DataFrame, report: ValidationReport) -> None:
    """Fail if any required column is missing."""
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        report.add_error(f"Missing required columns: {missing}")


def check_no_nulls(df: pd.DataFrame, report: ValidationReport) -> None:
    """Warn on any null values; error on nulls in critical columns."""
    critical = ["customer_id", "segment", "engagement_score", "needs_intervention"]
    for col in critical:
        if col not in df.columns:
            continue
        null_count = df[col].isna().sum()
        if null_count > 0:
            report.add_error(f"Nulls in critical column '{col}': {null_count} rows")

    non_critical = [c for c in REQUIRED_COLUMNS if c not in critical and c in df.columns]
    for col in non_critical:
        null_count = df[col].isna().sum()
        if null_count > 0:
            report.add_warning(f"Nulls in '{col}': {null_count} rows")


def check_binary_columns(df: pd.DataFrame, report: ValidationReport) -> None:
    """Binary columns must only contain 0 or 1."""
    for col in BINARY_COLUMNS:
        if col not in df.columns:
            continue
        invalid = df[~df[col].isin([0, 1])][col]
        if len(invalid) > 0:
            report.add_error(f"Column '{col}' contains non-binary values: {invalid.unique().tolist()}")


def check_allowed_values(df: pd.DataFrame, report: ValidationReport) -> None:
    """Categorical columns must only contain expected values."""
    for col, allowed in ALLOWED_VALUES.items():
        if col not in df.columns:
            continue
        actual = set(df[col].dropna().unique())
        unexpected = actual - allowed
        if unexpected:
            report.add_error(f"Column '{col}' has unexpected values: {unexpected}")


def check_engagement_score_range(df: pd.DataFrame, report: ValidationReport) -> None:
    """Engagement score must be in [0.0, 1.0]."""
    if "engagement_score" not in df.columns:
        return
    out_of_range = df[(df["engagement_score"] < 0) | (df["engagement_score"] > 1)]
    if len(out_of_range) > 0:
        report.add_error(
            f"engagement_score out of [0,1] range: {len(out_of_range)} rows"
        )


def check_numeric_ranges(df: pd.DataFrame, report: ValidationReport) -> None:
    """Numeric fields must be within reasonable bounds."""
    checks = {
        "tenure_months":          (1, 120),
        "days_since_last_contact": (0, 365),
    }
    for col, (lo, hi) in checks.items():
        if col not in df.columns:
            continue
        bad = df[(df[col] < lo) | (df[col] > hi)]
        if len(bad) > 0:
            report.add_warning(
                f"Column '{col}' has {len(bad)} values outside [{lo}, {hi}]"
            )


def check_row_count(df: pd.DataFrame, report: ValidationReport, min_rows: int = 100) -> None:
    """Warn if the dataset seems suspiciously small."""
    if len(df) < min_rows:
        report.add_warning(f"Dataset has only {len(df)} rows (expected >= {min_rows})")


def check_intervention_rate(df: pd.DataFrame, report: ValidationReport) -> None:
    """
    Warn if intervention rate is implausibly extreme.
    Expect roughly 10–60% intervention rate for a useful model.
    """
    if "needs_intervention" not in df.columns:
        return
    rate = df["needs_intervention"].mean()
    report.stats["intervention_rate"] = round(rate, 4)
    if rate < 0.05:
        report.add_warning(f"Intervention rate very low: {rate:.1%} — model may struggle")
    elif rate > 0.80:
        report.add_warning(f"Intervention rate very high: {rate:.1%} — check target definition")


def compute_stats(df: pd.DataFrame, report: ValidationReport) -> None:
    """Add summary statistics to the report."""
    report.stats.update({
        "row_count":       len(df),
        "column_count":    len(df.columns),
        "null_total":      int(df.isna().sum().sum()),
        "engagement_mean": round(df["engagement_score"].mean(), 4) if "engagement_score" in df.columns else None,
        "complaint_rate":  round(df["complaint_flag"].mean(), 4)   if "complaint_flag"   in df.columns else None,
    })


# ── Public API ─────────────────────────────────────────────────────────────────

def validate(df: pd.DataFrame) -> ValidationReport:
    """
    Run all validation checks on a DataFrame.

    Args:
        df: DataFrame to validate (should match schema from generate_data.py).

    Returns:
        ValidationReport with .passed, .errors, .warnings, .stats.
    """
    report = ValidationReport()

    check_required_columns(df, report)

    # Only run further checks if columns are present
    if not report.errors:
        check_no_nulls(df, report)
        check_binary_columns(df, report)
        check_allowed_values(df, report)
        check_engagement_score_range(df, report)
        check_numeric_ranges(df, report)
        check_row_count(df, report)
        check_intervention_rate(df, report)
        compute_stats(df, report)

    return report


# ── CLI entry point ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    from src.data.generate_data import generate_synthetic_data

    print("Generating data for validation test...")
    df = generate_synthetic_data()

    print("Running validation...")
    report = validate(df)

    print(report.summary())
    print("\nStats:")
    for k, v in report.stats.items():
        print(f"  {k}: {v}")
