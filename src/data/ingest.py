"""
ingest.py — Load validated data into DuckDB.

Pipeline:
  1. Generate synthetic data
  2. Validate
  3. Write to DuckDB (replace existing table)
  4. Export processed CSV for Power BI / downstream use

Run directly:
    python -m src.data.ingest
"""

import pandas as pd
from pathlib import Path
import sys
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.data.generate_data import generate_synthetic_data, save_raw_csv
from src.data.validate import validate
from src.utils.db import write_df, get_row_count, get_db_path


# ── Config ─────────────────────────────────────────────────────────────────────

def _load_config() -> dict:
    config_path = Path(__file__).resolve().parents[2] / "config" / "config.yaml"
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


# ── Cleaning ───────────────────────────────────────────────────────────────────

def clean(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply lightweight cleaning steps after generation and before ingestion.

    Steps:
    - Ensure sent_date is a proper date string (DuckDB-friendly)
    - Strip whitespace from string columns
    - Enforce correct dtypes on binary columns
    - Drop any accidental duplicate customer_id rows

    Args:
        df: Raw generated DataFrame.

    Returns:
        Cleaned DataFrame.
    """
    df = df.copy()

    # Ensure sent_date is string in ISO format for DuckDB compatibility
    df["sent_date"] = pd.to_datetime(df["sent_date"]).dt.date.astype(str)

    # Strip whitespace from object columns
    str_cols = df.select_dtypes(include="str").columns
    for col in str_cols:
        df[col] = df[col].str.strip()

    # Enforce binary int columns
    binary_cols = [
        "opened", "clicked", "response_flag", "complaint_flag",
        "escalation_flag", "opt_out_flag", "needs_intervention"
    ]
    for col in binary_cols:
        if col in df.columns:
            df[col] = df[col].astype(int)

    # Drop duplicates (generate_data is non-deterministic on IDs, so rare)
    before = len(df)
    df = df.drop_duplicates(subset=["customer_id"])
    dropped = before - len(df)
    if dropped > 0:
        print(f"  Dropped {dropped} duplicate customer_id rows")

    return df


def export_processed_csv(df: pd.DataFrame, path: str | None = None) -> str:
    """
    Export the processed DataFrame as a CSV for Power BI and notebook use.

    Args:
        df:   Processed DataFrame.
        path: Output path. Defaults to config value.

    Returns:
        Absolute path string of saved file.
    """
    cfg = _load_config()
    path = path or cfg["data"]["processed_csv_path"]
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return str(Path(path).resolve())


# ── Main pipeline ──────────────────────────────────────────────────────────────

def run_ingestion(
    n_rows: int | None = None,
    seed: int | None = None,
    skip_validation: bool = False,
) -> pd.DataFrame:
    """
    Full ingestion pipeline: generate → validate → clean → write to DuckDB.

    Args:
        n_rows:           Number of synthetic rows to generate.
        seed:             Random seed.
        skip_validation:  If True, skip validation checks (not recommended).

    Returns:
        Cleaned DataFrame that was written to DuckDB.

    Raises:
        ValueError: If validation fails with errors.
    """
    cfg = _load_config()
    table_name = cfg["data"]["table_name"]

    print("── Step 1: Generate synthetic data ──────────────────────")
    df_raw = generate_synthetic_data(n_rows=n_rows, seed=seed)
    raw_path = save_raw_csv(df_raw)
    print(f"  Generated {len(df_raw):,} rows → {raw_path}")

    if not skip_validation:
        print("\n── Step 2: Validate ──────────────────────────────────────")
        report = validate(df_raw)
        print(report.summary())
        if not report.passed:
            raise ValueError(f"Validation failed:\n{report.summary()}")
        print(f"  Intervention rate : {report.stats.get('intervention_rate', 'N/A'):.1%}")

    print("\n── Step 3: Clean ─────────────────────────────────────────")
    df_clean = clean(df_raw)
    print(f"  Cleaned rows : {len(df_clean):,}")

    print("\n── Step 4: Write to DuckDB ───────────────────────────────")
    db_path = get_db_path()
    write_df(df_clean, table_name, if_exists="replace")
    row_count = get_row_count(table_name)
    print(f"  Table '{table_name}' : {row_count:,} rows in {db_path}")

    print("\n── Step 5: Export processed CSV ──────────────────────────")
    csv_path = export_processed_csv(df_clean)
    print(f"  CSV exported to : {csv_path}")

    print("\n✓ Ingestion complete.")
    return df_clean


# ── CLI entry point ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    run_ingestion()
