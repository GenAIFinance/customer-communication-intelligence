"""
db.py — DuckDB connection and query utilities.

Provides a single connection helper used by all modules.
DuckDB file path is read from config or .env (DUCKDB_PATH).
"""

import os
import duckdb
import pandas as pd
import yaml
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# ── Config ────────────────────────────────────────────────────────────────────

def load_config() -> dict:
    """Load config.yaml from the project root config/ directory."""
    config_path = Path(__file__).resolve().parents[2] / "config" / "config.yaml"
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def get_db_path() -> str:
    """
    Resolve DuckDB file path.
    Priority: DUCKDB_PATH env var → config.yaml → default fallback.
    """
    env_path = os.getenv("DUCKDB_PATH")
    if env_path:
        return env_path

    try:
        config = load_config()
        return config["data"]["duckdb_path"]
    except Exception:
        return "data/processed/communications.duckdb"


# ── Connection ─────────────────────────────────────────────────────────────────

def get_connection(read_only: bool = False) -> duckdb.DuckDBPyConnection:
    """
    Return a DuckDB connection to the project database file.

    Args:
        read_only: Open in read-only mode (safe for concurrent reads).

    Returns:
        duckdb.DuckDBPyConnection
    """
    db_path = get_db_path()

    # Ensure parent directory exists
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)

    conn = duckdb.connect(database=db_path, read_only=read_only)
    return conn


# ── Query helpers ──────────────────────────────────────────────────────────────

def query_df(sql: str, params: list | None = None) -> pd.DataFrame:
    """
    Execute a SQL query and return results as a DataFrame.

    Args:
        sql:    SQL string (use ? for positional params).
        params: Optional list of positional parameter values.

    Returns:
        pd.DataFrame with query results.
    """
    conn = get_connection(read_only=True)
    try:
        if params:
            result = conn.execute(sql, params).df()
        else:
            result = conn.execute(sql).df()
        return result
    finally:
        conn.close()


def execute_statement(sql: str, params: list | None = None) -> None:
    """
    Execute a non-SELECT SQL statement (INSERT, CREATE, DROP, etc.).

    Args:
        sql:    SQL string.
        params: Optional list of positional parameter values.
    """
    conn = get_connection(read_only=False)
    try:
        if params:
            conn.execute(sql, params)
        else:
            conn.execute(sql)
        conn.commit()
    finally:
        conn.close()


def table_exists(table_name: str) -> bool:
    """
    Check whether a table exists in the DuckDB database.

    Args:
        table_name: Name of the table to check.

    Returns:
        True if table exists, False otherwise.
    """
    conn = get_connection(read_only=True)
    try:
        result = conn.execute(
            "SELECT count(*) FROM information_schema.tables WHERE table_name = ?",
            [table_name]
        ).fetchone()
        return result[0] > 0
    finally:
        conn.close()


def write_df(df: pd.DataFrame, table_name: str, if_exists: str = "replace") -> None:
    """
    Write a DataFrame to a DuckDB table.

    Args:
        df:         DataFrame to write.
        table_name: Target table name.
        if_exists:  'replace' (default) or 'append'.
    """
    conn = get_connection(read_only=False)
    try:
        if if_exists == "replace":
            conn.execute(f"DROP TABLE IF EXISTS {table_name}")
        conn.execute(
            f"CREATE TABLE IF NOT EXISTS {table_name} AS SELECT * FROM df"
        )
        if if_exists == "append":
            conn.execute(f"INSERT INTO {table_name} SELECT * FROM df")
        conn.commit()
    finally:
        conn.close()


def get_row_count(table_name: str) -> int:
    """Return the number of rows in a table."""
    result = query_df(f"SELECT COUNT(*) AS n FROM {table_name}")
    return int(result["n"].iloc[0])
