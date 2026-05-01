"""
run_pipeline.py — Single entry point to run the full CCI pipeline.

Steps:
  1. Generate synthetic data + ingest to DuckDB
  2. Train the intervention model
  3. Run anomaly detection
  4. Generate a sample AI summary
  5. Print final status

Run with:
    python run_pipeline.py

Optional flags:
    python run_pipeline.py --skip-genai    # skip AI summary step
    python run_pipeline.py --rows 1000     # generate fewer rows (faster)
"""

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))


def banner(title: str) -> None:
    width = 56
    print(f"\n{'─' * width}")
    print(f"  {title}")
    print(f"{'─' * width}")


def run_pipeline(n_rows: int = 5000, skip_genai: bool = False) -> None:
    start_total = time.time()

    # ── Step 1: Data ingestion ──────────────────────────────────────────────
    banner("Step 1 — Data Ingestion")
    t = time.time()
    from src.data.ingest import run_ingestion
    df = run_ingestion(n_rows=n_rows)
    print(f"\n  ✓ Done in {time.time() - t:.1f}s")

    # ── Step 2: Model training ──────────────────────────────────────────────
    banner("Step 2 — Model Training")
    t = time.time()
    from src.modeling.train_model import train
    model, metrics, feature_names = train()
    print(f"\n  ✓ Done in {time.time() - t:.1f}s")
    print(f"  ROC-AUC : {metrics['roc_auc']}")
    print(f"  F1      : {metrics['f1']}")

    # ── Step 3: Anomaly detection ───────────────────────────────────────────
    banner("Step 3 — Anomaly Detection")
    t = time.time()
    from src.modeling.anomaly import run_all_detectors, anomaly_summary_text
    from src.utils.db import query_df
    df_full = query_df("SELECT * FROM customer_communications")
    anomaly_results = run_all_detectors(df_full)
    anomaly_text    = anomaly_summary_text(anomaly_results)
    flagged = sum(1 for r in anomaly_results.values() if r.flagged)
    print(anomaly_text)
    print(f"\n  ✓ Done in {time.time() - t:.1f}s — {flagged}/3 detectors flagged")

    # ── Step 4: Sample AI summary ───────────────────────────────────────────
    if not skip_genai:
        banner("Step 4 — Sample AI Summary")
        t = time.time()
        from src.modeling.score import score_customer
        from src.genai.summarizer import generate_summary

        # Pick a high-risk case
        sample = df_full[df_full["needs_intervention"] == 1].iloc[0].to_dict()
        score  = score_customer(sample)
        result = generate_summary(
            customer=sample,
            score_result=score,
            anomaly_summary=anomaly_text,
        )
        print(f"\n  Customer  : {result.customer_id}")
        print(f"  Source    : {result.source}")
        print(f"  Risk band : {score.risk_band}")
        print(f"\n  {result.summary}")
        print(f"\n  ✓ Done in {time.time() - t:.1f}s")
    else:
        banner("Step 4 — AI Summary (skipped)")
        print("  Use --skip-genai=False to enable")

    # ── Summary ─────────────────────────────────────────────────────────────
    total_time = time.time() - start_total
    banner("Pipeline Complete")
    print(f"  Total time       : {total_time:.1f}s")
    print(f"  Rows in database : {len(df_full):,}")
    print(f"  Model AUC        : {metrics['roc_auc']}")
    print(f"  Anomalies found  : {flagged}/3 detectors")
    print(f"\n  Next steps:")
    print(f"    API    : uvicorn src.api.main:app --reload")
    print(f"    Dashboard: python -m streamlit run app/streamlit_app.py")
    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run the full Customer Communication Intelligence pipeline."
    )
    parser.add_argument(
        "--rows", type=int, default=5000,
        help="Number of synthetic rows to generate (default: 5000)"
    )
    parser.add_argument(
        "--skip-genai", action="store_true",
        help="Skip the AI summary step (faster, no OpenAI key needed)"
    )
    args = parser.parse_args()
    run_pipeline(n_rows=args.rows, skip_genai=args.skip_genai)
