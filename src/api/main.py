"""
main.py — FastAPI application for the Customer Communication Intelligence Platform.

Endpoints:
    GET  /health           — liveness check
    POST /score-customer   — score a single customer case
    POST /detect-anomaly   — run anomaly detectors on live DB data
    POST /generate-summary — generate a next-best-action summary

Run with:
    uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload
"""

from fastapi import FastAPI, HTTPException
from contextlib import asynccontextmanager
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.api.schemas import (
    HealthResponse,
    CustomerRequest,
    ScoreResponse,
    AnomalyRequest,
    AnomalyResponse,
    AnomalyItem,
    SummaryRequest,
    SummaryResponse,
)


# ── App state — loaded once at startup ────────────────────────────────────────

_model = None
_feature_names = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model and feature names once at startup — avoids per-request disk reads."""
    global _model, _feature_names
    try:
        from src.modeling.train_model import load_feature_names
        from src.modeling.score import load_model
        _model = load_model()
        _feature_names = load_feature_names()
        print(f"  Model loaded: {len(_feature_names)} features")
    except FileNotFoundError:
        print("  Warning: model not found — run train_model.py first. Score endpoint will error.")
    yield
    # Cleanup on shutdown (nothing needed for this MVP)


# ── App init ───────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Customer Communication Intelligence API",
    description="Anomaly detection, intervention scoring, and AI-assisted summaries.",
    version="1.0.0",
    lifespan=lifespan,
)


# ── GET /health ────────────────────────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse, tags=["System"])
def health():
    """Liveness check — returns 200 if the API is running."""
    return HealthResponse(status="ok", version="1.0.0")


# ── POST /score-customer ───────────────────────────────────────────────────────

@app.post("/score-customer", response_model=ScoreResponse, tags=["Scoring"])
def score_customer(request: CustomerRequest):
    """
    Score a single customer case for intervention risk.

    Returns an intervention probability score (0-1), a binary flag,
    and a risk band (Low / Medium / High).

    Accepts partial payloads — missing categorical fields default to
    safe values defined in build_features.py.
    """
    if _model is None or _feature_names is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Run 'python -m src.modeling.train_model' first."
        )

    try:
        from src.modeling.score import score_customer as _score
        customer_dict = request.model_dump()
        result = _score(
            customer_dict,
            model=_model,
            expected_columns=_feature_names,
        )
        return ScoreResponse(
            customer_id=result.customer_id,
            intervention_score=result.intervention_score,
            needs_intervention=result.needs_intervention,
            risk_band=result.risk_band,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ── POST /detect-anomaly ───────────────────────────────────────────────────────

@app.post("/detect-anomaly", response_model=AnomalyResponse, tags=["Anomaly"])
def detect_anomaly(request: AnomalyRequest):
    """
    Run anomaly detectors on the current database data.

    Set detector_type to run a specific detector:
      - segment_engagement_drop
      - complaint_spike
      - campaign_underperformance
      - all (default)

    Returns flagged anomalies and a combined summary.
    """
    try:
        from src.modeling.anomaly import (
            run_all_detectors,
            detect_segment_engagement_drop,
            detect_complaint_spike,
            detect_campaign_underperformance,
        )
        from src.utils.db import query_df

        df = query_df("SELECT * FROM customer_communications")
        detector_type = (request.detector_type or "all").lower()

        # Run selected detector(s)
        if detector_type == "segment_engagement_drop":
            raw = {"segment_engagement_drop": detect_segment_engagement_drop(df)}
        elif detector_type == "complaint_spike":
            raw = {"complaint_spike": detect_complaint_spike(df)}
        elif detector_type == "campaign_underperformance":
            raw = {"campaign_underperformance": detect_campaign_underperformance(df)}
        else:
            raw = run_all_detectors(df)

        results = [
            AnomalyItem(
                detector=name,
                flagged=result.flagged,
                summary=result.summary,
                anomalies=result.anomalies,
            )
            for name, result in raw.items()
        ]

        any_flagged   = any(r.flagged for r in results)
        total_flagged = sum(1 for r in results if r.flagged)

        return AnomalyResponse(
            results=results,
            any_flagged=any_flagged,
            total_flagged=total_flagged,
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ── POST /generate-summary ─────────────────────────────────────────────────────

@app.post("/generate-summary", response_model=SummaryResponse, tags=["GenAI"])
def generate_summary(request: SummaryRequest):
    """
    Generate a next-best-action summary for a customer case.

    Uses OpenAI if OPENAI_API_KEY is set, otherwise falls back
    to a template-based stub summary. Set force_stub=true to
    always use the stub (useful for testing and offline demos).
    """
    try:
        from src.genai.summarizer import generate_summary as _summarize
        from src.modeling.score import score_customer as _score

        customer_dict = request.customer.model_dump()

        # Score the customer to provide risk context to the summariser
        if _model is not None and _feature_names is not None:
            score_result = _score(
                customer_dict,
                model=_model,
                expected_columns=_feature_names,
            )
        else:
            score_result = None

        result = _summarize(
            customer=customer_dict,
            score_result=score_result,
            anomaly_summary=request.anomaly_summary or "",
            force_stub=request.force_stub,
        )

        return SummaryResponse(
            customer_id=result.customer_id,
            summary=result.summary,
            source=result.source,
            model=result.model,
        )

    except RuntimeError as e:
        # Config error (e.g. no API key and stub disabled) — bad request
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
