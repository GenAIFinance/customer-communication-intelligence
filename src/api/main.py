"""
main.py — FastAPI application for the Customer Communication Intelligence Platform.

Endpoints:
    GET  /health           — liveness check
    GET  /ready            — readiness check (model + DB loaded)
    POST /score-customer   — score a single customer case
    POST /detect-anomaly   — run anomaly detectors on live DB data
    POST /generate-summary — generate a next-best-action summary

Run with:
    uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload

Codex fixes applied:
  - sys.path.insert removed; run via uvicorn from project root
  - Structured logging replaces print statements
  - Specific exception types mapped to correct HTTP status codes
  - Model state stored in app.state (not global mutables)
  - DetectorType enum enforces valid detector names (invalid input -> 422)
  - Lifespan catches broad Exception not just FileNotFoundError
  - /ready endpoint reflects true readiness (model + DB)
"""

import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse

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

# ── Logging ────────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger("cci.api")


# ── Lifespan ───────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Load model and feature names once at startup into app.state.
    Catches all load errors — use /ready to check readiness.
    """
    app.state.model = None
    app.state.feature_names = None
    app.state.model_error = None

    try:
        from src.modeling.score import load_model
        from src.modeling.train_model import load_feature_names
        app.state.model = load_model()
        app.state.feature_names = load_feature_names()
        logger.info("Model loaded — %d features", len(app.state.feature_names))
    except Exception as exc:
        app.state.model_error = str(exc)
        logger.error("Model failed to load: %s", exc)

    yield
    logger.info("API shutting down.")


# ── App ────────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Customer Communication Intelligence API",
    description="Anomaly detection, intervention scoring, and AI-assisted summaries.",
    version="1.0.0",
    lifespan=lifespan,
)


def _require_model(request: Request):
    """Return (model, feature_names) or raise 503."""
    if request.app.state.model is None:
        detail = request.app.state.model_error or "Model not loaded."
        raise HTTPException(status_code=503, detail=f"Model unavailable: {detail}")
    return request.app.state.model, request.app.state.feature_names


# ── GET /health ────────────────────────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse, tags=["System"])
def health():
    """Liveness check — returns 200 if the process is running."""
    return HealthResponse(status="ok", version="1.0.0")


# ── GET /ready ─────────────────────────────────────────────────────────────────

@app.get("/ready", tags=["System"])
def ready(request: Request):
    """
    Readiness check — returns 200 only when model and DB are available.
    Use this to gate traffic in orchestrated deployments.
    """
    model_ok = request.app.state.model is not None
    try:
        from src.utils.db import query_df
        query_df("SELECT 1")
        db_ok = True
    except Exception:
        db_ok = False

    is_ready = model_ok and db_ok
    return JSONResponse(
        status_code=200 if is_ready else 503,
        content={"ready": is_ready, "model_ok": model_ok, "db_ok": db_ok},
    )


# ── POST /score-customer ───────────────────────────────────────────────────────

@app.post("/score-customer", response_model=ScoreResponse, tags=["Scoring"])
def score_customer(request: Request, payload: CustomerRequest):
    """
    Score a single customer case for intervention risk.
    Partial payloads accepted — missing categoricals use safe defaults.
    """
    model, feature_names = _require_model(request)
    customer_dict = payload.model_dump()
    # Convert date object to ISO string — downstream modules expect str
    if hasattr(customer_dict.get("sent_date"), "isoformat"):
        customer_dict["sent_date"] = customer_dict["sent_date"].isoformat()

    try:
        from src.modeling.score import score_customer as _score
        result = _score(customer_dict, model=model, expected_columns=feature_names)
    except ValueError as exc:
        logger.warning("Validation error scoring %s: %s", payload.customer_id, exc)
        raise HTTPException(status_code=422, detail=str(exc))
    except Exception as exc:
        logger.error("Scoring error for %s: %s", payload.customer_id, exc, exc_info=True)
        raise HTTPException(status_code=500, detail="Internal scoring error.")

    logger.info("Scored %s — %.3f (%s)", result.customer_id,
                result.intervention_score, result.risk_band)
    return ScoreResponse(
        customer_id=result.customer_id,
        intervention_score=result.intervention_score,
        needs_intervention=result.needs_intervention,
        risk_band=result.risk_band,
    )


# ── POST /detect-anomaly ───────────────────────────────────────────────────────

@app.post("/detect-anomaly", response_model=AnomalyResponse, tags=["Anomaly"])
def detect_anomaly(request: Request, payload: AnomalyRequest):
    """
    Run anomaly detectors on current database data.
    detector_type is validated by DetectorType enum — invalid values return 422.
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
        detector_type = (payload.detector_type or "all").value

        if detector_type == "segment_engagement_drop":
            raw = {"segment_engagement_drop": detect_segment_engagement_drop(df)}
        elif detector_type == "complaint_spike":
            raw = {"complaint_spike": detect_complaint_spike(df)}
        elif detector_type == "campaign_underperformance":
            raw = {"campaign_underperformance": detect_campaign_underperformance(df)}
        else:
            raw = run_all_detectors(df)

    except Exception as exc:
        logger.error("Anomaly detection error: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail="Anomaly detection failed.")

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

    logger.info("Anomaly check — %d/%d flagged", total_flagged, len(results))
    return AnomalyResponse(
        results=results,
        any_flagged=any_flagged,
        total_flagged=total_flagged,
    )


# ── POST /generate-summary ─────────────────────────────────────────────────────

@app.post("/generate-summary", response_model=SummaryResponse, tags=["GenAI"])
def generate_summary(request: Request, payload: SummaryRequest):
    """
    Generate a next-best-action summary.
    Uses OpenAI when key is set; falls back to stub otherwise.
    Set force_stub=true for offline/demo use.
    """
    customer_dict = payload.customer.model_dump()
    # Convert date object to ISO string for downstream compatibility
    if hasattr(customer_dict.get("sent_date"), "isoformat"):
        customer_dict["sent_date"] = customer_dict["sent_date"].isoformat()

    try:
        from src.genai.summarizer import generate_summary as _summarize
        from src.modeling.score import score_customer as _score

        score_result = None
        if request.app.state.model is not None:
            try:
                score_result = _score(
                    customer_dict,
                    model=request.app.state.model,
                    expected_columns=request.app.state.feature_names,
                )
            except Exception as exc:
                logger.warning("Scoring failed during summary: %s", exc)

        result = _summarize(
            customer=customer_dict,
            score_result=score_result,
            anomaly_summary=payload.anomaly_summary or "",
            force_stub=payload.force_stub,
        )

    except RuntimeError as exc:
        logger.warning("Summary config error: %s", exc)
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        logger.error("Summary error: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail="Summary generation failed.")

    logger.info("Summary for %s — source=%s", result.customer_id, result.source)
    return SummaryResponse(
        customer_id=result.customer_id,
        summary=result.summary,
        source=result.source,
        model=result.model,
    )
