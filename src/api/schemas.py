"""
schemas.py — Pydantic request and response models for the FastAPI endpoints.

Keeping schemas separate from main.py makes them independently testable
and easy to version. All fields include descriptions for the auto-generated
/docs page.
"""

from pydantic import BaseModel, Field
from typing import Optional, Any
from enum import Enum
from datetime import date


class DetectorType(str, Enum):
    segment_engagement_drop   = "segment_engagement_drop"
    complaint_spike           = "complaint_spike"
    campaign_underperformance = "campaign_underperformance"
    all                       = "all"


# ── Health ─────────────────────────────────────────────────────────────────────

class HealthResponse(BaseModel):
    status: str = Field(..., description="API health status")
    version: str = Field(..., description="API version string")


# ── Score Customer ─────────────────────────────────────────────────────────────

class CustomerRequest(BaseModel):
    """
    Raw customer feature payload for scoring.
    All fields match the dataset schema exactly.
    Categorical fields default to safe values when omitted.
    """
    customer_id:             str   = Field("UNKNOWN",  description="Customer ID — defaults to UNKNOWN for demo use; provide real ID in production")
    segment:                 str   = Field("Standard", description="Premium / Standard / Basic")
    product_type:            str   = Field("Auto",     description="Auto / Home / Life / Health")
    channel:                 str   = Field("Email",    description="Email / SMS / Phone / Direct Mail")
    campaign_id:             str   = Field("CAMP_001", description="Campaign identifier")
    sent_date:               date  = Field(date(2026, 1, 1), description="ISO date (YYYY-MM-DD)")
    opened:                  int   = Field(0, ge=0, le=1, description="1 if communication was opened")
    clicked:                 int   = Field(0, ge=0, le=1, description="1 if link was clicked")
    response_flag:           int   = Field(0, ge=0, le=1, description="1 if customer responded")
    complaint_flag:          int   = Field(0, ge=0, le=1, description="1 if complaint raised")
    escalation_flag:         int   = Field(0, ge=0, le=1, description="1 if escalation raised")
    engagement_score:        float = Field(0.5, ge=0.0, le=1.0, description="Composite engagement 0-1")
    sentiment_text:          str   = Field("neutral",  description="positive/neutral/negative/mixed")
    premium_bucket:          str   = Field("Mid",      description="Low / Mid / High")
    tenure_months:           int   = Field(12, ge=1, le=120, description="Months as customer")
    days_since_last_contact: int   = Field(30, ge=0, description="Days since last outreach")
    opt_out_flag:            int   = Field(0, ge=0, le=1, description="1 if customer opted out")


class ScoreResponse(BaseModel):
    customer_id:        str   = Field(..., description="Customer identifier")
    intervention_score: float = Field(..., description="Probability of needing intervention (0-1)")
    needs_intervention: int   = Field(..., description="Binary flag: 1 = intervention needed")
    risk_band:          str   = Field(..., description="Low / Medium / High")


# ── Detect Anomaly ─────────────────────────────────────────────────────────────

class AnomalyRequest(BaseModel):
    """
    Request to run anomaly detectors.
    detector_type controls which detector to run.
    If omitted, all three detectors run.
    """
    detector_type: Optional[DetectorType] = Field(
        DetectorType.all,
        description="segment_engagement_drop / complaint_spike / campaign_underperformance / all"
    )


class AnomalyItem(BaseModel):
    """A single flagged anomaly from any detector."""
    detector:  str  = Field(..., description="Which detector produced this result")
    flagged:   bool = Field(..., description="True if anomaly was detected")
    summary:   str  = Field(..., description="Human-readable summary")
    anomalies: list[dict[str, Any]] = Field(default_factory=list, description="List of flagged items with detector-specific detail")


class AnomalyResponse(BaseModel):
    results:      list[AnomalyItem] = Field(..., description="One result per detector run")
    any_flagged:  bool              = Field(..., description="True if any detector flagged an anomaly")
    total_flagged: int              = Field(..., description="Number of detectors that flagged")


# ── Generate Summary ───────────────────────────────────────────────────────────

class SummaryRequest(BaseModel):
    """
    Request to generate a next-best-action summary.
    customer is required. score and anomaly_summary are optional context.
    """
    customer:        CustomerRequest = Field(..., description="Customer feature payload")
    anomaly_summary: str             = Field("", description="Text from anomaly detector output — empty string if none")
    force_stub:      bool            = Field(False, description="Force stub fallback (no OpenAI call)")


class SummaryResponse(BaseModel):
    customer_id: str = Field(..., description="Customer identifier")
    summary:     str = Field(..., description="Next-best-action summary text")
    source:      str = Field(..., description="openai or stub")
    model:       str = Field(..., description="Model used to generate summary")
