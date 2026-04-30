"""
summarizer.py — Generate next-best-action summaries via OpenAI or stub fallback.

Flow:
  1. Build context dict from customer + score + anomaly data
  2. If OPENAI_API_KEY is set → call OpenAI API
  3. If key is missing or call fails → fall back to template stub
  4. Return structured SummaryResult

Run directly:
    python -m src.genai.summarizer
"""

import os
import yaml
from pathlib import Path
from dataclasses import dataclass
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from dotenv import load_dotenv
load_dotenv()

from src.genai.prompts import build_user_prompt, build_stub_summary, SYSTEM_PROMPT


# ── Config ─────────────────────────────────────────────────────────────────────

def _load_config() -> dict:
    config_path = Path(__file__).resolve().parents[2] / "config" / "config.yaml"
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


# ── Result dataclass ───────────────────────────────────────────────────────────

@dataclass
class SummaryResult:
    """
    Structured output from the summariser.

    Attributes:
        summary:    The next-best-action summary text.
        source:     'openai' | 'stub' — which path produced the summary.
        model:      Model name used (or 'stub').
        customer_id: ID of the customer case summarised.
    """
    summary: str
    source: str        # 'openai' or 'stub'
    model: str
    customer_id: str


# ── Context builder ────────────────────────────────────────────────────────────

def build_context(
    customer: dict,
    score_result=None,
    anomaly_summary: str = "",
) -> dict:
    """
    Assemble the context dict that prompts.py uses to build the prompt.

    Args:
        customer:       Raw customer field dict.
        score_result:   ScoreResult from score.py (optional).
        anomaly_summary: Text output of anomaly_summary_text() (optional).

    Returns:
        Context dict with all fields prompts.py expects.
    """
    intervention_score = 0.0
    risk_band = "Unknown"

    if score_result is not None:
        intervention_score = getattr(score_result, "intervention_score", 0.0)
        risk_band          = getattr(score_result, "risk_band", "Unknown")

    return {
        "customer_id":        customer.get("customer_id", "Unknown"),
        "segment":            customer.get("segment", "Unknown"),
        "channel":            customer.get("channel", "Unknown"),
        "sentiment":          customer.get("sentiment_text", "neutral"),
        "engagement_score":   float(customer.get("engagement_score", 0.0)),
        "complaint_flag":     int(customer.get("complaint_flag", 0)),
        "escalation_flag":    int(customer.get("escalation_flag", 0)),
        "days_since_contact": int(customer.get("days_since_last_contact", 0)),
        "intervention_score": float(intervention_score),
        "risk_band":          risk_band,
        "anomaly_summary":    anomaly_summary,
    }


# ── OpenAI call ────────────────────────────────────────────────────────────────

def _call_openai(context: dict, cfg: dict) -> str:
    """
    Call the OpenAI chat completions API.

    Args:
        context: Context dict for prompt building.
        cfg:     Config dict with genai settings.

    Returns:
        Summary string from the model.

    Raises:
        Exception: Any API error — caller handles fallback.
    """
    from openai import OpenAI

    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    response = client.chat.completions.create(
        model=os.getenv("OPENAI_MODEL", cfg["genai"]["model"]),
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": build_user_prompt(context)},
        ],
        max_tokens=cfg["genai"]["max_tokens"],
        temperature=cfg["genai"]["temperature"],
    )

    return response.choices[0].message.content.strip()


# ── Main summariser ────────────────────────────────────────────────────────────

def generate_summary(
    customer: dict,
    score_result=None,
    anomaly_summary: str = "",
    force_stub: bool = False,
) -> SummaryResult:
    """
    Generate a next-best-action summary for a customer case.

    Tries OpenAI first if API key is present; falls back to the
    template stub automatically on missing key or any API error.

    Args:
        customer:        Raw customer field dict.
        score_result:    ScoreResult from score.py (optional).
        anomaly_summary: Text from anomaly_summary_text() (optional).
        force_stub:      If True, skip OpenAI and use stub directly.
                         Useful for testing and offline demos.

    Returns:
        SummaryResult with summary text and source indicator.
    """
    cfg = _load_config()
    context = build_context(customer, score_result, anomaly_summary)
    customer_id = context["customer_id"]

    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    # Only fall back to stub when:
    #   1. force_stub=True (explicit override), OR
    #   2. API key is missing AND config says to use stub as fallback
    # When a key IS present, always attempt OpenAI regardless of config default.
    no_key = not api_key
    use_stub = force_stub or (no_key and cfg["genai"].get("use_stub_if_no_key", True))

    if not use_stub:
        try:
            summary = _call_openai(context, cfg)
            return SummaryResult(
                summary=summary,
                source="openai",
                model=os.getenv("OPENAI_MODEL", cfg["genai"]["model"]),
                customer_id=customer_id,
            )
        except Exception as e:
            # Log and fall through to stub
            print(f"  [summarizer] OpenAI call failed ({type(e).__name__}: {e}). Using stub.")

    # Stub fallback — always works, no external dependency
    summary = build_stub_summary(context)
    return SummaryResult(
        summary=summary,
        source="stub",
        model="stub",
        customer_id=customer_id,
    )


# ── CLI entry point ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    from src.utils.db import query_df
    from src.modeling.score import score_customer
    from src.modeling.anomaly import run_all_detectors, anomaly_summary_text

    print("Loading sample data...")
    df = query_df("SELECT * FROM customer_communications LIMIT 50")

    print("Running anomaly detectors...")
    anomaly_results = run_all_detectors(df)
    anomaly_text    = anomaly_summary_text(anomaly_results)

    # Pick a high-risk customer to summarise
    sample = df[df["needs_intervention"] == 1].iloc[0].to_dict()

    print("Scoring customer...")
    score = score_customer(sample)

    print("Generating summary...")
    result = generate_summary(sample, score_result=score, anomaly_summary=anomaly_text)

    print(f"\n── Next-Best-Action Summary ─────────────────────────")
    print(f"  Customer  : {result.customer_id}")
    print(f"  Source    : {result.source}")
    print(f"  Model     : {result.model}")
    print(f"\n  {result.summary}")
    print()

    # Also show a low-risk customer for contrast
    safe = df[df["needs_intervention"] == 0].iloc[0].to_dict()
    safe_score  = score_customer(safe)
    safe_result = generate_summary(safe, score_result=safe_score, anomaly_summary=anomaly_text)

    print(f"── Low-risk customer for contrast ───────────────────")
    print(f"  Customer  : {safe_result.customer_id}")
    print(f"  Risk band : {safe_score.risk_band}")
    print(f"\n  {safe_result.summary}")
