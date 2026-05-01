"""
prompts.py — Prompt templates for the GenAI next-best-action summariser.

Keeps all prompt logic in one place so it can be reviewed, versioned,
and tuned independently of the API call mechanics in summarizer.py.
"""

# ── System prompt ──────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are a customer communications analyst at a financial services company.
Your job is to produce concise, actionable next-best-action summaries for customer cases.

Rules:
- Maximum 3 sentences
- Use plain business language — no jargon
- Always include one specific recommended action
- Never invent data not provided in the context
- Never mention that you are an AI"""


# ── User prompt template ───────────────────────────────────────────────────────

def build_user_prompt(context: dict) -> str:
    """
    Build the user-facing prompt from a structured context dict.

    Expected context keys:
        customer_id         (str)
        segment             (str)
        channel             (str)
        engagement_score    (float)
        complaint_flag      (int)
        escalation_flag     (int)
        days_since_contact  (int)
        intervention_score  (float)
        risk_band           (str)
        anomaly_summary     (str)  — output of anomaly_summary_text()
        sentiment           (str)

    Args:
        context: Dict of customer and anomaly data.

    Returns:
        Formatted prompt string ready to send to the model.
    """
    return f"""Customer case summary request:

Customer ID     : {context.get('customer_id', 'Unknown')}
Segment         : {context.get('segment', 'Unknown')}
Channel         : {context.get('channel', 'Unknown')}
Sentiment       : {context.get('sentiment', 'Unknown')}
Engagement score: {context.get('engagement_score', 0.0):.2f} / 1.00
Complaint flag  : {'Yes' if context.get('complaint_flag') else 'No'}
Escalation flag : {'Yes' if context.get('escalation_flag') else 'No'}
Days since last contact: {context.get('days_since_contact', 0)}
Model intervention score: {context.get('intervention_score', 0.0):.2f} (risk: {context.get('risk_band', 'Unknown')})

Anomaly context:
{context.get('anomaly_summary', 'No anomaly data available.')}

Write a 2-3 sentence next-best-action summary for this customer case."""


# ── Stub template ──────────────────────────────────────────────────────────────

def build_stub_summary(context: dict) -> str:
    """
    Build a template-filled summary without calling any external API.

    Used when OPENAI_API_KEY is not set or as a fallback in tests/demos.
    Produces a realistic-looking summary using real data values so the
    dashboard still demonstrates the feature credibly.

    Args:
        context: Same dict as build_user_prompt().

    Returns:
        Formatted summary string.
    """
    risk_band          = context.get("risk_band", "Medium")
    segment            = context.get("segment", "Standard")
    channel            = context.get("channel", "Email")
    engagement_score   = context.get("engagement_score", 0.0)
    days_since_contact = context.get("days_since_contact", 0)
    complaint_flag     = context.get("complaint_flag", 0)
    escalation_flag    = context.get("escalation_flag", 0)
    intervention_score = context.get("intervention_score", 0.0)
    sentiment          = context.get("sentiment", "neutral")

    # Build a contextual opening based on risk level
    if risk_band == "High" or escalation_flag:
        opening = (
            f"This {segment} segment customer is showing elevated intervention risk "
            f"(score: {intervention_score:.2f}) with {sentiment} sentiment and "
            f"{days_since_contact} days since last contact."
        )
    elif complaint_flag:
        opening = (
            f"A complaint has been raised by this {segment} customer, "
            f"with engagement score at {engagement_score:.2f} and "
            f"risk band flagged as {risk_band}."
        )
    else:
        opening = (
            f"Engagement for this {segment} customer has declined "
            f"to {engagement_score:.2f} over the past period, "
            f"with {days_since_contact} days since last meaningful contact."
        )

    # Build channel-specific recommendation
    if channel == "Email":
        recommendation = (
            f"Recommend reducing {channel} frequency and switching to a "
            f"personalised SMS or phone outreach within the next 7 days."
        )
    elif channel == "SMS":
        recommendation = (
            f"Recommend escalating from {channel} to a direct phone call "
            f"to re-establish engagement and address any outstanding concerns."
        )
    else:
        recommendation = (
            f"Recommend a direct follow-up via {channel} with a personalised "
            f"retention offer tailored to the {segment} segment."
        )

    # Add anomaly context if present
    anomaly_summary = context.get("anomaly_summary", "")
    if "FLAGGED" in anomaly_summary:
        closing = (
            "Broader segment-level anomalies have been detected — "
            "review campaign messaging for this cohort before next outreach."
        )
    else:
        closing = (
            "Monitor engagement over the next 14 days and escalate "
            "if intervention score remains above 0.45."
        )

    return f"{opening} {recommendation} {closing}"
