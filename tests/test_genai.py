"""
test_genai.py — Tests for the GenAI summariser (prompts + summarizer).

All tests use force_stub=True so they run without an OpenAI API key.

Run with:
    python -m pytest tests/test_genai.py -v
"""

import pytest
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.genai.prompts import build_user_prompt, build_stub_summary, SYSTEM_PROMPT
from src.genai.summarizer import generate_summary, build_context, SummaryResult


# ── Fixtures ───────────────────────────────────────────────────────────────────

@pytest.fixture
def high_risk_customer():
    return {
        "customer_id":             "CUST_HIGH_001",
        "segment":                 "Basic",
        "channel":                 "Email",
        "sentiment_text":          "negative",
        "engagement_score":        0.08,
        "complaint_flag":          1,
        "escalation_flag":         1,
        "days_since_last_contact": 90,
    }


@pytest.fixture
def low_risk_customer():
    return {
        "customer_id":             "CUST_LOW_001",
        "segment":                 "Premium",
        "channel":                 "Phone",
        "sentiment_text":          "positive",
        "engagement_score":        0.92,
        "complaint_flag":          0,
        "escalation_flag":         0,
        "days_since_last_contact": 3,
    }


@pytest.fixture
def mock_score_result():
    from src.modeling.score import ScoreResult
    return ScoreResult(
        customer_id="CUST_HIGH_001",
        intervention_score=0.87,
        needs_intervention=1,
        risk_band="High",
    )


# ── Prompt tests ───────────────────────────────────────────────────────────────

class TestPrompts:

    def test_system_prompt_is_string(self):
        assert isinstance(SYSTEM_PROMPT, str)
        assert len(SYSTEM_PROMPT) > 50

    def test_build_user_prompt_contains_customer_id(self, high_risk_customer):
        ctx = build_context(high_risk_customer)
        prompt = build_user_prompt(ctx)
        assert "CUST_HIGH_001" in prompt

    def test_build_user_prompt_contains_segment(self, high_risk_customer):
        ctx = build_context(high_risk_customer)
        prompt = build_user_prompt(ctx)
        assert "Basic" in prompt

    def test_build_user_prompt_contains_engagement(self, high_risk_customer):
        ctx = build_context(high_risk_customer)
        prompt = build_user_prompt(ctx)
        assert "0.08" in prompt

    def test_build_user_prompt_shows_complaint_yes(self, high_risk_customer):
        ctx = build_context(high_risk_customer)
        prompt = build_user_prompt(ctx)
        assert "Yes" in prompt

    def test_build_user_prompt_shows_complaint_no(self, low_risk_customer):
        ctx = build_context(low_risk_customer)
        prompt = build_user_prompt(ctx)
        assert "No" in prompt

    def test_build_user_prompt_includes_anomaly_context(self, high_risk_customer):
        ctx = build_context(high_risk_customer, anomaly_summary="⚠ FLAGGED complaint_spike")
        prompt = build_user_prompt(ctx)
        assert "FLAGGED" in prompt

    def test_build_user_prompt_is_string(self, high_risk_customer):
        ctx = build_context(high_risk_customer)
        assert isinstance(build_user_prompt(ctx), str)


# ── Stub summary tests ─────────────────────────────────────────────────────────

class TestStubSummary:

    def test_returns_non_empty_string(self, high_risk_customer):
        ctx = build_context(high_risk_customer)
        summary = build_stub_summary(ctx)
        assert isinstance(summary, str)
        assert len(summary) > 50

    def test_contains_segment(self, high_risk_customer):
        ctx = build_context(high_risk_customer)
        summary = build_stub_summary(ctx)
        assert "Basic" in summary

    def test_high_risk_mentions_risk(self, high_risk_customer, mock_score_result):
        ctx = build_context(high_risk_customer, score_result=mock_score_result)
        summary = build_stub_summary(ctx)
        assert any(word in summary.lower() for word in ["risk", "escalat", "intervention"])

    def test_complaint_mentioned_when_flagged(self, high_risk_customer):
        ctx = build_context(high_risk_customer)
        ctx["complaint_flag"] = 1
        summary = build_stub_summary(ctx)
        assert "complaint" in summary.lower() or "risk" in summary.lower()

    def test_anomaly_flag_reflected_in_summary(self, high_risk_customer):
        ctx = build_context(
            high_risk_customer,
            anomaly_summary="[⚠ FLAGGED] complaint_spike: 3 days detected."
        )
        summary = build_stub_summary(ctx)
        assert "anomal" in summary.lower() or "segment" in summary.lower() or "campaign" in summary.lower()

    def test_different_channels_produce_different_recommendations(self):
        base = {
            "customer_id": "C1", "segment": "Standard",
            "sentiment_text": "neutral", "engagement_score": 0.3,
            "complaint_flag": 0, "escalation_flag": 0,
            "days_since_last_contact": 30,
        }
        email_ctx = build_context({**base, "channel": "Email"})
        sms_ctx   = build_context({**base, "channel": "SMS"})
        assert build_stub_summary(email_ctx) != build_stub_summary(sms_ctx)


# ── build_context tests ────────────────────────────────────────────────────────

class TestBuildContext:


    def test_null_engagement_score_no_crash(self):
        """build_context must not raise on None numeric values."""
        ctx = build_context({"customer_id": "C1", "engagement_score": None,
                             "complaint_flag": None, "days_since_last_contact": None})
        assert ctx["engagement_score"] == 0.0
        assert ctx["complaint_flag"] == 0
        assert ctx["days_since_contact"] == 0

    def test_nan_values_no_crash(self):
        """build_context must not raise on NaN numeric values."""
        ctx = build_context({"engagement_score": float("nan"), "complaint_flag": float("nan")})
        assert ctx["engagement_score"] == 0.0
        assert ctx["complaint_flag"] == 0

    def test_unparseable_string_no_crash(self):
        """build_context must not raise on unparseable string values."""
        ctx = build_context({"engagement_score": "bad", "complaint_flag": "also_bad"})
        assert ctx["engagement_score"] == 0.0
        assert ctx["complaint_flag"] == 0

    def test_null_customer_generates_summary(self):
        """generate_summary must not crash when customer has None fields."""
        result = generate_summary(
            {"customer_id": "C_NULL", "engagement_score": None, "complaint_flag": None},
            force_stub=True,
        )
        assert isinstance(result.summary, str)
        assert len(result.summary) > 0

    def test_returns_dict(self, high_risk_customer):
        ctx = build_context(high_risk_customer)
        assert isinstance(ctx, dict)

    def test_customer_id_preserved(self, high_risk_customer):
        ctx = build_context(high_risk_customer)
        assert ctx["customer_id"] == "CUST_HIGH_001"

    def test_score_result_applied(self, high_risk_customer, mock_score_result):
        ctx = build_context(high_risk_customer, score_result=mock_score_result)
        assert ctx["intervention_score"] == 0.87
        assert ctx["risk_band"] == "High"

    def test_defaults_when_no_score(self, high_risk_customer):
        ctx = build_context(high_risk_customer)
        assert ctx["intervention_score"] == 0.0
        assert ctx["risk_band"] == "Unknown"

    def test_anomaly_summary_stored(self, high_risk_customer):
        ctx = build_context(high_risk_customer, anomaly_summary="test anomaly text")
        assert ctx["anomaly_summary"] == "test anomaly text"

    def test_missing_fields_get_defaults(self):
        ctx = build_context({})
        assert ctx["customer_id"] == "Unknown"
        assert ctx["engagement_score"] == 0.0
        assert ctx["complaint_flag"] == 0


# ── generate_summary tests ─────────────────────────────────────────────────────

class TestGenerateSummary:

    def test_returns_summary_result(self, high_risk_customer):
        result = generate_summary(high_risk_customer, force_stub=True)
        assert isinstance(result, SummaryResult)

    def test_stub_source_when_forced(self, high_risk_customer):
        result = generate_summary(high_risk_customer, force_stub=True)
        assert result.source == "stub"
        assert result.model == "stub"

    def test_summary_is_non_empty_string(self, high_risk_customer):
        result = generate_summary(high_risk_customer, force_stub=True)
        assert isinstance(result.summary, str)
        assert len(result.summary) > 50

    def test_customer_id_in_result(self, high_risk_customer):
        result = generate_summary(high_risk_customer, force_stub=True)
        assert result.customer_id == "CUST_HIGH_001"

    def test_with_score_result(self, high_risk_customer, mock_score_result):
        result = generate_summary(
            high_risk_customer,
            score_result=mock_score_result,
            force_stub=True,
        )
        assert isinstance(result.summary, str)
        assert len(result.summary) > 0

    def test_with_anomaly_context(self, high_risk_customer):
        result = generate_summary(
            high_risk_customer,
            anomaly_summary="[⚠ FLAGGED] complaint_spike detected",
            force_stub=True,
        )
        assert isinstance(result.summary, str)


    def test_transient_error_falls_back_to_stub(self, high_risk_customer, monkeypatch):
        """Transient API errors (timeout, rate limit) must fall back to stub."""
        import src.genai.summarizer as mod
        def fake_call(context, cfg):
            raise Exception("Connection timeout after 30s")
        monkeypatch.setattr(mod, "_call_openai", fake_call)
        monkeypatch.setenv("OPENAI_API_KEY", "sk-fake-key")
        result = mod.generate_summary(high_risk_customer)
        assert result.source == "stub"

    def test_auth_error_raises(self, high_risk_customer, monkeypatch):
        """Auth errors must re-raise so config problems are visible."""
        import src.genai.summarizer as mod
        def fake_call(context, cfg):
            raise Exception("401 Unauthorized - invalid api key")
        monkeypatch.setattr(mod, "_call_openai", fake_call)
        monkeypatch.setenv("OPENAI_API_KEY", "sk-fake-key")
        with pytest.raises(Exception, match="401"):
            mod.generate_summary(high_risk_customer)

    def test_stub_used_when_no_api_key(self, high_risk_customer, monkeypatch):
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        result = generate_summary(high_risk_customer)
        assert result.source == "stub"

    def test_high_and_low_risk_produce_different_summaries(
        self, high_risk_customer, low_risk_customer, mock_score_result
    ):
        from src.modeling.score import ScoreResult
        low_score = ScoreResult(
            customer_id="CUST_LOW_001",
            intervention_score=0.10,
            needs_intervention=0,
            risk_band="Low",
        )
        high_result = generate_summary(high_risk_customer, score_result=mock_score_result, force_stub=True)
        low_result  = generate_summary(low_risk_customer,  score_result=low_score,         force_stub=True)
        assert high_result.summary != low_result.summary
