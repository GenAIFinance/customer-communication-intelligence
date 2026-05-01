"""
test_api.py — Tests for all FastAPI endpoints.

Uses TestClient (synchronous) so no async setup needed.
Model is loaded once per module for speed.

Run with:
    python -m pytest tests/test_api.py -v
"""

import pytest
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fastapi.testclient import TestClient
from src.api.main import app, _model, _feature_names
import src.api.main as main_module


# ── Setup ──────────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module", autouse=True)
def load_model_for_tests():
    """Load model into app state once for all tests."""
    from src.modeling.score import load_model
    from src.modeling.train_model import load_feature_names
    main_module._model = load_model()
    main_module._feature_names = load_feature_names()


@pytest.fixture(scope="module")
def client():
    return TestClient(app)


@pytest.fixture
def sample_customer():
    return {
        "customer_id":             "CUST_TEST_001",
        "segment":                 "Premium",
        "product_type":            "Home",
        "channel":                 "Email",
        "campaign_id":             "CAMP_001",
        "sent_date":               "2026-01-01",
        "opened":                  0,
        "clicked":                 0,
        "response_flag":           0,
        "complaint_flag":          1,
        "escalation_flag":         0,
        "engagement_score":        0.12,
        "sentiment_text":          "negative",
        "premium_bucket":          "High",
        "tenure_months":           36,
        "days_since_last_contact": 75,
        "opt_out_flag":            0,
    }


# ── GET /health ────────────────────────────────────────────────────────────────

class TestHealth:

    def test_health_returns_200(self, client):
        response = client.get("/health")
        assert response.status_code == 200

    def test_health_response_shape(self, client):
        data = client.get("/health").json()
        assert "status" in data
        assert "version" in data

    def test_health_status_ok(self, client):
        data = client.get("/health").json()
        assert data["status"] == "ok"

    def test_health_version_string(self, client):
        data = client.get("/health").json()
        assert isinstance(data["version"], str)
        assert len(data["version"]) > 0


# ── POST /score-customer ───────────────────────────────────────────────────────

class TestScoreCustomer:

    def test_returns_200(self, client, sample_customer):
        response = client.post("/score-customer", json=sample_customer)
        assert response.status_code == 200

    def test_response_has_required_fields(self, client, sample_customer):
        data = client.post("/score-customer", json=sample_customer).json()
        assert "customer_id" in data
        assert "intervention_score" in data
        assert "needs_intervention" in data
        assert "risk_band" in data

    def test_intervention_score_in_range(self, client, sample_customer):
        data = client.post("/score-customer", json=sample_customer).json()
        assert 0.0 <= data["intervention_score"] <= 1.0

    def test_needs_intervention_is_binary(self, client, sample_customer):
        data = client.post("/score-customer", json=sample_customer).json()
        assert data["needs_intervention"] in [0, 1]

    def test_risk_band_valid(self, client, sample_customer):
        data = client.post("/score-customer", json=sample_customer).json()
        assert data["risk_band"] in ["Low", "Medium", "High"]

    def test_customer_id_preserved(self, client, sample_customer):
        data = client.post("/score-customer", json=sample_customer).json()
        assert data["customer_id"] == "CUST_TEST_001"

    def test_partial_payload_defaults(self, client):
        """Minimal payload — missing categoricals use defaults."""
        minimal = {
            "opened": 0, "clicked": 0, "response_flag": 0,
            "complaint_flag": 1, "escalation_flag": 0,
            "engagement_score": 0.1, "tenure_months": 12,
            "days_since_last_contact": 60, "opt_out_flag": 0,
        }
        response = client.post("/score-customer", json=minimal)
        assert response.status_code == 200

    def test_high_risk_scores_high(self, client):
        """High-risk customer should score above 0.5."""
        risky = {
            "segment": "Basic", "product_type": "Auto", "channel": "Email",
            "opened": 0, "clicked": 0, "response_flag": 0,
            "complaint_flag": 1, "escalation_flag": 1,
            "engagement_score": 0.05, "tenure_months": 6,
            "days_since_last_contact": 90, "opt_out_flag": 0,
        }
        data = client.post("/score-customer", json=risky).json()
        assert data["intervention_score"] > 0.5

    def test_engagement_score_out_of_range_rejected(self, client, sample_customer):
        """engagement_score > 1.0 should be rejected by Pydantic."""
        bad = {**sample_customer, "engagement_score": 1.5}
        response = client.post("/score-customer", json=bad)
        assert response.status_code == 422


# ── POST /detect-anomaly ───────────────────────────────────────────────────────

class TestDetectAnomaly:

    def test_returns_200_no_body(self, client):
        response = client.post("/detect-anomaly", json={})
        assert response.status_code == 200

    def test_response_has_required_fields(self, client):
        data = client.post("/detect-anomaly", json={}).json()
        assert "results" in data
        assert "any_flagged" in data
        assert "total_flagged" in data

    def test_all_detectors_run_by_default(self, client):
        data = client.post("/detect-anomaly", json={}).json()
        assert len(data["results"]) == 3

    def test_detector_names_correct(self, client):
        data = client.post("/detect-anomaly", json={}).json()
        names = {r["detector"] for r in data["results"]}
        assert "segment_engagement_drop" in names
        assert "complaint_spike" in names
        assert "campaign_underperformance" in names

    def test_single_detector_segment(self, client):
        data = client.post("/detect-anomaly",
                           json={"detector_type": "segment_engagement_drop"}).json()
        assert len(data["results"]) == 1
        assert data["results"][0]["detector"] == "segment_engagement_drop"

    def test_single_detector_complaint(self, client):
        data = client.post("/detect-anomaly",
                           json={"detector_type": "complaint_spike"}).json()
        assert len(data["results"]) == 1
        assert data["results"][0]["detector"] == "complaint_spike"

    def test_single_detector_campaign(self, client):
        data = client.post("/detect-anomaly",
                           json={"detector_type": "campaign_underperformance"}).json()
        assert len(data["results"]) == 1

    def test_any_flagged_is_bool(self, client):
        data = client.post("/detect-anomaly", json={}).json()
        assert isinstance(data["any_flagged"], bool)

    def test_total_flagged_consistent(self, client):
        data = client.post("/detect-anomaly", json={}).json()
        flagged_count = sum(1 for r in data["results"] if r["flagged"])
        assert data["total_flagged"] == flagged_count


# ── POST /generate-summary ─────────────────────────────────────────────────────

class TestGenerateSummary:

    def test_returns_200(self, client, sample_customer):
        response = client.post("/generate-summary", json={
            "customer": sample_customer,
            "force_stub": True,
        })
        assert response.status_code == 200

    def test_response_has_required_fields(self, client, sample_customer):
        data = client.post("/generate-summary", json={
            "customer": sample_customer,
            "force_stub": True,
        }).json()
        assert "customer_id" in data
        assert "summary" in data
        assert "source" in data
        assert "model" in data

    def test_summary_is_non_empty(self, client, sample_customer):
        data = client.post("/generate-summary", json={
            "customer": sample_customer,
            "force_stub": True,
        }).json()
        assert len(data["summary"]) > 50

    def test_stub_source_when_forced(self, client, sample_customer):
        data = client.post("/generate-summary", json={
            "customer": sample_customer,
            "force_stub": True,
        }).json()
        assert data["source"] == "stub"

    def test_with_anomaly_context(self, client, sample_customer):
        data = client.post("/generate-summary", json={
            "customer": sample_customer,
            "anomaly_summary": "[⚠ FLAGGED] complaint_spike detected",
            "force_stub": True,
        }).json()
        assert isinstance(data["summary"], str)

    def test_customer_id_in_response(self, client, sample_customer):
        data = client.post("/generate-summary", json={
            "customer": sample_customer,
            "force_stub": True,
        }).json()
        assert data["customer_id"] == "CUST_TEST_001"

    def test_partial_customer_no_crash(self, client):
        """Partial customer payload must not crash the endpoint."""
        data = client.post("/generate-summary", json={
            "customer": {
                "complaint_flag": 1,
                "engagement_score": 0.1,
            },
            "force_stub": True,
        }).json()
        assert "summary" in data
