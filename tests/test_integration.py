"""Integration tests: full API end-to-end via FastAPI TestClient."""
from __future__ import annotations

from fastapi.testclient import TestClient

from tshepo.api import app


def test_health_endpoint():
    client = TestClient(app)
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["ok"] is True
    assert "parser" in body and "features" in body and "scorecard" in body


def test_predict_endpoint_rules():
    client = TestClient(app)
    r = client.post(
        "/predict",
        json={"body": "RK12AB34CD Confirmed. You have received M100.00 from THABO "
                    "58123456 on 19/10/26 at 2:15 PM. New M-PESA balance is M900.00.",
              "sender": "MPESA", "model": "rules"},
    )
    assert r.status_code == 200
    data = r.json()
    assert data["label"] in ("genuine_transaction", "suspicious", "noise")
    assert data["model"] == "rules"
    assert "correlation_id" in data and data["correlation_id"]


def test_predict_endpoint_rejects_oversized():
    client = TestClient(app)
    r = client.post(
        "/predict",
        json={"body": "x" * 5000, "sender": "MPESA", "model": "rules"},
    )
    # Pydantic validates max_length=4096 before the security layer runs.
    assert r.status_code in (400, 422)


def test_score_endpoint_end_to_end():
    client = TestClient(app)
    r = client.post(
        "/score",
        json={
            "messages": [
                {"body": "RK12AB34CD Confirmed. You have received M100.00 from THABO "
                         "58123456 on 19/10/26 at 2:15 PM. New balance is M900.00.",
                 "sender": "MPESA", "user_id": "U1",
                 "received_at": "2026-10-19T14:15:00"},
                {"body": "Your PIN is required to claim your prize. Call 0800SCAM now.",
                 "sender": "UNKNOWN", "user_id": "U1",
                 "received_at": "2026-10-19T14:20:00"},
            ]
        },
    )
    assert r.status_code == 200
    data = r.json()
    assert data["users"] == 1
    assert "passports" in data
    assert "U1" in data["passports"]
    p = data["passports"]["U1"]
    assert 300 <= p["score"] <= 850
    assert p["band"] in ("Very low", "Low", "Fair", "Good", "Very good")


def test_metrics_endpoint():
    client = TestClient(app)
    r = client.get("/metrics")
    assert r.status_code == 200
    assert "tshepo_http_requests_total" in r.text
    assert r.headers["content-type"].startswith("text/plain")
