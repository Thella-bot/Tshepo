"""Security headers tests."""
from __future__ import annotations

from fastapi.testclient import TestClient

from tshepo.api import app


def test_security_headers_present():
    client = TestClient(app)
    r = client.get("/health")
    assert r.headers["x-content-type-options"] == "nosniff"
    assert r.headers["x-frame-options"] == "DENY"
    assert r.headers["referrer-policy"] == "no-referrer"
    assert r.headers["content-security-policy"] == "default-src 'none'"


def test_correlation_id_header_returned():
    client = TestClient(app)
    cid = "test-cid-123"
    r = client.get("/health", headers={"x-correlation-id": cid})
    assert r.headers["x-correlation-id"] == cid
    assert r.json()["ok"] is True


def test_predict_response_has_correlation_id():
    client = TestClient(app)
    r = client.post(
        "/predict",
        json={"body": "RK12AB34CD Confirmed. You have received M100.00 from THABO "
                      "on 19/10/26 at 2:15 PM.", "sender": "MPESA", "model": "rules"},
    )
    assert r.status_code == 200
    assert "correlation_id" in r.json() and r.json()["correlation_id"]
    assert "x-correlation-id" in r.headers
