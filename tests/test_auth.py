"""Auth behaviour tests.

Patches ``tshepo.api.API_KEY`` per test so the write endpoints require a valid
``X-API-Key`` header without polluting other test modules.
"""
from __future__ import annotations

from unittest import mock

from fastapi.testclient import TestClient

from tshepo.api import app

PAYLOAD = {"body": "RK12AB34CD Confirmed. You have received M100.00 from THABO "
                   "on 19/10/26 at 2:15 PM.", "sender": "MPESA", "model": "rules"}

SCORE_PAYLOAD = {
    "messages": [
        {"body": "RK12AB34CD Confirmed. You have received M100.00 from THABO "
                 "on 19/10/26 at 2:15 PM.", "sender": "MPESA", "user_id": "U1"}
    ]
}


def _client() -> TestClient:
    return TestClient(app)


def test_predict_requires_api_key():
    """No API key -> 401."""
    with mock.patch("tshepo.api.API_KEY", "testkey12345"):
        r = _client().post("/predict", json=PAYLOAD)
    assert r.status_code == 401
    assert r.json()["detail"] == "invalid or missing API key"


def test_predict_rejects_wrong_key():
    """Wrong API key -> 401."""
    with mock.patch("tshepo.api.API_KEY", "testkey12345"):
        r = _client().post("/predict", json=PAYLOAD, headers={"x-api-key": "wrong"})
    assert r.status_code == 401


def test_predict_accepts_correct_key():
    """Correct API key -> 200."""
    with mock.patch("tshepo.api.API_KEY", "testkey12345"):
        r = _client().post(
            "/predict", json=PAYLOAD, headers={"x-api-key": "testkey12345"}
        )
    assert r.status_code == 200
    assert r.json()["label"] in ("genuine_transaction", "suspicious", "noise")


def test_score_requires_api_key():
    """No API key on /score -> 401."""
    with mock.patch("tshepo.api.API_KEY", "testkey12345"):
        r = _client().post("/score", json=SCORE_PAYLOAD)
    assert r.status_code == 401


def test_score_accepts_correct_key():
    """Correct API key on /score -> 200."""
    with mock.patch("tshepo.api.API_KEY", "testkey12345"):
        r = _client().post(
            "/score", json=SCORE_PAYLOAD, headers={"x-api-key": "testkey12345"}
        )
    assert r.status_code == 200
    assert "U1" in r.json()["passports"]


def test_health_and_metrics_stay_open():
    """Read-only endpoints do not require auth even when TSHEPO_API_KEY is set."""
    with mock.patch("tshepo.api.API_KEY", "testkey12345"):
        r = _client().get("/health")
    assert r.status_code == 200
    r = _client().get("/metrics")
    assert r.status_code == 200
