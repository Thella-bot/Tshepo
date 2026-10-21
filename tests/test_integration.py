"""Integration tests: full API end-to-end via FastAPI TestClient."""
from __future__ import annotations

from datetime import datetime, timedelta

from fastapi.testclient import TestClient

from tshepo.api import app


def coherent_history(days: int = 30, per_day: int = 2) -> list[dict]:
    """A wallet history that passes every gate the service applies.

    A score is only issued when the data supports it: the history has to
    cover a quarter of the 13-week window, the balance chain has to hold, no
    gap of 24 hours or more may sit between two alerts, and every message
    has to be readable. Two alerts a day for a month does all of that.
    """
    msgs: list[dict] = []
    bal = 0.0
    start = datetime(2026, 9, 20, 8, 0)
    n = 0
    for d in range(days):
        for k in range(per_day):
            t = start + timedelta(days=d, hours=6 * k)
            n += 1
            stamp = f"{t.day}/{t.month}/{t:%y} at {t.hour % 12 or 12}:{t:%M} {'AM' if t.hour < 12 else 'PM'}"
            if k == 0:
                bal += 100.0
                body = (f"Transact ID 08TEST{n:05d} Confirmed. You have received M100.00 from "
                        f"THABO 58123456 on {stamp}. New M-Pesa balance is M{bal:.2f}.")
            else:
                bal -= 20.0
                body = (f"Transact ID 08TEST{n:05d} Confirmed. You have paid M20.00 to "
                        f"KHALA 58123456 on {stamp}. New M-Pesa balance is M{bal:.2f}.")
            msgs.append({"body": body, "sender": "MPESA", "user_id": "U1",
                         "received_at": t.isoformat(timespec="minutes")})
    return msgs


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
            "messages": coherent_history() + [
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
    assert p["decision"] == "score"
    # the scam message is vetoed, and the passport says how much was read
    assert p["data_quality"]["messages_unread"] == 0
    assert p["data_quality"]["readable_share"] == 1.0


def test_score_endpoint_withholds_the_score_when_the_history_is_too_short():
    """One transaction cannot support a lending decision.

    The service must say so rather than return a confident number, and it
    must still answer.
    """
    client = TestClient(app)
    r = client.post(
        "/score",
        json={
            "messages": [
                {"body": "RK12AB34CD Confirmed. You have received M100.00 from THABO "
                         "58123456 on 19/10/26 at 2:15 PM. New balance is M900.00.",
                 "sender": "MPESA", "user_id": "U1",
                 "received_at": "2026-10-19T14:15:00"},
            ]
        },
    )
    assert r.status_code == 200
    p = r.json()["passports"]["U1"]
    assert p["score"] is None
    assert p["score_withheld"] is True
    assert p["decision"] == "insufficient_data"
    assert p["status"] == "insufficient_data"


def test_metrics_endpoint():
    client = TestClient(app)
    r = client.get("/metrics")
    assert r.status_code == 200
    assert "tshepo_http_requests_total" in r.text
    assert r.headers["content-type"].startswith("text/plain")


def test_version_endpoint():
    client = TestClient(app)
    r = client.get("/version")
    assert r.status_code == 200
    assert r.json()["service"] == "tshepo"
    assert r.json()["version"] == "0.2.0"
