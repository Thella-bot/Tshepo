"""Anti-fraud tests: the attacks an applicant who controls the phone can run.

The point of these tests is not that the system is unbreakable - it is not,
and no amount of message parsing changes that - but that the attacks that can
be seen from inside the data are seen, and that the honest wallets that carry
the target population are not sent to manual review for nothing.
"""
from __future__ import annotations

import json
from datetime import datetime, timedelta

import pytest
from fastapi.testclient import TestClient

from tshepo.api import app
from tshepo.antiabuse import body_hash, message_hashes, replay_reuse
from tshepo.features.cashflow import build_features, component_values
from tshepo.parsing.rules import build_ledger
from tshepo.store import Store

MP = "MPESA"
END = datetime(2026, 10, 20, 12, 0, 0)


def recv_in(amt, phone, name, bal, when, ref):
    return (f"{ref} Confirmed. You have received M{amt:.2f} from {name} {phone} "
            f"on {when.strftime('%d/%m/%y')} at {when.strftime('%I:%M %p')}. "
            f"New M-Pesa balance is M{bal:.2f}. Customer Care :114")


def send_out(amt, phone, name, bal, when, ref):
    return (f"{ref} Confirmed. M{amt:.2f} sent to {phone} - {name} on "
            f"{when.strftime('%d/%m/%y')} at {when.strftime('%I:%M %p')} "
            f"New M-Pesa balance is M{bal:.2f} Reference: .Customer Care :114")


def _history(days=40, per_day=2, own_wallets=0, clients=10, repayments=0):
    """A wallet history. ``own_wallets`` makes the money come from the
    applicant's own accounts and go straight back to them."""
    recs = []
    bal = 0.0
    t = END - timedelta(days=days)
    n = 0
    day = 0
    while t <= END:
        n += 1
        ref = f"08DNB{n:06d}"
        if own_wallets:
            phone, name = f"581234{day % own_wallets:02d}", f"MY WALLET {day % own_wallets}"
        else:
            phone, name = f"630590{day % clients:02d}", f"CLIENT PAYPOINT {day % clients}"
        bal += 500.0
        recs.append({"body": recv_in(500.0, phone, name, bal, t, ref), "sender": MP,
                     "received_at": t.isoformat(timespec="minutes")})
        if own_wallets:
            bal -= 500.0
            recs.append({"body": send_out(500.0, phone, name, bal, t + timedelta(hours=6),
                                          f"08DNB{n:06d}X"),
                         "sender": MP,
                         "received_at": (t + timedelta(hours=6)).isoformat(timespec="minutes")})
        t += timedelta(hours=12)
        day += 1
    return recs


def _score(recs):
    from tshepo.pipeline import score_user
    return score_user(recs, user_label="T")


# ---------------------------------------------------------------- circular flow
def test_money_that_comes_back_out_is_flagged():
    """The applicant's own wallets, paying in and taking the money back out.

    Every message here is a genuine operator alert. The only thing wrong with
    it is that the money never left the applicant.
    """
    p = _score(_history(own_wallets=10))
    assert p["fraud_signals"]["circular_share"] > 0.9
    assert p["decision"] == "refer_manual_review"
    assert any("circulating" in i for i in p["data_quality"]["issues"])


def test_a_genuine_wallet_is_not_flagged_as_circular():
    p = _score(_history())
    assert p["fraud_signals"]["circular_share"] == 0.0
    assert not any("circulating" in i for i in p["data_quality"]["issues"])


def test_a_small_back_and_forth_is_not_circular():
    """Two payments to and from the same person is a favour, not a scheme."""
    recs = [{"body": recv_in(500.0, "58123456", "MY BROTHER", 500.0, END, "08DNB000001"),
              "sender": MP, "received_at": END.isoformat(timespec="minutes")},
            {"body": send_out(500.0, "58123456", "MY BROTHER", 0.0, END, "08DNB000002"),
             "sender": MP, "received_at": END.isoformat(timespec="minutes")}]
    p = _score(recs)
    assert not any("circulating" in i for i in p["data_quality"]["issues"])


# ---------------------------------------------------------------- repayment
def test_repayments_need_corroboration():
    """Four identical 'repayments' spread over months are still a line in a
    script: a real repayment changes amount from month to month."""
    recs = _history(days=40, clients=10)
    bal = 0.0
    t = END - timedelta(days=80)
    for i in range(4):
        bal -= 300.0
        recs.append({"body": send_out(300.0, "57850818", "MOFO TSHEPO LOAN REPAYMENT",
                                      bal, t, f"08DNBR{i:06d}"),
                     "sender": MP, "received_at": t.isoformat(timespec="minutes")})
        t += timedelta(days=25)
    ledger = build_ledger(recs)
    f = build_features(ledger.transactions, ledger=ledger)
    values = component_values(f)
    assert f["n_loan_repay"] == 4
    assert f["repay_months"] >= 3
    assert f["repay_variety"] == 1
    assert values["repayment_habit"] < 1.0


def test_a_repayment_habit_still_scores():
    """Repayments that recur over months, with the amount moving, keep the marks."""
    recs = _history(days=40, clients=10)
    bal = 0.0
    t = END - timedelta(days=80)
    for i, amount in enumerate((300.0, 310.0, 295.0, 305.0)):
        bal -= amount
        recs.append({"body": send_out(amount, "57850818", "MOFO TSHEPO LOAN REPAYMENT",
                                      bal, t, f"08DNBR{i:06d}"),
                     "sender": MP, "received_at": t.isoformat(timespec="minutes")})
        t += timedelta(days=25)
    ledger = build_ledger(recs)
    f = build_features(ledger.transactions, ledger=ledger)
    values = component_values(f)
    assert f["repay_months"] >= 3
    assert f["repay_variety"] >= 2
    assert values["repayment_habit"] > 0.5


# ---------------------------------------------------------------- timestamps
def test_a_edited_timestamp_is_counted():
    """The message says 10/10/26, the export says 20/10/26."""
    recs = [{"body": recv_in(500.0, "63059001", "CLIENT", 500.0,
                             datetime(2026, 10, 10, 12, 0), "08DNB000001"),
             "sender": MP, "received_at": "2026-10-20T12:00"}]
    ledger = build_ledger(recs)
    assert ledger.stats["timestamp_mismatch"] == 1
    p = _score(recs)
    assert p["fraud_signals"]["timestamp_mismatches"] == 1
    assert any("disagrees" in i for i in p["data_quality"]["issues"])


def test_matching_timestamps_are_not_a_mismatch():
    recs = [{"body": recv_in(500.0, "63059001", "CLIENT", 500.0,
                             datetime(2026, 10, 20, 12, 0), "08DNB000001"),
             "sender": MP, "received_at": "2026-10-20T12:00"}]
    assert build_ledger(recs).stats["timestamp_mismatch"] == 0


# ---------------------------------------------------------------- the shape
def test_a_wallet_that_only_receives_is_flagged_when_it_is_big_enough():
    recs = _history(days=40, per_day=1, clients=10)
    p = _score(recs)
    assert p["fraud_signals"]["income_concentration"] > 0.9
    assert any("only receives" in i for i in p["data_quality"]["issues"])


def test_a_thin_file_of_incoming_payments_is_not_flagged():
    """Two payments are not a pattern."""
    recs = [{"body": recv_in(500.0, "63059001", "CLIENT", 500.0, END, "08DNB000001"),
             "sender": MP, "received_at": END.isoformat(timespec="minutes")},
            {"body": recv_in(500.0, "63059002", "CLIENT", 1000.0, END + timedelta(hours=1),
                             "08DNB000002"),
             "sender": MP, "received_at": (END + timedelta(hours=1)).isoformat(timespec="minutes")}]
    p = _score(recs)
    assert not any("only receives" in i for i in p["data_quality"]["issues"])


# ---------------------------------------------------------------- the passport
def test_the_passport_says_where_the_score_comes_from():
    p = _score(_history())
    assert p["verification"] == "sms_unverified"
    assert set(p["fraud_signals"]) >= {"circular_share", "income_concentration",
                                       "top_payer_share", "repayments_seen"}


# ---------------------------------------------------------------- replay
def test_the_same_history_used_by_two_applicants_is_flagged(tmp_path, monkeypatch):
    store = Store(str(tmp_path / "replay.db"))
    monkeypatch.setattr("tshepo.api.get_store", lambda: store)
    client = TestClient(app)
    messages = [{"body": f"08DNB{i:06d} Confirmed. You have received M500.00 from CLIENT 63059001 "
                        f"on 10/10/26 at 12:00 PM. New M-Pesa balance is M500.00.",
                 "sender": MP, "received_at": "2026-10-10T12:00"} for i in range(12)]

    first = client.post("/v1/consents", json={"user_id": "U1"}).json()
    h1 = {"X-Consent-Id": first["consent_id"], "X-Consent-Token": first["consent_token"]}
    client.post("/v1/ingest", headers=h1, json={"messages": messages})
    r1 = client.post(f"/v1/score?consent_id={first['consent_id']}", headers=h1)
    assert r1.status_code == 200
    assert "replayed_history" not in r1.json()["fraud_signals"]

    second = client.post("/v1/consents", json={"user_id": "U2"}).json()
    h2 = {"X-Consent-Id": second["consent_id"], "X-Consent-Token": second["consent_token"]}
    client.post("/v1/ingest", headers=h2, json={"messages": messages})
    r2 = client.post(f"/v1/score?consent_id={second['consent_id']}", headers=h2)
    body = r2.json()
    assert body["fraud_signals"]["replayed_history"] == "U1"
    assert body["decision"] == "refer_manual_review"


def test_scoring_the_same_history_twice_is_not_a_replay(tmp_path, monkeypatch):
    store = Store(str(tmp_path / "replay2.db"))
    monkeypatch.setattr("tshepo.api.get_store", lambda: store)
    client = TestClient(app)
    messages = [{"body": f"08DNB{i:06d} Confirmed. You have received M500.00 from CLIENT 63059001 "
                        f"on 10/10/26 at 12:00 PM. New M-Pesa balance is M500.00.",
                 "sender": MP, "received_at": "2026-10-10T12:00"} for i in range(12)]
    consent = client.post("/v1/consents", json={"user_id": "U1"}).json()
    h = {"X-Consent-Id": consent["consent_id"], "X-Consent-Token": consent["consent_token"]}
    client.post("/v1/ingest", headers=h, json={"messages": messages})
    client.post(f"/v1/score?consent_id={consent['consent_id']}", headers=h)
    r = client.post(f"/v1/score?consent_id={consent['consent_id']}", headers=h)
    assert "replayed_history" not in r.json()["fraud_signals"]


def test_a_shared_message_is_not_enough_on_its_own():
    """Two people both paid by the same agent share a few messages. That is
    not a replay, and it must not stop a score."""
    assert replay_reuse({"U2": 3}, 40) == (False, "U2", 3)
    assert replay_reuse({"U2": 30}, 40) == (True, "U2", 30)


def test_message_hashes_ignore_whitespace():
    assert body_hash("  a  b ") == body_hash("a b")
    assert len(message_hashes([{"body": "a"}, {"body": "a"}, {"body": "b"}])) == 2


# ---------------------------------------------------------------- erasure
def test_erasing_a_user_removes_their_message_hashes(tmp_path):
    store = Store(str(tmp_path / "erase.db"))
    store.record_message_hashes("U1", ["h1", "h2"])
    assert store.shared_message_counts("U2", ["h1", "h2"]) == {"U1": 2}
    store.delete_user_data("U1")
    assert store.shared_message_counts("U2", ["h1", "h2"]) == {}
