"""Cash-flow features, ledger integrity, scorecard and passport."""
import pytest

from tshepo.features.cashflow import (FEATURES, band, build_features, ledger_integrity,
                                       make_passport, scorecard)
from tshepo.parsing.parser import extract_transactions


def _row(ts, direction, amount, balance, category="transfer", fee=0.0, op="mpesa"):
    return dict(timestamp=ts, direction=direction, amount=amount, fee=fee, balance=balance,
                category=category, operator=op, counterparty_name="PAYER", counterparty_phone=None)


def test_features_shape_and_bounds():
    rows = [_row("2026-09-01T10:00", "in", 1000, 1000),
            _row("2026-09-15T10:00", "out", 200, 800, "merchant")]
    f = build_features(rows)
    assert set(f) == set(FEATURES)
    assert f["n_txn"] == 2
    assert f["monthly_income"] > 0
    assert 0 <= f["integrity"] <= 1


def test_integrity_detects_broken_chain():
    good = [_row("2026-09-01T10:00", "in", 1000, 1000),
            _row("2026-09-02T10:00", "out", 200, 800, "merchant")]
    share, n = ledger_integrity(good)
    assert share == 1.0 and n == 1

    forged = [_row("2026-09-01T10:00", "in", 1000, 1000),
              _row("2026-09-02T10:00", "in", 5000, 9999)]  # balance does not follow the amount
    share, _ = ledger_integrity(forged)
    assert share == 0.0


def test_empty_ledger_is_safe():
    f = build_features([])
    assert f["n_txn"] == 0
    assert f["integrity"] == 1.0
    # No transactions means no evidence for any component, so there is no
    # score: an empty history is a data finding, not a low-risk finding.
    assert scorecard(f)["score"] is None
    assert scorecard(f)["provisional"] is True
    assert len(scorecard(f)["withheld"]) == len(scorecard(f)["components"])


def test_scorecard_range_and_bands():
    rows = [_row(f"2026-09-{d:02d}T10:00", "in", 3000, 3000 + d * 100) for d in range(1, 20)]
    sc = scorecard(build_features(rows))
    assert 300 <= sc["score"] <= 850
    assert band(sc["score"]) in {"Very low", "Low", "Fair", "Good", "Very good"}
    scored = {k: v for k, v in sc["components"].items() if v is not None}
    assert all(0 <= v <= 1 for v in scored.values())
    # A component with no evidence is withheld, not scored as a zero.
    assert sc["components"]["history_depth"] is None
    assert "history_depth" in sc["withheld"]


def test_passport_contains_no_raw_message():
    records = [{"body": "RK12AB34CD Confirmed. You have received M100.00 from THABO MOKOENA 58123456 "
                         "on 19/10/26 at 2:15 PM. New M-PESA balance is M900.00.",
                "sender": "MPESA", "received_at": "2026-10-19T14:15", "user_id": "U1"}]
    p = make_passport(extract_transactions(records), user_label="U1")
    blob = str(p)
    assert "MOKOENA" not in blob and "58123456" not in blob and "RK12AB34CD" not in blob
    assert p["applicant"] == "U1"
    # One message is not a history: the passport says so instead of scoring.
    assert p["score"] is None
    assert p["status"] == "insufficient_data"
    assert p["data_integrity"] in {"consistent", "REVIEW: balance chain has breaks"}


@pytest.mark.parametrize("n", [0, 1, 2, 5])
def test_build_features_handles_any_ledger_size(n):
    rows = [_row("2026-09-01T10:00", "in", 100, 100 + i) for i in range(n)]
    f = build_features(rows)
    assert f["n_txn"] == n


def test_new_features_present_and_bounded():
    rows = [_row(f"2026-09-{d:02d}T{h:02d}:00", "in", 3000, 3000 + d * 100) for d in range(1, 20)
            for h in (10, 23)]
    f = build_features(rows)
    for k in FEATURES:
        assert k in f
    assert 0 <= f["night_txn_share"] <= 1
    assert 0 <= f["weekend_txn_share"] <= 1
    assert 0 <= f["fee_burden"] <= 1
    assert 0 <= f["round_amount_share"] <= 1
    assert 0 <= f["dow_regularity"] <= 1
    assert f["txn_per_day"] >= 0
    assert f["amount_velocity"] >= 0


def test_night_txn_share_detects_night_activity():
    rows = [_row("2026-09-01T23:00", "in", 100, 100), _row("2026-09-02T10:00", "out", 10, 90)]
    f = build_features(rows)
    assert f["night_txn_share"] == 0.5


def test_fee_burden_computed():
    rows = [_row("2026-09-01T10:00", "out", 100, 900, fee=5.0),
            _row("2026-09-02T10:00", "out", 100, 795, fee=5.0)]
    f = build_features(rows)
    assert abs(f["fee_burden"] - (10.0 / 210.0)) < 0.01


def test_empty_ledger_new_features_zero():
    f = build_features([])
    assert f["txn_per_day"] == 0.0
    assert f["amount_velocity"] == 0.0
    assert f["night_txn_share"] == 0.0
    assert f["dow_regularity"] == 0.0

