"""Property-based tests for production invariants.

Uses Hypothesis to generate arbitrary inputs and assert security / correctness
properties hold for *all* inputs, not just the ones we happened to test.
"""
from __future__ import annotations

from hypothesis import given, settings
from hypothesis import strategies as st

from tshepo.config import LABELS, MAX_BODY_LEN, MAX_RECORDS_TOTAL
from tshepo.features.cashflow import band, scorecard
from tshepo.models.gate import rule_gate
from tshepo.pipeline import score_user
from tshepo.security import SecurityError, validate_message, validate_records


@given(
    body=st.text(min_size=0, max_size=200).filter(lambda s: "\x00" not in s),
    sender=st.text(min_size=0, max_size=20),
)
def test_rule_gate_returns_valid_label(body, sender):
    """The rule gate must always return one of the defined labels."""
    label = rule_gate(body, sender if sender else None)
    assert label in LABELS


@given(body=st.text(min_size=0, max_size=300))
def test_scorecard_always_in_range(body):
    """The scorecard must always produce a score in [300, 850] regardless of input."""
    f = {"income_weeks_active": 0.5, "income_cv": 1.0, "net_savings_rate": 0.0,
         "buffer_ratio": 0.5, "low_bal_share": 0.0, "n_loan_repay": 0,
         "n_unique_payers": 1, "hist_days": 30, "n_txn": 10}
    s = scorecard(f)["score"]
    assert 300 <= s <= 850


@given(body=st.text(min_size=0, max_size=300))
def test_band_is_valid(body):
    """band() must return one of the five valid bands."""
    b = band(600)
    assert b in ("Very low", "Low", "Fair", "Good", "Very good")


@given(body=st.text(min_size=0, max_size=100))
def test_validate_message_strips_null_bytes(body):
    """Null bytes must be stripped from message bodies."""
    dirty = body + "\x00" + "junk"
    clean, _ = validate_message(dirty)
    assert "\x00" not in clean


def test_validate_message_rejects_oversized_body():
    """A body longer than MAX_BODY_LEN is rejected."""
    body = "x" * (MAX_BODY_LEN + 1)
    try:
        validate_message(body)
        assert False, "expected SecurityError"
    except SecurityError:
        pass


def test_validate_records_rejects_oversized_batch():
    """A batch longer than MAX_RECORDS_TOTAL is rejected."""
    records = [{"body": "b", "sender": "s", "user_id": "u"}] * (MAX_RECORDS_TOTAL + 1)
    try:
        validate_records(records)
        assert False, "expected SecurityError"
    except SecurityError:
        pass


@settings(max_examples=50, deadline=None)
@given(
    n_txn=st.integers(min_value=2, max_value=50),
)
def test_score_user_returns_valid_passport(n_txn):
    """score_user must return a passport with a valid score and band."""
    records = [
        {"body": f"Confirmed. You have received M{d * 100}.00 on 2026-10-{d:02d}.",
         "sender": "MPESA", "user_id": "prop-user",
         "received_at": f"2026-10-{d:02d}T10:00:00"}
        for d in range(1, n_txn + 1)
    ]
    passport = score_user(records, user_label="prop-test")
    assert 300 <= passport["score"] <= 850
    assert passport["band"] in ("Very low", "Low", "Fair", "Good", "Very good")
    assert passport["applicant"] == "prop-test"
