"""End-to-end pipeline: messages -> gate -> ledger -> features -> score.

This is the reusable core; the CLI scripts in tshepo.cli are thin wrappers around it.
It is the piece a production service would call per applicant:

    from tshepo.pipeline import score_user
    passport = score_user(messages, label_fn=load_gate("models/hybrid.joblib"))
"""
from __future__ import annotations

from collections import defaultdict

from .features.cashflow import build_features, make_passport, scorecard
from .models.gate import rule_gate
from .parsing.parser import extract_transactions


def group_by_user(records):
    """records -> {user_id: [record, ...]}"""
    out = defaultdict(list)
    for r in records:
        out[r.get("user_id")].append(r)
    return out


def ledger_for(records, label_fn=None):
    """The transactions the gate trusts, as ledger rows."""
    return extract_transactions(records, label_fn=label_fn or rule_gate)


def features_for(records, label_fn=None):
    """Cash-flow features for ONE user from their message list."""
    return build_features(ledger_for(records, label_fn=label_fn))


def score_user(records, label_fn=None, user_label="applicant", pd_model=None):
    """Full passport for one user: gate -> ledger -> features -> scorecard.

    pd_model: optional fitted ScoreModel; when given its probability of default is
    reported alongside the transparent score.
    """
    rows = ledger_for(records, label_fn=label_fn)
    f = build_features(rows)
    sc = scorecard(f)
    pd_score = None
    if pd_model is not None:
        pd_score = round(float(850 - 550 * pd_model.predict_proba(f)), 1)
    passport = make_passport(rows, user_label=user_label, pd_model_score=pd_score)
    passport["score"] = sc["score"]
    return passport


def score_dataset(records, label_fn=None, pd_model=None):
    """Score every user in a record set. Returns {user_id: passport}."""
    out = {}
    for uid, recs in group_by_user(records).items():
        out[uid] = score_user(recs, label_fn=label_fn, user_label=uid, pd_model=pd_model)
    return out
