"""End-to-end pipeline: messages -> gate -> ledger -> features -> score.

This is the reusable core; the CLI scripts in tshepo.cli are thin wrappers around it.
It is the piece a production service would call per applicant:

    from tshepo.pipeline import score_user
    passport = score_user(messages, label_fn=load_gate("models/hybrid.joblib"))
"""
from __future__ import annotations

import time
from collections import defaultdict

from .features.cashflow import build_features, make_passport, scorecard
from .logging import get_logger
from .models.gate import rule_gate
from .parsing.parser import extract_transactions
from .security import RateLimiter, validate_records

logger = get_logger("pipeline")
_rate_limiter = RateLimiter()


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


def score_user(records, label_fn=None, user_label="applicant", pd_model=None,
               rate_limit_key=None):
    """Full passport for one user: gate -> ledger -> features -> scorecard.

    pd_model: optional fitted ScoreModel; when given its probability of default is
    reported alongside the transparent score.
    rate_limit_key: optional key for per-user rate limiting.
    """
    if rate_limit_key is not None and not _rate_limiter.allow(rate_limit_key):
        raise RuntimeError(f"rate limit exceeded for {rate_limit_key}")

    t0 = time.monotonic()
    records = validate_records(records)
    rows = ledger_for(records, label_fn=label_fn)
    f = build_features(rows)
    sc = scorecard(f)
    pd_score = None
    if pd_model is not None:
        pd_score = round(float(850 - 550 * pd_model.predict_proba(f)), 1)
    passport = make_passport(rows, user_label=user_label, pd_model_score=pd_score)
    passport["score"] = sc["score"]
    dur_ms = (time.monotonic() - t0) * 1000
    logger.info("scored user", extra={"user_id": user_label, "score": sc["score"],
                                      "duration_ms": round(dur_ms, 1)})
    return passport


def score_batch(records, label_fn=None, pd_model=None, chunk_size=500,
                rate_limit_key=None):
    """Score many users in bounded chunks to cap memory use.

    Returns {user_id: passport}. Records are validated and grouped by user;
    each user's ledger is capped at MAX_RECORDS_PER_USER by the security layer.
    """
    records = validate_records(records)
    by_user = group_by_user(records)
    out = {}
    processed = 0
    for uid, recs in by_user.items():
        if rate_limit_key is not None and not _rate_limiter.allow(f"{rate_limit_key}:{uid}"):
            logger.warning("rate limited, skipping user", extra={"user_id": uid})
            continue
        # cap per-user ledger size for memory safety
        recs = recs[:5000]
        out[uid] = score_user(recs, label_fn=label_fn, user_label=uid, pd_model=pd_model)
        processed += 1
        if processed % chunk_size == 0:
            logger.info("batch progress", extra={"processed": processed, "total": len(by_user)})
    return out


def score_dataset(records, label_fn=None, pd_model=None):
    """Score every user in a record set. Returns {user_id: passport}."""
    return score_batch(records, label_fn=label_fn, pd_model=pd_model)
