"""End-to-end pipeline: messages -> gate -> ledger -> features -> score.

This is the reusable core; the CLI scripts in tshepo.cli are thin wrappers around it.
It is the piece a production service would call per applicant:

    from tshepo.pipeline import score_user
    from tshepo.models.gate import load_gate
    from tshepo.resilience import fallback_gate

    gate = fallback_gate(load_gate("models/hybrid.joblib"))  # degrades to rule gate
    passport = score_user(messages, label_fn=gate, correlation_id_value="req-123")
"""
from __future__ import annotations

import time
from collections import defaultdict
from typing import Any, Optional, cast

from .features.cashflow import build_features, make_passport, scorecard
from .logging import correlation_id, get_logger, new_correlation_id
from .models.gate import GateFn, rule_gate
from .parsing.parser import extract_transactions
from .resilience import ModelError, fallback_gate, time_budget
from .security import RateLimiter, RateLimitExceeded, validate_records

logger = get_logger("pipeline")
_rate_limiter = RateLimiter()


def group_by_user(records: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    """records -> {user_id: [record, ...]}"""
    out: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in records:
        out[str(r.get("user_id"))].append(r)
    return out


def ledger_for(records: list[dict[str, Any]], label_fn: Optional[GateFn] = None) -> list[dict[str, Any]]:
    """The transactions the gate trusts, as ledger rows."""
    return cast(list[dict[str, Any]], extract_transactions(records, label_fn=label_fn or rule_gate))


def features_for(records: list[dict[str, Any]], label_fn: Optional[GateFn] = None) -> dict[str, float]:
    """Cash-flow features for ONE user from their message list."""
    return cast(dict[str, float], build_features(ledger_for(records, label_fn=label_fn)))


def score_user(
    records: list[dict[str, Any]],
    label_fn: Optional[GateFn] = None,
    user_label: str = "applicant",
    pd_model: Optional[Any] = None,
    rate_limit_key: Optional[str] = None,
    correlation_id_value: Optional[str] = None,
) -> dict[str, Any]:
    """Full passport for one user: gate -> ledger -> features -> score.

    pd_model: optional fitted ScoreModel; when given its probability of default is
    reported alongside the transparent score.
    rate_limit_key: optional key for per-user rate limiting.
    correlation_id_value: request-scoped correlation id for tracing.
    """
    cid = correlation_id_value or new_correlation_id()
    token = correlation_id.set(cid)
    try:
        if rate_limit_key is not None and not _rate_limiter.allow(rate_limit_key):
            raise RateLimitExceeded(f"rate limit exceeded for {rate_limit_key}")

        # graceful degradation: ensure the gate never fails the whole request
        gate = label_fn or rule_gate
        if label_fn is not None:
            gate = fallback_gate(label_fn)

        with time_budget(5.0, "score_user"):
            t0 = time.monotonic()
            records = validate_records(records)
            rows = ledger_for(records, label_fn=gate)
            f = build_features(rows)
            sc = scorecard(f)
            pd_score: Optional[float] = None
            if pd_model is not None:
                pd_score = round(float(850 - 550 * pd_model.predict_proba(f)), 1)
            passport = cast(dict[str, Any], make_passport(rows, user_label=user_label, pd_model_score=pd_score))
            passport["score"] = sc["score"]
            dur_ms = (time.monotonic() - t0) * 1000
            logger.info(
                "scored user",
                extra={"user_id": user_label, "score": sc["score"], "duration_ms": round(dur_ms, 1), "correlation_id": cid},
            )
            return passport
    except RateLimitExceeded:
        raise
    except ModelError:
        raise
    except Exception as e:  # noqa: BLE001 - surface as a Tshepo error for the API layer
        logger.exception("score_user failed", extra={"user_id": user_label, "correlation_id": cid})
        raise ModelError(f"scoring failed for {user_label}: {e}") from e
    finally:
        correlation_id.reset(token)


def score_batch(
    records: list[dict[str, Any]],
    label_fn: Optional[GateFn] = None,
    pd_model: Optional[Any] = None,
    chunk_size: int = 500,
    rate_limit_key: Optional[str] = None,
    correlation_id_value: Optional[str] = None,
) -> dict[str, dict[str, Any]]:
    """Score many users in bounded chunks to cap memory use.

    Returns {user_id: passport}. Records are validated and grouped by user;
    each user's ledger is capped at MAX_RECORDS_PER_USER by the security layer.
    """
    cid = correlation_id_value or new_correlation_id()
    token = correlation_id.set(cid)
    try:
        records = validate_records(records)
        by_user = group_by_user(records)
        out: dict[str, dict[str, Any]] = {}
        processed = 0
        with time_budget(30.0, "score_batch"):
            for uid, recs in by_user.items():
                if rate_limit_key is not None and not _rate_limiter.allow(f"{rate_limit_key}:{uid}"):
                    logger.warning("rate limited, skipping user", extra={"user_id": uid, "correlation_id": cid})
                    continue
                recs = recs[:5000]
                out[uid] = score_user(
                    recs,
                    label_fn=label_fn,
                    user_label=uid,
                    pd_model=pd_model,
                    correlation_id_value=cid,
                )
                processed += 1
                if processed % chunk_size == 0:
                    logger.info(
                        "batch progress",
                        extra={"processed": processed, "total": len(by_user), "correlation_id": cid},
                    )
        return out
    finally:
        correlation_id.reset(token)


def score_dataset(
    records: list[dict[str, Any]],
    label_fn: Optional[GateFn] = None,
    pd_model: Optional[Any] = None,
    correlation_id_value: Optional[str] = None,
) -> dict[str, dict[str, Any]]:
    """Score every user in a record set. Returns {user_id: passport}."""
    return score_batch(
        records,
        label_fn=label_fn,
        pd_model=pd_model,
        correlation_id_value=correlation_id_value,
    )
