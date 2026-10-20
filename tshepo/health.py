"""Production health checks.

    from tshepo.health import check_all
    check_all()  # dict of component -> ok/error
"""
from __future__ import annotations

import time

from .features.cashflow import FEATURES, build_features, scorecard
from .logging import get_logger
from .models.scoring import ScorecardModel
from .parsing.parser import classify

logger = get_logger("tshepo.health")


def check_parser() -> dict:
    t0 = time.monotonic()
    try:
        res = classify("RK12AB34CD Confirmed. You have received M100.00 from THABO "
                       "58123456 on 19/10/26 at 2:15 PM. New M-PESA balance is M900.00.",
                       "MPESA")
        ok = res["label"] == "genuine_transaction"
        return {"ok": ok, "ms": round((time.monotonic() - t0) * 1000, 1),
                "detail": res["label"]}
    except Exception as e:  # noqa: BLE001
        return {"ok": False, "ms": round((time.monotonic() - t0) * 1000, 1),
                "error": str(e)}


def check_features() -> dict:
    t0 = time.monotonic()
    try:
        rows = [{"timestamp": "2026-10-01T10:00", "direction": "in", "amount": 1000.0,
                 "fee": 0.0, "balance": 1000.0, "category": "transfer", "operator": "mpesa"}]
        f = build_features(rows)
        ok = set(f) == set(FEATURES) and 300 <= scorecard(f)["score"] <= 850
        return {"ok": ok, "ms": round((time.monotonic() - t0) * 1000, 1),
                "detail": f"n_features={len(f)}"}
    except Exception as e:  # noqa: BLE001
        return {"ok": False, "ms": round((time.monotonic() - t0) * 1000, 1),
                "error": str(e)}


def check_scorecard() -> dict:
    t0 = time.monotonic()
    try:
        m = ScorecardModel()
        f = {k: 0.5 for k in FEATURES}
        s = m.score(f)
        ok = 300 <= s <= 850
        return {"ok": ok, "ms": round((time.monotonic() - t0) * 1000, 1),
                "detail": f"score={s}"}
    except Exception as e:  # noqa: BLE001
        return {"ok": False, "ms": round((time.monotonic() - t0) * 1000, 1),
                "error": str(e)}


def check_models_dir() -> dict:
    import os
    from .config import MODELS_DIR
    t0 = time.monotonic()
    try:
        files = [f for f in os.listdir(MODELS_DIR) if f.endswith(".joblib")]
        return {"ok": True, "ms": round((time.monotonic() - t0) * 1000, 1),
                "detail": f"{len(files)} joblib artefact(s)"}
    except Exception as e:  # noqa: BLE001
        return {"ok": False, "ms": round((time.monotonic() - t0) * 1000, 1),
                "error": str(e)}


def check_all() -> dict:
    results = {
        "parser": check_parser(),
        "features": check_features(),
        "scorecard": check_scorecard(),
        "models_dir": check_models_dir(),
    }
    results["ok"] = all(v["ok"] for v in results.values())
    if not results["ok"]:
        logger.warning("health check failed", extra={"results": results})
    else:
        logger.info("health check passed")
    return results
