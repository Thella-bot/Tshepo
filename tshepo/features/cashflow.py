"""
Cash-flow features, ledger integrity check, transparent scorecard and the shareable "credit passport".

Input: ledger rows for ONE user, as returned by tshepo.parsing.rules.build_ledger():
transactions (money moved), balance updates (balance restatements) and failed
transactions (the operator refused the transaction).

Design notes that matter more than they look:

* **One window.** Every rate-based feature is computed over the same 13-week
  window that ends at the user's most recent transaction. Mixing an all-time
  numerator with a 13-week denominator (the old `fee_burden`, `n_txn`) makes
  a long-history user look busier than a short-history one for reasons that
  have nothing to do with their finances.
* **The chain tolerates an unstated fee.** Operators rarely print the fee on
  the alert, but they always apply it. A link therefore holds when the balance
  moves by the amount plus a plausible fee, not only on an exact match. Links
  that only hold inside that tolerance are counted separately, and the exact
  share is reported as `strict_integrity` because that is the forgery signal.
* **The passport can refuse to score.** `status` and `decision` say whether
  the data supports a decision at all. A score computed on a truncated export
  or a broken chain is not a score, and saying so is the point.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Optional

import numpy as np

from ..config import get

FEATURES = [
    "hist_days", "n_txn", "monthly_income", "income_weeks_active", "income_cv", "out_in_ratio",
    "net_savings_rate", "bal_mean", "bal_min", "bal_cv", "low_bal_share", "buffer_ratio",
    "n_loan_repay", "airtime_share", "withdrawal_share", "merchant_share",
    "n_unique_payers", "payer_diversity", "top_payer_share", "integrity",
    # velocity & timing
    "txn_per_day", "amount_velocity", "night_txn_share", "weekend_txn_share",
    # regularity & fraud signals
    "fee_burden", "round_amount_share", "dow_regularity",
    # data quality
    "window_coverage", "strict_integrity", "fee_variance_share", "failed_txn_share",
    "n_txn_total", "n_balance_updates",
]
WEEKS = 13
WINDOW_DAYS = WEEKS * 7

# How far a balance may move beyond the stated amount before the link is
# treated as broken rather than as an unprinted operator fee.
FEE_TOL_RATE = 0.05
FEE_TOL_FLAT = 3.0
FEE_TOL_CAP = 15.0

STATUS_SCORED = "scored"
STATUS_REVIEW = "review"
STATUS_INSUFFICIENT = "insufficient_data"
DECISION_SCORED = "score"
DECISION_REVIEW = "refer_manual_review"
DECISION_INSUFFICIENT = "insufficient_data"


@dataclass
class Ledger:
    """Everything the parser made of one user's messages."""

    transactions: list[dict[str, Any]] = field(default_factory=list)
    balance_updates: list[dict[str, Any]] = field(default_factory=list)
    failed: list[dict[str, Any]] = field(default_factory=list)
    stats: dict[str, Any] = field(default_factory=dict)

    def all_rows(self) -> list[dict[str, Any]]:
        return self.transactions + self.balance_updates


def _ts(s):
    if s is None:
        return None
    s = str(s)
    try:
        return datetime.fromisoformat(s if "T" in s else s + "T00:00")
    except ValueError:
        return None


def _min_integrity() -> float:
    return float(get("MIN_INTEGRITY", 0.9))


def _min_window_coverage() -> float:
    return float(get("MIN_WINDOW_COVERAGE", 0.25))


def _fee_tolerance(amount) -> float:
    return min(FEE_TOL_RATE * abs(float(amount or 0)) + FEE_TOL_FLAT, FEE_TOL_CAP)


def _expected_balances(last: float, direction, amount, fee) -> list[float]:
    """Balances the operator could have reported, given an unstated fee."""
    a = abs(float(amount or 0))
    f = abs(float(fee or 0))
    if direction == "in":
        return [last + a, last + a + f, last + a - f]
    return [last - a, last - a - f, last - a + f]


def integrity_report(rows) -> dict[str, Any]:
    """Chain the balance observations of one account and report the result.

    A row is a transaction (money moved) or a balance update (the operator
    restated the balance). Transactions are linked to the previous balance on
    the same account; balance updates re-anchor it, because they are the
    operator telling us what the balance is, not that money moved.

    Returns a dict with the share of links that hold, the share that hold
    exactly, the share that only hold inside the fee tolerance, and a bounded
    list of the breaks.
    """
    ordered = sorted(
        [r for r in rows if r.get("timestamp") and r.get("balance") is not None],
        key=lambda r: (_ts(r["timestamp"]), 0 if r.get("kind") == "transaction" else 1),
    )
    last: dict[str, float] = {}
    checked = ok = strict = fee_var = reanchor_gaps = 0
    breaks: list[dict[str, Any]] = []
    largest_gap_hours = 0.0
    gap_at: Optional[str] = None
    prev_t = None
    for r in ordered:
        op = r.get("operator") or "?"
        bal = float(r["balance"])
        t_now = _ts(r["timestamp"])
        if prev_t is not None and t_now is not None:
            hours = (t_now - prev_t).total_seconds() / 3600.0
            if hours > largest_gap_hours:
                largest_gap_hours = hours
                gap_at = r.get("timestamp")
        if t_now is not None:
            prev_t = t_now
        if r.get("kind") == "balance_update":
            if op in last and abs(bal - last[op]) > 0.011:
                reanchor_gaps += 1
            last[op] = bal
            continue
        if op not in last:
            last[op] = bal
            continue
        checked += 1
        cands = _expected_balances(last[op], r.get("direction"), r.get("amount"), r.get("fee"))
        gap = min(abs(bal - c) for c in cands)
        if gap <= 0.011:
            ok += 1
            strict += 1
        elif gap <= _fee_tolerance(r.get("amount")):
            ok += 1
            fee_var += 1
        else:
            if len(breaks) < 10:
                breaks.append({
                    "operator": op,
                    "timestamp": r.get("timestamp"),
                    "expected": round(cands[0], 2),
                    "observed": round(bal, 2),
                    "gap": round(gap, 2),
                })
        last[op] = bal
    return {
        "links_checked": checked,
        "links_ok": ok,
        "integrity": (ok / checked) if checked else 1.0,
        "strict_integrity": (strict / checked) if checked else 1.0,
        "fee_variance_share": (fee_var / checked) if checked else 0.0,
        "reanchor_gaps": reanchor_gaps,
        "largest_gap_hours": round(largest_gap_hours, 2),
        "largest_gap_at": gap_at,
        "breaks": breaks,
    }


def ledger_integrity(rows):
    """Share of balance-chain links that hold. Kept for callers that want the pair."""
    report = integrity_report(rows)
    return report["integrity"], report["links_checked"]


def _row_kind(r) -> str:
    return r.get("kind") or "transaction"


def build_features(rows, ledger: Optional[Ledger] = None,
                   history_span_days: Optional[int] = None) -> dict[str, float]:
    """Cash-flow features for one user over a single 13-week window."""
    txns = list(rows) if ledger is None else list(ledger.transactions)
    updates = [] if ledger is None else list(ledger.balance_updates)
    f = dict.fromkeys(FEATURES, 0.0)

    dated = [r for r in txns if r.get("timestamp")]
    if not dated:
        f["integrity"] = 1.0
        f["strict_integrity"] = 1.0
        return f

    dated.sort(key=lambda r: _ts(r["timestamp"]))
    f["n_txn_total"] = float(len(dated))
    f["n_balance_updates"] = float(len(updates))
    span = (max((_ts(dated[-1]["timestamp"]) - _ts(dated[0]["timestamp"])).days, 1)
            if len(dated) >= 2 else 1)
    if history_span_days:
        span = max(span, int(history_span_days))
    f["hist_days"] = float(span)

    t1 = _ts(dated[-1]["timestamp"])
    start = t1 - timedelta(days=WINDOW_DAYS)
    window = [r for r in dated if _ts(r["timestamp"]) >= start]
    f["n_txn"] = float(len(window))
    if not window:
        f["integrity"] = 1.0
        f["strict_integrity"] = 1.0
        return f

    covered = (_ts(window[-1]["timestamp"]) - _ts(window[0]["timestamp"])).days
    f["window_coverage"] = float(min(1.0, max(covered, 0) / WINDOW_DAYS))

    wk = np.zeros(WEEKS)
    income = cash_in = outflow = 0.0
    payers: set[str] = set()
    n_inc = 0
    inc_amts: list[float] = []
    out_by = {"airtime": 0.0, "withdrawal": 0.0, "merchant": 0.0}
    n_loan = 0
    total_fees = 0.0
    for r in window:
        t = _ts(r["timestamp"])
        a = float(r.get("amount") or 0)
        fee = float(r.get("fee") or 0)
        total_fees += fee
        if r.get("direction") == "in":
            if r.get("category") == "deposit":
                cash_in += a
            else:
                income += a
                n_inc += 1
                inc_amts.append(a)
                wk[min(int((t - start).days // 7), WEEKS - 1)] += a
                payers.add(r.get("counterparty_phone") or r.get("counterparty_name") or f"anon-{n_inc}")
        else:
            out = a + fee
            outflow += out
            if r.get("category") in out_by:
                out_by[r["category"]] += out
            if r.get("category") == "loan_repayment":
                n_loan += 1

    f["monthly_income"] = income / (WINDOW_DAYS / 30)
    f["income_weeks_active"] = float((wk > 0).mean())
    f["income_cv"] = float(wk.std() / wk.mean()) if wk.mean() > 0 else 3.0
    f["out_in_ratio"] = outflow / (income + cash_in + 1.0)
    f["net_savings_rate"] = float(np.clip((income + cash_in - outflow) / (income + cash_in + 1.0), -1, 1))
    f["n_loan_repay"] = n_loan
    f["airtime_share"] = out_by["airtime"] / (outflow + 1.0)
    f["withdrawal_share"] = out_by["withdrawal"] / (outflow + 1.0)
    f["merchant_share"] = out_by["merchant"] / (outflow + 1.0)
    f["n_unique_payers"] = len(payers)
    f["payer_diversity"] = len(payers) / max(n_inc, 1)
    f["top_payer_share"] = max(inc_amts) / income if income > 0 else 1.0

    # -------- velocity & timing
    total_amount = sum(float(r.get("amount") or 0) for r in window)
    f["txn_per_day"] = f["n_txn"] / max(covered, 1)
    f["amount_velocity"] = total_amount / max(covered, 1)
    night = weekend = round_amt = 0
    dow_counts: dict[int, int] = {}
    for r in window:
        t = _ts(r["timestamp"])
        if t is None:
            continue
        dow_counts[t.weekday()] = dow_counts.get(t.weekday(), 0) + 1
        if t.hour >= 22 or t.hour < 5:
            night += 1
        if t.weekday() >= 5:
            weekend += 1
        amt = float(r.get("amount") or 0)
        if amt > 0 and (amt % 10 == 0 or amt % 100 == 0):
            round_amt += 1
    f["night_txn_share"] = night / f["n_txn"] if f["n_txn"] else 0.0
    f["weekend_txn_share"] = weekend / f["n_txn"] if f["n_txn"] else 0.0
    f["fee_burden"] = total_fees / (outflow + 1.0)
    f["round_amount_share"] = round_amt / f["n_txn"] if f["n_txn"] else 0.0
    if dow_counts:
        total = sum(dow_counts.values())
        probs = np.array(list(dow_counts.values())) / total
        entropy = float(-np.sum(probs * np.log(probs + 1e-9)))
        max_entropy = np.log(7)
        f["dow_regularity"] = 1.0 - (entropy / max_entropy) if max_entropy > 0 else 0.0

    # -------- balance behaviour, over the same window
    totals, running = [], {}
    for r in sorted(window, key=lambda r: _ts(r["timestamp"])):
        running[r.get("operator") or "?"] = float(r.get("balance") or 0)
        totals.append(sum(running.values()))
    totals = np.array(totals)
    f["bal_mean"], f["bal_min"] = float(totals.mean()), float(totals.min())
    f["bal_cv"] = float(totals.std() / totals.mean()) if totals.mean() > 0 else 3.0
    f["low_bal_share"] = float((totals < 20).mean())
    f["buffer_ratio"] = f["bal_mean"] / (outflow / (WINDOW_DAYS / 30) + 1.0)

    chain_rows = list(window) + [u for u in updates if u.get("timestamp")
                                  and _ts(u["timestamp"]) >= start]
    report = integrity_report(chain_rows)
    f["integrity"] = report["integrity"]
    f["strict_integrity"] = report["strict_integrity"]
    f["fee_variance_share"] = report["fee_variance_share"]
    if ledger is not None and ledger.failed:
        f["failed_txn_share"] = len(ledger.failed) / max(len(window) + len(ledger.failed), 1)
    return f


# ----------------------------------------------------------------- transparent scorecard
COMPONENTS = {  # name: (weight, human text when weak)
    "income_regularity": (25, "Income arrives irregularly or in few weeks"),
    "cash_surplus":      (20, "Spending is close to or above income"),
    "balance_buffer":    (20, "Balance is thin relative to monthly spending"),
    "repayment_habit":   (15, "No recent loan repayments visible"),
    "customer_base":     (10, "Few distinct people or businesses paying in"),
    "history_depth":     (10, "Short transaction history"),
}


def component_values(f):
    clip = lambda x: float(min(max(x, 0.0), 1.0))
    return {
        "income_regularity": clip(0.6 * f["income_weeks_active"] + 0.4 * (1 - min(f["income_cv"], 2) / 2)),
        "cash_surplus": clip((f["net_savings_rate"] + 0.2) / 0.6),
        "balance_buffer": clip(0.7 * min(f["buffer_ratio"], 1.0) + 0.3 * (1 - f["low_bal_share"])),
        "repayment_habit": clip(f["n_loan_repay"] / 4),
        "customer_base": clip(f["n_unique_payers"] / 10),
        "history_depth": clip(f["hist_days"] / 90),
    }


def scorecard(f):
    vals = component_values(f)
    pts = sum(COMPONENTS[k][0] * v for k, v in vals.items())  # 0..100
    score = int(round(300 + 5.5 * pts))
    gaps = sorted(((COMPONENTS[k][0] * (1 - v), k) for k, v in vals.items()), reverse=True)
    strengths = sorted(((COMPONENTS[k][0] * v, k) for k, v in vals.items()), reverse=True)
    return dict(score=score, components={k: round(v, 2) for k, v in vals.items()},
                reasons_down=[COMPONENTS[k][1] for g, k in gaps[:3] if g > 3],
                reasons_up=[k.replace("_", " ") for s, k in strengths[:2] if s > 10])


def band(score):
    return ("Very low" if score < 450 else "Low" if score < 550 else "Fair" if score < 650
            else "Good" if score < 750 else "Very good")


def assess(f, integrity_report_: Optional[dict[str, Any]] = None) -> dict[str, Any]:
    """Decide whether the data supports a lending decision.

    Returns the status, the decision, and the reasons.

    * ``insufficient_data`` - there is not enough to score on: no readable
      transactions, a history too short for the window, or messages dropped.
    * ``review`` - the data exists but does not hang together: the balance
      chain has breaks, so a human has to look before a decision is made.
    * ``scored`` - the data supports the score.
    """
    insufficient: list[str] = []
    review: list[str] = []
    integrity = float(f.get("integrity", 1.0))
    coverage = float(f.get("window_coverage", 0.0))
    n_txn = int(f.get("n_txn", 0))

    if n_txn == 0:
        insufficient.append("no transactions could be read from the messages supplied")
    if coverage < _min_window_coverage():
        insufficient.append(
            f"history covers only {coverage:.0%} of the {WINDOW_DAYS}-day window "
            f"(minimum {_min_window_coverage():.0%})")
    if integrity < _min_integrity():
        review.append(
            f"balance chain holds for only {integrity:.0%} of links "
            f"(minimum {_min_integrity():.0%}); the export is incomplete or inconsistent")
    if integrity_report_:
        if integrity_report_.get("reanchor_gaps"):
            review.append(
                f"{integrity_report_['reanchor_gaps']} balance restatement(s) disagree with the "
                f"running balance, so some messages are missing")
        gap_h = integrity_report_.get("largest_gap_hours") or 0
        if gap_h >= 24:
            review.append(
                f"the export has a {gap_h:.0f} hour gap (around {integrity_report_.get('largest_gap_at')}); "
                f"export the full message history to close it")

    if insufficient:
        status, decision = STATUS_INSUFFICIENT, DECISION_INSUFFICIENT
    elif review:
        status, decision = STATUS_REVIEW, DECISION_REVIEW
    else:
        status, decision = STATUS_SCORED, DECISION_SCORED
    return {"status": status, "decision": decision, "issues": insufficient + review}


def make_passport(rows, user_label="applicant", pd_model_score=None, ledger: Optional[Ledger] = None,
                 history_span_days: Optional[int] = None):
    """The consent-gated, shareable summary. Contains aggregates only: no raw messages, names or numbers."""
    f = build_features(rows, ledger=ledger, history_span_days=history_span_days)
    sc = scorecard(f)
    integrity = f["integrity"]
    chain = integrity_report((ledger.all_rows() if ledger is not None else list(rows)))
    quality = assess(f, chain)
    return {
        "applicant": user_label,
        "score": sc["score"], "band": band(sc["score"]),
        "model_pd_score": pd_model_score,
        "status": quality["status"],
        "decision": quality["decision"],
        "data_quality": {
            "window_coverage": round(f["window_coverage"], 3),
            "integrity_share": round(integrity, 3),
            "strict_integrity_share": round(f["strict_integrity"], 3),
            "links_checked": chain["links_checked"],
            "balance_chain_breaks": len(chain["breaks"]),
            "largest_gap_hours": chain.get("largest_gap_hours", 0.0),
            "issues": quality["issues"],
        },
        "history_days": int(f["hist_days"]), "transactions_used": int(f["n_txn"]),
        "monthly_income_estimate": round(f["monthly_income"], 2),
        "income_active_weeks": f"{round(f['income_weeks_active'] * WEEKS)}/{WEEKS}",
        "avg_balance": round(f["bal_mean"], 2), "loan_repayments_seen": int(f["n_loan_repay"]),
        "components": sc["components"],
        "reasons_down": sc["reasons_down"], "reasons_up": sc["reasons_up"],
        "data_integrity": "consistent" if integrity >= _min_integrity() else "REVIEW: balance chain has breaks",
        "integrity_share": round(integrity, 3),
        "notes": "Built from consented SMS alerts. Scorecard weights are provisional until calibrated on repayment outcomes.",
    }
