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
from .graph import (CIRCULAR_SHARE_ALERT, INCOME_CONCENTRATION_ALERT,
                    graph_signals)

FEATURES = [
    "hist_days", "n_txn", "monthly_income", "income_weeks_active", "income_weeks_n", "income_cv",
    "out_in_ratio",
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
    # anti-fraud: corroboration of the repayment habit, and the shape of the
    # counterparty graph (money that comes back out, and how lopsided the
    # wallet is). These are features, not decoration: the fitted model reads
    # them too.
    "repay_months", "repay_variety", "circular_share", "both_ways",
    "income_concentration", "top_payer_share",
]
WEEKS = 13
WINDOW_DAYS = WEEKS * 7


def _window() -> tuple[int, int]:
    """The scoring window as (weeks, days), read from the configuration.

    The window is a product decision: it decides what a year of income
    regularity means, how long a "history" is, and what a score is allowed to
    claim. It is configurable so it can be changed deliberately and
    re-validated, not so that it drifts.
    """
    days = int(get("SCORE_WINDOW_DAYS", WINDOW_DAYS))
    return max(1, days // 7), max(1, days)

# How far a balance may move beyond the stated amount before the link is
# treated as broken rather than as an unprinted operator fee.
FEE_TOL_RATE = 0.05
FEE_TOL_FLAT = 3.0
FEE_TOL_CAP = 15.0

STATUS_SCORED = "scored"
STATUS_REVIEW = "review"
STATUS_INSUFFICIENT = "insufficient_data"
STATUS_REJECTED = "rejected"
DECISION_SCORED = "score"
DECISION_REVIEW = "refer_manual_review"
DECISION_INSUFFICIENT = "insufficient_data"
DECISION_REJECT = "reject"

#: How many transactions a window needs before the share of incoming
#: transactions says something about the wallet rather than about its size.
CONCENTRATION_MIN_TXNS = 20
#: The same guard for the circular-flow signal. Two payments to and from the
#: same person is a favour, not a laundering scheme.
CIRCULAR_MIN_TXNS = 10
#: The evidence a high-confidence score needs, over and above the minimum.
CONFIDENCE_HIGH_TXNS = 60
CONFIDENCE_HIGH_DAYS = 90
CONFIDENCE_HIGH_WEEKS = 12


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


def _min_readable_share() -> float:
    """How much of the supplied history has to be readable to score on it."""
    return float(get("MIN_READABLE_SHARE", 0.2))


def _min_transactions() -> int:
    return int(get("MIN_TRANSACTIONS", 25))


def _min_history_days() -> int:
    return int(get("MIN_HISTORY_DAYS", 60))


def _min_income_weeks() -> int:
    return int(get("MIN_INCOME_WEEKS", 8))


def _max_unread_share() -> float:
    return float(get("MAX_UNREAD_SHARE", 0.4))


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
    """Cash-flow features for one user over a single window."""
    weeks, window_days = _window()
    txns = list(rows) if ledger is None else list(ledger.transactions)
    updates = [] if ledger is None else list(ledger.balance_updates)
    f = dict.fromkeys(FEATURES, 0.0)

    dated = [(r, _ts(r["timestamp"])) for r in txns if r.get("timestamp")]
    # A timestamp the platform cannot read is not a timestamp: an export with
    # an impossible date ("2026-10-32") must not break the sort.
    dated = [(r, t) for r, t in dated if t is not None]
    if not dated:
        f["integrity"] = 1.0
        f["strict_integrity"] = 1.0
        return f

    dated.sort(key=lambda rt: rt[1])
    f["n_txn_total"] = float(len(dated))
    f["n_balance_updates"] = float(len(updates))
    span = max((dated[-1][1] - dated[0][1]).days, 1) if len(dated) >= 2 else 1
    if history_span_days:
        span = max(span, int(history_span_days))
    f["hist_days"] = float(span)

    t1 = dated[-1][1]
    start = t1 - timedelta(days=window_days)
    window = [(r, t) for r, t in dated if t >= start]
    f["n_txn"] = float(len(window))
    if not window:
        f["integrity"] = 1.0
        f["strict_integrity"] = 1.0
        return f

    covered = (window[-1][1] - window[0][1]).days
    f["window_coverage"] = float(min(1.0, max(covered, 0) / window_days))

    wk = np.zeros(weeks)
    income = cash_in = outflow = 0.0
    payers: set[str] = set()
    n_inc = 0
    inc_amts: list[float] = []
    out_by = {"airtime": 0.0, "withdrawal": 0.0, "merchant": 0.0}
    n_loan = 0
    loan_months: set[str] = set()
    loan_amounts: set[float] = set()
    total_fees = 0.0
    for r, t in window:
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
                wk[min(int((t - start).days // 7), weeks - 1)] += a
                payers.add(r.get("counterparty_phone") or r.get("counterparty_name") or f"anon-{n_inc}")
        else:
            out = a + fee
            outflow += out
            if r.get("category") in out_by:
                out_by[r.get("category")] += out
            if r.get("category") == "loan_repayment":
                n_loan += 1
                loan_amounts.add(round(a, 2))
                if r.get("timestamp"):
                    loan_months.add(str(r["timestamp"])[:7])

    f["monthly_income"] = income / (window_days / 30)
    f["income_weeks_active"] = float((wk > 0).mean())
    f["income_weeks_n"] = float((wk > 0).sum())
    f["income_cv"] = float(wk.std() / wk.mean()) if wk.mean() > 0 else 3.0
    f["out_in_ratio"] = outflow / (income + cash_in + 1.0)
    f["net_savings_rate"] = float(np.clip((income + cash_in - outflow) / (income + cash_in + 1.0), -1, 1))
    f["n_loan_repay"] = n_loan
    # A borrower repays month after month, and the amount moves. Four identical
    # "repayments" in one window are a line in a script, not a habit, so the
    # scorecard only pays for the corroboration.
    f["repay_months"] = float(len(loan_months))
    f["repay_variety"] = float(len(loan_amounts))
    f["airtime_share"] = out_by["airtime"] / (outflow + 1.0)
    f["withdrawal_share"] = out_by["withdrawal"] / (outflow + 1.0)
    f["merchant_share"] = out_by["merchant"] / (outflow + 1.0)
    f["n_unique_payers"] = len(payers)
    f["payer_diversity"] = len(payers) / max(n_inc, 1)
    f["top_payer_share"] = max(inc_amts) / income if income > 0 else 1.0

    # -------- velocity & timing
    total_amount = sum(float(r.get("amount") or 0) for r, _ in window)
    f["txn_per_day"] = f["n_txn"] / max(covered, 1)
    f["amount_velocity"] = total_amount / max(covered, 1)
    night = weekend = round_amt = 0
    dow_counts: dict[int, int] = {}
    for r, t in window:
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
    for r, _t in window:
        running[r.get("operator") or "?"] = float(r.get("balance") or 0)
        totals.append(sum(running.values()))
    totals = np.array(totals) if totals else np.array([0.0])
    f["bal_mean"], f["bal_min"] = float(totals.mean()), float(totals.min())
    f["bal_cv"] = float(totals.std() / totals.mean()) if totals.mean() > 0 else 3.0
    f["low_bal_share"] = float((totals < 20).mean())
    f["buffer_ratio"] = f["bal_mean"] / (outflow / (window_days / 30) + 1.0)

    chain_rows = [r for r, _ in window] + [u for u in updates if u.get("timestamp")
                                           and _ts(u["timestamp"]) >= start]
    report = integrity_report(chain_rows)
    f["integrity"] = report["integrity"]
    f["strict_integrity"] = report["strict_integrity"]
    f["fee_variance_share"] = report["fee_variance_share"]
    if ledger is not None and ledger.failed:
        f["failed_txn_share"] = len(ledger.failed) / max(len(window) + len(ledger.failed), 1)

    # -------- who the money came from, and where it went
    graph = graph_signals([r for r, _ in window])
    f["circular_share"] = graph["circular_share"]
    f["income_concentration"] = graph["income_concentration"]
    f["top_payer_share"] = graph["top_payer_share"]
    f["both_ways"] = float(graph["both_ways"])
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
    # A repayment only counts as a habit when it recurs (several months) and
    # the amount moves. A single line of "repayment" messages, or four of them
    # with the same amount, is a claim, not a habit.
    repayment = min(f["n_loan_repay"] / 4.0,
                    f.get("repay_months", 0.0) / 3.0,
                    f.get("repay_variety", 0.0) / 2.0)
    _weeks, window_days = _window()
    return {
        "income_regularity": clip(0.6 * f["income_weeks_active"] + 0.4 * (1 - min(f["income_cv"], 2) / 2)),
        "cash_surplus": clip((f["net_savings_rate"] + 0.2) / 0.6),
        "balance_buffer": clip(0.7 * min(f["buffer_ratio"], 1.0) + 0.3 * (1 - f["low_bal_share"])),
        "repayment_habit": clip(repayment),
        "customer_base": clip(f["n_unique_payers"] / 10),
        "history_depth": clip(f["hist_days"] / window_days),
    }


#: What each component needs before it can say anything about the applicant.
#: Below the line the component is *withheld*, not scored zero: "not enough
#: evidence" must not look like "this applicant failed", and a withheld
#: component must not drag the score down for something nobody measured.
COMPONENT_EVIDENCE = {
    "income_regularity": ("income-bearing weeks", 4),
    "cash_surplus":      ("transactions", 10),
    "balance_buffer":    ("transactions", 10),
    "repayment_habit":   ("days of history", 60),
    "customer_base":     ("transactions", 10),
    "history_depth":     ("days of history", 60),
}


def component_evidence(f) -> dict[str, tuple[float, float]]:
    """component -> (observed, required) for the evidence it needs."""
    weeks = float(f.get("income_weeks_n", 0.0) or 0.0)
    txns = float(f.get("n_txn", 0.0) or 0.0)
    days = float(f.get("hist_days", 0.0) or 0.0)
    return {
        "income_regularity": (weeks, COMPONENT_EVIDENCE["income_regularity"][1]),
        "cash_surplus":      (txns, COMPONENT_EVIDENCE["cash_surplus"][1]),
        "balance_buffer":    (txns, COMPONENT_EVIDENCE["balance_buffer"][1]),
        "repayment_habit":   (days, COMPONENT_EVIDENCE["repayment_habit"][1]),
        "customer_base":     (txns, COMPONENT_EVIDENCE["customer_base"][1]),
        "history_depth":     (days, COMPONENT_EVIDENCE["history_depth"][1]),
    }


def withheld_components(f) -> dict[str, str]:
    """Components that cannot be scored from this history, and why."""
    out: dict[str, str] = {}
    for name, (observed, required) in component_evidence(f).items():
        if observed < required:
            label, need = COMPONENT_EVIDENCE[name]
            out[name] = f"not enough evidence: {int(observed)} {label}, {int(need)} needed"
    return out


def scorecard(f):
    """The transparent score.

    Components with enough evidence are scored and weighted as usual. A
    component without enough evidence is withheld and its weight is left out
    of the denominator rather than counted as a zero, so a partly measured
    applicant is not punished for the part nobody could see. When every
    component is scored - the normal case for a file that passed the evidence
    rules - this is exactly the old formula.
    """
    vals = component_values(f)
    withheld = withheld_components(f)
    scored = {k: v for k, v in vals.items() if k not in withheld}
    pts_earned = sum(COMPONENTS[k][0] * v for k, v in scored.items())
    pts_available = sum(COMPONENTS[k][0] for k in scored)
    score = int(round(300 + 550 * (pts_earned / pts_available))) if pts_available else None
    gaps = sorted(((COMPONENTS[k][0] * (1 - v), k) for k, v in scored.items()), reverse=True)
    strengths = sorted(((COMPONENTS[k][0] * v, k) for k, v in scored.items()), reverse=True)
    return dict(
        score=score,
        components={k: (round(v, 2) if k not in withheld else None) for k, v in vals.items()},
        withheld=withheld,
        provisional=bool(withheld),
        reasons_down=[COMPONENTS[k][1] for g, k in gaps[:3] if g > 3],
        reasons_up=[k.replace("_", " ") for s, k in strengths[:2] if s > 10],
    )


def band(score):
    return ("Very low" if score < 450 else "Low" if score < 550 else "Fair" if score < 650
            else "Good" if score < 750 else "Very good")


def evidence_summary(f, ledger: Optional[Ledger] = None) -> dict[str, Any]:
    """What the score is standing on: how much history, how readable it is."""
    stats = ledger.stats if ledger is not None else {}
    messages = int(stats.get("messages", 0) or 0)
    unread = int(stats.get("unread", 0) or 0)
    return {
        "transactions": int(f.get("n_txn", 0) or 0),
        "history_days": int(f.get("hist_days", 0) or 0),
        "income_weeks": int(f.get("income_weeks_n", 0) or 0),
        "window_coverage": round(float(f.get("window_coverage", 0.0) or 0.0), 3),
        "messages_supplied": messages,
        "messages_unread": unread,
        "readable_share": round((messages - unread) / messages, 3) if messages else None,
    }


def confidence_of(f) -> Optional[str]:
    """How much the evidence behind a score is worth.

    ``high`` - comfortably above the evidence rules; ``medium`` - the minimum
    a score may be issued on; ``low`` - something was scored on thin data and
    a human should look; ``None`` - there is nothing to be confident about.
    """
    txns = int(f.get("n_txn", 0) or 0)
    days = int(f.get("hist_days", 0) or 0)
    weeks = int(f.get("income_weeks_n", 0) or 0)
    if txns == 0 and days == 0:
        return None
    if txns >= CONFIDENCE_HIGH_TXNS and days >= CONFIDENCE_HIGH_DAYS \
            and weeks >= CONFIDENCE_HIGH_WEEKS:
        return "high"
    if txns >= _min_transactions() and days >= _min_history_days() \
            and weeks >= _min_income_weeks():
        return "medium"
    return "low"


def _missing_evidence(f, ledger: Optional[Ledger] = None) -> tuple[list[dict[str, Any]], int]:
    """What is missing before a score can be issued, and how long to wait.

    The second element is the number of days the applicant would have to wait
    for their own history to grow into a score, estimated from the rate their
    own messages arrive at.
    """
    _weeks, window_days = _window()
    n_txn = int(f.get("n_txn", 0) or 0)
    days = int(f.get("hist_days", 0) or 0)
    income_weeks = int(f.get("income_weeks_n", 0) or 0)
    coverage = float(f.get("window_coverage", 0.0) or 0.0)
    missing: list[dict[str, Any]] = []
    waits: list[int] = []

    if n_txn < _min_transactions():
        shortfall = _min_transactions() - n_txn
        missing.append({"code": "transactions",
                        "message": f"{n_txn} transaction(s) in the last {window_days} days, "
                                   f"{_min_transactions()} are needed for a score",
                        "shortfall": shortfall})
        if n_txn > 0 and days > 0:
            waits.append(int(-(-shortfall * days // n_txn)))  # ceil
    if days < _min_history_days():
        shortfall = _min_history_days() - days
        missing.append({"code": "history_days",
                        "message": f"{days} day(s) of history, {_min_history_days()} are needed "
                                   f"for a score",
                        "shortfall": shortfall})
        waits.append(shortfall)
    if income_weeks < _min_income_weeks():
        shortfall = _min_income_weeks() - income_weeks
        missing.append({"code": "income_weeks",
                        "message": f"money came in during {income_weeks} week(s), "
                                   f"{_min_income_weeks()} are needed to judge how regular it is",
                        "shortfall": shortfall})
        waits.append(shortfall * 7)
    if coverage < _min_window_coverage():
        missing.append({"code": "window_coverage",
                        "message": f"the history covers {coverage:.0%} of the {window_days}-day "
                                   f"window, {_min_window_coverage():.0%} is needed",
                        "shortfall": 0})
    return missing, (max(waits) if waits else 0)


def assess(f, integrity_report_: Optional[dict[str, Any]] = None,
           ledger: Optional[Ledger] = None) -> dict[str, Any]:
    """Decide whether the data supports a lending decision.

    Four states, and the difference between them matters to the applicant:

    * ``rejected`` - the export cannot be used: most of it could not be read.
      The applicant has to export again; waiting will not help.
    * ``insufficient_data`` - the export is readable but too thin to score on.
      The applicant is told what is missing and when to come back.
    * ``review`` - there is enough to score on, but something does not hang
      together, so a human has to look before a decision is made.
    * ``scored`` - the data supports the score.

    A score is only ever issued on the last two, and the decision is always
    reported with the evidence behind it.
    """
    rejected: list[str] = []
    insufficient: list[str] = []
    review: list[str] = []
    integrity = float(f.get("integrity", 1.0))
    coverage = float(f.get("window_coverage", 0.0))
    n_txn = int(f.get("n_txn", 0))
    _weeks, window_days = _window()

    unread = int(ledger.stats.get("unread", 0) or 0) if ledger is not None else 0
    messages = int(ledger.stats.get("messages", 0) or 0) if ledger is not None else 0
    unread_share = unread / max(messages, 1)

    # ---- 1. the export cannot be used at all
    if messages and unread_share >= _max_unread_share():
        rejected.append(
            f"{unread} of {messages} message(s) could not be read ({unread_share:.0%}); the export "
            f"has to be exported again, from the phone's full message history")
    if n_txn == 0 and messages:
        insufficient.append("no transactions could be read from the messages supplied")

    # ---- 2. readable, but too thin to score on
    missing, wait_days = _missing_evidence(f, ledger)
    for item in missing:
        insufficient.append(item["message"])
    if unread and unread_share < _max_unread_share():
        if unread_share >= _min_readable_share():
            insufficient.append(
                f"{unread} of {messages} message(s) could not be read ({unread_share:.0%}); "
                f"the history is incomplete")
        else:
            review.append(f"{unread} of {messages} message(s) could not be read")

    # ---- 3. enough to score on, but it does not hang together
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
    if ledger is not None:
        unchained = int(ledger.stats.get("unchained", 0) or 0)
        if unchained:
            review.append(
                f"{unchained} transaction(s) carried no balance statement and could not be "
                f"checked against the balance chain")
        mismatch = int(ledger.stats.get("timestamp_mismatch", 0) or 0)
        if mismatch:
            review.append(
                f"{mismatch} message(s) carry a timestamp that disagrees with the export's own "
                f"timestamp; the message and the file it came from were not produced together")

    # Signals that do not need a second source to be worth a human's time.
    circular = float(f.get("circular_share", 0.0) or 0.0)
    if circular > CIRCULAR_SHARE_ALERT and n_txn >= CIRCULAR_MIN_TXNS:
        review.append(
            f"{circular:.0%} of the money received came from accounts that also took money out; "
            f"the money may be circulating rather than earned")
    concentration = float(f.get("income_concentration", 0.0) or 0.0)
    # A small file of incoming payments only is a thin file, not a fraud
    # signal: the ratio is only meaningful once there is enough of a history
    # for the ratio to mean something.
    if concentration > INCOME_CONCENTRATION_ALERT and n_txn >= CONCENTRATION_MIN_TXNS:
        review.append(
            f"{concentration:.0%} of the transactions are incoming; a wallet that only receives "
            f"is not a wallet that earns")
    # The top-payer share is reported, not used to stop a score: a wallet paid
    # by one employer is normal, and refusing to score it would exclude a
    # large part of the people this service exists for.

    if rejected:
        status, decision = STATUS_REJECTED, DECISION_REJECT
    elif insufficient:
        status, decision = STATUS_INSUFFICIENT, DECISION_INSUFFICIENT
    elif review:
        status, decision = STATUS_REVIEW, DECISION_REVIEW
    else:
        status, decision = STATUS_SCORED, DECISION_SCORED

    retry = None
    if status == STATUS_INSUFFICIENT and missing:
        retry = {
            "kind": "wait",
            "after_days": wait_days,
            "message": (f"a score needs {_min_transactions()} transactions, "
                        f"{_min_history_days()} days of history and {_min_income_weeks()} weeks "
                        f"with money in; this history is about {wait_days} day(s) short of that"),
        }
    elif status == STATUS_REJECTED:
        retry = {
            "kind": "reexport",
            "after_days": 0,
            "message": "export the full message history from the phone again; most of the last "
                       "export could not be read",
        }

    return {
        "status": status,
        "decision": decision,
        "issues": rejected + insufficient + review,
        "rejected": rejected,
        "missing": missing,
        "retry": retry,
        "evidence": evidence_summary(f, ledger),
        "confidence": confidence_of(f),
    }


def make_passport(rows, user_label="applicant", pd_model_score=None, ledger: Optional[Ledger] = None,
                 history_span_days: Optional[int] = None, features: Optional[dict[str, float]] = None):
    """The consent-gated, shareable summary. Contains aggregates only: no raw messages, names or numbers."""
    f = features if features is not None else build_features(
        rows, ledger=ledger, history_span_days=history_span_days)
    sc = scorecard(f)
    integrity = f["integrity"]
    chain = integrity_report((ledger.all_rows() if ledger is not None else list(rows)))
    quality = assess(f, chain, ledger=ledger)
    stats = ledger.stats if ledger is not None else {}
    messages = int(stats.get("messages", 0) or 0)
    unread = int(stats.get("unread", 0) or 0)
    readable_share = round((messages - unread) / messages, 3) if messages else None
    fraud_signals = {
        "circular_share": round(float(f.get("circular_share", 0.0) or 0.0), 3),
        "both_ways_counterparties": int(f.get("both_ways", 0) or 0),
        "income_concentration": round(float(f.get("income_concentration", 0.0) or 0.0), 3),
        "top_payer_share": round(float(f.get("top_payer_share", 0.0) or 0.0), 3),
        "repayments_seen": int(f.get("n_loan_repay", 0) or 0),
        "repayment_months": int(f.get("repay_months", 0) or 0),
        "timestamp_mismatches": int(stats.get("timestamp_mismatch", 0) or 0),
    }
    score = sc["score"]
    # A score that was computed on evidence that does not meet the rules is
    # never presented as a decision. The number is still there for a human
    # reading the record, and the API withholds it from the applicant.
    if quality["status"] in (STATUS_INSUFFICIENT, STATUS_REJECTED):
        score = None
    weeks_total, _days = _window()
    return {
        "applicant": user_label,
        "score": score, "band": band(score) if score is not None else None,
        "model_pd_score": pd_model_score,
        "status": quality["status"],
        "decision": quality["decision"],
        # How much the evidence behind the score is worth, and what it is
        # standing on. A lender can set their own policy on top of this.
        "score_confidence": quality["confidence"],
        "score_provisional": sc["provisional"],
        "evidence": quality["evidence"],
        "missing": quality["missing"],
        "retry": quality["retry"],
        # What this score is worth depends on where the messages came from.
        # Every score here rests on SMS the applicant supplied, and nothing in
        # the pipeline can prove those messages came from the operator.
        "verification": "sms_unverified",
        "fraud_signals": fraud_signals,
        "data_quality": {
            "window_coverage": round(f["window_coverage"], 3),
            "integrity_share": round(integrity, 3),
            "strict_integrity_share": round(f["strict_integrity"], 3),
            "links_checked": chain["links_checked"],
            "balance_chain_breaks": len(chain["breaks"]),
            "largest_gap_hours": chain.get("largest_gap_hours", 0.0),
            "messages_read": messages - unread if messages else None,
            "messages_unread": unread if messages else None,
            "readable_share": readable_share,
            "issues": quality["issues"],
        },
        "history_days": int(f["hist_days"]), "transactions_used": int(f["n_txn"]),
        "monthly_income_estimate": round(f["monthly_income"], 2),
        "income_active_weeks": f"{int(f['income_weeks_n'])}/{weeks_total}",
        "avg_balance": round(f["bal_mean"], 2), "loan_repayments_seen": int(f["n_loan_repay"]),
        "components": sc["components"],
        "withheld_components": sc["withheld"],
        "reasons_down": sc["reasons_down"], "reasons_up": sc["reasons_up"],
        "data_integrity": "consistent" if integrity >= _min_integrity() else "REVIEW: balance chain has breaks",
        "integrity_share": round(integrity, 3),
        "notes": "Built from consented SMS alerts. Scorecard weights are provisional until calibrated on repayment outcomes.",
    }
