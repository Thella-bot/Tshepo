"""
Cash-flow features, ledger integrity check, transparent scorecard and the shareable "credit passport".

Input: ledger rows for ONE user, as returned by sms_parser.extract_transactions()
(dicts with timestamp, direction, category, amount, fee, balance, operator, counterparty_*).

The scorecard weights below are hand-set for a first demo. They are NOT calibrated on real repayment
outcomes. Calibrate (or replace with a fitted model) once a lender partner shares anonymised outcomes.
"""
from __future__ import annotations

import math
from datetime import datetime, timedelta

import numpy as np

FEATURES = [
    "hist_days", "n_txn", "monthly_income", "income_weeks_active", "income_cv", "out_in_ratio",
    "net_savings_rate", "bal_mean", "bal_min", "bal_cv", "low_bal_share", "buffer_ratio",
    "n_loan_repay", "airtime_share", "withdrawal_share", "merchant_share",
    "n_unique_payers", "payer_diversity", "top_payer_share", "integrity",
]
WEEKS = 13


def _ts(s):
    if s is None:
        return None
    s = str(s)
    return datetime.fromisoformat(s if "T" in s else s + "T00:00")


def ledger_integrity(rows):
    """Share of transactions whose balance equals previous balance +/- amount (+/- fee) on the same account.
    Forged or missing messages break the chain. Returns (share, n_checked)."""
    last, ok, n = {}, 0, 0
    for r in rows:
        op = r["operator"]
        if op in last:
            exp = last[op] + r["amount"] if r["direction"] == "in" else last[op] - r["amount"] - (r.get("fee") or 0)
            n += 1
            ok += abs(exp - r["balance"]) < 0.011
        last[op] = r["balance"]
    return (ok / n if n else 1.0), n


def build_features(rows):
    rows = sorted([r for r in rows if r.get("timestamp")], key=lambda r: _ts(r["timestamp"]))  # stable
    f = dict.fromkeys(FEATURES, 0.0)
    if len(rows) < 2:
        f["n_txn"] = len(rows)
        f["integrity"] = 1.0
        return f
    t0, t1 = _ts(rows[0]["timestamp"]), _ts(rows[-1]["timestamp"])
    f["hist_days"] = max((t1 - t0).days, 1)
    f["n_txn"] = len(rows)

    start = t1 - timedelta(days=WEEKS * 7)
    wk = np.zeros(WEEKS)
    income = cash_in = outflow = 0.0
    payers, n_inc, inc_amts = set(), 0, []
    out_by = {"airtime": 0.0, "withdrawal": 0.0, "merchant": 0.0}
    n_loan = 0
    for r in rows:
        t = _ts(r["timestamp"])
        if t < start:
            continue
        a = r["amount"]
        if r["direction"] == "in":
            if r["category"] == "deposit":
                cash_in += a
            else:
                income += a
                n_inc += 1
                inc_amts.append(a)
                wk[min(int((t - start).days // 7), WEEKS - 1)] += a
                payers.add(r.get("counterparty_phone") or r.get("counterparty_name") or f"anon-{n_inc}")
        else:
            out = a + (r.get("fee") or 0)
            outflow += out
            if r["category"] in out_by:
                out_by[r["category"]] += out
            if r["category"] == "loan_repayment":
                n_loan += 1

    f["monthly_income"] = income / (WEEKS * 7 / 30)
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

    last, totals = {}, []
    for r in rows:
        last[r["operator"]] = r["balance"]
        totals.append(sum(last.values()))
    totals = np.array(totals)
    f["bal_mean"], f["bal_min"] = float(totals.mean()), float(totals.min())
    f["bal_cv"] = float(totals.std() / totals.mean()) if totals.mean() > 0 else 3.0
    f["low_bal_share"] = float((totals < 20).mean())
    f["buffer_ratio"] = f["bal_mean"] / (outflow / (WEEKS * 7 / 30) + 1.0)
    f["integrity"] = ledger_integrity(rows)[0]
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


def make_passport(rows, user_label="applicant", pd_model_score=None):
    """The consent-gated, shareable summary. Contains aggregates only: no raw messages, names or numbers."""
    f = build_features(rows)
    sc = scorecard(f)
    integrity = f["integrity"]
    return {
        "applicant": user_label,
        "score": sc["score"], "band": band(sc["score"]),
        "model_pd_score": pd_model_score,
        "history_days": int(f["hist_days"]), "transactions_used": int(f["n_txn"]),
        "monthly_income_estimate": round(f["monthly_income"], 2),
        "income_active_weeks": f"{round(f['income_weeks_active'] * WEEKS)}/{WEEKS}",
        "avg_balance": round(f["bal_mean"], 2), "loan_repayments_seen": int(f["n_loan_repay"]),
        "components": sc["components"],
        "reasons_down": sc["reasons_down"], "reasons_up": sc["reasons_up"],
        "data_integrity": "consistent" if integrity >= 0.9 else "REVIEW: balance chain has breaks",
        "integrity_share": round(integrity, 3),
        "notes": "Built from consented SMS alerts. Scorecard weights are provisional until calibrated on repayment outcomes.",
    }
