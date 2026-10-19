#!/usr/bin/env python3
"""
End-to-end: synthetic SMS -> classifier gate -> ledger -> cash-flow features -> score.

    python train_score.py                  # rules as the gate
    python train_score.py --gate models/hybrid.joblib

Outcomes (`defaulted`) are SYNTHETIC: they come from a hidden reliability variable that also shapes the
simulated behaviour. A good AUC here only proves the pipeline works. It says nothing about real default risk.
"""
import argparse
import json
import random
from collections import defaultdict

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

from cashflow import FEATURES, band, build_features, make_passport, scorecard
from generator import build, phone, ref, stamp_mpesa
from sms_parser import classify, extract_transactions
from datetime import datetime


def ks(y, s):
    fpr, tpr, _ = roc_curve(y, s)
    return float((tpr - fpr).max())


def by_user(records):
    d = defaultdict(list)
    for r in records:
        d[r["user_id"]].append(r)
    return d


def forge(rng, user_recs, n, skilled):
    """Fake 'received' alerts. naive: random sender. skilled: spoofed MPESA sender ID, balances not chained."""
    out = []
    times = [datetime.fromisoformat(r["received_at"]) for r in user_recs]
    lo, hi = min(times), max(times)
    for _ in range(n):
        dt = lo + (hi - lo) * rng.random()
        amt = rng.choice([1500, 2000, 2500, 3000])
        bal = rng.randint(2000, 9000)
        body = (f"{ref(rng)} Confirmed. You have received M{amt:,.2f} from {rng.choice(['THABO', 'LERATO', 'MPHO'])} "
                f"{rng.choice(['MOKOENA', 'LETSIE'])} {phone(rng)} on {stamp_mpesa(dt)}. New M-PESA balance is M{bal:,.2f}.")
        sender = "MPESA" if skilled else rng.choice(["+266" + phone(rng), "MPESA-INFO"])
        out.append(dict(user_id=user_recs[0]["user_id"], sender=sender, body=body,
                        received_at=dt.isoformat(timespec="minutes"), label="forged"))
    return out


def naive_rows(recs):
    """No shield: trust anything that parses like a transaction."""
    rows = extract_transactions(recs, label_fn=lambda b, s: "genuine_transaction")
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--users", type=int, default=1500)
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--gate", default=None, help="path to a joblib text classifier to gate messages (default: rules)")
    a = ap.parse_args()

    label_fn = None
    if a.gate:
        from ml_classifier import load_gate
        label_fn = load_gate(a.gate)

    records, users = build(n_users=a.users, seed=a.seed, events=(30, 70), credit_signal=True)
    recs_by_user = by_user(records)
    X, y, uids = [], [], []
    for u in users:
        rows = extract_transactions(recs_by_user[u["user_id"]], label_fn=label_fn)
        f = build_features(rows)
        X.append([f[k] for k in FEATURES])
        y.append(u["defaulted"])
        uids.append(u["user_id"])
    X, y = np.array(X), np.array(y)
    print(f"{len(y)} users, synthetic default rate {y.mean():.1%}, gate = {a.gate or 'rules'}")

    idx = list(range(len(y)))
    random.Random(0).shuffle(idx)
    cut = int(len(idx) * 0.7)
    tr, te = idx[:cut], idx[cut:]

    # 1) transparent scorecard (no training)
    card = np.array([scorecard(dict(zip(FEATURES, row)))["score"] for row in X])
    # 2) logistic regression, 3) gradient boosting
    lr = make_pipeline(StandardScaler(), LogisticRegression(C=0.5, max_iter=2000)).fit(X[tr], y[tr])
    gb = HistGradientBoostingClassifier(max_depth=3, learning_rate=0.06, max_iter=150, random_state=0).fit(X[tr], y[tr])
    p_lr, p_gb = lr.predict_proba(X[te])[:, 1], gb.predict_proba(X[te])[:, 1]

    print("\nHeld-out users (synthetic outcomes):")
    print(f"{'model':32}{'AUC':>8}{'KS':>8}")
    for name, s in [("Transparent scorecard", -card[te]), ("Logistic regression", p_lr), ("Gradient boosting", p_gb)]:
        print(f"{name:32}{roc_auc_score(y[te], s):>8.3f}{ks(y[te], s):>8.3f}")

    # default rate by scorecard band
    print("\nDefault rate by scorecard band (held-out users):")
    cs, yt = card[te], y[te]
    for lo, hi in [(0, 450), (450, 550), (550, 650), (650, 750), (750, 900)]:
        m = (cs >= lo) & (cs < hi)
        if m.sum():
            print(f"  {band(lo):10} score {lo:>3}-{hi:<3} n={m.sum():>4}  default rate {yt[m].mean():6.1%}")

    print("\nMost important features (logistic regression, standardised coefficients):")
    coef = lr[-1].coef_[0]
    for i in np.argsort(-np.abs(coef))[:6]:
        print(f"  {FEATURES[i]:22} {coef[i]:+.2f}  ({'raises' if coef[i] > 0 else 'lowers'} default risk)")

    # ---- Does the shield protect the score? Forged inflows added to the TEST users.
    rng = random.Random(5)
    print("\nTamper test: 12 forged 'received M1,500-3,000' alerts added to each test user")
    print(f"{'forger':26}{'avg score, no shield':>22}{'avg score, shield':>19}{'honest avg':>12}{'integrity flag':>16}")
    h_flag = np.mean([build_features(extract_transactions(recs_by_user[uids[i]], label_fn=label_fn))["integrity"] < 0.9 for i in te[:200]])
    honest = np.mean([scorecard(build_features(extract_transactions(recs_by_user[uids[i]], label_fn=label_fn)))["score"] for i in te[:200]])
    for kind, skilled in [("naive (random sender)", False), ("skilled (spoofed MPESA)", True)]:
        no_sh, sh, flagged = [], [], 0
        for i in te[:200]:
            base = recs_by_user[uids[i]]
            mixed = sorted(base + forge(rng, base, 12, skilled), key=lambda r: r["received_at"])
            r_no = naive_rows(mixed)
            r_sh = extract_transactions(mixed, label_fn=label_fn)
            no_sh.append(scorecard(build_features(r_no))["score"])
            sh.append(scorecard(build_features(r_sh))["score"])
            flagged += build_features(r_sh)["integrity"] < 0.9
        print(f"{kind:26}{np.mean(no_sh):>22.0f}{np.mean(sh):>19.0f}{honest:>12.0f}{flagged / 200:>16.0%}")
    print(f"(integrity flag = share of users whose balance chain breaks and are flagged for review; honest users flagged: {h_flag:.0%})")

    # ---- Example passport
    i = te[0]
    rows = extract_transactions(recs_by_user[uids[i]], label_fn=label_fn)
    print("\nExample passport:")
    print(json.dumps(make_passport(rows, user_label=uids[i], pd_model_score=round(float(850 - 550 * p_lr[0]))), indent=2))


if __name__ == "__main__":
    main()
