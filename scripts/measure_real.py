"""Measure balance-chain integrity on real imported messages."""
import json
import os
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tshepo.evaluation import load_jsonl
from tshepo.parsing.parser import extract_transactions
from tshepo.features.cashflow import ledger_integrity, build_features

for path in ["data/real_mpesa.jsonl", "data/real_ecocash.jsonl"]:
    recs = load_jsonl(path)
    rows = extract_transactions(recs)
    # group by operator since each account has its own chain
    last = {}
    ok = n = 0
    for r in rows:
        op = r["operator"]
        if op in last:
            exp = last[op] + r["amount"] if r["direction"] == "in" else last[op] - r["amount"] - (r.get("fee") or 0)
            n += 1
            ok += abs(exp - r["balance"]) < 0.011
        last[op] = r["balance"]
    print(f"{path}: {len(rows)} txn rows, {n} chain links, {ok/n:.1%} integrity")
    f = build_features(rows)
    print(f"  integrity feature: {f['integrity']:.3f}, hist_days: {f['hist_days']}, n_txn: {f['n_txn']}")
    print(f"  monthly_income: {f['monthly_income']:.2f}, out_in_ratio: {f['out_in_ratio']:.2f}, net_savings: {f['net_savings_rate']:.2f}")
