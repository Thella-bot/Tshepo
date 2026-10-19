#!/usr/bin/env python3
"""Run the rule baseline over a freshly generated synthetic set and print the numbers you'd quote."""
import argparse
from collections import Counter

from generator import generate_dataset
from sms_parser import classify

LABELS = ["genuine_transaction", "suspicious", "noise"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--users", type=int, default=300)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--show-errors", type=int, default=8)
    a = ap.parse_args()

    recs = generate_dataset(n_users=a.users, seed=a.seed)
    conf, errors = Counter(), []
    fields = Counter()
    fields_ok = Counter()
    for r in recs:
        res = classify(r["body"], r["sender"])
        conf[(r["label"], res["label"])] += 1
        if res["label"] != r["label"]:
            errors.append((r, res))
        if r["label"] == "genuine_transaction" and res["label"] == "genuine_transaction":
            t, p = r["truth"], res["parsed"]
            checks = {
                "operator": p.operator == t["operator"], "direction": p.direction == t["direction"],
                "category": p.category == t["category"], "amount": p.amount == t["amount"],
                "balance": p.balance == t["balance"], "timestamp": p.timestamp == t["timestamp"],
            }
            if t["fee"] is not None:
                checks["fee"] = p.fee == t["fee"]
            if t["cp_phone"] is not None:
                checks["cp_phone"] = p.counterparty_phone == t["cp_phone"]
            if t["cp_name"] is not None:
                checks["cp_name"] = (p.counterparty_name or "").lower() == t["cp_name"].lower()
            for k, ok in checks.items():
                fields[k] += 1
                fields_ok[k] += ok

    n = len(recs)
    correct = sum(conf[(l, l)] for l in LABELS)
    print(f"\nMessages: {n}   Overall accuracy: {correct / n:.1%}\n")
    print(f"{'true \\ predicted':22}" + "".join(f"{l[:12]:>14}" for l in LABELS) + f"{'recall':>10}")
    for t in LABELS:
        tot = sum(conf[(t, p)] for p in LABELS)
        print(f"{t:22}" + "".join(f"{conf[(t, p)]:>14}" for p in LABELS) + f"{conf[(t, t)] / max(tot, 1):>10.1%}")
    print()
    for l in LABELS:
        pred = sum(conf[(t, l)] for t in LABELS)
        print(f"precision {l:22} {conf[(l, l)] / max(pred, 1):.1%}")

    scam_total = sum(conf[("suspicious", p)] for p in LABELS)
    gen_total = sum(conf[("genuine_transaction", p)] for p in LABELS)
    gen_flagged = conf[("genuine_transaction", "suspicious")]
    print(f"\nScam detection rate (recall):          {conf[('suspicious', 'suspicious')] / scam_total:.1%}")
    print(f"Genuine txns wrongly flagged as scam:  {gen_flagged / gen_total:.1%}")
    print("\nField extraction accuracy on correctly-classified genuine messages:")
    for k in fields:
        print(f"  {k:10} {fields_ok[k] / fields[k]:.1%}  (n={fields[k]})")

    if errors and a.show_errors:
        print(f"\nSample misclassifications (first {a.show_errors}):")
        for r, res in errors[:a.show_errors]:
            print(f"- true={r['label']}/{r['subtype']} pred={res['label']} flags={res['flags']}\n  [{r['sender']}] {r['body']}")


if __name__ == "__main__":
    main()
