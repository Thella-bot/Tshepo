"""Write a synthetic dataset to disk.

    python -m tshepo.cli.generate --users 200 --seed 42 --out data/synthetic_sms.jsonl --csv
"""
import argparse
import csv
import json
import os
from collections import Counter

from ..data.generator import generate_dataset


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--users", type=int, default=200)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="data/synthetic_sms.jsonl")
    ap.add_argument("--csv", action="store_true", help="also write a flat CSV next to the JSONL")
    a = ap.parse_args()

    recs = generate_dataset(n_users=a.users, seed=a.seed)
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as f:
        for r in recs:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    if a.csv:
        with open(os.path.splitext(a.out)[0] + ".csv", "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["id", "user_id", "sender", "received_at", "body", "label", "subtype", "language", "truth_json"])
            for r in recs:
                w.writerow([r["id"], r["user_id"], r["sender"], r["received_at"], r["body"], r["label"],
                            r["subtype"], r["language"], json.dumps(r["truth"])])
    print(f"Wrote {len(recs)} messages for {a.users} users -> {a.out}")
    print(dict(Counter(r["label"] for r in recs)))


if __name__ == "__main__":
    main()
