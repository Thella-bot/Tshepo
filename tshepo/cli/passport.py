"""Build a credit passport for every user in a message file.

    python -m tshepo.cli.passport --file data/synthetic_sms.jsonl --gate models/hybrid.joblib --limit 3

Input: JSONL with body/sender/received_at and (optionally) user_id per line.
Output: one passport JSON per user, with aggregates only - no raw messages, names or numbers.
"""
import argparse
import json

from ..evaluation import load_jsonl
from ..models.gate import load_gate, load_transformer_gate
from ..pipeline import group_by_user, score_user


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", required=True, help="JSONL of messages")
    ap.add_argument("--gate", default=None, help="joblib classifier to gate messages")
    ap.add_argument("--transformer", default=None, help="fine-tuned transformer dir to gate messages")
    ap.add_argument("--out", default=None, help="write passports to this JSON file")
    ap.add_argument("--limit", type=int, default=3, help="how many users to print")
    a = ap.parse_args()

    label_fn = None
    if a.gate:
        label_fn = load_gate(a.gate)
    elif a.transformer:
        label_fn = load_transformer_gate(a.transformer)

    records = load_jsonl(a.file)
    by_user = group_by_user(records)
    passports = []
    for i, (uid, recs) in enumerate(by_user.items()):
        passports.append(score_user(recs, label_fn=label_fn, user_label=uid))
        if i + 1 >= a.limit:
            break

    for p in passports:
        print(json.dumps(p, indent=2, ensure_ascii=False))
        print()

    if a.out:
        with open(a.out, "w", encoding="utf-8") as f:
            json.dump(passports, f, indent=2, ensure_ascii=False)
        print(f"wrote {len(passports)} passports -> {a.out}")


if __name__ == "__main__":
    main()
