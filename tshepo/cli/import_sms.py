"""Import a real SMS export into the JSONL format the project uses.

Android "Messages" and Samsung "SMS Backup+" exports look like:

    Received from <sender> on <YYYY-MM-DD> <HH:MM>

     <message body>

    -----------------------------------------------------

    python -m tshepo.cli.import_sms data/real_ecocash_raw.txt --out data/real_ecocash.jsonl
    python -m tshepo.cli.import_sms data/real_ecocash_raw.txt --out data/real_ecocash.jsonl --label genuine_transaction

Labels are optional. The importer only labels what you tell it to; the classifier
decides the rest. Real data stays in data/ and is gitignored.
"""
import argparse
import json
import os
import re

from ..config import LABELS, ensure_dir
from ..evaluation import load_jsonl

BLOCK_RE = re.compile(
    r"^Received from (?P<sender>.+?) on (?P<date>\d{4}-\d{2}-\d{2})[ T](?P<time>\d{2}:\d{2})",
    re.M)
SEP_RE = re.compile(r"^-{3,}\s*$", re.M)


def parse_export(text):
    """Split a phone export into (sender, received_at, body) records."""
    parts = SEP_RE.split(text)
    out = []
    for part in parts:
        m = BLOCK_RE.search(part)
        if not m:
            continue
        body = part[m.end():]
        body = body.replace("\r", "").strip("\n")
        body = " ".join(body.split())          # the export wraps long lines
        if not body:
            continue
        out.append(dict(sender=m.group("sender").strip(),
                        received_at=f"{m.group('date')}T{m.group('time')}",
                        body=body))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("input", help="the exported .txt file")
    ap.add_argument("--out", default=None, help="output JSONL (default: <input>.jsonl)")
    ap.add_argument("--label", default=None, choices=LABELS,
                    help="label every message in the file (use only when you know)")
    ap.add_argument("--user-id", default="REAL01", help="user id for these messages")
    a = ap.parse_args()

    with open(a.input, encoding="utf-8") as f:
        text = f.read()
    recs = parse_export(text)
    out = a.out or os.path.splitext(a.input)[0] + ".jsonl"
    ensure_dir(os.path.dirname(out) or ".")

    with open(out, "w", encoding="utf-8") as f:
        for i, r in enumerate(recs):
            r = dict(id=i, user_id=a.user_id, **r)
            if a.label:
                r["label"] = a.label
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"{len(recs)} messages -> {out}")
    senders = {}
    for r in recs:
        senders[r["sender"]] = senders.get(r["sender"], 0) + 1
    print("senders:", senders)
    if a.label:
        print(f"labelled as: {a.label}")
    else:
        print("no label written - run the classifier to label them")


if __name__ == "__main__":
    main()
