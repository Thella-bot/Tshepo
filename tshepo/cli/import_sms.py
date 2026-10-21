"""Import a real SMS export into the JSONL format the project uses.

Supported exports (auto-detected):

* Google Messages "Export messages" .txt
* Samsung SMS Backup+ .xml
* Google Messages JSON backup
* CSV / TSV with a date, a sender and a body column

    python -m tshepo.cli.import_sms data/real_ecocash_raw.txt --out data/real_ecocash.jsonl
    python -m tshepo.cli.import_sms data/real_ecocash_raw.txt --label genuine_transaction

The importer reports what the parser made of the export, because an export
that only yields a third of its messages is a data problem, not a scoring
problem, and it is invisible unless somebody looks.
"""
import argparse
import json
import os

from ..config import LABELS, ensure_dir
from ..importers import detect_format, parse_file, to_jsonl
from ..parsing.parser import build_ledger


def _readable(records):
    """What the parser made of the export: transactions, updates, failures, noise."""
    ledger = build_ledger(records)
    stats = dict(ledger.stats)
    stats.setdefault("messages", len(records))
    stats.setdefault("unread", 0)
    stats["readable_share"] = round(
        (stats["messages"] - stats["unread"]) / max(stats["messages"], 1), 3)
    stats["usable_share"] = round(stats.get("usable", 0) / max(stats["messages"], 1), 3)
    return stats


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("input", help="the exported file (.txt, .xml, .json, .csv, .tsv)")
    ap.add_argument("--out", default=None, help="output JSONL (default: <input>.jsonl)")
    ap.add_argument("--label", default=None, choices=LABELS,
                    help="label every message in the file (use only when you know)")
    ap.add_argument("--user-id", default="REAL01", help="user id for these messages")
    ap.add_argument("--format", default=None,
                    choices=["google_txt", "samsung_xml", "google_json", "csv", "tsv"],
                    help="override the format detection")
    a = ap.parse_args()

    with open(a.input, encoding="utf-8", errors="replace") as fh:
        head = fh.read(4096)
    fmt = a.format or detect_format(head, os.path.basename(a.input))

    recs = parse_file(a.input, fmt=fmt)
    out = a.out or os.path.splitext(a.input)[0] + ".jsonl"
    ensure_dir(os.path.dirname(out) or ".")

    rows = to_jsonl(recs, user_id=a.user_id, label=a.label)
    with open(out, "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"{len(rows)} messages -> {out}  (format: {fmt})")
    senders = {}
    for r in rows:
        senders[r.get("sender")] = senders.get(r.get("sender"), 0) + 1
    print("senders:", senders)
    stats = _readable(rows)
    print(f"parsed: {stats.get('transactions', 0)} transactions, "
          f"{stats.get('balance_updates', 0)} balance updates, "
          f"{stats.get('failed', 0)} failed transactions, "
          f"{stats.get('noise', 0)} noise, {stats.get('duplicates', 0)} duplicates")
    print(f"readable share: {stats.get('readable_share', 0):.0%} "
          f"(usable share {stats.get('usable_share', 0):.0%})")
    if stats.get("unread"):
        print(f"WARNING: {stats['unread']} message(s) could not be read at all. "
              f"They are not in the ledger and no money was taken from them.")
    if a.label:
        print(f"labelled as: {a.label}")
    else:
        print("no label written - run the classifier to label them")


if __name__ == "__main__":
    main()
