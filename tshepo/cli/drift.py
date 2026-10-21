"""Watch the operator templates for drift.

    python -m tshepo.cli.drift --baseline data/real_ecocash.jsonl --current data/real_ecocash.jsonl
    python -m tshepo.cli.drift --current data/new_export.jsonl --baseline data/real_ecocash.jsonl

Exits 2 when the drift verdict is "drift", so it can be used as a scheduled
check that fails loudly instead of a number nobody reads.
"""
import argparse
import json
import sys

from ..evaluation import load_jsonl
from ..parsing.drift import drift_report


def main() -> int:
    ap = argparse.ArgumentParser(description="operator template drift monitor")
    ap.add_argument("--current", required=True, help="the new export to check")
    ap.add_argument("--baseline", default=None, help="the window the parser is known to read")
    ap.add_argument("--json", action="store_true", help="print the report as JSON")
    a = ap.parse_args()

    current = load_jsonl(a.current)
    baseline = load_jsonl(a.baseline) if a.baseline else None
    report = drift_report(current, baseline)

    if a.json:
        print(json.dumps(report, indent=2, default=str))
    else:
        print(f"verdict      : {report['verdict']}")
        print(f"drift score  : {report['drift_score']}")
        print(f"messages     : {report['window']['messages']}")
        print(f"parse rate   : {report['window']['parse_rate']:.1%}")
        if report["findings"]:
            print("\nfindings:")
            for f in report["findings"]:
                print(f"  - {f}")
        else:
            print("\nno drift detected")
    return 2 if report["verdict"] == "drift" else 0


if __name__ == "__main__":
    sys.exit(main())
