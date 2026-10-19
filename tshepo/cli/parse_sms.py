"""Parse and classify one SMS with the rule engine.

    python -m tshepo.cli.parse_sms "RK12AB34CD Confirmed. You have received M100.00 ..." --sender MPESA
"""
import argparse
import json
from dataclasses import asdict

from ..parsing.parser import classify


def main():
    ap = argparse.ArgumentParser(description="Parse and classify one SMS")
    ap.add_argument("body")
    ap.add_argument("--sender", default=None)
    a = ap.parse_args()
    res = classify(a.body, a.sender)
    res["parsed"] = asdict(res["parsed"])
    print(json.dumps(res, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
