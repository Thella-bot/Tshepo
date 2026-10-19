"""Dataset splitting and metrics for the classifier experiments."""
from __future__ import annotations

import json
import random
from collections import Counter

from .config import LABELS


def group_split(records, test_frac=0.2, seed=0):
    """Split by user_id, never by message: messages from one user share a spending
    pattern, so a random per-message split leaks the user across train and test."""
    users = sorted({r["user_id"] for r in records})
    random.Random(seed).shuffle(users)
    test_users = set(users[: int(len(users) * test_frac)])
    tr = [r for r in records if r["user_id"] not in test_users]
    te = [r for r in records if r["user_id"] in test_users]
    return tr, te


def load_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def report(name, y_true, y_pred, quiet=False):
    conf = Counter(zip(y_true, y_pred))
    n = len(y_true)
    acc = sum(conf[(l, l)] for l in LABELS) / n
    res = {"name": name, "acc": acc}
    for l in LABELS:
        tp = conf[(l, l)]
        pred = sum(conf[(t, l)] for t in LABELS)
        tot = sum(conf[(l, p)] for p in LABELS)
        pr, rc = tp / max(pred, 1), tp / max(tot, 1)
        res[l] = (pr, rc, 2 * pr * rc / max(pr + rc, 1e-9))
    gen_tot = sum(conf[("genuine_transaction", p)] for p in LABELS)
    res["false_flag"] = conf[("genuine_transaction", "suspicious")] / max(gen_tot, 1)
    res["conf"] = conf
    return res


def print_table(results):
    print(f"{'model':28}{'acc':>8}{'scam recall':>13}{'scam prec':>11}{'genuine->scam':>15}{'genuine F1':>12}")
    for r in results:
        print(f"{r['name']:28}{r['acc']:>8.1%}{r['suspicious'][1]:>13.1%}{r['suspicious'][0]:>11.1%}"
              f"{r['false_flag']:>15.2%}{r['genuine_transaction'][2]:>12.3f}")
