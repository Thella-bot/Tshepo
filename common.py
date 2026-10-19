"""Shared helpers for the classifiers: input prep, group split, noise stress test, metrics."""
from __future__ import annotations

import json
import random
import re
from collections import Counter

from generator import perturb
from sms_parser import KNOWN_SENDERS

LABELS = ["genuine_transaction", "suspicious", "noise"]
L2I = {l: i for i, l in enumerate(LABELS)}


def sender_tag(sender):
    s = (sender or "").strip().upper()
    if s in KNOWN_SENDERS:
        return "SNDKNOWN"
    if s.startswith("+") or s[:3].isdigit():
        return "SNDPHONE"
    return "SNDOTHER"  # lookalike or unknown alphanumeric ID


def prep(body, sender=None):
    """Model input: sender-type token + body with digits normalised so formats generalise."""
    return f"{sender_tag(sender)} " + re.sub(r"\d", "0", " ".join(body.split()))


def heavy_noise(rng, s):
    """Stress test: stack perturbations and drop/swap letters inside words."""
    for _ in range(3):
        s = perturb(rng, s)
    out = []
    for w in s.split(" "):
        if len(w) > 4 and w.isalpha() and rng.random() < 0.3:
            j = rng.randint(1, len(w) - 2)
            w = w[:j] + w[j + 1:] if rng.random() < 0.5 else w[:j] + w[j + 1] + w[j] + w[j + 2:] if j + 2 <= len(w) else w
        out.append(w)
    return " ".join(out)


def group_split(records, test_frac=0.2, seed=0):
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
