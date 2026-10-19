#!/usr/bin/env python3
"""
TF-IDF + logistic regression SMS classifier, compared with the rule baseline.

    python train_tfidf.py                       # synthetic train/test + noise stress test
    python train_tfidf.py --real data/real_labelled.jsonl
        # real file: one JSON per line with {"body": ..., "sender": ..., "label": genuine_transaction|suspicious|noise}
        # evaluates the model trained on synthetic data against YOUR real messages (the number that matters)
"""
import argparse
import os
import random

import joblib
import numpy as np
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from common import LABELS, group_split, heavy_noise, load_jsonl, prep, print_table, report
from generator import build
from sms_parser import classify, parse, red_flags


def rule_tokens(body, sender):
    """Turn the rule engine's output into tokens so the model can use (and override) it."""
    p = parse(body, sender)
    f = red_flags(body, sender, p)
    toks = [f"FLAG_{k}" for k in f] + [f"RULESCORE{min(sum(f.values()), 6)}"]
    if p.is_transaction:
        toks.append("PARSED_TXN")
    return " ".join(toks)


class TfidfSmsClassifier:
    def __init__(self, use_rules=False):
        self.use_rules = use_rules
        self.char = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=2, sublinear_tf=True, lowercase=True)
        self.word = TfidfVectorizer(analyzer="word", ngram_range=(1, 2), min_df=2, sublinear_tf=True, lowercase=True,
                                    token_pattern=r"[A-Za-z0-9_\-\*#:/\.]+")
        self.clf = LogisticRegression(C=8.0, max_iter=2000, class_weight="balanced")

    def _text(self, b, s):
        t = prep(b, s)
        return t + " " + rule_tokens(b, s) if self.use_rules else t

    def _x(self, texts, fit=False):
        if fit:
            return hstack([self.char.fit_transform(texts), self.word.fit_transform(texts)]).tocsr()
        return hstack([self.char.transform(texts), self.word.transform(texts)]).tocsr()

    def fit(self, bodies, senders, labels):
        self.clf.fit(self._x([self._text(b, s) for b, s in zip(bodies, senders)], fit=True), labels)
        return self

    def predict_proba(self, bodies, senders):
        p = self.clf.predict_proba(self._x([self._text(b, s) for b, s in zip(bodies, senders)]))
        return p, list(self.clf.classes_)

    def predict(self, bodies, senders):
        return list(self.clf.predict(self._x([self._text(b, s) for b, s in zip(bodies, senders)])))


def rules_predict(recs):
    return [classify(r["body"], r["sender"])["label"] for r in recs]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--users", type=int, default=1500)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--real", default=None)
    ap.add_argument("--out", default="models/tfidf.joblib")
    a = ap.parse_args()

    recs, _ = build(n_users=a.users, seed=a.seed)
    tr, te = group_split(recs, 0.2, seed=a.seed)
    print(f"train {len(tr)} msgs / test {len(te)} msgs (split by user, no leakage)")

    fit = lambda rows, rules: TfidfSmsClassifier(rules).fit([r["body"] for r in rows], [r["sender"] for r in rows], [r["label"] for r in rows])
    model, hybrid = fit(tr, False), fit(tr, True)
    pred = lambda m, d: m.predict([r["body"] for r in d], [r["sender"] for r in d])

    rng = random.Random(123)
    noisy = [dict(r, body=heavy_noise(rng, r["body"])) for r in te]

    sets = {"synthetic test": te, "synthetic + heavy typos": noisy}
    if a.real:
        sets["REAL messages"] = load_jsonl(a.real)

    for name, ds in sets.items():
        y = [r["label"] for r in ds]
        print(f"\n== {name} (n={len(ds)}) ==")
        print_table([
            report("rules baseline", y, rules_predict(ds)),
            report("TF-IDF + LogReg", y, pred(model, ds)),
            report("Hybrid (TF-IDF + rules)", y, pred(hybrid, ds)),
        ])

    # Generalisation test: never show the model these scam types, then see if it still catches them.
    held = {"fake_agent", "loan_scam", "pin_request"}
    tr2 = [r for r in tr if r["subtype"] not in held]
    m2, h2 = fit(tr2, False), fit(tr2, True)
    unseen = [r for r in te if r["subtype"] in held]
    others = [r for r in te if r["subtype"] not in held]
    print(f"\n== Unseen scam types {sorted(held)} (excluded from training) ==")
    for nm, pred_fn in [("rules baseline", rules_predict),
                        ("TF-IDF + LogReg", lambda d: pred(m2, d)),
                        ("Hybrid (TF-IDF + rules)", lambda d: pred(h2, d))]:
        rec = sum(p == "suspicious" for p in pred_fn(unseen)) / len(unseen)
        ff = sum(p == "suspicious" for r, p in zip(others, pred_fn(others)) if r["label"] != "suspicious") / \
             max(1, sum(r["label"] != "suspicious" for r in others))
        print(f"{nm:26} recall on unseen scam types: {rec:6.1%}   non-scam wrongly flagged: {ff:.2%}   (n={len(unseen)})")

    # which tokens drive "suspicious"?
    names = np.array(list(model.char.get_feature_names_out()) + list(model.word.get_feature_names_out()))
    ci = list(model.clf.classes_).index("suspicious")
    top = np.argsort(model.clf.coef_[ci])[-12:][::-1]
    print("\nTop features pushing toward 'suspicious':", ", ".join(str(t) for t in names[top]))

    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    joblib.dump(model, a.out)
    joblib.dump(hybrid, a.out.replace('tfidf', 'hybrid'))
    print(f"\nSaved {a.out} and {a.out.replace('tfidf', 'hybrid')}")


if __name__ == "__main__":
    main()
