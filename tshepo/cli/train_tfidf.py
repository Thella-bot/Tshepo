"""TF-IDF + logistic regression SMS classifier, compared with the rule baseline.

    python -m tshepo.cli.train_tfidf                       # synthetic train/test + noise stress test
    python -m tshepo.cli.train_tfidf --real data/real_labelled.jsonl
        # real file: one JSON per line with {"body": ..., "sender": ..., "label": genuine_transaction|suspicious|noise}
        # evaluates the model trained on synthetic data against YOUR real messages (the number that matters)

The hybrid variant sees the rule engine's flags as extra tokens, so it can learn from
them and override them when the text contradicts the rule.
"""
import argparse
import os
import random

import numpy as np

from ..config import LABELS, ensure_dir, model_path
from ..data.generator import build
from ..evaluation import group_split, load_jsonl, print_table, report
from ..models.tfidf import TfidfSmsClassifier
from ..parsing.parser import classify
from ..textprep import heavy_noise


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

    fit = lambda rows, rules: TfidfSmsClassifier(rules).fit([r["body"] for r in rows],
                                                            [r["sender"] for r in rows],
                                                            [r["label"] for r in rows])
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

    out = model_path(a.out)
    ensure_dir(os.path.dirname(out))
    import joblib
    joblib.dump(model, out)
    hybrid_out = out.replace("tfidf", "hybrid")
    joblib.dump(hybrid, hybrid_out)
    print(f"\nSaved {out} and {hybrid_out}")
    print("Label order:", list(model.clf.classes_), "| labels:", LABELS)


if __name__ == "__main__":
    main()
