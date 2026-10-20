"""TF-IDF (+ optional rule flags) SMS classifier.

Two variants:
  * plain   - char_wb 2-5 grams + word 1-2 grams, logistic regression
  * hybrid  - the same, but the rule engine's flags are appended as tokens, so the
              model can learn from them AND override them when the text contradicts.

The class lives in its own module on purpose: joblib stores the module path in the
artefact, so moving this class would break loading of previously saved models.
"""
from __future__ import annotations

from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from ..logging import get_logger
from ..parsing.parser import parse, red_flags
from ..textprep import prep

logger = get_logger("models.tfidf")


def rule_tokens(body, sender):
    """Turn the rule engine's output into tokens so the model can use (and override) it."""
    p = parse(body, sender)
    f = red_flags(body, sender, p)
    toks = [f"FLAG_{k}" for k in f] + [f"RULESCORE{min(sum(f.values()), 6)}"]
    if p.is_transaction:
        toks.append("PARSED_TXN")
    return " ".join(toks)


class TfidfSmsClassifier:
    """predict(bodies, senders) -> list of labels; predict_proba -> (proba, classes)."""

    def __init__(self, use_rules=False):
        self.use_rules = use_rules
        self.char = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=2,
                                    sublinear_tf=True, lowercase=True)
        self.word = TfidfVectorizer(analyzer="word", ngram_range=(1, 2), min_df=2, sublinear_tf=True,
                                    lowercase=True, token_pattern=r"[A-Za-z0-9_\-\*#:/\.]+")
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
