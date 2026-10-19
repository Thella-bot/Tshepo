"""Fitted cash-flow scoring models.

Three scorers, same feature vector (tshepo.features.cashflow.FEATURES):
  * ScorecardModel  - the transparent 300-850 scorecard, no training, hand-set weights
  * ScoreModel      - logistic regression or gradient boosting on the same features,
                      fitted on repayment outcomes, saved with joblib

The fitted models are only as good as the labels they were fitted on. The repo ships
SYNTHETIC outcomes (a hidden reliability variable drives both behaviour and default);
replace them with anonymised real repayment outcomes before trusting a score.
"""
from __future__ import annotations

import os

import joblib
import numpy as np

from ..config import model_path
from ..features.cashflow import FEATURES, band, build_features, scorecard


class ScorecardModel:
    """Transparent scorecard: hand-set weights, no fitted parameters, always auditable."""

    def __init__(self):
        self.name = "scorecard"

    def score(self, features):
        return scorecard(features)["score"]

    def score_rows(self, rows):
        return self.score(build_features(rows))

    def explain(self, features):
        return scorecard(features)

    def predict_proba(self, features):
        """Map the 300-850 score to a pseudo-probability of default (monotone, calibrated-ish)."""
        s = self.score(features)
        return 1.0 / (1.0 + np.exp((s - 575.0) / 75.0))


class ScoreModel:
    """Fitted model (logistic regression or gradient boosting) over the cash-flow features."""

    def __init__(self, estimator, feature_names=None, metadata=None):
        self.estimator = estimator
        self.feature_names = list(feature_names or FEATURES)
        self.metadata = metadata or {}

    @classmethod
    def fit(cls, X, y, kind="gb", seed=0, feature_names=None):
        from sklearn.ensemble import HistGradientBoostingClassifier
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler

        if kind == "lr":
            est = make_pipeline(StandardScaler(),
                                LogisticRegression(C=0.5, max_iter=2000))
        else:
            est = HistGradientBoostingClassifier(max_depth=3, learning_rate=0.06,
                                                  max_iter=150, random_state=seed)
        est.fit(X, y)
        return cls(est, feature_names=feature_names,
                   metadata={"kind": kind, "n_train": int(len(y)),
                             "train_base_rate": float(np.mean(y)) if len(y) else None})

    def predict_proba(self, features):
        """features: dict of FEATURES -> value."""
        x = np.array([[features[k] for k in self.feature_names]], dtype=float)
        return float(self.estimator.predict_proba(x)[0, 1])

    def score(self, features):
        """Convert a default probability into the same 300-850 range as the scorecard."""
        return int(round(300 + 550 * (1.0 - self.predict_proba(features))))

    def score_rows(self, rows):
        return self.score(build_features(rows))

    def explain(self, features):
        return {"score": self.score(features), "pd": self.predict_proba(features),
                "band": band(self.score(features))}

    def save(self, path):
        path = model_path(path)
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        joblib.dump(self, path)
        return path

    @classmethod
    def load(cls, path):
        return joblib.load(model_path(path))
