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
import time

import joblib
import numpy as np

from ..config import model_path
from ..features.cashflow import FEATURES, band, build_features, scorecard
from ..logging import get_logger
from ..security import load_model as _safe_load

logger = get_logger("models.scoring")

MODEL_META_VERSION = "1.0"


class ScorecardModel:
    """Transparent scorecard: hand-set weights, no fitted parameters, always auditable."""

    def __init__(self):
        self.name = "scorecard"
        self.version = MODEL_META_VERSION

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
    """Fitted model (logistic regression or gradient boosting) over the cash-flow features.

    Exposes feature importances (when available) so scoring decisions stay explainable.
    """

    def __init__(self, estimator, feature_names=None, metadata=None):
        self.estimator = estimator
        self.feature_names = list(feature_names or FEATURES)
        self.metadata = dict(metadata or {})
        self.metadata.setdefault("version", MODEL_META_VERSION)
        self.metadata.setdefault("created_at", time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))

    @classmethod
    def fit(cls, X, y, kind="gb", seed=0, feature_names=None, metadata=None):
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
        meta = {"kind": kind, "n_train": int(len(y)),
                "train_base_rate": float(np.mean(y)) if len(y) else None}
        if metadata:
            meta.update(metadata)
        return cls(est, feature_names=feature_names, metadata=meta)

    def _x(self, features):
        """Build a 2-D array from a features dict, tolerating missing keys."""
        return np.array([[float(features.get(k, 0.0)) for k in self.feature_names]], dtype=float)

    def predict_proba(self, features):
        """features: dict of FEATURES -> value. Returns P(default)."""
        return float(self.estimator.predict_proba(self._x(features))[0, 1])

    def predict_proba_batch(self, feature_dicts):
        """Batch predict_proba for many feature dicts."""
        X = np.array([[float(d.get(k, 0.0)) for k in self.feature_names] for d in feature_dicts],
                     dtype=float)
        return self.estimator.predict_proba(X)[:, 1]

    def score(self, features):
        """Convert a default probability into the same 300-850 range as the scorecard."""
        return int(round(300 + 550 * (1.0 - self.predict_proba(features))))

    def score_rows(self, rows):
        return self.score(build_features(rows))

    def explain(self, features):
        """Return score, probability of default, band and per-feature contribution hints."""
        pd = self.predict_proba(features)
        return {"score": self.score(features), "pd": pd,
                "band": band(self.score(features)),
                "feature_importances": self.feature_importances()}

    def feature_importances(self):
        """Per-feature importance magnitudes (model-dependent), aligned to feature_names."""
        est = self.estimator
        if hasattr(est, "feature_importances_"):
            imp = est.feature_importances_
        elif hasattr(est, "named_steps") and "logisticregression" in est.named_steps:
            coef = est.named_steps["logisticregression"].coef_[0]
            imp = np.abs(coef)
        else:
            return None
        return {k: float(v) for k, v in zip(self.feature_names, imp)}

    def save(self, path):
        path = model_path(path)
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        joblib.dump(self, path)
        logger.info("model saved", extra={"model": "score_model", "path": path,
                                           "version": self.metadata.get("version")})
        return path

    @classmethod
    def load(cls, path):
        """Load a fitted scoring model via the safe (path-confined) loader.

        WARNING: joblib/pickle can execute arbitrary code on load. Only load artefacts
        you have produced yourself or have audited.
        """
        obj = _safe_load(path)
        if not isinstance(obj, cls):
            raise TypeError(f"loaded artefact is not a {cls.__name__}")
        logger.info("model loaded", extra={"model": "score_model", "path": path,
                                           "version": obj.metadata.get("version")})
        return obj
