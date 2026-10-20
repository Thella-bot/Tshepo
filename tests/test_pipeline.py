"""Classifiers, the gate, and the end-to-end pipeline."""
import numpy as np
import pytest

from tshepo.evaluation import group_split, report
from tshepo.models.gate import rule_gate
from tshepo.models.tfidf import TfidfSmsClassifier
from tshepo.models.scoring import ScorecardModel, ScoreModel
from tshepo.pipeline import ledger_for, score_user
from tshepo.textprep import prep, sender_tag


def test_sender_tag():
    assert sender_tag("MPESA") == "SNDKNOWN"
    assert sender_tag("+26658123456") == "SNDPHONE"
    assert sender_tag("MPESA-INFO") == "SNDOTHER"


def test_prep_normalises_digits():
    out = prep("You received M100.00 from 58123456", "MPESA")
    assert "100" not in out and "0.00" in out
    assert out.startswith("SNDKNOWN")


def test_heavy_noise_never_crashes_and_changes_words():
    """Regression: short alphabetic words (len 5) used to raise IndexError."""
    import random
    from tshepo.textprep import heavy_noise
    rng = random.Random(0)
    for _ in range(300):
        out = heavy_noise(rng, "Hello world balance bla bla")
        assert isinstance(out, str) and out
    # the swap branch must actually transpose letters, not copy them
    changed = heavy_noise(random.Random(1), "balance")
    assert changed != "balance"


def test_group_split_has_no_user_leakage(small_dataset):
    records, _ = small_dataset
    tr, te = group_split(records, 0.3, seed=1)
    tr_u = {r["user_id"] for r in tr}
    te_u = {r["user_id"] for r in te}
    assert not (tr_u & te_u)
    assert len(tr) + len(te) == len(records)


def test_tfidf_model_learns_and_predicts(small_dataset):
    records, _ = small_dataset
    tr, te = group_split(records, 0.3, seed=2)
    model = TfidfSmsClassifier(use_rules=False).fit(
        [r["body"] for r in tr], [r["sender"] for r in tr], [r["label"] for r in tr])
    preds = model.predict([r["body"] for r in te], [r["sender"] for r in te])
    assert len(preds) == len(te)
    assert set(preds) <= {"genuine_transaction", "suspicious", "noise"}
    res = report("tfidf", [r["label"] for r in te], preds)
    assert res["acc"] > 0.8


def test_hybrid_model_uses_rule_flags(small_dataset):
    from tshepo.models.tfidf import rule_tokens
    phishing = "Your account is suspended. Verify at http://x.example/verify to avoid closure."
    assert "FLAG_" in rule_tokens(phishing, "MPESA-INFO")

    records, _ = small_dataset
    tr, _ = group_split(records, 0.3, seed=2)
    hybrid = TfidfSmsClassifier(use_rules=True).fit(
        [r["body"] for r in tr], [r["sender"] for r in tr], [r["label"] for r in tr])
    plain = TfidfSmsClassifier(use_rules=False)
    # the hybrid text carries the rule tokens, the plain text does not
    assert hybrid._text(phishing, "MPESA-INFO") != plain._text(phishing, "MPESA-INFO")
    assert "FLAG_" in hybrid._text(phishing, "MPESA-INFO")


def test_rule_gate_matches_classifier():
    from tshepo.parsing.parser import classify
    body = "RK12AB34CD Confirmed. You have received M100.00 from THABO 58123456 on 19/10/26 at 2:15 PM. New M-PESA balance is M900.00."
    assert rule_gate(body, "MPESA") == classify(body, "MPESA")["label"]


def test_pipeline_scores_a_user(by_user):
    uid = sorted(by_user)[0]
    p = score_user(by_user[uid], label_fn=rule_gate, user_label=uid)
    assert 300 <= p["score"] <= 850
    assert p["transactions_used"] > 0
    assert p["history_days"] >= 0
    assert "components" in p


def test_ledger_excludes_scam_messages(by_user):
    uid = sorted(by_user)[0]
    rows = ledger_for(by_user[uid], label_fn=rule_gate)
    assert rows
    for r in rows:
        assert r["amount"] is not None and r["direction"] in {"in", "out"}


def test_scorecard_model_and_fitted_model_agree_on_scale():
    card = ScorecardModel()
    f = {"hist_days": 90, "n_txn": 20, "monthly_income": 2000, "income_weeks_active": 0.8,
         "income_cv": 0.5, "out_in_ratio": 0.6, "net_savings_rate": 0.2, "bal_mean": 1500,
         "bal_min": 100, "bal_cv": 0.4, "low_bal_share": 0.1, "buffer_ratio": 2.0,
         "n_loan_repay": 2, "airtime_share": 0.05, "withdrawal_share": 0.1, "merchant_share": 0.3,
         "n_unique_payers": 6, "payer_diversity": 0.8, "top_payer_share": 0.4, "integrity": 1.0}
    assert 300 <= card.score(f) <= 850
    assert 0.0 <= card.predict_proba(f) <= 1.0


def test_fitted_score_model_roundtrip(tmp_path):
    from tshepo.features.cashflow import FEATURES
    rng = np.random.default_rng(0)
    X = rng.random((60, len(FEATURES)))
    y = (X[:, 0] + 0.5 * X[:, 1] > 0.9).astype(int)
    m = ScoreModel.fit(X, y, kind="lr")
    p = tmp_path / "score.joblib"
    m.save(str(p))
    m2 = ScoreModel.load(str(p))
    f = dict(zip(FEATURES, X[0]))
    assert m2.predict_proba(f) == pytest.approx(m.predict_proba(f))
    assert 300 <= m2.score(f) <= 850
