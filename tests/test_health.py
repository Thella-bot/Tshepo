"""Production health checks."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tshepo.health import check_all, check_features, check_parser, check_scorecard


def test_check_parser_ok():
    r = check_parser()
    assert r["ok"] is True


def test_check_features_ok():
    r = check_features()
    assert r["ok"] is True


def test_check_scorecard_ok():
    r = check_scorecard()
    assert r["ok"] is True


def test_check_all_structure():
    r = check_all()
    assert "ok" in r
    assert isinstance(r["ok"], bool)
    for k in ("parser", "features", "scorecard", "models_dir"):
        assert k in r
        assert "ok" in r[k]
        assert "ms" in r[k]
