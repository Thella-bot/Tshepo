"""Security: input validation, path confinement, rate limiting."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest

from tshepo.security import (MAX_BODY_LEN, RateLimiter, SecurityError, safe_model_path,
                             validate_message, validate_records)


def test_validate_message_ok():
    body, sender = validate_message("Hello world", "MPESA")
    assert body == "Hello world"
    assert sender == "MPESA"


def test_validate_message_oversize_rejected():
    with pytest.raises(SecurityError):
        validate_message("x" * (MAX_BODY_LEN + 1), "MPESA")


def test_validate_message_bad_sender_rejected():
    with pytest.raises(SecurityError):
        validate_message("hi", "../../../etc/passwd")


def test_validate_message_null_stripped():
    body, _ = validate_message("a\x00b  c", "MPESA")
    assert body == "ab c"


def test_validate_records_bounds():
    recs = [{"body": "b1", "sender": "MPESA", "user_id": "U1"} for _ in range(3)]
    clean = validate_records(recs)
    assert len(clean) == 3


def test_validate_records_oversize_rejected():
    recs = [{"body": "b", "sender": "MPESA"} for _ in range(250_000)]
    with pytest.raises(SecurityError):
        validate_records(recs)


def test_rate_limiter():
    rl = RateLimiter(max_calls=2, window=60)
    assert rl.allow("k") is True
    assert rl.allow("k") is True
    assert rl.allow("k") is False
    assert rl.remaining("k") == 0
    # other key unaffected
    assert rl.allow("other") is True


def test_safe_model_path_confined(tmp_path):
    # relative name confined under models/
    p = safe_model_path("tfidf.joblib")
    assert p.endswith(os.path.join("models", "tfidf.joblib"))
    # traversal rejected
    with pytest.raises(SecurityError):
        safe_model_path("../../../etc/passwd")
    # "models/" prefix handled
    p2 = safe_model_path("models/hybrid.joblib")
    assert p2.endswith(os.path.join("models", "hybrid.joblib"))


def test_safe_model_path_absolute_allowed(tmp_path):
    # absolute paths are explicit intent, allowed through
    p = os.path.join(tmp_path, "score.joblib")
    assert safe_model_path(p) == os.path.abspath(p)
