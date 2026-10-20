"""Security hardening for Tshepo.

Threat model
------------
* An attacker supplies a malicious ``.joblib`` artefact. joblib uses pickle under the
  hood, and unpickling is arbitrary code execution (RCE). We therefore treat every
  artefact as untrusted: paths are confined to ``models/``, and callers are warned
  before loading.
* An attacker floods the scorer with huge / malformed messages to cause denial of
  service (ReDoS, memory exhaustion). Message bodies are length-capped and senders
  are validated.
* Path traversal through ``load_gate`` / ``load_transformer_gate`` is prevented by
  resolving every artefact path under ``models/``.
"""
from __future__ import annotations

import os
import re
import time
from collections import defaultdict, deque
from functools import wraps
from typing import Callable, Optional

from ..config import MODELS_DIR, ROOT

# --------------------------------------------------------------------------- limits
MAX_BODY_LEN = 4096          # characters per SMS body
MAX_SENDER_LEN = 64
MAX_RECORDS_PER_USER = 5000  # ledger rows considered per user
MAX_RECORDS_TOTAL = 200_000  # records processed in a single batch call
RATE_WINDOW = 60.0           # seconds
RATE_MAX = 120               # scoring calls per window per key

SENDER_RE = re.compile(r"^[A-Za-z0-9_\-+\. ]{1,64}$")


class SecurityError(ValueError):
    """Raised when an input fails a security check."""


def validate_message(body: str, sender: Optional[str] = None) -> tuple[str, Optional[str]]:
    """Sanitise and bound a single message before it reaches the parser/model.

    Raises SecurityError on oversized or malformed input.
    """
    if not isinstance(body, str):
        raise SecurityError("body must be a string")
    if len(body) > MAX_BODY_LEN:
        raise SecurityError(f"body exceeds {MAX_BODY_LEN} characters")
    # Null bytes and control characters are not meaningful in SMS; strip them.
    body = body.replace("\x00", "").replace("\r", " ")
    body = " ".join(body.split())
    if sender is not None:
        if not isinstance(sender, str) or len(sender) > MAX_SENDER_LEN:
            raise SecurityError("sender must be a string of at most 64 characters")
        if not SENDER_RE.match(sender.strip()):
            raise SecurityError("sender contains disallowed characters")
        sender = sender.strip()
    return body, sender


def validate_records(records) -> list[dict]:
    """Bound and normalise a list of message records."""
    if not isinstance(records, (list, tuple)):
        raise SecurityError("records must be a list")
    if len(records) > MAX_RECORDS_TOTAL:
        raise SecurityError(f"batch exceeds {MAX_RECORDS_TOTAL} records")
    clean = []
    for i, r in enumerate(records):
        if not isinstance(r, dict):
            raise SecurityError(f"record {i} is not an object")
        body, sender = validate_message(r.get("body", ""), r.get("sender"))
        clean.append(
            {
                "id": r.get("id", i),
                "user_id": r.get("user_id", "unknown"),
                "sender": sender,
                "body": body,
                "received_at": r.get("received_at"),
                "label": r.get("label"),
            }
        )
    return clean


# --------------------------------------------------------------------------- path confinement
def _confine(path: str, base: str) -> str:
    """Resolve ``path`` and refuse anything outside ``base`` (path traversal guard)."""
    base_abs = os.path.abspath(base)
    target = os.path.abspath(os.path.join(base_abs, path)) if not os.path.isabs(path) else os.path.abspath(path)
    if not (target == base_abs or target.startswith(base_abs + os.sep)):
        raise SecurityError(f"path escapes model directory: {path}")
    return target


def safe_model_path(name: str) -> str:
    """Resolve a model artefact path, confined to ``models/``."""
    return _confine(name, MODELS_DIR)


# --------------------------------------------------------------------------- rate limiting
class RateLimiter:
    """Token-bucket-ish sliding-window rate limiter keyed by an arbitrary id."""

    def __init__(self, max_calls: int = RATE_MAX, window: float = RATE_WINDOW):
        self.max_calls = max_calls
        self.window = window
        self._hits: dict[str, deque[float]] = defaultdict(deque)

    def allow(self, key: str = "default") -> bool:
        now = time.monotonic()
        hits = self._hits[key]
        cutoff = now - self.window
        while hits and hits[0] < cutoff:
            hits.popleft()
        if len(hits) >= self.max_calls:
            return False
        hits.append(now)
        return True

    def remaining(self, key: str = "default") -> int:
        now = time.monotonic()
        hits = self._hits[key]
        cutoff = now - self.window
        while hits and hits[0] < cutoff:
            hits.popleft()
        return max(0, self.max_calls - len(hits))


# module-level limiter shared by the scoring API
rate_limiter = RateLimiter()


def rate_limited(key_fn: Optional[Callable[..., str]] = None):
    """Decorator: refuse calls once the per-key rate limit is exceeded."""

    def dec(fn):
        @wraps(fn)
        def wrapper(*args, **kwargs):
            key = key_fn(*args, **kwargs) if key_fn else "default"
            if not rate_limiter.allow(key):
                raise SecurityError("rate limit exceeded")
            return fn(*args, **kwargs)

        return wrapper

    return dec


# --------------------------------------------------------------------------- safe deserialisation
def load_model(path: str):
    """Load a joblib artefact after confining its path to ``models/``.

    WARNING: joblib/pickle can execute arbitrary code on load. Only load artefacts
    you have produced yourself or have audited. Consider re-saving untrusted models
    through ``tshepo.models.scoring`` before loading them here.
    """
    import joblib

    p = safe_model_path(path)
    if not os.path.exists(p):
        raise FileNotFoundError(f"no model at {p}")
    return joblib.load(p)
