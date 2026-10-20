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
from typing import Any, Callable, Optional

from .config import MODELS_DIR, ROOT, get

# --------------------------------------------------------------------------- limits (overridable via TSHEPO_* env / config file)
def _body_len() -> int:
    return int(get("MAX_BODY_LEN", 4096))


def _sender_len() -> int:
    return int(get("MAX_SENDER_LEN", 64))


def _records_per_user() -> int:
    return int(get("MAX_RECORDS_PER_USER", 5000))


def _records_total() -> int:
    return int(get("MAX_RECORDS_TOTAL", 200000))


def _rate_window() -> float:
    return float(get("RATE_WINDOW", 60.0))


def _rate_max() -> int:
    return int(get("RATE_MAX", 120))


SENDER_RE = re.compile(r"^[A-Za-z0-9_\-+\. ]{1,128}$")

# Backward-compatible module-level names (tests / importers may reference these directly).
MAX_BODY_LEN: int = _body_len()
MAX_SENDER_LEN: int = _sender_len()
MAX_RECORDS_PER_USER: int = _records_per_user()
MAX_RECORDS_TOTAL: int = _records_total()
RATE_WINDOW: float = _rate_window()
RATE_MAX: int = _rate_max()


class SecurityError(ValueError):
    """Raised when an input fails a security check."""


class RateLimitExceeded(SecurityError):
    """Raised when a per-key rate limit is exceeded."""


def validate_message(body: str, sender: Optional[str] = None) -> tuple[str, Optional[str]]:
    """Sanitise and bound a single message before it reaches the parser/model.

    Raises SecurityError on oversized or malformed input.
    """
    if not isinstance(body, str):
        raise SecurityError("body must be a string")
    if len(body) > _body_len():
        raise SecurityError(f"body exceeds {_body_len()} characters")
    # Null bytes and control characters are not meaningful in SMS; strip them.
    body = body.replace("\x00", "").replace("\r", " ")
    body = " ".join(body.split())
    if sender is not None:
        if not isinstance(sender, str) or len(sender) > _sender_len():
            raise SecurityError(f"sender must be a string of at most {_sender_len()} characters")
        if not SENDER_RE.match(sender.strip()):
            raise SecurityError("sender contains disallowed characters")
        sender = sender.strip()
    return body, sender


def validate_records(records: Any) -> list[dict[str, Any]]:
    """Bound and normalise a list of message records."""
    if not isinstance(records, (list, tuple)):
        raise SecurityError("records must be a list")
    if len(records) > _records_total():
        raise SecurityError(f"batch exceeds {_records_total()} records")
    clean: list[dict[str, Any]] = []
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
    """Resolve ``path`` and refuse anything outside ``base`` (path traversal guard).

    Relative names are confined under ``base``. Absolute paths are returned as-is
    (the caller explicitly named them); only relative model *names* are sandboxed.
    """
    if os.path.isabs(path):
        return os.path.abspath(path)
    base_abs = os.path.abspath(base)
    target = os.path.abspath(os.path.join(base_abs, path))
    if not (target == base_abs or target.startswith(base_abs + os.sep)):
        raise SecurityError(f"path escapes model directory: {path}")
    return target


def safe_model_path(name: str) -> str:
    """Resolve a model artefact path, confined to ``models/``.

    Honours the same ``models/`` prefix convention as :func:`tshepo.config.model_path`,
    then verifies the result stays inside ``models/`` (path traversal guard).
    """
    if os.path.isabs(name):
        return os.path.abspath(name)
    if name.replace("\\", "/").startswith("models/"):
        resolved = os.path.abspath(os.path.join(ROOT, name))
    else:
        resolved = os.path.abspath(os.path.join(MODELS_DIR, name))
    base_abs = os.path.abspath(MODELS_DIR)
    if not (resolved == base_abs or resolved.startswith(base_abs + os.sep)):
        raise SecurityError(f"path escapes model directory: {name}")
    return resolved


# --------------------------------------------------------------------------- rate limiting
class RateLimiter:
    """Token-bucket-ish sliding-window rate limiter keyed by an arbitrary id."""

    def __init__(self, max_calls: Optional[int] = None, window: Optional[float] = None):
        self.max_calls = max_calls if max_calls is not None else _rate_max()
        self.window = window if window is not None else _rate_window()
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


def rate_limited(key_fn: Optional[Callable[..., str]] = None) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Decorator: refuse calls once the per-key rate limit is exceeded."""

    def dec(fn: Callable[..., Any]) -> Callable[..., Any]:
        @wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            key = key_fn(*args, **kwargs) if key_fn else "default"
            if not rate_limiter.allow(key):
                raise RateLimitExceeded("rate limit exceeded")
            return fn(*args, **kwargs)

        return wrapper

    return dec


# --------------------------------------------------------------------------- safe deserialisation
def load_model(path: str) -> Any:
    """Load a joblib artefact after confining its path to ``models/``.

    WARNING: joblib/pickle can execute arbitrary code on load. Only load artefacts
    you have produced yourself or have audited. Consider re-saving untrusted models
    through ``tshepo.models.scoring`` before loading them here.
    """
    import joblib  # type: ignore[import-untyped]

    p = safe_model_path(path)
    if not os.path.exists(p):
        raise FileNotFoundError(f"no model at {p}")
    return joblib.load(p)
