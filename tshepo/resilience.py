"""Resilience primitives: exceptions, graceful degradation, timeouts, circuit breaker.

Production services must never fail closed when a model artefact is corrupt or
unavailable. The rule gate is the zero-dependency fallback.
"""
from __future__ import annotations

import time
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Callable, Optional, cast

from .logging import get_logger

GateFn = Callable[[str, Optional[str]], str]

logger = get_logger("resilience")


class TshepoError(Exception):
    """Base exception for all Tshepo errors."""


class ModelError(TshepoError):
    """Raised when a model artefact cannot be loaded or used."""


class ValidationError(TshepoError):
    """Raised when input fails validation."""


class GateUnavailable(TshepoError):
    """Raised when the primary gate is unavailable and no fallback is configured."""


# --------------------------------------------------------------------------- circuit breaker


class CircuitBreaker:
    """Circuit breaker that trips after ``failure_threshold`` consecutive failures
    and stays open for ``cooldown_seconds`` before allowing a half-open probe.

    States: closed (normal) -> open (failing) -> half-open (probing) -> closed.
    """

    def __init__(
        self,
        failure_threshold: int = 5,
        cooldown_seconds: float = 30.0,
        half_open_max_calls: int = 1,
    ):
        self.failure_threshold = failure_threshold
        self.cooldown_seconds = cooldown_seconds
        self.half_open_max_calls = half_open_max_calls
        self._state = "closed"
        self._failures = 0
        self._last_failure_time: Optional[float] = None
        self._half_open_calls = 0

    @property
    def state(self) -> str:
        return self._state

    def _trip(self) -> None:
        self._state = "open"
        self._last_failure_time = time.monotonic()
        self._half_open_calls = 0
        logger.warning("circuit breaker tripped", extra={"state": "open"})

    def _half_open(self) -> None:
        self._state = "half-open"
        self._half_open_calls = 0

    def allow(self) -> bool:
        """Return True if a call is permitted right now."""
        if self._state == "closed":
            return True
        if self._state == "open":
            if time.monotonic() - (self._last_failure_time or 0) >= self.cooldown_seconds:
                self._half_open()
                return True
            return False
        # half-open: allow a limited number of probe calls
        if self._half_open_calls < self.half_open_max_calls:
            self._half_open_calls += 1
            return True
        return False

    def record_success(self) -> None:
        self._state = "closed"
        self._failures = 0
        self._half_open_calls = 0

    def record_failure(self) -> None:
        self._failures += 1
        if self._state == "half-open":
            self._trip()
        elif self._failures >= self.failure_threshold:
            self._trip()


# --------------------------------------------------------------------------- graceful gate


def fallback_gate(
    primary: GateFn, fallback: Optional[GateFn] = None, breaker: Optional[CircuitBreaker] = None
) -> GateFn:
    """Wrap a gate callable so failures fall back to ``fallback`` (default: rule gate).

    If ``breaker`` is given and the circuit is open, calls short-circuit straight to
    the fallback without invoking ``primary``. On any exception from ``primary`` it
    logs a warning and delegates to ``fallback``.
    """
    from .parsing.parser import classify

    def _default_fallback(body: str, sender: Optional[str] = None) -> str:
        return cast(str, classify(body, sender)["label"])

    fb = fallback or _default_fallback
    cb = breaker or CircuitBreaker()

    def wrapped(body: str, sender: Optional[str] = None) -> str:
        if not cb.allow():
            logger.warning("circuit open, using fallback gate", extra={"fallback": "rule_gate"})
            return fb(body, sender)
        try:
            result = primary(body, sender)
            cb.record_success()
            return result
        except Exception as e:  # noqa: BLE001 - degrade gracefully on any model failure
            cb.record_failure()
            logger.warning(
                "primary gate failed, falling back to rule gate",
                extra={"error": str(e), "fallback": "rule_gate", "circuit": cb.state},
            )
            return fb(body, sender)

    return wrapped


# --------------------------------------------------------------------------- timeouts


@contextmanager
def time_budget(seconds: float, label: str = "operation") -> Iterator[None]:
    """Context manager that warns when an operation exceeds a time budget.

    Does not abort the operation; used for observability of slow paths.
    """
    t0 = time.monotonic()
    try:
        yield
    finally:
        dur = time.monotonic() - t0
        if dur > seconds:
            logger.warning("time budget exceeded", extra={"label": label, "seconds": seconds, "dur_ms": round(dur * 1000, 1)})

