"""Circuit breaker unit tests."""
from __future__ import annotations

import time

from tshepo.resilience import CircuitBreaker


def test_breaker_starts_closed():
    cb = CircuitBreaker(failure_threshold=3, cooldown_seconds=0.1)
    assert cb.state == "closed"
    assert cb.allow() is True


def test_breaker_trips_after_threshold():
    cb = CircuitBreaker(failure_threshold=2, cooldown_seconds=0.1)
    assert cb.allow() is True
    cb.record_failure()
    assert cb.state == "closed"
    assert cb.allow() is True
    cb.record_failure()
    assert cb.state == "open"
    assert cb.allow() is False


def test_breaker_half_open_after_cooldown():
    cb = CircuitBreaker(failure_threshold=1, cooldown_seconds=0.05)
    cb.record_failure()
    assert cb.state == "open"
    assert cb.allow() is False
    time.sleep(0.06)
    assert cb.allow() is True
    assert cb.state == "half-open"
    cb.record_success()
    assert cb.state == "closed"


def test_breaker_half_open_failure_retrips():
    cb = CircuitBreaker(failure_threshold=1, cooldown_seconds=0.05)
    cb.record_failure()
    time.sleep(0.06)
    assert cb.allow() is True
    assert cb.state == "half-open"
    cb.record_failure()
    assert cb.state == "open"
    assert cb.allow() is False
