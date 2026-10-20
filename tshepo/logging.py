"""Structured logging for Tshepo with correlation-ID tracing.

Configure once at application startup:

    import tshepo.logging as tlog
    tlog.configure(level="INFO", json=True)

All modules should use ``logging.getLogger(__name__)``. A JSON formatter is
available so production logs can be ingested by log collectors.

Correlation IDs
---------------
Use :class:`correlation_id` (a :class:`contextvars.ContextVar`) to tag every log
record emitted during a single request / scoring call so you can trace a request
across modules:

    from tshepo.logging import correlation_id, configure
    import uuid

    token = correlation_id.set(str(uuid.uuid4()))
    try:
        logger.info("scoring started")
        ...
    finally:
        correlation_id.reset(token)

The correlation id is automatically added to every JSON log record emitted while
the context var is set.
"""
from __future__ import annotations

import contextvars
import json
import logging
import sys
import uuid
from contextvars import Token
from typing import Any, Literal, Optional, TextIO

#: Context variable carrying the current correlation id (request-scoped).
correlation_id: contextvars.ContextVar[Optional[str]] = contextvars.ContextVar(
    "correlation_id", default=None
)


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
            "time": self.formatTime(record, datefmt="%Y-%m-%dT%H:%M:%S"),
        }
        cid = correlation_id.get()
        if cid:
            payload["correlation_id"] = cid
        if record.exc_info:
            payload["exc_info"] = self.formatException(record.exc_info)
        for k in ("user_id", "model", "score", "duration_ms", "request_id"):
            v = getattr(record, k, None)
            if v is not None:
                payload[k] = v
        return json.dumps(payload, ensure_ascii=False)


def configure(level: str = "INFO", json_logs: bool = False, stream: Optional[TextIO] = None) -> None:
    """Configure the root logger for the tshepo package."""
    root = logging.getLogger("tshepo")
    root.setLevel(getattr(logging, level.upper(), logging.INFO))
    root.handlers.clear()
    handler = logging.StreamHandler(stream or sys.stderr)
    handler.setFormatter(JsonFormatter() if json_logs else logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
    root.addHandler(handler)
    logging.getLogger("tshepo").info("logging configured", extra={"level": level})


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(f"tshepo.{name}")


def new_correlation_id() -> str:
    """Generate a fresh correlation id (UUID4 hex)."""
    return uuid.uuid4().hex


class log_context:
    """Context manager that binds a correlation id (and optional extra fields) to
    all logs emitted within the block.

    Usage::

        with log_context(correlation_id="abc123", user_id="U0042"):
            logger.info("processing")

    If no correlation id is given, one is generated automatically.
    """

    def __init__(self, correlation_id: Optional[str] = None, **extra: Any):
        self.cid = correlation_id or new_correlation_id()
        self.extra = extra
        self._token: Optional[Token[Optional[str]]] = None

    def __enter__(self) -> "log_context":
        self._token = correlation_id.set(self.cid)
        return self

    def __exit__(
        self, exc_type: Optional[type[BaseException]], exc: Optional[BaseException], tb: Any
    ) -> Literal[False]:
        if self._token is not None:
            correlation_id.reset(self._token)
        if exc_type is not None:
            logging.getLogger("tshepo").exception(
                "request failed", extra={"correlation_id": self.cid, **self.extra}
            )
        return False
