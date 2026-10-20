"""Structured logging for Tshepo.

Configure once at application startup:

    import tshepo.logging as tlog
    tlog.configure(level="INFO", json=True)

All modules should use ``logging.getLogger(__name__)``. A JSON formatter is
available so production logs can be ingested by log collectors.
"""
from __future__ import annotations

import json
import logging
import sys
from typing import Optional


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
            "time": self.formatTime(record, datefmt="%Y-%m-%dT%H:%M:%S"),
        }
        if record.exc_info:
            payload["exc_info"] = self.formatException(record.exc_info)
        for k in ("user_id", "model", "score", "duration_ms"):
            v = getattr(record, k, None)
            if v is not None:
                payload[k] = v
        return json.dumps(payload, ensure_ascii=False)


def configure(level: str = "INFO", json_logs: bool = False, stream=None) -> None:
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
