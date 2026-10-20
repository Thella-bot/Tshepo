"""Run the Tshepo HTTP API server.

    python -m tshepo.cli.serve --port 8080 --workers 1
    TSHEPO_LOG_LEVEL=DEBUG python -m tshepo.cli.serve --port 8080
"""
from __future__ import annotations

import argparse

from ..config import LOG_LEVEL


def main():
    ap = argparse.ArgumentParser(description="Serve the Tshepo HTTP API")
    ap.add_argument("--host", default="0.0.0.0", help="bind address")
    ap.add_argument("--port", type=int, default=8080, help="port to listen on")
    ap.add_argument("--workers", type=int, default=1, help="number of uvicorn workers")
    ap.add_argument("--reload", action="store_true", help="auto-reload on code changes")
    ap.add_argument("--log-level", default=LOG_LEVEL, help="uvicorn log level")
    a = ap.parse_args()

    import uvicorn

    uvicorn.run(
        "tshepo.api:app",
        host=a.host,
        port=a.port,
        workers=a.workers,
        reload=a.reload,
        log_level=a.log_level.lower(),
    )


if __name__ == "__main__":
    main()
