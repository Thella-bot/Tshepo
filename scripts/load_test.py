"""Load test / benchmark for the Tshepo HTTP API.

Usage:

    # sequential baseline
    python scripts/load_test.py --requests 200

    # concurrent load
    python scripts/load_test.py --requests 500 --concurrency 20

    # against a running server
    TSHEPO_API_KEY=testkey python scripts/load_test.py --requests 200 \
        --concurrency 10 --base-url http://localhost:8080
"""
from __future__ import annotations

import argparse
import concurrent.futures
import os
import statistics
import sys
import time
from typing import Optional

from fastapi.testclient import TestClient

try:  # prefer a real server when a base URL is given
    import httpx  # type: ignore[import-untyped]
except Exception:  # pragma: no cover
    httpx = None

from tshepo.api import app  # noqa: E402

MESSAGE = "RK12AB34CD Confirmed. You have received M100.00 from THABO 58123456 on 19/10/26 at 2:15 PM. New M-PESA balance is M900.00."


def predict_once(base_url: Optional[str], client, headers: dict) -> tuple[float, int]:
    """One /predict call; returns (seconds, status_code)."""
    payload = {"body": MESSAGE, "sender": "MPESA", "model": "rules"}
    t0 = time.monotonic()
    if base_url:
        assert httpx is not None, "httpx required for remote load test"
        with httpx.Client(base_url=base_url, timeout=30.0) as remote:
            r = remote.post("/predict", json=payload, headers=headers)
    else:
        r = client.post("/predict", json=payload, headers=headers)
    return time.monotonic() - t0, r.status_code


def main() -> int:
    ap = argparse.ArgumentParser(description="Load-test the Tshepo /predict endpoint")
    ap.add_argument("--requests", type=int, default=200, help="total requests")
    ap.add_argument("--concurrency", type=int, default=1, help="concurrent clients")
    ap.add_argument(
        "--base-url", default=None, help="remote server base URL (default: in-process TestClient)"
    )
    ap.add_argument(
        "--api-key", default=os.environ.get("TSHEPO_API_KEY"), help="API key (or TSHEPO_API_KEY)"
    )
    a = ap.parse_args()

    headers = {}
    if a.api_key:
        headers["x-api-key"] = a.api_key

    timings: list[float] = []
    errors = 0
    start = time.monotonic()

    if a.base_url:
        if httpx is None:
            print("httpx is required for --base-url", file=sys.stderr)
            return 2
        with concurrent.futures.ThreadPoolExecutor(max_workers=a.concurrency) as ex:
            futs = [ex.submit(predict_once, a.base_url, None, headers) for _ in range(a.requests)]
            for f in concurrent.futures.as_completed(futs):
                dur, status = f.result()
                timings.append(dur)
                if status != 200:
                    errors += 1
    else:
        client = TestClient(app)
        with concurrent.futures.ThreadPoolExecutor(max_workers=a.concurrency) as ex:
            futs = [ex.submit(predict_once, None, client, headers) for _ in range(a.requests)]
            for f in concurrent.futures.as_completed(futs):
                dur, status = f.result()
                timings.append(dur)
                if status != 200:
                    errors += 1

    total = time.monotonic() - start
    timings.sort()
    p50 = statistics.median(timings) if timings else 0
    p95 = timings[int(len(timings) * 0.95) - 1] if timings else 0
    p99 = timings[int(len(timings) * 0.99) - 1] if timings else 0

    print(f"requests:     {a.requests}")
    print(f"concurrency:  {a.concurrency}")
    print(f"errors:       {errors}")
    print(f"total_s:      {total:.3f}")
    print(f"req_per_s:    {a.requests / total:.2f}")
    print(f"latency_ms:   avg={statistics.mean(timings)*1000:.1f} "
          f"p50={p50*1000:.1f} p95={p95*1000:.1f} p99={p99*1000:.1f}")
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
