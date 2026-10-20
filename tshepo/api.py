"""HTTP API for Tshepo.

Production service entry point. Run with:

    uvicorn tshepo.api:app --host 0.0.0.0 --port 8080

or via the CLI:

    python -m tshepo.cli.serve --port 8080

Endpoints
---------
GET  /health        - production readiness probe (no auth)
GET  /metrics       - Prometheus metrics (no auth)
POST /predict       - classify a single message (API key required)
POST /score         - score one or more users (credit passports, API key required)

Auth
----
Set ``TSHEPO_API_KEY`` to require ``X-API-Key`` header on write endpoints.
"""
from __future__ import annotations

import hashlib
import time
from typing import Any, Callable, Optional, cast

from fastapi import Depends, FastAPI, HTTPException, Request, Response, Security
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, PlainTextResponse
from fastapi.security import APIKeyHeader
from pydantic import BaseModel, Field, validator

from .config import API_KEY, CORS_ORIGINS, ENABLE_DOCS, MAX_REQUEST_SIZE_MB, model_path
from .health import check_all
from .logging import get_logger, log_context, new_correlation_id
from .models.gate import load_gate, load_transformer_gate, rule_gate
from .pipeline import score_batch
from .resilience import CircuitBreaker, ModelError, fallback_gate
from .security import RateLimitExceeded, validate_message, validate_records

logger = get_logger("api")

try:
    from . import __version__  # noqa: WPS433
except ImportError:  # pragma: no cover
    __version__ = "0.0.0"

app = FastAPI(
    title="Tshepo SMS Credit Passport",
    version="0.2.0",
    docs_url="/docs" if ENABLE_DOCS else None,
    redoc_url="/redoc" if ENABLE_DOCS else None,
    openapi_url="/openapi.json" if ENABLE_DOCS else None,
)

# --------------------------------------------------------------------------- security headers
SECURITY_HEADERS = {
    "X-Content-Type-Options": "nosniff",
    "X-Frame-Options": "DENY",
    "Referrer-Policy": "no-referrer",
    "Content-Security-Policy": "default-src 'none'",
}


@app.middleware("http")
async def _security_headers(request: Request, call_next: Callable[[Request], Any]) -> Any:
    response = await call_next(request)
    for k, v in SECURITY_HEADERS.items():
        response.headers[k] = v
    return response


# --------------------------------------------------------------------------- CORS
if CORS_ORIGINS:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=CORS_ORIGINS,
        allow_credentials=True,
        allow_methods=["GET", "POST"],
        allow_headers=["content-type", "x-api-key", "x-correlation-id"],
        max_age=600,
    )


# --------------------------------------------------------------------------- request size limit
MAX_REQUEST_SIZE = int(MAX_REQUEST_SIZE_MB * 1024 * 1024)


@app.middleware("http")
async def _limit_request_size(request: Request, call_next: Callable[[Request], Any]) -> Any:
    content_length = request.headers.get("content-length")
    if content_length is not None and int(content_length) > MAX_REQUEST_SIZE:
        return JSONResponse(
            status_code=413,
            content={"detail": f"request body too large (max {MAX_REQUEST_SIZE_MB} MB)", "code": "request_too_large"},
        )
    return await call_next(request)

# --------------------------------------------------------------------------- auth
api_key_header = APIKeyHeader(name="x-api-key", auto_error=False)


def verify_api_key(api_key: Optional[str] = Security(api_key_header)) -> None:
    """Require a valid API key when ``TSHEPO_API_KEY`` is configured."""
    if API_KEY is None:
        return  # auth not configured; leave endpoints open
    if api_key is None or api_key != API_KEY:
        if AUTH_FAILURES is not None:
            AUTH_FAILURES.inc()
        raise HTTPException(status_code=401, detail="invalid or missing API key")


# --------------------------------------------------------------------------- Prometheus metrics
try:
    from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest

    REQUEST_COUNT = Counter(
        "tshepo_http_requests_total", "Total HTTP requests", ["method", "endpoint", "status"]
    )
    REQUEST_DURATION = Histogram(
        "tshepo_http_request_duration_seconds", "HTTP request duration", ["method", "endpoint"]
    )
    SCORING_COUNT = Counter("tshepo_scoring_total", "Total users scored", ["outcome"])
    PREDICT_COUNT = Counter("tshepo_predict_total", "Total predictions", ["model"])
    AUTH_FAILURES = Counter("tshepo_auth_failures_total", "Total failed auth attempts")
except Exception:  # pragma: no cover - prometheus_client optional at import
    CONTENT_TYPE_LATEST = "text/plain"
    REQUEST_COUNT = REQUEST_DURATION = SCORING_COUNT = PREDICT_COUNT = AUTH_FAILURES = None


def _observe(method: str, endpoint: str, status: int, duration: float) -> None:
    if REQUEST_COUNT is not None:
        REQUEST_COUNT.labels(method=method, endpoint=endpoint, status=status).inc()
    if REQUEST_DURATION is not None:
        REQUEST_DURATION.labels(method=method, endpoint=endpoint).observe(duration)

_GATE_CACHE: dict[str, Any] = {}


def _get_gate(model: Optional[str]) -> Any:
    """Resolve and cache a gate by model name/path."""
    if model in (None, "", "rules"):
        return rule_gate
    if model in _GATE_CACHE:
        return _GATE_CACHE[model]
    try:
        if model == "transformer":
            gate = load_transformer_gate()
        else:
            gate = load_gate(model_path(model if model.endswith(".joblib") else f"{model}.joblib"))
        _GATE_CACHE[model] = gate
        return gate
    except Exception as e:  # noqa: BLE001
        logger.warning("gate load failed, using rule gate", extra={"model": model, "error": str(e)})
        return rule_gate


# per-request circuit breaker so a repeatedly-failing model degrades fast
_GATE_BREAKER = CircuitBreaker(failure_threshold=3, cooldown_seconds=15.0)


# --------------------------------------------------------------------------- request / response models
class MessageInput(BaseModel):
    body: str = Field(..., min_length=1, max_length=4096, description="SMS message body")
    sender: Optional[str] = Field(None, max_length=64, description="Message sender id")
    user_id: Optional[str] = Field(None, description="User / applicant id")
    received_at: Optional[str] = Field(None, description="ISO-8601 timestamp")
    label: Optional[str] = Field(None, description="Ground-truth label (for evaluation)")

    @validator("body")
    def _strip_body(cls, v: str) -> str:
        return v.replace("\x00", "").strip()


class PredictRequest(BaseModel):
    body: str = Field(..., min_length=1, max_length=4096)
    sender: Optional[str] = None
    model: str = Field("rules", description="rules | tfidf | hybrid | transformer")


class PredictResponse(BaseModel):
    label: str
    rule_label: Optional[str] = None
    model: str
    correlation_id: str


class ScoreRequest(BaseModel):
    messages: list[MessageInput] = Field(..., max_items=200000)
    gate: Optional[str] = Field(None, description="joblib gate model path/name")
    pd_model: Optional[str] = Field(None, description="fitted score model path/name")
    correlation_id: Optional[str] = None


# --------------------------------------------------------------------------- startup validation
@app.on_event("startup")
def _validate_startup() -> None:
    """Fail fast if the runtime configuration is unusable."""
    problems: list[str] = []
    import os

    from .config import MODELS_DIR
    if not os.path.isdir(MODELS_DIR):
        problems.append(f"models dir missing: {MODELS_DIR}")
    if API_KEY is not None and len(API_KEY) < 8:
        problems.append("TSHEPO_API_KEY is too short (min 8 chars)")
    if problems:
        logger.error("startup validation failed", extra={"problems": problems})
        raise RuntimeError("; ".join(problems))
    logger.info("startup validation passed")


# --------------------------------------------------------------------------- middleware: correlation id + timing
@app.middleware("http")
async def _trace(request: Request, call_next: Callable[[Request], Any]) -> Any:
    cid = request.headers.get("x-correlation-id") or new_correlation_id()
    with log_context(correlation_id=cid):
        t0 = time.monotonic()
        try:
            response = await call_next(request)
        except Exception:
            logger.exception("unhandled request error", extra={"correlation_id": cid, "path": request.url.path})
            raise
        dur_ms = (time.monotonic() - t0) * 1000
        _observe(request.method, request.url.path, response.status_code, dur_ms / 1000)
        response.headers["x-correlation-id"] = cid
        response.headers["x-response-time-ms"] = str(round(dur_ms, 1))
        logger.info(
            "request",
            extra={"correlation_id": cid, "path": request.url.path,
                   "method": request.method, "status": response.status_code, "dur_ms": round(dur_ms, 1)},
        )
        return response


# --------------------------------------------------------------------------- endpoints
@app.get("/metrics")
async def metrics() -> PlainTextResponse:
    """Prometheus metrics endpoint."""
    return PlainTextResponse(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.get("/version")
async def version() -> dict[str, str]:
    """Service version and build info."""
    return {"version": __version__, "service": "tshepo"}


@app.get("/health")
async def health() -> Response:
    """Readiness probe: returns ok=true only when all subsystems are healthy."""
    report = check_all()
    report["circuit_breaker"] = _GATE_BREAKER.state
    report["api_configured"] = API_KEY is not None
    status = 200 if report.get("ok") else 503
    return JSONResponse(content=report, status_code=status)


@app.post("/predict", response_model=PredictResponse)
async def predict(
    req: PredictRequest, request: Request, _auth: None = Depends(verify_api_key)
) -> PredictResponse:
    """Classify a single message."""
    if AUTH_FAILURES is not None:
        pass  # auth failures counted in the dependency
    cid = request.headers.get("x-correlation-id") or new_correlation_id()
    body, sender = validate_message(req.body, req.sender)
    gate = _get_gate(req.model)
    rule_res = classify_msg(body, sender)
    label = gate(body, sender)
    if PREDICT_COUNT is not None:
        PREDICT_COUNT.labels(model=req.model).inc()
    body_digest = hashlib.sha256(body.encode("utf-8")).hexdigest()[:16]
    logger.info(
        "prediction",
        extra={
            "correlation_id": cid,
            "model": req.model,
            "label": label,
            "rule_label": rule_res["label"],
            "body_len": len(body),
            "body_digest": body_digest,
        },
    )
    return PredictResponse(
        label=label,
        rule_label=rule_res["label"],
        model=req.model,
        correlation_id=cid,
    )


@app.post("/score")
async def score(
    req: ScoreRequest, request: Request, _auth: None = Depends(verify_api_key)
) -> dict[str, Any]:
    """Score users and return credit passports.

    Input is a list of message records (body/sender/user_id/received_at).
    Output is {user_id: passport}.
    """
    cid = req.correlation_id or request.headers.get("x-correlation-id") or new_correlation_id()
    with log_context(correlation_id=cid):
        try:
            records = validate_records([m.model_dump() for m in req.messages])
        except RateLimitExceeded as e:
            raise HTTPException(status_code=429, detail=str(e))
        except Exception as e:  # noqa: BLE001
            raise HTTPException(status_code=400, detail=f"invalid input: {e}")

        gate = fallback_gate(_get_gate(req.gate), breaker=_GATE_BREAKER)
        pd_model = None
        if req.pd_model:
            from .models.scoring import ScoreModel
            try:
                pd_model = ScoreModel.load(model_path(req.pd_model))
            except Exception as e:  # noqa: BLE001
                logger.warning("pd model load failed, scoring without it", extra={"error": str(e), "correlation_id": cid})

        try:
            passports = score_batch(
                records,
                label_fn=gate,
                pd_model=pd_model,
                correlation_id_value=cid,
            )
        except ModelError as e:
            raise HTTPException(status_code=502, detail=str(e))
        except RateLimitExceeded as e:
            raise HTTPException(status_code=429, detail=str(e))

        if SCORING_COUNT is not None:
            SCORING_COUNT.labels(outcome="ok").inc(len(passports))

        # audit log: one line per scored user
        for uid, passport in passports.items():
            logger.info(
                "scoring decision",
                extra={"correlation_id": cid, "user_id": uid, "score": passport.get("score"),
                       "band": passport.get("band"), "gate": req.gate or "rules"},
            )

        return {"correlation_id": cid, "users": len(passports), "passports": passports}


def classify_msg(body: str, sender: Optional[str]) -> dict[str, Any]:
    from .parsing.parser import classify
    return cast(dict[str, Any], classify(body, sender))


@app.exception_handler(RateLimitExceeded)
async def _rate_limit_handler(request: Request, exc: RateLimitExceeded) -> JSONResponse:
    return JSONResponse(status_code=429, content={"detail": str(exc), "code": "rate_limit_exceeded"})


@app.exception_handler(ModelError)
async def _model_error_handler(request: Request, exc: ModelError) -> JSONResponse:
    return JSONResponse(status_code=502, content={"detail": str(exc), "code": "model_error"})
