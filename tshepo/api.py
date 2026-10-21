"""HTTP API for Tshepo.

Production service entry point. Run with:

    uvicorn tshepo.api:app --host 0.0.0.0 --port 8080

or via the CLI:
    python -m tshepo.cli.serve --port 8080

Endpoints
---------
GET  /health        - production readiness probe (no auth)
GET  /metrics       - Prometheus metrics (no auth)
GET  /              - the capture portal (no auth, serves the web app)
POST /predict       - classify a single message (API key required)
POST /score         - score one or more users (API key required)

Consent-gated capture and retrieval. These are the endpoints the applicant
uses, and they are what makes "no app" true in practice: the applicant opens
a page in the browser on their phone, reads the consent notice, sends their
export, and reads their own passport.

GET    /v1/consent           - the consent notice and the policy
POST   /v1/consents          - record consent, returns a one-time token
GET    /v1/consents/{id}     - consent status
DELETE /v1/consents/{id}     - withdraw consent and erase everything
POST   /v1/ingest            - submit an export (file or records) under a consent
POST   /v1/messages          - submit a single forwarded message
POST   /v1/score             - score the messages held for a consent
GET    /v1/passports/{user}  - fetch a passport
DELETE /v1/users/{user}/data - erase everything held for a user
POST   /v1/maintenance/purge - retention purge (API key)
GET    /v1/audit             - the access log (API key)

Auth
----
Operator endpoints use ``X-API-Key``. Applicant endpoints use
``X-Consent-Id`` plus ``X-Consent-Token``, so a holder of a consent can read
and erase their own data without an operator in the loop.
"""
from __future__ import annotations

import hashlib
import json
import os
import time
from typing import Any, Callable, Optional, cast

from fastapi import Depends, FastAPI, Header, HTTPException, Request, Response, Security
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse, PlainTextResponse
from fastapi.security import APIKeyHeader
from pydantic import BaseModel, Field, validator

from .config import (ALLOW_UNCALIBRATED_MODEL, API_KEY, CORS_ORIGINS, ENABLE_DOCS,
                     ENABLE_PORTAL, MAX_REQUEST_SIZE_MB, RETENTION_DAYS, WITHHOLD_SCORE,
                     model_path)
from .consent import consent_summary
from .health import check_all
from .importers import detect_format, parse_export
from .logging import get_logger, log_context, new_correlation_id
from .models.gate import load_gate, load_transformer_gate, rule_gate
from .pipeline import score_batch, score_user
from .resilience import CircuitBreaker, ModelError, fallback_gate
from .security import RateLimitExceeded, validate_message, validate_records
from .store import Store, get_store

logger = get_logger("api")

try:
    from . import __version__  # noqa: WPS433
except ImportError:  # pragma: no cover
    __version__ = "0.0.0"

app = FastAPI(
    title="Tshepo SMS Credit Passport",
    version="0.3.0",
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
# The portal is same-origin only: it loads its own script and stylesheet and
# talks to this service, and nothing else.
PORTAL_CSP = ("default-src 'none'; script-src 'self'; style-src 'self'; "
              "connect-src 'self'; img-src 'self' data:; base-uri 'none'; frame-ancestors 'none'")
PORTAL_PATHS = ("/", "/portal", "/portal.js", "/portal.css")


@app.middleware("http")
async def _security_headers(request: Request, call_next: Callable[[Request], Any]) -> Any:
    response = await call_next(request)
    for k, v in SECURITY_HEADERS.items():
        response.headers[k] = v
    if request.url.path in PORTAL_PATHS:
        response.headers["Content-Security-Policy"] = PORTAL_CSP
    return response


# --------------------------------------------------------------------------- CORS
if CORS_ORIGINS:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=CORS_ORIGINS,
        allow_credentials=True,
        allow_methods=["GET", "POST", "DELETE"],
        allow_headers=["content-type", "x-api-key", "x-correlation-id", "x-consent-id", "x-consent-token"],
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


def _store() -> Store:
    return get_store()


def require_consent(x_consent_id: Optional[str] = Header(None),
                    x_consent_token: Optional[str] = Header(None)) -> str:
    """Authenticate the applicant: a live consent plus its token.

    The consent must exist, the token must match, and the consent must not be
    revoked or expired. Nothing else is accepted.
    """
    if not x_consent_id or not x_consent_token:
        raise HTTPException(status_code=401, detail="X-Consent-Id and X-Consent-Token are required")
    store = _store()
    if not store.consent_token_ok(x_consent_id, x_consent_token):
        if AUTH_FAILURES is not None:
            AUTH_FAILURES.inc()
        raise HTTPException(status_code=401, detail="invalid consent token")
    if not store.consent_is_live(x_consent_id):
        raise HTTPException(status_code=403, detail="consent has expired or been withdrawn")
    return x_consent_id


def operator_or_consent(
    x_api_key: Optional[str] = Header(None),
    x_consent_id: Optional[str] = Header(None),
    x_consent_token: Optional[str] = Header(None),
) -> str:
    """Either the operator's API key, or a live consent token. Returns the actor."""
    if API_KEY is not None and x_api_key is not None and x_api_key == API_KEY:
        return "operator"
    consent_id = require_consent(x_consent_id, x_consent_token)
    return f"consent:{consent_id}"


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
    CONSENT_COUNT = Counter("tshepo_consents_total", "Consents granted", ["source"])
    ERASURE_COUNT = Counter("tshepo_erasures_total", "Data sets erased", ["scope"])
except Exception:  # pragma: no cover - prometheus_client optional at import
    CONTENT_TYPE_LATEST = "text/plain"
    REQUEST_COUNT = REQUEST_DURATION = SCORING_COUNT = PREDICT_COUNT = None
    AUTH_FAILURES = CONSENT_COUNT = ERASURE_COUNT = None


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
    origin: Optional[str] = Field(None, description="received | sent")

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


class ConsentRequest(BaseModel):
    user_id: str = Field(..., min_length=1, max_length=128)
    subject_ref: Optional[str] = Field(None, max_length=128,
                                       description="Phone number or reference for the applicant")
    purpose: str = Field("credit_assessment", max_length=64)
    source: Optional[str] = Field(None, max_length=64,
                                   description="Where the consent was captured: portal, agent, api")
    scope: Optional[str] = Field(None, max_length=256)


class IngestRequest(BaseModel):
    consent_id: Optional[str] = Field(None, description="Consent the messages belong to")
    user_id: Optional[str] = Field(None, description="User id (defaults to the consent's user)")
    export_text: Optional[str] = Field(None, description="The raw export file, as text")
    export_format: Optional[str] = Field(None, description="google_txt | samsung_xml | google_json | csv | tsv")
    messages: Optional[list[MessageInput]] = Field(None, description="Already-parsed messages")
    label: Optional[str] = Field(None, description="Ground-truth label (for evaluation datasets)")


# --------------------------------------------------------------------------- startup validation
@app.on_event("startup")
def _validate_startup() -> None:
    """Fail fast if the runtime configuration is unusable."""
    problems: list[str] = []
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
    report["portal_enabled"] = ENABLE_PORTAL
    status = 200 if report.get("ok") else 503
    return JSONResponse(content=report, status_code=status)


@app.post("/predict", response_model=PredictResponse)
async def predict(
    req: PredictRequest, request: Request, _auth: None = Depends(verify_api_key)
) -> PredictResponse:
    """Classify a single message."""
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


def _scoring_response(passports: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Shape the passports for a caller, withholding the score when the data
    does not support a decision."""
    out: dict[str, Any] = {}
    for uid, p in passports.items():
        item = dict(p)
        if WITHHOLD_SCORE and item.get("decision") != "score":
            item["score"] = None
            item["score_withheld"] = True
        out[uid] = item
    return out


def _load_pd_model(name: str) -> Optional[Any]:
    """Load a fitted score model, refusing one fitted on synthetic outcomes.

    The shipped model is fitted on synthetic labels. Serving it as if it
    predicted real default would be the single most damaging thing this
    service could do, so it is refused unless an operator opts in.
    """
    from .models.scoring import ScoreModel
    try:
        model = ScoreModel.load(model_path(name))
    except Exception as e:  # noqa: BLE001
        logger.warning("pd model load failed, scoring without it", extra={"error": str(e)})
        return None
    outcomes = str(model.metadata.get("outcomes", "")).lower()
    if outcomes == "synthetic" and not ALLOW_UNCALIBRATED_MODEL:
        logger.warning("refusing uncalibrated score model",
                       extra={"model": name, "outcomes": outcomes})
        raise HTTPException(
            status_code=409,
            detail="the requested score model was fitted on synthetic outcomes and is not "
                   "calibrated on real repayment data; set TSHEPO_ALLOW_UNCALIBRATED_MODEL=1 "
                   "to serve it anyway")
    return model


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
            pd_model = _load_pd_model(req.pd_model)

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
                       "band": passport.get("band"), "gate": req.gate or "rules",
                       "status": passport.get("status"), "decision": passport.get("decision")},
            )

        return {"correlation_id": cid, "users": len(passports),
                "passports": _scoring_response(passports)}


# --------------------------------------------------------------------- consent-gated capture
@app.get("/v1/consent")
async def consent_notice() -> dict[str, Any]:
    """The consent notice, the retention policy and the withdrawal route."""
    return consent_summary()


@app.post("/v1/consents", status_code=201)
async def create_consent(req: ConsentRequest) -> dict[str, Any]:
    """Record the applicant's consent. Returns a one-time token.

    The token is shown once. The applicant keeps it; it is what lets them
    send their messages, read their passport and erase their data without
    asking an operator.
    """
    store = _store()
    consent = store.create_consent(user_id=req.user_id, purpose=req.purpose,
                                   subject_ref=req.subject_ref, scope=req.scope,
                                   source=req.source)
    store.log_access("applicant", "consent_granted", req.user_id, f"consent={consent['id']}")
    if CONSENT_COUNT is not None:
        CONSENT_COUNT.labels(source=req.source or "unknown").inc()
    return {
        "consent_id": consent["id"],
        "consent_token": consent["token"],
        "user_id": consent["user_id"],
        "granted_at": consent["granted_at"],
        "expires_at": consent["expires_at"],
        "purpose": consent["purpose"],
        "notice": consent_summary(),
    }


@app.get("/v1/consents/{consent_id}")
async def consent_status(consent_id: str, _actor: str = Depends(operator_or_consent)) -> dict[str, Any]:
    """The state of a consent: granted, withdrawn, expired."""
    store = _store()
    c = store.get_consent(consent_id)
    if not c:
        raise HTTPException(status_code=404, detail="unknown consent")
    return {
        "consent_id": c["id"], "user_id": c["user_id"], "purpose": c["purpose"],
        "granted_at": c["granted_at"], "expires_at": c["expires_at"],
        "revoked_at": c["revoked_at"], "live": store.consent_is_live(consent_id),
        "messages_held": store.message_count(c["user_id"]),
    }


@app.delete("/v1/consents/{consent_id}")
async def withdraw_consent(consent_id: str, _actor: str = Depends(operator_or_consent)) -> dict[str, Any]:
    """Withdraw consent and erase everything collected under it."""
    store = _store()
    result = store.revoke_and_erase(consent_id)
    if not result.get("ok"):
        raise HTTPException(status_code=404, detail="unknown consent")
    if ERASURE_COUNT is not None:
        ERASURE_COUNT.labels(scope="consent").inc()
    return result


@app.post("/v1/ingest")
async def ingest(req: IngestRequest, consent_id: str = Depends(require_consent)) -> dict[str, Any]:
    """Accept an export, or a set of messages, under a live consent.

    The applicant sends the file as text (the portal reads it in the browser
    and posts the text), or posts messages one at a time. Nothing is stored
    without a live consent.
    """
    store = _store()
    consent = store.get_consent(consent_id)
    if not consent:
        raise HTTPException(status_code=404, detail="unknown consent")
    user_id = req.user_id or str(consent["user_id"])

    records: list[dict[str, Any]] = []
    detected = None
    if req.export_text:
        detected = req.export_format or detect_format(req.export_text[:4096], "export.txt")
        records = parse_export(req.export_text, fmt=req.export_format)
    if req.messages:
        for m in req.messages:
            body, sender = validate_message(m.body, m.sender)
            records.append({"body": body, "sender": sender, "received_at": m.received_at,
                            "origin": m.origin or "received"})
    if not records:
        raise HTTPException(status_code=400,
                            detail="nothing to ingest: send export_text or messages")

    stored = store.insert_messages(consent_id, user_id, records)
    store.log_access(f"consent:{consent_id}", "ingest", user_id,
                      f"messages={stored} format={detected or 'json'}")
    return {
        "consent_id": consent_id, "user_id": user_id, "messages_received": len(records),
        "messages_stored": stored, "format": detected or "json",
    }


@app.post("/v1/messages", status_code=201)
async def ingest_message(msg: MessageInput, consent_id: str = Depends(require_consent)) -> dict[str, Any]:
    """Accept a single forwarded message under a live consent."""
    store = _store()
    consent = store.get_consent(consent_id)
    if not consent:
        raise HTTPException(status_code=404, detail="unknown consent")
    body, sender = validate_message(msg.body, msg.sender)
    store.insert_messages(consent_id, str(consent["user_id"]),
                          [{"body": body, "sender": sender, "received_at": msg.received_at}])
    store.log_access(f"consent:{consent_id}", "ingest_message", str(consent["user_id"]), None)
    return {"stored": 1, "consent_id": consent_id, "user_id": str(consent["user_id"])}


@app.post("/v1/score")
async def score_consent(consent_id: str, request: Request,
                         _actor: str = Depends(operator_or_consent)) -> dict[str, Any]:
    """Score the messages held under a consent, without the messages leaving
    the server a second time."""
    store = _store()
    consent = store.get_consent(consent_id)
    if not consent:
        raise HTTPException(status_code=404, detail="unknown consent")
    user_id = str(consent["user_id"])
    rows = store.messages_for(user_id)
    if not rows:
        raise HTTPException(status_code=409, detail="no messages have been submitted under this consent")
    records = [{"body": r["body"], "sender": r["sender"], "received_at": r["received_at"],
                "user_id": user_id, "origin": "received"} for r in rows]
    records = [r for r in records if r.get("origin") != "sent"]
    gate = fallback_gate(_get_gate(None), breaker=_GATE_BREAKER)
    passport = score_user(records, label_fn=gate, user_label=user_id)
    store.put_passport(user_id, consent_id, passport)
    store.log_access(f"consent:{consent_id}", "score", user_id, None)
    if SCORING_COUNT is not None:
        SCORING_COUNT.labels(outcome="ok").inc()
    out = dict(passport)
    if WITHHOLD_SCORE and out.get("decision") != "score":
        out["score"] = None
        out["score_withheld"] = True
    return out


@app.get("/v1/passports/{user_id}")
async def get_passport(user_id: str, _actor: str = Depends(operator_or_consent)) -> dict[str, Any]:
    """Fetch the passport issued for a user."""
    store = _store()
    p = store.get_passport(user_id)
    if not p:
        raise HTTPException(status_code=404, detail="no passport for this user")
    store.log_access("actor", "read_passport", user_id, None)
    out = dict(p)
    if WITHHOLD_SCORE and out.get("decision") != "score":
        out["score"] = None
        out["score_withheld"] = True
    return out


@app.delete("/v1/users/{user_id}/data")
async def erase_user(user_id: str, _actor: str = Depends(operator_or_consent)) -> dict[str, Any]:
    """Erase every message and passport held for a user."""
    store = _store()
    counts = store.delete_user_data(user_id)
    store.log_access("actor", "erase_user", user_id, None)
    if ERASURE_COUNT is not None:
        ERASURE_COUNT.labels(scope="user").inc()
    return {"user_id": user_id, **counts}


@app.post("/v1/maintenance/purge")
async def purge(_auth: None = Depends(verify_api_key)) -> dict[str, Any]:
    """Run the retention purge: delete data whose consent expired, was
    withdrawn, or aged past the retention window."""
    store = _store()
    result = store.purge_expired()
    store.log_access("operator", "purge", None, str(result))
    return result


@app.get("/v1/audit")
async def audit(limit: int = 100, _auth: None = Depends(verify_api_key)) -> dict[str, Any]:
    """The access log: who looked at whose data, and when."""
    return {"entries": _store().access_log(limit)}


# --------------------------------------------------------------------- capture portal
_WEB_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "web")


def _web_file(name: str) -> str:
    return os.path.join(_WEB_DIR, name)


@app.get("/", response_class=HTMLResponse)
async def portal() -> HTMLResponse:
    """The capture portal.

    It is a plain page: it shows the consent notice, takes the export file
    the applicant already has on their phone, and posts it to this service.
    No app to install, no bank integration, no third party holding the file.
    """
    if not ENABLE_PORTAL:
        raise HTTPException(status_code=404, detail="the capture portal is disabled")
    if not os.path.isfile(_web_file("index.html")):
        raise HTTPException(status_code=404, detail="the capture portal is not installed")
    with open(_web_file("index.html"), encoding="utf-8") as fh:
        return HTMLResponse(fh.read())


@app.get("/portal.js")
async def portal_js() -> Response:
    """The portal's script, served from this origin so the CSP can forbid inline code."""
    if not os.path.isfile(_web_file("portal.js")):
        raise HTTPException(status_code=404, detail="not found")
    with open(_web_file("portal.js"), encoding="utf-8") as fh:
        return Response(fh.read(), media_type="application/javascript")


@app.get("/portal.css")
async def portal_css() -> Response:
    """The portal's stylesheet."""
    if not os.path.isfile(_web_file("portal.css")):
        raise HTTPException(status_code=404, detail="not found")
    with open(_web_file("portal.css"), encoding="utf-8") as fh:
        return Response(fh.read(), media_type="text/css")


def classify_msg(body: str, sender: Optional[str]) -> dict[str, Any]:
    from .parsing.parser import classify
    return cast(dict[str, Any], classify(body, sender))


@app.exception_handler(RateLimitExceeded)
async def _rate_limit_handler(request: Request, exc: RateLimitExceeded) -> JSONResponse:
    return JSONResponse(status_code=429, content={"detail": str(exc), "code": "rate_limit_exceeded"})


@app.exception_handler(ModelError)
async def _model_error_handler(request: Request, exc: ModelError) -> JSONResponse:
    return JSONResponse(status_code=502, content={"detail": str(exc), "code": "model_error"})
