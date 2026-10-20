# HTTP API reference

The Tshepo HTTP API is a small FastAPI service. Run it with:

    pip install -e ".[api]"
    TSHEPO_API_KEY=<strong-key> python -m tshepo.cli.serve --port 8080

or with Docker:

    docker run -e TSHEPO_API_KEY=<strong-key> -p 8080:8080 tshepo

## Authentication

When `TSHEPO_API_KEY` is set, all write endpoints (`POST /predict`, `POST /score`)
require an `X-API-Key` header. Read endpoints (`GET /health`, `GET /version`,
`GET /metrics`) are always open.

    curl -H "X-API-Key: <strong-key>" \
      -H "Content-Type: application/json" \
      -d '{"body":"RK12AB34CD Confirmed. You have received M100.00 from THABO on 19/10/26 at 2:15 PM.","sender":"MPESA","model":"rules"}' \
      http://localhost:8080/predict

Missing or wrong key → `401 {"detail":"invalid or missing API key"}`.

## Endpoints

### GET /health — readiness probe

Returns `ok=true` only when every subsystem (parser, features, scorecard, models dir)
is healthy. Also reports circuit-breaker state and whether API-key auth is configured.

**Example response** (200):

```json
{
  "parser": {"ok": true, "ms": 0.6, "detail": "genuine_transaction"},
  "features": {"ok": true, "ms": 1.2, "detail": "n_features=27"},
  "scorecard": {"ok": true, "ms": 0.4, "detail": "score=620"},
  "models_dir": {"ok": true, "ms": 0.1, "detail": "3 joblib artefact(s)"},
  "circuit_breaker": "closed",
  "api_configured": true,
  "ok": true
}
```

Returns `503` when any check fails. Suitable for Kubernetes liveness/readiness probes.

### GET /version — service info

Returns the service name and version.

**Example response**:

```json
{"version": "0.2.0", "service": "tshepo"}
```

### GET /metrics — Prometheus metrics

Plain-text Prometheus exposition format. Counters and histograms:

* `tshepo_http_requests_total` (labels: method, endpoint, status)
* `tshepo_http_request_duration_seconds` (labels: method, endpoint)
* `tshepo_scoring_total` (labels: outcome)
* `tshepo_predict_total` (labels: model)
* `tshepo_auth_failures_total` (total failed auth attempts)

**Content-Type**: `text/plain; version=0.0.4; charset=utf-8`

### POST /predict — classify a single message

Classify one SMS as `genuine_transaction`, `suspicious` or `noise`.

**Request**:

| field | type | required | description |
|---|---|---|---|
| body | string | yes | SMS body (max 4096 chars) |
| sender | string | no | Sender id (max 64 chars) |
| model | string | no | `rules` \| `tfidf` \| `hybrid` \| `transformer` (default `rules`) |

**Example request**:

```json
{
  "body": "RK12AB34CD Confirmed. You have received M100.00 from THABO 58123456 on 19/10/26 at 2:15 PM. New M-PESA balance is M900.00.",
  "sender": "MPESA",
  "model": "rules"
}
```

**Example response** (200):

```json
{
  "label": "genuine_transaction",
  "rule_label": "genuine_transaction",
  "model": "rules",
  "correlation_id": "a1b2c3d4e5f6"
}
```

The response includes `X-Correlation-ID` and `X-Response-Time-Ms` headers.

**Errors**:
* `400` — invalid input (e.g. body too long, bad characters)
* `401` — missing or wrong API key (when configured)
* `429` — rate limit exceeded
* `413` — request body too large (> `TSHEPO_MAX_REQUEST_SIZE_MB`, default 10 MB)

### POST /score — score users into credit passports

Score many users at once. Input is a list of message records; output is one passport
per user.

**Request**:

| field | type | required | description |
|---|---|---|---|
| messages | array | yes | Message records (max 200,000) |
| gate | string | no | joblib gate model path/name (default `rules`) |
| pd_model | string | no | fitted score model path/name (optional) |
| correlation_id | string | no | request-scoped correlation id for tracing |

Each message record:

| field | type | required | description |
|---|---|---|---|
| body | string | yes | SMS body |
| sender | string | no | Sender id |
| user_id | string | no | User / applicant id |
| received_at | string | no | ISO-8601 timestamp |
| label | string | no | Ground-truth label (for evaluation) |

**Example request**:

```json
{
  "messages": [
    {
      "body": "RK12AB34CD Confirmed. You have received M100.00 from THABO 58123456 on 19/10/26 at 2:15 PM. New balance is M900.00.",
      "sender": "MPESA",
      "user_id": "U1",
      "received_at": "2026-10-19T14:15:00"
    },
    {
      "body": "Your PIN is required to claim your prize. Call 0800SCAM now.",
      "sender": "UNKNOWN",
      "user_id": "U1",
      "received_at": "2026-10-19T14:20:00"
    }
  ]
}
```

**Example response** (200):

```json
{
  "correlation_id": "a1b2c3d4e5f6",
  "users": 1,
  "passports": {
    "U1": {
      "applicant": "U1",
      "score": 640,
      "band": "Fair",
      "model_pd_score": null,
      "history_days": 1,
      "transactions_used": 1,
      "monthly_income_estimate": 100.0,
      "income_active_weeks": "0/13",
      "avg_balance": 900.0,
      "loan_repayments_seen": 0,
      "components": {...},
      "reasons_down": [...],
      "reasons_up": [...],
      "data_integrity": "consistent",
      "integrity_share": 1.0,
      "notes": "Built from consented SMS alerts. Scorecard weights are provisional until calibrated on repayment outcomes."
    }
  }
}
```

Passports contain aggregates only — no raw messages, names or phone numbers — so they
are safe to share with a lending partner.

**Errors**:
* `400` — invalid input (batch too large, bad record shape)
* `401` — missing or wrong API key (when configured)
* `429` — rate limit exceeded
* `413` — request body too large
* `502` — model error (gate failed and no fallback available)

## Rate limiting

A sliding-window limiter caps scoring calls per key (default 120 per 60 seconds per
key). Exceeding it returns `429`. Tune with `TSHEPO_RATE_MAX` and `TSHEPO_RATE_WINDOW`.

## Headers

Every response carries:

* `X-Content-Type-Options: nosniff`
* `X-Frame-Options: DENY`
* `Referrer-Policy: no-referrer`
* `Content-Security-Policy: default-src 'none'`
* `X-Correlation-ID` (echoed from request, or auto-generated)
* `X-Response-Time-Ms`

## Error envelope

All errors return a JSON body:

```json
{"detail": "human-readable message", "code": "error_code"}
```

Error codes: `rate_limit_exceeded`, `model_error`, `request_too_large`,
`invalid or missing API key`.

## Disabling the interactive docs

`GET /docs`, `GET /redoc` and `GET /openapi.json` are disabled by default. Set
`TSHEPO_ENABLE_DOCS=1` to enable (not recommended in production).
