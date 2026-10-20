# Deployment guide

## Quick start

```bash
# install with API extras
pip install -e ".[api]"

# run with uvicorn (dev)
TSHEPO_API_KEY=<strong-key> python -m tshepo.cli.serve --port 8080

# or with Docker
docker build -t tshepo .
docker run -e TSHEPO_API_KEY=<strong-key> -p 8080:8080 tshepo

# verify
curl http://localhost:8080/health
curl http://localhost:8080/metrics
```

## Configuration

All settings are environment variables (or a `tshepo.yaml`/`tshepo.json` config file in
the cwd or repo root). Env vars take precedence.

| variable | default | description |
|---|---|---|
| `TSHEPO_LOG_LEVEL` | `INFO` | logging level |
| `TSHEPO_JSON_LOGS` | `0` | set `1` for JSON-structured logs |
| `TSHEPO_MAX_BODY_LEN` | `4096` | max SMS body length in characters |
| `TSHEPO_MAX_SENDER_LEN` | `64` | max sender id length |
| `TSHEPO_MAX_RECORDS_PER_USER` | `5000` | ledger rows considered per user |
| `TSHEPO_MAX_RECORDS_TOTAL` | `200000` | max records in a single batch call |
| `TSHEPO_RATE_WINDOW` | `60` | rate-limit window in seconds |
| `TSHEPO_RATE_MAX` | `120` | scoring calls per window per key |
| `TSHEPO_API_KEY` | *(unset)* | require `X-API-Key` on write endpoints |
| `TSHEPO_ENABLE_DOCS` | `0` | set `1` to enable `/docs`/`/redoc` (not for prod) |
| `TSHEPO_CORS_ORIGINS` | *(unset)* | comma-separated allowed CORS origins |
| `TSHEPO_MAX_REQUEST_SIZE_MB` | `10` | max request body size in MB |

Create a `.env` from `.env.example` and adjust:

```bash
cp .env.example .env
# edit .env with your values
```

## Docker

### Build and run

```bash
docker build -t tshepo .
docker run --rm -e TSHEPO_API_KEY=<strong-key> -p 8080:8080 tshepo
```

The image runs as a non-root user (`tshepo`, uid 10001). Trained artefacts in
`models/` and data in `data/` are mounted read-only so they cannot be modified at
runtime.

A `HEALTHCHECK` hits `GET /health` every 30s; the container is marked unhealthy if it
fails 3 times.

### docker-compose (local development)

```bash
docker-compose up --build
```

This starts the `tshepo` service on port 8080. The compose file also defines a
`tshepo-lint` service that runs `pyflakes`, `mypy` and `pytest` inside the container
— useful for pre-merge checks without a local Python environment.

## Health checks

`GET /health` returns `200` only when all subsystems are healthy:

```json
{
  "parser": {"ok": true, ...},
  "features": {"ok": true, ...},
  "scorecard": {"ok": true, ...},
  "models_dir": {"ok": true, ...},
  "circuit_breaker": "closed",
  "api_configured": true,
  "ok": true
}
```

Use it as a Kubernetes readiness/liveness probe:

```yaml
readinessProbe:
  httpGet:
    path: /health
    port: 8080
  initialDelaySeconds: 5
  periodSeconds: 10
```

## Observability

### Prometheus metrics

`GET /metrics` exposes:

* `tshepo_http_requests_total` — method, endpoint, status
* `tshepo_http_request_duration_seconds` — method, endpoint
* `tshepo_scoring_total` — outcome
* `tshepo_predict_total` — model
* `tshepo_auth_failures_total`

Scrape with Prometheus and alert on, e.g., `tshepo_http_requests_total{status=~"5.."}`.

### Structured logs

Set `TSHEPO_JSON_LOGS=1` for JSON lines ingestible by your log collector. Every log
record includes `level`, `logger`, `msg`, `time` plus request-scoped fields
(`correlation_id`, `user_id`, `score`, `duration_ms`) when available.

Correlation IDs flow through the whole request: pass `X-Correlation-ID` on inbound
requests, or one is generated. It appears in every log line and response header.

## Circuit breaker

The gate is wrapped in a `CircuitBreaker` (3 failures → open, 15s cooldown → half-open
probe). While open, calls short-circuit to the zero-dependency rule gate. State is
reported in `GET /health` under `circuit_breaker` (`closed` / `open` / `half-open`).
This means a corrupt model artefact never takes the service down.

## CI/CD

`.github/workflows/ci.yml` runs on every push and PR:

1. install `.[dev,api]`
2. `pyflakes tshepo tests`
3. `mypy tshepo/`
4. `pytest tests -q`
5. on push to `main`: build the Docker image

Pre-commit hooks (`.pre-commit-config.yaml`) run pyflakes, mypy and pytest on every
commit.

## Production checklist

- [ ] Set `TSHEPO_API_KEY` to a strong value (min 8 chars; enforced at startup)
- [ ] Set `TSHEPO_JSON_LOGS=1` for structured logging
- [ ] Mount `models/` and `data/` read-only in the container
- [ ] Use the non-root `tshepo` user (Dockerfile does this by default)
- [ ] Configure a readiness probe on `GET /health`
- [ ] Scrape `GET /metrics` with Prometheus
- [ ] Disable interactive docs (`TSHEPO_ENABLE_DOCS` unset or `0`) in production
- [ ] Restrict CORS origins via `TSHEPO_CORS_ORIGINS` if the API is browser-facing
- [ ] Never commit real SMS data: `data/real_*` is gitignoted

## Load testing

```bash
python scripts/load_test.py --requests 500 --concurrency 20
```

Against a running server:

```bash
TSHEPO_API_KEY=<key> python scripts/load_test.py --requests 500 \
  --concurrency 20 --base-url http://localhost:8080
```
