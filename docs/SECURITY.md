# Security

## Threat model

| threat | mitigation |
|---|---|
| Malicious `.joblib` artefact → arbitrary code execution on load (pickle/RCE) | `tshepo.security.safe_model_path` confines loads to `models/`; only load artefacts you produced or audited; RCE warning in every loader |
| Path traversal through model path | every artefact path resolved under `models/`; relative names sandboxed, absolute paths passed through unchanged |
| Huge / malformed messages → ReDoS, memory exhaustion (DoS) | bodies capped at 4 KiB (`TSHEPO_MAX_BODY_LEN`), null/control bytes stripped, senders validated against an allow-list pattern |
| Flood of scoring calls → DoS | sliding-window rate limiter, default 120 calls/60s per key (`TSHEPO_RATE_MAX`, `TSHEPO_RATE_WINDOW`) |
| Batch too large → memory exhaustion | batches capped at 200k records (`TSHEPO_MAX_RECORDS_TOTAL`); per-user ledger capped at 5k rows |
| Unauthenticated access to scoring | optional API-key auth (`TSHEPO_API_KEY`, `X-API-Key` header) on `POST /predict` and `POST /score`; read endpoints stay open |
| Corrupt / slow model artefact takes down the service | `CircuitBreaker` trips after consecutive failures and falls back to the zero-dependency rule gate |
| Request body too large → abuse | middleware caps bodies at `TSHEPO_MAX_REQUEST_SIZE_MB` (default 10 MB) → `413` |
| Log leakage of PII | message bodies logged as sha256 digest + length only; raw bodies never written to logs |

## Authentication

Set `TSHEPO_API_KEY` (min 8 chars; enforced at startup) to require an `X-API-Key`
header on `POST /predict` and `POST /score`. Read endpoints (`/health`, `/version`,
`/metrics`) are always open.

```bash
export TSHEPO_API_KEY=<strong-key>
curl -H "X-API-Key: <strong-key>" http://localhost:8080/predict -d '{"body":"...","sender":"MPESA"}'
```

Missing or wrong key → `401`. Failed attempts are counted in the
`tshepo_auth_failures_total` Prometheus counter.

When `TSHEPO_API_KEY` is unset, all endpoints are open — appropriate only for local
development.

## Rate limiting

A sliding-window limiter caps scoring calls per key:

* default: 120 calls per 60 seconds per key
* tune with `TSHEPO_RATE_MAX` and `TSHEPO_RATE_WINDOW`
* exceeded → `429 {"code":"rate_limit_exceeded"}`

The limiter is keyed per caller; in the API the key is derived from the request
context. Use a reverse proxy (nginx, Cloudflare, etc.) in front of the API for
IP-based rate limiting in addition.

## Safe model loading

`joblib`/`pickle` can execute arbitrary code on load. Treat every artefact as
untrusted:

1. Never load a `.joblib` you did not produce yourself.
2. All loads go through `tshepo.security.load_model`, which confines the path to
   `models/` (`tshepo.security.safe_model_path`).
3. Relative model names are sandboxed under `models/`; path traversal (`../`) is
   rejected.
4. Audit artefacts before loading: re-save them through `tshepo.models.scoring`
   if you receive them from an untrusted source.

```python
from tshepo.security import load_model, SecurityError
from tshepo.models.scoring import ScoreModel

# safe: confined to models/
m = ScoreModel.load("models/score_gb.joblib")
```

## PII handling

* Message bodies are never written to logs in raw form. `tshepo.api` logs a sha256
  digest (`body_digest`) and `body_len` instead.
* Passports (`make_passport`) contain aggregates only — score, band, components,
  reasons, integrity share. No raw messages, names, phone numbers or references.
* Real user data lives in `data/` and is gitignored; `data/real_*.jsonl` and
  `data/real_*.csv` are explicitly excluded from git.

## Circuit breaker

The gate (`label_fn`) is wrapped in `CircuitBreaker` (failure_threshold=3,
cooldown=15s):

* `closed` — normal operation, primary gate is called
* `open` — tripped after consecutive failures; calls short-circuit to the rule gate
* `half-open` — after cooldown, a limited probe call is allowed; success closes the
  breaker, failure trips it again

State is exposed in `GET /health` under `circuit_breaker`. A corrupt model artefact
never takes the service down because the rule gate needs no artefact at all.

## Operational security

* Run the container as the non-root `tshepo` user (Dockerfile default).
* Mount `models/` and `data/` read-only.
* Put a reverse proxy (nginx, Cloudflare, ALB) in front for TLS termination, IP-based
  rate limiting and WAF rules.
* Restrict CORS via `TSHEPO_CORS_ORIGINS` if the API is browser-facing.
* Disable interactive docs in production (`TSHEPO_ENABLE_DOCS` unset or `0`).
* Rotate `TSHEPO_API_KEY` regularly.

## Security headers

Every response carries:

* `X-Content-Type-Options: nosniff`
* `X-Frame-Options: DENY`
* `Referrer-Policy: no-referrer`
* `Content-Security-Policy: default-src 'none'`

## Reporting security issues

Report vulnerabilities at https://github.com/Kilo-Org/kilocode/issues
