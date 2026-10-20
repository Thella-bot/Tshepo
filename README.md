# Tshepo - SMS credit passport

Builds a credit score for someone who has no formal credit history, using only the
mobile-money and bank SMS alerts they already receive. No app, no data sharing, no
bank integration: the user's phone is the data source.

The pipeline is four layers, and each layer can be replaced independently:

```
SMS messages  ->  GATE (is this a real transaction?)  ->  LEDGER (structured rows)
              ->  CASH-FLOW FEATURES  ->  SCORE (300-850) + credit passport
```

The gate is the only part that classifies text, and it has three interchangeable
implementations: hand-written rules, TF-IDF + logistic regression (with an optional
hybrid that also reads the rule flags), and a fine-tuned multilingual transformer.

## Quick start

    pip install -r requirements.txt
    pip install -e .                     # optional: gives you the tshepo-* commands

    python -m tshepo.cli.generate --users 200 --seed 42 --out data/synthetic_sms.jsonl --csv
    python -m tshepo.cli.evaluate       # rule-baseline numbers on a fresh synthetic set
    python -m tshepo.cli.parse_sms "RK12AB34CD Confirmed. You have received M100.00 ..." --sender MPESA

## Project structure

    tshepo/
      config.py            paths, label set
      logging.py           structured logging (JSON/text)
      security.py          input validation, safe model loading, rate limiting
      health.py            production health checks
      textprep.py          model input prep (sender tag + digit normalisation)
      evaluation.py        user-level split, metrics
      pipeline.py          end-to-end core: messages -> gate -> ledger -> features -> score
      data/
        generator.py       synthetic SMS streams (genuine / suspicious / noise) with chained balances
      parsing/
        fields.py          regex field extraction (Parsed, parse)
        rules.py           scam red flags + rule classifier (red_flags, classify)
        parser.py          compatibility shim (re-exports fields + rules)
      features/
        cashflow.py        27 cash-flow features, balance-chain integrity, transparent scorecard
      models/
        tfidf.py           TF-IDF (char+word) + logistic regression, and the hybrid variant
        transformer.py     fine-tuned multilingual transformer, same interface as tfidf
        scoring.py         fitted score models (logistic regression, gradient boosting)
        gate.py            the gate interface: label_fn(body, sender) -> label
      cli/
        generate.py        write a synthetic dataset
        evaluate.py        rule baseline report
        parse_sms.py       parse one SMS
        import_sms.py      import a phone export (txt -> JSONL)
        predict.py         classify messages with rules / tfidf / hybrid / transformer
        passport.py        build a credit passport per user from a message file
        train_tfidf.py     train the TF-IDF and hybrid classifiers
        train_transformer.py  fine-tune the transformer
        train_score.py     train the cash-flow scoring model end to end
    tests/                 pytest suite (parser, generator, features, pipeline, security, health)
    data/                  generated or hand-labelled message files (gitignored)
    models/                trained artefacts (gitignored)
    docs/                  architecture notes

## Training the models

    pip install -r requirements.txt
    python -m tshepo.cli.train_tfidf                     # -> models/tfidf.joblib, models/hybrid.joblib
    python -m tshepo.cli.train_score --gate models/hybrid.joblib   # -> models/score_gb.joblib

The TF-IDF script also runs two stress tests: a heavy-typo version of the test set,
and a set of scam types that were never in training (to see if the model generalises
or only memorised).

### Transformer (optional, needs torch + transformers)

    pip install torch transformers accelerate
    # GPU (Colab T4 or better):
    python -m tshepo.cli.train_transformer --model Davlan/afro-xlmr-base --users 1500 --epochs 3
    # CPU (small model, fewer users - minutes, not hours):
    python -m tshepo.cli.train_transformer --model distilbert-base-multilingual-cased \
        --users 300 --epochs 1 --batch 16 --device cpu

Model choices: `Davlan/afro-xlmr-base` (XLM-R adapted to African languages, reported
to include Sesotho) is the first choice; `xlm-roberta-base` is the general fallback;
`distilbert-base-multilingual-cased` is the CPU-friendly one.

## Using a model as the gate

    from tshepo.models.gate import load_gate
    from tshepo.pipeline import score_user

    label_fn = load_gate("models/hybrid.joblib")
    passport = score_user(messages, label_fn=label_fn, user_label="U0042")

or from the command line:

    python -m tshepo.cli.predict "Your PIN is required to..." --sender +26658123456 --model hybrid
    python -m tshepo.cli.passport --file data/synthetic_sms.jsonl --gate models/hybrid.joblib

## Security hardening

* **Safe model loading**: artefacts are loaded only from `models/` (path traversal is
  blocked). `joblib`/`pickle` can execute arbitrary code on load — only load artefacts
  you produced or audited.
* **Input validation**: message bodies are capped at 4 KiB, null/control bytes are
  stripped, and senders are validated against an allow-list pattern.
* **Rate limiting**: a sliding-window limiter caps scoring calls per key (default 120/min).
* **Circuit breaker**: the gate trips after consecutive model failures and falls back to
  the zero-dependency rule gate; state is reported in `/health`.
* **API key auth**: set `TSHEPO_API_KEY` to require `X-API-Key` on `/predict` and `/score`.
* **Bounds**: batches are capped at 200k records; per-user ledgers at 5k rows.

    from tshepo.security import validate_message, SecurityError
    body, sender = validate_message(body, sender)   # raises on oversized / malformed input

## Health checks

Run before deploying or in a readiness probe:

    from tshepo.health import check_all
    report = check_all()   # {"parser": {...}, "features": {...}, "scorecard": {...},
                           #  "models_dir": {...}, "ok": bool}

## Running the HTTP API

Serve the API with uvicorn:

    pip install -e ".[api]"
    TSHEPO_API_KEY=<strong-key> python -m tshepo.cli.serve --port 8080

Or with Docker:

    docker build -t tshepo .
    docker run -e TSHEPO_API_KEY=<strong-key> -p 8080:8080 tshepo

Endpoints:

* `GET /health` — readiness probe (no auth). Returns `ok=true` only when all
  subsystems are healthy, plus circuit-breaker state.
* `GET /metrics` — Prometheus metrics (no auth).
* `POST /predict` — classify one message (API key required when
  `TSHEPO_API_KEY` is set).
* `POST /score` — score many users into credit passports (API key required).
* `GET /docs`, `GET /redoc` — disabled by default; set `TSHEPO_ENABLE_DOCS=1` to
  enable (not recommended in production).

Security headers (`X-Content-Type-Options`, `X-Frame-Options`, `CSP`) are set on
every response. Request bodies are capped (`TSHEPO_MAX_REQUEST_SIZE_MB`, default
10 MB). CORS is disabled unless `TSHEPO_CORS_ORIGINS` is set.

    # classify one message
    curl -H "X-API-Key: <strong-key>" -H "Content-Type: application/json" \
      -d '{"body":"RK12AB34CD Confirmed. You have received M100.00 ...","sender":"MPESA","model":"rules"}' \
      http://localhost:8080/predict

    # score many users
    curl -H "X-API-Key: <strong-key>" -H "Content-Type: application/json" \
      -d '{"messages":[{"body":"...","sender":"MPESA","user_id":"U1"}]}' \
      http://localhost:8080/score

## Circuit breaker

The gate uses a circuit breaker (`tshepo.resilience.CircuitBreaker`) that trips
after consecutive model failures and falls back to the zero-dependency rule gate.
State is reported in `/health` under `circuit_breaker` (`closed`/`open`/`half-open`).

## Load testing

    python scripts/load_test.py --requests 500 --concurrency 20

## Batch scoring

Score many users with bounded memory (per-user ledger capped, progress logged):

    from tshepo.pipeline import score_batch
    passports = score_batch(records, label_fn=load_gate("models/hybrid.joblib"))

## Model versioning

Fitted scoring models carry `metadata` (kind, n_train, base rate, version, created_at)
and expose `feature_importances()` so scoring decisions stay explainable:

    from tshepo.models.scoring import ScoreModel
    m = ScoreModel.load("models/score_gb.joblib")
    print(m.metadata["version"])            # "1.0"
    print(m.feature_importances())          # per-feature importance magnitudes

## Real-data file format

One JSON per line:

    {"body": "...", "sender": "MPESA", "label": "genuine_transaction|suspicious|noise"}

Then:

    python -m tshepo.cli.train_tfidf --real data/real_labelled.jsonl
    python -m tshepo.cli.train_transformer --real data/real_labelled.jsonl

## Be honest about the numbers

The templates only approximate real operator formats, so ~99% on synthetic data is
optimistic (the parser was written against the same templates). The remaining errors
are mostly typos in keywords like "balance", which is exactly what the ML classifier
is for. The real number that matters is the one on held-out REAL messages.

## Next steps

1. Add 50-200 REAL anonymised messages (your own + volunteers). Mask names, phone
   numbers, refs. Hold them out as the test set. Adjust templates and regexes to
   match real formats.
2. Have a Sesotho speaker review the scam/noise wording; add more code-switched variants.
3. Report the lift of TF-IDF/transformer over the rules on the real hold-out set.
4. Get anonymised repayment outcomes from a lender/SACCO partner and refit the
   scoring model on them.

## Scoring caveat

Default outcomes in the training data are SYNTHETIC (a hidden reliability variable
drives both behaviour and outcome). The AUC proves the pipeline runs end to end, not
that the score predicts real defaults. Scorecard weights are hand-set.

See `docs/ARCHITECTURE.md` for how the layers fit together.

## Real data (phone exports)

Real operator messages don't match the synthetic templates exactly. Import a phone export
and the parser adapts to the real formats (short-code senders, amount spacing, merchant IDs,
failed transactions, etc.):

    python -m tshepo.cli.import_sms data/real_export.txt --out data/real_messages.jsonl
    python -m tshepo.cli.train_tfidf --real data/real_labelled.jsonl   # label first!

Real data goes in `data/` and is gitignored — never commit messages that contain names,
phone numbers or references.

On the real EcoCash/M-Pesa sample imported here, the rule parser extracts 98/119 EcoCash
and 14/29 M-Pesa messages as genuine transactions. The balance-chain integrity check then
flags incomplete exports (missing messages break the chain) for review instead of scoring
them on partial data.
