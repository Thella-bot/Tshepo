# Architecture

## The problem

A lender cannot assess someone with no payslip and no credit history. But the person
uses mobile money daily, so their phone already contains a record of their money:
what comes in, what goes out, how regular it is, how much buffer they keep.

Tshepo reads that record and turns it into a score. The user never installs anything -
they forward or grant read access to their SMS, or an agent does it for them.

## The four layers

```
  raw SMS  ──▶  GATE  ──▶  LEDGER  ──▶  FEATURES  ──▶  SCORE
               "real?"      rows        27 numbers     300-850 + reasons
```

### 1. Gate - "is this message a real transaction?"

The gate is the only text classifier in the system. It labels every message
`genuine_transaction`, `suspicious` or `noise`. Everything downstream only sees
`genuine_transaction` messages, so a scam SMS can never become income in the ledger.

Three interchangeable implementations, all behind the same one-function interface
(`label_fn(body, sender) -> label`, in `tshepo/models/gate.py`):

| implementation | where | why |
|---|---|---|
| rules | `parsing/rules.py` | zero dependencies, auditable, the baseline every model must beat |
| TF-IDF + logistic regression | `models/tfidf.py` | char n-grams survive typos; cheap enough for a server |
| hybrid | `models/tfidf.py` (`use_rules=True`) | same model, but the rule flags are fed in as extra tokens |
| transformer | `models/transformer.py` | fine-tuned multilingual model; best on unseen wording, heaviest to serve |

The hybrid deserves a note. The rules and the model disagree in two directions:
rules fire on a PIN request that the model reads as a legitimate notice, and the model
fires on a URL the rules miss. Feeding the rule flags in as tokens (`FLAG_asks_secret`,
`RULESCORE4`, `PARSED_TXN`) lets the learned model weigh them against the text instead
of being bound by a hard threshold. In the unseen-scam-type test the plain TF-IDF model
recalled 91% of scam types it had never seen, and the hybrid recalled 100%.

**Resilience**: in production the gate is wrapped in a circuit breaker
(`tshepo/resilience.py`). After consecutive failures it trips open and short-circuits
straight to the zero-dependency rule gate, so a corrupt or slow model artefact never
takes the whole service down. The breaker state is reported in `GET /health`.

### 2. Ledger - structure

`parsing/parser.py` pulls fields out of each accepted message: operator, direction,
category, amount, fee, balance, counterparty, reference, timestamp. Regex-based, because
operator formats are rigid and a regex is inspectable and cheap.

One subtlety: a message that the gate says is genuine but that the parser cannot read
as a transaction is dropped, not guessed at. A missing field is a data-quality signal,
not something to fabricate.

### 3. Features - cash flow

`features/cashflow.py` turns a user's ledger into 27 numbers over a rolling 13-week
window: income level and regularity, outflow ratio, savings rate, balance volatility
and floor, loan repayment habit, payer diversity and concentration, ledger integrity,
plus velocity/timing signals (txn per day, amount velocity, night/weekend share),
round-amount share, fee burden and day-of-week regularity.

**Ledger integrity** is the anti-fraud check: for each account, does the balance in
message *n* equal the balance in message *n-1* plus or minus the amount and fee? A
forged "you received M5,000" alert breaks the chain. The share of unbroken links is a
feature *and* a flag: below 0.9 the passport is marked for review.

### 4. Score

Two scorers over the same features:

* **Scorecard** (`features/cashflow.py`) - hand-set weights over six components
  (income regularity, cash surplus, balance buffer, repayment habit, customer base,
  history depth), each 0-1, mapped to 300-850. No fitting, so every point is explainable
  to the applicant: "your income arrives in only 3 of 13 weeks".
* **Fitted model** (`models/scoring.py`) - logistic regression or gradient boosting on
  the same feature vector, saved to `models/score_gb.joblib`. Better ranking, no
  explanation.

The passport (`make_passport`) is the shareable output: score, band, the components, the
reasons, and aggregates only. No raw messages, no names, no phone numbers.

## The anti-forgery property

The chain of custody matters: a scam message has to beat the gate to reach the ledger,
and if it does, it still has to survive the balance-chain check to influence the score.
`train_score.py` tests this directly by forging "you received M1,500-3,000" alerts into
held-out users' histories and comparing scores with and without the gate.

## Data flow of the training pipeline

    generator.build()            synthetic per-user streams, balances chained, labels attached
      -> group_split()           split by user, never by message
      -> classifier.fit()        TF-IDF / transformer
      -> report()                accuracy, scam recall, false-flag rate
    generator.build(credit_signal=True)
      -> extract_transactions()  gate + parser
      -> build_features()        27 features per user
      -> fit score models        AUC / KS on held-out users
      -> forgery test            does the gate + integrity check hold?

## The fraud problem, stated honestly

The applicant holds the phone. Every message Tshepo reads arrives through the
applicant, and the pipeline has no way to tell an operator's alert from a
well-written forgery. That is a property of the product, not a bug: the whole
point is that no bank integration is needed.

The measured consequence, from the red-team harness in `tests/test_antifraud.py`:
a fabricated 90-day history that is internally consistent reaches **849 of the
maximum 850**, passes every data-quality gate (balance chain 100%, coverage 99%,
no unread messages, no gap over 24 hours) and is issued as `status: scored` over
the API with nothing but a self-issued consent token. The honest EcoCash export
scores 523 on the same pipeline.

So the design answer is not "detect the forgery" - that is not possible from
inside the data. It is three things:

1. **Never let a number stand alone.** Every passport carries
   `verification: sms_unverified` and a `fraud_signals` block, so the lender
   sees what the number rests on.
2. **Catch the attacks that leave a trace.** Some manoeuvres cannot be
   perfectly imitated because they depend on relationships rather than on
   wording: money that goes out to the same accounts it came from, a history
   used by two applicants, a message whose timestamp disagrees with the export,
   a repayment that never recurs.
3. **Ask for provenance the applicant cannot produce.** The signal that
   forgery cannot reach is an independent one.

## Detection signals, and what each one is worth

| signal | where | catches | defeated by |
|---|---|---|---|
| counterparty graph | `features/graph.py` | money that arrives from an account and goes back to the same account - circular flow, a laundering pattern that needs no forgery at all | paying through intermediate accounts the applicant does not control |
| wallet shape | `features/cashflow.py` | a wallet where nearly every transaction is incoming | adding synthetic outgoing transactions |
| repayment corroboration | `features/cashflow.py` | a repayment habit that does not recur over months or never changes amount | spacing the forgeries across months with different amounts |
| timestamp cross-check | `parsing/rules.py` | a message edited after the export was written | editing both timestamps together |
| message fingerprints | `antiabuse.py`, `store.py` | one history submitted by two applicants | changing one message in the copy |
| balance-chain continuity | `features/cashflow.py` | messages deleted from the middle of the history | deleting only messages that did not move the balance |
| gate (rules / TF-IDF / hybrid) | `parsing/rules.py` | a forgery that does not look like the operator's template | copying the operator's real template |

Thresholds are set clear of the real population on purpose: on the real EcoCash
and M-Pesa exports and on 200 synthetic users the circular share never exceeded
0.15 and the incoming share never exceeded 0.62, while a circular-flow attack
measures 1.00 and a fabrication 0.86. The signals are review triggers, not
score changes, so a false positive costs a human's time and never a decision.

## Provenance without a bank integration

Without a bank, the service cannot prove a message is genuine. It can only make
the applicant's claim harder to fake, and it can ask for evidence the applicant
cannot manufacture alone:

* **A live challenge through the operator.** The service sends a one-time code
  to the applicant's wallet through the operator's USSD/app channel and asks
  for it in the portal. The applicant cannot pre-compute the code, so a forged
  history no longer proves control of a phone; it proves control of a message
  inbox. This needs an operator partnership, not a bank.
* **A signed export.** Some operators already put an export signature or a
  verification code in the export. Where it exists, verify it; where it does
  not, record that it does not.
* **Two-channel agreement.** Require the export to be corroborated by something
  the applicant cannot edit in one place: the phone number of the sender for
  every message, a second device or a second operator for the same period, or
  an employer/payroll statement for the income the score credits.
* **Time-boxed consent and an upload token.** A consent that is bound to the
  applicant's declared phone number, expires, and can only be used once, removes
  the "score this history for many people" attack and the replay attack.
* **Device attestation where the platform offers it.** Play Integrity and
  Apple App Attest attest that the app running on the phone is the real one
  and that the SMS database was read in place, not pasted. It does not prove
  the messages are genuine, but it removes the "edit the database" attack
  without requiring the applicant to install anything unusual.

## What is real and what is not

Real: the parser, the rule engine, the feature engineering, the scorecard, the
pipeline, the tests, the model interfaces.

Synthetic: the message templates, the Sesotho phrasing, the bank sender IDs, and the
repayment outcomes used to fit the scoring model. Numbers on synthetic data measure
the pipeline, not the product. The number that matters is the one on held-out real
messages, and the score that matters is the one fitted on real repayment outcomes.

## Security & scalability

**Security**
* `tshepo.security` validates every message: bodies capped at 4 KiB, null/control bytes
  stripped, senders validated against an allow-list pattern.
* Model artefacts are loaded only from `models/`; relative model names are path-confined
  to prevent traversal. `joblib`/`pickle` can execute arbitrary code on load — only load
  artefacts you produced or audited.
* A sliding-window rate limiter caps scoring calls per key (default 120/min).
* Batches are capped at 200k records; per-user ledgers at 5k rows.
* `tshepo.resilience.CircuitBreaker` trips after consecutive model failures and falls
  back to the rule gate; state is exposed in `GET /health`.

**Scalability**
* `score_batch` processes users in bounded chunks with per-user ledger capping and
  progress logging, so memory stays flat as the user count grows.
* `ScoreModel.predict_proba_batch` vectorises predictions across many feature dicts.
* Regexes are compiled once at module load (no per-message recompilation).
* `CircuitBreaker` prevents a repeatedly-failing model from hammering the service.

**Health checks**
* `tshepo.health.check_all` verifies the parser, features, scorecard and models dir
  before deploying or in a readiness probe. `GET /health` also reports circuit-breaker
  state and whether API-key auth is configured.
