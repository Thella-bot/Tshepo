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
              "real?"      rows        20 numbers     300-850 + reasons
```

### 1. Gate - "is this message a real transaction?"

The gate is the only text classifier in the system. It labels every message
`genuine_transaction`, `suspicious` or `noise`. Everything downstream only sees
`genuine_transaction` messages, so a scam SMS can never become income in the ledger.

Three interchangeable implementations, all behind the same one-function interface
(`label_fn(body, sender) -> label`, in `tshepo/models/gate.py`):

| implementation | where | why |
|---|---|---|
| rules | `parsing/parser.py` | zero dependencies, auditable, the baseline every model must beat |
| TF-IDF + logistic regression | `models/tfidf.py` | char n-grams survive typos; cheap enough for a server |
| hybrid | `models/tfidf.py` (`use_rules=True`) | same model, but the rule flags are fed in as extra tokens, so it can override a wrong rule |
| transformer | `models/transformer.py` | fine-tuned multilingual model; best on unseen wording, heaviest to serve |

The hybrid deserves a note. The rules and the model disagree in two directions:
rules fire on a PIN request that the model reads as a legitimate notice, and the model
fires on a URL the rules miss. Feeding the rule flags in as tokens (`FLAG_asks_secret`,
`RULESCORE4`, `PARSED_TXN`) lets the learned model weigh them against the text instead
of being bound by a hard threshold. In the unseen-scam-type test the plain TF-IDF model
recalled 91% of scam types it had never seen, and the hybrid recalled 100%.

### 2. Ledger - structure

`parsing/parser.py` pulls fields out of each accepted message: operator, direction,
category, amount, fee, balance, counterparty, reference, timestamp. Regex-based, because
operator formats are rigid and a regex is inspectable and cheap.

One subtlety: a message that the gate says is genuine but that the parser cannot read
as a transaction is dropped, not guessed at. A missing field is a data-quality signal,
not something to fabricate.

### 3. Features - cash flow

`features/cashflow.py` turns a user's ledger into 20 numbers over a rolling 13-week
window: income level and regularity, outflow ratio, savings rate, balance volatility
and floor, loan repayment habit, payer diversity and concentration, and a ledger
integrity share.

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
      -> build_features()        20 features per user
      -> fit score models        AUC / KS on held-out users
      -> forgery test            does the gate + integrity check hold?

## What is real and what is not

Real: the parser, the rule engine, the feature engineering, the scorecard, the
pipeline, the tests, the model interfaces.

Synthetic: the message templates, the Sesotho phrasing, the bank sender IDs, and the
repayment outcomes used to fit the scoring model. Numbers on synthetic data measure
the pipeline, not the product. The number that matters is the one on held-out real
messages, and the score that matters is the one fitted on real repayment outcomes.
