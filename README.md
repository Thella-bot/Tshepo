# Tshepo - SMS generator + parser (starter)

Pure Python 3.9+, no dependencies.

    python generator.py --users 200 --seed 42 --out data/synthetic_sms.jsonl --csv
    python evaluate.py                      # rule-baseline numbers on a fresh synthetic set
    python sms_parser.py "RK12AB34CD Confirmed. You have received M100.00 from THABO MOKOENA 58123456 on 19/10/26 at 2:15 PM. New M-PESA balance is M900.00." --sender MPESA

## Files
- generator.py   per-user synthetic streams (genuine / suspicious / noise), running balances, typo and casing noise
- sms_parser.py  field extraction, scam red-flag rules, classify(), extract_transactions() ledger for the credit score
- evaluate.py    confusion matrix, scam recall, false-flag rate, field accuracy

## Added: classifiers and scoring
- common.py              shared prep, user-level split, noise stress test, metrics
- train_tfidf.py         TF-IDF (char+word) + logistic regression, and a hybrid that also sees the rule flags;
                         tests: synthetic, heavy typos, UNSEEN scam types, and `--real your_file.jsonl`
- train_transformer.py   fine-tunes Davlan/afro-xlmr-base (or xlm-roberta-base). NOT RUN in the authoring sandbox: smoke-test on Colab first
- ml_classifier.py       load a trained model as the gate: extract_transactions(records, label_fn=load_gate())
- cashflow.py            20 cash-flow features, balance-chain integrity check, transparent 300-850 scorecard, make_passport()
- train_score.py         SMS -> gate -> ledger -> features -> scorecard / logistic / boosting, plus a forged-SMS tamper test

    pip install scikit-learn numpy scipy joblib
    python train_tfidf.py && python train_score.py --gate models/hybrid.joblib

Real-data file format (one JSON per line): {"body": "...", "sender": "MPESA", "label": "genuine_transaction|suspicious|noise"}

## Be honest about the numbers
The templates only approximate real operator formats, so ~99% on synthetic data is optimistic
(the parser was written against the same templates). The remaining errors are mostly typos in
keywords like "balance", which is exactly what the ML classifier is for.

## Next steps
1. Add 50-200 REAL anonymised messages (your own + volunteers). Mask names, phone numbers, refs.
   Hold them out as the test set. Adjust templates and regexes to match real formats.
2. Have a Sesotho speaker review the scam/noise wording; add more code-switched variants.
3. Train a TF-IDF (char n-gram) + logistic regression baseline, then a small multilingual transformer.
   Report the lift over the rules on the real hold-out set.
4. Feed extract_transactions() output into the cash-flow features (income regularity, inflow/outflow
   ratio, balance volatility, loan repayments) and fit the scoring model.

## Scoring caveat
Default outcomes in train_score.py are SYNTHETIC (a hidden reliability variable drives both behaviour and outcome).
The AUC proves the pipeline runs end to end, not that the score predicts real defaults. Scorecard weights are hand-set.
To make it real: get anonymised repayment outcomes from a lender/SACCO partner and refit.
