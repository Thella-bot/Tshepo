#!/usr/bin/env python3
"""
Fine-tune a small multilingual transformer for the 3-way SMS classifier.

NOT RUN in the authoring sandbox (no GPU, and Hugging Face downloads were blocked there), so treat
the first run as a smoke test. Recommended: Google Colab (free T4) or any machine with a GPU.

    pip install torch transformers accelerate scikit-learn
    python train_transformer.py --model Davlan/afro-xlmr-base --users 1500 --epochs 3
    python train_transformer.py --real data/real_labelled.jsonl --model-dir models/transformer   # evaluate only

Model choice (verify coverage on each model card):
  * Davlan/afro-xlmr-base   - XLM-R adapted to African languages (reported to include Sesotho); first choice
  * xlm-roberta-base        - strong general multilingual fallback, thinner Sesotho coverage
  * distilbert-base-multilingual-cased - smaller/faster if you need CPU inference
Evaluate against the TF-IDF/hybrid models with the SAME splits; only keep the transformer if it wins on
REAL messages or on the unseen-scam-type test, otherwise TF-IDF is cheaper and easier to deploy on USSD/SMS servers.
"""
import argparse
import os
import random

import numpy as np

from common import L2I, LABELS, group_split, heavy_noise, load_jsonl, prep, print_table, report
from generator import build


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Davlan/afro-xlmr-base")
    ap.add_argument("--users", type=int, default=1500)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--max-len", type=int, default=96)
    ap.add_argument("--real", default=None)
    ap.add_argument("--model-dir", default="models/transformer")
    ap.add_argument("--eval-only", action="store_true")
    a = ap.parse_args()

    import torch
    from torch.utils.data import Dataset
    from transformers import (AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments)

    class Sms(Dataset):
        def __init__(self, rows, tok):
            self.enc = tok([prep(r["body"], r["sender"]) for r in rows], truncation=True, max_length=a.max_len)
            self.y = [L2I[r["label"]] for r in rows]

        def __len__(self):
            return len(self.y)

        def __getitem__(self, i):
            d = {k: v[i] for k, v in self.enc.items()}
            d["labels"] = self.y[i]
            return d

    def predict(trainer, ds):
        return [LABELS[i] for i in trainer.predict(ds).predictions.argmax(-1)]

    recs, _ = build(n_users=a.users, seed=a.seed)
    tr, te = group_split(recs, 0.2, seed=a.seed)
    rng = random.Random(123)
    noisy = [dict(r, body=heavy_noise(rng, r["body"])) for r in te]

    src = a.model_dir if a.eval_only else a.model
    tok = AutoTokenizer.from_pretrained(src)
    model = AutoModelForSequenceClassification.from_pretrained(
        src, num_labels=len(LABELS), id2label=dict(enumerate(LABELS)), label2id=L2I)

    args = TrainingArguments(
        output_dir=a.model_dir, num_train_epochs=a.epochs, per_device_train_batch_size=a.batch,
        per_device_eval_batch_size=64, learning_rate=a.lr, weight_decay=0.01, warmup_ratio=0.1,
        logging_steps=50, save_strategy="no", report_to=[], fp16=torch.cuda.is_available(), seed=a.seed)
    trainer = Trainer(model=model, args=args, train_dataset=None if a.eval_only else Sms(tr, tok))
    if not a.eval_only:
        trainer.train()
        trainer.save_model(a.model_dir)
        tok.save_pretrained(a.model_dir)

    sets = {"synthetic test": te, "synthetic + heavy typos": noisy}
    if a.real:
        sets["REAL messages"] = load_jsonl(a.real)
    for name, ds in sets.items():
        y = [r["label"] for r in ds]
        print(f"\n== {name} (n={len(ds)}) ==")
        print_table([report(f"transformer ({a.model.split('/')[-1]})", y, predict(trainer, Sms(ds, tok)))])


if __name__ == "__main__":
    main()
