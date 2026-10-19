"""Fine-tune a small multilingual transformer for the 3-way SMS classifier.

    pip install torch transformers accelerate
    python -m tshepo.cli.train_transformer --model Davlan/afro-xlmr-base --users 1500 --epochs 3
    python -m tshepo.cli.train_transformer --real data/real_labelled.jsonl --model-dir models/transformer  # evaluate only

Model choice (verify coverage on each model card):
  * Davlan/afro-xlmr-base   - XLM-R adapted to African languages (reported to include Sesotho); first choice
  * xlm-roberta-base        - strong general multilingual fallback, thinner Sesotho coverage
  * distilbert-base-multilingual-cased - smaller/faster, runs on CPU
Evaluate against the TF-IDF/hybrid models with the SAME splits; only keep the transformer if it wins on
REAL messages or on the unseen-scam-type test, otherwise TF-IDF is cheaper and easier to deploy on USSD/SMS servers.

Hardware notes: on CPU use --model distilbert-base-multilingual-cased and a smaller --users
(it is minutes, not hours). fp16 is only used on CUDA; CPU runs in fp32.
"""
import argparse
import os
import random

import numpy as np

from ..config import LABELS, L2I, ensure_dir, model_path
from ..data.generator import build
from ..evaluation import group_split, load_jsonl, print_table, report
from ..textprep import heavy_noise


def build_training_args(a, cuda):
    """TrainingArguments renamed evaluation_strategy -> eval_strategy in transformers 4.46."""
    from transformers import TrainingArguments

    common = dict(
        output_dir=a.model_dir, num_train_epochs=a.epochs,
        per_device_train_batch_size=a.batch, per_device_eval_batch_size=64,
        learning_rate=a.lr, weight_decay=0.01, warmup_ratio=0.1, logging_steps=50,
        save_strategy="no", report_to=[], fp16=cuda, seed=a.seed,
        dataloader_num_workers=0,  # multiprocessing is fragile on Windows
        disable_tqdm=not a.progress,
    )
    try:
        return TrainingArguments(eval_strategy="no", **common)
    except TypeError:
        return TrainingArguments(evaluation_strategy="no", **common)


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
    ap.add_argument("--device", default=None, help="cuda | cpu (default: auto)")
    ap.add_argument("--progress", action="store_true", help="show the training progress bar")
    a = ap.parse_args()

    import torch
    from torch.utils.data import Dataset
    from transformers import (AutoModelForSequenceClassification, AutoTokenizer, Trainer)

    from ..textprep import prep

    device = a.device or ("cuda" if torch.cuda.is_available() else "cpu")
    cuda = device == "cuda"
    print(f"device: {device} | model: {a.model} | torch {torch.__version__}")

    class Sms(Dataset):
        def __init__(self, rows, tok):
            self.enc = tok([prep(r["body"], r["sender"]) for r in rows],
                           truncation=True, max_length=a.max_len)
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
    print(f"train {len(tr)} msgs / test {len(te)} msgs (split by user, no leakage)")

    src = a.model_dir if a.eval_only else a.model
    tok = AutoTokenizer.from_pretrained(src)
    model = AutoModelForSequenceClassification.from_pretrained(
        src, num_labels=len(LABELS), id2label=dict(enumerate(LABELS)), label2id=L2I)

    args = build_training_args(a, cuda)
    trainer = Trainer(model=model, args=args, train_dataset=None if a.eval_only else Sms(tr, tok))
    if not a.eval_only:
        trainer.train()
        ensure_dir(a.model_dir)
        trainer.save_model(a.model_dir)
        tok.save_pretrained(a.model_dir)
        print(f"saved transformer -> {model_path(a.model_dir)}")

    sets = {"synthetic test": te, "synthetic + heavy typos": noisy}
    if a.real:
        sets["REAL messages"] = load_jsonl(a.real)
    for name, ds in sets.items():
        y = [r["label"] for r in ds]
        print(f"\n== {name} (n={len(ds)}) ==")
        print_table([report(f"transformer ({a.model.split('/')[-1]})", y, predict(trainer, Sms(ds, tok)))])


if __name__ == "__main__":
    main()
