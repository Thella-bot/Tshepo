"""Transformer SMS classifier (fine-tuned sequence classification).

Wraps a Hugging Face checkpoint so it can be used exactly like the TF-IDF model:
    clf = TransformerSmsClassifier.load("models/transformer")
    clf.predict(bodies, senders)  -> list of labels

The transformer is optional: torch/transformers are only needed if you actually use
this model. The TF-IDF model is cheaper and is the default for a USSD/SMS server.
"""
from __future__ import annotations

import os

import torch

from ..config import LABELS, L2I, model_path
from ..logging import get_logger
from ..security import _confine
from ..textprep import prep

logger = get_logger("models.transformer")


class TransformerSmsClassifier:
    """Thin wrapper: tokenise -> model -> labels."""

    def __init__(self, model=None, tokenizer=None, device="cpu"):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device

    @classmethod
    def load(cls, path="models/transformer", device=None):
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        path = _confine(path, os.path.dirname(model_path(path)))
        if not os.path.isdir(path):
            raise FileNotFoundError(
                f"no transformer at {path}. Train one: python -m tshepo.cli.train_transformer")
        device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        tok = AutoTokenizer.from_pretrained(path)
        model = AutoModelForSequenceClassification.from_pretrained(path)
        model.to(device).eval()
        logger.info("transformer loaded", extra={"model": "transformer", "path": path,
                                                   "device": device})
        return cls(model=model, tokenizer=tok, device=device)

    def predict_proba(self, bodies, senders, batch_size=32):
        import torch

        probs = []
        texts = [prep(b, s) for b, s in zip(bodies, senders)]
        for i in range(0, len(texts), batch_size):
            chunk = texts[i:i + batch_size]
            enc = self.tokenizer(chunk, truncation=True, max_length=96, padding=True,
                                 return_tensors="pt").to(self.device)
            with torch.no_grad():
                logits = self.model(**enc).logits
            probs.extend(torch.softmax(logits, dim=-1).cpu().tolist())
        return probs, list(self.model.config.id2label.values())

    def predict(self, bodies, senders, batch_size=32):
        probs, classes = self.predict_proba(bodies, senders, batch_size=batch_size)
        return [classes[max(range(len(p)), key=lambda i: p[i])] for p in probs]

    def predict_label(self, body, sender=None):
        """label_fn(body, sender) interface, for use as a gate."""
        return self.predict([body], [sender])[0]


__all__ = ["TransformerSmsClassifier", "LABELS", "L2I"]
