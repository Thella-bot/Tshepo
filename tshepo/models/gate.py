"""The gate: whichever classifier you trust sits between the raw messages and the ledger.

    from tshepo.models.gate import load_gate
    from tshepo.parsing.parser import extract_transactions
    rows = extract_transactions(records, label_fn=load_gate("models/hybrid.joblib"))

A gate is a callable label_fn(body, sender) -> "genuine_transaction" | "suspicious" | "noise".
Rule-based gates need no artefact at all (see tshepo.parsing.parser.classify).
"""
from __future__ import annotations

from ..logging import get_logger
from ..parsing.parser import classify
from ..security import load_model as _safe_load

logger = get_logger("models.gate")


def rule_gate(body, sender=None):
    """The zero-dependency baseline gate."""
    return classify(body, sender)["label"]


def load_gate(path="models/hybrid.joblib"):
    """Load a trained classifier and wrap it in the label_fn interface.

    Uses the path-confined safe loader (see tshepo.security). Only load artefacts
    you have produced yourself or have audited.
    """
    model = _safe_load(path)
    logger.info("gate loaded", extra={"model": "gate", "path": path})
    return lambda body, sender=None: model.predict([body], [sender])[0]


def load_transformer_gate(path="models/transformer"):
    """Load the fine-tuned transformer as a gate (requires torch + transformers)."""
    from .transformer import TransformerSmsClassifier

    return TransformerSmsClassifier.load(path).predict_label
