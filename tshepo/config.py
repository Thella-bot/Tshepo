"""Paths and shared settings.

Everything the project writes goes under the repository root:
    data/    generated or hand-labelled message files (never commit real data)
    models/  trained artefacts (tfidf.joblib, hybrid.joblib, transformer/, score models)
"""
from __future__ import annotations

import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_DIR = os.path.join(ROOT, "data")
MODELS_DIR = os.path.join(ROOT, "models")
DOCS_DIR = os.path.join(ROOT, "docs")

LABELS = ["genuine_transaction", "suspicious", "noise"]
L2I = {l: i for i, l in enumerate(LABELS)}


def ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path


def model_path(name: str) -> str:
    """models/<name> unless an absolute path is given."""
    return name if os.path.isabs(name) else os.path.join(MODELS_DIR, name)
