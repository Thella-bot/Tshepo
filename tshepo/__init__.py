"""Tshepo - SMS-based credit passport for mobile-money users (Lesotho).

Layers
------
data/      synthetic SMS generation (per-user chronologically coherent streams)
parsing/   regex field extraction + rule-based scam flags (the baseline)
features/  cash-flow features, ledger integrity check, transparent scorecard
models/    TF-IDF, hybrid and transformer classifiers + the gate loader
pipeline/  end-to-end: messages -> gate -> ledger -> features -> score
cli/       runnable entry points (generate, evaluate, train_*, predict)
"""
from __future__ import annotations

__version__ = "0.1.0"

__all__ = ["__version__"]
