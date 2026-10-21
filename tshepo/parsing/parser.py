#!/usr/bin/env python3
"""
Tshepo - SMS parser + rule-based classifier (baseline).

Backward-compatible shim: parsing lives in :mod:`tshepo.parsing.fields` and the
scam rules in :mod:`tshepo.parsing.rules`. Import from those modules directly in
new code.

parse(body, sender)      -> Parsed   structured fields from a mobile-money / bank alert
red_flags(body, sender)  -> dict     scam indicators with weights
classify(body, sender)   -> dict     label in {genuine_transaction, suspicious, noise}
build_ledger(r)          -> Ledger   transactions + balance updates + failed transactions
extract_transactions(r)  -> list     clean ledger rows (genuine messages only)

CLI: python -m tshepo.cli.parse_sms "<message>" --sender MPESA
"""
from __future__ import annotations

from .fields import (  # noqa: F401
    KNOWN_SENDERS, SENDER_OP, SHORT_CODES, Parsed, parse,
)
from .rules import (  # noqa: F401
    THRESHOLD, WEIGHTS, build_ledger, classify, extract_transactions, red_flags,
)

__all__ = [
    "Parsed", "parse", "red_flags", "classify", "extract_transactions", "build_ledger",
    "KNOWN_SENDERS", "SENDER_OP", "SHORT_CODES", "THRESHOLD", "WEIGHTS",
]
