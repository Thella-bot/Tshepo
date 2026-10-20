"""Rule-based scam indicators and the baseline classifier.

:func:`red_flags` returns a dict of indicator -> weight. :func:`classify`
combines the flags with the parsed fields into a label in
{genuine_transaction, suspicious, noise}.

This is the zero-dependency baseline that every learned model must beat.
"""
from __future__ import annotations

import re
from dataclasses import asdict
from typing import Optional

from .fields import (Parsed, IN_RE, OUT_RE, KNOWN_SENDERS, SENDER_OP, SENDER_TYPO,
                     SHORT_CODES, parse)

URL_RE = re.compile(r"https?://|www\.|bit\.ly|\b[\w-]+\.(?:com|net|org|info|xyz|top|ml|tk|example|co\.ls)\b", re.I)
SECRET_RE = re.compile(r"\b(?:pin|password|passcode|otp)\b", re.I)
ASK_RE = re.compile(r"\b(?:send|share|give|reply|enter|confirm|provide|romela|mpolelle)\b", re.I)
NEGATION_RE = re.compile(r"\b(?:do not|don't|dont|never|u se)\b", re.I)
REWARD_RE = re.compile(r"\bwon\b|\bwin\b|prize|hapile|loan|moputso|claim|release|pending|refund|congratulations", re.I)
FEE_WORD_RE = re.compile(r"\bfee\b|processing|lefa|registration|charges?\b", re.I)
PRIZE_RE = re.compile(r"congratulations|you (?:have )?won|u hapile|winner|lebohoa", re.I)
MISTAKE_RE = re.compile(r"by mistake|in error|wrong number|ka phoso", re.I)
RETURN_RE = re.compile(r"send (?:it )?back|reverse|return|refund|khutlisetse|khutlisa|romela hape", re.I)
THREAT_RE = re.compile(r"suspend|blocked|deactivat|closure|closed|koetsoe|within 24", re.I)
ACCOUNT_RE = re.compile(r"account|akhaonto|wallet", re.I)
CLAIMS_OFFICIAL_RE = re.compile(r"confirmed|balance|dear customer|your .{0,30}account|security|akhaonto", re.I)
URGENT_RE = re.compile(r"urgent|immediately|hanghang|kapele|within \d+ (?:hours|minutes)", re.I)

WEIGHTS = dict(url=1, asks_secret=3, advance_fee=3, prize=2, reversal_request=3,
               account_threat=2, unverifiable_txn=2, spoofed_sender=3, urgent=1)
THRESHOLD = 3


def red_flags(body: str, sender: Optional[str] = None, parsed: Optional[Parsed] = None) -> dict:
    """Scam indicators with weights. Empty dict means no indicators found."""
    text = " ".join(body.split())
    low = text.lower()
    parsed = parsed or parse(body, sender)
    s = (sender or "").strip().upper()
    f = {}
    if URL_RE.search(text):
        f["url"] = WEIGHTS["url"]
    for sent in re.split(r"(?<=[.!?])\s+", text):  # ignore "Never share your PIN"
        if SECRET_RE.search(sent) and ASK_RE.search(sent) and not NEGATION_RE.search(sent):
            f["asks_secret"] = WEIGHTS["asks_secret"]
    if REWARD_RE.search(low) and FEE_WORD_RE.search(low):
        f["advance_fee"] = WEIGHTS["advance_fee"]
    if PRIZE_RE.search(low):
        f["prize"] = WEIGHTS["prize"]
    if MISTAKE_RE.search(low) and RETURN_RE.search(low):
        f["reversal_request"] = WEIGHTS["reversal_request"]
    if THREAT_RE.search(low) and ACCOUNT_RE.search(low):
        f["account_threat"] = WEIGHTS["account_threat"]
    if (parsed.amount is not None and (IN_RE.search(low) or OUT_RE.search(low))
            and parsed.balance is None
            and re.search(r"\bconfirmed\b|amohetse|you have received", low)):
        f["unverifiable_txn"] = WEIGHTS["unverifiable_txn"]
    resolved_sender = SHORT_CODES[s] if s in SHORT_CODES else SENDER_TYPO.get(s, s)
    if resolved_sender not in KNOWN_SENDERS and resolved_sender not in SENDER_OP.values() \
            and CLAIMS_OFFICIAL_RE.search(low):
        f["spoofed_sender"] = WEIGHTS["spoofed_sender"]
    if URGENT_RE.search(low):
        f["urgent"] = WEIGHTS["urgent"]
    return f


def classify(body: str, sender: Optional[str] = None) -> dict:
    """Label a single message using the rule baseline."""
    parsed = parse(body, sender)
    flags = red_flags(body, sender, parsed)
    score = sum(flags.values())
    if score >= THRESHOLD:
        label = "suspicious"
    elif parsed.is_transaction:
        label = "genuine_transaction"
    else:
        label = "noise"
    return dict(label=label, score=score, flags=flags, parsed=parsed)


def extract_transactions(records, label_fn=None):
    """Return ledger rows for genuine transactions only.

    ``label_fn(body, sender) -> label`` lets a trained model act as the gate.
    Rows the parser cannot read as transactions are skipped.
    """
    rows = []
    for r in records:
        res = classify(r["body"], r.get("sender"))
        label = label_fn(r["body"], r.get("sender")) if label_fn else res["label"]
        if label != "genuine_transaction" or not res["parsed"].is_transaction:
            continue
        d = asdict(res["parsed"])
        d["timestamp"] = d["timestamp"] or r.get("received_at")
        d["user_id"] = r.get("user_id")
        rows.append(d)
    return rows


__all__ = ["red_flags", "classify", "extract_transactions", "THRESHOLD", "WEIGHTS"]
