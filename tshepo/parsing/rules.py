"""Rule-based scam indicators and the baseline classifier.

:func:`red_flags` returns a dict of indicator -> weight. :func:`classify`
combines the flags with the parsed fields into a label in
{genuine_transaction, suspicious, noise}.

:func:`build_ledger` turns a user's messages into the ledger the score is
computed from. It applies one rule that matters: **the gate can only veto.**

The parser decides what is structurally a transaction (an amount, a direction
and a balance all present). The gate decides what is a scam. A message the
parser can read is a transaction unless the gate calls it suspicious; a
message the parser cannot read is noise whatever the gate says.

The old arrangement let a learned gate delete real transactions simply by
labelling them "noise" - which is exactly what the hybrid model did to the
real EcoCash export: 98 readable rows became 82. Scam rejection still works,
because a forged alert is structurally a transaction and still has to pass
the gate.
"""
from __future__ import annotations

import re
from dataclasses import asdict
from typing import Optional

from ..features.cashflow import Ledger
from .fields import (IN_RE, KNOWN_SENDERS, OUT_RE, SENDER_OP, SENDER_TYPO,
                     SHORT_CODES, Parsed, parse, parse_all)

URL_RE = re.compile(r"https?://|www\.|\.bit\.ly|\b[\w-]+\.(?:com|net|org|info|xyz|top|ml|tk|example|co\.ls)\b", re.I)
SECRET_RE = re.compile(r"\b(?:pin|password|passcode|otp)\b", re.I)
ASK_RE = re.compile(r"\b(?:send|share|give|reply|enter|confirm|provide|romela|mpolelle"
                    r"|required|needed|verify|update)\b", re.I)
NEGATION_RE = re.compile(r"\b(?:do not|don't|dont|never|u se)\b", re.I)
REWARD_RE = re.compile(r"\bwon\b|\bwin\b|prize|hapile|loan|moputso|claim|release|pending|refund|congratulations", re.I)
FEE_WORD_RE = re.compile(r"\bfee\b|processing|lefa|registration|charges?\b", re.I)
PRIZE_RE = re.compile(r"congratulations|you (?:have )?won|u hapile|winner|lebohoa|"
                      r"your prize|claim your|to claim", re.I)
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
    parts = parse_all(body, sender) if parsed is None else (
        [parsed] if isinstance(parsed, Parsed) else list(parsed))
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
    # An alert that states an amount but no balance cannot be checked against
    # the rest of the history, so it cannot be verified either.
    unverifiable = any(
        p.amount is not None and p.balance is None
        and (IN_RE.search(low) or OUT_RE.search(low))
        and re.search(r"\bconfirmed\b|amohetse|you have received", low)
        for p in parts)
    if unverifiable:
        f["unverifiable_txn"] = WEIGHTS["unverifiable_txn"]
    resolved_sender = SHORT_CODES[s] if s in SHORT_CODES else SENDER_TYPO.get(s, s)
    if resolved_sender not in KNOWN_SENDERS and resolved_sender not in SENDER_OP.values() \
            and CLAIMS_OFFICIAL_RE.search(low):
        f["spoofed_sender"] = WEIGHTS["spoofed_sender"]
    if URGENT_RE.search(low):
        f["urgent"] = WEIGHTS["urgent"]
    return f


def classify(body: str, sender: Optional[str] = None,
             parts: Optional[list[Parsed]] = None) -> dict:
    """Label a single message using the rule baseline.

    ``parts`` lets a caller that has already parsed the body pass the result
    in, so the message is not parsed twice.
    """
    if parts is None:
        parts = parse_all(body, sender)
    parsed = parts[0] if parts else parse(body, sender)
    flags = red_flags(body, sender, parts)
    score = sum(flags.values())
    if score >= THRESHOLD:
        label = "suspicious"
    elif parsed.is_transaction:
        label = "genuine_transaction"
    else:
        label = "noise"
    return dict(label=label, score=score, flags=flags, parsed=parsed)


def _row(r, parsed: Parsed) -> dict:
    d = asdict(parsed)
    d["timestamp"] = d["timestamp"] or r.get("received_at")
    d["user_id"] = r.get("user_id")
    return d


def _dedup_key(row: dict) -> tuple:
    """Identity of a transaction, used to drop the "initiated"/"processed" pair.

    The reference alone is not enough: an operator that prints "Reference: ."
    would make two different transactions look identical.
    """
    return (row.get("reference") or "", row.get("direction"),
            row.get("amount"), row.get("balance"))


#: How far the timestamp printed in a message and the timestamp the export
#: recorded for it may differ before the message is treated as edited.
TIMESTAMP_TOLERANCE_HOURS = 36.0


def _timestamp_gap_hours(a: str, b: str) -> float:
    """Hours between two timestamps, or 0.0 when either cannot be read."""
    from ..features.cashflow import _ts
    ta, tb = _ts(a), _ts(b)
    if ta is None or tb is None:
        return 0.0
    if (ta.tzinfo is None) != (tb.tzinfo is None):
        ta, tb = ta.replace(tzinfo=None), tb.replace(tzinfo=None)
    return abs((ta - tb).total_seconds()) / 3600.0


def build_ledger(records, label_fn=None) -> Ledger:
    """Build the ledger for one user from their messages.

    The gate may only veto. A message the parser reads as a transaction joins
    the ledger unless the gate calls it suspicious; a balance restatement
    re-anchors the chain; a refused transaction is kept as a behavioural
    signal rather than thrown away.

    Every message that ends up in no list at all is counted in
    ``stats["unread"]``. A message the parser could not read used to vanish
    without a trace, which made a partial export look like a complete one.
    """
    ledger = Ledger()
    seen: set[tuple] = set()
    ledger.stats["unread"] = 0
    ledger.stats["timestamp_mismatch"] = 0
    for r in records:
        if r.get("origin") == "sent":
            ledger.stats["sent"] = ledger.stats.get("sent", 0) + 1
            continue
        ledger.stats["messages"] = ledger.stats.get("messages", 0) + 1
        parts = parse_all(r["body"], r.get("sender"))
        res = classify(r["body"], r.get("sender"), parts)
        label = label_fn(r["body"], r.get("sender")) if label_fn else res["label"]
        if label == "suspicious":
            ledger.stats["suspicious"] = ledger.stats.get("suspicious", 0) + 1
            ledger.stats["vetoed"] = ledger.stats.get("vetoed", 0) + 1
            continue
        ledger.stats[label] = ledger.stats.get(label, 0) + 1
        used = False
        # A message carries the timestamp the operator printed in it, and the
        # export carries the timestamp the phone recorded. When both exist and
        # they disagree, the message and the file it came from were not
        # produced at the same time - which is what editing an export looks
        # like. A message without its own timestamp cannot be checked this way.
        for parsed in parts:
            if parsed.timestamp and r.get("received_at"):
                if _timestamp_gap_hours(parsed.timestamp, r["received_at"]) > TIMESTAMP_TOLERANCE_HOURS:
                    ledger.stats["timestamp_mismatch"] = ledger.stats.get("timestamp_mismatch", 0) + 1
        # One message can hold more than one operator alert; each is a row.
        for parsed in parts:
            if parsed.kind == "transaction":
                row = _row(r, parsed)
                key = _dedup_key(row)
                if key in seen:
                    ledger.stats["duplicates"] = ledger.stats.get("duplicates", 0) + 1
                    used = True
                    continue
                seen.add(key)
                ledger.transactions.append(row)
                if parsed.balance is None:
                    ledger.stats["unchained"] = ledger.stats.get("unchained", 0) + 1
                used = True
            elif parsed.kind == "balance_update":
                ledger.balance_updates.append(_row(r, parsed))
                used = True
            elif parsed.kind == "failed":
                ledger.failed.append(_row(r, parsed))
                used = True
        if not used:
            ledger.stats["unread"] = ledger.stats.get("unread", 0) + 1
    ledger.stats["transactions"] = len(ledger.transactions)
    ledger.stats["balance_updates"] = len(ledger.balance_updates)
    ledger.stats["failed"] = len(ledger.failed)
    ledger.stats["usable"] = len(ledger.transactions) + len(ledger.balance_updates)
    messages = ledger.stats.get("messages", 0)
    # Two different shares, and they answer different questions:
    #   readable_share - could the parser read the message at all
    #   usable_share   - did it become a row in the ledger
    ledger.stats["readable_share"] = round(
        (messages - ledger.stats.get("unread", 0)) / max(messages, 1), 3)
    ledger.stats["usable_share"] = round(ledger.stats["usable"] / max(messages, 1), 3)
    return ledger


def extract_transactions(records, label_fn=None):
    """Return ledger rows for genuine transactions only.

    Kept for callers that only want the money-moving rows. The gate may only
    veto: see :func:`build_ledger`.
    """
    ledger = build_ledger(records, label_fn=label_fn)
    return ledger.transactions


__all__ = ["red_flags", "classify", "extract_transactions", "build_ledger", "Ledger",
           "THRESHOLD", "WEIGHTS"]
