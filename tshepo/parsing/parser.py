#!/usr/bin/env python3
"""
Tshepo - SMS parser + rule-based classifier (baseline).

parse(body, sender)      -> Parsed   structured fields from a mobile-money / bank alert
red_flags(body, sender)  -> dict     scam indicators with weights
classify(body, sender)   -> dict     label in {genuine_transaction, suspicious, noise}
extract_transactions(r)  -> list     clean ledger rows (genuine messages only) for the scoring model

This is the BASELINE. The plan is to train a text classifier on top and show the lift
over these rules on held-out REAL messages. Rules are tolerant of case, spacing and typos
in the surrounding text, but they are tuned to the synthetic templates - treat any
accuracy on synthetic data as optimistic.

CLI: python -m tshepo.cli.parse_sms "<message>" --sender MPESA
"""
from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from typing import Optional

KNOWN_SENDERS = {"MPESA", "M-PESA", "ECOCASH", "BANKA", "BANKB", "VODACOM", "ECONET"}
SENDER_OP = {"MPESA": "mpesa", "M-PESA": "mpesa", "ECOCASH": "ecocash", "BANKA": "bank", "BANKB": "bank"}
SENDER_TYPO = {"LPESA": "MPESA"}  # common typo in real exports
# Operator short codes (Lesotho). EcoCash authentic sender is 199; M-Pesa uses M-Pesa.
SHORT_CODES = {"199": "ecocash"}
CUR = r"(?<![A-Za-z0-9])(?:LSL\s?|M)\s*"
# Allow operator spacing ("M 102.07") and stray artifacts ("M1.36.00").
NUM = r"((?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d{1,2})?(?:\.\d{2})?)"
AMT_RE = re.compile(CUR + NUM, re.I)
BAL_RE = re.compile(rf"bal(?:ance)?\s*(?:is|:)?\s*{CUR}{NUM}", re.I)
FEE_RE = re.compile(rf"(?:transaction\s+cost|fee|charge)s?(?:\s+of)?\s*(?:is|:)?\s*{CUR}{NUM}", re.I)
PHONE_RE = re.compile(r"(?<![A-Za-z0-9])(?:\+?266)?([56]\d{7})(?![A-Za-z0-9])")
NAME = r"[A-Za-z][A-Za-z .'\-]{1,48}?"
NAME_A = re.compile(rf"\b(?:to|from)\s+({NAME})\s*(?=\d{{8}}|\s(?:on|for)\b|\.|,|$)", re.I)
NAME_B = re.compile(rf"(?<![A-Za-z0-9])[56]\d{{7}}\s+({NAME})(?=\.|,|$)", re.I)
NAME_C = re.compile(rf"\bagent\s+({NAME})(?=\.|,|\s(?:on|for)\b|$)", re.I)
# {agent/merchant id}-{name}, e.g. "37810-Sima Communication Msu Qoaling"
MERCHANT_ID = re.compile(rf"(?<![A-Za-z0-9])(\d{{4,6}})\s*-\s*({NAME})(?=\s*(?:Approval|for\b|Wallet|New|\.|,|$))", re.I)
IN_RE = re.compile(r"received|credited|deposit|cash\s?in|amohetse", re.I)
OUT_RE = re.compile(r"\bsent\b|\bpaid\b|bought|debited|withdr|purchase|cash\s?out|cashout|\bgive\b", re.I)
MONTHS = {m: i + 1 for i, m in enumerate("jan feb mar apr may jun jul aug sep oct nov dec".split())}


@dataclass
class Parsed:
    operator: Optional[str] = None
    is_transaction: bool = False
    direction: Optional[str] = None
    category: Optional[str] = None
    amount: Optional[float] = None
    fee: Optional[float] = None
    balance: Optional[float] = None
    counterparty_name: Optional[str] = None
    counterparty_phone: Optional[str] = None
    counterparty_id: Optional[str] = None
    reference: Optional[str] = None
    timestamp: Optional[str] = None
    confidence: float = 0.0


def _num(s):
    """Parse a number, tolerating stray artifacts like 'M1.36.00' (take first decimal pair)."""
    s = s.replace(",", "")
    m = re.match(r"\d+(?:\.\d{1,2})?", s)
    return float(m.group(0)) if m else float(s)


def _timestamp(text):
    m = re.search(r"(\d{1,2})/(\d{1,2})/(\d{2,4})(?:\s+at\s+|\s+)(\d{1,2}):(\d{2})\s*([AP]M)?", text, re.I)
    if m:
        d, mo, y, h, mi, ap = m.groups()
        y = int(y) + (2000 if int(y) < 100 else 0)
        h = int(h)
        if ap:
            h = h % 12 + (12 if ap.upper() == "PM" else 0)
        return f"{y:04d}-{int(mo):02d}-{int(d):02d}T{h:02d}:{mi}"
    m = re.search(r"(\d{4})-(\d{2})-(\d{2})[ T](\d{2}):(\d{2})", text)
    if m:
        y, mo, d, h, mi = m.groups()
        return f"{y}-{mo}-{d}T{h}:{mi}"
    m = re.search(r"(?<![A-Za-z0-9])(\d{1,2})([A-Za-z]{3})(\d{2})(?![A-Za-z0-9])", text)
    if m and m.group(2).lower() in MONTHS:
        d, mon, y = m.groups()
        return f"{2000 + int(y):04d}-{MONTHS[mon.lower()]:02d}-{int(d):02d}"
    return None


def _clean_name(n):
    n = re.sub(r"^(?:at\s+)?agent\s+", "", n.strip(), flags=re.I)
    return n.strip(" .,-") or None


def parse(body: str, sender: Optional[str] = None) -> Parsed:
    text = " ".join(body.split())
    low = text.lower()
    s = (sender or "").strip().upper()
    p = Parsed()

    ss = SENDER_TYPO.get(s, s)                       # tolerate typos like LPESA -> MPESA
    if "m-pesa" in low or "mpesa" in low:
        p.operator = "mpesa"
    elif "ecocash" in low:
        p.operator = "ecocash"
    elif ss in SHORT_CODES:                          # EcoCash authentic sender is 199
        p.operator = SHORT_CODES[ss]
    elif "avail bal" in low or re.search(r"\bacc\b", low):
        p.operator = "bank"
    else:
        p.operator = SENDER_OP.get(ss)
    resolved = ss if ss in KNOWN_SENDERS else (SHORT_CODES.get(ss) or SENDER_OP.get(ss))

    bal = BAL_RE.search(text)
    fee = FEE_RE.search(text)
    masked = text
    for m in (bal, fee):
        if m:
            masked = masked[:m.start()] + " " * (m.end() - m.start()) + masked[m.end():]
    amt = AMT_RE.search(masked)
    p.balance = _num(bal.group(1)) if bal else None
    p.fee = _num(fee.group(1)) if fee else None
    p.amount = _num(amt.group(1)) if amt else None

    if IN_RE.search(low):
        p.direction = "in"
    elif OUT_RE.search(low):
        p.direction = "out"

    if re.search(r"loan|repay", low):
        p.category = "loan_repayment"
    elif re.search(r"cash\s?out|cashout", low):
        p.category = "withdrawal"
    elif "withdr" in low:
        p.category = "withdrawal"
    elif "airtime" in low:
        p.category = "airtime"
    elif "bill" in low:
        p.category = "bill"
    elif re.search(r"deposit|cash\s?in", low):
        p.category = "deposit"
    elif re.search(r"\bpaid\b|\bpos\b|purchase|cash\s?to", low):
        p.category = "merchant"
    else:
        p.category = "transfer"

    if p.operator == "bank":
        d = re.search(r"ref:\s*(.+?)\.\s*avail", text, re.I)
        p.reference = d.group(1).strip() if d else None
    else:
        r = (re.match(r"^\s*([A-Z0-9]{10})\s+confirmed", text, re.I)
             or re.search(r"\bref(?:erence)?\W{0,3}([A-Z0-9]{6,12})\b", text, re.I)
             or re.search(r"TxnId\s*:\s*([A-Z0-9]{6,12})", text, re.I))
        p.reference = r.group(1).upper() if r else None
        ph = PHONE_RE.search(text)
        p.counterparty_phone = ph.group(1) if ph else None
        n = NAME_A.search(text) or NAME_B.search(text) or NAME_C.search(text)
        p.counterparty_name = _clean_name(n.group(1)) if n else None
        m = MERCHANT_ID.search(text)
        p.counterparty_id = m.group(1).upper() if m else None
        if m and not p.counterparty_name:
            p.counterparty_name = _clean_name(m.group(2))

    p.timestamp = _timestamp(text)
    p.is_transaction = bool(p.operator and p.amount is not None and p.direction and p.balance is not None)
    got = [p.operator, p.amount, p.balance, p.direction, p.timestamp, p.reference]
    p.confidence = round(sum(x is not None for x in got) / len(got), 2)
    p.resolved_sender = resolved
    return p


# ------------------------------------------------------------------ scam rules
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
    text = " ".join(body.split())
    low = text.lower()
    parsed = parsed or parse(body, sender)
    s = (sender or "").strip().upper()
    f = {}
    resolved = parsed.resolved_sender or (SHORT_CODES.get(s) or SENDER_TYPO.get(s, s))
    if URL_RE.search(text):
        f["url"] = WEIGHTS["url"]
    for sent in re.split(r"(?<=[.!?])\s+", text):  # ignore sentences like "Never share your PIN"
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
    if (parsed.amount is not None and (IN_RE.search(low) or OUT_RE.search(low)) and parsed.balance is None
            and re.search(r"\bconfirmed\b|amohetse|you have received", low)):
        f["unverifiable_txn"] = WEIGHTS["unverifiable_txn"]
    if s in SHORT_CODES:
        resolved_sender = SHORT_CODES[s]
    else:
        resolved_sender = SENDER_TYPO.get(s, s)
    if resolved_sender not in KNOWN_SENDERS and resolved_sender not in SENDER_OP.values() \
            and CLAIMS_OFFICIAL_RE.search(low):
        f["spoofed_sender"] = WEIGHTS["spoofed_sender"]
    if URGENT_RE.search(low):
        f["urgent"] = WEIGHTS["urgent"]
    return f


def classify(body: str, sender: Optional[str] = None) -> dict:
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
    """records: iterable of dicts with body, sender, received_at (and optional user_id).
    Returns ledger rows for genuine transactions only - the input to the credit-passport features.
    label_fn(body, sender) -> label lets a trained model (see tshepo.models.gate) act as the gate;
    the rule parser still extracts the fields, and rows it cannot parse as transactions are skipped."""
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
