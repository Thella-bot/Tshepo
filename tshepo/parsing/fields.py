"""SMS field extraction.

Pure parsing: given a message body and sender, return a :class:`Parsed` record
with operator, direction, category, amount, fee, balance, counterparty,
reference and timestamp. No scam classification happens here - that lives in
:mod:`tshepo.parsing.rules`.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional

KNOWN_SENDERS = {"MPESA", "M-PESA", "ECOCASH", "BANKA", "BANKB", "VODACOM", "ECONET"}
SENDER_OP = {"MPESA": "mpesa", "M-PESA": "mpesa", "ECOCASH": "ecocash", "BANKA": "bank", "BANKB": "bank"}
SENDER_TYPO = {"LPESA": "MPESA"}
# Operator short codes (Lesotho). EcoCash authentic sender is 199; M-Pesa uses M-Pesa.
SHORT_CODES = {"199": "ecocash"}
CUR = r"(?<![A-Za-z0-9])(?:LSL\s?|M)\s*"
# Allow operator spacing ("M 102.07") and stray artifacts ("M1.36.00").
NUM = r"((?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d{1,2})?(?:\.\d{2})?)"
AMT_RE = re.compile(CUR + NUM, re.I)
BAL_RE = re.compile(rf"bal(?:ance)?\s*(?:is|:)?\s*{CUR}{NUM}", re.I)
FEE_RE = re.compile(rf"(?:transaction\s+cost|fee|charge)s?(?:\s+of)?\W{{0,4}}{CUR}{NUM}", re.I)
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
    resolved_sender: Optional[str] = None


def _num(s: str) -> float:
    """Parse a number, tolerating stray artifacts like 'M1.36.00' (first decimal pair)."""
    s = s.replace(",", "")
    m = re.match(r"\d+(?:\.\d{1,2})?", s)
    return float(m.group(0)) if m else float(s)


def _timestamp(text: str) -> Optional[str]:
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


def _clean_name(n: Optional[str]) -> Optional[str]:
    if not n:
        return None
    n = re.sub(r"^(?:at\s+)?agent\s+", "", n.strip(), flags=re.I)
    return n.strip(" .,-") or None


def parse(body: str, sender: Optional[str] = None) -> Parsed:
    """Extract structured fields from an SMS. Returns a :class:`Parsed` object."""
    text = " ".join(body.split())
    low = text.lower()
    s = (sender or "").strip().upper()
    p = Parsed()

    ss = SENDER_TYPO.get(s, s)
    if "m-pesa" in low or "mpesa" in low:
        p.operator = "mpesa"
    elif "ecocash" in low:
        p.operator = "ecocash"
    elif ss in SHORT_CODES:
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


__all__ = ["Parsed", "parse", "KNOWN_SENDERS", "SENDER_OP", "SHORT_CODES"]
