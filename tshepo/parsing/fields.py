"""SMS field extraction.

Pure parsing: given a message body and sender, return a :class:`Parsed` record
with operator, direction, category, amount, fee, balance, counterparty,
reference and timestamp. No scam classification happens here - that lives in
:mod:`tshepo.parsing.rules`.

The parser is written against the real operator formats in ``data/real_*.jsonl``:

* EcoCash ``Cash In Confirmation. Received M100.00 from 37810-Sima ...``
* EcoCash ``Transfer Confirmation.M85 to TSEPO NTLHANE. Wallet balance: M 9.07``
* EcoCash ``CashOut Confirmation: M 20 from 26921-Mantsiuoa ...``
* M-Pesa ``You have received M300.00 and withdrawal fee of M6.32 from ...``
* M-Pesa ``Give M100.00 cash to Bafokeng General Cafe ...``
* M-Pesa ``00000WF0B1RS Confirmed. Your balance is M247.23 ...``  (balance only)

Three things this module gets right that a naive keyword parser does not:

1. **Direction is inferred from the wallet, not from English.** ``Give M100 cash
   to <agent>`` and ``received ... for Withdrawal`` both *credit* the wallet,
   even though both read like spending. The rule-based fallback also reads
   ``Transfer Confirmation.M85 to X`` as out and ``M100 from X`` as in.
2. **Operator marketing is stripped before extraction**, so the promo tail
   (``Send money to your loved ones with EcoCash``) stops being parsed as a
   counterparty name or an amount.
3. **Balance-only messages are kept as balance updates.** They are not
   transactions, but they anchor the balance chain, and dropping them makes
   the integrity check report a broken chain for a perfectly good export.
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
NAME_A = re.compile(
    rf"\b(?:to|from)\s+({NAME})"
    r"(?=\d{8}|\s(?:on|for|approval|code|wallet|balance|new|txnid|ref|transaction|cash|"
    r"in|out|payment|merchant|has|been|initiated|processed)\b|[.,]|$)", re.I)
NAME_B = re.compile(rf"(?<![A-Za-z0-9])[56]\d{{7}}\s+({NAME})(?=\.|,|$)", re.I)
NAME_C = re.compile(rf"\bagent\s+({NAME})(?=\.|\s(?:on|for)\b|$)", re.I)
# {agent/merchant id}-{name}, e.g. "37810-Sima Communication Msu Qoaling"
# Transaction references. Every pattern must capture a token containing a
# digit: "Reference: ." in an M-Pesa alert must not become the reference
# "erence", or two unrelated transactions look like duplicates.
_REFERENCE_PATTERNS = [
    r"^\s*([A-Z0-9]*\d[A-Z0-9]{5,13})\.?\s+(?:confirmed|you\s+have)",
    r"Transact\s+ID\s*:?\s*([A-Z0-9]*\d[A-Z0-9]{5,13})",
    r"\bTxnId\s*:\s*([A-Z0-9]*\d[A-Z0-9]{4,})",
    r"Approval\s*Code\s*:?\s*([A-Z0-9]*\d[A-Z0-9]{4,})",
    r"\bRef(?:erence)?\s*:?\s*([A-Z0-9]*\d[A-Z0-9]{4,})",
]
_REFERENCE_RES = [re.compile(p, re.I) for p in _REFERENCE_PATTERNS]
MERCHANT_ID = re.compile(rf"(?<![A-Za-z0-9])(\d{{4,8}})\s*-\s*({NAME})(?=\s*(?:Approval|for\b|Wallet|New|\.|,|$))", re.I)
IN_RE = re.compile(r"received|credited|deposit|cash\s?in|amohetse", re.I)
OUT_RE = re.compile(r"\bsent\b|\bpaid\b|bought|debited|withdr|purchase|cash\s?out|cashout", re.I)
MONTHS = {m: i + 1 for i, m in enumerate("jan feb mar apr may jun jul aug sep oct nov dec".split())}

# --- wallet-direction rules that plain keywords get wrong ---------------------
# "Give M100.00 cash to Bafokeng General Cafe" credits the wallet: the customer
# hands cash to the agent and the agent credits the account.
GIVE_CASH_RE = re.compile(r"\bgive\b.{0,60}?\bcash\b", re.I)
# "You have received M26.00 from <agent> for Withdrawal" also credits the wallet.
FOR_WITHDRAWAL_RE = re.compile(r"\bfor\s+withdraw(?:al|al\s+fee)\b", re.I)
# A transaction the operator refused. Not income, not noise: a behavioural signal.
FAILED_RE = re.compile(
    r"\bis\s+failed\b|unable\s+to\s+process|is\s+insufficient\s+for\s+the\s+transaction|"
    r"transaction\s+is\s+failed|dormant\s+state", re.I)
# "Your balance is M247.23" - a balance restatement, not a transaction.
BALANCE_ONLY_RE = re.compile(
    r"\b(?:your\s+|new\s+|m-?pesa\s+|wallet\s+)?(?:available\s+)?balance\s+is\b|\bavail(?:able)?\s+bal\b", re.I)
# An airtime or data balance is not a wallet balance and must not re-anchor the chain.
NOT_WALLET_BALANCE_RE = re.compile(
    r"\b(?:airtime|data|sms|voice|bundle|minutes?|credit|loan|debt|interest)\s+balance\b", re.I)

# --- operator marketing tails ------------------------------------------------
# These ride along on every EcoCash alert. Left in place they are parsed as
# counterparty names ("your loved ones with EcoCash"), as amounts, and as
# transaction direction cues.
MARKETING_TAILS = [
    r"send\s+money\s+to\s+your\s+(?:loved\s+ones|family|friends)[^.]*\.?",
    r"pay\s+bills\s+from\s+the\s+comfort[^.]*\.?",
    r"use\s+ecocash\s+app[^.]*\.?",
    r"avoid\s+queues[^.]*\.?",
    r"dial\s+\*\d+\*#\s+to\s+pay\.?",
    r"quick,?\s+easy\s+transactions\s+anytime\.?",
    r"customer\s+care\s*:?\s*\.?\s*\d*",
    r"thank\s+you!",
    r"for\s+spot\b[^.]*\.?",
    r"spot\b[^.]*\.?",
]
_MARKETING_RE = re.compile("|".join(MARKETING_TAILS), re.I)

# --- counterparty hygiene -----------------------------------------------------
# Names the operator (or a promo line) that is not a payer.
OPERATOR_NAMES = {
    "vodacom m-pesa", "vodacom", "m-pesa", "mpesa", "ecocash", "econet", "orange",
    "your wallet", "your account", "your money", "your balance", "cashier", "customer care",
}
BOILERPLATE_NAME_RE = re.compile(
    r"^your\b|^(?:the\s+)?(?:ecocash|m-?pesa|app)\b|loved\s+ones|comfort\s+of\s+your|"
    r"quick,?\s+easy|anytime|avoid\s+queues|customer\s+care|thank\s+you|"
    r"pay\s+bills|send\s+money|spot\s+fine|dial\s+\*|"
    r"\bhas\s+been\b|successfully|initiated|processed|confirmed|"
    r"\bthe\b|\bfor\b|\band\b|\bwith\b|\bfrom\b", re.I)

# Sentence fragments that leak into the counterparty slot when a phone number
# is followed by prose ("... 57850818 has been initiated successfully").
NOT_A_NAME_RE = re.compile(
    r"\b(?:has|have|been|was|is|successfully|initiated|processed|confirmed|"
    r"transaction|approval|code|wallet|balance|new|cash|in|out|payment|merchant|"
    r"the|for|and|with|from|to|your|our|please|kindly)\b", re.I)

# --- transfer direction fallback ---------------------------------------------
_XFER_TO = re.compile(r"\bto\s+(?:\+?\d{8,10}|[A-Za-z])", re.I)
_XFER_FROM = re.compile(r"\bfrom\s+(?:\+?\d{8,10}|[A-Za-z])", re.I)
_TRANSFER_RE = re.compile(r"\btransfer(?:red|ence)?\b", re.I)


def strip_marketing(text: str) -> str:
    """Remove the operator promo tails that ride along on real alerts."""
    return _MARKETING_RE.sub(" ", text)


def _is_operator_name(name: Optional[str]) -> bool:
    if not name:
        return False
    n = re.sub(r"[^a-z0-9 ]+", " ", name.lower())
    n = re.sub(r"\s+", " ", n).strip()
    return n in OPERATOR_NAMES or bool(BOILERPLATE_NAME_RE.search(name))


def _looks_like_name(n: str) -> bool:
    """True when the string reads like a person or business, not a sentence."""
    if not n or len(n) > 60:
        return False
    if len(n.split()) > 5:
        return False
    if NOT_A_NAME_RE.search(n):
        return False
    return True


def _clean_name(n: Optional[str]) -> Optional[str]:
    """Normalise a counterparty name and drop promo/operator strings."""
    if not n:
        return None
    n = re.sub(r"^(?:at\s+)?agent\s+", "", n.strip(), flags=re.I)
    n = n.strip(" .,-")
    if not n or _is_operator_name(n):
        return None
    if not _looks_like_name(n):
        return None
    return n


def _infer_direction(text: str, low: str) -> Optional[str]:
    """Wallet direction: which way did the money move?

    Order matters. "You have received ... for Withdrawal" matches both the
    income and the spend keyword sets, and the wallet-side truth is that it
    credits the account.
    """
    if GIVE_CASH_RE.search(text):
        return "in"
    if FOR_WITHDRAWAL_RE.search(text):
        return "in"
    if IN_RE.search(low):
        return "in"
    if OUT_RE.search(low):
        return "out"
    return None


def _transfer_direction(text: str) -> Optional[str]:
    """Fallback for the "Transfer Confirmation" family, which states no verb.

    "Transfer Confirmation.M85 to TSEPO NTLHANE"   -> out
    "Transfer Confirmation. M 100 from MAPASEKA"    -> in
    """
    if not _TRANSFER_RE.search(text):
        return None
    to_m, from_m = _XFER_TO.search(text), _XFER_FROM.search(text)
    if to_m and from_m:
        return "out" if to_m.start() < from_m.start() else "in"
    if to_m:
        return "out"
    if from_m:
        return "in"
    return None


@dataclass
class Parsed:
    operator: Optional[str] = None
    is_transaction: bool = False
    is_balance_update: bool = False
    is_failed: bool = False
    kind: str = "other"           # transaction | balance_update | failed | other
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
    m = re.search(r"(\d{4})-(\d{2})-(\d{2})[ T](\d{2}):(\d{2}):(\d{2})", text)
    if m:
        y, mo, d, h, mi, s = m.groups()
        return f"{y}-{mo}-{d}T{h}:{mi}:{s}"
    return None


def _category(text: str, low: str, direction: Optional[str]) -> Optional[str]:
    """Transaction category.

    For incoming money the split between "deposit" (cash converted to e-money,
    not earned) and "transfer" (earned, from a named third party) drives the
    income features, so it has to match the generator's own labels: the
    synthetic "transfer" template is the EcoCash "Cash In Confirmation" line,
    and the synthetic "deposit" template always says "deposit" or "agent".

    Incoming rows only ever carry an inflow category; outflow categories
    ("airtime", "withdrawal", "merchant") on an incoming row are a parsing
    artefact of the operator's wording, not a fact about the money.
    """
    if direction == "in":
        if GIVE_CASH_RE.search(text) or FOR_WITHDRAWAL_RE.search(text):
            return "deposit"
        if re.search(r"deposit|cash\s?in", low) and not re.search(r"cash\s?in\s+confirmation", low):
            return "deposit"
        if re.search(r"\bfrom\s+agent\b", low):
            return "deposit"
        return "transfer"
    if re.search(r"loan|repay", low):
        return "loan_repayment"
    if re.search(r"cash\s?out|cashout", low):
        return "withdrawal"
    if "withdr" in low and not FOR_WITHDRAWAL_RE.search(text):
        return "withdrawal"
    if "airtime" in low:
        return "airtime"
    if "bill" in low:
        return "bill"
    if re.search(r"\bpaid\b|\bpos\b|purchase|cash\s?to", low):
        return "merchant"
    return "transfer"


def parse(body: str, sender: Optional[str] = None) -> Parsed:
    """Extract structured fields from an SMS. Returns a :class:`Parsed` object."""
    raw = " ".join(body.split())
    text = strip_marketing(raw)
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

    p.direction = _infer_direction(text, low)
    if p.direction is None:
        p.direction = _transfer_direction(text)

    p.category = _category(text, low, p.direction)

    if p.operator == "bank":
        d = re.search(r"ref:\s*(.+?)\.\s*avail", text, re.I)
        p.reference = d.group(1).strip() if d else None
    else:
        p.reference = None
        for rx in _REFERENCE_RES:
            r = rx.search(text)
            if r:
                p.reference = r.group(1).upper()
                break
        ph = PHONE_RE.search(text)
        p.counterparty_phone = ph.group(1) if ph else None
        n = NAME_A.search(text) or NAME_B.search(text) or NAME_C.search(text)
        p.counterparty_name = _clean_name(n.group(1)) if n else None
        m = MERCHANT_ID.search(text)
        p.counterparty_id = m.group(1).upper() if m else None
        if m and not p.counterparty_name:
            p.counterparty_name = _clean_name(m.group(2))

    # A credit from the operator itself (or an agent) is cash converted to
    # e-money, not earned income.
    if p.direction == "in" and p.category == "transfer":
        if _is_operator_name(p.counterparty_name) or p.counterparty_name is None:
            if re.search(r"\bfor\s+withdraw|\bgive\b[^.]{0,40}\bcash\b|cash\s?deposit", low):
                p.category = "deposit"

    p.timestamp = _timestamp(text)
    p.is_transaction = bool(p.operator and p.amount is not None and p.direction and p.balance is not None)
    p.is_balance_update = bool(
        not p.is_transaction and p.operator and p.balance is not None
        and p.amount is None and BALANCE_ONLY_RE.search(text)
        and not NOT_WALLET_BALANCE_RE.search(text))
    p.is_failed = bool(not p.is_transaction and p.operator and FAILED_RE.search(text))
    if p.is_transaction:
        p.kind = "transaction"
    elif p.is_balance_update:
        p.kind = "balance_update"
    elif p.is_failed:
        p.kind = "failed"
    else:
        p.kind = "other"
    got = [p.operator, p.amount, p.balance, p.direction, p.timestamp, p.reference]
    p.confidence = round(sum(x is not None for x in got) / len(got), 2)
    p.resolved_sender = resolved
    return p


__all__ = ["Parsed", "parse", "KNOWN_SENDERS", "SENDER_OP", "SHORT_CODES", "IN_RE", "OUT_RE",
           "strip_marketing"]
