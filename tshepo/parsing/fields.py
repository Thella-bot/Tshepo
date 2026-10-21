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

# --- money tokens -----------------------------------------------------------
# The operators write amounts as "M100.00", "M 100.00", "LSL100.00" and "R500.00".
# When the operator changes a template the currency can move to the other side
# ("100.00 M") or disappear altogether, so a balance or a fee may be written
# without one. The currency is therefore optional where a strong word anchors
# the figure (balance, fee) and required for the transaction amount unless a
# money verb sits right next to the digits.
_CUR = r"(?:LSL|M|R)"
_CUR_PRE = rf"(?<![A-Za-z0-9]){_CUR}\s*"
_CUR_POST = rf"\s*{_CUR}(?![A-Za-z0-9])"
# 1,500.00 / 1 500.00 / 1.500,00 / 1.36.00 (operator artefact) / 1500 / 100,50
# Grouped: the alternation must not escape into the pattern it is used in.
_NUM = (r"(?:\d{1,3}(?:[,\u00a0\u202f ]\d{3})+(?:[.,]\d{1,2})?"
        r"|\d{1,3}(?:[.,]\d{3})+(?:[.,]\d{1,2})?"
        r"|\d+\.\d{1,2}\.\d{2}"
        r"|\d+(?:[.,]\d{1,2})?)")
AMT_RE = re.compile(_CUR_PRE + f"(-?{_NUM})", re.I)
AMT_SUFFIX_RE = re.compile(f"(-?{_NUM}){_CUR_POST}", re.I)
BAL_RE = re.compile(rf"(?:bal(?:ance)?|boemong\s+ba\s+hae|balanse(?:\s+ea\s+hao)?)"
                    rf"\s*(?:is|ke|:)?\s*(?:{_CUR}\s*)?(-?{_NUM})", re.I)
FEE_RE = re.compile(rf"(?:transaction\s+cost|fee|charge)s?(?:\s+of)?\W{{0,4}}"
                    rf"(?:{_CUR}\s*)?(-?{_NUM})(?!/)", re.I)
# A bare number is only money when a money word sits right next to it. The
# separator may not cross a "/", so a date is never read as an amount.
_BARE_AMT_RE = re.compile(
    rf"(?:received|paid|sent|bought|credited|debited|withdrawn|withdraw|cash\s?in"
    rf"|amount\s+of|total\s+of|value\s+of)[^0-9/]{{0,8}}?(-?{_NUM})(?!/)", re.I)
_BARE_AMT_REV_RE = re.compile(
    rf"(-?{_NUM})(?!/)\s*(?:received|paid|sent|bought|credited|debited|withdrawn)", re.I)
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
OUT_RE = re.compile(r"\bsent\b|\bpaid\b|bought|debited|withdr|purchase|cash\s?out|cashout|"
                     r"\bromela\b", re.I)
MONTHS = {m: i + 1 for i, m in enumerate("jan feb mar apr may jun jul aug sep oct nov dec".split())}

# --- wallet-direction rules that plain keywords get wrong ---------------------
# "Give M100.00 cash to Bafokeng General Cafe" credits the wallet: the customer
# hands cash to the agent and the agent credits the account.
GIVE_CASH_RE = re.compile(r"\bgive\b.{0,60}?\bcash\b", re.I)
# "You have received M26.00 from <agent> for Withdrawal" also credits the wallet.
FOR_WITHDRAWAL_RE = re.compile(r"\bfor\s+withdraw(?:al|al\s+fee)\b", re.I)
# "You have received M300.00 and withdrawal fee of M6.32 from <agent>" is the
# same event written the other way round: the customer took cash out, so the
# wallet is credited. This is cash in, not income.
WITHDRAWAL_FEE_RE = re.compile(r"\bwithdrawal\s+fee\b", re.I)
# A data/airtime bundle bought from a bundle seller: "5 MB @ M10.00".
BUNDLE_RE = re.compile(r"\b(?:mb|minutes|bundle|combo|data)\b[^.]{0,40}?@\s*"
                       rf"(?:{_CUR}\s*)?{_NUM}", re.I)
# Money that leaves the wallet, whichever way the operator words it.
LOAN_REPAY_RE = re.compile(r"\bloan\s+(?:repayment|payment)\b|\brepayments?\b", re.I)
BILL_PAY_RE = re.compile(r"\bbill\s+payment\b|\bpayment\s+of\b|\bpayment\s+for\b", re.I)
# The operator gave the money back, or the customer asked for it back. Either
# way no new money moved, so this is not a transaction.
REVERSED_RE = re.compile(r"\b(?:reversed|reversal|refunded|re-funded|cancelled|canceled)\b", re.I)
RETURN_REQUEST_RE = re.compile(r"\bsend\s+(?:it\s+)?back\b|\bsend\s+back\b|\bromela\s+hape\b", re.I)
MISTAKE_RE = re.compile(r"\bby\s+mistake\b|\bka\s+phoso\b|\bin\s+error\b", re.I)
# A transaction the operator refused, reversed or cancelled. Not income, not
# noise: a behavioural signal.
FAILED_RE = re.compile(
    r"\bis\s+failed\b|unable\s+to\s+process|is\s+insufficient\s+for\s+the\s+transaction|"
    r"transaction\s+is\s+failed|transaction\s+failed|dormant\s+state|"
    r"^failed\b|\bfailed\.|\breversed\b|\breversal\b|\brefunded\b|\bcancell?ed\b", re.I)
# "Your balance is M247.23" - a balance restatement, not a transaction.
# "Balanse ea hao" / "Boemong ba hae" are the Sesotho for the same thing.
BALANCE_ONLY_RE = re.compile(
    r"\b(?:your\s+|new\s+|m-?pesa\s+|wallet\s+)?(?:available\s+)?balance\s+is\b|\bavail(?:able)?\s+bal\b|"
    r"\bbalanse\s+ea\s+hao\b|\bboemong\s+ba\s+hae\b", re.I)# An airtime or data balance is not a wallet balance and must not re-anchor the chain.
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


# The header a second operator alert starts with. A phone export can write two
# alerts as one line, and reading only the first one silently loses a
# transaction, so the body is split on these.
_ALERT_HEAD_RE = re.compile(r"(?:EcoCash\s*:|Ecocash\s*:|Transact\s+ID\s|M-Pesa\s*:)", re.I)


def split_alerts(body: str) -> list[str]:
    """Split a body that holds more than one operator alert.

    "New wallet balance: M 102.07.EcoCash: CashOut Confirmation: M 20 ..."
    is two alerts, not one. The second one is a real withdrawal, and reading
    only the first half loses it from the ledger without a word about it.
    """
    text = " ".join(str(body or "").split())
    if not text:
        return []
    parts: list[str] = []
    last = 0
    for m in _ALERT_HEAD_RE.finditer(text):
        start = m.start()
        if start == 0:
            continue
        prev = text[start - 1]
        if prev not in ".!?;:,)" and not prev.isspace():
            continue
        chunk = text[last:start].strip()
        if chunk:
            parts.append(chunk)
        last = start
    tail = text[last:].strip()
    if tail:
        parts.append(tail)
    return parts


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
    if FOR_WITHDRAWAL_RE.search(text) or WITHDRAWAL_FEE_RE.search(text):
        return "in"
    if LOAN_REPAY_RE.search(low):
        return "out"
    if BILL_PAY_RE.search(low):
        return "out"
    if BUNDLE_RE.search(text):
        return "out"
    if IN_RE.search(low):
        return "in"
    if OUT_RE.search(low):
        return "out"
    return None


def _no_money_moved(text: str, low: str) -> bool:
    """True when the message says the money did not move.

    An operator reversal, a cancellation and a "send it back" request all look
    like a transaction to a keyword parser, and all three would otherwise be
    booked as income.
    """
    if REVERSED_RE.search(low) or RETURN_REQUEST_RE.search(low):
        return True
    if MISTAKE_RE.search(low) and RETURN_REQUEST_RE.search(low):
        return True
    return False


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
    """Parse a money token into a float.

    Handles the formats the operators and the phones actually produce:
    ``1,500.00`` (comma thousands), ``1 500.00`` (space thousands),
    ``1.500,00`` (European), ``100,50`` (European decimal comma),
    ``1.36.00`` (a doubled decimal in one real EcoCash alert) and a leading
    minus sign. The old version read "M1 500.00" as 1.0 and "M100,50" as
    100.0, which is a silent 1000x error on a real transaction.
    """
    if s is None:
        return 0.0
    s = str(s).strip()
    sign = 1.0
    if s.startswith("-"):
        sign, s = -1.0, s[1:]
    elif s.startswith("+"):
        s = s[1:]
    s = s.replace("\u00a0", "").replace("\u202f", "").replace(" ", "")
    if "." in s and "," in s:
        # the right-most separator is the decimal one
        if s.rfind(".") > s.rfind(","):
            s = s.replace(",", "")
        else:
            s = s.replace(".", "").replace(",", ".")
    elif "," in s:
        if s.count(",") > 1:
            s = s.replace(",", "")
        else:
            head, _, tail = s.rpartition(",")
            s = head + tail if (len(tail) == 3 and head) else head + "." + tail
    elif s.count(".") > 1:
        # 1.36.00 -> 1.36 (the operator doubled a decimal point)
        parts = s.split(".")
        s = parts[0] + "." + (parts[1][:2] or "0")
    m = re.match(r"\d+(?:\.\d+)?", s)
    if not m:
        return 0.0
    try:
        return sign * float(m.group(0))
    except ValueError:
        return 0.0


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


# "Cash In Confirmation" / "Cash in of M100 received from agent ..." - the
# customer handed cash to an agent and the agent credited the wallet.
CASH_IN_RE = re.compile(r"\bcash\s?in\b", re.I)
# "from 37810-Sima Communication" - the payer is an agent (a numbered till),
# not a person paying the customer.
AGENT_PAYOUT_RE = re.compile(r"from\s+\d{4,11}\s*[-–]\s*[A-Za-z]", re.I)
# A merchant payment, as opposed to paying a person. "You have paid M100 to
# THABO 58123456" is a transfer; "You have paid Nkalime of M13.29" is a
# merchant payment, and so is "for Merchant Payment" and "POS PURCHASE".
MERCHANT_RE = re.compile(
    r"for\s+merchant\s+payment|for\s+0\b|\bpos\b|purchase|"
    r"paid\s+[A-Za-z][\w.'\- ]{0,40}?\s+of\s+(?:LSL|M|R)\s*[\d,.]+|cash\s?to", re.I)
PAID_TO_RE = re.compile(r"\bpaid\s+to\s+", re.I)


def _is_merchant_payment(text: str) -> bool:
    """Is this a payment to a business rather than to a person?

    The operators do not say "merchant" in most of their alerts, so the
    counterparty is the clue: a payment to a shop has no phone number after
    the name, a payment to a person does.
    """
    if MERCHANT_RE.search(text):
        return True
    m = PAID_TO_RE.search(text)
    if m and not PHONE_RE.search(text[m.end():m.end() + 40]):
        return True
    return False


def _category(text: str, low: str, direction: Optional[str]) -> Optional[str]:
    """Transaction category.

    For incoming money the split between "deposit" (cash converted to e-money,
    not earned) and "transfer" (earned, from a named third party) drives the
    income features. Money that arrives from an agent or through a "Cash In"
    is the customer's own cash coming back in: counting it as income is the
    single most expensive mistake this parser can make, because in the real
    EcoCash export the agent cash-in is the largest incoming category.

    Incoming rows only ever carry an inflow category; outflow categories
    ("airtime", "withdrawal", "merchant") on an incoming row are a parsing
    artefact of the operator's wording, not a fact about the money.
    """
    if LOAN_REPAY_RE.search(low):
        return "loan_repayment"
    if BILL_PAY_RE.search(low):
        return "bill"
    if direction == "in":
        if GIVE_CASH_RE.search(text) or FOR_WITHDRAWAL_RE.search(text) \
                or WITHDRAWAL_FEE_RE.search(text):
            return "deposit"
        if CASH_IN_RE.search(low):
            return "deposit"
        if re.search(r"\bdeposit\b", low):
            return "deposit"
        if AGENT_PAYOUT_RE.search(text):
            return "deposit"
        return "transfer"
    if re.search(r"cash\s?out|cashout", low):
        return "withdrawal"
    if "withdr" in low and not FOR_WITHDRAWAL_RE.search(text):
        return "withdrawal"
    if "airtime" in low or BUNDLE_RE.search(text):
        return "airtime"
    if "bill" in low:
        return "bill"
    # A merchant payment is not just "paid": paying a person is a transfer.
    # The operators mark a merchant payment with "for Merchant Payment", with
    # the item code ("for 0"), with "POS", or with the "<business> of M<x>"
    # wording M-Pesa uses.
    if _is_merchant_payment(text):
        return "merchant"
    return "transfer"


def _find_amount(masked: str) -> Optional[float]:
    """The transaction amount: the first money figure that is not the balance
    or the fee.

    Tried in order of how much the operator's wording anchors it: a currency
    in front of the digits, a currency behind them, then a bare number next
    to a money word. A bare number is only accepted when a money word sits
    right next to it, so a reference or a phone number is never read as money.
    """
    m = AMT_RE.search(masked)
    if m:
        return _num(m.group(1))
    m = AMT_SUFFIX_RE.search(masked)
    if m:
        return _num(m.group(1))
    m = _BARE_AMT_REV_RE.search(masked) or _BARE_AMT_RE.search(masked)
    if m:
        return _num(m.group(1))
    return None


def parse(body: str, sender: Optional[str] = None) -> Parsed:
    """Extract structured fields from an SMS. Returns a :class:`Parsed` object.

    A body that holds more than one operator alert is split (see
    :func:`split_alerts`) and the first alert is returned here; callers that
    need every alert in the body use :func:`parse_all`.
    """
    parts = split_alerts(body)
    return _parse_one(parts[0] if parts else "", sender)


def parse_all(body: str, sender: Optional[str] = None) -> list[Parsed]:
    """Parse every operator alert in a body.

    Phones sometimes concatenate two alerts into one message (an export
    writes them as one line). Reading only the first one silently loses a
    transaction, which is worse than dropping the message: the money is
    missing from the ledger and nothing says so.
    """
    out = []
    for part in split_alerts(body):
        p = _parse_one(part, sender)
        if p.kind != "other" or p.amount is not None or p.balance is not None:
            out.append(p)
    return out or [_parse_one("", sender)]


def _parse_one(body: str, sender: Optional[str] = None) -> Parsed:
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
    p.balance = _num(bal.group(1)) if bal else None
    p.fee = _num(fee.group(1)) if fee else None
    p.amount = _find_amount(masked)

    p.direction = _infer_direction(text, low)
    if p.direction is None:
        p.direction = _transfer_direction(text)

    if p.direction is not None and _no_money_moved(text, low):
        # A reversal, a cancellation or a "send it back" request: no money
        # moved, so this is not an inflow and not an outflow.
        p.direction = None

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
    # A transaction needs an amount and a direction. The balance is what makes
    # it checkable against the rest of the history, not what makes it a
    # transaction, so an alert that the operator truncated (or wrote in a new
    # template without a balance) is still a transaction - it just cannot be
    # chained. A missing balance is reported, never silently ignored.
    p.is_transaction = bool(p.operator and p.amount is not None and p.direction)
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


__all__ = ["Parsed", "parse", "parse_all", "split_alerts", "KNOWN_SENDERS", "SENDER_OP",
           "SHORT_CODES", "IN_RE", "OUT_RE", "strip_marketing"]
