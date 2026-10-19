"""Rule baseline: field extraction, scam flags, classification."""
from tshepo.parsing.parser import classify, extract_transactions, parse, red_flags

MPESA_IN = ("RK12AB34CD Confirmed. You have received M100.00 from THABO MOKOENA 58123456 "
            "on 19/10/26 at 2:15 PM. New M-PESA balance is M900.00.")
MPESA_OUT = ("LJKPXGF0GY Confirmed. M362.00 paid to Thaba Bosiu Grocers on 30/7/26 at 7:01 AM. "
             "M-PESA balance is M1,136.66. Transaction cost, M4.34.")
ECOCASH_WD = ("EcoCash: You have withdrawn M322.00 at agent Mpho Tau. Ref XELVTWW8V2. "
              "Fee M3.86. New balance M23,792.05. 2026-09-05 04:41")
BANK_TXN = ("BANKB: Acc XXXX6475 debited M624.00 on 13Aug26. Ref: ATM WITHDRAWAL. Avail bal M2,204.00.")
PHISHING = ("Dear customer, your M-PESA account has been suspended. Verify your details now at "
            "http://mpesa-verify.example/verify to avoid closure.")
PIN_BAIT = "M-PESA Security: To avoid account closure, reply with your PIN to 58123456."
NOISE = "Buy 1GB data for M50 valid 30 days. Dial *133# to subscribe."


def test_parse_mpesa_incoming():
    p = parse(MPESA_IN, "MPESA")
    assert p.operator == "mpesa"
    assert p.direction == "in"
    assert p.amount == 100.00
    assert p.balance == 900.00
    assert p.reference == "RK12AB34CD"
    assert p.counterparty_name == "THABO MOKOENA"
    assert p.counterparty_phone == "58123456"
    assert p.timestamp == "2026-10-19T14:15"
    assert p.is_transaction


def test_parse_amount_not_confused_by_balance():
    p = parse(MPESA_OUT, "MPESA")
    assert p.amount == 362.00
    assert p.balance == 1136.66
    assert p.fee == 4.34
    assert p.category == "merchant"
    assert p.direction == "out"


def test_parse_ecocash_and_bank():
    p = parse(ECOCASH_WD, "ECOCASH")
    assert p.operator == "ecocash" and p.amount == 322.00 and p.balance == 23792.05
    assert p.category == "withdrawal"
    assert p.timestamp == "2026-09-05T04:41"

    b = parse(BANK_TXN, "BANKB")
    assert b.operator == "bank" and b.amount == 624.00 and b.balance == 2204.00
    assert b.direction == "out" and b.category == "withdrawal"


def test_classify_genuine_and_noise():
    assert classify(MPESA_IN, "MPESA")["label"] == "genuine_transaction"
    assert classify(BANK_TXN, "BANKB")["label"] == "genuine_transaction"
    assert classify(NOISE, "MPESA")["label"] == "noise"


def test_classify_scam():
    for body, sender in [(PHISHING, "MPESA-INFO"), (PIN_BAIT, "MPESA")]:
        res = classify(body, sender)
        assert res["label"] == "suspicious", res
        assert res["score"] >= 3
        assert res["flags"]


def test_negated_secret_not_flagged():
    """'Never share your PIN' is a safety tip, not a scam."""
    res = classify("BANKA: Never share your PIN or password with anyone, including bank staff.", "BANKA")
    assert "asks_secret" not in res["flags"]
    assert res["label"] == "noise"


def test_extract_transactions_skips_non_genuine():
    records = [
        {"body": MPESA_IN, "sender": "MPESA", "received_at": "2026-10-19T14:15", "user_id": "U1"},
        {"body": PHISHING, "sender": "MPESA-INFO", "received_at": "2026-10-19T14:20", "user_id": "U1"},
        {"body": NOISE, "sender": "MPESA", "received_at": "2026-10-19T14:25", "user_id": "U1"},
    ]
    rows = extract_transactions(records)
    assert len(rows) == 1
    assert rows[0]["amount"] == 100.00
    assert rows[0]["user_id"] == "U1"


def test_extract_transactions_honours_label_fn():
    """A gate that blocks a real transaction removes it from the ledger."""
    records = [
        {"body": MPESA_IN, "sender": "MPESA", "received_at": "2026-10-19T14:15"},
    ]
    assert len(extract_transactions(records, label_fn=lambda b, s: "genuine_transaction")) == 1
    assert len(extract_transactions(records, label_fn=lambda b, s: "suspicious")) == 0


def test_extract_transactions_skips_messages_that_are_not_transactions():
    """Even a permissive gate cannot invent a transaction out of a promo SMS."""
    records = [
        {"body": MPESA_IN, "sender": "MPESA", "received_at": "2026-10-19T14:15"},
        {"body": NOISE, "sender": "MPESA", "received_at": "2026-10-19T14:25"},
    ]
    rows = extract_transactions(records, label_fn=lambda b, s: "genuine_transaction")
    assert len(rows) == 1
