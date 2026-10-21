"""Regression tests for the parser hardening.

Every test here corresponds to a measured failure on the real exports or on a
message an operator can actually send. The numbers matter: "M1 500.00" used to
parse as 1.0, which is a silent 1000x error on a real transaction.
"""
from __future__ import annotations

import pytest

from tshepo.parsing.fields import parse, parse_all, split_alerts
from tshepo.parsing.rules import build_ledger, classify

EC = "199"
MP = "MPESA"


# --- money tokens ----------------------------------------------------------
@pytest.mark.parametrize("text,amount", [
    ("M1,500.00", 1500.0),
    ("M1 500.00", 1500.0),
    ("M1.500,00", 1500.0),
    ("M100,50", 100.50),
    ("M1500", 1500.0),
    ("M0.10", 0.10),
    ("M1.36.00", 1.36),
    ("M-50.00", -50.0),
    ("LSL 300.00", 300.0),
    ("LSL300.00", 300.0),
    ("R500.00", 500.0),
])
def test_amount_formats(text, amount):
    body = (f"Transact ID 08AAA Confirmed. You have received {text} from THABO 58123456 "
            f"on 19/10/26 at 2:15 PM. New M-Pesa balance is M9,000.00.")
    p = parse(body, MP)
    assert p.amount == pytest.approx(amount)


def test_currency_after_the_number():
    body = ("Cash In Confirmation. Received 100.00 M from 37810-Sima Comm. "
            "New wallet balance: 202.07 M.")
    p = parse(body, EC)
    assert p.amount == pytest.approx(100.0)
    assert p.balance == pytest.approx(202.07)
    assert p.kind == "transaction"


def test_bare_number_only_next_to_a_money_word():
    body = "M-Pesa: 100.00 received. 19/10/26 14:15. Balance: 900.00. Thank you."
    p = parse(body, MP)
    assert p.amount == pytest.approx(100.0)
    assert p.balance == pytest.approx(900.0)
    assert p.kind == "transaction"


def test_a_date_is_never_read_as_an_amount():
    body = "M-Pesa: received 100.00 on 19/10/26 at 14:15. Balance: 900.00."
    p = parse(body, MP)
    assert p.amount == pytest.approx(100.0)


def test_a_reference_is_never_read_as_an_amount():
    body = "08DNBQCR6KGY. You have received M130.00 from Vodacom M-Pesa on 11/8/26. New M-Pesa balance is M130.00."
    p = parse(body, MP)
    assert p.amount == pytest.approx(130.0)


# --- direction and category ------------------------------------------------
def test_agent_cash_in_is_a_deposit_not_income():
    """The largest incoming category in the real EcoCash export.

    Counting it as income reported M1,355 a month of income for a customer
    who was mostly paying cash into their own wallet.
    """
    body = ("EcoCash:Cash In Confirmation. Received M100.00 from 37810-Sima Communication Msu Qoaling. "
            "New wallet balance:M 102.07. Send money to your loved ones with EcoCash")
    p = parse(body, EC)
    assert p.direction == "in"
    assert p.category == "deposit"


def test_cash_in_without_the_confirmation_words_is_a_deposit():
    body = ("You have received M10 from 0000098907-EcoCash Commissions 2025. "
            "New wallet balance: M 29.35.")
    p = parse(body, EC)
    assert p.category == "deposit"


def test_cash_withdrawal_is_a_deposit_not_income():
    body = ("Transact ID 08AIB0D3RWR6 Confirmed.You have received M300.00 and withdrawal fee of M6.32 "
            "from 26650707067 - Manthabeleng Mapitsi on 8/10/26 at 5:45 PM New M-Pesa balance is M306.32.")
    p = parse(body, MP)
    assert p.direction == "in"
    assert p.category == "deposit"
    assert p.fee == pytest.approx(6.32)


def test_give_cash_to_an_agent_is_a_deposit():
    body = ("Transact ID 08DNBY4FOVVT Confirmed. on 10/10/26 at 10:39 AM Give M100.00 cash to "
            "Bafokeng General Cafe A2023/11873 Moeti. New M-Pesa balance is M103.01Customer Care :114")
    p = parse(body, MP)
    assert p.direction == "in"
    assert p.category == "deposit"


def test_person_to_person_transfer_is_not_a_merchant_payment():
    body = ("LV1FX0W7CE. You have paid M577.00 to TUMELO TAU 60262174 on 1/10/26 at 8:30 AM. "
            "New M-Pesa balance is M10,597.91.")
    p = parse(body, MP)
    assert p.category == "transfer"


def test_merchant_payment_is_recognised():
    body = ("Transact ID 08DNBXVLACWU Confirmed.You have paid Nkalime of M13.29 on 8/10/26 at 5:45 PM "
            "New M-Pesa balance is M260.52.Customer Care :114")
    p = parse(body, MP)
    assert p.category == "merchant"


def test_data_bundle_is_airtime_and_goes_out():
    body = "You received GSM Social WhatsApp Daily 5 MB @ M3.00 from MOHAKALA MOHAKALA 62848628."
    p = parse(body, EC)
    assert p.direction == "out"
    assert p.category == "airtime"
    assert p.amount == pytest.approx(3.0)


def test_bundle_purchase_is_airtime():
    body = ("You have paid M6.00 from your wallet for GSM Social Combo Daily 500 MB @ M6.00 purchase "
            "for 62928888. New Balance is M96.07.")
    p = parse(body, EC)
    assert p.direction == "out"
    assert p.category == "airtime"


def test_loan_repayment_leaves_the_wallet():
    body = "Loan repayment of M250.00 received. Remaining balance M1,200.00."
    p = parse(body, MP)
    assert p.direction == "out"
    assert p.category == "loan_repayment"


def test_bill_payment_leaves_the_wallet():
    body = "Electricity bill payment of M450.00 successful. New M-Pesa balance is M200.00."
    p = parse(body, MP)
    assert p.direction == "out"
    assert p.category == "bill"


def test_operator_reversal_is_not_a_transaction():
    body = "EcoCash:Transfer Confirmation.M500.00 from 57850818 has been reversed. Wallet balance: M 0.00."
    p = parse(body, EC)
    assert p.kind == "failed"
    assert p.direction is None


def test_reversal_request_is_not_a_transaction():
    body = "You have received M500.00 by mistake. Please send it back to 57850818. New wallet balance: M 500.00."
    p = parse(body, EC)
    assert p.direction is None
    assert classify(body, EC)["label"] == "suspicious"


# --- completeness ----------------------------------------------------------
def test_a_truncated_alert_is_still_a_transaction():
    """A truncated alert is a real transaction with a missing balance.

    It must not be silently dropped, and it must be reported as unchained.
    """
    body = "Transact ID 08AAA Confirmed. You have received M300.00 from THABO 58123456 on 19/10/26 at 2:15 PM."
    p = parse(body, MP)
    assert p.kind == "transaction"
    assert p.amount == pytest.approx(300.0)
    assert p.balance is None


def test_two_alerts_in_one_body_both_count():
    body = ("EcoCash:Cash In Confirmation. Received M100.00 from 37810-Sima Comm. "
            "New wallet balance: M 102.07.EcoCash: CashOut Confirmation: M 20 from 26921-Mantsiuoa. "
            "New Wallet balance: M 82.07.")
    parts = split_alerts(body)
    assert len(parts) == 2
    rows = parse_all(body, EC)
    assert len(rows) == 2
    assert rows[0].amount == pytest.approx(100.0)
    assert rows[1].amount == pytest.approx(20.0)
    assert rows[1].direction == "out"

    ledger = build_ledger([{"body": body, "sender": EC, "user_id": "U1"}])
    assert len(ledger.transactions) == 2
    assert ledger.stats["unread"] == 0


def test_a_balance_only_alert_is_still_a_balance_update():
    body = "00000WF0B1RS Confirmed. Your balance is M247.23 on 8/10/26 at 5:47 PM.Customer Care: 114."
    p = parse(body, MP)
    assert p.kind == "balance_update"
    assert p.balance == pytest.approx(247.23)


def test_an_airtime_balance_is_not_a_wallet_balance():
    body = "Your airtime balance is M5.00. Data balance is 1GB."
    p = parse(body, MP)
    assert p.kind == "other"


# --- nothing disappears quietly --------------------------------------------
def test_unreadable_messages_are_counted():
    records = [
        {"body": "Transact ID 08AAA Confirmed. You have received M300.00 from THABO 58123456 "
                 "on 19/10/26 at 2:15 PM. New M-Pesa balance is M900.00.",
         "sender": MP, "user_id": "U1"},
        {"body": "Something the parser cannot read at all.", "sender": MP, "user_id": "U1"},
    ]
    ledger = build_ledger(records)
    assert ledger.stats["unread"] == 1
    assert ledger.stats["messages"] == 2


def test_the_passport_reports_how_much_was_read():
    from tshepo.pipeline import score_user
    records = [
        {"body": "Transact ID 08AAA Confirmed. You have received M300.00 from THABO 58123456 "
                 "on 19/10/26 at 2:15 PM. New M-Pesa balance is M900.00.",
         "sender": MP, "user_id": "U1", "received_at": "2026-10-19T14:15"},
        {"body": "Something the parser cannot read at all.", "sender": MP, "user_id": "U1",
         "received_at": "2026-10-19T14:16"},
    ]
    passport = score_user(records, user_label="U1")
    dq = passport["data_quality"]
    assert dq["messages_unread"] == 1
    assert dq["messages_read"] == 1
    assert any("could not be read" in issue for issue in dq["issues"])


def test_a_scam_alert_is_still_rejected():
    body = ("EcoCash:Cash In Confirmation. Received M5000.00 from 37810-Sima Comm. "
            "New wallet balance: M5000.00. Reply with your PIN 1234 to confirm.")
    assert classify(body, EC)["label"] == "suspicious"
