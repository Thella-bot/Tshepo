"""Template-drift monitor tests."""
from __future__ import annotations

from tshepo.parsing.drift import (drift_report, observe, reference_shape,
                                  template_signature, VERDICT_DRIFT, VERDICT_OK)

EC = "199"

BASE = [
    {"sender": EC, "body": "EcoCash:Cash In Confirmation. Received M100.00 from 37810-Sima Comm. "
                            "New wallet balance:M 102.07."},
    {"sender": EC, "body": "EcoCash: Transfer Confirmation.M85 to TSEPO NTLHANE. Wallet balance: M 9.07."},
    {"sender": EC, "body": "Ecocash: CashOut Confirmation: M 20 from 26921-Mantsiuoa. "
                           "Approval Code :CO260922.0848. New Wallet balance: M 22.38."},
]


def test_template_signature_ignores_the_numbers():
    a = template_signature("Received M100.00 from THABO")
    b = template_signature("Received M250.00 from LERATO")
    assert a == b
    assert template_signature("Transfer Confirmation") != a


def test_reference_shape():
    assert reference_shape("08DNBQCR6KGY") == "99AAAAAA9AAA"
    assert reference_shape("MP260921") == "AA999999"
    assert reference_shape(None) == "none"


def test_an_unchanged_window_has_no_drift():
    report = drift_report(BASE, BASE)
    assert report["verdict"] == VERDICT_OK
    assert report["drift_score"] == 0
    assert report["findings"] == []


def test_a_new_template_is_reported_as_churn():
    changed = list(BASE) + [{"sender": EC, "body": "EcoCash: Money received M100.00 from 37810-Sima. "
                                                    "Current balance is M102.07."}]
    report = drift_report(changed, BASE)
    assert report["verdict"] != VERDICT_OK
    assert any("template" in f for f in report["findings"])


def test_an_unreadable_template_shows_up_as_a_parse_rate_drop():
    changed = [{"sender": EC, "body": "Your money has been credited. Thank you for banking with us."}
               for _ in BASE]
    report = drift_report(changed, BASE)
    assert report["verdict"] == VERDICT_DRIFT
    assert any("parse rate fell" in f for f in report["findings"])


def test_a_new_sender_is_reported():
    report = drift_report([{"sender": "NEWNAME", "body": "Hello there"}], BASE)
    assert any("not in the baseline" in f for f in report["findings"])


def test_a_changed_reference_shape_is_reported():
    changed = [dict(r, body=r["body"].replace("EcoCash:", "EcoCash ref 12345:")) for r in BASE]
    report = drift_report(changed, BASE)
    assert any("reference changed shape" in f for f in report["findings"])


def test_observe_counts_the_readable_share():
    stats = observe(BASE)
    assert stats[EC]["messages"] == 3
    assert stats[EC]["readable"] == 3


def test_the_window_parse_rate_is_message_weighted():
    """A rate over many senders must not be a plain mean of the per-sender rates."""
    many = [{"sender": "BIG", "body": BASE[0]["body"]} for _ in range(90)]
    few = [{"sender": "SMALL", "body": "hello there"} for _ in range(10)]
    report = drift_report(many + few, many + few)
    assert report["window"]["parse_rate"] == 0.9
