"""The generator must produce self-consistent data, otherwise every downstream
number (and the whole credit story) is built on sand."""
from collections import Counter

from tshepo.data.generator import build


def test_labels_and_counts(small_dataset):
    records, users = small_dataset
    assert len(users) == 6
    counts = Counter(r["label"] for r in records)
    assert set(counts) <= {"genuine_transaction", "suspicious", "noise"}
    assert counts["genuine_transaction"] > counts["noise"] > 0
    assert all(r["body"] for r in records)


def test_balances_chain_per_user():
    """A genuine transaction's balance must equal the previous balance +/- amount +/- fee."""
    records, _ = build(n_users=8, seed=17, events=(15, 25))
    last = {}
    checked = 0
    for r in records:
        if r["label"] != "genuine_transaction":
            continue
        t = r["truth"]
        key = (r["user_id"], t["operator"])  # each user and each account has its own chain
        if key in last:
            prev = last[key]
            exp = prev + t["amount"] if t["direction"] == "in" else prev - t["amount"] - (t["fee"] or 0)
            assert abs(exp - t["balance"]) < 0.011, (r["body"], prev, t)
            checked += 1
        last[key] = t["balance"]
    assert checked > 10  # the check actually ran


def test_scams_use_untrusted_senders_mostly():
    records, _ = build(n_users=30, seed=23)
    scams = [r for r in records if r["label"] == "suspicious"]
    assert scams
    # at least some scams arrive from a phone number, which no operator uses
    assert any(r["sender"].startswith("+266") for r in scams)


def test_credit_signal_shapes_behaviour():
    """Reliable users must earn more and save more than unreliable ones."""
    records, users = build(n_users=300, seed=31, events=(30, 70), credit_signal=True)
    from tshepo.features.cashflow import build_features
    from tshepo.parsing.parser import extract_transactions
    by_u = {}
    for r in records:
        by_u.setdefault(r["user_id"], []).append(r)
    incomes = []
    for u in users:
        f = build_features(extract_transactions(by_u[u["user_id"]]))
        incomes.append((u["reliability"], f["monthly_income"]))
    reliable = [i for rel, i in incomes if rel > 0.7]
    unreliable = [i for rel, i in incomes if rel < 0.3]
    assert reliable and unreliable
    assert sum(reliable) / len(reliable) > sum(unreliable) / len(unreliable)
