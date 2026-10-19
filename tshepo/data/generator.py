"""
Tshepo - synthetic SMS generator.

Produces per-user, chronologically coherent streams of:
  * genuine_transaction  (mobile money / bank alerts, running balance stays consistent)
  * suspicious           (fake reversals, phishing, prize/loan/advance-fee, PIN requests)
  * noise                (promos, OTPs, bank notices, personal chat incl. Sesotho/English)

IMPORTANT: the templates below only APPROXIMATE real operator formats and the Sesotho
phrasing is a first draft. Before you quote any accuracy number, replace/extend the
templates with a sample of REAL (anonymised) messages and have a Sesotho speaker review
the scam/noise wording. Sender IDs (BANKA, BANKB, ...) are placeholders.

CLI: python -m tshepo.cli.generate --users 200 --seed 42 --out data/synthetic_sms.jsonl --csv
"""
from __future__ import annotations

import csv
import json
import math
import random
from datetime import datetime, timedelta

BASE = datetime(2026, 10, 19, 12, 0)  # "now" for the synthetic window

FIRST = ["Thabo", "Lerato", "Palesa", "Tumelo", "Mpho", "Neo", "Karabo", "Nthabiseng",
         "Teboho", "Lineo", "Puleng", "Tsepo", "Mosa", "Limpho", "Rethabile", "Itumeleng"]
LAST = ["Mokoena", "Letsie", "Molapo", "Ramokoatsi", "Lebona", "Mohapi", "Khali",
        "Tau", "Sello", "Mofokeng", "Thamae", "Seeiso"]
MERCHANTS = ["Mama Lineo Tuckshop", "Maseru Fresh Veg", "Corner Hardware",
             "Ha Thetsane Butchery", "Sefika Stationers", "Taxi Rank Kiosk",
             "Thaba Bosiu Grocers", "Mohale Spares"]
BILLERS = ["Water Utility", "Electricity Prepaid", "School Fees Office", "TV Subscription"]
DOMAINS = ["mpesa-verify.example", "ecocash-secure.example", "bank-update.example"]  # .example = never resolves
LOOKALIKES = ["MPESA-INFO", "ECO-CASH", "M-PESA1", "BANKA-SECURE", "VODA-PROMO"]

PROFILES = {  # drives how a synthetic user's money moves (useful later for the credit score)
    "steady_trader": dict(in_rate=0.55, in_mean=180, out_mean=120, start=(200, 1500)),
    "irregular":     dict(in_rate=0.40, in_mean=120, out_mean=140, start=(20, 400)),
    "salaried":      dict(in_rate=0.30, in_mean=2500, out_mean=250, start=(500, 3000)),
}

CATS = {
    ("wallet", "in"):  (["transfer", "deposit"], [6, 4]),
    ("wallet", "out"): (["transfer", "merchant", "airtime", "bill", "withdrawal"], [25, 30, 15, 10, 20]),
    ("bank", "in"):    (["transfer", "deposit"], [8, 2]),
    ("bank", "out"):   (["transfer", "merchant", "withdrawal", "loan_repayment", "airtime", "bill"],
                        [15, 30, 20, 15, 10, 10]),
}


# ---------------------------------------------------------------- helpers
def person(rng):
    n = f"{rng.choice(FIRST)} {rng.choice(LAST)}"
    return n.upper() if rng.random() < 0.5 else n


def phone(rng):
    return rng.choice("56") + "".join(rng.choice("0123456789") for _ in range(7))


def ref(rng):
    a = "ABCDEFGHJKLMNPQRSTUVWXYZ"
    return "".join(rng.choice(a) for _ in range(2)) + "".join(rng.choice(a + "0123456789") for _ in range(8))


def fmt_amt(x):
    return f"{x:,.2f}"


def stamp_mpesa(dt):
    return f"{dt.day}/{dt.month}/{dt:%y} at {dt.hour % 12 or 12}:{dt:%M} {'AM' if dt.hour < 12 else 'PM'}"


def stamp_eco(dt):
    return dt.strftime("%Y-%m-%d %H:%M")


def stamp_bank(dt):
    return dt.strftime("%d%b%y")


# ---------------------------------------------------------------- genuine
def build_genuine(rng, op, direction, cat, amt, fee, bal, dt, cp_name, cp_phone, bank_id="BANKA"):
    """Return (sender, body, shown) where `shown` says which truth fields appear in the text."""
    r, A, B = ref(rng), fmt_amt(amt), fmt_amt(bal)
    F = fmt_amt(fee) if fee is not None else None
    shown = dict(fee=fee, cp_name=None, cp_phone=None)

    if op == "mpesa":
        ts, s = stamp_mpesa(dt), "MPESA"
        if cat == "transfer" and direction == "out":
            body = f"{r} Confirmed. M{A} sent to {cp_name} {cp_phone} on {ts}. M-PESA balance is M{B}. Transaction cost, M{F}."
            shown.update(cp_name=cp_name, cp_phone=cp_phone)
        elif cat == "transfer":
            body = f"{r} Confirmed. You have received M{A} from {cp_name} {cp_phone} on {ts}. New M-PESA balance is M{B}."
            shown.update(cp_name=cp_name, cp_phone=cp_phone)
        elif cat == "merchant":
            body = f"{r} Confirmed. M{A} paid to {cp_name} on {ts}. M-PESA balance is M{B}. Transaction cost, M{F}."
            shown.update(cp_name=cp_name)
        elif cat == "bill":
            body = f"{r} Confirmed. M{A} paid to {cp_name} for bill payment on {ts}. M-PESA balance is M{B}. Transaction cost, M{F}."
            shown.update(cp_name=cp_name)
        elif cat == "airtime":
            body = f"{r} Confirmed. You bought M{A} of airtime on {ts}. M-PESA balance is M{B}."
        elif cat == "deposit":
            body = f"{r} Confirmed. You have received M{A} cash deposit from agent {cp_name} on {ts}. New M-PESA balance is M{B}."
            shown.update(cp_name=cp_name)
        else:  # withdrawal
            body = f"{r} Confirmed. You have withdrawn M{A} from agent {cp_name} on {ts}. M-PESA balance is M{B}. Transaction cost, M{F}."
            shown.update(cp_name=cp_name)
        return s, body, shown

    if op == "ecocash":
        ts, s = stamp_eco(dt), "ECOCASH"
        if cat == "transfer" and direction == "out":
            body = f"EcoCash: You have sent M{A} to {cp_phone} {cp_name}. Ref {r}. Fee M{F}. New balance M{B}. {ts}"
            shown.update(cp_name=cp_name, cp_phone=cp_phone)
        elif cat == "transfer":
            body = f"EcoCash: You have received M{A} from {cp_phone} {cp_name}. Ref {r}. New balance M{B}. {ts}"
            shown.update(cp_name=cp_name, cp_phone=cp_phone)
        elif cat == "merchant":
            body = f"EcoCash: You have paid M{A} to {cp_name}. Ref {r}. Fee M{F}. New balance M{B}. {ts}"
            shown.update(cp_name=cp_name)
        elif cat == "bill":
            body = f"EcoCash: You have paid M{A} to {cp_name} for bill payment. Ref {r}. Fee M{F}. New balance M{B}. {ts}"
            shown.update(cp_name=cp_name)
        elif cat == "airtime":
            body = f"EcoCash: You have bought M{A} airtime. Ref {r}. New balance M{B}. {ts}"
        elif cat == "deposit":
            body = f"EcoCash: Cash in of M{A} received from agent {cp_name}. Ref {r}. New balance M{B}. {ts}"
            shown.update(cp_name=cp_name)
        else:
            body = f"EcoCash: You have withdrawn M{A} at agent {cp_name}. Ref {r}. Fee M{F}. New balance M{B}. {ts}"
            shown.update(cp_name=cp_name)
        return s, body, shown

    # bank
    ts = stamp_bank(dt)
    mask = "XXXX" + "".join(rng.choice("0123456789") for _ in range(4))
    if direction == "in":
        desc = "CASH DEPOSIT" if cat == "deposit" else rng.choice(["SALE PROCEEDS", "TRANSFER IN", "SALARY", "PAYMENT RECEIVED"])
        verb = "credited"
    else:
        desc = {"transfer": "TRANSFER OUT", "merchant": "POS PURCHASE", "withdrawal": "ATM WITHDRAWAL",
                "loan_repayment": "LOAN REPAYMENT", "airtime": "AIRTIME PURCHASE", "bill": "BILL PAYMENT"}[cat]
        verb = "debited"
    body = f"{bank_id}: Acc {mask} {verb} M{A} on {ts}. Ref: {desc}. Avail bal M{B}."
    shown["fee"] = None
    return bank_id, body, shown


def pick_amount(rng, p, direction, cat):
    if cat == "airtime":
        return float(rng.choice([5, 10, 20, 30, 50, 100]))
    mean = p["in_mean"] if direction == "in" else p["out_mean"]
    return float(max(5, round(rng.lognormvariate(math.log(mean), 0.7))))


def fee_for(op, direction, cat, amt):
    if op == "bank" or direction == "in" or cat in ("airtime", "deposit"):
        return None
    return round(min(max(1.0, amt * 0.012), 30.0), 2)  # synthetic fee rule, NOT a real tariff


def make_txn(rng, p, bal, wallet, bank_id, dt, loan_mult=1.0):
    ops = [wallet] + (["bank"] if bank_id else [])
    op = ops[0] if len(ops) == 1 or rng.random() < 0.75 else "bank"
    direction = "in" if (rng.random() < p["in_rate"] or bal[op] < 15) else "out"
    kind = "bank" if op == "bank" else "wallet"
    cats, w = CATS[(kind, direction)]
    w = [x * loan_mult if c == "loan_repayment" else x for c, x in zip(cats, w)]
    cat = rng.choices(cats, w)[0]
    amt = pick_amount(rng, p, direction, cat)
    fee = fee_for(op, direction, cat, amt)
    if direction == "out" and amt + (fee or 0) > bal[op]:
        amt = float(max(5, int(bal[op] * rng.uniform(0.2, 0.7))))
        fee = fee_for(op, direction, cat, amt)
    bal[op] = round(bal[op] + amt if direction == "in" else bal[op] - amt - (fee or 0), 2)

    if cat in ("deposit", "withdrawal"):
        cp_name = person(rng)
    elif cat == "merchant":
        cp_name = rng.choice(MERCHANTS)
    elif cat == "bill":
        cp_name = rng.choice(BILLERS)
    elif cat == "transfer":
        cp_name = person(rng)
    else:
        cp_name = None
    cp_phone = phone(rng) if (cat == "transfer" and op != "bank") else None

    sender, body, shown = build_genuine(rng, op, direction, cat, amt, fee, bal[op], dt, cp_name, cp_phone, bank_id or "BANKA")
    truth = dict(operator=op, direction=direction, category=cat, amount=amt, balance=bal[op],
                 fee=shown["fee"], cp_name=shown["cp_name"], cp_phone=shown["cp_phone"],
                 timestamp=dt.strftime("%Y-%m-%d") if op == "bank" else dt.strftime("%Y-%m-%dT%H:%M"))
    return sender, body, cat, truth


# ---------------------------------------------------------------- noise
def noise_message(rng, dt, wallet, bank_id):
    p = rng.choice([50, 100, 150, 200])
    code = rng.randint(100, 199)
    bal = fmt_amt(rng.randint(0, 300) + rng.choice([0, 0.5]))
    items = [
        ("MPESA", "promo", "en", f"Buy 1GB data for M{p} valid 30 days. Dial *{code}# to subscribe."),
        ("VODACOM", "promo", "st", f"Reka data ea 1GB ka M{p}. Nako ke matsatsi a 30. Hatella *{code}#."),
        ("ECOCASH", "balance_notice", "en", f"Dear customer, your airtime balance is M{bal}."),
        ("VODACOM", "bundle_notice", "en", "Your data bundle expires in 24 hours. Top up to keep browsing."),
        ("MPESA", "otp", "en", f"Your verification code is {rng.randint(100000, 999999)}. Do not share this code with anyone."),
        ("ECOCASH", "otp", "en", f"Your OTP is {rng.randint(100000, 999999)}. Never share your OTP with anyone."),
        (bank_id or "BANKA", "bank_notice", "en", f"{bank_id or 'BANKA'}: Your account statement for {dt:%B} is ready. Visit any branch to collect it."),
        (bank_id or "BANKA", "bank_security_tip", "en", f"{bank_id or 'BANKA'}: Never share your PIN or password with anyone, including bank staff."),
        (bank_id or "BANKA", "bank_security_tip", "st", f"{bank_id or 'BANKA'}: U se ke ua fa motho PIN ea hau."),
        ("+266" + phone(rng), "chat", "en", "Hi, are we still meeting tomorrow at the market?"),
        ("+266" + phone(rng), "chat", "st", "Khotso! U phela joang? Ke tla u letsetsa mantsiboea."),
        ("+266" + phone(rng), "chat_money", "st", "Ke u rometse chelete ea koloi, u e fumane na?"),
        ("+266" + phone(rng), "chat_money", "mixed", f"Bro, ke u rometse M{rng.choice([20, 50, 100])} ka mpesa, check."),
        ("+266" + phone(rng), "chat_money", "st", "Ke tla u romela chelete hosane ha ke fumana."),
    ]
    s, sub, lang, body = rng.choice(items)
    return s, body, sub, lang


# ---------------------------------------------------------------- scams
def scam_message(rng, dt, wallet):
    op = "M-PESA" if wallet == "mpesa" else "EcoCash"
    name, ph, r = person(rng), phone(rng), ref(rng)
    amt = fmt_amt(float(rng.choice([50, 100, 200, 500, 1000, 2000])))
    big = f"{rng.choice([5000, 10000, 25000, 50000]):,}.00"
    fee = f"{rng.choice([50, 100, 150, 200, 300]):.2f}"
    dom = rng.choice(DOMAINS)
    ts = stamp_mpesa(dt)
    items = [
        ("fake_reversal", "en", f"{r} Confirmed. You have received M{amt} from {name} {ph} on {ts}. Sorry I sent it by mistake, please send back M{amt} to {ph}."),
        ("fake_reversal", "st", f"{r} Confirmed. U amohetse M{amt} ho tsoa ho {name}. Ke romile ka phoso, ka kopo u e khutlisetse ho {ph}."),
        ("phishing_link", "en", f"Dear customer, your {op} account has been suspended. Verify your details now at http://{dom}/verify to avoid closure."),
        ("phishing_link", "st", f"Akhaonto ea hau ea {op} e koetsoe. Tobetsa http://{dom}/bula hanghang ho e bulela."),
        ("fake_prize", "en", f"CONGRATULATIONS! You have won M{big} in the {op} festive promo. Pay M{fee} processing fee to {ph} to claim."),
        ("fake_prize", "st", f"U hapile M{big} ka {op}! Lefa M{fee} ho {ph} hore u fumane moputso."),
        ("pin_request", "en", f"{op} Security: To avoid account closure, reply with your PIN to {ph}."),
        ("loan_scam", "en", f"Get a loan of M{big} in 5 minutes! No collateral. Send M{fee} registration fee to {ph}."),
        ("fake_agent", "en", f"Agent {name}: your cash deposit of M{amt} is pending. Send M{fee} release fee to {ph} to receive it."),
    ]
    sub, lang, body = rng.choice(items)
    roll = rng.random()
    if roll < 0.55:
        sender = "+266" + phone(rng)
    elif roll < 0.90:
        sender = rng.choice(LOOKALIKES)
    else:
        sender = "MPESA" if wallet == "mpesa" else "ECOCASH"  # spoofed real sender ID: content must catch it
    return sender, body, sub, lang


# ---------------------------------------------------------------- realism
def perturb(rng, s):
    r = rng.random()
    if r < 0.05:
        s = s.lower()
    elif r < 0.10:
        s = s.upper()
    if rng.random() < 0.08:
        s = s.replace(". ", ".")
    if rng.random() < 0.06:
        s = s.replace(" ", "  ", 1)
    if rng.random() < 0.05:
        words = s.split(" ")
        idx = [i for i, w in enumerate(words) if len(w) > 5 and w.isalpha()]
        if idx:
            i = rng.choice(idx)
            w = words[i]
            j = rng.randint(1, len(w) - 3)
            words[i] = w[:j] + w[j + 1] + w[j] + w[j + 2:]
            s = " ".join(words)
    return s


# ---------------------------------------------------------------- dataset
def _sigmoid(x):
    return 1 / (1 + math.exp(-x))


def build(n_users=200, events=(15, 40), days=90, seed=42, scam_rate=0.08, noise_rate=0.20,
          perturb_rate=0.25, credit_signal=False):
    """Returns (records, users).

    credit_signal=False reproduces the plain classifier dataset exactly.
    credit_signal=True gives each user a hidden `reliability` that shapes behaviour (steadier income,
    bigger buffers, more loan repayments, tighter spending) and a SYNTHETIC repayment outcome
    (`defaulted`). That is only good for building and testing the scoring pipeline; real outcomes
    must come from a lender partner before any score is trusted.
    """
    rng = random.Random(seed)
    records, users, rid = [], [], 0
    for u in range(n_users):
        uid = f"U{u:04d}"
        profile = rng.choice(list(PROFILES))
        p = PROFILES[profile]
        wallet = rng.choice(["mpesa", "ecocash"])
        bank_id = rng.choice(["BANKA", "BANKB"]) if rng.random() < 0.4 else None

        rel_rng = random.Random(f"{seed}-{u}")  # separate stream: main stream unchanged when credit_signal=False
        reliability = rel_rng.betavariate(2, 2)
        defaulted = rel_rng.random() < _sigmoid(6 * (0.3 - reliability))
        loan_mult = 1.0
        if credit_signal:
            p = dict(p)
            p["in_rate"] = min(0.7, max(0.15, p["in_rate"] + 0.4 * (reliability - 0.5)))
            p["out_mean"] = p["out_mean"] * (1.6 - 1.2 * reliability)
            p["start"] = (int(p["start"][0] * (0.3 + 1.4 * reliability)), int(p["start"][1] * (0.3 + 1.4 * reliability)))
            loan_mult = 0.1 + 2.0 * reliability

        bal = {wallet: float(rng.randint(*p["start"]))}
        if bank_id:
            bal["bank"] = float(rng.randint(*p["start"]) * 2)
        n = rng.randint(*events)
        if credit_signal:
            span = days * 24 * 60
            jitter = (1.3 - reliability) * span / n * 1.5
            pts = [span * (i + 0.5) / n + rel_rng.gauss(0, jitter) for i in range(n)]
            times = sorted(BASE - timedelta(minutes=int(min(max(x, 0), span))) for x in pts)
        else:
            times = sorted(BASE - timedelta(minutes=rng.randint(0, days * 24 * 60)) for _ in range(n))
        for dt in times:
            roll = rng.random()
            truth = None
            if roll < scam_rate:
                sender, body, sub, lang = scam_message(rng, dt, wallet)
                label = "suspicious"
            elif roll < scam_rate + noise_rate:
                sender, body, sub, lang = noise_message(rng, dt, wallet, bank_id)
                label = "noise"
            else:
                sender, body, sub, truth = make_txn(rng, p, bal, wallet, bank_id, dt, loan_mult)
                label, lang = "genuine_transaction", "en"
            if rng.random() < perturb_rate:
                body = perturb(rng, body)
            records.append(dict(id=rid, user_id=uid, profile=profile, sender=sender,
                                received_at=dt.isoformat(timespec="minutes"), body=body,
                                label=label, subtype=sub, language=lang, truth=truth))
            rid += 1
        users.append(dict(user_id=uid, profile=profile, reliability=round(reliability, 4), defaulted=int(defaulted)))
    return records, users


def generate_dataset(n_users=200, events=(15, 40), days=90, seed=42,
                     scam_rate=0.08, noise_rate=0.20, perturb_rate=0.25):
    return build(n_users, events, days, seed, scam_rate, noise_rate, perturb_rate, credit_signal=False)[0]
