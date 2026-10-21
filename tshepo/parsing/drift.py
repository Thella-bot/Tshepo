"""Template-drift monitoring.

An operator changes an alert template without telling anyone. The parser does
not crash: it keeps answering, it just stops understanding the new wording,
and the symptom is a ledger with holes in it rather than an error. The score
still comes out, it is just quietly wrong, and nothing in the system says so.

This module watches the signals that move when a template changes:

1. **parse rate** - the share of a sender's messages that produce a usable
   row. A template change that the parser cannot read shows up here first,
   usually within a day of the change.
2. **template churn** - the share of messages whose fingerprint was not seen
   in the baseline window. A brand new fingerprint is not proof of drift (a
   new message type is normal), but a *rising* share of them is.
3. **field null rates** - the share of messages with no amount, no balance or
   no direction. A template that drops the balance takes the chain with it.
4. **reference shape** - the shape of the transaction reference ("08DNBQCR6KGY"
   -> "99AAAAAAAAAA"). Operators change reference formats when they change
   anything, and a changed shape breaks duplicate detection.

Each signal is compared against a baseline window and combined into a drift
score. The score says "look at this", it does not say "the parser is broken":
a new template is often a new product rather than a broken one.

    from tshepo.parsing.drift import drift_report
    report = drift_report(current_records, baseline_records)
    if report["verdict"] != "ok":
        print(report["findings"])
"""
from __future__ import annotations

import re
from collections import Counter
from typing import Any, Iterable, Optional

from .fields import parse

#: Weights for the drift score. Parse rate dominates because it is the
#: signal that reaches the score; the rest explain it.
WEIGHTS = dict(parse_rate=40.0, template_churn=25.0, field_nulls=20.0, reference_shape=15.0)

#: A parse-rate drop of this many points is drift, not noise.
PARSE_RATE_DROP = 0.10
#: More than this share of unseen templates is churn worth reporting.
TEMPLATE_CHURN = 0.15
#: A rise of this many points in the share of messages missing a field.
FIELD_NULL_RISE = 0.10

VERDICT_OK = "ok"
VERDICT_WATCH = "watch"
VERDICT_DRIFT = "drift"


#: Words the operators actually use. Everything else in a message is a name,
#: a merchant or a phone number, and is masked out of the fingerprint. A
#: fingerprint that kept the counterparty would give every message its own
#: "new template" and the churn signal would be pure noise.
OPERATOR_WORDS = frozenset("""
ecocash m-pesa mpesa lsl m r cash in out confirmation confirmed receive received
pay paid send sent balance new wallet available avail acc transfer amount fee
withdrawal withdraw withdrawn airtime bill bills payment merchant agent deposit
purchase bought give for of to from the is on at has have been reference ref
txnid id code approval approved customer care successfully initiated processed
reversed reversal failed unable process insufficient transaction dormant state
credited debited loan repayment data combo social daily weekly bundle gsm
babatone minutes mb you your my our and or not no this that it as be
""".split())


def template_signature(body: str) -> str:
    """A stable fingerprint of a message template.

    Numbers, names, merchants and phone numbers are masked, so "Received
    M100.00 from THABO" and "Received M250.00 from LERATO" share a
    fingerprint, while a reworded template gets a new one.
    """
    from .fields import strip_marketing
    t = strip_marketing(" ".join(str(body or "").split())).lower()
    out: list[str] = []
    for tok in re.findall(r"[a-z][a-z\-']*|\d[\d.,]*|[^\sa-z0-9]", t):
        if tok.isdigit() or tok[0].isdigit():
            out.append("#")
        elif tok in OPERATOR_WORDS:
            out.append(tok)
        elif tok.isalpha():
            out.append("#")
    return " ".join(out)[:160]


def reference_shape(reference: Optional[str]) -> str:
    """The shape of a transaction reference: "08DNBQCR6KGY" -> "99AAAAAAAAAA".

    A change in shape means the operator changed how references are printed,
    which breaks duplicate detection and the "initiated"/"processed" pairing.
    """
    if not reference:
        return "none"
    return re.sub(r"[A-Z]", "A", re.sub(r"\d", "9", str(reference).upper()))


def _blank() -> dict[str, Any]:
    return dict(messages=0, readable=0, no_amount=0, no_balance=0, no_direction=0,
                signatures=Counter(), reference_shapes=Counter())


def observe(records: Iterable[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Summarise one window of messages, per sender."""
    out: dict[str, dict[str, Any]] = {}
    for r in records:
        sender = (str(r.get("sender") or "").strip() or "<none>")
        body = r.get("body") or ""
        s = out.setdefault(sender, _blank())
        s["messages"] += 1
        s["signatures"][template_signature(body)] += 1
        p = parse(body, r.get("sender"))
        if p.kind in ("transaction", "balance_update", "failed"):
            s["readable"] += 1
        if p.amount is None:
            s["no_amount"] += 1
        if p.balance is None:
            s["no_balance"] += 1
        if p.direction is None:
            s["no_direction"] += 1
        s["reference_shapes"][reference_shape(p.reference)] += 1
    return out


def _rate(n: int, d: int) -> float:
    return round(n / d, 4) if d else 0.0


def _summarise(s: dict[str, Any]) -> dict[str, Any]:
    n = max(s["messages"], 1)
    shapes = s["reference_shapes"]
    dominant = shapes.most_common(1)[0][0] if shapes else "none"
    return dict(
        messages=s["messages"],
        parse_rate=_rate(s["readable"], n),
        no_amount_rate=_rate(s["no_amount"], n),
        no_balance_rate=_rate(s["no_balance"], n),
        no_direction_rate=_rate(s["no_direction"], n),
        distinct_templates=len(s["signatures"]),
        dominant_reference_shape=dominant,
        reference_shapes=dict(shapes),
        signatures=set(s["signatures"]),
    )


def _share(summaries: list[dict[str, Any]], key: str) -> float:
    """A message-weighted mean over senders, not a plain mean of rates."""
    total = sum(s["messages"] for s in summaries) or 1
    return sum(s[key] * s["messages"] for s in summaries) / total


def drift_report(current: Iterable[dict[str, Any]],
                 baseline: Optional[Iterable[dict[str, Any]]] = None) -> dict[str, Any]:
    """Compare a window of messages against a baseline window.

    ``baseline`` is the window the parser is known to read. It defaults to the
    current window, which is only useful for reporting, not for detecting
    drift.
    """
    cur_obs = observe(current)
    base_obs = observe(baseline) if baseline is not None else cur_obs
    findings: list[str] = []
    senders: list[dict[str, Any]] = []
    score = 0.0

    for sender, cur_raw in sorted(cur_obs.items()):
        cur = _summarise(cur_raw)
        base_raw = base_obs.get(sender)
        base = _summarise(base_raw) if base_raw else None
        entry: dict[str, Any] = {"sender": sender, "current": {
            k: v for k, v in cur.items() if k not in ("signatures", "reference_shapes")},
            "baseline": None, "signals": {}}
        if base is None:
            entry["signals"]["new_sender"] = True
            findings.append(f"{sender}: a sender that is not in the baseline "
                            f"({cur['messages']} message(s)) - every message from it is unverified")
            senders.append(entry)
            continue
        signals: dict[str, Any] = {}

        drop = base["parse_rate"] - cur["parse_rate"]
        if drop > PARSE_RATE_DROP:
            signals["parse_rate_drop"] = round(drop, 4)
            score += WEIGHTS["parse_rate"] * min(drop / PARSE_RATE_DROP, 2.0)
            findings.append(
                f"{sender}: parse rate fell from {base['parse_rate']:.0%} to "
                f"{cur['parse_rate']:.0%} - the template probably changed")

        unseen = {sig for sig, _ in cur_raw["signatures"].items()} - base["signatures"]
        churn = _rate(sum(cur_raw["signatures"][s] for s in unseen), max(cur_raw["messages"], 1))
        if churn > TEMPLATE_CHURN:
            signals["template_churn"] = round(churn, 4)
            score += WEIGHTS["template_churn"] * min(churn / TEMPLATE_CHURN, 2.0)
            findings.append(
                f"{sender}: {churn:.0%} of the messages use a template that was not in the "
                f"baseline ({len(unseen)} new template(s))")

        for key, label in (("no_amount_rate", "amount"), ("no_balance_rate", "balance"),
                           ("no_direction_rate", "direction")):
            rise = cur[key] - base[key]
            if rise > FIELD_NULL_RISE:
                signals[f"{label}_null_rise"] = round(rise, 4)
                score += WEIGHTS["field_nulls"] / 3 * min(rise / FIELD_NULL_RISE, 2.0)
                findings.append(
                    f"{sender}: {rise:.0%} more messages have no {label} than in the baseline")

        if cur["dominant_reference_shape"] != base["dominant_reference_shape"]:
            signals["reference_shape_changed"] = {
                "baseline": base["dominant_reference_shape"],
                "current": cur["dominant_reference_shape"],
            }
            score += WEIGHTS["reference_shape"]
            findings.append(
                f"{sender}: the transaction reference changed shape "
                f"({base['dominant_reference_shape']} -> {cur['dominant_reference_shape']}); "
                f"duplicate detection may be affected")

        entry["baseline"] = {k: v for k, v in base.items() if k not in ("signatures", "reference_shapes")}
        entry["signals"] = signals
        senders.append(entry)

    score = round(min(score, 100.0), 1)
    verdict = VERDICT_OK if score == 0 else (VERDICT_WATCH if score < 40 else VERDICT_DRIFT)
    return {
        "verdict": verdict,
        "drift_score": score,
        "findings": findings,
        "senders": senders,
        "window": {
            "messages": sum(s["messages"] for s in cur_obs.values()),
            "parse_rate": round(_share([_summarise(v) for v in cur_obs.values()], "parse_rate"), 4),
        },
    }


__all__ = ["template_signature", "reference_shape", "observe", "drift_report",
           "VERDICT_OK", "VERDICT_WATCH", "VERDICT_DRIFT", "WEIGHTS"]
