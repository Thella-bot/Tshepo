"""Consent: what the applicant is agreeing to, and how it is enforced.

Nothing is collected before a consent record exists. A consent record names
the purpose, the scope, the grant time and the expiry, and it is the key that
the API checks before it will accept a single message.

The applicant holds a token that is shown once. With it they can submit their
own messages, read their own passport and erase their own data at any time,
without an operator in the loop.
"""
from __future__ import annotations

from typing import Any, Optional

from .config import CONSENT_TTL_DAYS

CONSENT_VERSION = "1.0"
CONSENT_PURPOSE = "credit_assessment"

CONSENT_TEXT = """\
Tshepo credit passport - what we collect and why

WHAT WE COLLECT
If you send us your text messages, we read the mobile-money and bank alerts
among them. We keep the amount, the direction, the balance and the date of
each transaction. We keep the messages themselves only for as long as the
consent below lasts, and we delete them when you withdraw consent or when
the retention period ends.

WHAT WE DO NOT COLLECT
We do not use your contacts, your photos, your location, your browsing or
your passwords. We do not read the content of a message beyond working out
whether it is a transaction, and the passport we produce contains no names,
no phone numbers and no message text.

WHY
To build a credit passport that shows a lender how your money arrives and
leaves, so that you can be assessed on your actual cash flow instead of on
having no payslip.

WHO SEES IT
The applicant (you) and the lender your passport is sent to, under a data
sharing agreement. Nobody else.

HOW LONG
The messages are deleted when you withdraw consent, and in any case within
{retention_days} days of collection. The consent itself lasts {ttl_days} days.

YOUR RIGHTS
You can ask what we hold about you, ask us to correct it, or ask us to
delete it, at any time, using the code we give you. Deleting your data also
deletes your passport.
"""

CONSENT_TEXT_SN = """\
Tshepo - tumello ea hao

O roma SMS ea hao re sebatsa. Re boloka melao ea chelete eo re
sehlopong, hore re tsebe ka mese-tsetsi ea hao. Re se boloke mola o kahoo
mape a hao, lifa tsebe, kapa sengolo sa hao. Re boloka feela ka mor'a
{ttl_days} maha, mme re tlatsa hore SMS ea hao e tla sireloa.

O ka khutla ka nako e o batlang ha o kopa hore re hlakole tsebo ea hao
kapa re tlatsa hore re hlakole sengolo sa hao.
(Seotlha sa Sesotho: kopo ngaka ho sekhotsi se se lekale sengolo.)
"""

RETENTION_DAYS_TEXT = 180


def consent_text(ttl_days: Optional[int] = None, retention_days: Optional[int] = None) -> str:
    """The consent notice shown to the applicant before they send anything."""
    from .config import RETENTION_DAYS
    ttl = int(ttl_days if ttl_days is not None else CONSENT_TTL_DAYS)
    keep = int(retention_days if retention_days is not None else RETENTION_DAYS)
    return CONSENT_TEXT.format(ttl_days=ttl, retention_days=keep)


def consent_text_sn(ttl_days: Optional[int] = None) -> str:
    """Sesotho summary. Draft: a native speaker must sign this off before
    it is shown to applicants, because a wrong legal notice in Sesotho is
    worse than no Sesotho notice."""
    ttl = int(ttl_days if ttl_days is not None else CONSENT_TTL_DAYS)
    return CONSENT_TEXT_SN.format(ttl_days=ttl)


def consent_summary(ttl_days: Optional[int] = None, retention_days: Optional[int] = None) -> dict[str, Any]:
    """Everything the capture portal needs to render the consent step."""
    from .config import RETENTION_DAYS
    return {
        "version": CONSENT_VERSION,
        "purpose": CONSENT_PURPOSE,
        "text": consent_text(ttl_days, retention_days),
        "text_sesotho_draft": consent_text_sn(ttl_days),
        "ttl_days": int(ttl_days if ttl_days is not None else CONSENT_TTL_DAYS),
        "retention_days": int(retention_days if retention_days is not None else RETENTION_DAYS),
        "withdrawable": True,
    }


__all__ = ["CONSENT_VERSION", "CONSENT_PURPOSE", "CONSENT_TEXT", "consent_text",
           "consent_text_sn", "consent_summary"]
