"""Provenance helpers: can this set of messages be tied to this applicant?

Every message in Tshepo arrives from the applicant. The parser can tell what
a message says; nothing in the pipeline can tell that the operator sent it.
What it *can* do is notice when the same messages are used twice, by two
different people: a history that is sold, shared or borrowed between
applicants is one history, and scoring it twice is scoring one wallet twice.

The fingerprint is a hash of the message text, so it says nothing about the
messages themselves and can be stored, compared and deleted like any other
aggregate.

    from tshepo.antiabuse import message_hashes
    hashes = message_hashes(records)
"""
from __future__ import annotations

import hashlib
from typing import Iterable


def body_hash(body: str) -> str:
    """A stable hash of one message body. Whitespace is normalised first, so
    the same message pasted with different spacing is the same message."""
    text = " ".join(str(body or "").split())
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def message_hashes(records: Iterable[dict]) -> list[str]:
    """The hashes of a set of messages, without duplicates, in a stable order."""
    return sorted({body_hash(r.get("body") or "") for r in records if r.get("body")})


def history_fingerprint(records: Iterable[dict]) -> str:
    """One fingerprint for a whole set of messages.

    Two applicants who submit exactly the same set of messages get the same
    fingerprint. Adding one message changes it, which is why the per-message
    overlap check is the stronger of the two.
    """
    joined = "\n".join(message_hashes(records))
    return hashlib.sha256(joined.encode("utf-8")).hexdigest()


#: How many messages two applicants have to share before it is a replay
#: rather than a coincidence (two people both paid by the same agent).
REPLAY_MIN_MESSAGES = 5
#: The share of the applicant's messages that another applicant already has.
REPLAY_MIN_SHARE = 0.5


def replay_reuse(shared: dict[str, int], total: int) -> tuple[bool, str, int]:
    """Decide whether a message overlap is a replay.

    ``shared`` maps another applicant to the number of messages they have in
    common with this one. Returns ``(is_replay, other_user_id, count)``.
    """
    if not shared or total <= 0:
        return False, "", 0
    other, count = max(shared.items(), key=lambda kv: kv[1])
    if count >= REPLAY_MIN_MESSAGES and count >= REPLAY_MIN_SHARE * total:
        return True, other, count
    return False, other, count


__all__ = ["body_hash", "message_hashes", "history_fingerprint", "replay_reuse",
           "REPLAY_MIN_MESSAGES", "REPLAY_MIN_SHARE"]
