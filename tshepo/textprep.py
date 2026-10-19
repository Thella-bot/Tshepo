"""Text preparation shared by every classifier.

The model input is deliberately format-agnostic:
  * a sender-type token (known operator / phone number / unknown alphanumeric ID)
  * the body with every digit replaced by 0, so that amounts, references and phone
    numbers do not become thousands of sparse features that never repeat.
"""
from __future__ import annotations

import re

from .parsing.parser import KNOWN_SENDERS


def sender_tag(sender):
    s = (sender or "").strip().upper()
    if s in KNOWN_SENDERS:
        return "SNDKNOWN"
    if s.startswith("+") or s[:3].isdigit():
        return "SNDPHONE"
    return "SNDOTHER"  # lookalike or unknown alphanumeric ID


def prep(body, sender=None):
    """Model input: sender-type token + body with digits normalised so formats generalise."""
    return f"{sender_tag(sender)} " + re.sub(r"\d", "0", " ".join(body.split()))


def heavy_noise(rng, s):
    """Stress test: stack perturbations and drop/swap letters inside words."""
    from .data.generator import perturb

    for _ in range(3):
        s = perturb(rng, s)
    out = []
    for w in s.split(" "):
        if len(w) > 4 and w.isalpha() and rng.random() < 0.3:
            j = rng.randint(1, len(w) - 2)
            w = w[:j] + w[j + 1:] if rng.random() < 0.5 else w[:j] + w[j + 1] + w[j + 2] if j + 2 <= len(w) else w
        out.append(w)
    return " ".join(out)
