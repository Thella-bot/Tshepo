"""Import a phone export into the JSONL format the project uses.

The capture path fails on format, not on willingness: a user who exports from
Samsung Messages, from Samsung SMS Backup+, from a Google Messages JSON
backup or from a CSV gets a file this module can read, where the old
importer only understood the Google Messages text export.

Supported formats, detected from the content:

* ``google_txt``   - the Google Messages "Export messages" .txt file
                     ("Received from <sender> on <date> <time>")
* ``samsung_xml``  - Samsung SMS Backup+ XML (``<sms date= address= body= type=>``)
* ``google_json``  - the Google Messages JSON backup
* ``csv``          - a CSV with a date, a sender and a body column
* ``tsv``          - the same, tab separated

    from tshepo.importers import parse_file
    records = parse_file("data/real_ecocash_raw.txt")

Each record is ``{sender, received_at, body, origin}`` where ``origin`` is
"received" or "sent". Sent messages are kept in the file for the audit trail
but are not put into the ledger, because the money alert for a transfer the
customer sent is the authoritative record, not the customer's own request.
"""
from __future__ import annotations

import csv
import io
import json
import os
import re
import xml.etree.ElementTree as ET
from typing import Any, Iterable, Optional

GOOGLE_TXT_BLOCK = re.compile(
    r"^Received from (?P<sender>.+?) on (?P<date>\d{4}-\d{2}-\d{2})[ T](?P<time>\d{2}:\d{2})",
    re.M)
SEP_RE = re.compile(r"^-{3,}\s*$", re.M)

_CSV_DATE_KEYS = ("received_at", "date", "time", "datetime", "timestamp")
_CSV_SENDER_KEYS = ("sender", "address", "from", "number", "phone")
_CSV_BODY_KEYS = ("body", "message", "text", "content")


def detect_format(text: str, filename: str = "") -> str:
    """Work out which export format a file is in."""
    stripped = text.lstrip()
    if stripped.startswith("<"):
        return "samsung_xml"
    if stripped.startswith("[") or stripped.startswith("{"):
        return "google_json"
    if GOOGLE_TXT_BLOCK.search(text):
        return "google_txt"
    if filename.lower().endswith((".csv",)):
        return "csv"
    if filename.lower().endswith((".tsv",)):
        return "tsv"
    if "," in text.splitlines()[0] if text.splitlines() else False:
        return "csv"
    return "unknown"


def _epoch_to_iso(ms: Any) -> Optional[str]:
    try:
        value = int(ms)
    except (TypeError, ValueError):
        return None
    if value > 10_000_000_000:      # seconds, not milliseconds
        value *= 1000
    from datetime import datetime, timezone
    return datetime.fromtimestamp(value / 1000, tz=timezone.utc).replace(
        microsecond=0, tzinfo=None).isoformat()


def _clean(body: str) -> str:
    return " ".join(str(body).replace("\r", "").split())


def parse_google_txt(text: str) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for part in SEP_RE.split(text):
        m = GOOGLE_TXT_BLOCK.search(part)
        if not m:
            continue
        body = _clean(part[m.end():])
        if not body:
            continue
        out.append({"sender": m.group("sender").strip(),
                    "received_at": f"{m.group('date')}T{m.group('time')}",
                    "body": body, "origin": "received"})
    return out


def parse_samsung_xml(text: str) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    try:
        root = ET.fromstring(text)
    except ET.ParseError:
        return out
    for node in root.iter("sms"):
        body = _clean(node.get("body") or "")
        if not body:
            continue
        # Samsung type: 1 inbox, 2 sent, 3 draft, 4 outbox, 5 failed, 6 queued
        stype = (node.get("type") or "1").strip()
        out.append({"sender": (node.get("address") or "").strip(),
                    "received_at": _epoch_to_iso(node.get("date")),
                    "body": body,
                    "origin": "sent" if stype in ("2", "4", "5", "6") else "received"})
    return out


def parse_google_json(text: str) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return out
    if isinstance(data, dict):
        data = data.get("messages") or data.get("sms") or []
    for node in data:
        if not isinstance(node, dict):
            continue
        body = _clean(node.get("body") or node.get("message") or "")
        if not body:
            continue
        mtype = node.get("type")
        out.append({"sender": str(node.get("address") or node.get("sender") or "").strip(),
                    "received_at": _epoch_to_iso(node.get("date") or node.get("timestamp")),
                    "body": body,
                    "origin": "sent" if str(mtype) in ("2", "4", "5", "6") else "received"})
    return out


def _pick(row: dict[str, str], keys: Iterable[str]) -> Optional[str]:
    lowered = {str(k).strip().lower(): v for k, v in row.items()}
    for k in keys:
        if k in lowered and str(lowered[k]).strip():
            return str(lowered[k]).strip()
    return None


def parse_delimited(text: str, delimiter: str) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    reader = csv.DictReader(io.StringIO(text), delimiter=delimiter)
    for row in reader:
        body = _pick(row, _CSV_BODY_KEYS)
        if not body:
            continue
        out.append({"sender": _pick(row, _CSV_SENDER_KEYS) or "",
                    "received_at": _pick(row, _CSV_DATE_KEYS),
                    "body": _clean(body),
                    "origin": "received"})
    return out


def parse_export(text: str, fmt: Optional[str] = None, filename: str = "") -> list[dict[str, Any]]:
    """Parse an export into records. The format is detected when not given."""
    fmt = fmt or detect_format(text, filename)
    if fmt == "google_txt":
        return parse_google_txt(text)
    if fmt == "samsung_xml":
        return parse_samsung_xml(text)
    if fmt == "google_json":
        return parse_google_json(text)
    if fmt == "csv":
        return parse_delimited(text, ",")
    if fmt == "tsv":
        return parse_delimited(text, "\t")
    return []


def parse_file(path: str, fmt: Optional[str] = None) -> list[dict[str, Any]]:
    """Read an export file and return its records."""
    with open(path, encoding="utf-8", errors="replace") as fh:
        text = fh.read()
    return parse_export(text, fmt=fmt, filename=os.path.basename(path))


def to_jsonl(records: list[dict[str, Any]], user_id: str = "REAL01",
             label: Optional[str] = None) -> list[dict[str, Any]]:
    """Attach an id, a user and an optional label to each record."""
    out = []
    for i, r in enumerate(records):
        row = dict(id=i, user_id=user_id, **r)
        if label:
            row["label"] = label
        out.append(row)
    return out


__all__ = ["detect_format", "parse_export", "parse_file", "parse_google_txt",
           "parse_samsung_xml", "parse_google_json", "parse_delimited", "to_jsonl"]
