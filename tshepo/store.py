"""Storage for consent records, collected messages and issued passports.

Everything the service holds about a person lives in one SQLite database, so
that it can be deleted in one place. A consent record is created before a
single message is accepted; a passport is only ever written against a live
consent; deleting the consent deletes the messages and the passport.

    from tshepo.store import Store
    store = Store()
    consent = store.create_consent(user_id="U1", subject_ref="+266...")
    store.insert_messages(consent["id"], "U1", records)
    store.delete_user_data("U1")

Retention is enforced by :meth:`Store.purge_expired`, which removes messages
and passports whose consent has expired or been revoked, and anything older
than the configured retention window.
"""
from __future__ import annotations

import hashlib
import json
import os
import secrets
import sqlite3
import threading
import time
from datetime import datetime, timedelta, timezone
from typing import Any, Iterable, Optional

from .config import CONSENT_TTL_DAYS, DB_PATH, RETENTION_DAYS, ensure_dir
from .logging import get_logger

logger = get_logger("store")

SCHEMA = """
CREATE TABLE IF NOT EXISTS consents (
    id            TEXT PRIMARY KEY,
    user_id       TEXT NOT NULL,
    subject_ref   TEXT,
    purpose       TEXT NOT NULL,
    scope         TEXT,
    source        TEXT,
    token_hash    TEXT NOT NULL,
    granted_at    TEXT NOT NULL,
    expires_at    TEXT NOT NULL,
    revoked_at    TEXT,
    created_at    TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS messages (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    consent_id   TEXT NOT NULL,
    user_id      TEXT NOT NULL,
    body         TEXT NOT NULL,
    sender       TEXT,
    received_at  TEXT,
    created_at   TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS passports (
    user_id     TEXT PRIMARY KEY,
    consent_id  TEXT NOT NULL,
    passport    TEXT NOT NULL,
    created_at  TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS access_log (
    id        INTEGER PRIMARY KEY AUTOINCREMENT,
    actor     TEXT NOT NULL,
    action    TEXT NOT NULL,
    user_id   TEXT,
    detail    TEXT,
    at        TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS message_hashes (
    body_hash TEXT NOT NULL,
    user_id   TEXT NOT NULL,
    created_at TEXT NOT NULL,
    PRIMARY KEY (body_hash, user_id)
);
CREATE INDEX IF NOT EXISTS idx_messages_user ON messages(user_id);
CREATE INDEX IF NOT EXISTS idx_messages_consent ON messages(consent_id);
CREATE INDEX IF NOT EXISTS idx_message_hashes_hash ON message_hashes(body_hash);
"""


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _iso(dt: datetime) -> str:
    return dt.replace(microsecond=0).isoformat()


def hash_token(token: str) -> str:
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


class Store:
    """SQLite-backed store for consents, messages, passports and the access log."""

    def __init__(self, path: Optional[str] = None):
        self.path = path or DB_PATH
        ensure_dir(os.path.dirname(self.path) or ".")
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(self.path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        with self._lock:
            self._conn.executescript(SCHEMA)
            self._conn.commit()

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    # -------------------------------------------------------------- consents
    def create_consent(self, user_id: str, purpose: str = "credit_assessment",
                       subject_ref: Optional[str] = None, scope: Optional[str] = None,
                       source: Optional[str] = None, ttl_days: Optional[int] = None
                       ) -> dict[str, Any]:
        """Create a consent record and return it with its one-time token."""
        consent_id = "cons_" + secrets.token_urlsafe(16)
        token = secrets.token_urlsafe(24)
        now = _now()
        ttl = int(ttl_days if ttl_days is not None else CONSENT_TTL_DAYS)
        expires = now + timedelta(days=ttl)
        with self._lock:
            self._conn.execute(
                "INSERT INTO consents (id, user_id, subject_ref, purpose, scope, source, "
                "token_hash, granted_at, expires_at, created_at) "
                "VALUES (?,?,?,?,?,?,?,?,?,?)",
                (consent_id, user_id, subject_ref, purpose, scope, source,
                 hash_token(token), _iso(now), _iso(expires), _iso(now)))
            self._conn.commit()
        logger.info("consent granted", extra={"user_id": user_id, "consent_id": consent_id,
                                              "expires_at": _iso(expires)})
        return {"id": consent_id, "user_id": user_id, "token": token, "purpose": purpose,
                "scope": scope, "source": source, "granted_at": _iso(now),
                "expires_at": _iso(expires), "revoked_at": None}

    def get_consent(self, consent_id: str) -> Optional[dict[str, Any]]:
        with self._lock:
            row = self._conn.execute("SELECT * FROM consents WHERE id=?", (consent_id,)).fetchone()
        return dict(row) if row else None

    def consent_token_ok(self, consent_id: str, token: str) -> bool:
        c = self.get_consent(consent_id)
        if not c or not token:
            return False
        return secrets.compare_digest(str(c["token_hash"]), hash_token(token))

    def consent_is_live(self, consent_id: str) -> bool:
        """A consent is live when it is granted, not revoked and not expired."""
        c = self.get_consent(consent_id)
        if not c or c["revoked_at"]:
            return False
        return str(c["expires_at"]) > _iso(_now())

    def revoke_consent(self, consent_id: str) -> bool:
        c = self.get_consent(consent_id)
        if not c:
            return False
        with self._lock:
            self._conn.execute("UPDATE consents SET revoked_at=? WHERE id=?",
                               (_iso(_now()), consent_id))
            self._conn.commit()
        logger.info("consent revoked", extra={"consent_id": consent_id, "user_id": c["user_id"]})
        return True

    def list_consents(self, user_id: Optional[str] = None) -> list[dict[str, Any]]:
        q = "SELECT * FROM consents"
        args: tuple = ()
        if user_id:
            q += " WHERE user_id=?"
            args = (user_id,)
        q += " ORDER BY created_at DESC"
        with self._lock:
            rows = self._conn.execute(q, args).fetchall()
        return [dict(r) for r in rows]

    # -------------------------------------------------------------- messages
    def insert_messages(self, consent_id: str, user_id: str,
                        records: Iterable[dict[str, Any]]) -> int:
        now = _iso(_now())
        payload = [(consent_id, user_id, r.get("body") or "", r.get("sender"),
                    r.get("received_at"), now) for r in records]
        if not payload:
            return 0
        with self._lock:
            self._conn.executemany(
                "INSERT INTO messages (consent_id, user_id, body, sender, received_at, created_at) "
                "VALUES (?,?,?,?,?,?)", payload)
            self._conn.commit()
        return len(payload)

    def messages_for(self, user_id: str) -> list[dict[str, Any]]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT id, consent_id, user_id, body, sender, received_at FROM messages "
                "WHERE user_id=? ORDER BY received_at, id", (user_id,)).fetchall()
        return [dict(r) for r in rows]

    def message_count(self, user_id: Optional[str] = None) -> int:
        q = "SELECT COUNT(*) AS n FROM messages"
        args: tuple = ()
        if user_id:
            q += " WHERE user_id=?"
            args = (user_id,)
        with self._lock:
            return int(self._conn.execute(q, args).fetchone()["n"])

    # ------------------------------------------------- message provenance
    def record_message_hashes(self, user_id: str, body_hashes: Iterable[str]) -> int:
        """Remember which messages this applicant submitted.

        The hash says nothing about the message, and it is deleted with the
        rest of the applicant's data.
        """
        now = _iso(_now())
        payload = [(h, str(user_id), now) for h in set(body_hashes) if h]
        if not payload:
            return 0
        with self._lock:
            self._conn.executemany(
                "INSERT OR IGNORE INTO message_hashes (body_hash, user_id, created_at) "
                "VALUES (?,?,?)", payload)
            self._conn.commit()
        return len(payload)

    def shared_message_counts(self, user_id: str, body_hashes: Iterable[str]) -> dict[str, int]:
        """How many of these messages another applicant has already submitted."""
        hashes = [h for h in set(body_hashes) if h]
        if not hashes:
            return {}
        out: dict[str, int] = {}
        with self._lock:
            for start in range(0, len(hashes), 400):
                chunk = hashes[start:start + 400]
                placeholders = ",".join("?" * len(chunk))
                rows = self._conn.execute(
                    f"SELECT user_id, COUNT(*) AS n FROM message_hashes "
                    f"WHERE body_hash IN ({placeholders}) AND user_id != ? GROUP BY user_id",
                    (*chunk, str(user_id))).fetchall()
                for row in rows:
                    out[str(row["user_id"])] = out.get(str(row["user_id"]), 0) + int(row["n"])
        return out

    def clear_message_hashes(self, user_id: str) -> int:
        with self._lock:
            cur = self._conn.execute("DELETE FROM message_hashes WHERE user_id=?", (user_id,))
            self._conn.commit()
        return cur.rowcount

    def delete_messages(self, user_id: str) -> int:
        with self._lock:
            cur = self._conn.execute("DELETE FROM messages WHERE user_id=?", (user_id,))
            self._conn.commit()
            return cur.rowcount

    # -------------------------------------------------------------- passports
    def put_passport(self, user_id: str, consent_id: str, passport: dict[str, Any]) -> None:
        with self._lock:
            self._conn.execute(
                "INSERT INTO passports (user_id, consent_id, passport, created_at) VALUES (?,?,?,?) "
                "ON CONFLICT(user_id) DO UPDATE SET consent_id=excluded.consent_id, "
                "passport=excluded.passport, created_at=excluded.created_at",
                (user_id, consent_id, json.dumps(passport, ensure_ascii=False), _iso(_now())))
            self._conn.commit()

    def get_passport(self, user_id: str) -> Optional[dict[str, Any]]:
        with self._lock:
            row = self._conn.execute("SELECT passport FROM passports WHERE user_id=?",
                                      (user_id,)).fetchone()
        return json.loads(row["passport"]) if row else None

    def delete_passport(self, user_id: str) -> int:
        with self._lock:
            cur = self._conn.execute("DELETE FROM passports WHERE user_id=?", (user_id,))
            self._conn.commit()
            return cur.rowcount

    # -------------------------------------------------------------- erasure
    def delete_user_data(self, user_id: str) -> dict[str, int]:
        """Right to erasure: remove every trace of this person except the
        consent record itself, which is kept as the proof that consent was
        given and then withdrawn (and contains no message content)."""
        with self._lock:
            messages = self._conn.execute("DELETE FROM messages WHERE user_id=?", (user_id,)).rowcount
            passports = self._conn.execute("DELETE FROM passports WHERE user_id=?", (user_id,)).rowcount
            hashes = self._conn.execute("DELETE FROM message_hashes WHERE user_id=?", (user_id,)).rowcount
            self._conn.commit()
        logger.info("user data erased", extra={"user_id": user_id,
                                              "messages": messages, "passports": passports})
        return {"messages": messages, "passports": passports, "message_hashes": hashes}

    def revoke_and_erase(self, consent_id: str) -> dict[str, Any]:
        c = self.get_consent(consent_id)
        if not c:
            return {"ok": False, "reason": "unknown consent"}
        self.revoke_consent(consent_id)
        counts = self.delete_user_data(str(c["user_id"]))
        self.log_access("consent-holder", "revoke_and_erase", str(c["user_id"]),
                        f"consent={consent_id}")
        return {"ok": True, "user_id": c["user_id"], **counts}

    # -------------------------------------------------------------- retention
    def purge_expired(self, retention_days: Optional[int] = None) -> dict[str, Any]:
        """Delete data whose consent expired, was revoked, or aged out."""
        keep_days = int(retention_days if retention_days is not None else RETENTION_DAYS)
        now = _now()
        cutoff = _iso(now - timedelta(days=keep_days))
        expired = [str(r["id"]) for r in self.list_consents()
                   if r["revoked_at"] or str(r["expires_at"]) <= _iso(now)]
        purged_users: set[str] = set()
        with self._lock:
            for cid in expired:
                row = self._conn.execute("SELECT user_id FROM consents WHERE id=?", (cid,)).fetchone()
                if row:
                    purged_users.add(str(row["user_id"]))
            aged = {str(r["user_id"]) for r in
                    self._conn.execute("SELECT DISTINCT user_id FROM messages WHERE created_at < ?",
                                       (cutoff,)).fetchall()}
        for uid in sorted(purged_users | aged):
            self.delete_user_data(uid)
        logger.info("retention purge", extra={"users": len(purged_users | aged),
                                              "expired_consents": len(expired)})
        return {"users_purged": len(purged_users | aged), "expired_consents": len(expired),
                "retention_days": keep_days}

    # -------------------------------------------------------------- audit
    def log_access(self, actor: str, action: str, user_id: Optional[str] = None,
                   detail: Optional[str] = None) -> None:
        with self._lock:
            self._conn.execute(
                "INSERT INTO access_log (actor, action, user_id, detail, at) VALUES (?,?,?,?,?)",
                (actor, action, user_id, detail, _iso(_now())))
            self._conn.commit()

    def access_log(self, limit: int = 100) -> list[dict[str, Any]]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM access_log ORDER BY id DESC LIMIT ?", (limit,)).fetchall()
        return [dict(r) for r in rows]


_store: Optional[Store] = None
_store_lock = threading.Lock()


def get_store(path: Optional[str] = None) -> Store:
    """Process-wide store instance."""
    global _store
    with _store_lock:
        if _store is None:
            _store = Store(path)
        return _store


def reset_store() -> None:
    """Drop the cached store (tests)."""
    global _store
    with _store_lock:
        if _store is not None:
            _store.close()
        _store = None


__all__ = ["Store", "get_store", "reset_store", "hash_token"]
