"""Append-only, tamper-evident audit log.

Each query writes one structured event with the question, the route, the
records used (ids and content hashes), an answer hash, latencies and the
policy in force. SQLite triggers reject UPDATE and DELETE, so the log is
append-only at the database level, not by convention.

Events are additionally hash-chained: every event stores the previous
event's hash and its own hash over the canonical payload plus that
predecessor. Triggers stop the application layer; the chain makes
after-the-fact edits by anyone with file access detectable — verify()
reports the first broken link. Honest scope: this is tamper-EVIDENT,
not tamper-proof; an attacker who rewrites the whole chain from the
first altered event onward is only caught if a chain head was anchored
elsewhere (export the latest event_sha256 to write-once storage for
that).

License: Apache-2.0
Copyright 2026 Michael Kupermann
"""

import hashlib
import json
import sqlite3
from datetime import datetime, timezone
from typing import List, Optional


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


class AuditLog:
    """Append-only query audit log backed by SQLite."""

    def __init__(self, path: str):
        self.path = path
        con = sqlite3.connect(path)
        try:
            con.executescript(
                "CREATE TABLE IF NOT EXISTS events ("
                " id INTEGER PRIMARY KEY AUTOINCREMENT,"
                " ts TEXT NOT NULL,"
                " principal TEXT,"
                " policy TEXT,"
                " question TEXT NOT NULL,"
                " answer_path TEXT,"
                " refused INTEGER,"
                " policy_denied INTEGER,"
                " blocked_generation INTEGER,"
                " record_ids TEXT,"
                " record_hashes TEXT,"
                " answer_sha256 TEXT,"
                " retrieval_ms REAL,"
                " generation_ms REAL,"
                " prev_sha256 TEXT,"
                " event_sha256 TEXT);"
                "CREATE TRIGGER IF NOT EXISTS events_no_update"
                " BEFORE UPDATE ON events BEGIN"
                " SELECT RAISE(ABORT, 'audit log is append-only'); END;"
                "CREATE TRIGGER IF NOT EXISTS events_no_delete"
                " BEFORE DELETE ON events BEGIN"
                " SELECT RAISE(ABORT, 'audit log is append-only'); END;"
            )
            for col in ("prev_sha256", "event_sha256"):
                try:  # logs created before hash chaining get the columns added
                    con.execute(f"ALTER TABLE events ADD COLUMN {col} TEXT")
                except sqlite3.OperationalError:
                    pass
            con.commit()
        finally:
            con.close()

    @staticmethod
    def _event_hash(payload: tuple, prev: str) -> str:
        return _sha256(json.dumps(list(payload), sort_keys=False) + prev)

    def record(self, result, principal: Optional[str] = None) -> int:
        """Append one hash-chained event for a RAGResult. Returns the id."""
        record_ids = [r.title for r in result.retrieval_results]
        record_hashes = [_sha256(r.content) for r in result.retrieval_results]
        payload = (datetime.now(timezone.utc).isoformat(), principal,
                   result.policy, result.question, result.answer_path,
                   int(result.refused), int(result.policy_denied),
                   int(result.blocked_generation is not None),
                   json.dumps(record_ids), json.dumps(record_hashes),
                   _sha256(result.answer),
                   result.retrieval_latency * 1000,
                   result.generation_latency * 1000)
        con = sqlite3.connect(self.path)
        try:
            row = con.execute(
                "SELECT event_sha256 FROM events"
                " ORDER BY id DESC LIMIT 1").fetchone()
            prev = row[0] if row and row[0] else ""
            cur = con.execute(
                "INSERT INTO events (ts, principal, policy, question,"
                " answer_path, refused, policy_denied, blocked_generation,"
                " record_ids, record_hashes, answer_sha256, retrieval_ms,"
                " generation_ms, prev_sha256, event_sha256)"
                " VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                payload + (prev, self._event_hash(payload, prev)),
            )
            con.commit()
            return int(cur.lastrowid)
        finally:
            con.close()

    def verify(self) -> tuple:
        """Walk the hash chain. Returns (ok, first_bad_id or None)."""
        con = sqlite3.connect(self.path)
        try:
            rows = con.execute(
                "SELECT id, ts, principal, policy, question, answer_path,"
                " refused, policy_denied, blocked_generation, record_ids,"
                " record_hashes, answer_sha256, retrieval_ms, generation_ms,"
                " prev_sha256, event_sha256 FROM events ORDER BY id").fetchall()
        finally:
            con.close()
        prev = ""
        for row in rows:
            event_id, payload = row[0], tuple(row[1:14])
            if row[14] != prev or row[15] != self._event_hash(payload, prev):
                return False, event_id
            prev = row[15]
        return True, None

    def events(self, last: int = 50) -> List[dict]:
        """The most recent events, oldest first."""
        con = sqlite3.connect(self.path)
        try:
            con.row_factory = sqlite3.Row
            rows = con.execute(
                "SELECT * FROM events ORDER BY id DESC LIMIT ?", (last,)
            ).fetchall()
            return [dict(r) for r in reversed(rows)]
        finally:
            con.close()

    def count(self) -> int:
        con = sqlite3.connect(self.path)
        try:
            return con.execute("SELECT COUNT(*) FROM events").fetchone()[0]
        finally:
            con.close()
