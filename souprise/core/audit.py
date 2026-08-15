"""Append-only audit log: every query, provable, immutable.

Each query writes one structured event with the question, the route, the
records used (ids and content hashes), an answer hash, latencies and the
policy in force. SQLite triggers reject UPDATE and DELETE, so the log is
append-only at the database level, not by convention.

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
                " generation_ms REAL);"
                "CREATE TRIGGER IF NOT EXISTS events_no_update"
                " BEFORE UPDATE ON events BEGIN"
                " SELECT RAISE(ABORT, 'audit log is append-only'); END;"
                "CREATE TRIGGER IF NOT EXISTS events_no_delete"
                " BEFORE DELETE ON events BEGIN"
                " SELECT RAISE(ABORT, 'audit log is append-only'); END;"
            )
            con.commit()
        finally:
            con.close()

    def record(self, result, principal: Optional[str] = None) -> int:
        """Append one event for a RAGResult. Returns the event id."""
        record_ids = [r.title for r in result.retrieval_results]
        record_hashes = [_sha256(r.content) for r in result.retrieval_results]
        con = sqlite3.connect(self.path)
        try:
            cur = con.execute(
                "INSERT INTO events (ts, principal, policy, question,"
                " answer_path, refused, policy_denied, blocked_generation,"
                " record_ids, record_hashes, answer_sha256, retrieval_ms,"
                " generation_ms) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (datetime.now(timezone.utc).isoformat(), principal,
                 result.policy, result.question, result.answer_path,
                 int(result.refused), int(result.policy_denied),
                 int(result.blocked_generation is not None),
                 json.dumps(record_ids), json.dumps(record_hashes),
                 _sha256(result.answer),
                 result.retrieval_latency * 1000,
                 result.generation_latency * 1000),
            )
            con.commit()
            return int(cur.lastrowid)
        finally:
            con.close()

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
