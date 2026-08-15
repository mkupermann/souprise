"""Tests for index-side access policies and the append-only audit log.

License: Apache-2.0
Copyright 2026 Michael Kupermann
"""

import sqlite3

import pytest

from souprise import RAGConfig, SoupriseRAG
from souprise.core.access import AccessPolicy, redact_text, visible_mask
from souprise.core.audit import AuditLog

ENTRIES = [
    {"id": "Invoice EU_01 Jan", "text": "Invoice EU_01 Jan\nCustomer: EU_01\n"
     "Amount: $100.00\nMargin: 21.0%\nStatus: overdue\nRegion: EU"},
    {"id": "Invoice EU_02 Feb", "text": "Invoice EU_02 Feb\nCustomer: EU_02\n"
     "Amount: $250.00\nMargin: 33.0%\nStatus: paid\nRegion: EU"},
    {"id": "Invoice US_01 Mar", "text": "Invoice US_01 Mar\nCustomer: US_01\n"
     "Amount: $999.00\nMargin: 44.0%\nStatus: overdue\nRegion: US"},
]

EU_SALES = AccessPolicy(name="eu_sales",
                        visible_where={"Region": frozenset({"EU"})},
                        hidden_fields=frozenset({"Margin"}))


def build_rag(**cfg):
    rag = SoupriseRAG(RAGConfig(retriever="simple", answer_mode="verified",
                                **cfg))
    rag.index_from_entries(ENTRIES)
    return rag


class TestPolicyPrimitives:
    def test_visible_mask(self):
        mask = visible_mask(ENTRIES, EU_SALES)
        assert list(mask) == [True, True, False]

    def test_redact_text(self):
        out = redact_text(ENTRIES[0]["text"], EU_SALES)
        assert "Margin" not in out
        assert "redacted by policy" in out
        assert "Amount: $100.00" in out


class TestRBACEnforcement:
    def test_forbidden_record_never_retrieved(self):
        rag = build_rag()
        result = rag.query("overdue invoice US_01", policy=EU_SALES)
        titles = [r.title for r in result.retrieval_results]
        assert all("US_01" not in t for t in titles)
        assert "999" not in result.answer
        assert result.policy == "eu_sales"

    def test_forbidden_entity_yields_refusal_not_lookalike(self):
        rag = build_rag()
        result = rag.query("What is the amount for US_01?", policy=EU_SALES)
        assert result.refused
        assert "999" not in result.answer

    def test_hidden_field_denied(self):
        rag = build_rag()
        result = rag.query("What is the margin for EU_01?", policy=EU_SALES)
        assert result.policy_denied
        assert "21.0" not in result.answer

    def test_hidden_field_redacted_from_record_dumps(self):
        rag = build_rag()
        result = rag.query("Pull up the profile for EU_01", policy=EU_SALES)
        assert "Margin" not in result.answer
        assert "Amount: $100.00" in result.answer

    def test_aggregation_respects_visibility(self):
        rag = build_rag()
        result = rag.query("What is the total amount of all overdue invoices?",
                           policy=EU_SALES)
        assert result.computed
        assert "100.00" in result.answer      # EU overdue only
        assert "1,099" not in result.answer   # would include the US record

    def test_visible_lookup_still_exact(self):
        rag = build_rag()
        result = rag.query("What is the amount for EU_02?", policy=EU_SALES)
        assert "$250.00" in result.answer
        assert result.verified

    def test_unrestricted_unchanged(self):
        rag = build_rag()
        result = rag.query("What is the amount for US_01?")
        assert "$999.00" in result.answer


class TestAuditLog:
    def test_every_query_logged_with_hashes(self, tmp_path):
        audit_path = str(tmp_path / "audit.db")
        rag = build_rag(audit_path=audit_path)
        rag.query("What is the amount for EU_01?", policy=EU_SALES)
        rag.query("What is the margin for EU_01?", policy=EU_SALES)  # denial
        rag.query("What is the amount for Zorblatt_9?")              # refusal

        log = AuditLog(audit_path)
        assert log.count() == 3
        events = log.events()
        assert events[0]["policy"] == "eu_sales"
        assert events[1]["policy_denied"] == 1
        assert events[2]["refused"] == 1
        import hashlib
        result = rag.query("What is the amount for EU_01?", policy=EU_SALES)
        assert (AuditLog(audit_path).events()[-1]["answer_sha256"]
                == hashlib.sha256(result.answer.encode()).hexdigest())

    def test_append_only_enforced_by_triggers(self, tmp_path):
        audit_path = str(tmp_path / "audit.db")
        rag = build_rag(audit_path=audit_path)
        rag.query("What is the amount for EU_01?")
        con = sqlite3.connect(audit_path)
        with pytest.raises(sqlite3.DatabaseError):
            con.execute("UPDATE events SET question='tampered' WHERE id=1")
        with pytest.raises(sqlite3.DatabaseError):
            con.execute("DELETE FROM events WHERE id=1")
        con.close()
        assert AuditLog(audit_path).count() == 1
