"""Tests for physically isolated tenants.

License: Apache-2.0
Copyright 2026 Michael Kupermann
"""

import pytest

from souprise import RAGConfig, SoupriseRAG
from souprise.core.audit import AuditLog
from souprise.core.hdc import SimpleHDCRetriever
from souprise.core.tenants import TenantError, TenantManager

ACME = [{"id": "Invoice INV_001", "text": "Invoice INV_001\nCustomer: "
         "Shared_Corp\nAmount: $111.00\nStatus: overdue"}]
GLOBEX = [{"id": "Invoice INV_001", "text": "Invoice INV_001\nCustomer: "
           "Shared_Corp\nAmount: $999.00\nStatus: paid"}]


def build_tenant(mgr, name, entries):
    tenant = mgr.create(name)
    retriever = SimpleHDCRetriever()
    retriever.index(entries)
    retriever.save(tenant.index_path)
    return tenant


class TestTenantManager:
    def test_create_get_list(self, tmp_path):
        mgr = TenantManager(str(tmp_path))
        mgr.create("acme")
        mgr.create("globex")
        assert mgr.list() == ["acme", "globex"]
        assert mgr.get("acme").policies_dir.is_dir()

    def test_invalid_names_rejected(self, tmp_path):
        mgr = TenantManager(str(tmp_path))
        for bad in ("../escape", "a/b", "UPPER", "", ".", "a" * 65):
            with pytest.raises(TenantError):
                mgr.create(bad)

    def test_unknown_tenant_rejected(self, tmp_path):
        with pytest.raises(TenantError):
            TenantManager(str(tmp_path)).get("ghost")

    def test_policy_path_stays_inside(self, tmp_path):
        tenant = TenantManager(str(tmp_path)).create("acme")
        with pytest.raises(TenantError):
            tenant.policy_path("../../etc/passwd")


class TestIsolation:
    def test_same_entity_different_values_never_cross(self, tmp_path):
        mgr = TenantManager(str(tmp_path))
        acme = build_tenant(mgr, "acme", ACME)
        globex = build_tenant(mgr, "globex", GLOBEX)

        for tenant, own, other in ((acme, "$111.00", "999"),
                                   (globex, "$999.00", "111.00")):
            rag = SoupriseRAG(RAGConfig(retriever="simple",
                                        answer_mode="verified"))
            rag.retriever = SimpleHDCRetriever.load(tenant.index_path)
            result = rag.query("What is the amount for INV_001?")
            assert own in result.answer
            assert other not in result.answer

    def test_audit_logs_are_separate(self, tmp_path):
        mgr = TenantManager(str(tmp_path))
        acme = build_tenant(mgr, "acme", ACME)
        globex = build_tenant(mgr, "globex", GLOBEX)

        rag = SoupriseRAG(RAGConfig(retriever="simple",
                                    answer_mode="verified",
                                    audit_path=acme.audit_path))
        rag.retriever = SimpleHDCRetriever.load(acme.index_path)
        rag.query("What is the amount for INV_001?")
        rag.query("What is the status of INV_001?")

        rag2 = SoupriseRAG(RAGConfig(retriever="simple",
                                     answer_mode="verified",
                                     audit_path=globex.audit_path))
        rag2.retriever = SimpleHDCRetriever.load(globex.index_path)
        rag2.query("What is the amount for INV_001?")

        assert AuditLog(acme.audit_path).count() == 2
        assert AuditLog(globex.audit_path).count() == 1
