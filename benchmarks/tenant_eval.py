"""BENCH-10: multi-tenant isolation, correctness and audit separation.

Pre-registered bars in benchmarks/PROTOCOL.md.

Usage:
    PYTHONPATH=. python3 benchmarks/tenant_eval.py

License: Apache-2.0
Copyright 2026 Michael Kupermann
"""

import json
import sqlite3
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from finetune_eval import _norm, make_eval_set  # noqa: E402

from souprise import RAGConfig, SoupriseRAG  # noqa: E402
from souprise.core.audit import AuditLog  # noqa: E402
from souprise.core.hdc import SimpleHDCRetriever  # noqa: E402
from souprise.core.tenants import TenantManager  # noqa: E402
from souprise.data.generators.business import generate_business_data  # noqa: E402


def build(mgr, name, seed):
    """One tenant with its own corpus. Same generator, different seed, so
    entity names collide across tenants but values differ."""
    tenant = mgr.create(name)
    entries = generate_business_data(n=1000, seed=seed)
    index_entries = [e.to_retrieval_format() for e in entries]
    retriever = SimpleHDCRetriever()
    retriever.index(index_entries)
    retriever.save(tenant.index_path)
    return tenant, entries, index_entries


def main():
    base = tempfile.mkdtemp()
    mgr = TenantManager(base)
    acme, acme_entries, acme_idx = build(mgr, "acme", 7)
    globex, globex_entries, globex_idx = build(mgr, "globex", 8)

    # Values unique to each tenant (exclude figures both corpora share)
    def values_of(index_entries):
        vals = set()
        for e in index_entries:
            for line in e["text"].splitlines():
                if line.startswith(("Amount: ", "Annual Revenue: ")):
                    vals.add(_norm(line.split(": ", 1)[1]).lstrip("$"))
        return vals

    acme_vals, globex_vals = values_of(acme_idx), values_of(globex_idx)
    only_globex = globex_vals - acme_vals
    only_acme = acme_vals - globex_vals
    globex_blob = _norm(" ".join(e["text"] for e in globex_idx))
    acme_blob = _norm(" ".join(e["text"] for e in acme_idx))

    def rag_for(tenant):
        rag = SoupriseRAG(RAGConfig(retriever="simple",
                                    answer_mode="verified",
                                    audit_path=tenant.audit_path))
        rag.retriever = SimpleHDCRetriever.load(tenant.index_path)
        return rag

    # T1: cross-tenant leaks + T2: own-value accuracy, both directions
    leaks = 0
    t2 = {}
    n_queries = {}
    for tenant, entries, foreign_vals, foreign_blob, label in (
            (acme, acme_entries, only_globex, globex_blob, "acme"),
            (globex, globex_entries, only_acme, acme_blob, "globex")):
        rag = rag_for(tenant)
        eval_set = make_eval_set(entries, 100, 13)
        correct = answered = 0
        for item in eval_set[:100]:
            result = rag.query(item["question"])
            blob = _norm(result.answer + " ".join(
                r.title + " " + r.content for r in result.retrieval_results))
            if any(v in blob for v in foreign_vals):
                leaks += 1
            if not result.refused:
                answered += 1
                correct += item["expected"] in _norm(result.answer)
        t2[label] = correct / answered if answered else 0.0
        n_queries[label] = 100
        print(f"{label}: {answered} answered, accuracy "
              f"{t2[label]:.3f}, running leak count {leaks}")

    # T3: audit separation and immutability
    a_count = AuditLog(acme.audit_path).count()
    g_count = AuditLog(globex.audit_path).count()
    t3_counts = (a_count == n_queries["acme"] and g_count == n_queries["globex"])
    t3_immutable = True
    for path in (acme.audit_path, globex.audit_path):
        con = sqlite3.connect(path)
        try:
            con.execute("DELETE FROM events")
            t3_immutable = False
        except sqlite3.DatabaseError:
            pass
        finally:
            con.close()
    print(f"T3 audit events acme={a_count} globex={g_count}, "
          f"immutable={t3_immutable}")

    results = {
        "t1_cross_tenant_leaks": leaks,
        "t2_accuracy_acme": round(t2["acme"], 4),
        "t2_accuracy_globex": round(t2["globex"], 4),
        "t3_counts_separate": bool(t3_counts),
        "t3_immutable": bool(t3_immutable),
        "bars": {"t1": "= 0", "t2": "= 1.000 each", "t3": "all true"},
        "verdicts": {
            "t1": "pass" if leaks == 0 else "fail",
            "t2": ("pass" if t2["acme"] == 1.0 and t2["globex"] == 1.0
                   else "fail"),
            "t3": "pass" if (t3_counts and t3_immutable) else "fail",
        },
    }
    Path("benchmarks/results").mkdir(parents=True, exist_ok=True)
    Path("benchmarks/results/tenant_eval.json").write_text(
        json.dumps(results, indent=2))
    print(json.dumps(results["verdicts"], indent=2))


if __name__ == "__main__":
    main()
