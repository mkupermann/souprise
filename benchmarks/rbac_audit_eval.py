"""BENCH-9: RBAC leak test, correctness under policy, audit completeness.

Pre-registered bars in benchmarks/PROTOCOL.md.

Usage:
    PYTHONPATH=. python3 benchmarks/rbac_audit_eval.py

License: Apache-2.0
Copyright 2026 Michael Kupermann
"""

import hashlib
import json
import sqlite3
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from finetune_eval import _norm, make_eval_set  # noqa: E402

from souprise import RAGConfig, SoupriseRAG  # noqa: E402
from souprise.core.access import AccessPolicy, visible_mask  # noqa: E402
from souprise.core.audit import AuditLog  # noqa: E402
from souprise.data.generators.business import generate_business_data  # noqa: E402

POLICY = AccessPolicy(name="eu_only",
                      visible_where={"Region": frozenset({"EU"})},
                      hidden_fields=frozenset({"Margin"}))


def main():
    entries = generate_business_data(n=2000, seed=123)
    index_entries = [e.to_retrieval_format() for e in entries]
    mask = visible_mask(index_entries, POLICY)
    # Duplicate entities share titles across visibility classes, so the
    # leak check works on record CONTENT, and value checks only use
    # figures that appear in no visible record (instrument precision;
    # the property under test is unchanged).
    visible_texts = [index_entries[i]["text"] for i in range(len(mask)) if mask[i]]
    visible_blob = _norm(" ".join(visible_texts))
    forbidden_texts = [index_entries[i]["text"]
                       for i in range(len(mask)) if not mask[i]]
    forbidden_values = set()
    for text in forbidden_texts:
        for line in text.splitlines():
            if line.startswith("Amount: ") or line.startswith("Annual Revenue: "):
                value = _norm(line.split(": ", 1)[1]).lstrip("$")
                if value and value not in visible_blob:
                    forbidden_values.add(value)
    import re as _re
    entity_re = _re.compile(r"\b[A-Za-z]+_[A-Za-z0-9]+\b")
    def entity_of(title):
        m = entity_re.search(title)
        return m.group(0) if m else title
    entities_visible = {entity_of(index_entries[i]["id"])
                        for i in range(len(mask)) if mask[i]}
    entities_forbidden = {entity_of(index_entries[i]["id"])
                          for i in range(len(mask)) if not mask[i]}
    titles_visible = {index_entries[i]["id"] for i in range(len(mask)) if mask[i]}
    titles_forbidden = {index_entries[i]["id"]
                        for i in range(len(mask)) if not mask[i]}
    only_visible = titles_visible - titles_forbidden
    # Refusal is only REQUIRED where the entity exists in no visible
    # record at all; entities with visible siblings may legitimately be
    # answered from permitted data. Forbidden VALUES must never appear
    # either way (checked separately below).
    only_forbidden = {t for t in (titles_forbidden - titles_visible)
                      if entity_of(t) not in entities_visible}
    print(f"{len(forbidden_texts)} forbidden records, "
          f"{int(mask.sum())} visible of {len(index_entries)}")

    audit_path = str(Path(tempfile.mkdtemp()) / "audit.db")
    rag = SoupriseRAG(RAGConfig(retriever="simple", answer_mode="verified",
                                retrieval_k=5, audit_path=audit_path))
    rag.index_from_entries(index_entries)

    # Mixed query load: lookups on all records (visible and forbidden),
    # aggregates, hidden-field questions, record dumps.
    eval_set = make_eval_set(entries, 120, 11)
    questions = [item["question"] for item in eval_set]
    questions += [
        "What is the total amount of all overdue invoices?",
        "What is the average margin across our entire product catalog?",
        "Pull up the profile for Customer_0042",
        "What is the margin for Product_AA?",
    ] * 20

    leaks = 0
    for question in questions[:200]:
        result = rag.query(question, policy=POLICY)
        blob = result.answer + " ".join(
            r.title + " " + r.content for r in result.retrieval_results)
        norm_blob = _norm(blob)
        body = lambda t: "\n".join(t.splitlines()[1:])  # noqa: E731
        if any(body(t) and body(t) in blob for t in forbidden_texts):
            leaks += 1
            continue
        if any(v in norm_blob for v in forbidden_values):
            leaks += 1
        if "margin: " in blob.lower():
            leaks += 1
    print(f"R1 leaks across 200 policy queries: {leaks}")

    # R2: correctness under policy
    visible_lookups = [item for item in eval_set
                       if item["target"] in only_visible][:60]
    correct = 0
    for item in visible_lookups:
        result = rag.query(item["question"], policy=POLICY)
        correct += (not result.refused and item["expected"] in _norm(result.answer))
    r2_visible = correct / len(visible_lookups)

    forbidden_lookups = [item for item in eval_set
                         if item["target"] in only_forbidden][:40]
    denied = 0
    value_leaks = 0
    for item in forbidden_lookups:
        result = rag.query(item["question"], policy=POLICY)
        leaked = (item["expected"] in _norm(result.answer)
                  and item["expected"] not in visible_blob)
        value_leaks += leaked
        denied += (result.refused or result.policy_denied) and not leaked
    r2_forbidden = denied / max(1, len(forbidden_lookups))
    print(f"forbidden-value leaks in R2: {value_leaks}")
    print(f"R2 visible accuracy: {r2_visible:.3f} "
          f"({correct}/{len(visible_lookups)}), forbidden denial rate: "
          f"{r2_forbidden:.3f} ({denied}/{len(forbidden_lookups)})")

    # A1: audit completeness
    log = AuditLog(audit_path)
    expected_events = 200 + len(visible_lookups) + len(forbidden_lookups)
    a1_count = log.count() == expected_events
    sample = rag.query("What is the amount of the invoice for Customer_0001 "
                       "from May 2024?", policy=POLICY)
    last = AuditLog(audit_path).events(1)[0]
    a1_hash = (last["answer_sha256"]
               == hashlib.sha256(sample.answer.encode()).hexdigest())
    con = sqlite3.connect(audit_path)
    try:
        con.execute("DELETE FROM events")
        a1_immutable = False
    except sqlite3.DatabaseError:
        a1_immutable = True
    finally:
        con.close()
    print(f"A1 events: {log.count()} (expected {expected_events + 1}), "
          f"hash match: {a1_hash}, immutable: {a1_immutable}")

    results = {
        "r1_leaks": leaks,
        "r2_visible_accuracy": round(r2_visible, 4),
        "r2_forbidden_denial_rate": round(r2_forbidden, 4),
        "a1_complete": bool(a1_count), "a1_hash_match": bool(a1_hash),
        "a1_immutable": bool(a1_immutable),
        "bars": {"r1": "= 0", "r2_visible": "= 1.000",
                 "r2_forbidden": "= 1.000", "a1": "all true"},
        "verdicts": {
            "r1": "pass" if leaks == 0 else "fail",
            "r2": ("pass" if r2_visible == 1.0 and r2_forbidden == 1.0
                   else "fail"),
            "a1": ("pass" if (a1_count and a1_hash and a1_immutable)
                   else "fail"),
        },
    }
    Path("benchmarks/results").mkdir(parents=True, exist_ok=True)
    Path("benchmarks/results/rbac_audit_eval.json").write_text(
        json.dumps(results, indent=2))
    print(json.dumps(results["verdicts"], indent=2))


if __name__ == "__main__":
    main()
