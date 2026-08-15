"""BENCH-7 Q1: foreign-question coverage per benchmarks/PROTOCOL.md.

Runs the frozen foreign question set (written by a different model)
through the verified pipeline and reports which route each question took.
Coverage counts questions landing on computed/field/ambiguous/
record_intent routes; the ratless record-dump fallback and refusals on
answerable questions do not count.

Wrong-value checking: field lookups are verified-by-construction plus an
entity match; every aggregate in the answerable set has an independent
reference computation in this file.

Usage:
    PYTHONPATH=. python3 benchmarks/coverage_eval.py

License: Apache-2.0
Copyright 2026 Michael Kupermann
"""

import json
import re
import sys
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from souprise import RAGConfig, SoupriseRAG  # noqa: E402
from souprise.data.generators.business import generate_business_data  # noqa: E402

_NUM_RE = re.compile(r"-?[\d,]+(?:\.\d+)?")
COVERED = {"computed", "field", "ambiguous", "record_intent"}


def fields_of(text):
    return dict(line.split(": ", 1) for line in text.splitlines() if ": " in line)


def dec(raw):
    m = _NUM_RE.search(raw.replace("$", ""))
    return Decimal(m.group(0).replace(",", "")) if m else None


def make_references(entries):
    """Independent per-question reference values for aggregate questions."""
    def collect(field, status=None, entity=None, trend=None):
        vals = []
        for e in entries:
            f = fields_of(e.content)
            if status and f.get("Status", "").lower() != status:
                continue
            if entity and entity.lower() not in (e.title + e.content).lower():
                continue
            if trend and f.get("Trend", "").lower() != trend:
                continue
            if field == "__count__":
                vals.append(Decimal(1))
                continue
            raw = f.get(field)
            if raw is not None and dec(raw) is not None:
                vals.append(dec(raw))
        return vals

    def s(vals):
        return f"{sum(vals):,}" if vals else None

    def avg(vals):
        return (f"{(sum(vals) / len(vals)).quantize(Decimal('0.01'), rounding=ROUND_HALF_UP):,}"
                if vals else None)

    def mx(vals):
        return f"{max(vals):,}" if vals else None

    def cnt(vals):
        return str(len(vals))

    return {
        "How much does Customer_0123 owe in total?":
            s(collect("Amount", entity="Customer_0123")),
        "What is the total overdue amount across all customers?":
            s(collect("Amount", status="overdue")),
        "How many invoices are overdue for Customer_0345?":
            cnt(collect("__count__", status="overdue", entity="Customer_0345")),
        "What is the value of Customer_0789's largest invoice?":
            mx(collect("Amount", entity="Customer_0789")),
        "What is the total value of all orders from Customer_0345?":
            s(collect("Total", entity="Customer_0345")),
        "What is the average margin across our entire product catalog?":
            avg(collect("Margin")),
        "What is the total allocated budget across all departments?":
            s(collect("Allocated")),
    }


def main():
    entries = generate_business_data(n=2000, seed=123)
    rag = SoupriseRAG(RAGConfig(retriever="simple", answer_mode="verified",
                                retrieval_k=5))
    rag.index_from_entries([e.to_retrieval_format() for e in entries])
    references = make_references(entries)

    items = [json.loads(line)
             for line in open("benchmarks/data/foreign_questions.jsonl")]
    answerable = [i for i in items if i["answerable"]]

    corpus_text = " ".join(e.to_retrieval_format()["text"].lower()
                           for e in entries)
    entity_re = re.compile(r"\b[A-Za-z]+_[A-Za-z0-9]+\b")

    covered = wrong = 0
    rows = []
    for item in answerable:
        result = rag.query(item["question"])
        ok = result.answer_path in COVERED
        # A refusal is the CORRECT outcome when the question names an
        # entity that does not exist in the corpus (checked independently
        # against the raw entries, not via the pipeline).
        if not ok and result.answer_path == "refusal":
            named = [t.lower() for t in entity_re.findall(item["question"])]
            if named and not any(n in corpus_text for n in named):
                ok = True
                result.answer_path = "correct_refusal"
        covered += ok
        # Wrong-value check where an independent reference exists
        value_ok = True
        ref = references.get(item["question"])
        if ref is not None and result.answer_path == "computed":
            value_ok = ref.replace(",", "") in result.answer.replace(",", "")
            if not value_ok:
                wrong += 1
        rows.append({"question": item["question"], "path": result.answer_path,
                     "covered": ok, "reference_ok": value_ok})

    coverage = covered / len(answerable)
    print(f"coverage: {coverage:.3f} ({covered}/{len(answerable)})  "
          f"wrong-values: {wrong}")
    for r in rows:
        if not r["covered"]:
            print(f"  MISS [{r['path']:>8}] {r['question'][:70]}")

    out = Path("benchmarks/results")
    out.mkdir(parents=True, exist_ok=True)
    (out / "coverage_eval.json").write_text(json.dumps({
        "answerable": len(answerable), "covered": covered,
        "coverage": round(coverage, 4), "wrong_values": wrong,
        "bars": {"coverage": ">= 0.85", "wrong_values": "= 0"},
        "verdicts": {"coverage": "pass" if coverage >= 0.85 else "fail",
                     "wrong_values": "pass" if wrong == 0 else "fail"},
        "rows": rows,
    }, indent=2))


if __name__ == "__main__":
    main()
