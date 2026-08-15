"""BENCH-6 evaluation per benchmarks/PROTOCOL.md.

C1: exactness of deterministic aggregates against independent ground truth.
C2: shipped figure-mismatch rate in styled mode with the real model.

Usage:
    PYTHONPATH=. python3 benchmarks/compute_eval.py [--skip-styled]

License: Apache-2.0
Copyright 2026 Michael Kupermann
"""

import argparse
import json
import random
import re
import sys
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from finetune_eval import make_eval_set  # noqa: E402

from souprise import RAGConfig, SoupriseRAG  # noqa: E402
from souprise.core.compute import compute_aggregate  # noqa: E402
from souprise.data.generators.business import generate_business_data  # noqa: E402

_NUM_RE = re.compile(r"-?[\d,]+(?:\.\d+)?")


def ground_truth(entries, field_name, op, status=None):
    """Independent reference computation from the raw entries."""
    values = []
    for e in entries:
        fields = dict(line.split(": ", 1)
                      for line in e.content.splitlines() if ": " in line)
        if status and fields.get("Status", "").lower() != status:
            continue
        raw = fields.get(field_name)
        if raw is None:
            continue
        m = _NUM_RE.search(raw.replace("$", ""))
        if m:
            values.append(Decimal(m.group(0).replace(",", "")))
    if not values:
        return None
    if op == "sum":
        r = sum(values)
    elif op == "avg":
        r = (sum(values) / len(values)).quantize(Decimal("0.01"),
                                                 rounding=ROUND_HALF_UP)
    elif op == "max":
        r = max(values)
    elif op == "min":
        r = min(values)
    else:
        r = Decimal(len(values))
    return f"{r:,}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-styled", action="store_true")
    parser.add_argument("--out", default="benchmarks/results")
    args = parser.parse_args()

    entries = generate_business_data(n=2000, seed=123)
    index_entries = [e.to_retrieval_format() for e in entries]

    # C1: 40 aggregate questions with independent ground truth
    rng = random.Random(3)
    cases = []
    for _ in range(40):
        op_word, op = rng.choice([("total", "sum"), ("average", "avg"),
                                  ("highest", "max"), ("lowest", "min")])
        field_word, field_name = rng.choice([
            ("amount", "Amount"), ("annual revenue", "Annual Revenue"),
            ("stock", "Stock")])
        status = (rng.choice(["overdue", "paid", "open"])
                  if field_name == "Amount" and rng.random() < 0.6 else None)
        question = (f"What is the {op_word} {field_word} of all "
                    f"{status + ' ' if status else ''}"
                    f"{'invoices' if field_name == 'Amount' else 'records'}?")
        cases.append((question, field_name, op, status))

    exact = 0
    for question, field_name, op, status in cases:
        computed = compute_aggregate(question, index_entries)
        truth = ground_truth(entries, field_name, op, status)
        if computed is not None and truth is not None and computed.value == truth:
            exact += 1
        elif computed is None and truth is None:
            exact += 1
    c1 = exact / len(cases)
    print(f"C1 aggregate exactness: {c1:.3f} ({exact}/{len(cases)})")

    results = {"c1_exactness": round(c1, 4),
               "bars": {"c1": "= 1.000", "c2": "= 0.000"}}

    # C2: styled mode with the real model
    if not args.skip_styled:
        rag = SoupriseRAG(RAGConfig(
            retriever="simple", answer_mode="styled",
            model_path="mlx-community/Qwen2.5-0.5B-Instruct-4bit",
            temperature=0.0))
        rag.index_from_entries(index_entries)
        rag.load_model()

        deduped = {}
        for e in entries:
            deduped[e.title] = e
        lookups = make_eval_set(list(deduped.values()), 60, 11)
        questions = [item["question"] for item in lookups]
        questions += [c[0] for c in cases[:20]]

        shipped_mismatch = fallbacks = 0
        from souprise.core.pipeline import check_grounding
        for question in questions:
            result = rag.query(question)
            if result.refused:
                continue
            if result.blocked_generation is not None:
                fallbacks += 1
            # Independent recheck of the SHIPPED text
            deterministic = rag._deterministic_answer(
                question, result.retrieval_results)["text"]
            if check_grounding(result.answer, deterministic, question):
                shipped_mismatch += 1
        c2 = shipped_mismatch / len(questions)
        results["c2_shipped_mismatch_rate"] = round(c2, 4)
        results["c2_fallback_rate"] = round(fallbacks / len(questions), 4)
        print(f"C2 shipped mismatch rate: {c2:.3f}  "
              f"(fallback rate: {fallbacks / len(questions):.3f})")

    results["verdicts"] = {
        "c1": "pass" if results["c1_exactness"] == 1.0 else "fail",
        "c2": ("pass" if results.get("c2_shipped_mismatch_rate", 1) == 0.0
               else "fail" if "c2_shipped_mismatch_rate" in results
               else "skipped"),
    }
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "compute_eval.json").write_text(json.dumps(results, indent=2))
    print(json.dumps(results["verdicts"], indent=2))


if __name__ == "__main__":
    main()
