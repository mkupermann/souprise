"""BENCH-5 evaluation: verified answer mode per benchmarks/PROTOCOL.md.

V1 value accuracy on a deduplicated corpus (bar: 1.000)
V2 wrong-value rate under entity ambiguity (bar: 0.000)
V3 refusal rate on unknown entities (bar: 1.000)
V4 shipped ungrounded-figure rate in gated generative mode (bar: 0.000)

Usage:
    PYTHONPATH=. python3 benchmarks/verified_eval.py [--skip-generative]

License: Apache-2.0
Copyright 2026 Michael Kupermann
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from finetune_eval import _norm, make_eval_set  # noqa: E402

from souprise import RAGConfig, SoupriseRAG  # noqa: E402
from souprise.data.generators.business import generate_business_data  # noqa: E402


def dedupe(entries):
    seen = {}
    for e in entries:
        seen[e.title] = e
    return list(seen.values())


def build_rag(entries, mode):
    rag = SoupriseRAG(RAGConfig(
        retriever="simple", answer_mode=mode, retrieval_k=5,
        model_path="mlx-community/Qwen2.5-0.5B-Instruct-4bit"))
    rag.index_from_entries([e.to_retrieval_format() for e in entries])
    return rag


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-generative", action="store_true")
    parser.add_argument("--out", default="benchmarks/results")
    args = parser.parse_args()

    raw = generate_business_data(n=2000, seed=123)
    deduped = dedupe(raw)

    # V1: value accuracy on the deduplicated corpus
    eval_set = make_eval_set(deduped, 60, 11)
    rag = build_rag(deduped, "verified")
    v1_correct = 0
    for item in eval_set:
        result = rag.query(item["question"])
        v1_correct += (not result.refused
                       and item["expected"] in _norm(result.answer))
    v1 = v1_correct / len(eval_set)
    print(f"V1 value accuracy (deduped): {v1:.3f}")

    # V2: wrong-value rate on the ambiguous raw corpus. A failure is
    # asserting a single incorrect value; correct value, an explicit
    # multi-candidate listing, or a refusal all pass.
    eval_raw = make_eval_set(raw, 60, 11)
    rag_raw = build_rag(raw, "verified")
    wrong = 0
    for item in eval_raw:
        result = rag_raw.query(item["question"])
        if result.refused or result.ambiguous:
            continue
        if item["expected"] not in _norm(result.answer):
            wrong += 1
    v2 = wrong / len(eval_raw)
    print(f"V2 wrong-value rate (ambiguous): {v2:.3f}")

    # V3: refusal on unknown entities
    unknown_questions = [
        f"What is the amount of the invoice for Zorblatt_{i:03d} from Jan 2031?"
        for i in range(10)
    ] + [
        f"What is the annual revenue of Phantom_{i:03d}?" for i in range(10)
    ]
    refusals = 0
    for question in unknown_questions:
        result = rag.query(question)
        refusals += result.refused
    v3 = refusals / len(unknown_questions)
    print(f"V3 refusal rate (unknown entities): {v3:.3f}")

    results = {
        "v1_value_accuracy_deduped": round(v1, 4),
        "v2_wrong_value_rate_ambiguous": round(v2, 4),
        "v3_refusal_rate_unknown": round(v3, 4),
        "bars": {"v1": "= 1.000", "v2": "= 0.000", "v3": "= 1.000",
                 "v4": "= 0.000"},
    }

    # V4: gated generative mode with the real local model
    if not args.skip_generative:
        rag_gen = build_rag(deduped, "generative")
        rag_gen.load_model()
        shipped_ungrounded = 0
        for item in eval_set:
            result = rag_gen.query(item["question"])
            shipped_ungrounded += bool(result.ungrounded_numbers)
        v4 = shipped_ungrounded / len(eval_set)
        results["v4_shipped_ungrounded_rate"] = round(v4, 4)
        print(f"V4 shipped ungrounded rate (gated generative): {v4:.3f}")

    results["verdicts"] = {
        "v1": "pass" if results["v1_value_accuracy_deduped"] == 1.0 else "fail",
        "v2": "pass" if results["v2_wrong_value_rate_ambiguous"] == 0.0 else "fail",
        "v3": "pass" if results["v3_refusal_rate_unknown"] == 1.0 else "fail",
        "v4": ("pass" if results.get("v4_shipped_ungrounded_rate", 1) == 0.0
               else "fail" if "v4_shipped_ungrounded_rate" in results
               else "skipped"),
    }
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "verified_eval.json").write_text(json.dumps(results, indent=2))
    print(json.dumps(results["verdicts"], indent=2))


if __name__ == "__main__":
    main()
