"""BENCH-3: fine-tuning failure analysis per benchmarks/PROTOCOL.md.

Runs the pre-registered scenarios:
  A  error decomposition of the BENCH-2 setting (per-question dump)
  S1 harder contexts (corpus 5,000, k=5)
  S2 format fidelity (concise correctness, computed from the same dumps)
  S3 memorization negative control (no records in the prompt)

Usage:
    PYTHONPATH=. python3 benchmarks/finetune_analysis.py \
        --adapter ./souprise_model

License: Apache-2.0
Copyright 2026 Michael Kupermann
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from finetune_eval import _norm, make_eval_set  # noqa: E402

from souprise.core.hdc import SimpleHDCRetriever
from souprise.core.pipeline import wrap_chat  # noqa: E402
from souprise.data.generators.business import generate_business_data  # noqa: E402


def build(corpus_n, corpus_seed, questions, qseed, k):
    entries = generate_business_data(n=corpus_n, seed=corpus_seed)
    retriever = SimpleHDCRetriever()
    retriever.index([e.to_retrieval_format() for e in entries])
    eval_set = make_eval_set(entries, questions, qseed)
    contexts, target_in_context = [], []
    for item in eval_set:
        results = retriever.search(item["question"], k=k)
        contexts.append("\n\n".join(f"--- {r.title} ---\n{r.content}" for r in results))
        target_in_context.append(any(r.title == item["target"] for r in results))
    return eval_set, contexts, target_in_context


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", default="mlx-community/Qwen2.5-0.5B-Instruct-4bit")
    parser.add_argument("--adapter", required=True)
    parser.add_argument("--out", default="benchmarks/results")
    args = parser.parse_args()

    from mlx_lm import generate, load
    from mlx_lm.sample_utils import make_sampler

    sampler = make_sampler(temp=0.0)
    models = {}

    def answer(tag, prompt):
        if tag not in models:
            models[tag] = (load(args.base, adapter_path=args.adapter)
                           if tag == "tuned" else load(args.base))
        model, tokenizer = models[tag]
        return generate(model, tokenizer, prompt=wrap_chat(tokenizer, prompt), max_tokens=80,
                        sampler=sampler)

    def run(scenario, eval_set, contexts, with_context=True):
        out = {}
        for tag in ("untuned", "tuned"):
            rows = []
            for item, context in zip(eval_set, contexts):
                if with_context:
                    prompt = (f"RECORDS:\n{context}\nEND OF RECORDS\n\n"
                              f"QUESTION: {item['question']}\n"
                              f"ANSWER (based only on the records above):")
                else:
                    prompt = (f"QUESTION: {item['question']}\nANSWER:")
                text = answer(tag, prompt)
                norm = _norm(text)
                first20 = " ".join(norm.split()[:20])
                rows.append({
                    "question": item["question"],
                    "expected": item["expected"],
                    "target": item["target"],
                    "answer": text[:300],
                    "correct": item["expected"] in norm,
                    "concise_correct": item["expected"] in first20,
                })
            acc = sum(r["correct"] for r in rows) / len(rows)
            concise = sum(r["concise_correct"] for r in rows) / len(rows)
            print(f"{scenario:8s} {tag:8s} acc={acc:.3f} concise={concise:.3f}")
            out[tag] = {"accuracy": round(acc, 4),
                        "concise_accuracy": round(concise, 4), "rows": rows}
        return out

    # A + S2 base setting: BENCH-2 replication with dumps
    eval_a, ctx_a, tic_a = build(2000, 123, 60, 11, 3)
    print(f"A/base: {len(eval_a)} questions, target in context for "
          f"{sum(tic_a)}/{len(tic_a)}")
    a = run("base", eval_a, ctx_a)

    # Error decomposition on the base setting
    decomp = {}
    for tag in ("untuned", "tuned"):
        rows = a[tag]["rows"]
        misses = [i for i, r in enumerate(rows) if not r["correct"]]
        decomp[tag] = {
            "misses": len(misses),
            "retrieval_miss": sum(1 for i in misses if not tic_a[i]),
            "extraction_failure": sum(1 for i in misses if tic_a[i]),
            "miss_indices": misses,
        }
    both = set(decomp["untuned"]["miss_indices"]) & set(decomp["tuned"]["miss_indices"])
    decomp["shared_misses"] = len(both)

    # S1 harder contexts
    eval_s1, ctx_s1, tic_s1 = build(5000, 123, 60, 11, 5)
    s1 = run("S1", eval_s1, ctx_s1)

    # S3 memorization control: same questions as base, no context
    s3 = run("S3", eval_a, ["" for _ in eval_a], with_context=False)

    results = {
        "protocol": "benchmarks/PROTOCOL.md BENCH-3",
        "base": {t: {k2: v for k2, v in a[t].items() if k2 != "rows"} for t in a},
        "decomposition": decomp,
        "s1_harder": {t: {k2: v for k2, v in s1[t].items() if k2 != "rows"} for t in s1},
        "s2_concise_base": {t: a[t]["concise_accuracy"] for t in a},
        "s3_no_context": {t: {k2: v for k2, v in s3[t].items() if k2 != "rows"} for t in s3},
        "bars": {
            "s1": "tuned - untuned >= +0.05 (accuracy)",
            "s2": "tuned - untuned >= +0.05 (concise accuracy, base setting)",
            "s3": "memorization flagged iff tuned - untuned >= +0.10 without context",
        },
    }
    results["verdicts"] = {
        "s1": ("helps" if s1["tuned"]["accuracy"] - s1["untuned"]["accuracy"] >= 0.05
               else "no_benefit"),
        "s2": ("helps" if a["tuned"]["concise_accuracy"] - a["untuned"]["concise_accuracy"] >= 0.05
               else "no_benefit"),
        "s3_memorization": ("flagged" if s3["tuned"]["accuracy"] - s3["untuned"]["accuracy"] >= 0.10
                            else "not_detected"),
    }

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "finetune_analysis.json").write_text(json.dumps(results, indent=2))
    dumps = {"base": a, "s1": s1, "s3": s3}
    (out / "finetune_analysis_dumps.json").write_text(json.dumps(dumps, indent=2))
    print(json.dumps(results["verdicts"], indent=2))


if __name__ == "__main__":
    main()
