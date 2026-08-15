"""Tuned-vs-untuned evaluation on point-lookup questions.

Protocol and bars are pre-registered in benchmarks/PROTOCOL.md. Retrieval
context is computed once and reused identically for both models; an answer
counts as correct iff the expected value (normalized number or status word)
appears in the output.

Usage:
    PYTHONPATH=. python3 benchmarks/finetune_eval.py \
        --tuned ./souprise_model --base mlx-community/Qwen2.5-0.5B-Instruct-4bit

License: Apache-2.0
Copyright 2026 Michael Kupermann
"""

import argparse
import json
import random
import re
from pathlib import Path

from souprise.core.hdc import SimpleHDCRetriever
from souprise.data.generators.business import generate_business_data


def _field(content: str, key: str) -> str:
    for line in content.splitlines():
        if line.startswith(key + ":"):
            return line.split(": ", 1)[1]
    return ""


def _norm(text: str) -> str:
    return re.sub(r"[,$]", "", text.lower())


def make_eval_set(entries, n_questions: int, seed: int):
    rng = random.Random(seed)
    pool = list(range(len(entries)))
    rng.shuffle(pool)
    items = []
    for idx in pool:
        if len(items) >= n_questions:
            break
        e = entries[idx]
        parts = e.title.split()
        if e.title.startswith("Invoice "):
            customer, month, year = parts[1], parts[2], parts[3]
            if rng.random() < 0.5:
                q = f"What is the amount of the invoice for {customer} from {month} {year}?"
                expected = _norm(_field(e.content, "Amount")).lstrip("$")
            else:
                q = f"What is the status of the invoice for {customer} from {month} {year}?"
                expected = _norm(_field(e.content, "Status"))
        elif e.title.startswith("Customer Profile"):
            customer = parts[2]
            q = f"What is the annual revenue of {customer}?"
            expected = _norm(_field(e.content, "Annual Revenue")).lstrip("$")
        elif e.title.startswith("Product ") and "Metrics" in e.title:
            product = parts[1]
            q = f"How many units of {product} are in stock?"
            expected = _norm(_field(e.content, "Stock")).replace(" units", "")
        else:
            continue
        items.append({"question": q, "expected": expected, "target": e.title})
    return items


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tuned", required=True, help="Path to the fine-tuned model")
    parser.add_argument("--base", default="mlx-community/Qwen2.5-0.5B-Instruct-4bit")
    parser.add_argument("--adapter", default=None,
                        help="LoRA adapter path (if tuned model is an adapter)")
    parser.add_argument("--corpus", type=int, default=2000)
    parser.add_argument("--corpus-seed", type=int, default=123)
    parser.add_argument("--questions", type=int, default=60)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--k", type=int, default=3)
    parser.add_argument("--out", default="benchmarks/results")
    args = parser.parse_args()

    entries = generate_business_data(n=args.corpus, seed=args.corpus_seed)
    retriever = SimpleHDCRetriever()
    retriever.index([e.to_retrieval_format() for e in entries])
    eval_set = make_eval_set(entries, args.questions, args.seed)
    print(f"{len(eval_set)} questions over {args.corpus} records")

    # Retrieval context once, reused for both models
    contexts = []
    for item in eval_set:
        results = retriever.search(item["question"], k=args.k)
        contexts.append("\n\n".join(f"--- {r.title} ---\n{r.content}" for r in results))

    from mlx_lm import generate, load
    from mlx_lm.sample_utils import make_sampler

    def run(model_spec, adapter=None):
        model, tokenizer = (load(model_spec, adapter_path=adapter)
                            if adapter else load(model_spec))
        sampler = make_sampler(temp=0.0)  # deterministic for eval
        correct = 0
        for item, context in zip(eval_set, contexts):
            prompt = (f"RECORDS:\n{context}\nEND OF RECORDS\n\n"
                      f"QUESTION: {item['question']}\n"
                      f"ANSWER (based only on the records above):")
            answer = generate(model, tokenizer, prompt=prompt,
                              max_tokens=80, sampler=sampler)
            if item["expected"] in _norm(answer):
                correct += 1
        return correct / len(eval_set)

    print("evaluating untuned base model...")
    base_acc = run(args.base)
    print(f"untuned accuracy: {base_acc:.3f}")
    print("evaluating tuned model...")
    tuned_acc = run(args.tuned) if not args.adapter else run(args.base, args.adapter)
    print(f"tuned accuracy:   {tuned_acc:.3f}")

    delta = tuned_acc - base_acc
    verdict = "adds_value" if delta >= 0.05 else "no_measurable_benefit"
    print(f"delta: {delta:+.3f}  verdict: {verdict}")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "finetune_eval.json").write_text(json.dumps({
        "questions": len(eval_set), "corpus": args.corpus,
        "untuned_accuracy": round(base_acc, 4),
        "tuned_accuracy": round(tuned_acc, 4),
        "delta": round(delta, 4),
        "bar": "adds value iff delta >= +0.05",
        "verdict": verdict,
    }, indent=2))


if __name__ == "__main__":
    main()
