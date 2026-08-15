"""BENCH-8 E: encoding v1 vs v2 per benchmarks/PROTOCOL.md.

Measures both encoders on the BENCH-1 paraphrase set, on a deterministic
robustness set (seeded typos in non-entity words, word-order shuffles),
and v2 on the frozen foreign-question coverage set. Adoption rule is
locked in the protocol: v2 must hold everything v1 holds AND win on
robustness, otherwise v1 stays and the loss is published.

Usage:
    PYTHONPATH=. python3 benchmarks/encoding_eval.py

License: Apache-2.0
Copyright 2026 Michael Kupermann
"""

import json
import random
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from recall_bench import make_queries  # noqa: E402

from souprise import RAGConfig, SoupriseRAG  # noqa: E402
from souprise.core.hdc import SimpleHDCRetriever  # noqa: E402
from souprise.data.generators.business import generate_business_data  # noqa: E402

_ENTITY_RE = re.compile(r"\b[A-Za-z]+_[A-Za-z0-9]+\b")


def perturb_typos(query: str, rng: random.Random) -> str:
    """Swap two adjacent characters in non-entity words longer than 3."""
    words = query.split()
    out = []
    for word in words:
        if len(word) > 3 and not _ENTITY_RE.fullmatch(word) and rng.random() < 0.6:
            i = rng.randint(1, len(word) - 2)
            word = word[:i] + word[i + 1] + word[i] + word[i + 2:]
        out.append(word)
    return " ".join(out)


def perturb_shuffle(query: str, rng: random.Random) -> str:
    words = query.split()
    rng.shuffle(words)
    return " ".join(words)


def recall_at_5(retriever, queries):
    hits = 0
    for item in queries:
        got = [r.title for r in retriever.search(item["query"], k=5)]
        hits += item["target"] in got
    return hits / len(queries)


def main():
    entries = generate_business_data(n=5000, seed=42)
    index_entries = [e.to_retrieval_format() for e in entries]
    base_queries = make_queries(entries, 200, 7)

    rng = random.Random(13)
    typo_queries = [{"query": perturb_typos(q["query"], rng),
                     "target": q["target"]} for q in base_queries]
    rng = random.Random(13)
    shuffle_queries = [{"query": perturb_shuffle(q["query"], rng),
                        "target": q["target"]} for q in base_queries]

    results = {}
    for encoding in ("v1", "v2"):
        retriever = SimpleHDCRetriever(encoding=encoding,
                                       sketch_threshold=10**12)
        retriever.index(index_entries)
        results[encoding] = {
            "paraphrase": round(recall_at_5(retriever, base_queries), 4),
            "typos": round(recall_at_5(retriever, typo_queries), 4),
            "shuffle": round(recall_at_5(retriever, shuffle_queries), 4),
        }
        print(encoding, results[encoding])

    # v2 coverage on the frozen foreign-question set
    corpus = generate_business_data(n=2000, seed=123)
    rag = SoupriseRAG(RAGConfig(retriever="simple", answer_mode="verified",
                                retrieval_k=5))
    rag.retriever = SimpleHDCRetriever(encoding="v2", sketch_threshold=10**12)
    rag.index_from_entries([e.to_retrieval_format() for e in corpus])
    corpus_text = " ".join(e.to_retrieval_format()["text"].lower() for e in corpus)
    items = [json.loads(line)
             for line in open("benchmarks/data/foreign_questions.jsonl")]
    answerable = [i for i in items if i["answerable"]]
    covered = 0
    for item in answerable:
        result = rag.query(item["question"])
        ok = result.answer_path in {"computed", "field", "ambiguous", "record_intent"}
        if not ok and result.answer_path == "refusal":
            named = [t.lower() for t in _ENTITY_RE.findall(item["question"])]
            if named and not any(n in corpus_text for n in named):
                ok = True
        covered += ok
    coverage_v2 = covered / len(answerable)
    print(f"v2 foreign-question coverage: {coverage_v2:.3f}")

    v1, v2 = results["v1"], results["v2"]
    robustness_win = (v2["typos"] > v1["typos"]
                      and v2["shuffle"] >= v1["shuffle"])
    holds = v2["paraphrase"] >= v1["paraphrase"] and coverage_v2 == 1.0
    verdict = "adopt_v2" if (robustness_win and holds) else "keep_v1"

    out = {
        "v1": v1, "v2": v2, "v2_coverage": round(coverage_v2, 4),
        "bars": "v2 must hold paraphrase recall and 1.000 coverage AND beat "
                "v1 on typos while not losing on shuffle",
        "verdict": verdict,
    }
    Path("benchmarks/results").mkdir(parents=True, exist_ok=True)
    Path("benchmarks/results/encoding_eval.json").write_text(
        json.dumps(out, indent=2))
    print("verdict:", verdict)


if __name__ == "__main__":
    main()
