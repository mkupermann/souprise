"""Recall benchmark: built-in HDC retriever vs a BM25 baseline.

Protocol and verdict bars are pre-registered in benchmarks/PROTOCOL.md and
were committed before this script first ran. The queries paraphrase records
with templates and synonyms; no query repeats a record title.

Usage:
    python benchmarks/recall_bench.py            # protocol settings
    python benchmarks/recall_bench.py --n 1000 --queries 50   # smoke run

License: Apache-2.0
Copyright 2026 Michael Kupermann
"""

import argparse
import json
import math
import random
import re
from collections import Counter, defaultdict
from pathlib import Path

from souprise.core.hdc import SimpleHDCRetriever, _tokenize
from souprise.data.generators.business import generate_business_data

_WORD_RE = re.compile(r"[a-z0-9]+")

STATUS_SYNONYMS = {
    "overdue": ["late", "past due", "unpaid", "not settled"],
    "paid": ["settled", "cleared", "already payed"],
    "open": ["outstanding", "still unsettled", "awaiting payment"],
    "cancelled": ["voided", "called off"],
    "shipped": ["sent out", "dispatched"],
    "delivered": ["arrived", "received by the customer"],
    "confirmed": ["accepted", "acknowledged"],
    "in production": ["being manufactured", "currently built"],
}


def _field(content: str, key: str) -> str:
    for line in content.splitlines():
        if line.startswith(key + ":"):
            return line.split(": ", 1)[1]
    return ""


def make_queries(entries, n_queries: int, seed: int):
    """Paraphrased queries with a known target record."""
    rng = random.Random(seed)
    queries = []
    pool = list(range(len(entries)))
    rng.shuffle(pool)
    for idx in pool:
        if len(queries) >= n_queries:
            break
        e = entries[idx]
        title, content = e.title, e.content
        parts = title.split()
        if title.startswith("Invoice "):
            customer, month, year = parts[1], parts[2], parts[3]
            status = _field(content, "Status")
            syn = rng.choice(STATUS_SYNONYMS.get(status, [status]))
            q = rng.choice([
                f"how much does {customer} owe from {month} {year}",
                f"{syn} bill of {customer} for {month} {year}",
                f"what did we charge {customer} in {month} {year}",
                f"billing document {customer} {month} {year} payment state",
            ])
        elif title.startswith("Order "):
            customer, product = parts[1], parts[2]
            q = rng.choice([
                f"purchase of {product} by {customer}",
                f"what did {customer} buy, the {product} one",
                f"{customer} bought {product}, delivery state",
            ])
        elif title.startswith("Customer Profile"):
            customer = parts[2]
            q = rng.choice([
                f"yearly sales and segment of {customer}",
                f"who is our contact at {customer} and any open tickets",
                f"account overview {customer}",
            ])
        elif title.startswith("Product "):
            product = parts[1]
            q = rng.choice([
                f"inventory level and pricing of {product}",
                f"is {product} selling up or down lately",
                f"warehouse units of {product}",
            ])
        elif title.startswith("KPI "):
            dept, metric, quarter, year = parts[1], parts[2], parts[3], parts[4]
            q = rng.choice([
                f"did {dept} hit its {metric.lower()} goal in {quarter} {year}",
                f"{dept} {metric.lower()} figure {quarter} {year}",
            ])
        elif title.startswith("Budget "):
            dept, year = parts[1], parts[2]
            q = rng.choice([
                f"how much of the {year} {dept} money is spent",
                f"remaining funds {dept} {year}",
            ])
        else:
            continue
        queries.append({"query": q, "target": title})
    return queries


class BM25:
    """Plain BM25 (k1=1.5, b=0.75) over unigram tokens, NumPy-free."""

    def __init__(self, docs, k1=1.5, b=0.75):
        self.k1, self.b = k1, b
        self.doc_tokens = [_WORD_RE.findall(d.lower()) for d in docs]
        self.doc_len = [len(t) for t in self.doc_tokens]
        self.avg_len = sum(self.doc_len) / max(1, len(self.doc_len))
        self.tf = [Counter(t) for t in self.doc_tokens]
        df = defaultdict(int)
        for tokens in self.doc_tokens:
            for term in set(tokens):
                df[term] += 1
        n = len(docs)
        self.idf = {t: math.log((n - f + 0.5) / (f + 0.5) + 1) for t, f in df.items()}

    def top(self, query: str, k: int = 5):
        terms = _WORD_RE.findall(query.lower())
        scores = []
        for i, tf in enumerate(self.tf):
            s = 0.0
            for t in terms:
                if t not in tf:
                    continue
                num = tf[t] * (self.k1 + 1)
                den = tf[t] + self.k1 * (1 - self.b + self.b * self.doc_len[i] / self.avg_len)
                s += self.idf.get(t, 0.0) * num / den
            scores.append(s)
        order = sorted(range(len(scores)), key=lambda i: -scores[i])
        return order[:k]


def evaluate(name, ranks):
    recall = sum(1 for r in ranks if r is not None and r < 5) / len(ranks)
    mrr = sum(1 / (r + 1) for r in ranks if r is not None and r < 5) / len(ranks)
    print(f"{name:14s} Recall@5 = {recall:.3f}   MRR@5 = {mrr:.3f}")
    return {"recall_at_5": round(recall, 4), "mrr_at_5": round(mrr, 4)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=5000)
    parser.add_argument("--queries", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--query-seed", type=int, default=7)
    parser.add_argument("--out", default="benchmarks/results")
    args = parser.parse_args()

    print(f"Corpus {args.n}, {args.queries} paraphrased queries")
    entries = generate_business_data(n=args.n, seed=args.seed)
    docs = [f"{e.title}\n{e.content}" for e in entries]
    titles = [e.title for e in entries]
    queries = make_queries(entries, args.queries, args.query_seed)

    hdc = SimpleHDCRetriever()
    hdc.index([e.to_retrieval_format() for e in entries])
    bm25 = BM25(docs)

    hdc_ranks, bm_ranks = [], []
    for item in queries:
        got = [r.title for r in hdc.search(item["query"], k=5)]
        hdc_ranks.append(got.index(item["target"]) if item["target"] in got else None)
        got = [titles[i] for i in bm25.top(item["query"], k=5)]
        bm_ranks.append(got.index(item["target"]) if item["target"] in got else None)

    hdc_m = evaluate("HDC (ours)", hdc_ranks)
    bm_m = evaluate("BM25", bm_ranks)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    results = {
        "corpus": args.n, "queries": len(queries), "seed": args.seed,
        "query_seed": args.query_seed, "hdc": hdc_m, "bm25": bm_m,
        "bar": "HDC competitive iff hdc.recall_at_5 >= bm25.recall_at_5 - 0.02",
        "verdict": ("competitive"
                    if hdc_m["recall_at_5"] >= bm_m["recall_at_5"] - 0.02
                    else "bm25_wins"),
    }
    (out / "recall.json").write_text(json.dumps(results, indent=2))
    print(f"verdict: {results['verdict']}  -> {out}/recall.json")


if __name__ == "__main__":
    main()
