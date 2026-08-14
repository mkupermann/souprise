"""Retrieval benchmark for the built-in HDC retriever.

Measures index build time and query latency on YOUR hardware — Souprise
deliberately publishes no benchmark numbers of its own. Runs with the core
install only (no model download, no optional dependencies).

Usage:
    python benchmarks/retrieval_bench.py --n 10000
    python benchmarks/retrieval_bench.py --n 100000 --queries 50 --k 5

License: Apache-2.0
Copyright 2026 Michael Kupermann
"""

import argparse
import statistics
import time

from souprise.core.hdc import SimpleHDCRetriever
from souprise.data.generators.business import generate_business_data


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark the built-in HDC retriever.")
    parser.add_argument("--n", type=int, default=10_000, help="Corpus size (entries)")
    parser.add_argument("--queries", type=int, default=25, help="Number of measured queries")
    parser.add_argument("--k", type=int, default=5, help="Top-k results per query")
    parser.add_argument("--seed", type=int, default=42, help="Data generation seed")
    args = parser.parse_args()

    print(f"Generating {args.n:,} synthetic business entries (seed={args.seed})...")
    entries = [e.to_retrieval_format() for e in generate_business_data(n=args.n, seed=args.seed)]

    retriever = SimpleHDCRetriever()

    start = time.perf_counter()
    retriever.index(entries)
    index_seconds = time.perf_counter() - start

    # Query with entry titles spread across the corpus; skip one warm-up query.
    step = max(1, len(entries) // args.queries)
    queries = [entries[i]["id"] for i in range(0, len(entries), step)][: args.queries]
    retriever.search(queries[0], k=args.k)

    latencies = []
    hits = 0
    for query in queries:
        start = time.perf_counter()
        results = retriever.search(query, k=args.k)
        latencies.append((time.perf_counter() - start) * 1000)
        if any(r.title == query for r in results):
            hits += 1

    print()
    print(f"Corpus size        : {retriever.size:,} entries")
    print(f"Index size         : {retriever.index_bytes / 1_000_000:.2f} MB "
          f"({retriever.index_bytes // retriever.size} bytes/entry)")
    print(f"Index build        : {index_seconds:.2f} s "
          f"({retriever.size / index_seconds:,.0f} entries/s)")
    print(f"Query latency (k={args.k}), {len(queries)} queries:")
    print(f"  mean             : {statistics.mean(latencies):8.2f} ms")
    print(f"  median           : {statistics.median(latencies):8.2f} ms")
    print(f"  p95              : {sorted(latencies)[int(0.95 * (len(latencies) - 1))]:8.2f} ms")
    print(f"Self-retrieval@{args.k}  : {hits}/{len(queries)}")


if __name__ == "__main__":
    main()
