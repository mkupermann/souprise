"""BENCH-8 S: sketch prefilter vs exact scan at 1M records.

Pre-registered bars in benchmarks/PROTOCOL.md: recall@5 >= 0.99 vs the
exact scan, median query speedup >= 5x, measured at 1,000,000 records
with 200 lookup queries.

Usage:
    PYTHONPATH=. python3 benchmarks/ann_eval.py [--n 1000000]

License: Apache-2.0
Copyright 2026 Michael Kupermann
"""

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from souprise.core.hdc import SimpleHDCRetriever  # noqa: E402
from souprise.data.generators.business import generate_business_data  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=1_000_000)
    parser.add_argument("--queries", type=int, default=200)
    parser.add_argument("--out", default="benchmarks/results")
    args = parser.parse_args()

    print(f"building corpus of {args.n:,} records...")
    entries = [e.to_retrieval_format()
               for e in generate_business_data(n=args.n, seed=42)]
    retriever = SimpleHDCRetriever(sketch_threshold=10**12)  # exact for now
    t0 = time.perf_counter()
    retriever.index(entries)
    print(f"indexed in {time.perf_counter() - t0:.0f}s")

    step = max(1, len(entries) // args.queries)
    queries = [entries[i]["id"] for i in range(0, len(entries), step)][:args.queries]

    # Exact baseline
    retriever.search(queries[0], k=5)  # warm-up
    exact_tops, exact_lat = [], []
    for query in queries:
        t = time.perf_counter()
        results = retriever.search(query, k=5)
        exact_lat.append((time.perf_counter() - t) * 1000)
        exact_tops.append([r.title for r in results])

    # Prefiltered
    retriever.sketch_threshold = 0
    retriever._ensure_sketches()  # build outside timing, report separately
    t0 = time.perf_counter()
    retriever._sketches = None
    retriever._ensure_sketches()
    sketch_build_s = time.perf_counter() - t0

    retriever.search(queries[0], k=5)  # warm-up
    pre_tops, pre_lat = [], []
    for query in queries:
        t = time.perf_counter()
        results = retriever.search(query, k=5)
        pre_lat.append((time.perf_counter() - t) * 1000)
        pre_tops.append([r.title for r in results])

    overlap = [len(set(a) & set(b)) / 5 for a, b in zip(exact_tops, pre_tops)]
    recall = sum(overlap) / len(overlap)
    exact_med = statistics.median(exact_lat)
    pre_med = statistics.median(pre_lat)
    speedup = exact_med / pre_med

    print(f"recall@5 vs exact : {recall:.4f}")
    print(f"exact median      : {exact_med:.1f} ms")
    print(f"prefilter median  : {pre_med:.1f} ms  (speedup {speedup:.1f}x)")
    print(f"sketch build      : {sketch_build_s:.1f} s "
          f"({retriever._sketches.nbytes / 1e6:.0f} MB)")

    results = {
        "corpus": args.n, "queries": len(queries),
        "recall_at_5_vs_exact": round(recall, 4),
        "exact_median_ms": round(exact_med, 2),
        "prefilter_median_ms": round(pre_med, 2),
        "speedup": round(speedup, 2),
        "sketch_build_seconds": round(sketch_build_s, 2),
        "sketch_mb": round(retriever._sketches.nbytes / 1e6, 1),
        "bars": {"recall": ">= 0.99", "speedup": ">= 5x"},
        "verdicts": {
            "recall": "pass" if recall >= 0.99 else "fail",
            "speedup": "pass" if speedup >= 5 else "fail",
        },
    }
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "ann_eval.json").write_text(json.dumps(results, indent=2))
    print(json.dumps(results["verdicts"], indent=2))


if __name__ == "__main__":
    main()
