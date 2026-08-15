# Sub-Linear Search and Encoding v2 (BENCH-8)

Run per the pre-registered protocol in `benchmarks/PROTOCOL.md`. Both
candidate features failed their locked bars. Both results are published
here, and the one unplanned win ships instead.

## S: sketch prefilter — bars failed, not adopted

Two-stage search (1,024-bit byte-sampled sketch prefilter, exact
re-ranking of 25,000 candidates), measured at 1,000,000 records with 200
queries:

| Metric | Bar | Measured |
|---|---|---|
| Recall@5 vs exact scan | >= 0.99 | **0.218** |
| Median query speedup | >= 5x | **3.9x** |

At 100,000 records the same design reached 0.996 recall; at 1M it
collapses. The finding underneath: bundled HDC vectors produce tightly
packed distance distributions, so a sampled sketch cannot separate the
true top-5 from a million near-ties. Sampling-based prefilters are the
wrong tool for this vector family. The prefilter stays in the code as an
explicit opt-in (`sketch_threshold`), off by default. Next candidate for
the sub-linear goal is a graph-based index (HNSW-style) under the same
bars — tracked in #38.

## E: encoding v2 — v1 wins, v1 stays

v2 added character trigrams per token and a mild positional permutation.
Measured on the BENCH-1 paraphrase set and a deterministic robustness
set (seeded adjacent-character typos, word-order shuffles):

| Set | v1 | v2 |
|---|---|---|
| Paraphrase Recall@5 | **1.000** | 0.950 |
| Typo Recall@5 | **0.960** | 0.830 |
| Shuffle Recall@5 | **1.000** | 0.890 |

The extra trigram components dilute the majority bundle (bundling has a
capacity limit) and the positional permutation hurts word-order
robustness, exactly where it was supposed to be neutral. The "too
simple" v1 encoding is measurably the more robust one. v2 remains
available as `encoding="v2"` with its version stored in the index
metadata; v1 stays the default.

## The unplanned win: exact search got 2.7x faster

Rewiring distances onto padded uint64 views (hardware popcount over
8-byte words) sped up the exact scan with zero accuracy trade-off:

| Corpus | Exact median before | Exact median now |
|---|---|---|
| 1,000,000 records | 371 ms | **135.8 ms** |

This ships as the default. It also means the earlier scale recording's
371 ms figure is now conservative; the recording stays as is and this
report carries the current number.

## Reproduce

```bash
PYTHONPATH=. python3 benchmarks/ann_eval.py --n 1000000
PYTHONPATH=. python3 benchmarks/encoding_eval.py
```
