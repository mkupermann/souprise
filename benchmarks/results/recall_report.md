# Recall Benchmark: Built-in HDC vs BM25

Run per the pre-registered protocol in `benchmarks/PROTOCOL.md` (bars were
committed before this run). Reproduce with:

```bash
PYTHONPATH=. python3 benchmarks/recall_bench.py
```

## Setup

- Corpus: 5,000 synthetic business records (seed 42)
- Queries: 200 paraphrased lookups (seed 7), generated from randomly chosen
  records with templates and status synonyms (overdue -> "late", "unpaid",
  "past due"...). No query repeats a record title.
- Baseline: BM25, k1=1.5, b=0.75, same unigram tokenization, implemented in
  `benchmarks/recall_bench.py`.

## Results

| System | Recall@5 | MRR@5 |
|---|---|---|
| Built-in HDC (SimpleHDCRetriever) | 1.000 | 1.000 |
| BM25 | 0.965 | 0.908 |

**Verdict against the locked bar** (HDC competitive iff its Recall@5 is
within 0.02 of BM25): **competitive**. HDC did not lose to BM25 on this
query class, and ranked the target first more consistently (MRR 1.000 vs
0.908).

## Honest limitations

- Both systems sit near the ceiling because realistic business lookups carry
  unique entity identifiers (Customer_0057, Product_AB) that alone narrow
  the search drastically. This benchmark shows HDC holds up on the lookup
  class Souprise targets; it does not measure identifier-free semantic
  search, where both bag-of-words approaches would degrade and an embedding
  model would be expected to win.
- Synonyms in queries are limited to status words; the corpus vocabulary is
  synthetic and narrower than real ERP text.
- One machine, one seed pair. The script takes minutes; rerun it on your
  data before drawing conclusions for your corpus.
