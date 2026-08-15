# Deterministic Computation and Verbalizer Evaluation (BENCH-6)

Run per the pre-registered protocol in `benchmarks/PROTOCOL.md` (BENCH-6,
bars committed before implementation). Reproduce with:

```bash
PYTHONPATH=. python3 benchmarks/compute_eval.py
```

## The division of labor

The language model may write prose. Code owns every number. Two features
implement that rule:

- **Deterministic aggregation** (`souprise/core/compute.py`): sum, count,
  average, min and max with status/region/entity filters are parsed
  rule-based and computed with Decimal arithmetic over ALL index entries,
  not a top-k sample. Aggregate questions that used to get a "belongs in
  a database" hint now get an exact answer.
- **Styled mode** (`--mode styled`): the deterministic core (verified
  lookup or computed aggregate) produces the facts; the LLM only phrases
  the sentence. Every figure in the generated text must exactly match the
  deterministic values or the question, otherwise the deterministic text
  ships instead.

## Results: both locked bars pass

| Check | Bar | Measured |
|---|---|---|
| C1 aggregate exactness vs independent ground truth (40 cases: sum/avg/max/min of Amount, Annual Revenue, Stock, with and without status filters) | = 1.000 | **1.000** (40/40) |
| C2 shipped figure-mismatch rate in styled mode, real 0.5B model, 80 questions (60 lookups + 20 aggregates) | = 0.000 | **0.000** |

Fallback rate in C2: 0.000 — at temperature 0 the model phrased every
answer without altering a figure, so the gate never had to fire. It stays
in place regardless; the guarantee comes from the gate, not from trusting
the model.

## Scope, stated plainly

- The aggregate parser is rule-based and covers the shipped field/filter
  vocabulary (amounts, revenue, stock; status, region, entity). Questions
  outside it fall back to verified lookup or an honest hint, never to a
  guessed number.
- Aggregates trust the index. Records missing a field are skipped and the
  record count is always reported, so "sum over 37 records" is checkable.
- Styled mode inherits every verified-mode guarantee; it only changes the
  wording of an already-correct answer.
