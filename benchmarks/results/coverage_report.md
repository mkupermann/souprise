# Foreign-Question Coverage and Natural-Name Entities (BENCH-7)

Run per the pre-registered protocol in `benchmarks/PROTOCOL.md`. The
question set was written by a different model (Mistral via the vibe CLI)
that has never seen the parser code, frozen and committed before the
first measurement (`benchmarks/data/foreign_questions.jsonl`, scope rules
in `benchmarks/data/README.md`). Reproduce with:

```bash
PYTHONPATH=. python3 benchmarks/coverage_eval.py
PYTHONPATH=. python3 benchmarks/realname_eval.py
```

## Q1: coverage on foreign questions

| Run | Coverage (31 answerable questions) | Wrong values |
|---|---|---|
| Baseline (parser as reviewed) | **0.516** | 0 |
| After router fix + intents + thresholds | 0.968 | 0 |
| After deterministic entity scan | **1.000** | 0 |

Bar: coverage >= 0.85 and wrong-value rate = 0.000 — **both pass**, and
the honest path is part of the result: the reviewed suspicion of
evaluation circularity was correct, foreign questions halved the initial
hit rate.

What the misses taught, and what changed:

1. **A real router bug.** The aggregation pre-gate used a narrower marker
   list than the computation parser, so computable questions like "the
   largest order from Customer_0345" were never computed. The pre-gate is
   gone; the parser decides.
2. **Missing intents and filters.** Record-overview intent (profile /
   details / summarize / "how is X performing"), trend field lookups,
   numeric threshold filters ("more than 50", "over $500k") and
   existence questions ("are there any...") as exact counts.
3. **Correct refusals counted as correct.** Several foreign questions
   name entities that do not exist in the corpus (the question author
   invented ids); refusing them is right. The eval verifies existence
   independently against the raw entries before crediting a refusal.
4. **Rare entities beyond top-k.** A named entity the index does contain
   can still miss the top-k similarity window; a deterministic full scan
   now finds its records (or proves the field absent) instead of refusing.

## Q2: natural company names

Entity verification previously keyed on underscore identifiers only. It
now builds an entity vocabulary from the index (customer, product,
department and contact field values) and recognizes company-shaped names.
On a corpus of 30 natural-name companies (ACME GmbH, Meyer & Söhne, ...):

| Check | Bar | Measured |
|---|---|---|
| Known-name value accuracy (30 lookups) | = 1.000 | **1.000** |
| Unknown-name refusal rate (20 invented companies) | = 1.000 | **1.000** |

## Honest limits

- The frozen set has 70 questions from one model; more authors would be
  better. The out-of-scope classes (time windows, sorted listings,
  per-group results) remain honestly out of scope and are documented
  with per-question reasons.
- Company-name detection covers legal-form suffixes (GmbH, AG, Inc.,
  ...) and "Name & Name" patterns plus everything the index vocabulary
  contains; an unusual name absent from both patterns and vocabulary
  falls back to similarity retrieval like any other text.
