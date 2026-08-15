# BENCH-9: Index-side RBAC and audit trail

Pre-registered bars in `benchmarks/PROTOCOL.md` (committed before
implementation). Eval: `benchmarks/rbac_audit_eval.py`. Corpus: 2,000
synthetic records, seed 123. Policy under test: `eu_only` (visible only
where Region=EU, field Margin hidden). 131 of 2,000 records visible.

## Results

| Check | Bar | Result | Verdict |
|---|---|---|---|
| R1 leak count (200 mixed queries) | = 0 | 0 | pass |
| R2 value accuracy, visible targets | = 1.000 | 1.000 (8/8) | pass |
| R2 denial rate, forbidden targets | = 1.000 | 1.000 (40/40) | pass |
| R2 forbidden-value leaks | = 0 | 0 | pass |
| A1 events = queries | true | 249 = 249 | pass |
| A1 answer hash matches shipped answer | true | true | pass |
| A1 UPDATE/DELETE rejected | true | true | pass |

## How enforcement works

The policy's visibility mask is applied to the hypervector index BEFORE
distance computation (`search(..., subset=...)` in `hdc.py`). Similarity
scores over forbidden records never exist, so they cannot leak through
ranking, sources, or answers. Hidden fields are stripped from record
texts before the verified path, aggregation, and generative context see
them. Aggregates are computed over the visible subset only.

## Measurement-instrument amendments (logged per protocol discipline)

The bars were NOT changed. Two defects in the eval instrument were found
and fixed during the run; both are precision fixes, the property under
test is unchanged:

1. **Title collisions from duplicate entities.** The seed-123 corpus
   contains records with identical titles in different regions (a known
   BENCH-3 finding, e.g. two "Customer Profile Customer_0168" records,
   one EU, one US). The first instrument flagged a leak whenever a
   forbidden record's TITLE appeared in output, which also fired on the
   legitimately visible same-titled EU twin. Amended to compare record
   CONTENT (body text) and to restrict forbidden-value checks to figures
   that appear in no visible record. First run under the amended
   instrument: 101 false positives became 0 with zero true leaks.

2. **Refusal expectation at entity level.** R2 originally required a
   refusal for every question whose target record is forbidden. Where
   the same entity also has visible sibling records, an answer drawn
   from permitted data is correct behavior, not a leak. Amended: refusal
   is required only for entities with no visible record at all; a
   separate check asserts forbidden values never appear in any case.

## One product fix that fell out

Questions about entities absent from the visible set ("How many units of
Product_DW are in stock?") were answered by the aggregation path with
"count over 0 records: 0" — misleading for an unknown entity and an
existence oracle under a policy. `compute_aggregate` now returns None
when an entity filter matches nothing, so the pipeline refuses instead,
identically with and without a policy. Re-running BENCH-5 after this
change surfaced a separate parser bug (issue #52, "lowest" matched the
Amount pattern via the substring "owe"); fixed with word boundaries.
BENCH-5 passes again (C1 = 1.000, C2 = 0.000).

## Known limits

- Policies are in-process enforcement objects. Authentication and
  principal management are the REST API's job (issue #29).
- Timing side channels are a documented non-goal (see PROTOCOL.md).
- R2's visible-lookup sample is small (n=8) because the eval restricts
  it to titles that are unambiguously visible under the strict eu_only
  policy; the unrestricted value-accuracy bar is covered by BENCH-4 on
  larger samples.
