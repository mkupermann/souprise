# Verified Answer Mode Evaluation (BENCH-5)

Run per the pre-registered protocol in `benchmarks/PROTOCOL.md` (BENCH-5,
bars committed before implementation). Reproduce with:

```bash
PYTHONPATH=. python3 benchmarks/verified_eval.py
```

## The idea

Fabrications cannot be reliably suppressed inside a generative model, so
the factual path removes the model entirely. In verified mode a rule-based
detector maps the question to a record field, the value is **copied
verbatim** from the retrieved record, entity mentions in the question must
match the answering record, conflicting records yield an explicit list of
all candidates, and weak or empty retrieval yields a refusal. The
generative mode is opt-in and hard-gated: an answer containing any figure
not present in the retrieved records is replaced by the verified fallback
before it reaches the user.

## Results: all four locked bars pass

| Check | Bar | Measured |
|---|---|---|
| V1 value accuracy, deduplicated corpus (60 lookups) | = 1.000 | **1.000** |
| V2 wrong-value rate under entity ambiguity | = 0.000 | **0.000** |
| V3 refusal rate on 20 unknown entities | = 1.000 | **1.000** |
| V4 shipped ungrounded-figure rate, gated generative (real 0.5B model) | = 0.000 | **0.000** |

## One honest iteration note

The first run failed V3 at 0.000: similarity scores alone happily return
the closest OTHER entity for an unknown name. The fix is not a score
threshold but deterministic entity verification — if the question names an
entity, the answering record must actually carry it, otherwise the mode
refuses. With that in place all bars pass; no bar was changed.

## Scope, stated precisely

- "100 % correct" holds for what verified mode asserts: every value in a
  verified answer exists verbatim in a cited record, unknown entities are
  refused, and conflicts are listed rather than resolved by guessing.
- It does not mean the system understands every question: field detection
  is rule-based and may return the whole matching record instead of a
  specific value; that answer is verbatim data, still nothing invented.
- Records themselves are trusted as ground truth. If the source data is
  wrong, the answer repeats the source; keep one current record per
  entity (see the fine-tuning failure analysis).
- Verified mode covers point lookups. Aggregates remain database work,
  and the free-text generative mode remains available behind the gate.
