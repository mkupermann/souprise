# Fine-Tuning Failure Analysis (BENCH-3)

Run per the pre-registered protocol in `benchmarks/PROTOCOL.md` (BENCH-3,
bars committed before the runs). Reproduce with:

```bash
PYTHONPATH=. python3 benchmarks/finetune_analysis.py --adapter ./souprise_model
```

## Pre-registered results

**A. Error decomposition.** The target record was inside the provided
context for 60 of 60 questions. Every miss (16 untuned, 17 tuned, 15
shared) is an extraction failure, zero are retrieval misses. Retrieval is
not the bottleneck.

**S1, harder contexts** (5,000 records, k=5): untuned 0.533, tuned 0.533.
Delta 0.000, bar +0.05 — **no benefit**.

**S2, format fidelity** (expected value within the first 20 tokens):
untuned 0.733, tuned 0.700 — **no benefit**.

**S3, memorization negative control** (no records in the prompt): untuned
0.017, tuned **0.117** — over the +0.10 flag threshold. **The adapter
memorized training facts.** For daily-changing business data this is a
defect: a tuned model can emit stale memorized figures precisely when
retrieval returns nothing.

Per the locked applicability rule, fine-tuning moves out of the core pitch
and becomes a measure-first option.

## What the misses actually were (exploratory, not pre-registered)

Reading the per-question dumps showed the failed questions cluster on
entities with **multiple conflicting records** in the corpus: the synthetic
data contains e.g. three "Customer Profile Customer_0058" records with
three different annual revenues, and the model quotes one of the *other*
real values. These are ambiguity errors, not fabrications.

Two follow-up measurements, clearly marked exploratory:

| Intervention | Accuracy (base setting) |
|---|---|
| Untuned 0.5B on ambiguous corpus (baseline) | 0.733 |
| LoRA-tuned 0.5B | 0.717 |
| Untuned **1.5B** (twice the size) | 0.717 |
| Untuned 0.5B on **per-entity deduplicated** corpus | **1.000** |

Neither fine-tuning nor a bigger model fixed the errors. Deduplicating to
one current record per entity fixed all of them. That is exactly what
`souprise index add`'s upsert semantics maintain automatically when daily
updates carry stable entity ids.

## Applicable guidance

1. **Keep one current record per entity.** Use stable ids and `souprise
   index add` (upsert) for daily updates; don't append snapshots of the
   same entity. This is worth more than any tuning at this task.
2. **Don't fine-tune on your record values.** It bought nothing on any
   pre-registered scenario and it memorizes data that will go stale (S3).
3. If you fine-tune at all, do it for form (domain vocabulary, output
   format), measure with `benchmarks/finetune_eval.py` against your own
   data, and re-check the S3 memorization control.
