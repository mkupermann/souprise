# Corporate Style Tuning Evaluation (BENCH-4)

Run per the pre-registered protocol in `benchmarks/PROTOCOL.md` (BENCH-4,
bars committed before implementation). Feature under test:
`souprise train style` — company language and answer structure from a
glossary and template, record values randomized per run.

## Result: all four locked bars pass

| Metric | Untuned | Tuned | Bar | Verdict |
|---|---|---|---|---|
| Term compliance (company glossary terms used) | 0.000 | **0.983** | delta >= +0.20 | pass |
| Format compliance (template structure present) | 0.000 | **0.983** | delta >= +0.30 | pass |
| Factual accuracy (guard) | 0.750 | 0.700 | tuned >= untuned - 0.05 | holds, exactly at the limit |
| Memorization rate (no-context value reproduction) | 0.000 | 0.000 | tuned <= untuned + 0.02 | clean |

The prompt contains no glossary or template hints; the tuning carries the
voice by itself. 59 of 60 answers open with the trained structure
("Kurzüberblick: ... Quelle im Datenbestand: ...") and use the company
terminology (Faktura, Geschäftspartner, Rechnungsbetrag).

## What it took: two honest iterations

**Iteration 1 failed (0.000 compliance).** Two causes, both found by
inspecting real outputs, both fixed:

1. Instrument bug: generators and evals sent raw prompts while Soup
   trains behind the tokenizer's chat template, so the adapter never
   fired at inference. Fixed in `wrap_chat()` (pipeline + all evals).
2. Distribution gap: training used one record per prompt and
   company-term questions only; runtime has 1-5 records and users ask in
   generic language. The generator now mixes question registers, adds
   distractor records with shuffled target position, and trains on both
   raw and glossary-translated context.

**Iteration 2 passed** with the same locked bars — no bar was changed at
any point.

## Honest caveats

- The accuracy guard holds exactly at its limit (-0.05). Style tuning is
  not free; on 60 questions the tuned model missed 3 more lookups than
  the untuned one. Watch this on your own data.
- Term and format compliance are mechanical checks (glossary hits,
  template markers), not human judgments of tone. Blind A/B preference
  testing is yours to run — the DPO can evaluate locally.
- One template, one glossary, German example set, 0.5B model. Different
  templates or languages need their own run of `benchmarks/style_eval.py`.

Reproduce:

```bash
souprise train style --glossary examples/style/glossary_de.csv \
    --answer-template examples/style/answer_template_de.txt
souprise train create-config --data-path style_training.jsonl
soup train --config soup_config.yaml --yes
PYTHONPATH=. python3 benchmarks/style_eval.py --adapter ./style_model \
    --glossary examples/style/glossary_de.csv \
    --answer-template examples/style/answer_template_de.txt \
    --training-data style_training.jsonl
```
