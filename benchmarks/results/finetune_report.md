# Fine-Tuning Evaluation: Tuned vs Untuned

Run per the pre-registered protocol in `benchmarks/PROTOCOL.md` (bars
committed before the run). The training path was executed for real:
Soup/LoRA (soup-cli 0.73.1, optimizer adamw_torch) on
mlx-community/Qwen2.5-0.5B-Instruct-4bit, 3,883 Alpaca examples (seed 42),
3 epochs, 2,600 iterations, final train loss ~0.30.

## Setup

- Eval: 60 point-lookup questions (seed 11) over a 2,000-record corpus
  (seed 123, disjoint from any tuning assumption).
- Retrieval context computed once with the built-in retriever and reused
  identically for both models. Temperature 0.
- Correct iff the expected value (normalized number or status word) appears
  in the answer.

## Results

| Model | Accuracy |
|---|---|
| Untuned Qwen2.5-0.5B-Instruct-4bit | 0.733 |
| Tuned (LoRA on synthetic business Q&A) | 0.717 |

Delta: **-0.017**. Verdict against the locked bar (adds value iff
delta >= +0.05): **no measurable benefit** at this scale.

## What this means, honestly

- With retrieval doing the factual work, the untuned instruct model already
  extracts values from the provided records about as well as the tuned one.
  At 0.5B scale and with templated synthetic training data, LoRA tuning did
  not help and the small negative delta is within noise for n=60.
- Fine-tuning in Souprise is therefore an **optional** step, not a
  requirement. It may still pay off with real domain vocabulary, larger
  models, or task formats the base model handles poorly; that would need
  its own measurement.
- The training path itself works end to end and is reproducible:
  `souprise train generate` -> `souprise train create-config` ->
  `soup train --config soup_config.yaml --yes`.

Reproduce:

```bash
PYTHONPATH=. python3 benchmarks/finetune_eval.py \
    --tuned ./souprise_model --adapter ./souprise_model
```
