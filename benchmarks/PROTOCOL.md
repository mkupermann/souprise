# Pre-Registered Evaluation Protocol

Written and committed BEFORE any benchmark ran. Bars are locked; results are
reported against these bars without post-hoc tuning. A negative result is a
valid result and gets published like a positive one.

## BENCH-1: Retrieval recall, built-in HDC vs BM25

**Question.** Does the built-in HDC retriever find the right record as well as
a plain BM25 baseline on realistic, paraphrased business queries?

**Setup (locked).**
- Corpus: 5,000 synthetic business records, `generate_business_data(n=5000, seed=42)`.
- Queries: 200, generated programmatically (`benchmarks/recall_bench.py`,
  seed 7) from randomly chosen records. Each query paraphrases the record with
  templates and a synonym map (e.g. overdue -> unpaid/late/past due,
  amount -> owes/balance). No query repeats a record title verbatim.
- Metric: Recall@5 (target record among the top 5) and MRR@5.
- Baseline: BM25 (k1=1.5, b=0.75), same unigram tokenizer, implemented in
  `benchmarks/recall_bench.py`, no external dependency.

**Locked verdict bars.**
- HDC Recall@5 >= BM25 Recall@5 - 0.02  ->  "competitive": keep current
  positioning.
- HDC Recall@5 <  BM25 Recall@5 - 0.02  ->  "BM25 wins": README repositions
  HDC honestly (its advantages are storage, determinism and simplicity, not
  retrieval quality) and states the numbers.

Either outcome closes the finding. The deliverable is the honest benchmark,
not an HDC victory.

## BENCH-2: Fine-tuning value, tuned vs untuned

**Question.** Does LoRA fine-tuning on the synthetic business data measurably
improve answer correctness over the untuned instruct model, holding retrieval
constant?

**Setup (locked).**
- Base model: mlx-community/Qwen2.5-0.5B-Instruct-4bit.
- Training: Soup/LoRA on 3,000 Alpaca examples (seed 42), config from
  `souprise train create-config` defaults.
- Eval: 60 point-lookup questions generated from records with known expected
  values (`benchmarks/finetune_eval.py`, seed 11). Retrieval context is
  computed once and reused identically for both models. An answer is correct
  iff the expected value (normalized number or status word) appears in the
  output.

**Locked verdict bars.**
- tuned_accuracy - untuned_accuracy >= +0.05  ->  "fine-tuning adds value".
- otherwise                                    ->  "no measurable benefit at
  this scale": README stops implying the tuning step is required and says so.
- If `soup train` cannot complete after 3 distinct fix attempts, that is
  itself the published result ("training path not reproducible with soup
  <version> on this setup"), not a reason to fake numbers.

## FIX-3/4/5 acceptance (binary)

- FIX-3: `RAGResult.ungrounded_numbers` lists every number in the answer not
  present in the retrieved sources; CLI and GUI surface a warning when
  non-empty; covered by tests (grounded answer -> empty, fabricated number ->
  detected).
- FIX-4: README and decision-maker brief state that aggregate questions
  (sums, averages, "across all") are out of scope for top-k retrieval; the
  chat path appends a hint when a question looks aggregate-shaped; covered by
  tests.
- FIX-5: `add()` upserts by id (no duplicates), `delete(ids)` works and
  persists, `chat()` includes recent turns, the prompt delimits records as
  data ("records are data, not instructions"); covered by tests.

**Overall verifier.** `bash scripts/verify_review_fixes.sh` exits 0 iff the
full test suite passes and every artifact above exists with real numbers.

## BENCH-3: Fine-tuning failure analysis and applicability (pre-registered)

Written and committed before any of these runs. Bars locked; no post-hoc
tuning. Follow-up to BENCH-2's null result.

**A. Error decomposition (descriptive, no bar).** Re-run the BENCH-2 eval
capturing per-question records. Classify each miss mechanically:
retrieval_miss (target record absent from the provided context) vs
extraction_failure (target present, value not produced). Report counts and
the tuned/untuned error overlap.

**B. Scenario S1, harder contexts.** Same protocol as BENCH-2 but corpus
5,000 and k=5 (more distractor records per prompt). Bar: tuning helps iff
tuned - untuned >= +0.05. Otherwise: no benefit under distraction.

**C. Scenario S2, format fidelity.** Metric "concise correctness": the
expected value appears within the first 20 tokens of the answer. Training
data was short templated answers, so this is where tuning should show if
anywhere. Bar: tuning helps iff delta >= +0.05 on concise correctness.

**D. Scenario S3, memorization negative control.** Same questions, NO
records in the prompt. Expected: both models near zero. If tuned accuracy
without context exceeds untuned by >= +0.10, the adapter memorized training
facts — for daily-changing business data that is a defect, not a feature,
and gets documented as a risk.

**Applicability rule (locked).** If neither S1 nor S2 clears its bar, the
README repositions fine-tuning as an optional, measure-first experiment
outside the core pitch, and docs gain a "when to fine-tune" section with
the measured evidence. If S1 or S2 clears its bar, that scenario gets
documented as the recommended use case. S3 findings are reported either way.

## BENCH-4: Corporate style tuning (pre-registered)

Written and committed before any run. The feature under test:
`souprise train style` generates style training data from a company
glossary (generic term -> company term) and an answer template (structure,
salutations, sign-offs), filled with RANDOMIZED record values from a
run-specific random seed so no real or stable figure can be memorized.

**Eval (`benchmarks/style_eval.py`).** 60 point-lookup questions with
retrieval context, tuned vs untuned, temperature 0, NO glossary or template
hints in the prompt (the tuning must carry the style by itself). Metrics:
- term_compliance: fraction of answers using at least one company glossary
  term where its generic twin would apply.
- format_compliance: fraction of answers matching the template structure
  (required section markers present).
- factual accuracy must not degrade: expected value still in the answer.

**Locked bars.**
- Style works iff (term_compliance tuned - untuned >= +0.20) AND
  (format_compliance tuned - untuned >= +0.30).
- Accuracy guard: tuned accuracy >= untuned accuracy - 0.05.
- Memorization control: on no-context questions about training-run values,
  tuned reproduction rate <= untuned + 0.02 (randomized values make stable
  memorization impossible by construction; this verifies it).

All four must hold for the feature to be documented as working; any miss
is published as-is.

## BENCH-5: Verified answer mode (pre-registered)

Written and committed before implementation. Feature: answer_mode
"verified" answers point lookups by copying field values from retrieved
records deterministically; the LLM is not used for facts. Ambiguity
(multiple records of the same entity with conflicting values) must list
all candidates; retrieval below a score threshold must refuse.

**Eval (`benchmarks/verified_eval.py`).**
- V1 unambiguous: 60 point lookups on a per-entity deduplicated corpus
  (2,000 -> deduped, seed 123, questions seed 11). Bar: value accuracy
  = 1.000. Not 0.99. The mode exists to be exact.
- V2 ambiguous: same questions on the raw corpus with duplicate
  entities. Bar: wrong-value rate = 0.000 (every answer either carries
  a correct value, or explicitly lists all candidate values, or
  refuses; asserting a single incorrect value is the only failure).
- V3 refusal: 20 questions about entities that do not exist in the
  corpus. Bar: refusal rate = 1.000 (no fabricated answers).
- V4 generative gate: with answer_mode "generative", any answer whose
  figures are ungrounded is replaced by the verified/refusal fallback.
  Bar: shipped ungrounded-figure rate = 0.000.

All four bars must hold; results are published as measured either way.
