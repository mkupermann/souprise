# Foreign Question Set (BENCH-7)

70 questions written by Mistral (via the vibe CLI), which has never seen
Souprise's parser code. Committed before the first coverage measurement;
the set is frozen — the parser may improve, the questions may not.

Scope rules (documented before measuring): a question is "answerable"
if it is a field lookup for a named entity or an aggregate
(sum/count/average/min/max, optionally with a numeric threshold) filtered
by status, region, trend or entity. Out of scope, honestly: time-window
questions (no time axis in the data model), sorted or exhaustive
listings, per-group results and group-argmax ("which region has the
most..."), and corpus-percentage questions. Seven borderline items were
reclassified by hand; each carries its reason in the JSONL.
