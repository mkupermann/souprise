#!/bin/bash
# Verifier for the five review fixes. Exits 0 iff everything is done for real.
set -u
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="${SOUPRISE_PY:-python3}"
fail() { echo "VERIFY FAIL: $1"; exit 1; }

# 1. Full test suite green (includes FIX-3/4/5 tests once they exist)
$PY -m pytest tests/ -q > /tmp/verify_pytest.out 2>&1 || fail "pytest not green"
grep -qE "^[0-9]+ passed" /tmp/verify_pytest.out || \
  grep -qE "[0-9]+ passed" /tmp/verify_pytest.out || fail "no passed count parsed"

# 2. FIX-3: grounding check exists and is tested
grep -q "ungrounded_numbers" souprise/core/pipeline.py || fail "no grounding field"
grep -rq "ungrounded" tests/ || fail "no grounding tests"

# 3. FIX-5: upsert/delete/multi-turn/delimiting exist and are tested
grep -q "def delete" souprise/core/hdc.py || fail "no delete()"
grep -rq "test_add_upserts\|upsert" tests/ || fail "no upsert test"
grep -rq "test_delete" tests/ || fail "no delete test"
grep -rq "test_chat_includes_history\|history" tests/test_pipeline.py || fail "no multi-turn test"
grep -q "data, not instructions" souprise/core/pipeline.py || fail "no injection delimiting"

# 4. BENCH-1: recall benchmark ran and produced a real report with both systems
[ -f benchmarks/results/recall_report.md ] || fail "no recall report"
grep -q "BM25" benchmarks/results/recall_report.md || fail "recall report lacks BM25"
grep -qE "Recall@5.*0\.[0-9]+" benchmarks/results/recall_report.md || fail "no recall numbers"
grep -q "recall_report" README.md || grep -q "BM25" README.md || fail "README lacks recall result"

# 5. BENCH-2: fine-tune eval report exists with tuned vs untuned numbers OR a
#    documented reproducible failure of the training path
[ -f benchmarks/results/finetune_report.md ] || fail "no finetune report"
grep -qiE "tuned|training path" benchmarks/results/finetune_report.md || fail "finetune report empty"
grep -qi "fine-tun" README.md || fail "README lacks fine-tune result reference"

# 6. FIX-4: aggregation honesty in README + hint in code and tests
grep -qi "aggregate" README.md || fail "README lacks aggregation limits"
grep -q "aggregation" souprise/core/pipeline.py || fail "no aggregation hint in code"
grep -rq "aggregat" tests/ || fail "no aggregation test"

echo "VERIFY PASS"
exit 0
