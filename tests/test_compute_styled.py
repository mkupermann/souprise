"""Tests for deterministic aggregation and the styled (verbalizer) mode.

License: Apache-2.0
Copyright 2026 Michael Kupermann
"""

from souprise import RAGConfig, SoupriseRAG
from souprise.core.compute import compute_aggregate, parse_aggregate
from souprise.core.pipeline import BaseGenerator, GenerationResult

ENTRIES = [
    {"id": "I1", "text": "Invoice A_01 Jan 2025\nCustomer: A_01\n"
                         "Amount: $100.00\nStatus: overdue\nRegion: EU"},
    {"id": "I2", "text": "Invoice B_02 Feb 2025\nCustomer: B_02\n"
                         "Amount: $250.50\nStatus: overdue\nRegion: EU"},
    {"id": "I3", "text": "Invoice C_03 Mar 2025\nCustomer: C_03\n"
                         "Amount: $49.50\nStatus: paid\nRegion: US"},
]


class TestParseAggregate:
    def test_sum_with_status_filter(self):
        op, field, filters = parse_aggregate("What is the total amount of all overdue invoices?")
        assert (op, field) == ("sum", "Amount")
        assert filters.get("Status") == "overdue"

    def test_count(self):
        op, field, filters = parse_aggregate("How many invoices are overdue?")
        assert op == "count"
        assert filters.get("Status") == "overdue"

    def test_not_an_aggregate(self):
        assert parse_aggregate("What is the amount for A_01?") is None


class TestComputeAggregate:
    def test_sum_exact_decimal(self):
        result = compute_aggregate(
            "What is the total amount of all overdue invoices?", ENTRIES)
        assert result.computed
        assert result.value == "350.50"
        assert result.record_count == 2
        assert "350.50" in result.text

    def test_average(self):
        result = compute_aggregate("average amount of all invoices?", ENTRIES)
        assert result.value == "133.33"

    def test_count_plain(self):
        result = compute_aggregate("How many invoices are overdue?", ENTRIES)
        assert result.value == "2"

    def test_entity_filter(self):
        result = compute_aggregate("total amount for A_01?", ENTRIES)
        assert result.value == "100.00"
        assert result.record_count == 1

    def test_no_matches_is_explicit(self):
        result = compute_aggregate("total amount of all cancelled invoices?",
                                   ENTRIES)
        assert result.record_count == 0
        assert "nothing to compute" in result.text.lower()


class PhrasingGenerator(BaseGenerator):
    """Fake LLM used for the styled mode."""

    def __init__(self, reply):
        self.reply = reply

    def load(self, model_path):
        pass

    def generate(self, prompt, **kwargs):
        return GenerationResult(text=self.reply, latency=0.0)


class TestStyledMode:
    def _rag(self, reply):
        rag = SoupriseRAG(RAGConfig(retriever="simple", answer_mode="styled"))
        rag.generator = PhrasingGenerator(reply)
        rag.index_from_entries(ENTRIES)
        return rag

    def test_verified_pipeline_computes_aggregates(self):
        rag = SoupriseRAG(RAGConfig(retriever="simple", answer_mode="verified"))
        rag.index_from_entries(ENTRIES)
        result = rag.query("What is the total amount of all overdue invoices?")
        assert result.computed and result.verified
        assert "350.50" in result.answer
        assert not result.aggregation_hint  # computed, not just hinted

    def test_styled_keeps_faithful_phrasing(self):
        rag = self._rag("The two overdue invoices add up to 350.50 in total.")
        result = rag.query("What is the total amount of all overdue invoices?")
        assert result.answer == "The two overdue invoices add up to 350.50 in total."
        assert result.verified and result.computed
        assert result.blocked_generation is None

    def test_styled_blocks_wrong_figures(self):
        rag = self._rag("The overdue invoices total exactly $999.99.")
        result = rag.query("What is the total amount of all overdue invoices?")
        assert "999.99" not in result.answer
        assert "350.50" in result.answer  # deterministic text stands
        assert result.blocked_generation == "The overdue invoices total exactly $999.99."

    def test_styled_refusal_skips_llm(self):
        rag = self._rag("should never be used")
        result = rag.query("What is the amount for Zorblatt_777?")
        assert result.refused
        assert result.answer != "should never be used"
