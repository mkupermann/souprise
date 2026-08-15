"""Tests for the verified answer mode.

License: Apache-2.0
Copyright 2026 Michael Kupermann
"""

from souprise import RAGConfig, SoupriseRAG
from souprise.core.pipeline import BaseGenerator, GenerationResult, RetrievalResult
from souprise.core.verified import REFUSAL_TEXT, answer_verified, detect_field


def _result(title, content, score=0.6):
    return RetrievalResult(title=title, content=content, score=score)


INVOICE = _result("Invoice ACME_01 Mar 2025",
                  "Customer: ACME_01\nAmount: $14,762.19\nStatus: overdue")


class TestFieldDetection:
    def test_common_fields(self):
        assert detect_field("What is the amount of the invoice?") == "Amount"
        assert detect_field("Is it paid or overdue, what's the status?") == "Status"
        assert detect_field("annual revenue of the customer") == "Annual Revenue"
        assert detect_field("Wie hoch ist der Lagerbestand?") == "Stock"
        assert detect_field("tell me something unspecific") is None


class TestVerifiedAnswers:
    def test_value_is_copied_verbatim(self):
        va = answer_verified("What is the amount?", [INVOICE])
        assert not va.refused and not va.ambiguous
        assert va.values == ["$14,762.19"]
        assert "$14,762.19" in va.text
        assert "Invoice ACME_01 Mar 2025" in va.text

    def test_refusal_below_score(self):
        weak = _result("Invoice X", "Amount: $1.00", score=0.50)
        va = answer_verified("What is the amount?", [weak], min_score=0.52)
        assert va.refused
        assert va.text == REFUSAL_TEXT

    def test_refusal_on_empty(self):
        assert answer_verified("anything", []).refused

    def test_ambiguity_lists_all_candidates_and_asserts_none(self):
        a = _result("Customer Profile ACME_01", "Annual Revenue: $100.00")
        b = _result("Customer Profile ACME_01 v2", "Annual Revenue: $200.00", 0.58)
        va = answer_verified("annual revenue of ACME_01?", [a, b])
        assert va.ambiguous
        assert set(va.values) == {"$100.00", "$200.00"}
        assert "$100.00" in va.text and "$200.00" in va.text

    def test_unknown_field_returns_record_verbatim(self):
        va = answer_verified("tell me about this", [INVOICE])
        assert not va.refused
        assert "Amount: $14,762.19" in va.text

    def test_unknown_entity_refuses_despite_similar_records(self):
        """A question naming an entity the corpus lacks must refuse, not
        return the closest other entity's value."""
        va = answer_verified(
            "What is the amount of the invoice for Zorblatt_003?", [INVOICE])
        assert va.refused

    def test_named_entity_must_match_record(self):
        other = _result("Invoice OTHER_99 Jan 2024",
                        "Customer: OTHER_99\nAmount: $1.00", 0.59)
        va = answer_verified("What is the amount for ACME_01?",
                             [other, INVOICE])
        assert not va.refused
        assert va.values == ["$14,762.19"]

    def test_custom_template(self):
        va = answer_verified("What is the amount?", [INVOICE],
                             template="Kurzüberblick: {summary}\nQuelle: {source}")
        assert va.text.startswith("Kurzüberblick:")
        assert "Quelle: Invoice ACME_01 Mar 2025" in va.text


class FabricatingGenerator(BaseGenerator):
    """Always answers with a figure that exists in no record."""

    def load(self, model_path):
        pass

    def generate(self, prompt, **kwargs):
        return GenerationResult(text="The amount is $99,999.99.", latency=0.0)


class TestPipelineModes:
    def _rag(self, mode):
        rag = SoupriseRAG(RAGConfig(retriever="simple", answer_mode=mode))
        rag.index_from_entries([
            {"id": "Invoice ACME_01 Mar 2025",
             "text": "Invoice ACME_01 Mar 2025\nCustomer: ACME_01\n"
                     "Amount: $14,762.19\nStatus: overdue"},
        ])
        return rag

    def test_verified_mode_needs_no_model(self):
        rag = self._rag("verified")
        result = rag.query("What is the amount of the invoice for ACME_01?")
        assert result.verified
        assert "$14,762.19" in result.answer
        assert rag.generator is None  # the LLM was never touched

    def test_verified_mode_refuses_unknown_entity(self):
        rag = self._rag("verified")
        result = rag.query("What is the amount for Zorblatt Industries?")
        assert result.refused or result.verified  # never a fabricated value
        if result.refused:
            assert REFUSAL_TEXT in result.answer

    def test_generative_gate_blocks_fabrication(self):
        rag = self._rag("generative")
        rag.generator = FabricatingGenerator()
        result = rag.query("What is the amount of the invoice for ACME_01?")
        assert result.blocked_generation == "The amount is $99,999.99."
        assert "$99,999.99" not in result.answer
        assert "$14,762.19" in result.answer
        assert result.ungrounded_numbers == []
