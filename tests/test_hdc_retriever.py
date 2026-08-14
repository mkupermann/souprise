"""Tests for the built-in pure-NumPy HDC retriever.

These tests verify the documented claims: 10,000-bit hypervectors stored as
1,250 bytes per entry, deterministic encoding, and XOR + popcount search
returning relevant results.

License: Apache-2.0
Copyright 2026 Michael Kupermann
"""

import pytest

from souprise.core.hdc import DIMENSIONS, PACKED_BYTES, SimpleHDCRetriever
from souprise.data.generators.business import generate_business_data


def make_entries():
    return [
        {"id": "INV-001", "text": "Invoice ACME Corp amount 12400 status overdue"},
        {"id": "INV-002", "text": "Invoice Globex amount 800 status paid"},
        {"id": "ORD-001", "text": "Order Initech Product_AB quantity 40 shipped"},
        {"id": "CUS-001", "text": "Customer profile Globex segment Enterprise region EU"},
    ]


class TestStorageClaims:
    def test_dimensions_and_packing(self):
        """A hypervector is 10,000 bits packed into exactly 1,250 bytes."""
        assert DIMENSIONS == 10_000
        assert PACKED_BYTES == 1_250

        retriever = SimpleHDCRetriever()
        retriever.index(make_entries())
        assert retriever.index_bytes == len(make_entries()) * 1_250

    def test_index_scales_linearly(self):
        entries = [
            e.to_retrieval_format() for e in generate_business_data(n=200, seed=42)
        ]
        retriever = SimpleHDCRetriever()
        retriever.index(entries)
        assert retriever.size == 200
        assert retriever.index_bytes == 200 * 1_250


class TestEncoding:
    def test_deterministic(self):
        r1, r2 = SimpleHDCRetriever(), SimpleHDCRetriever()
        text = "Invoice ACME Corp amount 12400"
        assert (r1._encode(text) == r2._encode(text)).all()

    def test_different_texts_differ(self):
        r = SimpleHDCRetriever()
        a = r._encode("Invoice ACME Corp overdue")
        b = r._encode("Product stock trend declining")
        assert (a != b).any()


class TestSearch:
    def test_returns_relevant_result_first(self):
        retriever = SimpleHDCRetriever()
        retriever.index(make_entries())
        results = retriever.search("overdue invoice for ACME Corp", k=2)
        assert results[0].title == "INV-001"
        assert 0.0 <= results[0].score <= 1.0
        assert results[0].score >= results[1].score

    def test_search_on_synthetic_corpus(self):
        entries = [
            e.to_retrieval_format() for e in generate_business_data(n=500, seed=42)
        ]
        retriever = SimpleHDCRetriever()
        retriever.index(entries)

        # Pick a known record and query for it by its distinctive title terms.
        target = entries[0]
        results = retriever.search(target["id"], k=5)
        assert any(r.title == target["id"] for r in results)

    def test_k_capped_at_corpus_size(self):
        retriever = SimpleHDCRetriever()
        retriever.index(make_entries())
        assert len(retriever.search("invoice", k=50)) == len(make_entries())

    def test_search_before_index_raises(self):
        with pytest.raises(RuntimeError):
            SimpleHDCRetriever().search("anything")

    def test_clear(self):
        retriever = SimpleHDCRetriever()
        retriever.index(make_entries())
        retriever.clear()
        assert retriever.size == 0
        with pytest.raises(RuntimeError):
            retriever.search("anything")


class TestScale:
    """The retriever must handle corpora well beyond 10,000 records."""

    def test_search_on_25k_corpus(self):
        entries = [
            e.to_retrieval_format() for e in generate_business_data(n=25_000, seed=7)
        ]
        retriever = SimpleHDCRetriever()
        retriever.index(entries)
        assert retriever.size == 25_000
        assert retriever.index_bytes == 25_000 * 1_250

        target = entries[12_345]
        results = retriever.search(target["text"], k=5)
        assert results[0].content == target["text"]

    def test_chunked_search_matches_unchunked(self):
        """Chunked distance computation must not change the results."""
        entries = [
            e.to_retrieval_format() for e in generate_business_data(n=1_000, seed=3)
        ]
        small_chunks = SimpleHDCRetriever(chunk_rows=64)
        one_chunk = SimpleHDCRetriever(chunk_rows=10_000_000)
        small_chunks.index(entries)
        one_chunk.index(entries)

        a = small_chunks.search("open invoice Customer_0042", k=10)
        b = one_chunk.search("open invoice Customer_0042", k=10)
        assert [r.title for r in a] == [r.title for r in b]
        assert [r.score for r in a] == [r.score for r in b]

    def test_add_appends_without_reindex(self):
        first = make_entries()
        retriever = SimpleHDCRetriever()
        retriever.index(first)
        retriever.add([{"id": "NEW-001", "text": "Budget Finance 2025 allocated 500000"}])
        assert retriever.size == len(first) + 1
        results = retriever.search("Finance budget 2025 allocation", k=1)
        assert results[0].title == "NEW-001"
