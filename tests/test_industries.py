"""Tests for configuration-driven industry profiles.

License: Apache-2.0
Copyright 2026 Michael Kupermann
"""

import pytest

from souprise import RAGConfig, SoupriseRAG
from souprise.data.industries import (
    ProfileError,
    generate_industry_data,
    list_profiles,
    load_profile,
)


class TestProfiles:
    def test_shipped_profiles_load(self):
        names = list_profiles()
        assert "finance_insurance" in names
        assert "logistics" in names
        for name in names:
            profile = load_profile(name)
            assert profile["record_types"]

    def test_unknown_profile_rejected(self):
        with pytest.raises(ProfileError):
            load_profile("no_such_industry")

    def test_generation_is_deterministic(self):
        profile = load_profile("logistics")
        a = generate_industry_data(profile, n=50, seed=9)
        b = generate_industry_data(profile, n=50, seed=9)
        assert [e.content for e in a] == [e.content for e in b]

    def test_records_use_field_value_format(self):
        for name in list_profiles():
            entries = generate_industry_data(load_profile(name), n=30, seed=3)
            for e in entries:
                assert all(": " in line for line in e.content.splitlines())
                assert name in e.tags


class TestVerifiedOverProfiles:
    @pytest.mark.parametrize("name", ["finance_insurance", "logistics"])
    def test_field_lookup_is_exact(self, name):
        entries = generate_industry_data(load_profile(name), n=400, seed=11)
        rag = SoupriseRAG(RAGConfig(retriever="simple",
                                    answer_mode="verified"))
        rag.index_from_entries([e.to_retrieval_format() for e in entries])

        target = entries[0]
        entity = target.title.split()[-1]
        field, value = target.content.splitlines()[1].split(": ", 1)
        result = rag.query(f"What is the {field.lower()} for {entity}?")
        assert value in result.answer or result.ambiguous

    def test_aggregation_over_profile_data(self):
        entries = generate_industry_data(load_profile("logistics"),
                                         n=400, seed=11)
        rag = SoupriseRAG(RAGConfig(retriever="simple",
                                    answer_mode="verified"))
        rag.index_from_entries([e.to_retrieval_format() for e in entries])
        result = rag.query("What is the total amount of all overdue invoices?")
        assert result.computed
