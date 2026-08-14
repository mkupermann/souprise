"""Tests for data generators.

License: Apache-2.0
Copyright 2026 Michael Kupermann
"""

import pytest
from souprise.data.generators.business import (
    generate_business_data,
    generate_alpaca_training_data,
    BusinessEntry,
)


class TestBusinessEntry:
    """Tests for BusinessEntry dataclass."""

    def test_to_dict(self):
        """Test conversion to dictionary."""
        entry = BusinessEntry(
            title="Test Entry",
            content="Test content",
            tags=["test", "example"]
        )
        result = entry.to_dict()
        
        assert result["title"] == "Test Entry"
        assert result["content"] == "Test content"
        assert result["tags"] == ["test", "example"]

    def test_to_alpaca_format(self):
        """Test conversion to Alpaca format."""
        entry = BusinessEntry(
            title="Test Invoice",
            content="Customer: Test\nAmount: $100",
            tags=["invoice"]
        )
        result = entry.to_alpaca_format()
        
        assert "instruction" in result
        assert "input" in result
        assert "output" in result
        assert "Test Invoice" in result["instruction"]

    def test_to_retrieval_format(self):
        """Test conversion to retrieval format."""
        entry = BusinessEntry(
            title="Test Entry",
            content="Test content",
            tags=["test"]
        )
        result = entry.to_retrieval_format()
        
        assert "id" in result
        assert "text" in result
        assert "metadata" in result
        assert result["id"] == "Test Entry"


class TestGenerateBusinessData:
    """Tests for business data generation."""

    def test_generate_default(self):
        """Test default data generation."""
        entries = generate_business_data(n=100, seed=42)
        
        assert len(entries) == 100
        assert all(isinstance(e, BusinessEntry) for e in entries)

    def test_generate_reproducible(self):
        """Test that generation is reproducible with same seed."""
        entries1 = generate_business_data(n=50, seed=42)
        entries2 = generate_business_data(n=50, seed=42)
        
        assert len(entries1) == len(entries2)
        for e1, e2 in zip(entries1, entries2):
            assert e1.title == e2.title
            assert e1.content == e2.content
            assert e1.tags == e2.tags

    def test_generate_categories(self):
        """Test generation with specific categories."""
        entries = generate_business_data(n=100, seed=42, categories=["invoice"])
        
        assert len(entries) == 100
        # All entries should be invoices
        assert all("invoice" in e.tags for e in entries)

    def test_generate_custom_counts(self):
        """Test generation with custom entity counts."""
        entries = generate_business_data(
            n=100,
            seed=42,
            customer_count=50,
            product_count=50
        )
        
        assert len(entries) == 100
        # Check that we have reasonable variety in customers
        customers = set()
        for entry in entries:
            if "Invoice" in entry.title:
                # Extract customer from title
                customer = entry.title.split()[1]
                customers.add(customer)
        
        # Should have at least some customers
        assert len(customers) > 0
        assert len(customers) <= 50  # Should not exceed customer_count


class TestGenerateAlpacaTrainingData:
    """Tests for Alpaca format training data generation."""

    def test_generate_default(self):
        """Test default training data generation."""
        data = generate_alpaca_training_data(n=100, seed=42)
        
        assert len(data) > 0
        assert all(isinstance(item, dict) for item in data)
        assert all("instruction" in item for item in data)
        assert all("input" in item for item in data)
        assert all("output" in item for item in data)

    def test_generate_reproducible(self):
        """Test that training data generation is reproducible."""
        data1 = generate_alpaca_training_data(n=50, seed=42)
        data2 = generate_alpaca_training_data(n=50, seed=42)
        
        assert len(data1) == len(data2)
        for d1, d2 in zip(data1, data2):
            assert d1 == d2

    def test_save_to_file(self, tmp_path):
        """Test saving training data to file."""
        output_path = str(tmp_path / "test_data.jsonl")
        data = generate_alpaca_training_data(n=10, seed=42, output_path=output_path)
        
        # Check file exists
        import os
        assert os.path.exists(output_path)
        
        # Check file contents
        with open(output_path, "r") as f:
            lines = f.readlines()
        
        assert len(lines) == len(data)
        for line, item in zip(lines, data):
            import json
            loaded = json.loads(line.strip())
            assert loaded == item
