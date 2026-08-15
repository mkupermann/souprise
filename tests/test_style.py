"""Tests for corporate style training data generation.

License: Apache-2.0
Copyright 2026 Michael Kupermann
"""

import json

import pytest

from souprise.data.style import (
    generate_style_training,
    load_glossary,
    template_markers,
    write_style_training,
)

GLOSSARY = {"invoice": "Faktura", "customer": "Geschäftspartner",
            "amount": "Rechnungsbetrag", "stock": "Lagerbestand"}
TEMPLATE = ("Kurzüberblick: {summary}\n"
            "Quelle im Datenbestand: {source}\n"
            "Für Rückfragen steht das Team Finanzbuchhaltung gern bereit.")


class TestGlossaryAndTemplate:
    def test_load_glossary(self, tmp_path):
        path = tmp_path / "g.csv"
        path.write_text("generic,company\ninvoice,Faktura\ncustomer,Geschäftspartner\n",
                        encoding="utf-8")
        assert load_glossary(str(path)) == {"invoice": "Faktura",
                                            "customer": "Geschäftspartner"}

    def test_empty_glossary_raises(self, tmp_path):
        path = tmp_path / "g.csv"
        path.write_text("generic,company\n", encoding="utf-8")
        with pytest.raises(ValueError):
            load_glossary(str(path))

    def test_template_markers(self):
        markers = template_markers(TEMPLATE)
        assert "Kurzüberblick:" in markers
        assert "Quelle im Datenbestand:" in markers
        assert "Für Rückfragen steht das Team Finanzbuchhaltung gern bereit." in markers


class TestStyleGeneration:
    def test_examples_carry_company_terms_and_structure(self):
        examples = generate_style_training(GLOSSARY, TEMPLATE, n=20, seed=5)
        assert len(examples) == 20
        for ex in examples:
            assert "Kurzüberblick:" in ex["output"]
            assert "Für Rückfragen" in ex["output"]
            assert "RECORDS" in ex["instruction"]
            assert "QUESTION:" in ex["instruction"]
        joined = " ".join(e["output"] + e["instruction"] for e in examples)
        assert "Faktura" in joined or "Geschäftspartner" in joined
        assert "invoice" not in " ".join(e["output"] for e in examples)

    def test_seeded_generation_is_deterministic(self):
        a = generate_style_training(GLOSSARY, TEMPLATE, n=10, seed=5)
        b = generate_style_training(GLOSSARY, TEMPLATE, n=10, seed=5)
        assert a == b

    def test_default_seed_randomizes_values(self):
        a = generate_style_training(GLOSSARY, TEMPLATE, n=10)
        b = generate_style_training(GLOSSARY, TEMPLATE, n=10)
        assert a != b

    def test_write_style_training(self, tmp_path):
        g = tmp_path / "g.csv"
        g.write_text("generic,company\ninvoice,Faktura\n", encoding="utf-8")
        t = tmp_path / "t.txt"
        t.write_text(TEMPLATE, encoding="utf-8")
        out = tmp_path / "out.jsonl"
        count = write_style_training(str(g), str(t), str(out), n=5, seed=1)
        assert count == 5
        lines = out.read_text(encoding="utf-8").strip().splitlines()
        assert len(lines) == 5
        assert {"instruction", "input", "output"} <= set(json.loads(lines[0]))

    def test_template_without_summary_raises(self, tmp_path):
        g = tmp_path / "g.csv"
        g.write_text("generic,company\ninvoice,Faktura\n", encoding="utf-8")
        t = tmp_path / "t.txt"
        t.write_text("no placeholder here", encoding="utf-8")
        with pytest.raises(ValueError):
            write_style_training(str(g), str(t), str(tmp_path / "o.jsonl"), n=2)
