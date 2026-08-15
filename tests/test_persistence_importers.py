"""Tests for index persistence and the data importers.

Everything here runs against real files in tmp_path. The PostgreSQL test
runs only when SOUPRISE_TEST_PG_DSN points at a reachable database (CI
provides a service container on Linux; locally, any Postgres works).

License: Apache-2.0
Copyright 2026 Michael Kupermann
"""

import csv
import json
import os

import pytest

from souprise.core.hdc import SimpleHDCRetriever
from souprise.data import importers
from souprise.data.generators.business import generate_business_data

RECORDS = [
    {"invoice_id": "INV-001", "customer": "ACME Corp", "amount": "12400",
     "status": "overdue"},
    {"invoice_id": "INV-002", "customer": "Globex", "amount": "800",
     "status": "paid"},
    {"invoice_id": "INV-003", "customer": "Initech", "amount": "5300",
     "status": "open"},
]


class TestPersistence:
    def test_save_load_roundtrip(self, tmp_path):
        """A loaded index returns identical results without re-encoding."""
        entries = [
            e.to_retrieval_format() for e in generate_business_data(n=500, seed=11)
        ]
        original = SimpleHDCRetriever()
        original.index(entries)
        path = str(tmp_path / "index.db")
        original.save(path)

        restored = SimpleHDCRetriever.load(path)
        assert restored.size == original.size
        assert restored.index_bytes == original.index_bytes

        query = "overdue invoice for Customer_0042"
        a = original.search(query, k=5)
        b = restored.search(query, k=5)
        assert [r.title for r in a] == [r.title for r in b]
        assert [r.score for r in a] == [r.score for r in b]

    def test_metadata_survives_roundtrip(self, tmp_path):
        retriever = SimpleHDCRetriever()
        retriever.index([
            {"id": "X", "text": "some text", "metadata": {"tags": ["a", "b"]}},
        ])
        path = str(tmp_path / "index.db")
        retriever.save(path)
        restored = SimpleHDCRetriever.load(path)
        assert restored.search("some text", k=1)[0].metadata == {"tags": ["a", "b"]}

    def test_load_missing_file_raises(self, tmp_path):
        with pytest.raises(Exception):
            SimpleHDCRetriever.load(str(tmp_path / "missing.db"))


class TestCSVImporter:
    def _write_csv(self, tmp_path, delimiter=","):
        path = tmp_path / "invoices.csv"
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=RECORDS[0].keys(),
                                    delimiter=delimiter)
            writer.writeheader()
            writer.writerows(RECORDS)
        return str(path)

    def test_csv_to_search(self, tmp_path):
        """CSV rows become searchable entries end to end."""
        entries = importers.load_csv(
            self._write_csv(tmp_path), id_column="invoice_id",
            tag_columns=["status"],
        )
        assert len(entries) == 3
        assert entries[0]["id"] == "INV-001"
        assert "customer: ACME Corp" in entries[0]["text"]
        assert entries[0]["metadata"] == {"tags": ["overdue"]}

        retriever = SimpleHDCRetriever()
        retriever.index(entries)
        assert retriever.search("overdue ACME invoice", k=1)[0].title == "INV-001"

    def test_semicolon_delimiter_sniffed(self, tmp_path):
        entries = importers.load_csv(
            self._write_csv(tmp_path, delimiter=";"), id_column="invoice_id"
        )
        assert len(entries) == 3
        assert "Globex" in entries[1]["text"]

    def test_text_columns_subset(self, tmp_path):
        entries = importers.load_csv(
            self._write_csv(tmp_path), id_column="invoice_id",
            text_columns=["customer"],
        )
        assert "amount" not in entries[0]["text"]
        assert "ACME Corp" in entries[0]["text"]


class TestExcelImporter:
    def test_xlsx_to_search(self, tmp_path):
        openpyxl = pytest.importorskip("openpyxl")
        path = str(tmp_path / "invoices.xlsx")
        workbook = openpyxl.Workbook()
        sheet = workbook.active
        sheet.append(list(RECORDS[0].keys()))
        for record in RECORDS:
            sheet.append(list(record.values()))
        workbook.save(path)

        entries = importers.load_excel(path, id_column="invoice_id")
        assert len(entries) == 3
        retriever = SimpleHDCRetriever()
        retriever.index(entries)
        assert retriever.search("paid invoice Globex", k=1)[0].title == "INV-002"


class TestJSONLImporter:
    def test_native_and_tabular_lines(self, tmp_path):
        path = tmp_path / "data.jsonl"
        with open(path, "w") as f:
            f.write(json.dumps({"id": "A", "text": "native entry text",
                                "metadata": {"tags": ["x"]}}) + "\n")
            f.write(json.dumps({"id": "B", "customer": "ACME",
                                "status": "open"}) + "\n")
        entries = importers.load_jsonl(str(path))
        assert entries[0] == {"id": "A", "text": "native entry text",
                              "metadata": {"tags": ["x"]}}
        assert entries[1]["id"] == "B"
        assert "customer: ACME" in entries[1]["text"]


PG_DSN = os.environ.get("SOUPRISE_TEST_PG_DSN")


@pytest.mark.skipif(not PG_DSN, reason="SOUPRISE_TEST_PG_DSN not set")
class TestPostgresConnector:
    def test_postgres_to_search(self):
        """Round trip against a real PostgreSQL: create, load, index, search."""
        sqlalchemy = pytest.importorskip("sqlalchemy")
        engine = sqlalchemy.create_engine(PG_DSN)
        with engine.begin() as connection:
            connection.execute(sqlalchemy.text(
                "DROP TABLE IF EXISTS souprise_test_invoices"))
            connection.execute(sqlalchemy.text(
                "CREATE TABLE souprise_test_invoices "
                "(invoice_id TEXT, customer TEXT, amount INT, status TEXT)"))
            for record in RECORDS:
                connection.execute(
                    sqlalchemy.text(
                        "INSERT INTO souprise_test_invoices VALUES "
                        "(:invoice_id, :customer, :amount, :status)"),
                    {**record, "amount": int(record["amount"])},
                )
        try:
            entries = importers.load_postgres(
                PG_DSN,
                "SELECT invoice_id, customer, amount, status "
                "FROM souprise_test_invoices",
                id_column="invoice_id",
                tag_columns=["status"],
            )
            assert len(entries) == 3
            retriever = SimpleHDCRetriever()
            retriever.index(entries)
            assert retriever.search("overdue ACME", k=1)[0].title == "INV-001"
        finally:
            with engine.begin() as connection:
                connection.execute(sqlalchemy.text(
                    "DROP TABLE IF EXISTS souprise_test_invoices"))
            engine.dispose()
