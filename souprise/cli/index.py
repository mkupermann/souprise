"""Index commands: build, inspect, and query persistent HDC indexes.

License: Apache-2.0
Copyright 2026 Michael Kupermann
"""

from typing import List, Optional

import typer
from rich.console import Console
from rich.table import Table

app = typer.Typer(help="Build and query persistent HDC indexes.")
console = Console()


def _split(option: Optional[str]) -> Optional[List[str]]:
    return [part.strip() for part in option.split(",")] if option else None


@app.command()
def build(
    output: str = typer.Option("souprise_index.db", help="Path of the index file to write"),
    from_csv: Optional[str] = typer.Option(None, help="Build from a CSV file"),
    from_xlsx: Optional[str] = typer.Option(None, help="Build from an Excel workbook"),
    from_jsonl: Optional[str] = typer.Option(None, help="Build from a JSONL file"),
    from_postgres: Optional[str] = typer.Option(
        None, help="Build from PostgreSQL; a SQLAlchemy DSN like postgresql://user@host/db"
    ),
    query: Optional[str] = typer.Option(
        None, help="SELECT statement (required with --from-postgres)"
    ),
    sheet: Optional[str] = typer.Option(None, help="Excel sheet name (default: active)"),
    id_column: Optional[str] = typer.Option(None, help="Column to use as entry id"),
    text_columns: Optional[str] = typer.Option(
        None, help="Comma-separated columns for the searchable text (default: all)"
    ),
    tag_columns: Optional[str] = typer.Option(
        None, help="Comma-separated columns whose values become tags"
    ),
    synthetic: Optional[int] = typer.Option(
        None, help="Build from N synthetic business records instead of a file"
    ),
    seed: int = typer.Option(42, help="Seed for --synthetic"),
    tenant: Optional[str] = typer.Option(
        None, help="Write the index into this tenant's directory "
                   "(creates the tenant if needed)"),
    tenant_dir: str = typer.Option(
        "tenants", help="Base directory holding all tenants"),
):
    """Build a persistent HDC index from CSV, Excel, JSONL, PostgreSQL, or synthetic data.

    Examples:
        souprise index build --from-csv invoices.csv --id-column invoice_id
        souprise index build --from-postgres postgresql://localhost/erp \\
            --query "SELECT id, customer, amount, status FROM invoices" --id-column id
        souprise index build --synthetic 10000
    """
    from souprise.core.hdc import SimpleHDCRetriever
    from souprise.data import importers

    if tenant:
        from souprise.core.tenants import TenantError, TenantManager
        try:
            output = TenantManager(tenant_dir).create(tenant).index_path
        except TenantError as e:
            console.print(f"[red]{e}[/red]")
            raise typer.Exit(1)

    sources = [s for s in (from_csv, from_xlsx, from_jsonl, from_postgres, synthetic) if s]
    if len(sources) != 1:
        console.print("[red]Choose exactly one source: --from-csv, --from-xlsx, "
                      "--from-jsonl, --from-postgres, or --synthetic.[/red]")
        raise typer.Exit(1)

    text_cols, tag_cols = _split(text_columns), _split(tag_columns)

    if from_csv:
        entries = importers.load_csv(from_csv, id_column, text_cols, tag_cols)
    elif from_xlsx:
        entries = importers.load_excel(from_xlsx, sheet, id_column, text_cols, tag_cols)
    elif from_jsonl:
        entries = importers.load_jsonl(from_jsonl)
    elif from_postgres:
        if not query:
            console.print("[red]--from-postgres requires --query.[/red]")
            raise typer.Exit(1)
        entries = importers.load_postgres(from_postgres, query, id_column, text_cols, tag_cols)
    else:
        from souprise.data.generators.business import generate_business_data
        entries = [e.to_retrieval_format() for e in generate_business_data(n=synthetic, seed=seed)]

    if not entries:
        console.print("[red]Source produced no entries.[/red]")
        raise typer.Exit(1)

    console.print(f"[yellow]Encoding {len(entries):,} entries...[/yellow]")
    retriever = SimpleHDCRetriever()
    retriever.index(entries)
    retriever.save(output)
    console.print(f"[green]Indexed {retriever.size:,} entries "
                  f"({retriever.index_bytes / 1_000_000:.2f} MB of vectors) -> {output}[/green]")


@app.command()
def add(
    path: str = typer.Option("souprise_index.db", help="Existing index file to update"),
    from_csv: Optional[str] = typer.Option(None, help="Append entries from a CSV file"),
    from_xlsx: Optional[str] = typer.Option(None, help="Append entries from an Excel workbook"),
    from_jsonl: Optional[str] = typer.Option(None, help="Append entries from a JSONL file"),
    from_postgres: Optional[str] = typer.Option(
        None, help="Append entries from PostgreSQL; a SQLAlchemy DSN"
    ),
    query: Optional[str] = typer.Option(
        None, help="SELECT statement (required with --from-postgres)"
    ),
    sheet: Optional[str] = typer.Option(None, help="Excel sheet name (default: active)"),
    id_column: Optional[str] = typer.Option(None, help="Column to use as entry id"),
    text_columns: Optional[str] = typer.Option(
        None, help="Comma-separated columns for the searchable text (default: all)"
    ),
    tag_columns: Optional[str] = typer.Option(
        None, help="Comma-separated columns whose values become tags"
    ),
):
    """Append new records to an existing index without re-encoding it.

    Only the new entries are encoded, so a daily delta lands in seconds.
    No model training is involved; answers include the new records
    immediately.

    Example (nightly job):
        souprise index add --path souprise_index.db --from-csv todays_invoices.csv \\
            --id-column invoice_id
    """
    from souprise.core.hdc import SimpleHDCRetriever
    from souprise.data import importers

    sources = [s for s in (from_csv, from_xlsx, from_jsonl, from_postgres) if s]
    if len(sources) != 1:
        console.print("[red]Choose exactly one source: --from-csv, --from-xlsx, "
                      "--from-jsonl, or --from-postgres.[/red]")
        raise typer.Exit(1)

    text_cols, tag_cols = _split(text_columns), _split(tag_columns)

    if from_csv:
        entries = importers.load_csv(from_csv, id_column, text_cols, tag_cols)
    elif from_xlsx:
        entries = importers.load_excel(from_xlsx, sheet, id_column, text_cols, tag_cols)
    elif from_jsonl:
        entries = importers.load_jsonl(from_jsonl)
    else:
        if not query:
            console.print("[red]--from-postgres requires --query.[/red]")
            raise typer.Exit(1)
        entries = importers.load_postgres(from_postgres, query, id_column, text_cols, tag_cols)

    if not entries:
        console.print("[yellow]Source produced no entries, index unchanged.[/yellow]")
        return

    retriever = SimpleHDCRetriever.load(path)
    before = retriever.size
    retriever.add(entries)
    retriever.save(path)
    console.print(f"[green]Appended {retriever.size - before:,} entries "
                  f"({before:,} -> {retriever.size:,}) -> {path}[/green]")


@app.command()
def info(
    path: str = typer.Option("souprise_index.db", help="Index file to inspect"),
):
    """Show statistics of a persistent index."""
    from souprise.core.hdc import PACKED_BYTES, SimpleHDCRetriever

    retriever = SimpleHDCRetriever.load(path)
    table = Table(title=f"Index: {path}")
    table.add_column("Property")
    table.add_column("Value")
    table.add_row("Entries", f"{retriever.size:,}")
    table.add_row("Vector size", f"{PACKED_BYTES:,} bytes (10,000 bits)")
    table.add_row("Vector storage", f"{retriever.index_bytes / 1_000_000:.2f} MB")
    console.print(table)


@app.command("query")
def query_index(
    question: str = typer.Argument(..., help="The search query"),
    path: str = typer.Option("souprise_index.db", help="Index file to search"),
    k: int = typer.Option(5, help="Number of results"),
):
    """Search a persistent index directly, without loading an LLM."""
    from souprise.core.hdc import SimpleHDCRetriever

    retriever = SimpleHDCRetriever.load(path)
    for result in retriever.search(question, k=k):
        console.print(f"[bold cyan]{result.score:.3f}[/bold cyan]  "
                      f"[bold]{result.title}[/bold]")
        lines = [line for line in result.content.splitlines() if line != result.title]
        if lines:
            console.print(f"  {lines[0][:100]}")
