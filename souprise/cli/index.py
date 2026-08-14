"""Index command for Souprise.

Provides commands for managing the HDC index.

License: Apache-2.0
Copyright 2026 Michael Kupermann
"""

import typer
from pathlib import Path
from rich.console import Console
import json

from souprise.core.pipeline import SoupriseRAG, RAGConfig
from souprise.data.generators.business import generate_business_data

app = typer.Typer(help="Manage the HDC index for business data.")
console = Console()


@app.command()
def create(
    data_path: str = typer.Option(
        None,
        help="Path to JSONL file with data to index. If None, generates synthetic data."
    ),
    n: int = typer.Option(
        10000,
        help="Number of synthetic entries to generate (used if data_path is None)"
    ),
    seed: int = typer.Option(
        42,
        help="Random seed for synthetic data"
    ),
    output_path: str = typer.Option(
        "./souprise_index",
        help="Directory to save the index (not implemented in HDC yet)"
    ),
):
    """Create an HDC index from business data.
    
    Example:
        souprise index create --data-path my_data.jsonl
        souprise index create --n 5000  # Generate synthetic data
    """
    rag = SoupriseRAG()
    
    if data_path:
        # Load from file
        with open(data_path, "r") as f:
            entries = [
                {
                    "id": f"entry_{i}",
                    "text": json.dumps(line),
                    "metadata": {}
                }
                for i, line in enumerate(f)
            ]
        console.print(f"[yellow]Loading data from {data_path}...[/yellow]")
    else:
        # Generate synthetic data
        entries = generate_business_data(n=n, seed=seed)
        entries = [
            {
                "id": entry.title,
                "text": f"{entry.title}\n{entry.content}",
                "metadata": {"tags": entry.tags}
            }
            for entry in entries
        ]
        console.print(f"[yellow]Generating {n} synthetic business entries...[/yellow]")
    
    # Index data
    rag.index_from_entries(entries)
    console.print(f"[green]Indexed {len(entries)} entries[/green]")
    console.print("[yellow]Note: Current HDC implementation keeps index in memory only[/yellow]")
    console.print("[yellow]For persistent storage, see the JuiceHDC documentation[/yellow]")


@app.command()
def info(
    data_size: int = typer.Option(10000, help="Number of entries"),
):
    """Show information about the HDC index.
    
    Example:
        souprise index info --data-size 10000
    """
    # Generate sample data to show stats
    entries = generate_business_data(n=data_size, seed=42)
    
    # Count by category
    categories = {}
    for entry in entries:
        cat = entry.tags[0] if entry.tags else "unknown"
        categories[cat] = categories.get(cat, 0) + 1
    
    console.print("[bold blue]HDC Index Statistics[/bold blue]")
    console.print(f"Total entries: {len(entries)}")
    console.print(f"Vector dimension: 10,000 bits (HDC)")
    console.print(f"Storage per entry: ~1.25 KB (packed)")
    console.print(f"Total storage: ~{len(entries) * 1.25 / 1024:.2f} MB")
    console.print()
    console.print("[bold]Entries by category:[/bold]")
    for cat, count in sorted(categories.items()):
        console.print(f"  {cat}: {count} ({count/len(entries)*100:.1f}%)")
