"""Main CLI application for Souprise.

This module combines all CLI commands into a single entry point.

License: Apache-2.0
Copyright 2026 Michael Kupermann
"""

import typer
from rich.console import Console

from .chat import app as chat_app
from .train import app as train_app
from .index import app as index_app

app = typer.Typer(
    name="souprise",
    help="Offline RAG for Business Data: HDC Retrieval + LLM Generation",
    no_args_is_help=True,
    rich_markup_mode="rich",
)

# Register subcommands
app.add_typer(chat_app, name="chat", help="Interactive chat with business data")
app.add_typer(train_app, name="train", help="Fine-tune LLMs with synthetic data")
app.add_typer(index_app, name="index", help="Manage the HDC index")


@app.command()
def version():
    """Show Souprise version."""
    from souprise import __version__
    console = Console()
    console.print(f"Souprise version: {__version__}")


@app.command()
def info():
    """Show system information."""
    import sys
    from rich.table import Table
    
    console = Console()
    table = Table(title="Souprise System Info")
    table.add_column("Component")
    table.add_column("Status")
    
    # Python version
    table.add_row("Python", f"{sys.version.split()[0]}")
    
    # Check dependencies
    try:
        import numpy
        table.add_row("NumPy", f"{numpy.__version__}")
    except ImportError:
        table.add_row("NumPy", "[red]Not installed[/red]")
    
    try:
        import mlx
        table.add_row("MLX", f"{mlx.__version__}")
    except ImportError:
        table.add_row("MLX", "[yellow]Not installed (Apple Silicon only)[/yellow]")
    
    try:
        import torch
        table.add_row("PyTorch", f"{torch.__version__}")
    except ImportError:
        table.add_row("PyTorch", "[yellow]Not installed[/yellow]")
    
    try:
        from cortex import __version__ as cortex_version
        table.add_row("JuiceHDC", f"{cortex_version}")
    except ImportError:
        table.add_row("JuiceHDC", "[yellow]Not installed[/yellow]")
    
    try:
        from soup_cli import __version__ as soup_version
        table.add_row("Soup", f"{soup_version}")
    except ImportError:
        table.add_row("Soup", "[yellow]Not installed[/yellow]")
    
    console.print(table)


if __name__ == "__main__":
    app()
