"""Chat command for Souprise RAG system.

Provides an interactive chat interface for querying business data.

License: Apache-2.0
Copyright 2026 Michael Kupermann
"""


from typing import Optional

import typer
from rich.console import Console
from rich.panel import Panel

from souprise.core.pipeline import DEFAULT_MODELS, RAGConfig, SoupriseRAG, resolve_backend

app = typer.Typer(help="Interactive chat with business data using RAG.")
console = Console()


@app.callback(invoke_without_command=True)
def chat(
    ctx: typer.Context,
    model: Optional[str] = typer.Option(
        None,
        help="Path or HuggingFace ID for the LLM model "
             "(default: a small base model matching the backend)"
    ),
    backend: str = typer.Option(
        "auto",
        help="Backend: 'auto' (detect), 'mlx' (Apple Silicon), 'torch' (CUDA/ROCm/CPU)"
    ),
    data_size: int = typer.Option(
        10000,
        help="Number of synthetic business entries to generate"
    ),
    retrieval_k: int = typer.Option(
        5,
        help="Number of retrieval results to use"
    ),
    max_tokens: int = typer.Option(
        256,
        help="Maximum tokens for LLM response"
    ),
    temperature: float = typer.Option(
        0.7,
        help="Temperature for LLM generation (0.0-1.0)"
    ),
    verbose: bool = typer.Option(
        False,
        help="Show verbose output"
    ),
):
    """Start an interactive chat session with business data.

    Example:
        souprise chat --model ./souprise_model
    """
    # A subcommand like `souprise chat query` handles the call itself.
    if ctx.invoked_subcommand is not None:
        return

    console.print(Panel(
        "[bold blue]Souprise[/bold blue] - Offline RAG for Business Data",
        border_style="blue"
    ))

    resolved_backend = resolve_backend(backend)
    model = model or DEFAULT_MODELS[resolved_backend]

    if verbose:
        console.print(f"[yellow]Initializing RAG pipeline ({resolved_backend})...[/yellow]")

    # Initialize RAG pipeline
    config = RAGConfig(
        retrieval_k=retrieval_k,
        model_path=model,
        backend=resolved_backend,
        max_tokens=max_tokens,
        temperature=temperature
    )

    rag = SoupriseRAG(config=config)

    if verbose:
        console.print("[yellow]Generating synthetic business data...[/yellow]")

    # Generate and index synthetic data
    rag.index_from_business_data(n=data_size, seed=42)

    if verbose:
        console.print(f"[yellow]Loading LLM model: {model}...[/yellow]")

    # Load model
    try:
        rag.load_model()
    except Exception as e:
        console.print(f"[red]Error loading model: {e}[/red]")
        raise typer.Exit(1)

    console.print("[green]Ready! Type your questions (Ctrl+C to exit)[/green]")
    console.print()

    # Chat loop
    try:
        while True:
            try:
                # Get user input
                query = input("> ").strip()
                if not query:
                    continue

                # Execute query
                result = rag.query(query)

                # Print answer
                console.print("[bold green]Answer:[/bold green]")
                console.print(result.answer)
                console.print()

                if verbose:
                    console.print(f"[dim]Retrieval: {result.retrieval_latency*1000:.2f}ms | "
                                 f"Generation: {result.generation_latency*1000:.2f}ms | "
                                 f"Total: {result.total_latency*1000:.2f}ms[/dim]")

            except (KeyboardInterrupt, EOFError):
                console.print("\n[yellow]Goodbye![/yellow]")
                break
            except Exception as e:
                console.print(f"[red]Error: {e}[/red]")
    except Exception as e:
        console.print(f"[red]Fatal error: {e}[/red]")
        raise typer.Exit(1)


@app.command()
def query(
    question: str = typer.Argument(..., help="The question to ask"),
    model: Optional[str] = typer.Option(None, help="Model path or ID"),
    backend: str = typer.Option("auto", help="Backend: 'auto', 'mlx', or 'torch'"),
    data_size: int = typer.Option(10000, help="Number of data entries"),
    retrieval_k: int = typer.Option(5, help="Number of retrieval results"),
):
    """Ask a single question and get an answer.

    Example:
        souprise chat query "What are the open invoices?"
    """
    resolved_backend = resolve_backend(backend)
    config = RAGConfig(
        retrieval_k=retrieval_k,
        model_path=model or DEFAULT_MODELS[resolved_backend],
        backend=resolved_backend
    )
    rag = SoupriseRAG(config=config)
    rag.index_from_business_data(n=data_size, seed=42)
    rag.load_model()

    # Execute query
    result = rag.query(question)

    # Print result
    console.print(f"[bold blue]Question:[/bold blue] {question}")
    console.print(f"[bold green]Answer:[/bold green] {result.answer}")
    console.print(f"[dim]Total latency: {result.total_latency*1000:.2f}ms[/dim]")
