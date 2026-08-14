"""Chat command for Souprise RAG system.

Provides an interactive chat interface for querying business data.

License: Apache-2.0
Copyright 2026 Michael Kupermann
"""

import typer
from rich.console import Console
from rich.panel import Panel
from rich.text import Text
from typing import Optional, List

from souprise.core.pipeline import SoupriseRAG, RAGConfig, quickstart
from souprise.data.generators.business import generate_alpaca_training_data

app = typer.Typer(help="Interactive chat with business data using RAG.")
console = Console()


@app.command()
def chat(
    model: str = typer.Option(
        "mlx-community/Phi-2-4bit",
        help="Path or HuggingFace ID for the LLM model"
    ),
    backend: str = typer.Option(
        "mlx",
        help="Backend for LLM: 'mlx' for Apple Silicon, 'torch' for CUDA"
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
        souprise chat --model mlx-community/Phi-2-4bit --backend mlx
    """
    console.print(Panel(
        "[bold blue]Souprise[/bold blue] - Offline RAG for Business Data",
        border_style="blue"
    ))
    
    if verbose:
        console.print("[yellow]Initializing RAG pipeline...[/yellow]")
    
    # Initialize RAG pipeline
    config = RAGConfig(
        retrieval_k=retrieval_k,
        model_path=model,
        backend=backend,
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
                console.print(f"[bold green]Answer:[/bold green]")
                console.print(result.answer)
                console.print()
                
                if verbose:
                    console.print(f"[dim]Retrieval: {result.retrieval_latency*1000:.2f}ms | "
                                 f"Generation: {result.generation_latency*1000:.2f}ms | "
                                 f"Total: {result.total_latency*1000:.2f}ms[/dim]")
                
            except KeyboardInterrupt:
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
    model: str = typer.Option("mlx-community/Phi-2-4bit", help="Model path or ID"),
    backend: str = typer.Option("mlx", help="Backend: 'mlx' or 'torch'"),
    data_size: int = typer.Option(10000, help="Number of data entries"),
    retrieval_k: int = typer.Option(5, help="Number of retrieval results"),
):
    """Ask a single question and get an answer.
    
    Example:
        souprise chat query "What are the open invoices?"
    """
    # Initialize RAG
    config = RAGConfig(
        retrieval_k=retrieval_k,
        model_path=model,
        backend=backend
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
