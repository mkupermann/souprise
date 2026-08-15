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


def _resolve_tenant(tenant, tenant_dir, index, audit, policy):
    """Map a tenant name onto its private index, audit log and policies.

    Explicit --index/--audit paths still win; a bare --policy name is
    looked up in the tenant's policies directory.
    """
    if not tenant:
        return index, audit, policy
    from pathlib import Path

    from souprise.core.tenants import TenantError, TenantManager
    try:
        t = TenantManager(tenant_dir).get(tenant)
    except TenantError as e:
        console.print(f"[red]{e}[/red]")
        raise typer.Exit(1)
    index = index or t.index_path
    audit = audit or t.audit_path
    if policy and "/" not in policy and not Path(policy).exists():
        policy = t.policy_path(policy.removesuffix(".json"))
    console.print(f"[yellow]Tenant: {t.name}[/yellow]")
    return index, audit, policy


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
    mode: str = typer.Option(
        "verified",
        help="'verified' (default): values copied from records, no LLM on "
             "the factual path. 'styled': deterministic facts, LLM phrases "
             "the sentence behind an exact gate. 'generative': LLM answers "
             "behind a hard grounding gate."
    ),
    index: Optional[str] = typer.Option(
        None,
        help="Persistent index file built with 'souprise index build'. "
             "When omitted, synthetic data is generated and indexed in memory."
    ),
    data_size: int = typer.Option(
        10000,
        help="Number of synthetic business entries to generate (ignored with --index)"
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
    policy: Optional[str] = typer.Option(
        None,
        help="JSON file with an access policy (visible_where conditions "
             "and hidden_fields). Applied to the index before search."
    ),
    audit: Optional[str] = typer.Option(
        None,
        help="Path to an append-only SQLite audit log. Every query is "
             "recorded with record hashes and an answer hash."
    ),
    tenant: Optional[str] = typer.Option(
        None,
        help="Tenant name. Uses that tenant's own index, audit log and "
             "policies; nothing is shared between tenants."
    ),
    tenant_dir: str = typer.Option(
        "tenants",
        help="Base directory holding all tenants"
    ),
    industry: Optional[str] = typer.Option(
        None,
        help="Industry profile for the synthetic data (see 'souprise "
             "industries list'). Ignored with --index."
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
    index, audit, policy = _resolve_tenant(tenant, tenant_dir, index, audit, policy)

    if verbose:
        console.print(f"[yellow]Initializing RAG pipeline ({resolved_backend})...[/yellow]")

    # Initialize RAG pipeline
    config = RAGConfig(
        retrieval_k=retrieval_k,
        model_path=model,
        backend=resolved_backend,
        answer_mode=mode,
        max_tokens=max_tokens,
        temperature=temperature,
        audit_path=audit
    )

    rag = SoupriseRAG(config=config)

    active_policy = None
    if policy:
        from souprise.core.access import load_policy
        active_policy = load_policy(policy)
        console.print(f"[yellow]Access policy: {active_policy.name}[/yellow]")

    if index:
        from pathlib import Path as _Path
        if tenant and not _Path(index).exists():
            console.print(f"[red]Tenant '{tenant}' has no index yet. Build "
                          f"one with 'souprise index build --tenant "
                          f"{tenant} ...'[/red]")
            raise typer.Exit(1)
        from souprise.core.hdc import SimpleHDCRetriever
        if verbose:
            console.print(f"[yellow]Loading index {index}...[/yellow]")
        rag.retriever = SimpleHDCRetriever.load(index)
    elif industry:
        from souprise.data.industries import (
            ProfileError, generate_industry_data, load_profile,
        )
        try:
            profile = load_profile(industry)
        except ProfileError as e:
            console.print(f"[red]{e}[/red]")
            raise typer.Exit(1)
        console.print(f"[yellow]Industry profile: {profile['display']}[/yellow]")
        entries = generate_industry_data(profile, n=data_size, seed=42)
        rag.index_from_entries([e.to_retrieval_format() for e in entries])
    else:
        if verbose:
            console.print("[yellow]Generating synthetic business data...[/yellow]")
        rag.index_from_business_data(n=data_size, seed=42)

    if mode in ("generative", "styled"):
        if verbose:
            console.print(f"[yellow]Loading LLM model: {model}...[/yellow]")
        try:
            rag.load_model()
        except Exception as e:
            console.print(f"[red]Error loading model: {e}[/red]")
            raise typer.Exit(1)
    elif verbose:
        console.print("[green]Verified mode: no model needed, "
                      "values are copied from records.[/green]")

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
                result = rag.query(query, policy=active_policy)

                # Print answer
                console.print("[bold green]Answer:[/bold green]")
                console.print(result.answer)
                if result.policy_denied:
                    console.print("[yellow]Denied by your role's access "
                                  "policy.[/yellow]")
                if result.verified:
                    console.print("[green]Verified: every value above is "
                                  "copied from the cited records.[/green]")
                if result.refused:
                    console.print("[yellow]No sufficiently matching record; "
                                  "refusing beats guessing.[/yellow]")
                if result.blocked_generation:
                    console.print("[red]A generated answer contained figures "
                                  "not present in any record and was blocked; "
                                  "showing the verified fallback.[/red]")
                if result.ungrounded_numbers:
                    console.print(f"[red]Caution: these figures are not in the "
                                  f"retrieved records: "
                                  f"{', '.join(result.ungrounded_numbers)}[/red]")
                if result.aggregation_hint:
                    console.print("[yellow]Note: this looks like an aggregate "
                                  "question. Top-k retrieval only sees a few "
                                  "records; totals and averages need a database "
                                  "query.[/yellow]")
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
    mode: str = typer.Option("verified",
                             help="'verified' (values copied from records), "
                                  "'styled' (LLM phrases, code owns numbers) or "
                                  "'generative' (LLM behind a grounding gate)"),
    index: Optional[str] = typer.Option(None, help="Persistent index file to load"),
    data_size: int = typer.Option(10000, help="Number of data entries (ignored with --index)"),
    retrieval_k: int = typer.Option(5, help="Number of retrieval results"),
    policy: Optional[str] = typer.Option(
        None, help="JSON access policy file, applied before search"),
    audit: Optional[str] = typer.Option(
        None, help="Append-only SQLite audit log path"),
    tenant: Optional[str] = typer.Option(
        None, help="Tenant name; uses that tenant's index, audit and policies"),
    tenant_dir: str = typer.Option(
        "tenants", help="Base directory holding all tenants"),
    industry: Optional[str] = typer.Option(
        None, help="Industry profile for synthetic data (ignored with --index)"),
):
    """Ask a single question and get an answer.

    Example:
        souprise chat query "What are the open invoices?"
    """
    resolved_backend = resolve_backend(backend)
    index, audit, policy = _resolve_tenant(tenant, tenant_dir, index, audit, policy)
    config = RAGConfig(
        retrieval_k=retrieval_k,
        model_path=model or DEFAULT_MODELS[resolved_backend],
        backend=resolved_backend,
        answer_mode=mode,
        audit_path=audit
    )
    rag = SoupriseRAG(config=config)
    active_policy = None
    if policy:
        from souprise.core.access import load_policy
        active_policy = load_policy(policy)
    if index:
        from pathlib import Path as _Path
        if tenant and not _Path(index).exists():
            console.print(f"[red]Tenant '{tenant}' has no index yet. Build "
                          f"one with 'souprise index build --tenant "
                          f"{tenant} ...'[/red]")
            raise typer.Exit(1)
        from souprise.core.hdc import SimpleHDCRetriever
        rag.retriever = SimpleHDCRetriever.load(index)
    elif industry:
        from souprise.data.industries import (
            ProfileError, generate_industry_data, load_profile,
        )
        try:
            profile = load_profile(industry)
        except ProfileError as e:
            console.print(f"[red]{e}[/red]")
            raise typer.Exit(1)
        entries = generate_industry_data(profile, n=data_size, seed=42)
        rag.index_from_entries([e.to_retrieval_format() for e in entries])
    else:
        rag.index_from_business_data(n=data_size, seed=42)
    if mode in ("generative", "styled"):
        rag.load_model()

    # Execute query
    result = rag.query(question, policy=active_policy)

    # Print result
    console.print(f"[bold blue]Question:[/bold blue] {question}")
    console.print(f"[bold green]Answer:[/bold green] {result.answer}")
    if result.policy_denied:
        console.print("[yellow]Denied by your role's access policy.[/yellow]")
    if result.verified:
        console.print("[green]Verified: every value above is copied from "
                      "the cited records.[/green]")
    if result.refused:
        console.print("[yellow]No sufficiently matching record; refusing "
                      "beats guessing.[/yellow]")
    if result.blocked_generation:
        console.print("[red]A generated answer contained ungrounded figures "
                      "and was blocked; showing the verified fallback.[/red]")
    if result.ungrounded_numbers:
        console.print(f"[red]Caution: these figures are not in the retrieved "
                      f"records: {', '.join(result.ungrounded_numbers)}[/red]")
    if result.aggregation_hint:
        console.print("[yellow]Note: aggregate questions (totals, averages) "
                      "need a database query, not top-k retrieval.[/yellow]")
    console.print(f"[dim]Total latency: {result.total_latency*1000:.2f}ms[/dim]")
