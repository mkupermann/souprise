"""Tenant management commands.

License: Apache-2.0
Copyright 2026 Michael Kupermann
"""

import typer
from rich.console import Console
from rich.table import Table

from souprise.core.tenants import DEFAULT_BASE_DIR, TenantError, TenantManager

app = typer.Typer(help="Manage physically isolated tenants.")
console = Console()


@app.command()
def create(
    name: str = typer.Argument(..., help="Tenant name (lowercase slug)"),
    base_dir: str = typer.Option(DEFAULT_BASE_DIR,
                                 help="Base directory holding all tenants"),
):
    """Create a tenant directory with its own index, audit log and policies."""
    try:
        tenant = TenantManager(base_dir).create(name)
    except TenantError as e:
        console.print(f"[red]{e}[/red]")
        raise typer.Exit(1)
    console.print(f"[green]Tenant '{tenant.name}' ready at {tenant.root}[/green]")
    console.print(f"Index:    {tenant.index_path}")
    console.print(f"Audit:    {tenant.audit_path}")
    console.print(f"Policies: {tenant.policies_dir}/<name>.json")


@app.command("list")
def list_tenants(
    base_dir: str = typer.Option(DEFAULT_BASE_DIR,
                                 help="Base directory holding all tenants"),
):
    """List tenants under the base directory."""
    names = TenantManager(base_dir).list()
    if not names:
        console.print(f"No tenants under {base_dir}.")
        return
    table = Table(title=f"Tenants in {base_dir}")
    table.add_column("Name")
    table.add_column("Index")
    table.add_column("Audit events")
    mgr = TenantManager(base_dir)
    for name in names:
        tenant = mgr.get(name)
        from pathlib import Path
        has_index = "yes" if Path(tenant.index_path).exists() else "no"
        events = "0"
        if Path(tenant.audit_path).exists():
            from souprise.core.audit import AuditLog
            events = str(AuditLog(tenant.audit_path).count())
        table.add_row(name, has_index, events)
    console.print(table)
