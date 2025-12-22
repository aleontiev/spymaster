"""
Dataset configuration commands for SpyMaster CLI.

Commands for managing dataset configurations in the database.
"""

from pathlib import Path
from typing import Optional

import typer
from rich import box
from rich.panel import Panel
from rich.table import Table

from commands import console, PROJECT_ROOT

# Create the typer app for datasets commands
datasets_app = typer.Typer(help="Manage dataset configurations (database)", invoke_without_command=True)


@datasets_app.callback()
def datasets_callback(ctx: typer.Context):
    """Manage dataset configurations. Run without subcommand to list all."""
    if ctx.invoked_subcommand is None:
        datasets_list()


@datasets_app.command("list")
def datasets_list(
    dataset_type: Optional[str] = typer.Option(None, "--type", "-t", help="Filter by type (source/computed)"),
    show_deps: bool = typer.Option(False, "--deps", "-d", help="Show dependencies"),
):
    """List all dataset configurations."""
    from src.db.database import get_db

    # Handle direct calls (not through Typer)
    if hasattr(dataset_type, 'default'):
        dataset_type = None
    if hasattr(show_deps, 'default'):
        show_deps = False

    db = get_db()
    datasets = db.list_datasets(type=dataset_type)

    if not datasets:
        console.print("[yellow]No datasets configured in database.[/yellow]")
        console.print("[dim]Run 'manage.py datasets migrate' to import from datasets.json[/dim]")
        return

    table = Table(title="Dataset Configurations", box=box.ROUNDED)
    table.add_column("Name", style="cyan")
    table.add_column("Type", style="green")
    table.add_column("Provider/Computation", style="yellow")
    table.add_column("Granularity")
    if show_deps:
        table.add_column("Dependencies", style="dim")

    for ds in datasets:
        provider_or_comp = ds.provider if ds.type == "source" else ds.computation or "-"
        if provider_or_comp and len(provider_or_comp) > 40:
            provider_or_comp = "..." + provider_or_comp[-37:]

        row = [
            ds.name,
            ds.type,
            provider_or_comp or "-",
            ds.granularity,
        ]

        if show_deps:
            deps = [d.dependency_name for d in ds.dependencies]
            row.append(", ".join(deps) if deps else "-")

        table.add_row(*row)

    console.print(table)
    console.print(f"\n[dim]Total: {len(datasets)} datasets[/dim]")


@datasets_app.command("show")
def datasets_show(
    name: str = typer.Argument(..., help="Dataset name"),
):
    """Show detailed configuration for a dataset."""
    from src.db.database import get_db

    db = get_db()
    dataset = db.get_dataset(name)

    if not dataset:
        console.print(f"[red]Dataset '{name}' not found[/red]")
        raise typer.Exit(1)

    info_lines = [
        f"[bold]Name:[/bold] {dataset.name}",
        f"[bold]Type:[/bold] {dataset.type}",
        f"[bold]Description:[/bold] {dataset.description or '-'}",
        f"[bold]Path Pattern:[/bold] {dataset.path_pattern}",
        f"[bold]Granularity:[/bold] {dataset.granularity}",
        f"[bold]Ephemeral:[/bold] {dataset.ephemeral}",
    ]

    if dataset.type == "source":
        info_lines.append(f"[bold]Provider:[/bold] {dataset.provider}")
        if dataset.fetch_config:
            info_lines.append(f"[bold]Fetch Config:[/bold]")
            for k, v in dataset.fetch_config.items():
                info_lines.append(f"  {k}: {v}")
    else:
        info_lines.append(f"[bold]Computation:[/bold] {dataset.computation}")

    console.print(Panel("\n".join(info_lines), title=f"Dataset: {name}", border_style="cyan"))

    if dataset.fields:
        field_table = Table(title="Fields", box=box.SIMPLE)
        field_table.add_column("Name", style="cyan")
        field_table.add_column("Type")
        field_table.add_column("Source")
        field_table.add_column("Description", style="dim")

        for f in dataset.fields:
            field_table.add_row(f.name, f.type, f.source or "-", f.description or "-")

        console.print(field_table)

    if dataset.dependencies:
        dep_table = Table(title="Dependencies", box=box.SIMPLE)
        dep_table.add_column("Dataset", style="cyan")
        dep_table.add_column("Relation")
        dep_table.add_column("Days")
        dep_table.add_column("Required")

        for d in dataset.dependencies:
            dep_table.add_row(
                d.dependency_name,
                d.relation,
                str(d.days) if d.days else "-",
                "Yes" if d.required else "No",
            )

        console.print(dep_table)


@datasets_app.command("migrate")
def datasets_migrate(
    json_path: Optional[Path] = typer.Option(None, "--from", "-f", help="Path to datasets.json"),
    force: bool = typer.Option(False, "--force", help="Overwrite existing datasets"),
):
    """Migrate datasets from datasets.json to database."""
    from src.db.database import get_db

    if json_path is None:
        json_path = PROJECT_ROOT / "datasets.json"

    if not json_path.exists():
        console.print(f"[red]File not found: {json_path}[/red]")
        raise typer.Exit(1)

    db = get_db()

    existing = db.list_datasets()
    if existing and not force:
        console.print(f"[yellow]Database already has {len(existing)} datasets.[/yellow]")
        console.print("[dim]Use --force to overwrite, or use 'datasets delete-all' first[/dim]")
        return

    if force and existing:
        console.print(f"[yellow]Deleting {len(existing)} existing datasets...[/yellow]")
        for ds in existing:
            db.delete_dataset(ds.name)

    console.print(f"[cyan]Importing from {json_path}...[/cyan]")
    count = db.import_datasets_from_json(json_path)

    console.print(f"[green]Successfully imported {count} datasets![/green]")
    datasets_list()


@datasets_app.command("delete")
def datasets_delete(
    name: str = typer.Argument(..., help="Dataset name to delete"),
    force: bool = typer.Option(False, "--force", "-f", help="Skip confirmation"),
):
    """Delete a dataset configuration from database."""
    from src.db.database import get_db

    db = get_db()

    if not db.dataset_exists(name):
        console.print(f"[red]Dataset '{name}' not found[/red]")
        raise typer.Exit(1)

    if not force:
        confirm = typer.confirm(f"Delete dataset '{name}'?")
        if not confirm:
            console.print("[yellow]Cancelled[/yellow]")
            return

    db.delete_dataset(name)
    console.print(f"[green]Deleted dataset '{name}'[/green]")


@datasets_app.command("delete-all")
def datasets_delete_all(
    force: bool = typer.Option(False, "--force", "-f", help="Skip confirmation"),
):
    """Delete all dataset configurations from database."""
    from src.db.database import get_db

    db = get_db()
    datasets = db.list_datasets()

    if not datasets:
        console.print("[yellow]No datasets to delete[/yellow]")
        return

    if not force:
        confirm = typer.confirm(f"Delete all {len(datasets)} datasets?")
        if not confirm:
            console.print("[yellow]Cancelled[/yellow]")
            return

    for ds in datasets:
        db.delete_dataset(ds.name)

    console.print(f"[green]Deleted {len(datasets)} datasets[/green]")
