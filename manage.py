#!/usr/bin/env python3
"""
SpyMaster CLI - Unified management interface for strategies, checkpoints, and backtests.

Usage:
    python manage.py strategies list
    python manage.py strategies get <name>
    python manage.py strategies backtest <name>
    python manage.py checkpoints list
    python manage.py checkpoints create <name> --epochs 10
    python manage.py checkpoints train <name>
    python manage.py cache list
    python manage.py cache clear all
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Optional

import typer
from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.tree import Tree
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.syntax import Syntax

from src.managers import CheckpointManager, StrategyManager, CacheManager

# Import command apps from commands package
from commands.data import data_app
from commands.datasets import datasets_app
from commands.market_calendar import calendar_app

# Initialize
app = typer.Typer(
    name="spymaster",
    help="SpyMaster CLI - Manage strategies, checkpoints, and backtests",
    no_args_is_help=True,
)
console = Console()

# Sub-apps with invoke_without_command to allow default behavior
# Note: data_app, datasets_app, calendar_app are imported from commands package
strategies_app = typer.Typer(help="Manage trading strategies", invoke_without_command=True)
checkpoints_app = typer.Typer(help="Manage model checkpoints", invoke_without_command=True)
cache_app = typer.Typer(help="Manage data cache", invoke_without_command=True)

app.add_typer(strategies_app, name="strategies")
app.add_typer(checkpoints_app, name="checkpoints")
app.add_typer(cache_app, name="cache")
app.add_typer(data_app, name="data")
app.add_typer(datasets_app, name="datasets")
app.add_typer(calendar_app, name="calendar")


# Callbacks for default behavior (run list when no subcommand specified)
@strategies_app.callback()
def strategies_callback(ctx: typer.Context):
    """Manage trading strategies. Run without subcommand to list all."""
    if ctx.invoked_subcommand is None:
        strategies_list()


@checkpoints_app.callback()
def checkpoints_callback(ctx: typer.Context):
    """Manage model checkpoints. Run without subcommand to list all."""
    if ctx.invoked_subcommand is None:
        checkpoints_list()


@cache_app.callback()
def cache_callback(ctx: typer.Context):
    """Manage data cache. Run without subcommand to list all."""
    if ctx.invoked_subcommand is None:
        cache_list()


# Note: data_app, datasets_app, calendar_app callbacks are defined in their respective modules

# Paths
PROJECT_ROOT = Path(__file__).parent
CHECKPOINTS_DIR = PROJECT_ROOT / "checkpoints"
STRATEGIES_DIR = PROJECT_ROOT / "strategies"
BACKTESTS_DIR = PROJECT_ROOT / "backtests"
CACHE_DIR = PROJECT_ROOT / "data" / "cache"
DATA_DIR = PROJECT_ROOT / "data"

# Initialize managers
checkpoint_manager = CheckpointManager(CHECKPOINTS_DIR, PROJECT_ROOT)
strategy_manager = StrategyManager(STRATEGIES_DIR, BACKTESTS_DIR, CHECKPOINTS_DIR, PROJECT_ROOT)
cache_manager = CacheManager(CACHE_DIR, DATA_DIR, PROJECT_ROOT)


# =============================================================================
# Utility Functions
# =============================================================================


def format_file_size(size_bytes: int) -> str:
    """Format file size in human-readable format."""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size_bytes < 1024:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024
    return f"{size_bytes:.1f} TB"


def format_timestamp(timestamp: float) -> str:
    """Format timestamp as human-readable date."""
    return datetime.fromtimestamp(timestamp).strftime("%Y-%m-%d %H:%M")


def status_color(status: str) -> str:
    """Get color for status."""
    colors = {
        "Trained": "green",
        "In Progress": "yellow",
        "Pending": "blue",
        "Not Found": "red",
        "Empty": "dim",
        "Running": "yellow",
        "Completed": "green",
        "Failed": "red",
    }
    return colors.get(status, "white")


# =============================================================================
# STRATEGIES Commands
# =============================================================================


@strategies_app.command("list")
def strategies_list():
    """List all available strategies."""
    strategies = strategy_manager.list_all()

    table = Table(
        title="Trading Strategies",
        box=box.ROUNDED,
        show_header=True,
        header_style="bold cyan",
    )
    table.add_column("Name", style="bold")
    table.add_column("Type")
    table.add_column("Checkpoint")
    table.add_column("Created")
    table.add_column("Backtests", justify="right")

    if not strategies:
        console.print(Panel(
            "[dim]No strategies found. Create one with:[/dim]\n"
            "[cyan]python manage.py strategies create <name> --checkpoint <checkpoint>[/cyan]",
            title="Strategies",
            border_style="dim",
        ))
        return

    for info in strategies:
        config = info.config
        strategy_type = config.strategy_type if config else "unknown"
        checkpoint = config.checkpoint if config else "-"
        created = format_timestamp(info.modified)

        table.add_row(
            info.name,
            strategy_type,
            checkpoint,
            created,
            str(info.backtest_count),
        )

    console.print(table)


@strategies_app.command("get")
def strategies_get(
    name: str = typer.Argument(..., help="Strategy name"),
    backtests: bool = typer.Option(False, "--backtests", "-b", help="Show backtests"),
    backtest_id: Optional[str] = typer.Option(None, "--backtest-id", help="Specific backtest ID"),
):
    """Get details of a strategy."""
    info = strategy_manager.get(name)

    if not info:
        console.print(f"[red]Strategy '{name}' not found.[/red]")
        raise typer.Exit(1)

    # Show strategy details
    if not backtests and not backtest_id:
        tree = Tree(f"[bold cyan]Strategy: {name}[/bold cyan]")

        if info.config:
            details = tree.add("[bold]Configuration[/bold]")
            details.add(f"[dim]id:[/dim] {info.config.id}")
            details.add(f"[dim]type:[/dim] {info.config.strategy_type}")
            details.add(f"[dim]checkpoint:[/dim] {info.config.checkpoint}")
            details.add(f"[dim]created:[/dim] {info.config.created}")
            if info.config.parameters:
                for key, value in info.config.parameters.items():
                    details.add(f"[dim]{key}:[/dim] {value}")

        # Show files
        files = tree.add("[bold]Files[/bold]")
        for f in info.files:
            size = format_file_size(f.stat().st_size)
            files.add(f"[dim]{f.name}[/dim] ({size})")

        console.print(Panel(tree, border_style="cyan"))
        return

    # Show specific backtest
    if backtest_id:
        result = strategy_manager.get_backtest(name, backtest_id)
        if not result:
            console.print(f"[red]Backtest '{backtest_id}' not found.[/red]")
            raise typer.Exit(1)

        console.print(Panel(
            Syntax(json.dumps(result.raw_data, indent=2), "json", theme="monokai"),
            title=f"Backtest: {backtest_id}",
            border_style="green",
        ))
        return

    # List all backtests
    bt_results = strategy_manager.list_backtests(name)

    if not bt_results:
        console.print(f"[yellow]No backtests found for strategy '{name}'.[/yellow]")
        return

    table = Table(
        title=f"Backtests for {name}",
        box=box.ROUNDED,
        show_header=True,
        header_style="bold green",
    )
    table.add_column("ID", style="bold")
    table.add_column("Date Range")
    table.add_column("Sharpe", justify="right")
    table.add_column("Return %", justify="right")
    table.add_column("Win Rate", justify="right")
    table.add_column("Created")

    for result in bt_results:
        date_range = f"{result.start_date or '?'} - {result.end_date or '?'}"
        sharpe = f"{result.sharpe_ratio:.2f}"
        returns = f"{result.total_return_pct:.1f}%"
        win_rate = f"{result.win_rate*100:.1f}%"
        created = format_timestamp(result.created)

        table.add_row(
            result.backtest_id,
            date_range,
            sharpe,
            returns,
            win_rate,
            created,
        )

    console.print(table)


@strategies_app.command("create")
def strategies_create(
    name: str = typer.Argument(..., help="Strategy name"),
    checkpoint: str = typer.Option(..., "--checkpoint", "-c", help="Checkpoint to use"),
    strategy_type: str = typer.Option("entry_exit", "--type", "-t", help="Strategy type"),
):
    """Create a new strategy."""
    if strategy_manager.exists(name):
        console.print(f"[red]Strategy '{name}' already exists.[/red]")
        raise typer.Exit(1)

    if not checkpoint_manager.exists(checkpoint):
        console.print(f"[red]Checkpoint '{checkpoint}' not found.[/red]")
        raise typer.Exit(1)

    config = strategy_manager.create(name, checkpoint, strategy_type)

    console.print(Panel(
        f"[green]Strategy '{name}' created successfully![/green]\n\n"
        f"[dim]Checkpoint:[/dim] {checkpoint}\n"
        f"[dim]Type:[/dim] {strategy_type}",
        title="Strategy Created",
        border_style="green",
    ))


@strategies_app.command("remove")
def strategies_remove(
    name: str = typer.Argument(..., help="Strategy name"),
    force: bool = typer.Option(False, "--force", "-f", help="Skip confirmation"),
):
    """Remove a strategy."""
    if not strategy_manager.exists(name):
        console.print(f"[red]Strategy '{name}' not found.[/red]")
        raise typer.Exit(1)

    if not force:
        confirm = typer.confirm(f"Are you sure you want to remove strategy '{name}'?")
        if not confirm:
            console.print("[yellow]Cancelled.[/yellow]")
            raise typer.Exit(0)

    strategy_manager.remove(name, remove_backtests=True)
    console.print(f"[green]Strategy '{name}' removed.[/green]")


@strategies_app.command("backtest")
def strategies_backtest(
    name: str = typer.Argument(..., help="Strategy name"),
    start_date: str = typer.Option(None, "--start-date", "-s", help="Start date (YYYY-MM-DD)"),
    end_date: str = typer.Option(None, "--end-date", "-e", help="End date (YYYY-MM-DD)"),
    initial_capital: float = typer.Option(100000, "--capital", help="Initial capital"),
):
    """Run a backtest for a strategy."""
    info = strategy_manager.get(name)

    if not info:
        console.print(f"[red]Strategy '{name}' not found.[/red]")
        raise typer.Exit(1)

    if not info.config or not info.config.checkpoint:
        console.print("[red]Strategy has no checkpoint configured.[/red]")
        raise typer.Exit(1)

    cmd = strategy_manager.build_backtest_command(name, start_date, end_date, initial_capital)

    console.print(Panel(
        f"[cyan]Running backtest for strategy: {name}[/cyan]\n"
        f"[dim]Checkpoint:[/dim] {info.config.checkpoint}\n"
        f"[dim]Date range:[/dim] {start_date or 'all'} to {end_date or 'all'}\n"
        f"[dim]Capital:[/dim] ${initial_capital:,.0f}",
        title="Backtest",
        border_style="cyan",
    ))

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("Running backtest...", total=None)

        try:
            result = strategy_manager.run_backtest(name, start_date, end_date, initial_capital)

            if result and result.returncode == 0:
                console.print("[green]Backtest completed successfully![/green]")
                console.print(result.stdout)
            elif result:
                console.print(f"[red]Backtest failed:[/red]\n{result.stderr}")

        except Exception as e:
            console.print(f"[red]Error running backtest: {e}[/red]")


@strategies_app.command("edit")
def strategies_edit(
    name: str = typer.Argument(..., help="Strategy name"),
    checkpoint: Optional[str] = typer.Option(None, "--checkpoint", "-c", help="Checkpoint to use"),
    strategy_type: Optional[str] = typer.Option(None, "--type", "-t", help="Strategy type"),
    param: Optional[list[str]] = typer.Option(None, "--param", "-p", help="Set parameter as key=value (can be used multiple times)"),
    editor: bool = typer.Option(False, "--editor", "-e", help="Open config in $EDITOR"),
):
    """Edit a strategy configuration.

    Examples:
        python manage.py strategies edit my-strategy --checkpoint new-checkpoint
        python manage.py strategies edit my-strategy --type entry_exit
        python manage.py strategies edit my-strategy --param threshold=0.5 --param window=10
        python manage.py strategies edit my-strategy --editor  # Open in $EDITOR
    """
    import os
    import subprocess

    if not strategy_manager.exists(name):
        console.print(f"[red]Strategy '{name}' not found.[/red]")
        raise typer.Exit(1)

    if editor:
        # Open in editor
        config_path = strategy_manager.get_config_path(name)
        if not config_path:
            console.print(f"[red]No strategy.json found for strategy '{name}'.[/red]")
            raise typer.Exit(1)

        editor_cmd = os.environ.get("EDITOR", "vim")
        subprocess.run([editor_cmd, str(config_path)])
        console.print(f"[green]Config file saved.[/green]")
        return

    # Parse parameters from --param key=value options
    parameters = None
    if param:
        parameters = {}
        for p in param:
            if "=" not in p:
                console.print(f"[red]Invalid parameter format: {p}. Use key=value[/red]")
                raise typer.Exit(1)
            key, value = p.split("=", 1)
            # Try to parse value as number or boolean
            try:
                if value.lower() == "true":
                    parameters[key] = True
                elif value.lower() == "false":
                    parameters[key] = False
                elif "." in value:
                    parameters[key] = float(value)
                else:
                    parameters[key] = int(value)
            except ValueError:
                parameters[key] = value

    # Check if any options were provided
    has_updates = any([checkpoint, strategy_type, parameters])

    if not has_updates:
        console.print("[yellow]No changes specified. Use --editor to open in editor, or specify options to update.[/yellow]")
        console.print("[dim]Example: python manage.py strategies edit my-strategy --checkpoint new-checkpoint[/dim]")
        raise typer.Exit(0)

    # Validate checkpoint if provided
    if checkpoint and not checkpoint_manager.exists(checkpoint):
        console.print(f"[red]Checkpoint '{checkpoint}' not found.[/red]")
        raise typer.Exit(1)

    config = strategy_manager.update(
        name=name,
        checkpoint=checkpoint,
        strategy_type=strategy_type,
        parameters=parameters,
    )

    if config:
        console.print(f"[green]Strategy '{name}' updated successfully![/green]")
        # Show updated config
        strategies_get(name, backtests=False, backtest_id=None)
    else:
        console.print(f"[red]Failed to update strategy '{name}'.[/red]")
        raise typer.Exit(1)


# =============================================================================
# CHECKPOINTS Commands
# =============================================================================


@checkpoints_app.command("list")
def checkpoints_list():
    """List all checkpoints."""
    checkpoints = checkpoint_manager.list_all()

    table = Table(
        title="Model Checkpoints",
        box=box.ROUNDED,
        show_header=True,
        header_style="bold magenta",
    )
    table.add_column("Name", style="bold")
    table.add_column("Status")
    table.add_column("Type")
    table.add_column("Size", justify="right")
    table.add_column("Modified")

    if not checkpoints:
        console.print(Panel(
            "[dim]No checkpoints found. Create one with:[/dim]\n"
            "[cyan]python manage.py checkpoints create <name>[/cyan]",
            title="Checkpoints",
            border_style="dim",
        ))
        return

    for info in checkpoints:
        cp_type = info.config.model_type if info.config else "lejepa"
        size_str = format_file_size(info.total_size)
        modified = format_timestamp(info.modified)
        status_styled = f"[{status_color(info.status)}]{info.status}[/{status_color(info.status)}]"

        table.add_row(info.name, status_styled, cp_type, size_str, modified)

    console.print(table)


@checkpoints_app.command("get")
def checkpoints_get(
    name: str = typer.Argument(..., help="Checkpoint name"),
    show_status: bool = typer.Option(False, "--status", "-s", help="Show only status"),
):
    """Get details of a checkpoint."""
    info = checkpoint_manager.get(name)

    if not info:
        console.print(f"[red]Checkpoint '{name}' not found.[/red]")
        raise typer.Exit(1)

    if show_status:
        console.print(f"[{status_color(info.status)}]{info.status}[/{status_color(info.status)}]")
        return

    tree = Tree(f"[bold magenta]Checkpoint: {name}[/bold magenta]")

    # Status
    tree.add(f"[bold]Status:[/bold] [{status_color(info.status)}]{info.status}[/{status_color(info.status)}]")

    # Configuration
    if info.config:
        config_node = tree.add("[bold]Configuration[/bold]")
        config_node.add(f"[dim]id:[/dim] {info.config.id}")
        config_node.add(f"[dim]model_type:[/dim] {info.config.model_type}")
        config_node.add(f"[dim]created:[/dim] {info.config.created}")

        if info.config.hyperparameters:
            hp_node = config_node.add("[bold]Hyperparameters[/bold]")
            for key, value in info.config.hyperparameters.items():
                hp_node.add(f"[dim]{key}:[/dim] {value}")

        if info.config.data:
            data_node = config_node.add("[bold]Data[/bold]")
            for key, value in info.config.data.items():
                if value:
                    data_node.add(f"[dim]{key}:[/dim] {value}")

    # Files
    files_node = tree.add("[bold]Files[/bold]")
    for f in sorted(info.files, key=lambda x: x.name):
        size = format_file_size(f.stat().st_size)
        modified = format_timestamp(f.stat().st_mtime)
        files_node.add(f"[dim]{f.name}[/dim] ({size}, {modified})")

    console.print(Panel(tree, border_style="magenta"))


@checkpoints_app.command("create")
def checkpoints_create(
    name: str = typer.Argument(..., help="Checkpoint name"),
    model_type: str = typer.Option("lejepa", "--type", "-t", help="Model type (lejepa, entry-3class, entry-5class, exit)"),
    embedding_dim: int = typer.Option(256, "--embedding-dim", help="Embedding dimension"),
    num_layers: int = typer.Option(4, "--num-layers", help="Number of transformer layers"),
    num_heads: int = typer.Option(8, "--num-heads", help="Number of attention heads"),
    patch_length: int = typer.Option(32, "--patch-length", help="Patch length"),
    underlying: str = typer.Option("SPY", "--underlying", help="Underlying symbol"),
    start_date: Optional[str] = typer.Option(None, "--start-date", help="Training start date"),
    end_date: Optional[str] = typer.Option(None, "--end-date", help="Training end date"),
    epochs: int = typer.Option(10, "--epochs", help="Number of training epochs"),
    batch_size: int = typer.Option(2048, "--batch-size", help="Batch size"),
    lr: float = typer.Option(1e-4, "--lr", help="Learning rate"),
    lambda_sigreg: float = typer.Option(0.1, "--lambda-sigreg", help="SIGReg regularization weight"),
    # Entry policy specific options
    lejepa_checkpoint: Optional[str] = typer.Option(None, "--lejepa-checkpoint", help="Path to LeJEPA checkpoint (required for entry/exit)"),
    lookahead: int = typer.Option(15, "--lookahead", help="Minutes to look ahead for ROI labeling"),
    min_roi_threshold: float = typer.Option(20.0, "--min-roi-threshold", help="Minimum ROI %% for trade signal"),
    otm_buffer: float = typer.Option(1.2, "--otm-buffer", help="OTM must beat ATM by this factor"),
    focal_alpha_hold: float = typer.Option(0.35, "--focal-alpha-hold", help="Focal loss alpha for HOLD"),
    focal_alpha_atm: float = typer.Option(1.0, "--focal-alpha-atm", help="Focal loss alpha for ATM classes"),
    focal_alpha_otm: float = typer.Option(1.2, "--focal-alpha-otm", help="Focal loss alpha for OTM classes"),
    slippage_pct: float = typer.Option(0.5, "--slippage-pct", help="Slippage as %% of option price"),
    execution_delay: int = typer.Option(1, "--execution-delay", help="Execution delay in minutes"),
    # Partial patches for early morning data
    include_partial_patches: bool = typer.Option(False, "--include-partial-patches", help="Include partial patches for early morning data (9:30-10:00am simulation)"),
    min_partial_length: int = typer.Option(8, "--min-partial-length", help="Minimum candles for partial patches"),
):
    """Create a new checkpoint configuration."""
    # Validate entry/exit policies require lejepa checkpoint
    if model_type in ["entry-3class", "entry-5class", "exit"] and not lejepa_checkpoint:
        console.print(f"[red]--lejepa-checkpoint is required for {model_type} model type[/red]")
        raise typer.Exit(1)

    if checkpoint_manager.exists(name):
        console.print(f"[yellow]Checkpoint '{name}' already exists.[/yellow]")
        overwrite = typer.confirm("Overwrite configuration?")
        if not overwrite:
            raise typer.Exit(0)

    config = checkpoint_manager.create(
        name=name,
        model_type=model_type,
        embedding_dim=embedding_dim,
        num_layers=num_layers,
        num_heads=num_heads,
        patch_length=patch_length,
        underlying=underlying,
        start_date=start_date,
        end_date=end_date,
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        lambda_sigreg=lambda_sigreg,
        # Entry policy specific
        lejepa_checkpoint=lejepa_checkpoint,
        lookahead=lookahead,
        min_roi_threshold=min_roi_threshold,
        otm_buffer=otm_buffer,
        focal_alpha_hold=focal_alpha_hold,
        focal_alpha_atm=focal_alpha_atm,
        focal_alpha_otm=focal_alpha_otm,
        slippage_pct=slippage_pct,
        execution_delay=execution_delay,
        # Partial patches
        include_partial_patches=include_partial_patches,
        min_partial_length=min_partial_length,
    )

    # Show summary
    table = Table(title=f"Checkpoint: {name}", box=box.ROUNDED, show_header=False)
    table.add_column("Parameter", style="dim")
    table.add_column("Value", style="cyan")

    table.add_row("Model Type", model_type)
    table.add_row("Embedding Dim", str(embedding_dim))
    table.add_row("Layers", str(num_layers))
    table.add_row("Heads", str(num_heads))
    table.add_row("Patch Length", str(patch_length))
    table.add_row("Epochs", str(epochs))
    table.add_row("Batch Size", str(batch_size))
    table.add_row("Learning Rate", str(lr))
    table.add_row("Data", f"{underlying} ({start_date or 'all'} to {end_date or 'all'})")

    # Show entry-specific params if applicable
    if model_type in ["entry-3class", "entry-5class", "exit"]:
        table.add_row("LeJEPA Checkpoint", lejepa_checkpoint or "N/A")
    if model_type == "entry-5class":
        table.add_row("Lookahead", f"{lookahead} min")
        table.add_row("Min ROI Threshold", f"{min_roi_threshold}%")
        table.add_row("OTM Buffer", f"{otm_buffer}x")
        table.add_row("Slippage", f"{slippage_pct}%")

    console.print(table)
    console.print(f"\n[green]Checkpoint '{name}' created. Run training with:[/green]")
    console.print(f"[cyan]python manage.py checkpoints train {name}[/cyan]")


@checkpoints_app.command("train")
def checkpoints_train(
    name: str = typer.Argument(..., help="Checkpoint name"),
    resume: bool = typer.Option(False, "--resume", "-r", help="Resume from latest checkpoint"),
    background: bool = typer.Option(False, "--background", "-b", help="Run in background"),
):
    """Start training a checkpoint."""
    info = checkpoint_manager.get(name)

    if not info:
        console.print(f"[red]Checkpoint '{name}' not found. Create it first.[/red]")
        raise typer.Exit(1)

    if not info.config:
        console.print(f"[red]No config.json found in checkpoint '{name}'.[/red]")
        raise typer.Exit(1)

    cmd = checkpoint_manager.build_train_command(name, resume)

    console.print(Panel(
        f"[cyan]Starting training for: {name}[/cyan]\n\n"
        f"[dim]Command:[/dim] {' '.join(cmd)}",
        title="Training",
        border_style="cyan",
    ))

    if background:
        process = checkpoint_manager.train(name, resume, background=True)
        if process:
            console.print(f"[green]Training started in background (PID: {process.pid})[/green]")
            console.print(f"[dim]Check logs at: checkpoints/{name}/training.log[/dim]")
    else:
        try:
            checkpoint_manager.train(name, resume, background=False)
        except KeyboardInterrupt:
            console.print("\n[yellow]Training interrupted.[/yellow]")


@checkpoints_app.command("remove")
def checkpoints_remove(
    name: str = typer.Argument(..., help="Checkpoint name"),
    force: bool = typer.Option(False, "--force", "-f", help="Skip confirmation"),
):
    """Remove a checkpoint."""
    if not checkpoint_manager.exists(name):
        console.print(f"[red]Checkpoint '{name}' not found.[/red]")
        raise typer.Exit(1)

    if not force:
        confirm = typer.confirm(f"Are you sure you want to remove checkpoint '{name}'?")
        if not confirm:
            console.print("[yellow]Cancelled.[/yellow]")
            raise typer.Exit(0)

    checkpoint_manager.remove(name)
    console.print(f"[green]Checkpoint '{name}' removed.[/green]")


@checkpoints_app.command("edit")
def checkpoints_edit(
    name: str = typer.Argument(..., help="Checkpoint name"),
    model_type: Optional[str] = typer.Option(None, "--type", "-t", help="Model type (lejepa, entry, exit)"),
    embedding_dim: Optional[int] = typer.Option(None, "--embedding-dim", help="Embedding dimension"),
    num_layers: Optional[int] = typer.Option(None, "--num-layers", help="Number of transformer layers"),
    num_heads: Optional[int] = typer.Option(None, "--num-heads", help="Number of attention heads"),
    patch_length: Optional[int] = typer.Option(None, "--patch-length", help="Patch length"),
    underlying: Optional[str] = typer.Option(None, "--underlying", help="Underlying symbol"),
    start_date: Optional[str] = typer.Option(None, "--start-date", help="Training start date"),
    end_date: Optional[str] = typer.Option(None, "--end-date", help="Training end date"),
    epochs: Optional[int] = typer.Option(None, "--epochs", help="Number of training epochs"),
    batch_size: Optional[int] = typer.Option(None, "--batch-size", help="Batch size"),
    lr: Optional[float] = typer.Option(None, "--lr", help="Learning rate"),
    lambda_sigreg: Optional[float] = typer.Option(None, "--lambda-sigreg", help="SIGReg regularization weight"),
    editor: bool = typer.Option(False, "--editor", "-e", help="Open config in $EDITOR"),
):
    """Edit a checkpoint configuration.

    Examples:
        python manage.py checkpoints edit my-checkpoint --epochs 20
        python manage.py checkpoints edit my-checkpoint --start-date 2022-01-01 --end-date 2024-12-31
        python manage.py checkpoints edit my-checkpoint --editor  # Open in $EDITOR
    """
    import os
    import subprocess

    if not checkpoint_manager.exists(name):
        console.print(f"[red]Checkpoint '{name}' not found.[/red]")
        raise typer.Exit(1)

    if editor:
        # Open in editor
        config_path = checkpoint_manager.get_config_path(name)
        if not config_path:
            console.print(f"[red]No config.json found for checkpoint '{name}'.[/red]")
            raise typer.Exit(1)

        editor_cmd = os.environ.get("EDITOR", "vim")
        subprocess.run([editor_cmd, str(config_path)])
        console.print(f"[green]Config file saved.[/green]")
        return

    # Check if any options were provided
    has_updates = any([
        model_type, embedding_dim, num_layers, num_heads, patch_length,
        underlying, start_date, end_date, epochs, batch_size, lr, lambda_sigreg
    ])

    if not has_updates:
        console.print("[yellow]No changes specified. Use --editor to open in editor, or specify options to update.[/yellow]")
        console.print("[dim]Example: python manage.py checkpoints edit my-checkpoint --epochs 20[/dim]")
        raise typer.Exit(0)

    config = checkpoint_manager.update(
        name=name,
        model_type=model_type,
        embedding_dim=embedding_dim,
        num_layers=num_layers,
        num_heads=num_heads,
        patch_length=patch_length,
        underlying=underlying,
        start_date=start_date,
        end_date=end_date,
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        lambda_sigreg=lambda_sigreg,
    )

    if config:
        console.print(f"[green]Checkpoint '{name}' updated successfully![/green]")
        # Show updated config
        checkpoints_get(name, show_status=False)
    else:
        console.print(f"[red]Failed to update checkpoint '{name}'.[/red]")
        raise typer.Exit(1)


# =============================================================================
# CACHE Commands
# =============================================================================


@cache_app.command("list")
def cache_list():
    """List cached data files."""
    stats = cache_manager.get_all_stats()

    table = Table(
        title="Data Cache",
        box=box.ROUNDED,
        show_header=True,
        header_style="bold yellow",
    )
    table.add_column("Type")
    table.add_column("Files", justify="right")
    table.add_column("Date Range")
    table.add_column("Total Size", justify="right")

    for stat in stats:
        if stat.file_count > 0:
            date_range = f"{stat.date_range[0]} to {stat.date_range[1]}" if stat.date_range[0] else "-"
            table.add_row(
                stat.cache_type.capitalize(),
                str(stat.file_count),
                date_range,
                format_file_size(stat.total_size),
            )

    if all(s.file_count == 0 for s in stats):
        console.print(Panel(
            "[dim]No cached data found. Build cache with:[/dim]\n"
            "[cyan]python manage.py cache build[/cyan]",
            title="Cache",
            border_style="dim",
        ))
        return

    console.print(table)


@cache_app.command("clear")
def cache_clear(
    target: str = typer.Argument("all", help="Target: all, normalized, raw, or a specific date (YYYY-MM-DD)"),
    force: bool = typer.Option(False, "--force", "-f", help="Skip confirmation"),
):
    """Clear the data cache.

    Examples:
        python manage.py cache clear all          # Clear all cache
        python manage.py cache clear normalized   # Clear only normalized cache
        python manage.py cache clear raw          # Clear only raw cache
        python manage.py cache clear 2024-01-15   # Clear specific date
    """
    # Check if target is a date
    is_date = len(target) == 10 and target[4] == "-" and target[7] == "-"

    if not is_date and target not in ["all", "normalized", "raw"]:
        console.print(f"[red]Invalid target: {target}[/red]")
        console.print("[dim]Valid options: all, normalized, raw, or a date (YYYY-MM-DD)[/dim]")
        raise typer.Exit(1)

    if not force:
        if is_date:
            msg = f"Are you sure you want to clear cache for {target}?"
        else:
            msg = f"Are you sure you want to clear the {target} cache?"
        confirm = typer.confirm(msg)
        if not confirm:
            console.print("[yellow]Cancelled.[/yellow]")
            raise typer.Exit(0)

    if is_date:
        cleared = cache_manager.clear_date(target)
    else:
        cleared = cache_manager.clear(cache_type=target)

    console.print(f"[green]Cleared {cleared} files from cache.[/green]")


@cache_app.command("build")
def cache_build(
    start_date: Optional[str] = typer.Option(None, "--start-date", "-s", help="Start date"),
    end_date: Optional[str] = typer.Option(None, "--end-date", "-e", help="End date"),
    workers: int = typer.Option(8, "--workers", "-w", help="Number of workers"),
    background: bool = typer.Option(False, "--background", "-b", help="Run in background"),
):
    """Build/rebuild the data cache."""
    cmd = cache_manager.build_command(start_date, end_date, workers)

    console.print(Panel(
        f"[cyan]Building cache...[/cyan]\n"
        f"[dim]Date range:[/dim] {start_date or 'all'} to {end_date or 'all'}\n"
        f"[dim]Workers:[/dim] {workers}\n"
        f"[dim]Command:[/dim] {' '.join(cmd)}",
        title="Cache Build",
        border_style="cyan",
    ))

    if background:
        process = cache_manager.build(start_date, end_date, workers, background=True)
        if process:
            console.print(f"[green]Cache build started in background (PID: {process.pid})[/green]")
    else:
        cache_manager.build(start_date, end_date, workers, background=False)


@cache_app.command("check")
def cache_check(
    underlying: str = typer.Option("SPY", "--underlying", "-u", help="Underlying symbol"),
    start_date: str = typer.Option(..., "--start-date", "-s", help="Start date (YYYY-MM-DD)"),
    end_date: str = typer.Option(..., "--end-date", "-e", help="End date (YYYY-MM-DD)"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Show detailed output"),
):
    """Validate training cache files for completeness.

    Checks training-1m-normalized files for:
    - Missing trading days in date range
    - Row count (390 for regular days, ~210 for half days)

    Examples:
        python manage.py cache check --underlying SPY --start-date 2024-01-01 --end-date 2024-12-31
        python manage.py cache check -u SPY -s 2024-06-01 -e 2024-06-30 --verbose
    """
    from datetime import datetime as dt

    loader = _get_dag_loader()
    start = dt.strptime(start_date, "%Y-%m-%d").date()
    end = dt.strptime(end_date, "%Y-%m-%d").date()

    # Check training-1m-normalized (the main training cache)
    dataset = "training-1m-normalized"
    cached_dates = loader.get_cached_dates(dataset, underlying)
    dates_in_range = [d for d in cached_dates if start <= d <= end]

    console.print(f"[dim]Checking {len(dates_in_range)} {dataset} files...[/dim]\n")

    ok_count = 0
    short_count = 0
    error_count = 0

    for d in sorted(dates_in_range):
        try:
            df = loader.load_single_file(dataset, underlying, d)
            if df is None:
                error_count += 1
                if verbose:
                    console.print(f"  {d}: [red]LOAD ERROR[/red]")
                continue

            row_count = len(df)
            # Regular day = 390 bars, half day ~210
            if row_count < 200:
                short_count += 1
                if verbose:
                    console.print(f"  {d}: [yellow]SHORT ({row_count} rows)[/yellow]")
            else:
                ok_count += 1
                if verbose:
                    console.print(f"  {d}: [green]OK[/green] ({row_count} rows)")

        except Exception as e:
            error_count += 1
            if verbose:
                console.print(f"  {d}: [red]ERROR - {e}[/red]")

    console.print(f"\n[bold]Summary:[/bold]")
    console.print(f"  [green]OK:[/green] {ok_count}")
    console.print(f"  [yellow]Short:[/yellow] {short_count}")
    console.print(f"  [red]Errors:[/red] {error_count}")


@cache_app.command("stats")
def cache_stats():
    """Show detailed cache statistics."""
    summary = cache_manager.get_storage_summary()

    tree = Tree("[bold yellow]Cache Statistics[/bold yellow]")

    # Normalized cache
    norm = summary["normalized"]
    norm_node = tree.add("[bold]Normalized Cache[/bold]")
    norm_node.add(f"[dim]Files:[/dim] {norm['files']}")
    norm_node.add(f"[dim]Size:[/dim] {format_file_size(norm['size_bytes'])}")
    if norm['date_range'][0]:
        norm_node.add(f"[dim]Date range:[/dim] {norm['date_range'][0]} to {norm['date_range'][1]}")

    # Raw cache
    raw = summary["raw"]
    raw_node = tree.add("[bold]Raw Cache[/bold]")
    raw_node.add(f"[dim]Files:[/dim] {raw['files']}")
    raw_node.add(f"[dim]Size:[/dim] {format_file_size(raw['size_bytes'])}")
    if raw['date_range'][0]:
        raw_node.add(f"[dim]Date range:[/dim] {raw['date_range'][0]} to {raw['date_range'][1]}")

    # Total
    tree.add(f"[bold]Total Size:[/bold] {format_file_size(summary['total_size_bytes'])}")

    console.print(Panel(tree, border_style="yellow"))


# =============================================================================
# Root Commands
# =============================================================================


@app.command("status")
def status():
    """Show overall system status."""
    console.print(Panel(
        "[bold cyan]SpyMaster Trading System[/bold cyan]",
        border_style="cyan",
    ))

    # Get data from managers
    checkpoints = checkpoint_manager.list_all()
    strategies = strategy_manager.list_all()
    cache_summary = cache_manager.get_storage_summary()

    # Count trained checkpoints
    trained = sum(1 for cp in checkpoints if cp.status == "Trained")

    # Data counts
    stocks_dir = DATA_DIR / "stocks"
    options_dir = DATA_DIR / "options"
    stocks_count = len(list(stocks_dir.glob("*.parquet"))) if stocks_dir.exists() else 0
    options_count = len(list(options_dir.glob("*.parquet"))) if options_dir.exists() else 0

    table = Table(box=box.SIMPLE, show_header=False)
    table.add_column("Category", style="bold")
    table.add_column("Count", justify="right", style="cyan")
    table.add_column("Details", style="dim")

    table.add_row("Checkpoints", str(len(checkpoints)), f"{trained} trained")
    table.add_row("Strategies", str(len(strategies)), "")
    table.add_row("Cache Files", str(cache_summary["normalized"]["files"]), "normalized days")
    table.add_row("Cache Size", format_file_size(cache_summary["total_size_bytes"]), "")
    table.add_row("Raw Data", str(stocks_count), f"stocks, {options_count} options")

    console.print(table)

    console.print("\n[dim]Commands:[/dim]")
    console.print("  [cyan]python manage.py checkpoints list[/cyan]")
    console.print("  [cyan]python manage.py strategies list[/cyan]")
    console.print("  [cyan]python manage.py cache list[/cyan]")


@app.command("version")
def version():
    """Show version information."""
    console.print("[bold cyan]SpyMaster[/bold cyan] v1.0.0")
    console.print("[dim]LeJEPA-based 0DTE Options Trading System[/dim]")


@app.command("serve")
def serve(
    port: int = typer.Option(8050, "--port", "-p", help="Port to run the web server on"),
    host: str = typer.Option("0.0.0.0", "--host", "-h", help="Host to bind to"),
    debug: bool = typer.Option(True, "--debug/--no-debug", help="Enable debug mode"),
):
    """Start the web UI server.

    Examples:
        python manage.py serve
        python manage.py serve --port 8080
        python manage.py serve --host 127.0.0.1 --port 5000 --no-debug
    """
    import sys
    sys.path.insert(0, str(PROJECT_ROOT))

    from webui.app import app as flask_app

    console.print(Panel(
        f"[bold cyan]SpyMaster Web UI[/bold cyan]\n\n"
        f"[dim]Host:[/dim] {host}\n"
        f"[dim]Port:[/dim] {port}\n"
        f"[dim]Debug:[/dim] {debug}\n\n"
        f"Open [link=http://localhost:{port}]http://localhost:{port}[/link] in your browser",
        title="Web Server",
        border_style="cyan",
    ))

    flask_app.run(host=host, port=port, debug=debug)


# =============================================================================

if __name__ == "__main__":
    app()
