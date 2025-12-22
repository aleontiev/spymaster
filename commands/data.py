"""
Data commands for SpyMaster CLI.

Commands for downloading, loading, and managing raw and computed datasets.
"""

from typing import Optional

import typer
from rich import box
from rich.table import Table

from commands import (
    console,
    format_file_size,
    _get_dag_loader,
    DATA_DIR,
    PROJECT_ROOT,
)

# Create the typer app for data commands
data_app = typer.Typer(help="Download and manage raw data", invoke_without_command=True)


@data_app.callback()
def data_callback(ctx: typer.Context):
    """Download and manage raw data. Run without subcommand to show status."""
    if ctx.invoked_subcommand is None:
        data_status()


@data_app.command("status")
def data_status(
    underlying: str = typer.Option("SPY", "--underlying", "-u", help="Underlying symbol"),
):
    """Show data status using the new DAG-based hierarchical structure."""
    loader = _get_dag_loader()
    registry = loader.registry

    # Get all datasets from registry, ordered by dependency depth
    all_datasets = registry.list_datasets()

    table = Table(
        title=f"Data Status ({underlying})",
        box=box.ROUNDED,
        show_header=True,
        header_style="bold blue",
    )
    table.add_column("Dataset")
    table.add_column("Type")
    table.add_column("Files", justify="right")
    table.add_column("Date Range")
    table.add_column("Size", justify="right")

    for dataset_id in all_datasets:
        config = registry.get(dataset_id)
        display_name = registry.get_display_name(dataset_id)
        type_label = f"[cyan]{config.type}[/cyan]" if config.type == "source" else f"[yellow]{config.type}[/yellow]"
        try:
            cached_dates = loader.get_cached_dates(dataset_id, underlying)
            if cached_dates:
                total_size = 0
                for d in cached_dates:
                    path = loader.resolve_path(dataset_id, underlying, d)
                    if path.exists():
                        total_size += path.stat().st_size

                date_range = f"{min(cached_dates)} to {max(cached_dates)}"
                table.add_row(
                    f"{dataset_id} ({display_name})",
                    type_label,
                    str(len(cached_dates)),
                    date_range,
                    format_file_size(total_size),
                )
            else:
                table.add_row(f"{dataset_id} ({display_name})", type_label, "0", "-", "-")
        except Exception as e:
            table.add_row(f"{dataset_id} ({display_name})", type_label, "[red]ERROR[/red]", str(e)[:30], "-")

    console.print(table)


@data_app.command("download")
def data_download(
    data_type: str = typer.Argument(..., help="Type: stocks or options"),
    year: Optional[int] = typer.Option(None, "--year", "-y", help="Single year to download"),
    from_year: Optional[int] = typer.Option(None, "--from-year", help="Start year"),
    to_year: Optional[int] = typer.Option(None, "--to-year", help="End year"),
    month: Optional[int] = typer.Option(None, "--month", "-m", help="Specific month (1-12)"),
    workers: int = typer.Option(4, "--workers", "-w", help="Number of parallel workers"),
    skip_existing: bool = typer.Option(True, "--skip-existing/--no-skip-existing", help="Skip existing files"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Show what would be downloaded"),
):
    """Download raw data from Polygon S3 (stocks/options).

    For GEX flow data, use: python manage.py data load gex-flow

    Examples:
        python manage.py data download stocks --from-year 2020 --to-year 2024
        python manage.py data download options --year 2024
        python manage.py data download options --year 2024 --month 11
    """
    from src.data.providers.polygon import PolygonProvider

    if data_type not in ("stocks", "options"):
        console.print(f"[red]Invalid data type: {data_type}[/red]")
        console.print("[dim]Valid types: stocks, options[/dim]")
        console.print("[dim]For GEX flow, use: python manage.py data load gex-flow[/dim]")
        raise typer.Exit(1)

    # Determine year range
    if year:
        years = [year]
    elif from_year or to_year:
        start_year = from_year or 2020
        end_year = to_year or 2024
        years = list(range(start_year, end_year + 1))
    else:
        console.print("[red]Must specify --year or --from-year/--to-year[/red]")
        raise typer.Exit(1)

    months = [month] if month else list(range(1, 13))

    loader = _get_dag_loader()
    provider = PolygonProvider()
    dataset = f"{data_type}-1m"
    underlying = "SPY"

    from datetime import date
    from rich.progress import Progress, SpinnerColumn, TextColumn

    files_to_download = []
    for yr in years:
        for mo in months:
            try:
                first_day = date(yr, mo, 1)
                path = loader.resolve_path(dataset, underlying, first_day)
                month_dir = path.parent

                if skip_existing and month_dir.exists() and any(month_dir.glob("*.parquet")):
                    continue
                files_to_download.append((yr, mo))
            except ValueError:
                pass

    if not files_to_download:
        console.print("[green]All requested data already downloaded.[/green]")
        return

    if dry_run:
        console.print(f"[dim]Would download {len(files_to_download)} months of {data_type} data[/dim]")
        for yr, mo in files_to_download:
            console.print(f"  {yr}-{mo:02d}")
        return

    console.print(f"[bold]Downloading {len(files_to_download)} months of {data_type} data...[/bold]")

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("Downloading...", total=len(files_to_download))

        for yr, mo in files_to_download:
            progress.update(task, description=f"Downloading {data_type} {yr}-{mo:02d}...")
            try:
                first_day = date(yr, mo, 1)
                if data_type == "stocks":
                    provider.download_stocks_month(yr, mo, "SPY")
                else:
                    provider.download_options_month(yr, mo, "SPY")
            except Exception as e:
                console.print(f"[red]Error downloading {yr}-{mo:02d}: {e}[/red]")
            progress.advance(task)

    console.print("[green]Download complete![/green]")


@data_app.command("load")
def data_load(
    dataset: str = typer.Argument(..., help="Dataset to load (e.g., gex-flow-1m, training-raw)"),
    start_date: str = typer.Option(..., "--start-date", "-s", help="Start date (YYYY-MM-DD)"),
    end_date: Optional[str] = typer.Option(None, "--end-date", "-e", help="End date (YYYY-MM-DD)"),
    underlying: str = typer.Option("SPY", "--underlying", "-u", help="Underlying symbol"),
    force: bool = typer.Option(False, "--force", "-f", help="Force reload even if cached (ignore cache)"),
    rebuild: bool = typer.Option(False, "--rebuild", "-r", help="Delete existing files first, then recompute from scratch"),
    fix_zeros: bool = typer.Option(False, "--fix-zeros", help="Only process files with all-zero values"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Show what would be processed"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Show detailed progress"),
    concurrent: int = typer.Option(8, "--concurrent", "-c", help="Number of concurrent ThetaData requests"),
):
    """Load or compute a dataset.

    Datasets are organized in a DAG (Directed Acyclic Graph) with automatic dependency resolution.
    Source datasets (like trade-quote-1m) are fetched from ThetaData.
    Computed datasets (like gex-flow-1m, training-raw) are derived from source data.

    Examples:
        python manage.py data load trade-quote-1m --start-date 2024-12-15
        python manage.py data load gex-flow-1m --start-date 2024-12-01 --end-date 2024-12-15
        python manage.py data load training-raw --start-date 2024-01-01 --rebuild  # Delete + recompute
    """
    import asyncio
    import pandas as pd
    import numpy as np
    from datetime import datetime
    from src.data.dag import DAGLoader

    loader = _get_dag_loader()
    registry = loader.registry

    # Validate dataset exists in registry
    dataset_id = dataset
    if not registry.exists(dataset_id):
        console.print(f"[red]Unknown dataset: {dataset}[/red]")
        all_datasets = registry.list_datasets()
        console.print(f"[dim]Valid datasets: {', '.join(all_datasets)}[/dim]")
        raise typer.Exit(1)

    # Parse dates
    start = datetime.strptime(start_date, "%Y-%m-%d").date()
    end = datetime.strptime(end_date, "%Y-%m-%d").date() if end_date else start

    # Get trading days from market calendar
    from src.db.database import get_db
    db = get_db()
    calendar_dates = db.get_trading_days(start.isoformat(), end.isoformat())

    if calendar_dates:
        available_dates = sorted([datetime.strptime(d, "%Y-%m-%d").date() for d in calendar_dates])
        console.print(f"[dim]Found {len(available_dates)} trading days from market calendar[/dim]")
    else:
        from datetime import timedelta
        available_dates = []
        current = start
        while current <= end:
            if current.weekday() < 5:
                available_dates.append(current)
            current += timedelta(days=1)
        console.print(f"[dim]Using {len(available_dates)} weekdays (no calendar data)[/dim]")

    if not available_dates:
        console.print(f"[yellow]No dates in range {start} to {end}[/yellow]")
        raise typer.Exit(0)

    # Determine which dates need processing
    dates_to_process = []
    cached_dates = set(loader.get_cached_dates(dataset_id, underlying))

    if fix_zeros and dataset_id == "gex-flow-1m":
        console.print("[dim]Scanning for files with zero values...[/dim]")
        for d in available_dates:
            if d in cached_dates:
                try:
                    df = loader.load_single_file(dataset_id, underlying, d)
                    if df is not None and "net_gamma_flow" in df.columns:
                        nonzero = (df["net_gamma_flow"] != 0).sum()
                        if nonzero == 0:
                            dates_to_process.append(d)
                except:
                    dates_to_process.append(d)
        console.print(f"[dim]Found {len(dates_to_process)} files with zero values[/dim]")

    elif rebuild:
        dates_to_process = available_dates
        if not dry_run:
            deleted_count = 0
            for d in dates_to_process:
                if d in cached_dates:
                    loader.delete_file(dataset_id, underlying, d)
                    deleted_count += 1
            if deleted_count > 0:
                console.print(f"[dim]Deleted {deleted_count} existing files for rebuild[/dim]")

    elif force:
        dates_to_process = available_dates
    else:
        import datetime as dt
        today = dt.date.today()
        for d in available_dates:
            if d not in cached_dates:
                dates_to_process.append(d)
            elif d == today:
                dates_to_process.append(d)

    if not dates_to_process:
        console.print(f"[green]All dates already cached for {dataset_id}[/green]")
        raise typer.Exit(0)

    console.print(f"[bold]Processing {len(dates_to_process)} dates for {dataset_id}[/bold]")

    if dry_run:
        console.print("[dim]Dates to process:[/dim]")
        for d in dates_to_process[:20]:
            console.print(f"  {d}")
        if len(dates_to_process) > 20:
            console.print(f"  ... and {len(dates_to_process) - 20} more")
        raise typer.Exit(0)

    # Get dataset config to determine processing strategy
    dataset_config = registry.get(dataset_id)
    is_computed = dataset_config.type == "computed"
    is_thetadata_source = dataset_config.type == "source" and dataset_config.provider == "thetadata"

    # Process based on dataset type
    if is_thetadata_source and dataset_id == "trade-quote-1m":
        # ThetaData source - requires async client
        async def process_thetadata():
            from src.data.thetadata_client import ThetaDataClient
            from src.data.gex_flow_engine import GEXFlowEngine

            client = ThetaDataClient()
            engine = GEXFlowEngine()

            console.print("[dim]Checking ThetaData terminal...[/dim]")
            if not await client._check_terminal_running():
                console.print("[red]ThetaData terminal not running! Start with: java -jar ThetaTerminalv3.jar[/red]")
                return

            console.print("[green]Terminal running. Starting processing...[/green]\n")

            success, fail = 0, 0
            for i, d in enumerate(dates_to_process, 1):
                console.print(f"[{i}/{len(dates_to_process)}] Processing {d}...")

                try:
                    stocks_df = loader.load_single_file("stocks-1m", underlying, d)
                    if stocks_df is None or stocks_df.empty:
                        console.print(f"  [yellow]No stock data, skipping[/yellow]")
                        fail += 1
                        continue

                    if "timestamp" not in stocks_df.columns and "window_start" in stocks_df.columns:
                        stocks_df = stocks_df.rename(columns={"window_start": "timestamp"})
                    if "timestamp" in stocks_df.columns:
                        stocks_df["timestamp"] = pd.to_datetime(stocks_df["timestamp"])

                    oi_df = loader.load_single_file("oi-day", underlying, d)
                    if oi_df is None or oi_df.empty:
                        console.print("  [dim]Fetching OI...[/dim]")
                        oi_df = await client.fetch_open_interest_for_date(d, underlying, zero_dte=True)
                        if not oi_df.empty:
                            loader.save_to_cache(oi_df, "oi-day", underlying, d)

                    greeks_df = loader.load_single_file("greeks-1m", underlying, d)
                    if greeks_df is None or greeks_df.empty:
                        console.print("  [dim]Fetching Greeks...[/dim]")
                        greeks_df = await client.fetch_greeks_for_date(d, underlying, interval="1m", zero_dte=True)
                        if not greeks_df.empty:
                            loader.save_to_cache(greeks_df, "greeks-1m", underlying, d)

                    if greeks_df is None or greeks_df.empty:
                        console.print(f"  [yellow]No Greeks, skipping[/yellow]")
                        fail += 1
                        continue

                    trade_agg_df = loader.load_single_file("trade-quote-1m", underlying, d)
                    if trade_agg_df is None or trade_agg_df.empty:
                        console.print("  [dim]Fetching trade+quote data...[/dim]")
                        trade_quotes_df = await client.fetch_trade_quotes_parallel(
                            d, underlying, zero_dte=True, concurrent_requests=concurrent
                        )

                        trade_agg_df = pd.DataFrame()
                        if not trade_quotes_df.empty:
                            df = trade_quotes_df.copy()
                            df["sign"] = 0
                            price = df["price"].values
                            bid = df["bid"].values
                            ask = df["ask"].values
                            df.loc[price >= ask, "sign"] = -1
                            df.loc[price <= bid, "sign"] = 1

                            ts_col = "trade_timestamp" if "trade_timestamp" in df.columns else "timestamp"
                            df["minute"] = pd.to_datetime(df[ts_col]).dt.floor("min")
                            df["buy_volume"] = np.where(df["sign"] == -1, df["size"], 0)
                            df["sell_volume"] = np.where(df["sign"] == 1, df["size"], 0)
                            if "right" in df.columns:
                                df["right"] = df["right"].str.lower()

                            trade_agg_df = df.groupby(["minute", "strike", "right"]).agg({
                                "buy_volume": "sum",
                                "sell_volume": "sum",
                                "size": "count",
                            }).reset_index()
                            trade_agg_df = trade_agg_df.rename(columns={"minute": "timestamp", "size": "trade_count"})
                            trade_agg_df["ticker"] = underlying

                            loader.save_to_cache(trade_agg_df, "trade-quote-1m", underlying, d)

                    console.print("  [dim]Computing GEX flow features...[/dim]")
                    features_df = engine.compute_flow_features(
                        greeks_df, trade_agg_df, stocks_df,
                        daily_oi_df=oi_df if oi_df is not None else pd.DataFrame()
                    )

                    if features_df.empty:
                        console.print(f"  [yellow]No features computed[/yellow]")
                        fail += 1
                        continue

                    path = loader.save_to_cache(features_df, dataset_id, underlying, d)
                    console.print(f"  [green]Saved {len(features_df)} rows[/green]")
                    success += 1

                except Exception as e:
                    console.print(f"  [red]Error: {e}[/red]")
                    fail += 1

            console.print(f"\n[bold]Complete:[/bold] {success} succeeded, {fail} failed")

        asyncio.run(process_thetadata())

    elif is_computed:
        # Computed datasets - use DAG loader with full dependency resolution
        from src.data.dag.tree_display import (
            LiveTreeDisplay, TreeNode, NodeStatus,
            convert_resolution_node_to_tree_node, print_static_tree
        )

        success_count = 0
        fail_count = 0

        for d in dates_to_process:
            resolution_tree = loader.build_resolution_tree(dataset_id, underlying, d, force_recompute=force)
            tree_node = convert_resolution_node_to_tree_node(resolution_tree, force_recompute=force)

            from datetime import datetime as dt_module
            import pytz
            now_et = dt_module.now(pytz.timezone("US/Eastern"))
            current_minute = now_et.strftime("%H:%M")
            title = f"[blue]Loading [white]{dataset_id}[/white]: [green]{underlying}[/green] @ [green]{d}[/green] as of [yellow]{current_minute} ET[/yellow][/blue]"

            with LiveTreeDisplay(console) as display:
                display.set_root(tree_node, title=title)

                try:
                    def progress_callback(
                        dataset: str,
                        status: str,
                        progress: float = 0.0,
                        row_count: int = None,
                        error_msg: str = None,
                        status_message: str = None,
                    ):
                        status_map = {
                            "loading": NodeStatus.LOADING,
                            "computing": NodeStatus.COMPUTING,
                            "complete": NodeStatus.CACHED,
                            "error": NodeStatus.ERROR,
                            "cached": NodeStatus.CACHED,
                            "rate_limited": NodeStatus.RATE_LIMITED,
                        }
                        node_status = status_map.get(status, NodeStatus.LOADING)
                        if progress >= 1.0 and status in ("loading", "computing"):
                            node_status = NodeStatus.CACHED

                        status_msg = status_message
                        if status_msg is None:
                            if status == "loading":
                                status_msg = "Fetching..."
                            elif status == "computing":
                                status_msg = "Computing..."

                        display.update_node(
                            dataset=dataset,
                            date=d,
                            status=node_status,
                            progress=progress,
                            row_count=row_count,
                            error_msg=error_msg,
                            status_message=status_msg,
                        )

                    df = loader.load_day(
                        dataset_id, underlying, d,
                        force_recompute=force,
                        progress_callback=progress_callback
                    )

                    if df is not None and not df.empty:
                        # Update root node to complete
                        display.update_node(dataset_id, date=d, status=NodeStatus.COMPLETE, row_count=len(df))

                        # Update all child nodes to COMPLETE/CACHED status
                        # This ensures all dependencies show 100% when loading succeeds
                        def mark_children_complete(node):
                            for child in node.children:
                                if child.status not in (NodeStatus.ERROR,):
                                    # Get the base dataset name (strip any suffixes like " (3 days)")
                                    base_dataset = child.dataset.split(" (")[0]

                                    # Re-read the actual file to get accurate counts
                                    try:
                                        # Use the child's date if it has one, otherwise use the main date
                                        child_date = child.date if child.date else d
                                        child_path = loader.resolve_path(base_dataset, underlying, child_date)
                                        if child_path.exists():
                                            child_df = pd.read_parquet(child_path)
                                            from src.data.dag.loader import count_unique_minutes
                                            unique_mins = count_unique_minutes(child_df)
                                            display.update_node(
                                                base_dataset,
                                                date=child_date,
                                                status=NodeStatus.CACHED,
                                                row_count=len(child_df),
                                                unique_minutes=unique_mins,
                                                progress=1.0,
                                            )
                                        else:
                                            # File doesn't exist but we're marking as complete
                                            # (e.g., special nodes like prev_days_stocks)
                                            display.update_node(
                                                child.dataset,
                                                status=NodeStatus.CACHED,
                                                progress=1.0,
                                            )
                                    except Exception:
                                        # Just mark as complete without re-reading
                                        display.update_node(
                                            child.dataset,
                                            status=NodeStatus.CACHED,
                                            progress=1.0,
                                        )
                                mark_children_complete(child)

                        mark_children_complete(tree_node)
                        success_count += 1
                    else:
                        display.update_node(dataset_id, date=d, status=NodeStatus.ERROR, error_msg="No data")
                        fail_count += 1

                except Exception as e:
                    display.update_node(dataset_id, date=d, status=NodeStatus.ERROR, error_msg=str(e))
                    fail_count += 1

            console.print()

        console.print(f"\n[bold]Complete:[/bold] {success_count} succeeded, {fail_count} failed")

    else:
        console.print(f"[yellow]Direct processing for {dataset_id} not yet implemented[/yellow]")
        console.print("[dim]Use the cache build command for this dataset[/dim]")


@data_app.command("check")
def data_check(
    dataset: str = typer.Argument("gex-flow-1m", help="Dataset to check"),
    start_date: str = typer.Option(..., "--start-date", "-s", help="Start date (YYYY-MM-DD)"),
    end_date: Optional[str] = typer.Option(None, "--end-date", "-e", help="End date (YYYY-MM-DD)"),
    underlying: str = typer.Option("SPY", "--underlying", "-u", help="Underlying symbol"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Show all files"),
):
    """Check data quality for a dataset.

    Examples:
        python manage.py data check gex-flow-1m --start-date 2024-01-01 --end-date 2024-12-31
        python manage.py data check training-1m-normalized --start-date 2024-06-01 -v
    """
    from datetime import datetime
    import pandas as pd

    loader = _get_dag_loader()
    registry = loader.registry

    # Validate dataset exists in registry
    dataset_id = dataset
    if not registry.exists(dataset_id):
        console.print(f"[red]Unknown dataset: {dataset}[/red]")
        all_datasets = registry.list_datasets()
        console.print(f"[dim]Valid datasets: {', '.join(all_datasets)}[/dim]")
        raise typer.Exit(1)

    start = datetime.strptime(start_date, "%Y-%m-%d").date()
    end = datetime.strptime(end_date, "%Y-%m-%d").date() if end_date else start

    cached_dates = loader.get_cached_dates(dataset_id, underlying)
    dates_in_range = [d for d in cached_dates if start <= d <= end]

    console.print(f"[bold]Checking {dataset_id} for {underlying}[/bold]")
    console.print(f"[dim]Date range: {start} to {end}[/dim]")
    console.print(f"[dim]Found {len(dates_in_range)} cached files[/dim]\n")

    # Check columns based on dataset type
    config = registry.get(dataset_id)
    expected_cols = list(config.field_map.keys()) if config.field_map else []

    zero_dates = []
    partial_dates = []
    complete_dates = []

    for d in dates_in_range:
        df = loader.load_single_file(dataset_id, underlying, d)
        if df is None or df.empty:
            partial_dates.append((d, 0, "Empty"))
            continue

        rows = len(df)
        all_zeros = False

        if expected_cols:
            numeric_cols = [c for c in expected_cols if c in df.columns and df[c].dtype in ['float64', 'int64']]
            if numeric_cols:
                nonzero = (df[numeric_cols[0]] != 0).sum()
                if nonzero == 0:
                    all_zeros = True

        if all_zeros:
            zero_dates.append(d)
        elif rows < 300:
            partial_dates.append((d, rows, "Partial"))
        else:
            complete_dates.append((d, rows))

    console.print(f"[green]Complete:[/green] {len(complete_dates)} files")
    console.print(f"[yellow]Partial:[/yellow] {len(partial_dates)} files")
    console.print(f"[red]Zeros:[/red] {len(zero_dates)} files")

    if partial_dates and verbose:
        console.print(f"\n[yellow]Partial files:[/yellow]")
        for d, rows, reason in partial_dates:
            console.print(f"  {d}: {rows} rows ({reason})")

    if zero_dates and not verbose:
        console.print(f"\n[yellow]Files with zeros ({len(zero_dates)}):[/yellow]")
        for d in zero_dates[:10]:
            console.print(f"  {d}")
        if len(zero_dates) > 10:
            console.print(f"  ... and {len(zero_dates) - 10} more")
        console.print(f"\n[dim]Run with --force to reprocess these:[/dim]")
        console.print(f"[cyan]python manage.py data load {dataset} -s {start_date} -e {end_date or start_date} --fix-zeros[/cyan]")
