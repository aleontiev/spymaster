"""
Data commands for SpyMaster CLI.

Commands for downloading, loading, and managing raw and computed datasets.
"""

from typing import Optional

import pandas as pd
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
    use_alpaca: bool = typer.Option(False, "--alpaca", help="Use Alpaca instead of Polygon for stocks"),
):
    """Download raw data from Polygon S3 or Alpaca (stocks/options).

    For GEX flow data, use: python manage.py data load gex-flow

    Examples:
        python manage.py data download stocks --from-year 2020 --to-year 2024
        python manage.py data download stocks --from-year 2012 --to-year 2020 --alpaca
        python manage.py data download options --year 2024
        python manage.py data download options --year 2024 --month 11
    """
    from src.data.providers.polygon import PolygonProvider
    from src.data.providers.alpaca import AlpacaProvider

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

    # Validate provider choice
    if use_alpaca and data_type == "options":
        console.print("[red]Alpaca does not provide options data. Use Polygon for options.[/red]")
        raise typer.Exit(1)

    loader = _get_dag_loader()
    provider_name = "Alpaca" if use_alpaca else "Polygon"
    dataset = f"{data_type}-1m"
    underlying = "SPY"

    from datetime import date
    import calendar
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
        console.print(f"[dim]Would download {len(files_to_download)} months of {data_type} data from {provider_name}[/dim]")
        for yr, mo in files_to_download:
            console.print(f"  {yr}-{mo:02d}")
        return

    console.print(f"[bold]Downloading {len(files_to_download)} months of {data_type} data from {provider_name}...[/bold]")

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("Downloading...", total=len(files_to_download))

        for yr, mo in files_to_download:
            progress.update(task, description=f"Downloading {data_type} {yr}-{mo:02d}...")
            try:
                if use_alpaca:
                    # Use Alpaca for stocks
                    alpaca_provider = AlpacaProvider()
                    _, last_day = calendar.monthrange(yr, mo)
                    start_date = date(yr, mo, 1)
                    end_date = date(yr, mo, last_day)
                    df = alpaca_provider.fetch_bars_range(underlying, start_date, end_date, feed="sip")
                else:
                    # Use Polygon S3 for stocks/options
                    polygon_provider = PolygonProvider()
                    df = polygon_provider.fetch_month(data_type, yr, mo, underlying)

                if df.empty:
                    console.print(f"[yellow]No data for {yr}-{mo:02d}[/yellow]")
                else:
                    # Save each day separately
                    if "window_start" in df.columns:
                        df["date"] = df["window_start"].dt.date
                    elif "timestamp" in df.columns:
                        df["date"] = pd.to_datetime(df["timestamp"]).dt.date

                    days_saved = 0
                    for day_date, day_df in df.groupby("date"):
                        day_df = day_df.drop(columns=["date"])
                        path = loader.resolve_path(dataset, underlying, day_date)
                        path.parent.mkdir(parents=True, exist_ok=True)
                        day_df.to_parquet(path, index=False)
                        days_saved += 1

                    console.print(f"[green]Saved {len(df)} rows ({days_saved} days) for {yr}-{mo:02d}[/green]")
            except Exception as e:
                console.print(f"[red]Error downloading {yr}-{mo:02d}: {e}[/red]")
            progress.advance(task)

    console.print("[green]Download complete![/green]")


@data_app.command("load")
def data_load(
    dataset: str = typer.Argument(..., help="Dataset to load (e.g., options-flow-1m, training-raw)"),
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
    Source datasets (like option-trades-1m) are fetched from ThetaData.
    Computed datasets (like options-flow-1m, training-raw) are derived from source data.

    Examples:
        python manage.py data load option-trades-1m --start-date 2024-12-15
        python manage.py data load options-flow-1m --start-date 2024-12-01 --end-date 2024-12-15
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

    if fix_zeros and dataset_id == "options-flow-1m":
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
        # Note: files are deleted just before each date is rebuilt (not upfront)
        # This preserves data if the job is cancelled partway through

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
    is_alpaca_source = dataset_config.type == "source" and dataset_config.provider == "alpaca"

    # Process based on dataset type
    if is_thetadata_source and dataset_id == "option-trades-1m":
        # ThetaData source - requires async client
        from rich.progress import Progress, BarColumn, TextColumn, TimeElapsedColumn, SpinnerColumn

        async def process_thetadata():
            from src.data.thetadata_client import ThetaDataClient, ThetaDataAPIError

            client = ThetaDataClient()

            if not await client._check_terminal_running():
                console.print("[red]ThetaData terminal not running! Start with: java -jar ThetaTerminalv3.jar[/red]")
                return

            success, fail = 0, 0

            for d in dates_to_process:
                try:
                    # Delete existing file just before rebuild (preserves data if job cancelled)
                    if rebuild:
                        loader.delete_file(dataset_id, underlying, d)

                    with Progress(
                        SpinnerColumn(),
                        TextColumn("[blue]{task.fields[date]}[/blue]"),
                        BarColumn(bar_width=30),
                        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
                        TextColumn("[dim]{task.fields[status]}[/dim]"),
                        TimeElapsedColumn(),
                        console=console,
                        transient=False,
                    ) as progress:
                        task = progress.add_task(
                            f"Processing {d}",
                            total=100,
                            date=str(d),
                            status="loading stocks...",
                        )

                        # Step 1: Load stock data (5%)
                        stocks_df = loader.load_single_file("stocks-1m", underlying, d)
                        if stocks_df is None or stocks_df.empty:
                            progress.update(task, completed=100, status="[yellow]no stock data[/yellow]")
                            fail += 1
                            continue

                        if "timestamp" not in stocks_df.columns and "window_start" in stocks_df.columns:
                            stocks_df = stocks_df.rename(columns={"window_start": "timestamp"})
                        if "timestamp" in stocks_df.columns:
                            stocks_df["timestamp"] = pd.to_datetime(stocks_df["timestamp"])

                        progress.update(task, completed=5, status="checking existing...")

                        # Step 2: Check existing trade data
                        existing_trade_agg_df = None
                        resume_from = None
                        if not force:
                            try:
                                existing_trade_agg_df = loader.load_single_file("option-trades-1m", underlying, d)
                                if existing_trade_agg_df is not None and not existing_trade_agg_df.empty and "timestamp" in existing_trade_agg_df.columns:
                                    resume_from = existing_trade_agg_df["timestamp"].max()
                            except Exception:
                                pass

                        progress.update(task, completed=10)

                        # Step 3: Fetch trade+quote data in 120-minute batches (10% -> 95%)
                        trade_agg_df = existing_trade_agg_df
                        total_rows = len(existing_trade_agg_df) if existing_trade_agg_df is not None else 0
                        all_new_batches = []

                        if trade_agg_df is None or trade_agg_df.empty or resume_from is not None:
                            # Determine expiration: try 0DTE first, fallback to all expirations
                            # (ThetaData API quirk: specific expiration queries may fail with 472
                            # even when data exists, but expiration=* works)
                            use_zero_dte = True
                            filter_to_nearest_exp = False
                            progress.update(task, status="checking expiration...")
                            try:
                                # Probe 0DTE with a single minute fetch
                                probe_df = await client.fetch_trade_quotes(
                                    d, underlying, zero_dte=True,
                                    start_time="09:30:00", end_time="09:31:00"
                                )
                                if probe_df is None or probe_df.empty:
                                    raise ThetaDataAPIError("No 0DTE data")
                            except ThetaDataAPIError:
                                # 0DTE query failed, fetch all expirations and filter later
                                use_zero_dte = False
                                filter_to_nearest_exp = True

                            resume_str = f" from {resume_from.strftime('%H:%M')}" if resume_from else ""
                            exp_str = " (nearest exp)" if filter_to_nearest_exp else ""
                            progress.update(task, status=f"fetching{resume_str}{exp_str}...")

                            def update_batch_progress(batch_num, total_batches, minutes_done, total_minutes):
                                # Map 10% to 95% range based on minutes progress
                                if total_minutes > 0:
                                    pct = 10 + (minutes_done / total_minutes) * 85  # 10% to 95%
                                    progress.update(task, completed=pct, status=f"{minutes_done}/{total_minutes} min")

                            async for batch_df in client.fetch_trade_quotes_batched(
                                d, underlying,
                                zero_dte=use_zero_dte,
                                concurrent_requests=concurrent,
                                resume_from=resume_from,
                                batch_minutes=120,
                                progress_callback=update_batch_progress
                            ):
                                if batch_df.empty:
                                    continue

                                # Filter to nearest expiration if needed
                                df = batch_df.copy()
                                if filter_to_nearest_exp and "expiration" in df.columns:
                                    # Find the nearest expiration (prefer 0DTE, then closest future)
                                    unique_exps = pd.to_datetime(df["expiration"]).dt.date.unique()
                                    query_date = d
                                    nearest = None
                                    for exp in sorted(unique_exps):
                                        if exp >= query_date:
                                            nearest = exp
                                            break
                                    if nearest is not None:
                                        df = df[pd.to_datetime(df["expiration"]).dt.date == nearest]
                                    if df.empty:
                                        continue

                                # Aggregate this batch
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

                                batch_agg_df = df.groupby(["minute", "strike", "right"]).agg({
                                    "price": ["first", "max", "min", "last"],
                                    "buy_volume": "sum",
                                    "sell_volume": "sum",
                                    "size": ["sum", "count"],
                                }).reset_index()
                                batch_agg_df.columns = [
                                    "timestamp", "strike", "right",
                                    "open", "high", "low", "close",
                                    "buy_volume", "sell_volume",
                                    "volume", "trade_count"
                                ]
                                batch_agg_df["ticker"] = underlying

                                all_new_batches.append(batch_agg_df)
                                total_rows = (len(existing_trade_agg_df) if existing_trade_agg_df is not None else 0) + \
                                            sum(len(b) for b in all_new_batches)
                                progress.update(task, status=f"{total_rows} rows")

                                # Save incrementally after each batch
                                if all_new_batches:
                                    new_df = pd.concat(all_new_batches, ignore_index=True)
                                    new_df = new_df.drop_duplicates(subset=["timestamp", "strike", "right"], keep="last")
                                    new_df = new_df.sort_values("timestamp").reset_index(drop=True)

                                    if existing_trade_agg_df is not None and not existing_trade_agg_df.empty:
                                        combined_df = pd.concat([existing_trade_agg_df, new_df], ignore_index=True)
                                        combined_df = combined_df.drop_duplicates(subset=["timestamp", "strike", "right"], keep="last")
                                        combined_df = combined_df.sort_values("timestamp").reset_index(drop=True)
                                        loader.save_to_cache(combined_df, "option-trades-1m", underlying, d)
                                        trade_agg_df = combined_df
                                    else:
                                        loader.save_to_cache(new_df, "option-trades-1m", underlying, d)
                                        trade_agg_df = new_df
                                    total_rows = len(trade_agg_df)

                        progress.update(task, completed=100, status=f"✓ {total_rows} rows")
                        success += 1

                except Exception as e:
                    console.print(f"[blue]{d}[/blue] [red]Error: {e}[/red]")
                    fail += 1

            console.print(f"\n[bold]Complete:[/bold] {success} succeeded, {fail} failed")

        asyncio.run(process_thetadata())

    elif is_alpaca_source and dataset_id == "stock-trades-1m":
        # Alpaca source - fetch stock trades with Lee-Ready classification
        from src.data.providers.alpaca import AlpacaProvider
        from rich.progress import Progress, BarColumn, TextColumn, TimeElapsedColumn, SpinnerColumn

        provider = AlpacaProvider()
        success = 0
        fail = 0

        for d in dates_to_process:
            try:
                # Delete existing file just before rebuild (preserves data if job cancelled)
                if rebuild:
                    loader.delete_file(dataset_id, underlying, d)

                # Check for existing data (incremental loading)
                existing_df = None
                resume_from = None
                if not force:
                    try:
                        existing_df = loader.load_single_file(dataset_id, underlying, d)
                        if existing_df is not None and not existing_df.empty and "timestamp" in existing_df.columns:
                            resume_from = existing_df["timestamp"].max()
                    except Exception:
                        pass  # No existing file, fetch full day

                # Use batched fetching with progress bar
                all_batches = []
                total_rows = 0

                with Progress(
                    SpinnerColumn(),
                    TextColumn("[blue]{task.fields[date]}[/blue]"),
                    BarColumn(bar_width=30),
                    TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
                    TextColumn("[dim]{task.fields[status]}[/dim]"),
                    TimeElapsedColumn(),
                    console=console,
                    transient=False,
                ) as progress:
                    resume_str = f" (from {resume_from.strftime('%H:%M')})" if resume_from else ""
                    task = progress.add_task(
                        f"Fetching {d}",
                        total=390,  # ~390 minutes in trading day
                        date=str(d),
                        status=f"starting{resume_str}...",
                    )

                    def update_progress(batch_num, total_batches, minutes_done, total_minutes):
                        progress.update(
                            task,
                            completed=minutes_done,
                            total=total_minutes,
                        )

                    for batch_df in provider.fetch_stock_trade_quote_1m_batched(
                        underlying, d,
                        resume_from=resume_from,
                        batch_minutes=5,
                        progress_callback=update_progress,
                    ):
                        all_batches.append(batch_df)
                        total_rows += len(batch_df)

                        # Update status with row count
                        progress.update(task, status=f"{total_rows} rows")

                        # Commit incrementally: merge and save after each batch
                        if all_batches:
                            new_df = pd.concat(all_batches, ignore_index=True)
                            new_df = new_df.drop_duplicates(subset=["timestamp"], keep="last")
                            new_df = new_df.sort_values("timestamp").reset_index(drop=True)

                            if existing_df is not None and not existing_df.empty:
                                combined_df = pd.concat([existing_df, new_df], ignore_index=True)
                                combined_df = combined_df.drop_duplicates(subset=["timestamp"], keep="last")
                                combined_df = combined_df.sort_values("timestamp").reset_index(drop=True)
                                loader.save_to_cache(combined_df, dataset_id, underlying, d)
                            else:
                                loader.save_to_cache(new_df, dataset_id, underlying, d)

                    # Final status
                    progress.update(task, completed=390, status=f"✓ {total_rows} rows")

                if all_batches:
                    success += 1
                elif existing_df is not None and not existing_df.empty:
                    console.print(f"[blue]{d}[/blue] [cyan]✓ {len(existing_df)} rows (up to date)[/cyan]")
                    success += 1
                else:
                    console.print(f"[blue]{d}[/blue] [yellow]No data[/yellow]")
                    fail += 1

            except Exception as e:
                console.print(f"[blue]{d}[/blue] [red]Error: {e}[/red]")
                fail += 1

        console.print(f"\n[bold]Complete:[/bold] {success} succeeded, {fail} failed")

    elif is_computed:
        # Computed datasets - use DAG loader with full dependency resolution
        from src.data.dag.tree_display import (
            LiveTreeDisplay, TreeNode, NodeStatus,
            convert_resolution_node_to_tree_node, print_static_tree
        )

        success_count = 0
        fail_count = 0

        for d in dates_to_process:
            # Delete existing file just before rebuild (preserves data if job cancelled)
            if rebuild:
                loader.delete_file(dataset_id, underlying, d)

            resolution_tree = loader.build_resolution_tree(dataset_id, underlying, d, force_recompute=force, show_all_deps=True)
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
                        unique_minutes: int = None,
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
                            "gap_filling": NodeStatus.LOADING,
                        }
                        node_status = status_map.get(status, NodeStatus.LOADING)
                        if progress >= 1.0 and status in ("loading", "computing", "gap_filling"):
                            node_status = NodeStatus.CACHED

                        status_msg = status_message
                        if status_msg is None:
                            if status == "loading":
                                status_msg = "Fetching..."
                            elif status == "computing":
                                status_msg = "Computing..."
                            elif status == "gap_filling":
                                status_msg = "Filling gaps..."

                        display.update_node(
                            dataset=dataset,
                            date=d,
                            status=node_status,
                            progress=progress,
                            row_count=row_count,
                            unique_minutes=unique_minutes,
                            error_msg=error_msg,
                            status_message=status_msg,
                        )

                    df = loader.load_day(
                        dataset_id, underlying, d,
                        force_recompute=force,
                        progress_callback=progress_callback
                    )

                    if df is not None and not df.empty:
                        # Calculate unique minutes for root node
                        from src.data.dag.loader import count_unique_minutes
                        root_unique_mins = count_unique_minutes(df)
                        # Update root node to complete
                        display.update_node(dataset_id, date=d, status=NodeStatus.COMPLETE, row_count=len(df), unique_minutes=root_unique_mins)

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
    dataset: str = typer.Argument("options-flow-1m", help="Dataset to check"),
    start_date: str = typer.Option(..., "--start-date", "-s", help="Start date (YYYY-MM-DD)"),
    end_date: Optional[str] = typer.Option(None, "--end-date", "-e", help="End date (YYYY-MM-DD)"),
    underlying: str = typer.Option("SPY", "--underlying", "-u", help="Underlying symbol"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Show all files"),
):
    """Check data quality for a dataset.

    Examples:
        python manage.py data check options-flow-1m --start-date 2024-01-01 --end-date 2024-12-31
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


@data_app.command("alpaca-test")
def data_alpaca_test():
    """Test Alpaca API connection and credentials.

    Checks:
    - API key and secret are configured
    - Connection to Alpaca API
    - Market status
    - Latest quote for SPY

    Examples:
        python manage.py data alpaca-test
    """
    import os
    from dotenv import load_dotenv

    load_dotenv()

    console.print("[bold]Testing Alpaca Connection[/bold]\n")

    # Check credentials
    api_key = os.getenv("ALPACA_API_KEY")
    secret_key = os.getenv("ALPACA_SECRET_KEY")

    if not api_key:
        console.print("[red]✗ ALPACA_API_KEY not found in .env[/red]")
        console.print("[dim]Add ALPACA_API_KEY=your_key to .env file[/dim]")
        raise typer.Exit(1)
    else:
        console.print(f"[green]✓ ALPACA_API_KEY found[/green] ({api_key[:8]}...)")

    if not secret_key:
        console.print("[red]✗ ALPACA_SECRET_KEY not found in .env[/red]")
        console.print("[dim]Add ALPACA_SECRET_KEY=your_secret to .env file[/dim]")
        raise typer.Exit(1)
    else:
        console.print(f"[green]✓ ALPACA_SECRET_KEY found[/green] ({secret_key[:8]}...)")

    # Test provider
    try:
        from src.data.providers.alpaca import AlpacaProvider
        provider = AlpacaProvider()

        # Check market status
        is_open = provider.check_market_open()
        if is_open:
            console.print("[green]✓ Market is OPEN[/green]")
        else:
            console.print("[yellow]○ Market is CLOSED[/yellow]")

        # Get latest quote
        quote = provider.get_latest_quote("SPY")
        if quote:
            console.print(f"[green]✓ Latest SPY quote:[/green] bid=${quote['bid_price']:.2f} ask=${quote['ask_price']:.2f}")
        else:
            console.print("[yellow]○ Could not fetch latest quote (market may be closed)[/yellow]")

        # Get latest bar
        bar = provider.get_latest_bar("SPY")
        if bar:
            console.print(f"[green]✓ Latest SPY bar:[/green] O=${bar['open']:.2f} H=${bar['high']:.2f} L=${bar['low']:.2f} C=${bar['close']:.2f}")
        else:
            console.print("[yellow]○ Could not fetch latest bar[/yellow]")

        console.print("\n[green]Alpaca connection successful![/green]")

    except Exception as e:
        console.print(f"[red]✗ Error connecting to Alpaca: {e}[/red]")
        raise typer.Exit(1)


@data_app.command("alpaca-fetch")
def data_alpaca_fetch(
    start_date: str = typer.Option(..., "--start-date", "-s", help="Start date (YYYY-MM-DD)"),
    end_date: Optional[str] = typer.Option(None, "--end-date", "-e", help="End date (YYYY-MM-DD)"),
    underlying: str = typer.Option("SPY", "--underlying", "-u", help="Underlying symbol"),
    feed: str = typer.Option("sip", "--feed", "-f", help="Data feed (sip=premium, iex=free)"),
    force: bool = typer.Option(False, "--force", help="Overwrite existing files"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Show what would be fetched"),
):
    """Fetch historical stocks data from Alpaca API.

    Uses the Alpaca Markets API to fetch minute-level OHLCV data.
    Premium users should use --feed sip for the full SIP feed.
    Free users can use --feed iex for IEX data.

    Examples:
        python manage.py data alpaca-fetch --start-date 2024-12-20
        python manage.py data alpaca-fetch --start-date 2024-12-01 --end-date 2024-12-20
        python manage.py data alpaca-fetch --start-date 2024-12-20 --feed iex
    """
    from datetime import datetime, timedelta
    from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn

    from src.data.providers.alpaca import AlpacaProvider

    # Parse dates
    start = datetime.strptime(start_date, "%Y-%m-%d").date()
    end = datetime.strptime(end_date, "%Y-%m-%d").date() if end_date else start

    loader = _get_dag_loader()
    provider = AlpacaProvider()

    # Get trading days
    from src.db.database import get_db
    db = get_db()
    calendar_dates = db.get_trading_days(start.isoformat(), end.isoformat())

    if calendar_dates:
        dates_to_fetch = sorted([datetime.strptime(d, "%Y-%m-%d").date() for d in calendar_dates])
    else:
        dates_to_fetch = []
        current = start
        while current <= end:
            if current.weekday() < 5:
                dates_to_fetch.append(current)
            current += timedelta(days=1)

    # Check which dates need fetching
    if not force:
        cached_dates = set(loader.get_cached_dates("stocks-1m", underlying))
        dates_to_fetch = [d for d in dates_to_fetch if d not in cached_dates]

    if not dates_to_fetch:
        console.print("[green]All dates already cached.[/green]")
        return

    console.print(f"[bold]Fetching {len(dates_to_fetch)} days from Alpaca ({feed} feed)[/bold]")

    if dry_run:
        for d in dates_to_fetch[:20]:
            console.print(f"  {d}")
        if len(dates_to_fetch) > 20:
            console.print(f"  ... and {len(dates_to_fetch) - 20} more")
        return

    success_count = 0
    fail_count = 0

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Fetching...", total=len(dates_to_fetch))

        for d in dates_to_fetch:
            progress.update(task, description=f"Fetching {underlying} {d}...")

            try:
                fetch_config = {"method": "rest_api", "feed": feed}
                raw_df = provider.fetch(fetch_config, underlying, d)

                if raw_df.empty:
                    console.print(f"  [yellow]{d}: No data[/yellow]")
                    fail_count += 1
                else:
                    # Map to internal schema
                    registry = loader.registry
                    config = registry.get("stocks-1m")
                    df = provider.map_schema(raw_df, config.field_map)

                    # Save to cache
                    path = loader.save_to_cache(df, "stocks-1m", underlying, d)
                    console.print(f"  [green]{d}: {len(df)} bars saved[/green]")
                    success_count += 1

            except Exception as e:
                console.print(f"  [red]{d}: Error - {e}[/red]")
                fail_count += 1

            progress.advance(task)

    console.print(f"\n[bold]Complete:[/bold] {success_count} succeeded, {fail_count} failed")


@data_app.command("remove")
def data_remove(
    dataset: str = typer.Argument(..., help="Dataset to remove (e.g., options-flow-1m, greeks-1m)"),
    start_date: str = typer.Option(..., "--start-date", "-s", help="Start date (YYYY-MM-DD)"),
    end_date: Optional[str] = typer.Option(None, "--end-date", "-e", help="End date (YYYY-MM-DD)"),
    underlying: str = typer.Option("SPY", "--underlying", "-u", help="Underlying symbol"),
    deep: bool = typer.Option(False, "--deep", "-d", help="Also remove all dependent datasets (downstream)"),
    force: bool = typer.Option(False, "--force", "-f", help="Skip confirmation"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Show what would be removed"),
):
    """Remove data files for a dataset within a date range.

    By default, only removes files for the specified dataset.
    Use --deep to also remove all downstream dependent datasets.

    Examples:
        # Remove GOOGL options-flow-1m for Dec 22
        python manage.py data remove options-flow-1m -s 2025-12-22 -u GOOGL

        # Remove all GOOGL data for Dec 22 (including dependencies)
        python manage.py data remove greeks-1m -s 2025-12-22 -u GOOGL --deep

        # Remove SPY data for a date range
        python manage.py data remove training-1m-raw -s 2025-12-01 -e 2025-12-15

        # Preview what would be removed
        python manage.py data remove options-flow-1m -s 2025-12-22 -u GOOGL --deep --dry-run
    """
    from datetime import datetime, timedelta

    loader = _get_dag_loader()
    registry = loader.registry

    # Validate dataset exists in registry
    if not registry.exists(dataset):
        console.print(f"[red]Unknown dataset: {dataset}[/red]")
        all_datasets = registry.list_datasets()
        console.print(f"[dim]Valid datasets: {', '.join(all_datasets)}[/dim]")
        raise typer.Exit(1)

    # Parse dates
    start = datetime.strptime(start_date, "%Y-%m-%d").date()
    end = datetime.strptime(end_date, "%Y-%m-%d").date() if end_date else start

    # Generate date range
    dates_in_range = []
    current = start
    while current <= end:
        dates_in_range.append(current)
        current += timedelta(days=1)

    # Determine which datasets to remove
    if deep:
        # Get all datasets that depend on this one (downstream)
        datasets_to_remove = [dataset] + registry.get_dependents(dataset)
        console.print(f"[bold]Deep removal: {len(datasets_to_remove)} datasets[/bold]")
    else:
        datasets_to_remove = [dataset]

    # Find files to remove
    files_to_remove = []
    for ds in datasets_to_remove:
        cached_dates = set(loader.get_cached_dates(ds, underlying))
        for d in dates_in_range:
            if d in cached_dates:
                path = loader.resolve_path(ds, underlying, d)
                if path.exists():
                    files_to_remove.append((ds, d, path))

    if not files_to_remove:
        console.print(f"[yellow]No files found for {underlying} in date range {start} to {end}[/yellow]")
        raise typer.Exit(0)

    # Display files to remove
    console.print(f"\n[bold]Files to remove ({len(files_to_remove)}):[/bold]")

    # Group by dataset for display
    from collections import defaultdict
    by_dataset = defaultdict(list)
    total_size = 0
    for ds, d, path in files_to_remove:
        size = path.stat().st_size if path.exists() else 0
        total_size += size
        by_dataset[ds].append((d, path, size))

    for ds in datasets_to_remove:
        if ds in by_dataset:
            files = by_dataset[ds]
            dates = sorted([f[0] for f in files])
            ds_size = sum(f[2] for f in files)
            if len(dates) == 1:
                console.print(f"  [cyan]{ds}[/cyan]: {dates[0]} ({format_file_size(ds_size)})")
            else:
                console.print(f"  [cyan]{ds}[/cyan]: {dates[0]} to {dates[-1]} ({len(files)} files, {format_file_size(ds_size)})")

    console.print(f"\n[dim]Total: {len(files_to_remove)} files, {format_file_size(total_size)}[/dim]")

    if dry_run:
        console.print("\n[yellow]Dry run - no files removed[/yellow]")
        raise typer.Exit(0)

    # Confirm deletion
    if not force:
        confirm = typer.confirm(f"\nRemove {len(files_to_remove)} files for {underlying}?")
        if not confirm:
            console.print("[dim]Cancelled[/dim]")
            raise typer.Exit(0)

    # Remove files
    removed_count = 0
    failed_count = 0
    for ds, d, path in files_to_remove:
        try:
            if path.exists():
                path.unlink()
                removed_count += 1
                # Also remove manifest if it exists
                manifest_path = path.with_suffix(path.suffix + ".manifest.json")
                if manifest_path.exists():
                    manifest_path.unlink()
        except Exception as e:
            console.print(f"[red]Failed to remove {path}: {e}[/red]")
            failed_count += 1

    console.print(f"\n[green]Removed {removed_count} files[/green]")
    if failed_count > 0:
        console.print(f"[red]Failed to remove {failed_count} files[/red]")


@data_app.command("alpaca-stream")
def data_alpaca_stream(
    symbols: str = typer.Option("SPY", "--symbols", "-s", help="Comma-separated symbols to stream"),
    feed: str = typer.Option("sip", "--feed", "-f", help="Data feed (sip=premium, iex=free)"),
    duration: int = typer.Option(60, "--duration", "-d", help="Duration in seconds (0 for indefinite)"),
    show_bars: bool = typer.Option(True, "--bars/--no-bars", help="Show minute bars"),
    show_quotes: bool = typer.Option(False, "--quotes/--no-quotes", help="Show quotes"),
    show_trades: bool = typer.Option(False, "--trades/--no-trades", help="Show trades"),
):
    """Start real-time data streaming from Alpaca.

    Streams live market data including bars, quotes, and trades.
    Premium users should use --feed sip for the full SIP feed.

    Examples:
        python manage.py data alpaca-stream --symbols SPY
        python manage.py data alpaca-stream --symbols SPY,QQQ --duration 300
        python manage.py data alpaca-stream --symbols SPY --quotes --trades
    """
    import signal
    import sys
    from datetime import datetime

    from src.data.providers.alpaca import AlpacaProvider

    symbol_list = [s.strip() for s in symbols.split(",")]
    provider = AlpacaProvider()

    console.print(f"[bold]Starting Alpaca Stream[/bold]")
    console.print(f"[dim]Symbols: {symbol_list}[/dim]")
    console.print(f"[dim]Feed: {feed}[/dim]")
    console.print(f"[dim]Duration: {'indefinite' if duration == 0 else f'{duration}s'}[/dim]")
    console.print("[dim]Press Ctrl+C to stop[/dim]\n")

    bar_count = 0
    quote_count = 0
    trade_count = 0

    def on_bar(bar):
        nonlocal bar_count
        bar_count += 1
        if show_bars:
            console.print(
                f"[green]BAR[/green] {bar['symbol']} {bar['timestamp']} "
                f"O={bar['open']:.2f} H={bar['high']:.2f} L={bar['low']:.2f} C={bar['close']:.2f} "
                f"V={bar['volume']:,}"
            )

    def on_quote(quote):
        nonlocal quote_count
        quote_count += 1
        if show_quotes:
            console.print(
                f"[blue]QUOTE[/blue] {quote['symbol']} {quote['timestamp']} "
                f"bid={quote['bid_price']:.2f}x{quote['bid_size']} "
                f"ask={quote['ask_price']:.2f}x{quote['ask_size']}"
            )

    def on_trade(trade):
        nonlocal trade_count
        trade_count += 1
        if show_trades:
            console.print(
                f"[yellow]TRADE[/yellow] {trade['symbol']} {trade['timestamp']} "
                f"price={trade['price']:.2f} size={trade['size']}"
            )

    def signal_handler(sig, frame):
        console.print("\n[yellow]Stopping stream...[/yellow]")
        provider.stop_stream()
        console.print(f"\n[bold]Stream Summary:[/bold]")
        console.print(f"  Bars: {bar_count}")
        console.print(f"  Quotes: {quote_count}")
        console.print(f"  Trades: {trade_count}")
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)

    try:
        import threading

        if duration > 0:
            # Start a timer to stop the stream
            def stop_after_duration():
                import time
                time.sleep(duration)
                provider.stop_stream()

            timer = threading.Thread(target=stop_after_duration, daemon=True)
            timer.start()

        provider.start_stream(
            symbols=symbol_list,
            on_bar=on_bar if show_bars else None,
            on_quote=on_quote if show_quotes else None,
            on_trade=on_trade if show_trades else None,
            feed=feed,
        )

    except KeyboardInterrupt:
        pass
    finally:
        provider.stop_stream()
        console.print(f"\n[bold]Stream Summary:[/bold]")
        console.print(f"  Bars: {bar_count}")
        console.print(f"  Quotes: {quote_count}")
        console.print(f"  Trades: {trade_count}")
