"""
Calendar commands for SpyMaster CLI.

Commands for managing market calendar data.
"""

from typing import Optional

import typer
from rich import box
from rich.panel import Panel
from rich.table import Table

from commands import console

# Create the typer app for calendar commands
calendar_app = typer.Typer(help="Manage market calendar data", invoke_without_command=True)


@calendar_app.callback()
def calendar_callback(ctx: typer.Context):
    """Manage market calendar data. Run without subcommand to show status."""
    if ctx.invoked_subcommand is None:
        calendar_status()


@calendar_app.command("status")
def calendar_status():
    """Show market calendar coverage status."""
    from src.db.database import get_db

    db = get_db()
    coverage = db.get_calendar_coverage()

    if not coverage["min_date"]:
        console.print(Panel(
            "[yellow]No calendar data in database.[/yellow]\n\n"
            "[dim]Sync calendar with:[/dim]\n"
            "[cyan]python manage.py calendar sync --start-date 2020-01-01 --end-date 2025-12-31[/cyan]",
            title="Market Calendar",
            border_style="dim",
        ))
        return

    table = Table(title="Market Calendar Coverage", box=box.ROUNDED)
    table.add_column("Metric", style="bold")
    table.add_column("Value", style="cyan")

    table.add_row("Date Range", f"{coverage['min_date']} to {coverage['max_date']}")
    table.add_row("Total Days", str(coverage["total_days"]))
    table.add_row("Trading Days", str(coverage["trading_days"]))
    table.add_row("Non-Trading Days", str(coverage["total_days"] - coverage["trading_days"]))

    console.print(table)


@calendar_app.command("sync")
def calendar_sync(
    start_date: str = typer.Option(..., "--start-date", "-s", help="Start date (YYYY-MM-DD)"),
    end_date: str = typer.Option(..., "--end-date", "-e", help="End date (YYYY-MM-DD)"),
    force: bool = typer.Option(False, "--force", "-f", help="Re-fetch existing dates"),
):
    """Sync market calendar from ThetaData API.

    Fetches market schedule (open, early_close, full_close, weekend) for each
    date in the range and stores it in the database.

    Examples:
        python manage.py calendar sync --start-date 2020-01-01 --end-date 2025-12-31
        python manage.py calendar sync -s 2024-01-01 -e 2024-12-31 --force
    """
    import asyncio
    from src.db.database import get_db
    from src.data.thetadata_client import ThetaDataClient

    db = get_db()

    if force:
        missing_dates = None
    else:
        missing_dates = db.get_missing_calendar_dates(start_date, end_date)
        if not missing_dates:
            console.print("[green]All dates already in calendar![/green]")
            return
        console.print(f"[dim]Found {len(missing_dates)} dates not in calendar[/dim]")

    async def do_sync():
        client = ThetaDataClient()

        console.print("[dim]Checking ThetaData terminal...[/dim]")
        if not await client._check_terminal_running():
            console.print("[red]ThetaData terminal not running! Start with: java -jar ThetaTerminalv3.jar[/red]")
            return

        console.print("[green]Terminal running. Fetching calendar data...[/green]\n")

        calendars = await client.fetch_calendar_for_date_range(
            start_date, end_date, concurrent_requests=30
        )

        if not calendars:
            console.print("[yellow]No calendar data received[/yellow]")
            return

        saved = 0
        for cal in calendars:
            if not force and missing_dates and cal["date"] not in missing_dates:
                continue

            db.upsert_calendar_from_api(cal["date"], cal)
            saved += 1

        console.print(f"[green]Saved {saved} calendar entries to database[/green]")
        calendar_status()

    asyncio.run(do_sync())


@calendar_app.command("show")
def calendar_show(
    start_date: str = typer.Option(..., "--start-date", "-s", help="Start date (YYYY-MM-DD)"),
    end_date: Optional[str] = typer.Option(None, "--end-date", "-e", help="End date (YYYY-MM-DD)"),
    trading_only: bool = typer.Option(False, "--trading-only", "-t", help="Show only trading days"),
):
    """Show calendar entries for a date range.

    Examples:
        python manage.py calendar show --start-date 2024-12-01 --end-date 2024-12-31
        python manage.py calendar show -s 2024-12-01 -e 2024-12-31 --trading-only
        python manage.py calendar show -s 2024-12-25  # Single date
    """
    from datetime import datetime
    from src.db.database import get_db

    db = get_db()

    if end_date is None:
        end_date = start_date

    calendars = db.get_calendar_range(start_date, end_date)

    if not calendars:
        console.print(f"[yellow]No calendar data for {start_date} to {end_date}[/yellow]")
        console.print("[dim]Sync calendar first with: python manage.py calendar sync[/dim]")
        return

    table = Table(title=f"Market Calendar ({start_date} to {end_date})", box=box.ROUNDED)
    table.add_column("Date", style="bold")
    table.add_column("Day", style="dim")
    table.add_column("Schedule", style="cyan")
    table.add_column("Market Open")
    table.add_column("Market Close")
    table.add_column("Minutes", justify="right")

    for cal in calendars:
        if trading_only and not cal.is_trading_day:
            continue

        date_obj = datetime.strptime(cal.date, "%Y-%m-%d")
        day_name = date_obj.strftime("%a")

        schedule_colors = {
            "open": "green",
            "early_close": "yellow",
            "full_close": "red",
            "weekend": "dim",
        }
        color = schedule_colors.get(cal.schedule_type, "white")
        schedule_styled = f"[{color}]{cal.schedule_type}[/{color}]"

        table.add_row(
            cal.date,
            day_name,
            schedule_styled,
            cal.market_open or "-",
            cal.market_close or "-",
            str(cal.trading_minutes) if cal.trading_minutes > 0 else "-",
        )

    console.print(table)


@calendar_app.command("trading-days")
def calendar_trading_days(
    start_date: str = typer.Option(..., "--start-date", "-s", help="Start date (YYYY-MM-DD)"),
    end_date: str = typer.Option(..., "--end-date", "-e", help="End date (YYYY-MM-DD)"),
    count_only: bool = typer.Option(False, "--count", "-c", help="Show only count"),
):
    """List trading days in a date range.

    Examples:
        python manage.py calendar trading-days --start-date 2024-12-01 --end-date 2024-12-31
        python manage.py calendar trading-days -s 2024-01-01 -e 2024-12-31 --count
    """
    from datetime import datetime
    from collections import defaultdict
    from src.db.database import get_db

    db = get_db()
    trading_days = db.get_trading_days(start_date, end_date)

    if count_only:
        console.print(f"[cyan]{len(trading_days)}[/cyan] trading days from {start_date} to {end_date}")
        return

    if not trading_days:
        console.print(f"[yellow]No trading days found for {start_date} to {end_date}[/yellow]")
        console.print("[dim]Sync calendar first with: python manage.py calendar sync[/dim]")
        return

    console.print(f"[bold]Trading Days ({start_date} to {end_date}):[/bold]\n")

    by_month = defaultdict(list)
    for d in trading_days:
        month = d[:7]
        by_month[month].append(d)

    for month, days in sorted(by_month.items()):
        month_dt = datetime.strptime(month, "%Y-%m")
        month_name = month_dt.strftime("%B %Y")
        console.print(f"[cyan]{month_name}:[/cyan] {len(days)} days")

        if not count_only:
            for i in range(0, len(days), 7):
                row = days[i:i+7]
                day_nums = [d.split("-")[2] for d in row]
                console.print(f"  {', '.join(day_nums)}")

    console.print(f"\n[bold]Total:[/bold] [cyan]{len(trading_days)}[/cyan] trading days")


@calendar_app.command("check")
def calendar_check(
    date_str: str = typer.Argument(..., help="Date to check (YYYY-MM-DD)"),
):
    """Check if a specific date is a trading day.

    Examples:
        python manage.py calendar check 2024-12-25
        python manage.py calendar check 2024-12-16
    """
    from datetime import datetime
    from src.db.database import get_db

    db = get_db()
    cal = db.get_calendar_day(date_str)

    if cal is None:
        console.print(f"[yellow]Date {date_str} not in calendar[/yellow]")
        console.print("[dim]Sync calendar first with: python manage.py calendar sync[/dim]")
        return

    date_obj = datetime.strptime(date_str, "%Y-%m-%d")
    day_name = date_obj.strftime("%A")

    if cal.is_trading_day:
        console.print(f"[green]✓[/green] {date_str} ({day_name}) is a [green]trading day[/green]")
        console.print(f"  Schedule: {cal.schedule_type}")
        console.print(f"  Hours: {cal.market_open} - {cal.market_close} ET")
        console.print(f"  Minutes: {cal.trading_minutes}")
    else:
        console.print(f"[red]✗[/red] {date_str} ({day_name}) is [red]not a trading day[/red]")
        console.print(f"  Reason: {cal.schedule_type}")
