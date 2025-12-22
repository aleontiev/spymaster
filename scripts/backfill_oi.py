#!/usr/bin/env python3
"""
Backfill missing OI data from ThetaData.

This script identifies trading days that have stock data but no OI data,
then fetches the missing OI from ThetaData and saves it to per-day parquet files.

Usage:
    # First, start the ThetaData terminal:
    # java -jar ThetaTerminalv3.jar

    # Then run this script:
    uv run python scripts/backfill_oi.py

    # Dry run to see what would be fetched:
    uv run python scripts/backfill_oi.py --dry-run

    # Backfill specific date range:
    uv run python scripts/backfill_oi.py --start-date 2021-01-01 --end-date 2021-12-31
"""
import argparse
import asyncio
import re
import sys
from datetime import date, datetime
from pathlib import Path

import pandas as pd

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.thetadata_client import (
    ThetaDataClient,
    ThetaDataConnectionError,
    ThetaDataAPIError,
)


def get_missing_oi_dates(
    stocks_dir: Path,
    oi_dir: Path,
    underlying: str = "SPY",
    start_date: str | None = None,
    end_date: str | None = None,
) -> list[str]:
    """Find trading days with stock data but no OI data."""
    # Get all stock trading days
    stock_dates = set()
    for f in stocks_dir.glob(f"{underlying}_STOCKS-1M_*.parquet"):
        match = re.search(r"(\d{4}-\d{2}-\d{2})", f.stem)
        if match:
            stock_dates.add(match.group(1))

    # Get all OI dates
    oi_dates = set()
    for f in oi_dir.glob(f"{underlying}_OI-0DTE_*.parquet"):
        match = re.search(r"(\d{4}-\d{2}-\d{2})", f.stem)
        if match:
            oi_dates.add(match.group(1))

    # Find missing
    missing = sorted(stock_dates - oi_dates)

    # Filter by date range if specified
    if start_date:
        missing = [d for d in missing if d >= start_date]
    if end_date:
        missing = [d for d in missing if d <= end_date]

    return missing


def get_nearest_expiration(df: pd.DataFrame, query_date: date) -> pd.DataFrame:
    """
    Filter DataFrame to only include options with the nearest expiration to query_date.

    For GEX calculations, we want the options that are closest to expiring.
    This handles days where 0DTE doesn't exist (Tue/Thu before Nov 2022).

    The OI reported on query_date represents EOD OI from the previous trading day.
    So the nearest expiration might be:
    - Same day (0DTE) if options expire on query_date
    - Previous day's expiration (reported next morning)
    - Next available expiration
    """
    if df.empty or "expiration" not in df.columns:
        return df

    # Convert expiration to date for comparison
    expirations = df["expiration"].dt.date.unique()

    # Find the nearest expiration to the query date
    # Prefer same-day or past expirations (represents actual 0DTE or near-term activity)
    query_dt = query_date if isinstance(query_date, date) else datetime.strptime(query_date, "%Y-%m-%d").date()

    nearest = None
    min_diff = float("inf")

    for exp in expirations:
        diff = (exp - query_dt).days
        # Prioritize: 0DTE (diff=0), then past expirations (diff<0), then future
        # Use absolute value but prefer non-positive diffs
        abs_diff = abs(diff)
        if abs_diff < min_diff:
            min_diff = abs_diff
            nearest = exp
        elif abs_diff == min_diff and diff <= 0:
            # If same distance, prefer the past/current expiration
            nearest = exp

    if nearest is None:
        return df

    # Filter to nearest expiration
    mask = df["expiration"].dt.date == nearest
    return df[mask].copy()


async def backfill_oi_for_date(
    client: ThetaDataClient,
    query_date: str,
    oi_dir: Path,
    underlying: str = "SPY",
    use_nearest_expiration: bool = True,
) -> bool:
    """
    Fetch and save OI data for a single date.

    For days with 0DTE options available, fetches 0DTE OI directly.
    For days without 0DTE (Tue/Thu before Nov 2022), fetches all expirations
    and filters to the nearest one.

    Args:
        client: ThetaData client
        query_date: Date string in YYYY-MM-DD format
        oi_dir: Directory to save OI parquet files
        underlying: Ticker symbol (default: SPY)
        use_nearest_expiration: If True, fall back to nearest expiration when 0DTE unavailable
    """
    try:
        # Parse date
        dt = datetime.strptime(query_date, "%Y-%m-%d").date()

        df = pd.DataFrame()  # Start with empty df
        tried_0dte = False

        # First try to fetch 0DTE OI for this date
        try:
            df = await client.fetch_open_interest_for_date(
                query_date=dt,
                root=underlying,
                zero_dte=True,
            )
            tried_0dte = True
        except ThetaDataAPIError as e:
            # Status 472 = no data found, which is expected for Tue/Thu pre-Nov 2022
            if "472" in str(e):
                tried_0dte = True  # Mark that we tried 0DTE
            else:
                raise  # Re-raise other errors

        # If no 0DTE data (empty or 472 error), try fetching all expirations
        if df.empty and use_nearest_expiration and tried_0dte:
            print(f"  {query_date}: No 0DTE data, trying nearest expiration...")
            try:
                df = await client.fetch_open_interest_for_date(
                    query_date=dt,
                    root=underlying,
                    expiration="*",  # All expirations
                    zero_dte=False,
                )

                if not df.empty:
                    # Filter to nearest expiration
                    df = get_nearest_expiration(df, dt)
                    if not df.empty:
                        nearest_exp = df["expiration"].dt.date.iloc[0]
                        print(f"  {query_date}: Using nearest expiration {nearest_exp}")
            except ThetaDataAPIError as e:
                if "472" not in str(e):
                    raise  # Re-raise non-472 errors

        if df.empty:
            print(f"  {query_date}: No data returned")
            return False

        # Save to per-day file
        output_path = oi_dir / f"{underlying}_OI-0DTE_{query_date}.parquet"
        df.to_parquet(output_path, index=False)
        print(f"  {query_date}: Saved {len(df)} records")
        return True

    except ThetaDataAPIError as e:
        print(f"  {query_date}: API error - {e}")
        return False
    except Exception as e:
        print(f"  {query_date}: Error - {e}")
        return False


async def main():
    parser = argparse.ArgumentParser(
        description="Backfill missing OI data from ThetaData"
    )
    parser.add_argument(
        "--stocks-dir",
        type=str,
        default="data/stocks",
        help="Directory containing stock data files",
    )
    parser.add_argument(
        "--oi-dir",
        type=str,
        default="data/oi",
        help="Directory for OI data files",
    )
    parser.add_argument(
        "--underlying",
        type=str,
        default="SPY",
        help="Underlying symbol (default: SPY)",
    )
    parser.add_argument(
        "--start-date",
        type=str,
        help="Start date filter (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--end-date",
        type=str,
        help="End date filter (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be fetched without actually fetching",
    )
    parser.add_argument(
        "--concurrent",
        type=int,
        default=3,
        help="Number of concurrent requests (default: 3)",
    )
    args = parser.parse_args()

    stocks_dir = Path(args.stocks_dir)
    oi_dir = Path(args.oi_dir)

    if not stocks_dir.exists():
        print(f"Error: Stocks directory not found: {stocks_dir}")
        sys.exit(1)

    oi_dir.mkdir(parents=True, exist_ok=True)

    # Find missing dates
    print("Scanning for missing OI dates...")
    missing_dates = get_missing_oi_dates(
        stocks_dir=stocks_dir,
        oi_dir=oi_dir,
        underlying=args.underlying,
        start_date=args.start_date,
        end_date=args.end_date,
    )

    print(f"\nFound {len(missing_dates)} missing OI dates")

    if not missing_dates:
        print("Nothing to backfill!")
        return

    # Analyze by day of week
    from collections import Counter

    weekday_names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    weekday_counts = Counter()
    for d in missing_dates:
        dt = datetime.strptime(d, "%Y-%m-%d")
        weekday_counts[weekday_names[dt.weekday()]] += 1

    print("\nMissing dates by day of week:")
    for day in ["Mon", "Tue", "Wed", "Thu", "Fri"]:
        if weekday_counts[day] > 0:
            print(f"  {day}: {weekday_counts[day]}")

    if args.dry_run:
        print("\nDry run - would fetch these dates:")
        for d in missing_dates[:20]:
            print(f"  {d}")
        if len(missing_dates) > 20:
            print(f"  ... and {len(missing_dates) - 20} more")
        return

    # Initialize ThetaData client
    client = ThetaDataClient()

    print("\nChecking ThetaData terminal...")
    if not await client._check_terminal_running():
        print(
            "ERROR: ThetaData terminal is not running.\n"
            "Please start it with: java -jar ThetaTerminalv3.jar\n"
            "Make sure creds.txt contains your email and password."
        )
        sys.exit(1)

    print("Terminal is running. Starting backfill...\n")

    # Process dates with concurrency limit
    semaphore = asyncio.Semaphore(args.concurrent)
    success_count = 0
    fail_count = 0

    async def fetch_with_semaphore(query_date: str) -> bool:
        async with semaphore:
            return await backfill_oi_for_date(
                client=client,
                query_date=query_date,
                oi_dir=oi_dir,
                underlying=args.underlying,
            )

    # Process in batches for progress reporting
    batch_size = 20
    for i in range(0, len(missing_dates), batch_size):
        batch = missing_dates[i : i + batch_size]
        batch_num = i // batch_size + 1
        total_batches = (len(missing_dates) + batch_size - 1) // batch_size

        print(f"\nBatch {batch_num}/{total_batches} ({batch[0]} to {batch[-1]})")

        tasks = [fetch_with_semaphore(d) for d in batch]
        results = await asyncio.gather(*tasks)

        for success in results:
            if success:
                success_count += 1
            else:
                fail_count += 1

        # Small delay between batches
        await asyncio.sleep(0.1)

    # Summary
    print("\n" + "=" * 50)
    print("Backfill Complete")
    print("=" * 50)
    print(f"Successfully fetched: {success_count}")
    print(f"Failed: {fail_count}")

    # Verify final count
    final_oi_files = list(oi_dir.glob(f"{args.underlying}_OI-0DTE_*.parquet"))
    print(f"\nTotal OI files: {len(final_oi_files)}")


if __name__ == "__main__":
    asyncio.run(main())
