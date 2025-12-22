#!/usr/bin/env python3
"""
Convert naive Eastern Time timestamps to UTC in trade-quote-1m parquet files.

This script converts timestamps from naive datetime (assumed ET) to UTC timezone-aware
datetimes. It properly handles DST transitions.

Usage:
    # Dry run (show what would be converted)
    uv run python scripts/convert_timestamps_to_utc.py --dry-run

    # Convert all files
    uv run python scripts/convert_timestamps_to_utc.py

    # Convert with backup
    uv run python scripts/convert_timestamps_to_utc.py --backup

    # Convert specific date range
    uv run python scripts/convert_timestamps_to_utc.py --start 2025-01-01 --end 2025-06-30
"""

import argparse
import shutil
from datetime import datetime, date
from pathlib import Path
from typing import Optional
import logging

import pandas as pd
import pytz

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

ET = pytz.timezone("America/New_York")
UTC = pytz.UTC

DATA_DIR = Path("/home/ant/code/spymaster/data/trade-quote-1m")


def parse_date_from_path(file_path: Path) -> Optional[date]:
    """Extract date from file path like .../2025-01/15.parquet -> 2025-01-15."""
    try:
        day = int(file_path.stem)
        year_month = file_path.parent.name  # "2025-01"
        year, month = year_month.split("-")
        return date(int(year), int(month), day)
    except (ValueError, AttributeError):
        return None


def is_already_utc(df: pd.DataFrame) -> bool:
    """Check if timestamps are already in UTC format."""
    if df.empty or "timestamp" not in df.columns:
        return True

    # Check if timezone-aware
    ts = df["timestamp"].iloc[0]
    if hasattr(ts, "tzinfo") and ts.tzinfo is not None:
        return True

    # Check if times look like UTC (14:30 - 21:00 for market hours)
    # ET market hours: 9:30 - 16:00 -> UTC: 14:30 - 21:00 (EST) or 13:30 - 20:00 (EDT)
    first_hour = df["timestamp"].iloc[0].hour
    # If first timestamp hour is >= 13, it's likely already UTC
    return first_hour >= 13


def convert_et_to_utc(df: pd.DataFrame) -> pd.DataFrame:
    """Convert naive ET timestamps to UTC."""
    if df.empty or "timestamp" not in df.columns:
        return df

    df = df.copy()

    # Localize naive timestamps to ET, then convert to UTC
    # Use ambiguous='infer' for DST transitions
    df["timestamp"] = (
        pd.to_datetime(df["timestamp"])
        .dt.tz_localize(ET, ambiguous="infer", nonexistent="shift_forward")
        .dt.tz_convert(UTC)
    )

    return df


def process_file(
    file_path: Path,
    dry_run: bool = False,
    backup: bool = False,
) -> tuple[bool, str]:
    """
    Process a single parquet file.

    Returns:
        (success, message) tuple
    """
    try:
        df = pd.read_parquet(file_path)

        if is_already_utc(df):
            return True, "already UTC"

        if dry_run:
            sample_before = df["timestamp"].iloc[0]
            df_converted = convert_et_to_utc(df)
            sample_after = df_converted["timestamp"].iloc[0]
            return True, f"would convert: {sample_before} -> {sample_after}"

        # Create backup if requested
        if backup:
            backup_path = file_path.with_suffix(".parquet.bak")
            shutil.copy2(file_path, backup_path)

        # Convert and save
        df_converted = convert_et_to_utc(df)
        df_converted.to_parquet(file_path, index=False)

        return True, "converted"

    except Exception as e:
        return False, f"error: {e}"


def main():
    parser = argparse.ArgumentParser(
        description="Convert naive ET timestamps to UTC in trade-quote-1m files"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be converted without making changes",
    )
    parser.add_argument(
        "--backup",
        action="store_true",
        help="Create .bak backup files before converting",
    )
    parser.add_argument(
        "--start",
        type=str,
        help="Start date (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--end",
        type=str,
        help="End date (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--underlying",
        type=str,
        default="SPY",
        help="Underlying symbol (default: SPY)",
    )
    args = parser.parse_args()

    # Parse date range
    start_date = date.fromisoformat(args.start) if args.start else None
    end_date = date.fromisoformat(args.end) if args.end else None

    # Find all parquet files
    underlying_dir = DATA_DIR / args.underlying
    if not underlying_dir.exists():
        logger.error(f"Directory not found: {underlying_dir}")
        return 1

    files = sorted(underlying_dir.glob("**/*.parquet"))

    # Filter by date range
    if start_date or end_date:
        filtered_files = []
        for f in files:
            file_date = parse_date_from_path(f)
            if file_date is None:
                continue
            if start_date and file_date < start_date:
                continue
            if end_date and file_date > end_date:
                continue
            filtered_files.append(f)
        files = filtered_files

    if not files:
        logger.warning("No files found to process")
        return 0

    logger.info(f"Found {len(files)} files to process")
    if args.dry_run:
        logger.info("DRY RUN - no changes will be made")

    # Process files
    converted = 0
    skipped = 0
    errors = 0

    for i, file_path in enumerate(files):
        success, message = process_file(
            file_path,
            dry_run=args.dry_run,
            backup=args.backup,
        )

        rel_path = file_path.relative_to(DATA_DIR)

        if not success:
            logger.error(f"[{i+1}/{len(files)}] {rel_path}: {message}")
            errors += 1
        elif "already UTC" in message:
            logger.debug(f"[{i+1}/{len(files)}] {rel_path}: {message}")
            skipped += 1
        else:
            logger.info(f"[{i+1}/{len(files)}] {rel_path}: {message}")
            converted += 1

        # Progress update every 50 files
        if (i + 1) % 50 == 0:
            logger.info(f"Progress: {i+1}/{len(files)} files processed")

    # Summary
    logger.info(f"\nSummary:")
    logger.info(f"  Converted: {converted}")
    logger.info(f"  Skipped (already UTC): {skipped}")
    logger.info(f"  Errors: {errors}")

    return 0 if errors == 0 else 1


if __name__ == "__main__":
    exit(main())
