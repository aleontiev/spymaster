#!/usr/bin/env python3
"""
Cache Population CLI for Spymaster.

Populates both normalized and raw data caches for specified date ranges.

Usage:
    # Populate cache for specific date range
    uv run python populate_cache.py --start-date 2021-01-01 --end-date 2025-09-01

    # Populate cache for a single date
    uv run python populate_cache.py --start-date 2024-01-15 --end-date 2024-01-15

    # Force recompute (ignore existing cache)
    uv run python populate_cache.py --start-date 2021-01-01 --end-date 2021-12-31 --force

    # Only populate raw cache (skip normalization)
    uv run python populate_cache.py --start-date 2021-01-01 --end-date 2021-12-31 --raw-only

    # Clear all cache files
    uv run python populate_cache.py --clear
"""
import argparse
import re
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from src.data.loader import (
    NORMALIZED_CACHE_DIR,
    RAW_CACHE_DIR,
    CACHE_VERSION,
    load_normalized_day,
    load_raw_day,
    get_cache_path,
    get_raw_cache_path,
    extract_date_from_filename,
)
from src.utils.logger import get_logger

logger = get_logger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Populate normalized and raw data caches",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument(
        "--start-date",
        type=str,
        help="Start date (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--end-date",
        type=str,
        help="End date (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data",
        help="Root data directory (default: data)",
    )
    parser.add_argument(
        "--underlying",
        type=str,
        default="SPY",
        help="Underlying symbol (default: SPY)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force recompute even if cache exists",
    )
    parser.add_argument(
        "--raw-only",
        action="store_true",
        help="Only populate raw cache (skip normalization)",
    )
    parser.add_argument(
        "--normalized-only",
        action="store_true",
        help="Only populate normalized cache (skip raw)",
    )
    parser.add_argument(
        "--clear",
        action="store_true",
        help="Clear all cache files and exit",
    )
    parser.add_argument(
        "--gex-device",
        type=str,
        default="cpu",
        help="Device for GEX computation (default: cpu)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be done without actually doing it",
    )
    parser.add_argument(
        "--oi-dir",
        type=str,
        default="data/oi",
        help="Directory containing per-day OI parquet files (SPY_OI-0DTE_YYYY-MM-DD.parquet)",
    )

    return parser.parse_args()


def filter_files_by_date(files: list, start_date: str, end_date: str) -> list:
    """Filter files by date range."""
    date_pattern = re.compile(r'(\d{4}-\d{2}-\d{2})')
    filtered = []

    for f in files:
        match = date_pattern.search(f.stem)
        if match:
            file_date = match.group(1)
            if start_date and file_date < start_date:
                continue
            if end_date and file_date > end_date:
                continue
            filtered.append(f)

    return sorted(filtered)


def clear_cache() -> None:
    """Clear all cache files."""
    print("Clearing cache directories...")

    normalized_count = 0
    raw_count = 0

    if NORMALIZED_CACHE_DIR.exists():
        for f in NORMALIZED_CACHE_DIR.glob("*.parquet"):
            f.unlink()
            normalized_count += 1

    if RAW_CACHE_DIR.exists():
        for f in RAW_CACHE_DIR.glob("*.parquet"):
            f.unlink()
            raw_count += 1

    print(f"Cleared {normalized_count} normalized cache files")
    print(f"Cleared {raw_count} raw cache files")


def main() -> None:
    """Main function."""
    args = parse_args()

    print("=" * 70)
    print("Cache Population Tool")
    print("=" * 70)
    print(f"Cache version: {CACHE_VERSION}")
    print(f"Normalized cache: {NORMALIZED_CACHE_DIR}")
    print(f"Raw cache: {RAW_CACHE_DIR}")

    # Handle clear command
    if args.clear:
        clear_cache()
        return

    # Validate date arguments
    if not args.start_date or not args.end_date:
        print("\nError: --start-date and --end-date are required")
        print("Use --clear to clear all cache files")
        sys.exit(1)

    print(f"\nDate range: {args.start_date} to {args.end_date}")
    print(f"Underlying: {args.underlying}")
    print(f"Force recompute: {args.force}")
    print(f"Raw only: {args.raw_only}")
    print(f"Normalized only: {args.normalized_only}")
    print(f"OI directory: {args.oi_dir}")

    # Build paths
    data_dir = Path(args.data_dir)
    stocks_dir = data_dir / "stocks"
    options_dir = data_dir / "options"

    if not stocks_dir.exists():
        print(f"\nError: Stocks directory not found: {stocks_dir}")
        sys.exit(1)

    if not options_dir.exists():
        print(f"\nError: Options directory not found: {options_dir}")
        sys.exit(1)

    # Find all stocks files
    pattern = f"{args.underlying}_*.parquet"
    all_stocks_files = list(stocks_dir.glob(pattern))

    if not all_stocks_files:
        print(f"\nError: No stocks files found matching {stocks_dir / pattern}")
        sys.exit(1)

    # Filter by date range
    stocks_files = filter_files_by_date(all_stocks_files, args.start_date, args.end_date)

    if not stocks_files:
        print(f"\nNo files found in date range {args.start_date} to {args.end_date}")
        sys.exit(1)

    print(f"\nFound {len(stocks_files)} days to process")

    if args.dry_run:
        print("\nDry run - would process these files:")
        for f in stocks_files[:10]:
            date_str = extract_date_from_filename(f.stem)
            print(f"  {date_str}")
        if len(stocks_files) > 10:
            print(f"  ... and {len(stocks_files) - 10} more")
        return

    # Note: GEX features now come from pre-computed flow files in data/gex_flow/
    print(f"\nNote: GEX features are loaded from pre-computed flow files in data/gex_flow/")

    # Ensure cache directories exist
    NORMALIZED_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    RAW_CACHE_DIR.mkdir(parents=True, exist_ok=True)

    # Process each file
    print("\n" + "=" * 70)
    print("Processing Files")
    print("=" * 70)

    raw_count = 0
    normalized_count = 0
    skipped_count = 0
    error_count = 0

    for stocks_file in tqdm(stocks_files, desc="Populating cache", unit="day"):
        date_str = extract_date_from_filename(stocks_file.stem)

        # Find corresponding options file
        options_file = options_dir / f"{args.underlying}_{date_str}.parquet"
        if not options_file.exists():
            options_file = options_dir / f"{args.underlying}_OPTIONS-1M_{date_str}.parquet"

        if not options_file.exists():
            logger.warning(f"Options file not found for {date_str}")
            error_count += 1
            continue

        try:
            if args.raw_only:
                # Only populate raw cache
                raw_cache_path = get_raw_cache_path(date_str)
                if not args.force and raw_cache_path.exists():
                    skipped_count += 1
                    continue

                df = load_raw_day(
                    stocks_file,
                    options_file,
                    force_recompute=args.force,
                )
                if df is not None:
                    raw_count += 1
                else:
                    error_count += 1

            elif args.normalized_only:
                # Only populate normalized cache (but this also creates raw cache by default)
                normalized_cache_path = get_cache_path(date_str)
                if not args.force and normalized_cache_path.exists():
                    skipped_count += 1
                    continue

                df = load_normalized_day(
                    stocks_file,
                    options_file,
                    force_recompute=args.force,
                    also_cache_raw=False,  # Don't cache raw
                )
                if df is not None:
                    normalized_count += 1
                else:
                    error_count += 1

            else:
                # Populate both caches
                normalized_cache_path = get_cache_path(date_str)
                raw_cache_path = get_raw_cache_path(date_str)

                # Check if both already exist
                if not args.force and normalized_cache_path.exists() and raw_cache_path.exists():
                    skipped_count += 1
                    continue

                df = load_normalized_day(
                    stocks_file,
                    options_file,
                    force_recompute=args.force,
                    also_cache_raw=True,
                )
                if df is not None:
                    normalized_count += 1
                    raw_count += 1
                else:
                    error_count += 1

        except Exception as e:
            logger.error(f"Error processing {date_str}: {e}")
            error_count += 1

    # Summary
    print("\n" + "=" * 70)
    print("Summary")
    print("=" * 70)
    print(f"Normalized cache files created: {normalized_count}")
    print(f"Raw cache files created: {raw_count}")
    print(f"Skipped (already cached): {skipped_count}")
    print(f"Errors: {error_count}")

    # Show cache stats
    normalized_files = list(NORMALIZED_CACHE_DIR.glob("*.parquet"))
    raw_files = list(RAW_CACHE_DIR.glob("*.parquet"))
    print(f"\nTotal normalized cache files: {len(normalized_files)}")
    print(f"Total raw cache files: {len(raw_files)}")

    if normalized_files:
        total_size = sum(f.stat().st_size for f in normalized_files) / 1e6
        print(f"Normalized cache size: {total_size:.1f} MB")

    if raw_files:
        total_size = sum(f.stat().st_size for f in raw_files) / 1e6
        print(f"Raw cache size: {total_size:.1f} MB")

    print("\nDone!")


if __name__ == "__main__":
    main()
