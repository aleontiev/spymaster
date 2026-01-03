#!/usr/bin/env python3
"""
Rebuild GEX flow and training data for a date range.

This script uses the DAGLoader to force recomputation of:
- options-flow-1m: Options flow features
- training-1m-raw: Raw training data (includes GEX flow)

Usage:
    # Rebuild last 7 days for SPY
    uv run python scripts/rebuild_gex_training.py --days 7

    # Rebuild specific date range
    uv run python scripts/rebuild_gex_training.py --start 2025-12-13 --end 2025-12-19

    # Rebuild specific dataset only
    uv run python scripts/rebuild_gex_training.py --days 7 --dataset options-flow-1m
"""

import argparse
import sys
from datetime import date, timedelta
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.dag.loader import DAGLoader


def main():
    parser = argparse.ArgumentParser(
        description="Rebuild GEX flow and training data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument("--start", type=str, help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", type=str, help="End date (YYYY-MM-DD)")
    parser.add_argument("--days", type=int, default=7, help="Number of days back from today (default: 7)")
    parser.add_argument("--underlying", type=str, default="SPY", help="Underlying symbol (default: SPY)")
    parser.add_argument("--dataset", type=str, choices=["options-flow-1m", "training-1m-raw", "all"],
                       default="all", help="Dataset to rebuild (default: all)")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be rebuilt")

    args = parser.parse_args()

    # Determine date range
    if args.start and args.end:
        start_date = date.fromisoformat(args.start)
        end_date = date.fromisoformat(args.end)
    else:
        end_date = date.today()
        start_date = end_date - timedelta(days=args.days)

    print("=" * 60)
    print("GEX Flow & Training Data Rebuild")
    print("=" * 60)
    print(f"Underlying: {args.underlying}")
    print(f"Date range: {start_date} to {end_date}")
    print(f"Dataset: {args.dataset}")
    print()

    if args.dry_run:
        print("DRY RUN - showing what would be rebuilt:")
        print()

        with DAGLoader() as loader:
            # Show resolution trees
            if args.dataset in ("options-flow-1m", "all"):
                print("options-flow-1m resolution:")
                for d in loader._get_trading_dates(start_date, end_date):
                    tree = loader.build_resolution_tree("options-flow-1m", args.underlying, d, force_recompute=True)
                    loader.print_resolution_tree(tree)
                    print()

            if args.dataset in ("training-1m-raw", "all"):
                print("training-1m-raw resolution:")
                for d in loader._get_trading_dates(start_date, end_date):
                    tree = loader.build_resolution_tree("training-1m-raw", args.underlying, d, force_recompute=True)
                    loader.print_resolution_tree(tree)
                    print()
        return

    # Progress callback
    def progress_callback(dataset: str, status: str, progress: float = 0, row_count: int = 0,
                         status_message: str = "", error_msg: str = "", **kwargs):
        if status == "complete":
            print(f"  {dataset}: completed ({row_count} rows)")
        elif status == "computing":
            if status_message:
                print(f"  {dataset}: {status_message}")
        elif status == "loading":
            if status_message:
                print(f"  {dataset}: {status_message}")
        elif status == "cached":
            print(f"  {dataset}: cached ({row_count} rows)")
        elif error_msg:
            print(f"  {dataset}: ERROR - {error_msg}")

    # Rebuild datasets
    with DAGLoader() as loader:
        if args.dataset in ("options-flow-1m", "all"):
            print("Rebuilding options-flow-1m...")
            print("-" * 40)

            # First clear existing cache for the date range
            loader.clear_cache("options-flow-1m", args.underlying, start_date, end_date)

            # Then load with force_recompute
            df = loader.load(
                dataset="options-flow-1m",
                underlying=args.underlying,
                start_date=start_date,
                end_date=end_date,
                force_recompute=True,
                progress_callback=progress_callback,
            )
            print(f"\noptions-flow-1m: {len(df)} total rows rebuilt")
            print()

        if args.dataset in ("training-1m-raw", "all"):
            print("Rebuilding training-1m-raw...")
            print("-" * 40)

            # First clear existing cache for the date range
            loader.clear_cache("training-1m-raw", args.underlying, start_date, end_date)

            # Then load with force_recompute
            df = loader.load(
                dataset="training-1m-raw",
                underlying=args.underlying,
                start_date=start_date,
                end_date=end_date,
                force_recompute=True,
                progress_callback=progress_callback,
            )
            print(f"\ntraining-1m-raw: {len(df)} total rows rebuilt")
            print()

    print("=" * 60)
    print("Rebuild complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
