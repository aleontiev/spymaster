#!/usr/bin/env python3
"""
Backfill stock-trades-1m and option-trades-1m for SPY.
Runs one day at a time in reverse order (newest to oldest).
"""

import subprocess
import sys
from datetime import date
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.db.database import get_db
from commands import _get_dag_loader


def main():
    start_date = date(2020, 1, 1)
    end_date = date(2025, 12, 31)
    underlying = "SPY"
    datasets = ["stock-trades-1m", "option-trades-1m"]

    # Get trading days from database
    db = get_db()
    trading_days_str = db.get_trading_days(start_date.isoformat(), end_date.isoformat())
    trading_days = set(date.fromisoformat(d) for d in trading_days_str)
    print(f"Total trading days (2020-2025): {len(trading_days)}")

    loader = _get_dag_loader()

    for dataset in datasets:
        print(f"\n{'='*60}")
        print(f"Processing {dataset}")
        print(f"{'='*60}")

        # Get cached dates
        cached = set(d for d in loader.get_cached_dates(dataset, underlying)
                     if start_date <= d <= end_date)

        # Find missing dates
        missing = trading_days - cached
        missing_sorted = sorted(missing, reverse=True)  # newest first

        print(f"Cached: {len(cached)}, Missing: {len(missing)}")

        if not missing_sorted:
            print(f"All dates cached for {dataset}")
            continue

        print(f"Processing {len(missing_sorted)} dates in reverse order...")
        print(f"Newest: {missing_sorted[0]}, Oldest: {missing_sorted[-1]}")

        # Process in reverse order (newest first)
        for i, d in enumerate(missing_sorted, 1):
            print(f"\n[{i}/{len(missing_sorted)}] {dataset} @ {d}")

            cmd = [
                "uv", "run", "python", "manage.py", "data", "load",
                dataset,
                "--start-date", d.isoformat(),
                "--underlying", underlying,
            ]

            result = subprocess.run(cmd)
            if result.returncode != 0:
                print(f"Failed to backfill {dataset} for {d}")


if __name__ == "__main__":
    main()
