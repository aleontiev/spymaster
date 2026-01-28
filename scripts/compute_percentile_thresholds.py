#!/usr/bin/env python
"""
Compute percentile thresholds for policy training labels.

Analyzes raw close prices from the specified date range and computes
the Nth percentile thresholds for LONG and SHORT signals at different
lookahead periods (1m, 5m, 15m).

Usage:
    uv run python scripts/compute_percentile_thresholds.py \
        --start-date 2023-01-01 --end-date 2025-08-31 \
        --percentile 5.0

Output:
    For each lookahead (1m, 5m, 15m):
    - 95th percentile (LONG threshold): top N% upward moves
    - 5th percentile (SHORT threshold): top N% downward moves (absolute value)
"""
import argparse
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


def load_raw_closes(
    data_dir: Path,
    underlying: str,
    start_date: datetime,
    end_date: datetime,
) -> Tuple[np.ndarray, List[datetime]]:
    """
    Load raw close prices from parquet files.

    Returns:
        raw_closes: Array of close prices
        dates: List of dates corresponding to each price
    """
    raw_dir = data_dir / "training-1m-raw" / underlying

    if not raw_dir.exists():
        raise ValueError(f"Raw data directory not found: {raw_dir}")

    all_raw_closes = []
    all_dates = []
    file_count = 0

    for month_dir in sorted(raw_dir.iterdir()):
        if not month_dir.is_dir():
            continue
        for parquet_file in sorted(month_dir.glob("*.parquet")):
            try:
                month_str = month_dir.name
                day_str = parquet_file.stem
                file_date = datetime.strptime(f"{month_str}-{day_str}", "%Y-%m-%d").date()

                if file_date < start_date.date():
                    continue
                if file_date > end_date.date():
                    continue

                raw_df = pd.read_parquet(parquet_file)
                if "close" not in raw_df.columns:
                    continue

                # Filter to market hours (first 390 rows = 6.5 hours)
                market_len = min(390, len(raw_df))
                raw_close = raw_df["close"].values[:market_len].astype(np.float32)

                all_raw_closes.append(raw_close)
                all_dates.extend([file_date] * market_len)

                file_count += 1
            except Exception as e:
                continue

    if not all_raw_closes:
        raise ValueError(f"No data loaded from {raw_dir}")

    raw_closes = np.concatenate(all_raw_closes)
    print(f"Loaded {len(raw_closes):,} bars from {file_count} days")
    print(f"Date range: {all_dates[0]} to {all_dates[-1]}")
    print(f"Price range: ${raw_closes.min():.2f} - ${raw_closes.max():.2f}")

    return raw_closes, all_dates


def compute_forward_returns(
    raw_closes: np.ndarray,
    lookahead: int,
) -> np.ndarray:
    """
    Compute forward returns for each minute.

    Args:
        raw_closes: Array of close prices
        lookahead: Minutes to look ahead

    Returns:
        Array of percentage returns
    """
    n = len(raw_closes)
    valid_count = n - lookahead

    returns = []
    for i in range(valid_count):
        current_price = raw_closes[i]
        future_price = raw_closes[i + lookahead]
        if current_price > 0:
            pct_change = (future_price - current_price) / current_price * 100
            returns.append(pct_change)

    return np.array(returns)


def compute_percentile_thresholds(
    returns: np.ndarray,
    percentile: float,
) -> Tuple[float, float]:
    """
    Compute percentile thresholds for LONG and SHORT signals.

    Args:
        returns: Array of percentage returns
        percentile: Target percentile (e.g., 5.0 for top 5%)

    Returns:
        (long_threshold, short_threshold): Absolute values
    """
    # LONG threshold: (100 - percentile)th percentile (e.g., 95th for top 5%)
    long_threshold = np.percentile(returns, 100 - percentile)

    # SHORT threshold: percentile-th percentile, take absolute value
    short_threshold = abs(np.percentile(returns, percentile))

    return long_threshold, short_threshold


def main():
    parser = argparse.ArgumentParser(
        description="Compute percentile thresholds for policy training"
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data",
        help="Root data directory",
    )
    parser.add_argument(
        "--underlying",
        type=str,
        default="SPY",
        help="Underlying symbol",
    )
    parser.add_argument(
        "--start-date",
        type=str,
        default="2023-01-01",
        help="Start date (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--end-date",
        type=str,
        default="2025-08-31",
        help="End date (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--percentile",
        type=float,
        default=5.0,
        help="Target percentile for signals (e.g., 5.0 for top 5%%)",
    )

    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    start_date = datetime.strptime(args.start_date, "%Y-%m-%d")
    end_date = datetime.strptime(args.end_date, "%Y-%m-%d")

    print("=" * 70)
    print("Computing Percentile Thresholds for Policy Training")
    print("=" * 70)
    print(f"Data directory: {data_dir}")
    print(f"Underlying: {args.underlying}")
    print(f"Date range: {args.start_date} to {args.end_date}")
    print(f"Target percentile: {args.percentile}% (top {args.percentile}% moves)")
    print()

    # Load raw closes
    raw_closes, dates = load_raw_closes(
        data_dir=data_dir,
        underlying=args.underlying,
        start_date=start_date,
        end_date=end_date,
    )

    # Compute thresholds for each lookahead
    lookaheads = [1, 5, 15]

    print("\n" + "=" * 70)
    print("Percentile Thresholds (for --long-threshold-pct and --short-threshold-pct)")
    print("=" * 70)
    print(f"\nTarget: Top {args.percentile}% moves in each direction\n")

    results = {}
    for lookahead in lookaheads:
        returns = compute_forward_returns(raw_closes, lookahead)

        long_threshold, short_threshold = compute_percentile_thresholds(
            returns, args.percentile
        )

        results[lookahead] = {
            "long": long_threshold,
            "short": short_threshold,
            "n_samples": len(returns),
        }

        # Count actual labels at these thresholds
        n_long = (returns >= long_threshold).sum()
        n_short = (returns <= -short_threshold).sum()
        n_hold = len(returns) - n_long - n_short

        pct_long = n_long / len(returns) * 100
        pct_short = n_short / len(returns) * 100
        pct_hold = n_hold / len(returns) * 100

        print(f"{lookahead}m lookahead:")
        print(f"  LONG threshold:  >= +{long_threshold:.4f}%")
        print(f"  SHORT threshold: <= -{short_threshold:.4f}%")
        print(f"  Label distribution:")
        print(f"    HOLD:  {n_hold:,} ({pct_hold:.2f}%)")
        print(f"    LONG:  {n_long:,} ({pct_long:.2f}%)")
        print(f"    SHORT: {n_short:,} ({pct_short:.2f}%)")
        print()

    # Print summary for copy-paste into training commands
    print("=" * 70)
    print("Training Commands")
    print("=" * 70)

    for lookahead in lookaheads:
        r = results[lookahead]
        if lookahead == 15:
            context = 60
            checkpoint_id = "percentile-60-15-v3-5pct"
            lejepa = "data/checkpoints/lejepa-60-15/lejepa_best.pt"
        elif lookahead == 5:
            context = 15
            checkpoint_id = "percentile-15-5-v2-5pct"
            lejepa = "data/checkpoints/lejepa-15-5-v1/lejepa_best.pt"
        else:  # 1m
            context = 5
            checkpoint_id = "percentile-5-1-v2-5pct"
            lejepa = "data/checkpoints/lejepa-5-1-v1/lejepa_best.pt"

        print(f"\n# {lookahead}m model (context={context})")
        print(f"uv run python train.py --model-type entry-percentile \\")
        print(f"  --lejepa-checkpoint {lejepa} \\")
        print(f"  --checkpoint-id {checkpoint_id} \\")
        print(f"  --start-date {args.start_date} --end-date {args.end_date} \\")
        print(f"  --context-len {context} --lookahead {lookahead} \\")
        print(f"  --long-threshold-pct {r['long']:.4f} \\")
        print(f"  --short-threshold-pct {r['short']:.4f} \\")
        print(f"  --epochs 50 --batch-size 256 \\")
        print(f"  --use-focal-loss --focal-gamma 2.0 \\")
        print(f"  --focal-alpha-hold 0.1 --focal-alpha-signal 1.0")


if __name__ == "__main__":
    main()
