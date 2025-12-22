#!/usr/bin/env python3
"""
Fast parallel reprocessing of GEX flow features.

Computes Greeks from existing options data using Black-Scholes, then recomputes
GEX flow features. No ThetaData API calls required.

Usage:
    # Reprocess all existing GEX flow files
    uv run python scripts/reprocess_gex_flow.py --workers 16

    # Reprocess specific date range
    uv run python scripts/reprocess_gex_flow.py --start 2024-01-01 --end 2024-05-31 --workers 16

    # Dry run
    uv run python scripts/reprocess_gex_flow.py --dry-run
"""

import argparse
import re
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from scipy import stats

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.option_parser import parse_option_tickers_vectorized
from src.data.gex_flow_engine import GEXFlowEngine


# Constants
RISK_FREE_RATE = 0.05  # 5% risk-free rate assumption
DIVIDEND_YIELD = 0.013  # SPY approximate dividend yield


@dataclass
class ProcessingResult:
    """Result of processing a single day."""
    date_str: str
    success: bool
    message: str
    rows: int = 0


def compute_black_scholes_greeks(
    S: np.ndarray,
    K: np.ndarray,
    T: np.ndarray,
    sigma: np.ndarray,
    is_call: np.ndarray,
    r: float = RISK_FREE_RATE,
    q: float = DIVIDEND_YIELD,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute Black-Scholes Greeks (delta, gamma, IV passthrough).

    Args:
        S: Underlying prices
        K: Strike prices
        T: Time to expiration in years
        sigma: Implied volatility
        is_call: Boolean array, True for calls
        r: Risk-free rate
        q: Dividend yield

    Returns:
        Tuple of (delta, gamma, sigma) arrays
    """
    # Avoid division by zero
    T = np.maximum(T, 1e-10)
    sigma = np.maximum(sigma, 0.01)

    # Black-Scholes d1 and d2
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    # Delta
    delta = np.where(
        is_call,
        np.exp(-q * T) * stats.norm.cdf(d1),
        np.exp(-q * T) * (stats.norm.cdf(d1) - 1)
    )

    # Gamma (same for calls and puts)
    gamma = np.exp(-q * T) * stats.norm.pdf(d1) / (S * sigma * np.sqrt(T))

    return delta, gamma, sigma


def implied_volatility_newton(
    price: np.ndarray,
    S: np.ndarray,
    K: np.ndarray,
    T: np.ndarray,
    is_call: np.ndarray,
    r: float = RISK_FREE_RATE,
    q: float = DIVIDEND_YIELD,
    max_iter: int = 20,
    tol: float = 1e-6,
) -> np.ndarray:
    """
    Compute implied volatility using Newton-Raphson method.

    Returns IV array with NaN for failed convergence.
    """
    # Initial guess based on ATM approximation
    iv = np.full_like(price, 0.20, dtype=np.float64)

    # Avoid issues with zero time
    T = np.maximum(T, 1e-10)
    S = np.maximum(S, 0.01)
    K = np.maximum(K, 0.01)

    for _ in range(max_iter):
        # Black-Scholes price
        d1 = (np.log(S / K) + (r - q + 0.5 * iv**2) * T) / (iv * np.sqrt(T))
        d2 = d1 - iv * np.sqrt(T)

        if is_call.any():
            bs_price = np.where(
                is_call,
                S * np.exp(-q * T) * stats.norm.cdf(d1) - K * np.exp(-r * T) * stats.norm.cdf(d2),
                K * np.exp(-r * T) * stats.norm.cdf(-d2) - S * np.exp(-q * T) * stats.norm.cdf(-d1)
            )
        else:
            bs_price = K * np.exp(-r * T) * stats.norm.cdf(-d2) - S * np.exp(-q * T) * stats.norm.cdf(-d1)

        # Vega
        vega = S * np.exp(-q * T) * stats.norm.pdf(d1) * np.sqrt(T)
        vega = np.maximum(vega, 1e-10)

        # Newton step
        diff = price - bs_price
        iv_new = iv + diff / vega

        # Clamp to reasonable range
        iv_new = np.clip(iv_new, 0.01, 3.0)

        # Check convergence
        if np.all(np.abs(iv_new - iv) < tol):
            break

        iv = iv_new

    # Mark non-converged as NaN
    iv = np.where(np.abs(diff) > 0.1, np.nan, iv)

    return iv


def load_and_compute_greeks(
    options_file: Path,
    stocks_file: Path,
    oi_file: Optional[Path] = None,
) -> Optional[pd.DataFrame]:
    """
    Load options data and compute Greeks using Black-Scholes.

    Returns DataFrame with columns: timestamp, strike, is_call, delta, gamma, iv, volume
    """
    if not options_file.exists() or not stocks_file.exists():
        return None

    # Load options data
    options_df = pd.read_parquet(options_file)
    if options_df.empty:
        return None

    # Load stocks data for underlying prices
    stocks_df = pd.read_parquet(stocks_file)
    if "window_start" in stocks_df.columns:
        stocks_df = stocks_df.rename(columns={"window_start": "timestamp"})
    stocks_df["timestamp"] = pd.to_datetime(stocks_df["timestamp"])
    stocks_df = stocks_df.set_index("timestamp")

    # Parse option tickers
    if "ticker" not in options_df.columns:
        return None

    parsed = parse_option_tickers_vectorized(options_df["ticker"].values)
    options_df["underlying"] = parsed["underlying"]
    options_df["expiration"] = pd.to_datetime(parsed["expiration"])
    options_df["strike"] = parsed["strike"]
    options_df["option_type"] = parsed["option_type"]
    options_df["is_call"] = options_df["option_type"] == "C"

    # Parse timestamp
    if "window_start" in options_df.columns:
        options_df = options_df.rename(columns={"window_start": "timestamp"})
    options_df["timestamp"] = pd.to_datetime(options_df["timestamp"])

    # Filter to SPY only and 0DTE
    options_df = options_df[options_df["underlying"] == "SPY"].copy()
    trade_date = options_df["timestamp"].dt.date.iloc[0]
    options_df = options_df[options_df["expiration"].dt.date == trade_date]

    if options_df.empty:
        return None

    # Merge with underlying prices
    options_df = options_df.set_index("timestamp")
    options_df["underlying_price"] = options_df.index.map(
        lambda ts: stocks_df.loc[ts, "close"] if ts in stocks_df.index else np.nan
    )
    options_df = options_df.dropna(subset=["underlying_price"])
    options_df = options_df.reset_index()

    if options_df.empty:
        return None

    # Compute time to expiration (in years)
    # Market closes at 4:00 PM ET = 16:00
    market_close = options_df["expiration"] + pd.Timedelta(hours=16)
    options_df["T"] = (market_close - options_df["timestamp"]).dt.total_seconds() / (365.25 * 24 * 3600)
    options_df["T"] = options_df["T"].clip(lower=1e-10)

    # Use mid price for IV calculation
    options_df["mid_price"] = (options_df["open"] + options_df["close"]) / 2
    options_df["mid_price"] = options_df["mid_price"].clip(lower=0.01)

    # Compute IV
    options_df["iv"] = implied_volatility_newton(
        price=options_df["mid_price"].values,
        S=options_df["underlying_price"].values,
        K=options_df["strike"].values,
        T=options_df["T"].values,
        is_call=options_df["is_call"].values,
    )

    # Fill NaN IV with reasonable default
    options_df["iv"] = options_df["iv"].fillna(0.20)
    options_df["iv"] = options_df["iv"].clip(0.01, 3.0)

    # Compute Greeks
    delta, gamma, _ = compute_black_scholes_greeks(
        S=options_df["underlying_price"].values,
        K=options_df["strike"].values,
        T=options_df["T"].values,
        sigma=options_df["iv"].values,
        is_call=options_df["is_call"].values,
    )

    options_df["delta"] = delta
    options_df["gamma"] = gamma

    # Select relevant columns for GEXFlowEngine
    greeks_df = options_df[[
        "timestamp", "strike", "option_type", "delta", "gamma", "iv", "volume", "underlying_price"
    ]].copy()

    # Rename to match GEXFlowEngine expectations
    greeks_df = greeks_df.rename(columns={
        "option_type": "right",
        "iv": "implied_volatility",
    })

    return greeks_df


def process_single_day(args_tuple: tuple) -> ProcessingResult:
    """
    Process a single day: compute Greeks and GEX flow features.

    Args tuple: (date_str, options_file, stocks_file, oi_file, output_file)
    """
    date_str, options_file, stocks_file, oi_file, output_file = args_tuple

    try:
        # Load options and compute Greeks
        greeks_df = load_and_compute_greeks(
            Path(options_file),
            Path(stocks_file),
            Path(oi_file) if oi_file else None,
        )

        if greeks_df is None or greeks_df.empty:
            return ProcessingResult(date_str, False, "No options data", 0)

        # Load OI for baseline (if available)
        oi_df = pd.DataFrame()
        if oi_file and Path(oi_file).exists():
            oi_df = pd.read_parquet(oi_file)

        # Initialize GEX flow engine
        engine = GEXFlowEngine()

        # Load underlying prices
        stocks_df = pd.read_parquet(stocks_file)
        if "window_start" in stocks_df.columns:
            stocks_df = stocks_df.rename(columns={"window_start": "timestamp"})
        stocks_df["timestamp"] = pd.to_datetime(stocks_df["timestamp"])

        # Compute GEX flow features (pass OI for structural GEX calculations)
        # Note: We pass empty trade_quotes since we're computing from Greeks only
        features_df = engine.compute_flow_features(
            greeks_df=greeks_df,
            trade_quotes_df=pd.DataFrame(),  # No trade classification
            underlying_prices=stocks_df,
            daily_oi_df=oi_df,  # Pass OI for structural GEX features
        )

        if features_df.empty:
            return ProcessingResult(date_str, False, "No features computed", 0)

        # Save to parquet
        Path(output_file).parent.mkdir(parents=True, exist_ok=True)
        features_df.to_parquet(output_file)

        return ProcessingResult(date_str, True, "OK", len(features_df))

    except Exception as e:
        return ProcessingResult(date_str, False, str(e)[:100], 0)


def get_dates_to_process(
    gex_flow_dir: Path,
    options_dir: Path,
    stocks_dir: Path,
    oi_dir: Path,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
) -> list[tuple]:
    """Get list of dates to process with file paths."""
    dates_to_process = []

    # Get existing GEX flow dates
    existing_dates = set()
    for f in gex_flow_dir.glob("SPY_thetadata_1m_combined_*.parquet"):
        match = re.search(r"(\d{4}-\d{2}-\d{2})", f.stem)
        if match:
            existing_dates.add(match.group(1))

    # Get available options dates
    for options_file in sorted(options_dir.glob("SPY_OPTIONS-1M_*.parquet")):
        match = re.search(r"(\d{4}-\d{2}-\d{2})", options_file.stem)
        if not match:
            continue

        date_str = match.group(1)

        # Filter by date range
        if start_date and date_str < start_date:
            continue
        if end_date and date_str > end_date:
            continue

        # Check if stocks file exists
        stocks_file = stocks_dir / f"SPY_STOCKS-1M_{date_str}.parquet"
        if not stocks_file.exists():
            continue

        # OI file (optional)
        oi_file = oi_dir / f"SPY_OI-0DTE_{date_str}.parquet"
        oi_path = str(oi_file) if oi_file.exists() else None

        # Output file
        output_file = gex_flow_dir / f"SPY_thetadata_1m_combined_{date_str}.parquet"

        dates_to_process.append((
            date_str,
            str(options_file),
            str(stocks_file),
            oi_path,
            str(output_file),
        ))

    return dates_to_process


def main():
    parser = argparse.ArgumentParser(
        description="Reprocess GEX flow features from local data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument("--start", type=str, help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", type=str, help="End date (YYYY-MM-DD)")
    parser.add_argument("--workers", type=int, default=8, help="Number of parallel workers")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be processed")
    parser.add_argument("--gex-flow-dir", type=Path, default=Path("data/gex_flow"))
    parser.add_argument("--options-dir", type=Path, default=Path("data/options"))
    parser.add_argument("--stocks-dir", type=Path, default=Path("data/stocks"))
    parser.add_argument("--oi-dir", type=Path, default=Path("data/oi"))

    args = parser.parse_args()

    print("=" * 60)
    print("GEX Flow Reprocessing")
    print("=" * 60)

    # Get dates to process
    dates_to_process = get_dates_to_process(
        args.gex_flow_dir,
        args.options_dir,
        args.stocks_dir,
        args.oi_dir,
        args.start,
        args.end,
    )

    print(f"Found {len(dates_to_process)} dates to process")

    if not dates_to_process:
        print("Nothing to process!")
        return

    if args.dry_run:
        print("\nDry run - would process:")
        for date_str, *_ in dates_to_process[:10]:
            print(f"  {date_str}")
        if len(dates_to_process) > 10:
            print(f"  ... and {len(dates_to_process) - 10} more")
        return

    # Process in parallel
    start_time = time.time()
    success_count = 0
    fail_count = 0

    print(f"\nProcessing with {args.workers} workers...")
    print("-" * 60)

    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(process_single_day, args_tuple): args_tuple[0]
            for args_tuple in dates_to_process
        }

        for i, future in enumerate(as_completed(futures), 1):
            result = future.result()

            if result.success:
                success_count += 1
                status = "OK"
            else:
                fail_count += 1
                status = "FAIL"

            # Progress update
            if i % 50 == 0 or not result.success:
                elapsed = time.time() - start_time
                rate = i / elapsed if elapsed > 0 else 0
                eta = (len(dates_to_process) - i) / rate if rate > 0 else 0
                print(f"[{i}/{len(dates_to_process)}] {result.date_str}: {status} - {result.message} "
                      f"({rate:.1f}/s, ETA: {eta/60:.1f}m)")

    # Summary
    elapsed = time.time() - start_time
    print("\n" + "=" * 60)
    print("Reprocessing Complete")
    print("=" * 60)
    print(f"Success: {success_count}")
    print(f"Failed: {fail_count}")
    print(f"Time: {elapsed:.1f}s ({elapsed/60:.1f}m)")
    print(f"Rate: {len(dates_to_process)/elapsed:.2f} days/second")


if __name__ == "__main__":
    main()
