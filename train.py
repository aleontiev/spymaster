#!/usr/bin/env python3
"""
Unified Training Script for all model types.

Supports four model types:
- jepa: LeJEPA self-supervised pre-training
- entry-3class: 3-class entry policy (HOLD, BUY_CALL, BUY_PUT)
- entry-5class: 5-class entry policy (HOLD, CALL_ATM, CALL_OTM, PUT_ATM, PUT_OTM)
- entry-regression: Continuous regression policy (score in [-1, 1])

Usage:
    # Train LeJEPA
    uv run python train.py --model-type jepa --checkpoint-id my-lejepa --epochs 10

    # Train 3-class entry policy
    uv run python train.py --model-type entry-3class \
        --lejepa-checkpoint checkpoints/lejepa/lejepa_best.pt \
        --checkpoint-id entry-3class-v1 --epochs 50

    # Train 5-class entry policy with ROI-based labeling
    uv run python train.py --model-type entry-5class \
        --lejepa-checkpoint checkpoints/lejepa/lejepa_best.pt \
        --checkpoint-id entry-5class-v1 --epochs 50

    # Train continuous regression entry policy
    uv run python train.py --model-type entry-regression \
        --lejepa-checkpoint checkpoints/lejepa/lejepa_best.pt \
        --checkpoint-id entry-regression-v1 --epochs 50
"""
import argparse
import math
import re
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, time, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from src.data.processing import InMemoryDataset, MarketPatch, create_dataloader
from src.data.synthetic import create_test_dataset
from src.data.loader import (
    create_combined_dataset,
    load_normalized_data,
    NORMALIZED_CACHE_DIR,
    RAW_CACHE_DIR,
    CACHE_VERSION,
    extract_date_from_filename,
    get_cache_path,
    get_raw_cache_path,
    load_raw_day,
    normalize_features,
    parse_date,
)
from src.model.lejepa import LeJEPA
from src.model.policy import EntryPolicy, EntryAction, EntryActionLegacy, RegressionEntryPolicy
from src.model.loss import FocalLoss
from src.managers.checkpoint_manager import CheckpointManager


# =============================================================================
# JEPA Sliding Window Dataset
# =============================================================================


class JEPASlidingWindowDataset(Dataset):
    """
    Simple sliding window dataset for JEPA training.

    Creates (context, target) pairs from a DataFrame of normalized features.
    """

    def __init__(
        self,
        data: torch.Tensor,
        context_len: int,
        target_len: int,
        prediction_gap: int = 0,
        stride: int = 1,
    ):
        """
        Args:
            data: Tensor of shape (num_rows, num_features)
            context_len: Number of candles for context window
            target_len: Number of candles for target window
            prediction_gap: Gap between context end and target start
            stride: Step size for sliding window
        """
        self.data = data
        self.context_len = context_len
        self.target_len = target_len
        self.prediction_gap = prediction_gap
        self.stride = stride

        # Calculate valid indices
        total_len = context_len + prediction_gap + target_len
        max_start = len(data) - total_len
        self.indices = list(range(0, max_start + 1, stride))

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        start = self.indices[idx]
        context_end = start + self.context_len
        target_start = context_end + self.prediction_gap
        target_end = target_start + self.target_len

        context = self.data[start:context_end]
        target = self.data[target_start:target_end]

        return context, target


class JEPADatasetWithPremarket(Dataset):
    """
    JEPA dataset with premarket and opening range.

    Each day's data has:
    - premarket: 4 candles × 5 OHLCV (T-3, T-2, T-1, PM ranges)
    - opening_range: 1 candle × 5 OHLCV (aggregated 9:30-9:44 AM)
    - market: Variable candles × N features (9:45 AM onwards)

    Creates (premarket, opening_range, context, target) tuples.
    """

    def __init__(
        self,
        day_data: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]],  # (premarket, opening_range, market)
        context_len: int,
        target_len: int,
        prediction_gap: int = 0,
        stride: int = 1,
    ):
        self.day_data = day_data
        self.context_len = context_len
        self.target_len = target_len
        self.prediction_gap = prediction_gap
        self.stride = stride

        # Build index: (day_idx, start_within_day)
        self.indices = []
        total_len = context_len + prediction_gap + target_len
        for day_idx, (premarket, opening_range, market) in enumerate(day_data):
            max_start = len(market) - total_len
            for start in range(0, max_start + 1, stride):
                self.indices.append((day_idx, start))

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        day_idx, start = self.indices[idx]
        premarket, opening_range, market = self.day_data[day_idx]

        context_end = start + self.context_len
        target_start = context_end + self.prediction_gap
        target_end = target_start + self.target_len

        context = market[start:context_end]
        target = market[target_start:target_end]

        return premarket, opening_range, context, target


def create_sliding_window_dataset(
    df: pd.DataFrame,
    context_len: int,
    target_len: int,
    prediction_gap: int = 0,
    stride: int = 1,
) -> JEPASlidingWindowDataset:
    """
    Create a JEPA sliding window dataset from a DataFrame.

    Args:
        df: DataFrame with normalized features (excluding timestamp/date columns)
        context_len: Number of candles for context window
        target_len: Number of candles for target window
        prediction_gap: Gap between context end and target start
        stride: Step size for sliding window

    Returns:
        JEPASlidingWindowDataset ready for training
    """
    # Convert to tensor (float32 for training)
    data = torch.tensor(df.values, dtype=torch.float32)

    return JEPASlidingWindowDataset(
        data=data,
        context_len=context_len,
        target_len=target_len,
        prediction_gap=prediction_gap,
        stride=stride,
    )


def load_raw_training_data_with_premarket(
    raw_dir: Path,
    start_date,
    end_date,
    max_files: Optional[int] = None,
    opening_range_minutes: int = 15,
) -> Tuple[List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]], List]:
    """
    Load raw training data with premarket and opening range separated.

    Args:
        raw_dir: Path to training-1m-raw/{underlying} directory
        start_date: Start date
        end_date: End date
        max_files: Optional limit on number of files
        opening_range_minutes: Minutes for opening range aggregation (default: 15)

    Returns:
        Tuple of:
        - List of (premarket, opening_range, market) tensors per day
        - List of dates
    """
    # Find all parquet files
    all_files = []
    for month_dir in sorted(raw_dir.iterdir()):
        if not month_dir.is_dir():
            continue
        for parquet_file in sorted(month_dir.glob("*.parquet")):
            try:
                month_str = month_dir.name
                day_str = parquet_file.stem
                file_date = datetime.strptime(f"{month_str}-{day_str}", "%Y-%m-%d").date()

                if start_date and file_date < start_date:
                    continue
                if end_date and file_date > end_date:
                    continue
                all_files.append((file_date, parquet_file))
            except ValueError:
                continue

    all_files.sort(key=lambda x: x[0])

    if max_files:
        all_files = all_files[:max_files]

    day_data = []
    dates = []

    # OHLCV columns for premarket and opening range
    ohlcv_cols = ["open", "high", "low", "close", "volume"]

    # Canonical market features (consistent across all raw data files)
    # These 27 features are normalized in the model
    market_cols = [
        # Price features (5)
        "open", "high", "low", "close", "volume",
        # VWAP (1)
        "vwap",
        # Options flow (10)
        "atm_spread", "net_premium_flow",
        "call_strikes_active", "put_strikes_active",
        "atm_call_volume", "otm_call_volume",
        "atm_put_volume", "otm_put_volume",
        # Greeks/GEX (columns common to both old and new formats)
        "net_gex", "net_dex",
        # Technical indicators (4)
        "anchored_vwap_z", "vwap_divergence", "implied_volatility",
        # Time features (3)
        "time_to_close", "sin_time", "cos_time",
    ]

    for file_date, parquet_file in tqdm(all_files, desc="Loading raw data", unit="day"):
        df = pd.read_parquet(parquet_file)

        # Set timestamp as index if present
        if "timestamp" in df.columns:
            df = df.set_index("timestamp")

        # Determine if we have premarket rows (8 premarket + market rows)
        has_premarket = len(df) > 390
        premarket_offset = 8 if has_premarket else 0

        if has_premarket:
            # Extract T-3, T-2, T-1, PM (skip blanks at positions 1, 3, 5, 7)
            premarket_rows = df.iloc[:8:2]  # rows 0, 2, 4, 6
            premarket_df = premarket_rows[ohlcv_cols].copy()
        else:
            # No premarket - create zeros
            premarket_df = pd.DataFrame(0.0, index=range(4), columns=ohlcv_cols)

        # Market data starts after premarket
        market_full = df.iloc[premarket_offset:]

        # Reference price for normalization: first market candle's open
        ref_price = market_full["open"].iloc[0]
        if ref_price <= 0:
            ref_price = 1.0  # Fallback to avoid log(0)

        # Normalize premarket OHLCV using log returns relative to ref_price
        # Prices -> log(price / ref_price), Volume -> log(1 + volume / 1e6)
        # Clip prices to avoid log(0) - use a small epsilon
        eps = 1e-8
        if has_premarket:
            premarket_normalized = premarket_df.copy()
            for col in ["open", "high", "low", "close"]:
                prices = np.clip(premarket_df[col].values, eps, None)
                premarket_normalized[col] = np.log(prices / ref_price)
            premarket_normalized["volume"] = np.log1p(premarket_df["volume"] / 1e6)
            premarket_df = premarket_normalized
        else:
            # No premarket - zeros are fine (log return of 0 = price equals ref)
            pass

        # Compute opening range (aggregate first N minutes)
        opening_range_rows = market_full.iloc[:opening_range_minutes]
        if len(opening_range_rows) >= opening_range_minutes:
            raw_opening_range = {
                "open": max(opening_range_rows["open"].iloc[0], eps),
                "high": max(opening_range_rows["high"].max(), eps),
                "low": max(opening_range_rows["low"].min(), eps),
                "close": max(opening_range_rows["close"].iloc[-1], eps),
                "volume": opening_range_rows["volume"].sum(),
            }
            # Normalize opening range using log returns
            opening_range = {
                "open": np.log(raw_opening_range["open"] / ref_price),
                "high": np.log(raw_opening_range["high"] / ref_price),
                "low": np.log(raw_opening_range["low"] / ref_price),
                "close": np.log(raw_opening_range["close"] / ref_price),
                "volume": np.log1p(raw_opening_range["volume"] / 1e6),
            }
        else:
            # Not enough data - use zeros
            opening_range = {col: 0.0 for col in ohlcv_cols}

        opening_range_df = pd.DataFrame([opening_range])

        # Market data after opening range - select only canonical columns
        market_after_or = market_full.iloc[opening_range_minutes:]

        # Select canonical columns, filling missing with 0
        market_selected = pd.DataFrame(index=market_after_or.index)
        for col in market_cols:
            if col in market_after_or.columns:
                market_selected[col] = market_after_or[col]
            else:
                # Check for old column names and map them
                if col == "net_gex" and "cumulative_net_gex" in market_after_or.columns:
                    market_selected[col] = market_after_or["cumulative_net_gex"]
                elif col == "net_dex" and "net_delta_flow" in market_after_or.columns:
                    market_selected[col] = market_after_or["net_delta_flow"]
                else:
                    market_selected[col] = 0.0

        # Convert to tensors
        premarket_tensor = torch.tensor(premarket_df.values, dtype=torch.float32)
        opening_range_tensor = torch.tensor(opening_range_df.values, dtype=torch.float32).squeeze(0)
        market_tensor = torch.tensor(market_selected.values, dtype=torch.float32)

        day_data.append((premarket_tensor, opening_range_tensor, market_tensor))
        dates.append(file_date)

    return day_data, dates


# =============================================================================
# Parallel Cache Building
# =============================================================================


def _process_single_day_for_cache(args_tuple: tuple) -> tuple[str, bool, str]:
    """
    Process a single day and save to cache (worker function for parallel processing).

    Args tuple: (stocks_file, options_file, date_str, oi_dir, build_raw, build_normalized)

    Returns: (date_str, success, message)
    """
    stocks_file, options_file, date_str, oi_dir, build_raw, build_normalized = args_tuple

    try:
        # Check if already cached
        raw_path = get_raw_cache_path(date_str)
        norm_path = get_cache_path(date_str)

        if build_raw and raw_path.exists():
            build_raw = False
        if build_normalized and norm_path.exists():
            build_normalized = False

        if not build_raw and not build_normalized:
            return (date_str, True, "Already cached")

        # Load OI for this date (per-day file)
        oi_df = None
        if oi_dir:
            oi_file = Path(oi_dir) / f"SPY_OI-0DTE_{date_str}.parquet"
            if oi_file.exists():
                oi_df = pd.read_parquet(oi_file)

        # Create GEX engine with OI
        gex_engine = None
        if oi_df is not None and not oi_df.empty:
            gex_engine = GEXEngine(device="cpu")
            gex_engine.load_oi_chain(oi_df)

        # Load raw data (handles caching internally)
        raw_df = load_raw_day(
            stocks_file=Path(stocks_file),
            options_file=Path(options_file),
            oi_df=oi_df,
            gex_engine=gex_engine,
            force_recompute=build_raw,
        )

        if raw_df is None or raw_df.empty:
            return (date_str, False, "No data")

        # Build and save normalized cache
        if build_normalized:
            norm_df = normalize_features(raw_df.copy())
            norm_df.index.name = "timestamp"
            NORMALIZED_CACHE_DIR.mkdir(parents=True, exist_ok=True)
            norm_df.to_parquet(norm_path, index=True)

        return (date_str, True, f"OK ({len(raw_df)} rows)")

    except Exception as e:
        return (date_str, False, f"{str(e)[:80]}")


def get_available_files_for_cache(
    stocks_dir: Path,
    options_dir: Path,
    underlying: str = "SPY",
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
) -> list[tuple[Path, Path, str]]:
    """Get list of (stocks_file, options_file, date_str) tuples."""
    files = []
    for stocks_file in sorted(stocks_dir.glob(f"{underlying}_STOCKS-1M_*.parquet")):
        try:
            date_str = extract_date_from_filename(stocks_file.stem)
            if not date_str:
                continue

            if start_date and date_str < start_date:
                continue
            if end_date and date_str > end_date:
                continue

            # Find matching options file
            options_file = options_dir / f"{underlying}_OPTIONS-1M_{date_str}.parquet"
            if options_file.exists():
                files.append((stocks_file, options_file, date_str))
        except (ValueError, IndexError):
            continue
    return files


def get_cached_dates(cache_dir: Path, tag: str = CACHE_VERSION) -> set[str]:
    """Get set of dates that already have cache."""
    dates = set()
    for f in cache_dir.glob(f"SPY_{tag}_*.parquet"):
        try:
            parts = f.stem.split("_")
            if len(parts) >= 3:
                dates.add(parts[2])  # Format: SPY_v7_2025-01-01
        except (ValueError, IndexError):
            continue
    return dates


def get_cached_raw_dates(cache_dir: Path) -> set[str]:
    """Get set of dates that already have raw cache."""
    dates = set()
    for f in cache_dir.glob("SPY_raw_*.parquet"):
        try:
            parts = f.stem.split("_")
            if len(parts) >= 3:
                dates.add(parts[2])  # Format: SPY_raw_2025-01-01
        except (ValueError, IndexError):
            continue
    return dates


def build_cache_parallel(
    stocks_dir: Path,
    options_dir: Path,
    oi_dir: Optional[Path] = None,
    underlying: str = "SPY",
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    workers: int = 8,
    force: bool = False,
) -> tuple[int, int]:
    """
    Build cache in parallel using ProcessPoolExecutor.

    This efficiently processes multiple days concurrently, much faster than
    sequential processing for large date ranges.

    Args:
        stocks_dir: Directory containing raw stocks data
        options_dir: Directory containing raw options data
        oi_dir: Directory containing daily OI files
        underlying: Underlying symbol
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)
        workers: Number of parallel workers
        force: Force rebuild even if cached

    Returns:
        Tuple of (success_count, fail_count)
    """
    import time as time_module

    # Ensure cache directories exist
    RAW_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    NORMALIZED_CACHE_DIR.mkdir(parents=True, exist_ok=True)

    # Get available files
    all_files = get_available_files_for_cache(
        stocks_dir, options_dir, underlying, start_date, end_date
    )

    if not all_files:
        print(f"No data files found for {underlying} in {start_date} to {end_date}")
        return 0, 0

    # Filter out already cached dates (unless force)
    if not force:
        cached_raw = get_cached_raw_dates(RAW_CACHE_DIR)
        cached_norm = get_cached_dates(NORMALIZED_CACHE_DIR)

        files_to_process = []
        for stocks_file, options_file, date_str in all_files:
            needs_raw = date_str not in cached_raw
            needs_norm = date_str not in cached_norm
            if needs_raw or needs_norm:
                files_to_process.append((stocks_file, options_file, date_str, needs_raw, needs_norm))

        already_cached = len(all_files) - len(files_to_process)
        if already_cached > 0:
            print(f"Already cached: {already_cached} days (cache version: {CACHE_VERSION})")
        print(f"Days to process: {len(files_to_process)}")
    else:
        files_to_process = [
            (s, o, d, True, True)
            for s, o, d in all_files
        ]
        print(f"Force rebuild: processing all {len(files_to_process)} days")

    if not files_to_process:
        print("Cache is up to date!")
        return len(all_files), 0

    # Prepare args for workers (include oi_dir)
    oi_dir_str = str(oi_dir) if oi_dir else None
    worker_args = [
        (str(s), str(o), d, oi_dir_str, r, n)
        for s, o, d, r, n in files_to_process
    ]

    # Process in parallel
    start_time = time_module.time()
    success_count = 0
    fail_count = 0

    print(f"\nBuilding cache with {workers} parallel workers...")
    print("=" * 60)

    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(_process_single_day_for_cache, arg): arg[2]
            for arg in worker_args
        }

        for i, future in enumerate(as_completed(futures), 1):
            date_str, success, msg = future.result()
            if success:
                success_count += 1
                status = "OK"
            else:
                fail_count += 1
                status = "FAIL"

            # Progress update every 10 or on failure
            if i % 10 == 0 or not success:
                elapsed = time_module.time() - start_time
                rate = i / elapsed if elapsed > 0 else 0
                eta = (len(worker_args) - i) / rate if rate > 0 else 0
                print(f"[{i}/{len(worker_args)}] {date_str}: {status} - {msg} "
                      f"({rate:.1f}/s, ETA: {eta/60:.1f}m)")

    # Summary
    elapsed = time_module.time() - start_time
    print("\n" + "=" * 60)
    print("Cache Build Complete")
    print("=" * 60)
    print(f"Success: {success_count}")
    print(f"Failed: {fail_count}")
    print(f"Time: {elapsed:.1f}s ({elapsed/60:.1f}m)")
    if elapsed > 0:
        print(f"Rate: {len(worker_args)/elapsed:.2f} days/second")

    return success_count, fail_count


# =============================================================================
# Common Utilities
# =============================================================================


def set_seed(seed: int) -> None:
    """Set random seeds for reproducibility."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device(device_arg: str) -> torch.device:
    """Determine device based on argument and availability."""
    if device_arg == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
    return torch.device(device_arg)


def build_file_pattern(
    underlying: str,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
) -> str:
    """Build a glob pattern for data files based on underlying and date range."""
    if start_date and end_date:
        start_year = start_date[:4]
        end_year = end_date[:4]
        if start_year == end_year:
            return f"{underlying}_*_{start_year}-*.parquet"
        else:
            years = list(range(int(start_year), int(end_year) + 1))
            year_pattern = "[" + "".join(str(y)[-1] for y in years) + "]"
            return f"{underlying}_*_202{year_pattern}-*.parquet"
    elif start_date:
        start_year = start_date[:4]
        return f"{underlying}_*_{start_year}*.parquet"
    elif end_date:
        return f"{underlying}_*_*.parquet"
    else:
        return f"{underlying}_*.parquet"


def filter_files_by_date(
    files: list,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
) -> list:
    """Filter file list by date range."""
    if not start_date and not end_date:
        return files

    filtered = []
    date_pattern = re.compile(r'(\d{4}-\d{2}-\d{2})')

    for f in files:
        match = date_pattern.search(f.stem)
        if match:
            file_date = match.group(1)
            if start_date and file_date < start_date:
                continue
            if end_date and file_date > end_date:
                continue
            filtered.append(f)

    return filtered


def warmup_cosine_schedule(
    optimizer: AdamW,
    current_step: int,
    warmup_steps: int,
    total_steps: int,
    base_lr: float,
    min_lr: float = 1e-6,
) -> float:
    """Linear warmup followed by cosine decay."""
    if current_step < warmup_steps:
        lr = base_lr * (current_step + 1) / warmup_steps
    else:
        progress = (current_step - warmup_steps) / (total_steps - warmup_steps)
        lr = min_lr + 0.5 * (base_lr - min_lr) * (1 + math.cos(math.pi * progress))

    for param_group in optimizer.param_groups:
        param_group["lr"] = lr

    return lr


# =============================================================================
# Argument Parsing
# =============================================================================


def parse_args() -> argparse.Namespace:
    """Parse command line arguments for all model types."""
    parser = argparse.ArgumentParser(
        description="Unified training script for LeJEPA and entry policies",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Train LeJEPA
    uv run python train.py --model-type jepa --checkpoint-id lejepa-v1 --epochs 10

    # Train 3-class entry policy
    uv run python train.py --model-type entry-3class \\
        --lejepa-checkpoint checkpoints/lejepa/lejepa_best.pt \\
        --checkpoint-id entry-v1 --epochs 50

    # Train 5-class entry policy
    uv run python train.py --model-type entry-5class \\
        --lejepa-checkpoint checkpoints/lejepa/lejepa_best.pt \\
        --checkpoint-id entry-5class-v1 --epochs 50

    # Train directional move entry policy (5-minute smoothed candles)
    uv run python train.py --model-type entry-directional \\
        --lejepa-checkpoint checkpoints/lejepa/lejepa_best.pt \\
        --checkpoint-id entry-directional-v1 --epochs 50 \\
        --candle-agg-minutes 5 --min-candles 3 --max-candles 6
        """,
    )

    # Model type (required)
    parser.add_argument(
        "--model-type",
        type=str,
        required=True,
        choices=["jepa", "entry-3class", "entry-5class", "entry-regression", "entry-directional"],
        help="Type of model to train",
    )

    # Checkpoint management
    parser.add_argument(
        "--checkpoint-id",
        type=str,
        required=True,
        help="Checkpoint ID for the registry",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Checkpoint name/UUID or path to resume from",
    )

    # LeJEPA checkpoint (required for entry policies)
    parser.add_argument(
        "--lejepa-checkpoint",
        type=str,
        default=None,
        help="Path to pre-trained LeJEPA checkpoint (required for entry policies)",
    )

    # Data parameters
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data",
        help="Root data directory containing stocks/, options/, oi/ subdirs",
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
        default=None,
        help="Start date for training data (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--end-date",
        type=str,
        default=None,
        help="End date for training data (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--max-files",
        type=int,
        default=None,
        help="Max parquet files to load (for testing)",
    )
    parser.add_argument(
        "--pattern",
        type=str,
        default=None,
        help="Explicit glob pattern for parquet files (overrides date range)",
    )
    parser.add_argument(
        "--context-len",
        type=int,
        default=90,
        help="Context window length in minutes. 90 = look at last 90 minutes of history.",
    )
    parser.add_argument(
        "--target-len",
        type=int,
        default=30,
        help="Target window length in minutes. 30 = predict next 30 minutes.",
    )
    parser.add_argument(
        "--prediction-gap",
        type=int,
        default=0,
        help="Gap between context end and target start. 0 = predict immediately after context.",
    )
    parser.add_argument(
        "--patch-size",
        type=int,
        default=1,
        help="Token resolution in minutes. 1 = each token is 1 minute (no grouping).",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=1,
        help="Sliding window stride (JEPA only). Use 1 to cover all minute offsets for live inference.",
    )
    parser.add_argument(
        "--include-partial-patches",
        action="store_true",
        default=False,
        help="Include partial patches for early morning data (9:30-10:00am scenario). "
             "Uses DST-aware market open detection to create patches with fewer candles.",
    )
    parser.add_argument(
        "--min-partial-length",
        type=int,
        default=8,
        help="Minimum candles required for partial patches (default: 8)",
    )

    # LeJEPA model parameters
    parser.add_argument(
        "--embedding-dim",
        type=int,
        default=64,
        help="Embedding dimension (latent state size)",
    )
    parser.add_argument(
        "--num-heads",
        type=int,
        default=8,
        help="Number of attention heads",
    )
    parser.add_argument(
        "--num-layers",
        type=int,
        default=6,
        help="Number of transformer layers",
    )
    parser.add_argument(
        "--ff-dim",
        type=int,
        default=2048,
        help="Feed-forward dimension",
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=0.1,
        help="Dropout probability",
    )
    parser.add_argument(
        "--ema-momentum",
        type=float,
        default=0.90,
        help="EMA momentum for target encoder (JEPA). Lower = faster adaptation. 0.90 recommended for financial data.",
    )
    parser.add_argument(
        "--lambda-sigreg",
        type=float,
        default=0.5,
        help="Regularization loss weight (0.5 = equal weight prediction/regularization)",
    )
    parser.add_argument(
        "--reg-type",
        type=str,
        default="vicreg",
        choices=["vicreg", "sigreg"],
        help="Regularization type: 'vicreg' (exact covariance, better for D<=2048) "
             "or 'sigreg' (random projections, better for D>4096). Default: vicreg",
    )
    parser.add_argument(
        "--sigreg-threshold",
        type=float,
        default=60.0,
        help="SIGReg loss threshold for best model selection (JEPA). "
             "Only consider epochs with sigreg_loss < threshold for best model.",
    )
    parser.add_argument(
        "--min-effective-rank-pct",
        type=float,
        default=0.05,
        help="Minimum effective rank as fraction of embedding_dim (0.05 = 5%%). "
             "For financial data, rank 6-12 is realistic (reflects 3-5 market factors). "
             "Models below this threshold are considered collapsed.",
    )

    # Entry policy parameters
    parser.add_argument(
        "--policy-hidden-dim",
        type=int,
        default=256,
        help="Policy hidden dimension",
    )
    parser.add_argument(
        "--policy-layers",
        type=int,
        default=2,
        help="Number of policy layers",
    )

    # Label generation parameters (entry policies)
    parser.add_argument(
        "--lookahead",
        type=int,
        default=15,
        help="Minutes to look ahead for labeling",
    )
    parser.add_argument(
        "--threshold-pct",
        type=float,
        default=0.1,
        help="Price movement threshold %% for 3-class labels",
    )
    parser.add_argument(
        "--scale-factor",
        type=float,
        default=100.0,
        help="Scale factor for regression targets (100 means 1%% move → 1.0 score)",
    )
    parser.add_argument(
        "--min-roi-threshold",
        type=float,
        default=20.0,
        help="Minimum ROI threshold %% for 5-class trade signal",
    )
    parser.add_argument(
        "--otm-buffer",
        type=float,
        default=1.2,
        help="OTM must beat ATM by this factor (5-class)",
    )
    parser.add_argument(
        "--atm-offset",
        type=float,
        default=0.0,
        help="ATM strike offset from spot (5-class)",
    )
    parser.add_argument(
        "--otm-offset",
        type=float,
        default=2.0,
        help="OTM strike offset from ATM in dollars (5-class)",
    )
    parser.add_argument(
        "--base-iv",
        type=float,
        default=0.20,
        help="Base implied volatility for Black-Scholes fallback",
    )
    parser.add_argument(
        "--execution-delay",
        type=int,
        default=1,
        help="Execution delay in minutes (5-class)",
    )
    parser.add_argument(
        "--slippage-pct",
        type=float,
        default=0.5,
        help="Slippage as %% of option price (5-class)",
    )
    parser.add_argument(
        "--no-time-decay",
        action="store_true",
        default=False,
        help="Disable time-decay weighting in ROI labeling (5-class). Use uniform weights.",
    )

    # Directional move labeling parameters (entry-directional)
    parser.add_argument(
        "--candle-agg-minutes",
        type=int,
        default=5,
        help="Minutes to aggregate into each smoothed candle (entry-directional). Default: 5",
    )
    parser.add_argument(
        "--min-candles",
        type=int,
        default=3,
        help="Minimum smoothed candles to check (entry-directional). 3 = 15 minutes. Default: 3",
    )
    parser.add_argument(
        "--max-candles",
        type=int,
        default=6,
        help="Maximum smoothed candles to check (entry-directional). 6 = 30 minutes. Default: 6",
    )
    parser.add_argument(
        "--min-move-pct",
        type=float,
        default=0.05,
        help="Minimum price move %% for directional signal. Default: 0.05 (SPY ~30 cents)",
    )
    parser.add_argument(
        "--min-consistency-pct",
        type=float,
        default=1.0,
        help="Required %% of candles in right direction (1.0 = ALL). Default: 1.0",
    )
    parser.add_argument(
        "--wick-penalty-weight",
        type=float,
        default=0.3,
        help="Weight for wick penalty in score (0-1). Lower wicks = better. Default: 0.3",
    )
    parser.add_argument(
        "--move-size-weight",
        type=float,
        default=0.4,
        help="Weight for move size in score (0-1). Larger moves = better. Default: 0.4",
    )

    # Focal Loss parameters
    parser.add_argument(
        "--use-focal-loss",
        action="store_true",
        default=True,
        help="Use Focal Loss (entry policies)",
    )
    parser.add_argument(
        "--focal-gamma",
        type=float,
        default=2.0,
        help="Focal Loss gamma (focusing parameter)",
    )
    parser.add_argument(
        "--focal-alpha-hold",
        type=float,
        default=0.35,
        help="Focal Loss alpha weight for HOLD class",
    )
    parser.add_argument(
        "--focal-alpha-signal",
        type=float,
        default=1.0,
        help="Focal Loss alpha weight for BUY_CALL/BUY_PUT (3-class)",
    )
    parser.add_argument(
        "--focal-alpha-atm",
        type=float,
        default=1.0,
        help="Focal Loss alpha weight for ATM classes (5-class)",
    )
    parser.add_argument(
        "--focal-alpha-otm",
        type=float,
        default=1.2,
        help="Focal Loss alpha weight for OTM classes (5-class)",
    )

    # Training parameters
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4096,
        help="Batch size. Large batches (4096) improve SIGReg stability.",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=5e-4,
        help="Learning rate (5e-4 recommended for JEPA to escape collapse)",
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=1e-5,
        help="Weight decay (reduced to avoid shrinking weights)",
    )
    parser.add_argument(
        "--warmup-epochs",
        type=int,
        default=10,
        help="Linear warmup epochs for learning rate",
    )
    parser.add_argument(
        "--repr-warmup-epochs",
        type=int,
        default=0,
        help="Representation warmup epochs (SIGReg only, no prediction loss). "
             "Set to 1-2 to force the latent space to expand before learning predictions. "
             "This is the 'Big Bang' strategy to prevent dimensional collapse.",
    )
    parser.add_argument(
        "--train-split",
        type=float,
        default=0.8,
        help="Fraction of data for training",
    )

    # System parameters
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device: 'auto', 'cuda', or 'cpu'",
    )
    parser.add_argument(
        "--compile",
        action="store_true",
        help="Use torch.compile (JEPA only)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help="DataLoader workers",
    )
    parser.add_argument(
        "--cache-workers",
        type=int,
        default=8,
        help="Number of parallel workers for cache building (default: 8)",
    )
    parser.add_argument(
        "--force-rebuild-cache",
        action="store_true",
        help="Force rebuild of data cache even if already cached",
    )

    # Logging and saving
    parser.add_argument(
        "--save-every",
        type=int,
        default=10,
        help="Save checkpoint every N epochs",
    )
    parser.add_argument(
        "--log-every",
        type=int,
        default=10,
        help="Log every N steps/epochs",
    )

    args = parser.parse_args()

    # Validation
    if args.model_type in ["entry-3class", "entry-5class", "entry-regression", "entry-directional"] and not args.lejepa_checkpoint:
        parser.error(f"--lejepa-checkpoint is required for {args.model_type}")

    return args


# =============================================================================
# LeJEPA Training
# =============================================================================


def create_jepa_optimizer(
    model: LeJEPA, lr: float, weight_decay: float
) -> Tuple[AdamW, list]:
    """Create AdamW optimizer with proper parameter groups for LeJEPA."""
    decay_params = []
    no_decay_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if "bias" in name or "norm" in name or "ln" in name:
            no_decay_params.append(param)
        else:
            decay_params.append(param)

    param_groups = [
        {"params": decay_params, "weight_decay": weight_decay},
        {"params": no_decay_params, "weight_decay": 0.0},
    ]

    optimizer = AdamW(param_groups, lr=lr, betas=(0.9, 0.95))
    return optimizer, param_groups


def train_jepa_epoch(
    model: LeJEPA,
    dataloader: DataLoader,
    optimizer: AdamW,
    device: torch.device,
    epoch: int,
    global_step: int,
    warmup_steps: int,
    total_steps: int,
    base_lr: float,
    log_every: int = 10,
    use_amp: bool = True,
) -> Tuple[Dict[str, float], int]:
    """Train LeJEPA for one epoch."""
    model.train()

    total_loss = 0.0
    total_pred_loss = 0.0
    total_sigreg_loss = 0.0
    num_batches = 0

    for batch_idx, batch in enumerate(dataloader):
        if hasattr(torch.compiler, "cudagraph_mark_step_begin"):
            torch.compiler.cudagraph_mark_step_begin()

        # Handle both 2-tuple and 4-tuple batches
        if len(batch) == 4:
            premarket, opening_range, context, target = batch
            premarket = premarket.to(device, non_blocking=True)
            opening_range = opening_range.to(device, non_blocking=True)
            context = context.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
        else:
            context, target = batch
            premarket = None
            opening_range = None
            context = context.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)

        lr = warmup_cosine_schedule(
            optimizer, global_step, warmup_steps, total_steps, base_lr
        )

        optimizer.zero_grad(set_to_none=True)

        if use_amp and device.type == "cuda":
            with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
                output = model(
                    x_context=context,
                    x_target=target,
                    x_opening_range=opening_range,
                    x_premarket=premarket,
                    return_loss=True,
                )
                loss = output["loss"]
        else:
            output = model(
                x_context=context,
                x_target=target,
                x_opening_range=opening_range,
                x_premarket=premarket,
                return_loss=True,
            )
            loss = output["loss"]

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()
        total_pred_loss += output["pred_loss"].item()
        total_sigreg_loss += output["sigreg_loss"].item()
        num_batches += 1
        global_step += 1

        if batch_idx % log_every == 0:
            print(
                f"  Epoch {epoch} | Step {batch_idx}/{len(dataloader)} | "
                f"Loss: {loss.item():.4f} | Pred: {output['pred_loss'].item():.4f} | "
                f"SIGReg: {output['sigreg_loss'].item():.4f} | LR: {lr:.2e}"
            )

    metrics = {
        "loss": total_loss / num_batches,
        "pred_loss": total_pred_loss / num_batches,
        "sigreg_loss": total_sigreg_loss / num_batches,
    }

    return metrics, global_step


@torch.no_grad()
def validate_jepa(
    model: LeJEPA,
    dataloader: DataLoader,
    device: torch.device,
    embedding_dim: int,
    use_amp: bool = True,
) -> Dict[str, float]:
    """Validate LeJEPA model."""
    model.eval()

    total_pred_loss = 0.0
    num_batches = 0
    all_embeddings = []

    for batch in dataloader:
        if hasattr(torch.compiler, "cudagraph_mark_step_begin"):
            torch.compiler.cudagraph_mark_step_begin()

        # Handle both 2-tuple (context, target) and 4-tuple (premarket, opening_range, context, target)
        if len(batch) == 4:
            premarket, opening_range, context, target = batch
            premarket = premarket.to(device, non_blocking=True)
            opening_range = opening_range.to(device, non_blocking=True)
            context = context.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
        else:
            context, target = batch
            context = context.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            premarket = None
            opening_range = None

        if use_amp and device.type == "cuda":
            with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
                output = model(
                    x_context=context,
                    x_target=target,
                    x_opening_range=opening_range,
                    x_premarket=premarket,
                    return_loss=False,
                )
        else:
            output = model(
                x_context=context,
                x_target=target,
                x_opening_range=opening_range,
                x_premarket=premarket,
                return_loss=False,
            )

        pred_loss = torch.nn.functional.mse_loss(
            output["predicted_embedding"], output["target_embedding"]
        )
        total_pred_loss += pred_loss.item()
        num_batches += 1

        all_embeddings.append(output["context_embedding"].float().cpu())

    avg_pred_loss = total_pred_loss / num_batches
    metrics = {
        "val_loss": avg_pred_loss,
        "val_pred_loss": avg_pred_loss,
    }

    embeddings = torch.cat(all_embeddings, dim=0)
    metrics["embedding_std"] = embeddings.std().item()
    metrics["embedding_mean_norm"] = embeddings.norm(dim=1).mean().item()

    dim_variance = embeddings.var(dim=0)
    metrics["dim_variance_mean"] = dim_variance.mean().item()
    metrics["dim_variance_std"] = dim_variance.std().item()

    # Compute effective rank for collapse detection
    # Effective rank = number of eigenvalues > 1% of the largest eigenvalue
    cov = torch.cov(embeddings.T)
    eigenvalues = torch.linalg.eigvalsh(cov)
    eigenvalues = eigenvalues.sort(descending=True).values
    threshold = 0.01 * eigenvalues[0]
    effective_rank = (eigenvalues > threshold).sum().item()
    metrics["effective_rank"] = effective_rank
    metrics["top_eigenvalue"] = eigenvalues[0].item()

    return metrics


def train_jepa(args: argparse.Namespace) -> None:
    """Main LeJEPA training function."""
    print("=" * 70)
    print("LeJEPA Training")
    print("=" * 70)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    set_seed(args.seed)
    device = get_device(args.device)
    print(f"\nDevice: {device}")

    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name()}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # Create dataset
    print("\n" + "=" * 70)
    print("Creating Dataset")
    print("=" * 70)

    data_dir = Path(args.data_dir)
    normalized_dir = data_dir / "training-1m-normalized" / args.underlying

    print(f"Data directory: {data_dir}")
    print(f"Normalized data: {normalized_dir}")
    print(f"Underlying: {args.underlying}")
    if args.start_date:
        print(f"Start date: {args.start_date}")
    if args.end_date:
        print(f"End date: {args.end_date}")

    # Use normalized data (properly scaled, consistent features)
    if normalized_dir.exists():
        print("\n" + "-" * 70)
        print("Loading Normalized Training Data")
        print("-" * 70)

        # Parse date range
        start_date = datetime.strptime(args.start_date, "%Y-%m-%d").date() if args.start_date else None
        end_date = datetime.strptime(args.end_date, "%Y-%m-%d").date() if args.end_date else None

        # Load normalized data
        all_files = []
        for month_dir in sorted(normalized_dir.iterdir()):
            if not month_dir.is_dir():
                continue
            for parquet_file in sorted(month_dir.glob("*.parquet")):
                try:
                    month_str = month_dir.name
                    day_str = parquet_file.stem
                    file_date = datetime.strptime(f"{month_str}-{day_str}", "%Y-%m-%d").date()
                    if start_date and file_date < start_date:
                        continue
                    if end_date and file_date > end_date:
                        continue
                    all_files.append((file_date, parquet_file))
                except ValueError:
                    continue

        all_files.sort(key=lambda x: x[0])
        if args.max_files:
            all_files = all_files[:args.max_files]

        # Define canonical features to use (available in all normalized files)
        canonical_cols = [
            "open", "high", "low", "close", "volume",
            "atm_spread", "net_premium_flow", "implied_volatility",
            "call_strikes_active", "put_strikes_active",
            "atm_call_volume", "otm_call_volume",
            "atm_put_volume", "otm_put_volume",
            "net_gex", "net_dex",
            "time_to_close", "sin_time", "cos_time",
        ]

        # Load and concatenate data
        day_tensors = []
        dates = []
        for file_date, parquet_file in tqdm(all_files, desc="Loading normalized data", unit="day"):
            df = pd.read_parquet(parquet_file)

            # Select canonical columns, filling missing with 0
            selected = pd.DataFrame(index=df.index)
            for col in canonical_cols:
                if col in df.columns:
                    selected[col] = df[col].fillna(0)
                else:
                    selected[col] = 0.0

            # Convert to tensor
            tensor = torch.tensor(selected.values, dtype=torch.float32)
            day_tensors.append(tensor)
            dates.append(file_date)

        print(f"Loaded {len(dates)} days")
        if dates:
            print(f"Date range: {dates[0]} to {dates[-1]}")
            print(f"Features: {len(canonical_cols)}")
            print(f"Candles per day: {day_tensors[0].shape[0]}")

        # Validate data quality
        all_data = torch.cat(day_tensors, dim=0)
        print(f"\nData validation:")
        for i, col in enumerate(canonical_cols):
            vals = all_data[:, i]
            zeros_pct = (vals == 0).float().mean().item() * 100
            std = vals.std().item()
            if zeros_pct > 50 or std < 0.001:
                print(f"  ⚠️  {col}: {zeros_pct:.1f}% zeros, std={std:.4f}")

        # Split into train/val by date
        split_idx = int(len(dates) * args.train_split)
        train_tensors = day_tensors[:split_idx]
        val_tensors = day_tensors[split_idx:]

        print(f"\nTrain: {split_idx} days")
        print(f"Val: {len(dates) - split_idx} days")

        # Create simple sliding window datasets (no premarket for now)
        train_data = torch.cat(train_tensors, dim=0)
        val_data = torch.cat(val_tensors, dim=0)

        train_dataset = JEPASlidingWindowDataset(
            train_data,
            context_len=args.context_len,
            target_len=args.target_len,
            prediction_gap=args.prediction_gap,
            stride=args.stride,
        )
        val_dataset = JEPASlidingWindowDataset(
            val_data,
            context_len=args.context_len,
            target_len=args.target_len,
            prediction_gap=args.prediction_gap,
            stride=args.stride,
        )
    elif not normalized_dir.exists():
        print("\nUsing synthetic data (no normalized data directory found)")
        df = create_test_dataset(num_days=30, bars_per_day=390, random_seed=args.seed)
        split_idx = int(len(df) * args.train_split)
        df_train = df.iloc[:split_idx]
        df_val = df.iloc[split_idx:]

        train_dataset = create_sliding_window_dataset(
            df_train,
            context_len=args.context_len,
            target_len=args.target_len,
            prediction_gap=args.prediction_gap,
            stride=args.stride,
        )
        val_dataset = create_sliding_window_dataset(
            df_val,
            context_len=args.context_len,
            target_len=args.target_len,
            prediction_gap=args.prediction_gap,
            stride=args.stride,
        )

    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")

    train_loader = create_dataloader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = create_dataloader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
        drop_last=False,
    )

    # Initialize model
    print("\n" + "=" * 70)
    print("Initializing Model")
    print("=" * 70)

    # Check dataset type and extract sample
    sample = train_dataset[0]
    if len(sample) == 4:
        # JEPADatasetWithPremarket: (premarket, opening_range, context, target)
        sample_premarket, sample_opening_range, sample_context, sample_target = sample
        has_premarket = True
        premarket_dim = sample_premarket.shape[-1]
        premarket_len = sample_premarket.shape[0]
        opening_range_dim = sample_opening_range.shape[-1]
        print(f"Premarket: {premarket_len} candles × {premarket_dim} features")
        print(f"Opening range: {opening_range_dim} features")
    else:
        # Simple dataset: (context, target)
        sample_context, sample_target = sample
        has_premarket = False
        premarket_dim = 5
        premarket_len = 4
        opening_range_dim = 5

    feature_dim = sample_context.shape[-1]
    context_len = sample_context.shape[0]
    target_len = sample_target.shape[0]
    print(f"Context: {context_len} candles × {feature_dim} features")
    print(f"Target: {target_len} candles × {feature_dim} features")

    model = LeJEPA(
        input_dim=feature_dim,
        premarket_dim=premarket_dim,
        premarket_len=premarket_len,
        opening_range_dim=opening_range_dim,
        d_model=args.ff_dim // 4,  # Transformer dimension (ff_dim/4 for reasonable size)
        nhead=args.num_heads,
        num_layers=args.num_layers,
        embedding_dim=args.embedding_dim,
        max_context_len=context_len,
        dropout=args.dropout,
        lambda_reg=args.lambda_sigreg,
        reg_type=args.reg_type,
    )
    print(f"Regularization: {args.reg_type.upper()}")
    model = model.to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")

    if args.compile and hasattr(torch, "compile"):
        model.transformer = torch.compile(model.transformer, mode="reduce-overhead")
        model.predictor = torch.compile(model.predictor, mode="reduce-overhead")

    optimizer, _ = create_jepa_optimizer(model, args.lr, args.weight_decay)

    steps_per_epoch = len(train_loader)
    total_steps = steps_per_epoch * args.epochs
    warmup_steps = steps_per_epoch * args.warmup_epochs

    # Resume from checkpoint if specified
    start_epoch = 0
    global_step = 0

    # Get checkpoint manager
    project_root = Path(__file__).parent
    mgr = CheckpointManager(project_root / "checkpoints", project_root)

    if args.resume:
        resume_path = args.resume
        resume_entry = mgr.get(args.resume)
        if resume_entry:
            best_path = mgr.get_best_checkpoint(args.resume)
            if best_path:
                resume_path = str(best_path)
            else:
                resume_path = str(resume_entry.path / "lejepa_best.pt")

        print(f"\nResuming from checkpoint: {resume_path}")
        model, checkpoint = LeJEPA.load_checkpoint(
            resume_path, device=str(device), load_optimizer=True, optimizer=optimizer
        )
        start_epoch = checkpoint.get("epoch", 0) + 1
        global_step = checkpoint.get("global_step", 0)
        print(f"Resuming from epoch {start_epoch}, step {global_step}")

    # Register checkpoint in manager
    # Remove model_type and checkpoint_id from config to avoid duplicate keyword argument
    excluded_keys = {"model_type", "checkpoint_id"}
    config = {k: v for k, v in vars(args).items() if k not in excluded_keys}
    entry = mgr.get_or_create(
        args.checkpoint_id,
        model_type="lejepa",
        config=config,
    )
    checkpoint_dir = Path(entry["path"])
    print(f"\nCheckpoint registered: {entry['name']} (ID: {entry['id'][:8]}...)")
    print(f"Checkpoint directory: {checkpoint_dir}")

    # Training loop
    print("\n" + "=" * 70)
    print("Starting Training")
    print("=" * 70)

    # Store original lambda_reg for restoration after warmup
    repr_warmup_epochs = getattr(args, 'repr_warmup_epochs', 0)
    original_lambda_reg = model.lambda_reg

    if repr_warmup_epochs > 0:
        print(f"\n*** REPRESENTATION WARMUP: {repr_warmup_epochs} epoch(s) ***")
        print(f"    During warmup: lambda_reg=100.0 (SIGReg dominates, minimal prediction)")
        print(f"    This forces the latent space to expand before prediction learning")

    best_val_loss = float("inf")
    use_amp = device.type == "cuda"

    for epoch in range(start_epoch, args.epochs):
        # Handle representation warmup phase
        # During warmup: lambda_reg=100.0 (SIGReg dominates)
        # After warmup: restore original lambda_reg
        is_repr_warmup = epoch < repr_warmup_epochs
        if is_repr_warmup:
            model.lambda_reg = 100.0  # SIGReg dominates (100x weight)
            print(f"\n{'='*70}")
            print(f"Epoch {epoch + 1}/{args.epochs} [REPR WARMUP - SIGReg Dominant]")
            print("=" * 70)
        else:
            model.lambda_reg = original_lambda_reg
            print(f"\n{'='*70}")
            print(f"Epoch {epoch + 1}/{args.epochs}")
            print("=" * 70)

        train_metrics, global_step = train_jepa_epoch(
            model=model,
            dataloader=train_loader,
            optimizer=optimizer,
            device=device,
            epoch=epoch + 1,
            global_step=global_step,
            warmup_steps=warmup_steps,
            total_steps=total_steps,
            base_lr=args.lr,
            log_every=args.log_every,
            use_amp=use_amp,
        )

        print(f"\n  Train - Loss: {train_metrics['loss']:.4f} | "
              f"Pred: {train_metrics['pred_loss']:.4f} | "
              f"SIGReg: {train_metrics['sigreg_loss']:.4f}")

        val_metrics = validate_jepa(model, val_loader, device, args.embedding_dim, use_amp=use_amp)

        print(f"  Val   - Loss: {val_metrics['val_loss']:.4f} | "
              f"Pred: {val_metrics['val_pred_loss']:.4f}")
        effective_rank = val_metrics.get("effective_rank", 0)
        print(f"  Embedding Stats - STD: {val_metrics['embedding_std']:.4f} | "
              f"Norm: {val_metrics['embedding_mean_norm']:.4f} | "
              f"EffRank: {effective_rank:.0f}/{args.embedding_dim}")

        # Check for collapse using effective rank (much more reliable than global std)
        min_effective_rank = int(args.embedding_dim * args.min_effective_rank_pct)
        if effective_rank < min_effective_rank:
            print(f"  WARNING: Embedding collapse detected! Effective rank {effective_rank:.0f} < {min_effective_rank} ({args.min_effective_rank_pct:.0%} of {args.embedding_dim})")

        if (epoch + 1) % args.save_every == 0:
            checkpoint_path = checkpoint_dir / f"lejepa_epoch_{epoch + 1:04d}.pt"
            model.save_checkpoint(
                checkpoint_path,
                epoch=epoch + 1,
                optimizer=optimizer,
                global_step=global_step,
                train_metrics=train_metrics,
                val_metrics=val_metrics,
            )

        # Check if embedding geometry is healthy (the collapse safety gate)
        # Use effective rank as the primary indicator - it directly measures
        # how many dimensions are being utilized
        current_variance = val_metrics.get("embedding_std", 0.0)
        effective_rank = val_metrics.get("effective_rank", 0)
        min_effective_rank = int(args.embedding_dim * args.min_effective_rank_pct)
        is_rank_healthy = effective_rank >= min_effective_rank
        is_variance_healthy = (current_variance > 0.5) and (current_variance < 2.0)
        is_healthy = is_rank_healthy and is_variance_healthy
        sigreg_loss = train_metrics.get("sigreg_loss", float("inf"))

        if val_metrics["val_loss"] < best_val_loss:
            if is_healthy:
                best_val_loss = val_metrics["val_loss"]
                best_path = checkpoint_dir / "lejepa_best.pt"
                model.save_checkpoint(
                    best_path,
                    epoch=epoch + 1,
                    optimizer=optimizer,
                    global_step=global_step,
                    train_metrics=train_metrics,
                    val_metrics=val_metrics,
                )
                print(f"  New best model saved! Val loss: {best_val_loss:.4f} (EffRank: {effective_rank:.0f}, STD: {current_variance:.4f})")
            else:
                if not is_rank_healthy:
                    print(f"  Val loss {val_metrics['val_loss']:.4f} < best {best_val_loss:.4f}, but effective rank {effective_rank:.0f} < {min_effective_rank} - COLLAPSED, skipping")
                else:
                    print(f"  Val loss {val_metrics['val_loss']:.4f} < best {best_val_loss:.4f}, but variance {current_variance:.4f} outside [0.5, 2.0] - skipping")

    # Save final model
    final_path = checkpoint_dir / "lejepa_final.pt"
    model.save_checkpoint(
        final_path,
        epoch=args.epochs,
        optimizer=optimizer,
        global_step=global_step,
    )

    # Update checkpoint status and metrics
    mgr.set_training_complete(
        args.checkpoint_id,
        metrics={"best_val_loss": float(best_val_loss)},
        status="trained",
    )

    print("\n" + "=" * 70)
    print("Training Complete!")
    print("=" * 70)
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Final checkpoint: {final_path}")
    print(f"Best checkpoint: {checkpoint_dir / 'lejepa_best.pt'}")
    print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


# =============================================================================
# Entry Policy Datasets
# =============================================================================


class Entry3ClassDataset(Dataset):
    """Dataset for 3-class entry policy training (HOLD, BUY_CALL, BUY_PUT).

    Uses pre-normalized data (same format as LeJEPA training) for feature consistency.
    Labels are computed from raw close prices to get actual % price changes.
    """

    def __init__(
        self,
        normalized_df: pd.DataFrame,
        raw_closes: np.ndarray,
        context_len: int = 32,
        lookahead: int = 10,
        threshold_pct: float = 0.1,
    ):
        """
        Args:
            normalized_df: DataFrame with normalized features (same as LeJEPA input)
            raw_closes: Array of raw close prices (for label computation)
            context_len: Number of timesteps per context window (minutes of history)
            lookahead: Minutes ahead to look for price change
            threshold_pct: Price change threshold for BUY_CALL/BUY_PUT classification
        """
        self.context_len = context_len
        self.lookahead = lookahead
        self.threshold_pct = threshold_pct
        self.raw_closes = raw_closes

        # Convert normalized DataFrame to tensor
        self.data = torch.tensor(normalized_df.values, dtype=torch.float32)
        self.data = torch.nan_to_num(self.data, nan=0.0, posinf=10.0, neginf=-10.0)
        self.data = torch.clamp(self.data, -10, 10)

        # Valid indices: need context_len history and lookahead future
        self.valid_indices = []
        for i in range(context_len, len(normalized_df) - lookahead):
            self.valid_indices.append(i)

        self.labels = self._compute_labels()

    def _compute_labels(self) -> np.ndarray:
        """Compute labels using RAW close prices (not normalized)."""
        labels = []
        for idx in self.valid_indices:
            current_price = self.raw_closes[idx]
            future_price = self.raw_closes[idx + self.lookahead]

            # Guard against division by zero
            if current_price == 0:
                labels.append(EntryActionLegacy.HOLD)
                continue

            pct_change = (future_price - current_price) / current_price * 100

            if pct_change > self.threshold_pct:
                labels.append(EntryActionLegacy.BUY_CALL)
            elif pct_change < -self.threshold_pct:
                labels.append(EntryActionLegacy.BUY_PUT)
            else:
                labels.append(EntryActionLegacy.HOLD)

        return np.array(labels)

    def __len__(self) -> int:
        return len(self.valid_indices)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        data_idx = self.valid_indices[idx]
        start_idx = data_idx - self.context_len
        # Direct slice from pre-normalized tensor (matches LeJEPA input exactly)
        patch = self.data[start_idx:data_idx]  # [context_len, features]
        label = self.labels[idx]
        return patch, label

    def get_class_distribution(self) -> Dict[str, int]:
        unique, counts = np.unique(self.labels, return_counts=True)
        dist = {}
        for action in EntryActionLegacy:
            count = counts[unique == action.value][0] if action.value in unique else 0
            dist[action.name] = int(count)  # Convert numpy int64 to Python int for JSON
        return dist


class EntryRegressionDataset(Dataset):
    """Dataset for continuous regression entry policy training.

    Instead of discrete labels (HOLD/BUY_CALL/BUY_PUT), returns continuous targets
    in [-1, 1] representing direction and conviction:
    - Positive scores = bullish (buy call), magnitude = conviction
    - Negative scores = bearish (buy put), magnitude = conviction
    - Scores near 0 = no signal

    Target: Y = clip(FutureReturn * scale_factor, -1, 1)
    where FutureReturn = (future_price - current_price) / current_price

    A scale_factor of 100 means:
    - 0.5% move → score 0.5
    - 1%+ move → saturates at 1.0 (max conviction)
    """

    def __init__(
        self,
        normalized_df: pd.DataFrame,
        raw_closes: np.ndarray,
        context_len: int = 32,
        lookahead: int = 10,
        scale_factor: float = 100.0,
    ):
        """
        Args:
            normalized_df: DataFrame with normalized features (same as LeJEPA input)
            raw_closes: Array of raw close prices (for target computation)
            context_len: Number of timesteps per context window (minutes of history)
            lookahead: Minutes ahead to look for price change
            scale_factor: Multiplier for returns before clipping (default 100 → 1% = 1.0)
        """
        self.context_len = context_len
        self.lookahead = lookahead
        self.scale_factor = scale_factor
        self.raw_closes = raw_closes

        # Convert normalized DataFrame to tensor
        self.data = torch.tensor(normalized_df.values, dtype=torch.float32)
        self.data = torch.nan_to_num(self.data, nan=0.0, posinf=10.0, neginf=-10.0)
        self.data = torch.clamp(self.data, -10, 10)

        # Valid indices: need context_len history and lookahead future
        self.valid_indices = []
        for i in range(context_len, len(normalized_df) - lookahead):
            self.valid_indices.append(i)

        self.targets = self._compute_targets()

    def _compute_targets(self) -> np.ndarray:
        """Compute continuous targets using RAW close prices."""
        targets = []
        for idx in self.valid_indices:
            current_price = self.raw_closes[idx]
            future_price = self.raw_closes[idx + self.lookahead]

            # Guard against division by zero
            if current_price == 0:
                targets.append(0.0)
                continue

            # Compute return and scale
            pct_return = (future_price - current_price) / current_price
            scaled_return = pct_return * self.scale_factor

            # Clip to [-1, 1]
            target = np.clip(scaled_return, -1.0, 1.0)
            targets.append(target)

        return np.array(targets, dtype=np.float32)

    def __len__(self) -> int:
        return len(self.valid_indices)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, float]:
        data_idx = self.valid_indices[idx]
        start_idx = data_idx - self.context_len
        # Direct slice from pre-normalized tensor (matches LeJEPA input exactly)
        patch = self.data[start_idx:data_idx]  # [context_len, features]
        target = self.targets[idx]
        return patch, target

    def get_target_stats(self) -> Dict[str, float]:
        """Get statistics about the target distribution."""
        return {
            "mean": float(np.mean(self.targets)),
            "std": float(np.std(self.targets)),
            "min": float(np.min(self.targets)),
            "max": float(np.max(self.targets)),
            "bullish_pct": float((self.targets > 0.1).mean() * 100),
            "bearish_pct": float((self.targets < -0.1).mean() * 100),
            "neutral_pct": float((np.abs(self.targets) <= 0.1).mean() * 100),
        }


class RealOptionsROICalculator:
    """Calculates option ROI using real market prices from Polygon data."""

    def __init__(
        self,
        options_dir: str = "data/options",
        base_iv: float = 0.20,
        risk_free_rate: float = 0.05,
        execution_delay_minutes: int = 1,
        slippage_pct: float = 0.5,
    ) -> None:
        self.options_dir = Path(options_dir)
        self.base_iv = base_iv
        self.risk_free_rate = risk_free_rate
        self.execution_delay_minutes = execution_delay_minutes
        self.slippage_pct = slippage_pct

        self._options_cache: Dict[str, pd.DataFrame] = {}
        self._price_lookup_cache: Dict[str, Dict] = {}
        self.real_lookups = 0
        self.fallback_lookups = 0

    def _parse_ticker(self, ticker: str) -> Optional[Tuple[str, str, float]]:
        match = re.match(r'O:SPY(\d{6})([CP])(\d{8})', ticker)
        if match:
            exp_date = match.group(1)
            opt_type = 'call' if match.group(2) == 'C' else 'put'
            strike = int(match.group(3)) / 1000.0
            return exp_date, opt_type, strike
        return None

    def _load_options_for_date(self, date_str: str) -> Optional[pd.DataFrame]:
        if date_str in self._options_cache:
            return self._options_cache[date_str]

        file_path = self.options_dir / f"SPY_{date_str}.parquet"
        if not file_path.exists():
            self._options_cache[date_str] = None
            return None

        try:
            df = pd.read_parquet(file_path)
            parsed = df['ticker'].apply(self._parse_ticker)
            df['exp_date'] = parsed.apply(lambda x: x[0] if x else None)
            df['opt_type'] = parsed.apply(lambda x: x[1] if x else None)
            df['strike'] = parsed.apply(lambda x: x[2] if x else None)

            exp_code = date_str[2:4] + date_str[5:7] + date_str[8:10]
            df = df[df['exp_date'] == exp_code].copy()

            if len(df) == 0:
                self._options_cache[date_str] = None
                return None

            df['timestamp'] = pd.to_datetime(df['window_start'])
            df = df.set_index('timestamp').sort_index()
            self._options_cache[date_str] = df
            self._build_price_lookup(date_str, df)
            return df

        except Exception:
            self._options_cache[date_str] = None
            return None

    def _build_price_lookup(self, date_str: str, df: pd.DataFrame) -> None:
        lookup = {}
        for (strike, opt_type), group in df.groupby(['strike', 'opt_type']):
            key = (strike, opt_type)
            lookup[key] = {
                'open': group['open'].to_dict(),
                'close': group['close'].to_dict(),
            }
        self._price_lookup_cache[date_str] = lookup

    def _get_option_price(
        self,
        date_str: str,
        timestamp: datetime,
        strike: float,
        is_call: bool,
        price_type: str = "open",
        max_time_diff_minutes: int = 2,
    ) -> Optional[float]:
        if date_str not in self._price_lookup_cache:
            self._load_options_for_date(date_str)

        lookup = self._price_lookup_cache.get(date_str)
        if lookup is None:
            return None

        opt_type = 'call' if is_call else 'put'
        key = (strike, opt_type)

        if key not in lookup:
            return None

        price_data = lookup[key]
        prices = price_data.get(price_type, price_data.get('close', {}))

        target_ts = pd.Timestamp(timestamp)
        best_price = None
        best_diff = float('inf')

        for ts, price in prices.items():
            diff = abs((ts - target_ts).total_seconds())
            if diff < best_diff and diff <= max_time_diff_minutes * 60:
                best_diff = diff
                best_price = price

        return best_price

    def _black_scholes_price(
        self,
        S: float,
        K: float,
        T: float,
        is_call: bool,
        sigma: float,
    ) -> float:
        from scipy.stats import norm

        r = self.risk_free_rate
        T = max(T, 1e-8)

        sqrt_T = np.sqrt(T)
        d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * sqrt_T)
        d2 = d1 - sigma * sqrt_T

        if is_call:
            price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
        else:
            price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)

        return max(price, 0.01)

    def calculate_roi(
        self,
        timestamp_decision: datetime,
        timestamp_exit_decision: datetime,
        spot_now: float,
        spot_future: float,
        strike: float,
        is_call: bool,
    ) -> Tuple[float, bool]:
        delay = timedelta(minutes=self.execution_delay_minutes)
        timestamp_entry = timestamp_decision + delay
        timestamp_exit = timestamp_exit_decision + delay
        date_str = timestamp_decision.strftime("%Y-%m-%d")

        price_entry = self._get_option_price(date_str, timestamp_entry, strike, is_call, price_type="open")
        price_exit = self._get_option_price(date_str, timestamp_exit, strike, is_call, price_type="open")

        if price_entry is not None and price_exit is not None and price_entry > 0:
            slippage_mult = self.slippage_pct / 100.0
            price_entry_with_slippage = price_entry * (1 + slippage_mult)
            price_exit_with_slippage = price_exit * (1 - slippage_mult)
            self.real_lookups += 1
            roi = (price_exit_with_slippage - price_entry_with_slippage) / price_entry_with_slippage * 100.0
            return roi, True

        self.fallback_lookups += 1
        market_close_utc = timestamp_entry.replace(hour=21, minute=0, second=0)
        hours_to_close = max((market_close_utc - timestamp_entry).total_seconds() / 3600.0, 0.01)
        lookahead_hours = (timestamp_exit - timestamp_entry).total_seconds() / 3600.0

        T_entry = hours_to_close / (6.5 * 252)
        T_exit = max(hours_to_close - lookahead_hours, 0.01) / (6.5 * 252)

        price_entry = self._black_scholes_price(spot_now, strike, T_entry, is_call, self.base_iv)
        price_exit = self._black_scholes_price(spot_future, strike, T_exit, is_call, self.base_iv)

        slippage_mult = self.slippage_pct / 100.0
        price_entry = price_entry * (1 + slippage_mult)
        price_exit = price_exit * (1 - slippage_mult)

        roi = (price_exit - price_entry) / price_entry * 100.0
        return roi, False

    def get_atm_otm_strikes(
        self,
        spot: float,
        is_call: bool,
        atm_offset: float = 0.0,
        otm_offset: float = 2.0,
    ) -> Tuple[float, float]:
        base_strike = round(spot)
        if is_call:
            atm_strike = base_strike + atm_offset
            otm_strike = atm_strike + otm_offset
        else:
            atm_strike = base_strike - atm_offset
            otm_strike = atm_strike - otm_offset
        return atm_strike, otm_strike

    def get_stats(self) -> Dict[str, int]:
        total = self.real_lookups + self.fallback_lookups
        return {
            "real_lookups": self.real_lookups,
            "fallback_lookups": self.fallback_lookups,
            "total_lookups": total,
            "real_pct": self.real_lookups / total * 100 if total > 0 else 0,
        }


class Entry5ClassDataset(Dataset):
    """Dataset for 5-class entry policy training with ROI-based labeling."""

    def __init__(
        self,
        df: pd.DataFrame,
        patcher: MarketPatch,
        roi_calculator: RealOptionsROICalculator,
        context_len: int = 32,
        lookahead: int = 15,
        min_roi_threshold: float = 20.0,
        otm_buffer: float = 1.2,
        atm_offset: float = 0.0,
        otm_offset: float = 2.0,
        no_time_decay: bool = False,
    ):
        self.df = df
        self.patcher = patcher
        self.roi_calc = roi_calculator
        self.context_len = context_len
        self.lookahead = lookahead
        self.min_roi_threshold = min_roi_threshold
        self.otm_buffer = otm_buffer
        self.atm_offset = atm_offset
        self.otm_offset = otm_offset
        self.no_time_decay = no_time_decay

        self.opens = df["open"].values
        self.closes = df["close"].values
        self.timestamps = df.index.to_pydatetime()

        # Use context_len as max lookahead for time-weighted max ROI
        self.valid_indices = []
        for i in range(context_len, len(df) - context_len):
            self.valid_indices.append(i)

        self.labels = self._compute_labels()

    def _compute_labels(self) -> np.ndarray:
        """Compute labels using time-weighted max ROI within context window.

        For each option type, we check all exit times from 1 to context_len minutes,
        calculate ROI at each time, apply weights, and use the maximum weighted ROI.

        Two weighting factors:
        1. Time decay: earlier exits are valued more
           - Formula: time_weight = 1.0 - 0.5 * (exit_offset - 1) / (context_len - 1)
           - At minute 1: weight = 1.0 (earliest exit, highest value)
           - At minute 32: weight = 0.5 (latest exit, still meaningful)

        2. Directional consistency: reward clean moves in the right direction
           - For calls: % of green candles (close > open) in the window
           - For puts: % of red candles (close < open) in the window
           - Formula: consistency_weight = 0.5 + 0.5 * favorable_pct
           - Range: 0.5 (all candles wrong direction) to 1.0 (all candles right direction)

        Final weighted ROI = raw_roi * time_weight * consistency_weight
        """
        labels = []

        for idx in tqdm(self.valid_indices, desc="Computing time-weighted ROI labels"):
            spot_now = self.closes[idx]
            timestamp_decision = self.timestamps[idx]

            call_atm_strike, call_otm_strike = self.roi_calc.get_atm_otm_strikes(
                spot_now, is_call=True, atm_offset=self.atm_offset, otm_offset=self.otm_offset
            )
            put_atm_strike, put_otm_strike = self.roi_calc.get_atm_otm_strikes(
                spot_now, is_call=False, atm_offset=self.atm_offset, otm_offset=self.otm_offset
            )

            # Track best weighted ROI for each option type
            best_call_atm_weighted = -float('inf')
            best_call_otm_weighted = -float('inf')
            best_put_atm_weighted = -float('inf')
            best_put_otm_weighted = -float('inf')

            # Also track raw ROI at the best weighted point for comparison
            call_atm_roi_at_best = 0.0
            call_otm_roi_at_best = 0.0
            put_atm_roi_at_best = 0.0
            put_otm_roi_at_best = 0.0

            # Check all exit times from 1 to context_len
            for exit_offset in range(1, self.context_len + 1):
                spot_future = self.closes[idx + exit_offset]
                timestamp_exit = self.timestamps[idx + exit_offset]

                # Time decay weight: earlier exits are valued more (1.0 at minute 1, 0.5 at minute 32)
                # When no_time_decay is True, use uniform weight of 1.0 for all exit times
                if self.no_time_decay:
                    time_weight = 1.0
                    call_consistency = 1.0
                    put_consistency = 1.0
                else:
                    time_weight = 1.0 - 0.5 * (exit_offset - 1) / (self.context_len - 1)

                    # Directional consistency: count green/red candles in the window
                    window_opens = self.opens[idx + 1 : idx + exit_offset + 1]
                    window_closes = self.closes[idx + 1 : idx + exit_offset + 1]
                    green_candles = np.sum(window_closes > window_opens)
                    red_candles = np.sum(window_closes < window_opens)
                    # Calls benefit from green candles, puts benefit from red candles
                    call_consistency = 0.5 + 0.5 * (green_candles / exit_offset)
                    put_consistency = 0.5 + 0.5 * (red_candles / exit_offset)

                call_atm_roi, _ = self.roi_calc.calculate_roi(
                    timestamp_decision, timestamp_exit, spot_now, spot_future, call_atm_strike, is_call=True
                )
                call_otm_roi, _ = self.roi_calc.calculate_roi(
                    timestamp_decision, timestamp_exit, spot_now, spot_future, call_otm_strike, is_call=True
                )
                put_atm_roi, _ = self.roi_calc.calculate_roi(
                    timestamp_decision, timestamp_exit, spot_now, spot_future, put_atm_strike, is_call=False
                )
                put_otm_roi, _ = self.roi_calc.calculate_roi(
                    timestamp_decision, timestamp_exit, spot_now, spot_future, put_otm_strike, is_call=False
                )

                # Calculate weighted ROI (time_weight * consistency_weight)
                call_atm_weighted = call_atm_roi * time_weight * call_consistency
                call_otm_weighted = call_otm_roi * time_weight * call_consistency
                put_atm_weighted = put_atm_roi * time_weight * put_consistency
                put_otm_weighted = put_otm_roi * time_weight * put_consistency

                # Update best if this exit time gives better weighted ROI
                if call_atm_weighted > best_call_atm_weighted:
                    best_call_atm_weighted = call_atm_weighted
                    call_atm_roi_at_best = call_atm_roi
                if call_otm_weighted > best_call_otm_weighted:
                    best_call_otm_weighted = call_otm_weighted
                    call_otm_roi_at_best = call_otm_roi
                if put_atm_weighted > best_put_atm_weighted:
                    best_put_atm_weighted = put_atm_weighted
                    put_atm_roi_at_best = put_atm_roi
                if put_otm_weighted > best_put_otm_weighted:
                    best_put_otm_weighted = put_otm_weighted
                    put_otm_roi_at_best = put_otm_roi

            # Use weighted ROI for comparison and threshold checking
            best_call_weighted = max(best_call_atm_weighted, best_call_otm_weighted)
            call_prefers_otm = best_call_otm_weighted > best_call_atm_weighted * self.otm_buffer

            best_put_weighted = max(best_put_atm_weighted, best_put_otm_weighted)
            put_prefers_otm = best_put_otm_weighted > best_put_atm_weighted * self.otm_buffer

            # Apply threshold to weighted ROI
            if best_call_weighted < self.min_roi_threshold and best_put_weighted < self.min_roi_threshold:
                label = EntryAction.HOLD
            elif best_call_weighted >= best_put_weighted:
                if best_call_weighted >= self.min_roi_threshold:
                    label = EntryAction.BUY_CALL_OTM if call_prefers_otm else EntryAction.BUY_CALL_ATM
                else:
                    label = EntryAction.HOLD
            else:
                if best_put_weighted >= self.min_roi_threshold:
                    label = EntryAction.BUY_PUT_OTM if put_prefers_otm else EntryAction.BUY_PUT_ATM
                else:
                    label = EntryAction.HOLD

            labels.append(label)

        return np.array(labels)

    def __len__(self) -> int:
        return len(self.valid_indices)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        data_idx = self.valid_indices[idx]
        start_idx = data_idx - self.context_len
        patch = self.patcher.create_patch(self.df, start_idx)
        label = self.labels[idx]
        return patch, label

    def get_class_distribution(self) -> Dict[str, int]:
        unique, counts = np.unique(self.labels, return_counts=True)
        dist = {}
        for action in EntryAction:
            count = counts[unique == action.value][0] if action.value in unique else 0
            dist[action.name] = count
        return dist


class DirectionalMoveDataset(Dataset):
    """Dataset for entry policy training using directional move labeling.

    This labeling scheme focuses on finding moments before clean, directional moves by:
    1. Using 5-minute smoothed candles (aggregate 5 consecutive 1-minute bars)
    2. Looking at the next 3 or 6 5-minute candles (15 or 30 minutes prediction)
    3. Requiring ALL candles to trend in the direction of the play
    4. First 50% of the move must be immediately in the right direction
    5. Rewarding larger moves and penalizing wicky (choppy) candles

    The label score considers:
    - Move size: Larger price movements get higher scores
    - Directional consistency: All N candles must close in the expected direction
    - Immediate direction: First half of candles must show immediate momentum
    - Wick penalty: Shorter wicks (cleaner candles) are rewarded

    IMPORTANT: This dataset requires TWO data sources:
    - normalized_df: Normalized data for creating patches (fed to LeJEPA)
    - raw_ohlcv: Raw OHLCV prices for labeling (determining good entries)
    """

    def __init__(
        self,
        normalized_df: pd.DataFrame,
        raw_ohlcv: Dict[str, np.ndarray],
        patcher: MarketPatch,
        roi_calculator: RealOptionsROICalculator,
        context_len: int = 90,
        candle_agg_minutes: int = 5,
        min_candles: int = 3,
        max_candles: int = 6,
        min_move_pct: float = 0.05,  # 0.05% for SPY is ~30 cents, sufficient for 0DTE ROI
        min_consistency_pct: float = 1.0,  # 1.0 = ALL candles must be consistent
        wick_penalty_weight: float = 0.3,
        move_size_weight: float = 0.4,
        atm_offset: float = 0.0,
        otm_offset: float = 2.0,
        otm_buffer: float = 1.2,
        min_roi_threshold: float = 10.0,  # 10% ROI for directional moves
    ):
        """
        Args:
            normalized_df: Normalized DataFrame for creating patches (LeJEPA input)
            raw_ohlcv: Dict with 'open', 'high', 'low', 'close' arrays of raw prices for labeling
            patcher: MarketPatch for creating input patches
            roi_calculator: Calculator for options ROI
            context_len: Context window length in minutes (default: 90)
            candle_agg_minutes: Minutes to aggregate into each smoothed candle (default: 5)
            min_candles: Minimum smoothed candles to check (default: 3 = 15 min)
            max_candles: Maximum smoothed candles to check (default: 6 = 30 min)
            min_move_pct: Minimum price move % to consider a signal (default: 0.3%)
            min_consistency_pct: Required % of candles in right direction (default: 1.0 = 100%)
            wick_penalty_weight: Weight for wick penalty in score (default: 0.3)
            move_size_weight: Weight for move size in score (default: 0.4)
            atm_offset: ATM strike offset from spot (default: 0.0)
            otm_offset: OTM strike offset from ATM (default: 2.0)
            otm_buffer: Multiplier for OTM preference (default: 1.2)
            min_roi_threshold: Minimum ROI % to label as non-HOLD (default: 15.0)
        """
        self.df = normalized_df  # For patches
        self.patcher = patcher
        self.roi_calc = roi_calculator
        self.context_len = context_len
        self.candle_agg_minutes = candle_agg_minutes
        self.min_candles = min_candles
        self.max_candles = max_candles
        self.min_move_pct = min_move_pct
        self.min_consistency_pct = min_consistency_pct
        self.wick_penalty_weight = wick_penalty_weight
        self.move_size_weight = move_size_weight
        self.atm_offset = atm_offset
        self.otm_offset = otm_offset
        self.otm_buffer = otm_buffer
        self.min_roi_threshold = min_roi_threshold

        # Raw OHLCV arrays for labeling (NOT normalized)
        self.opens = raw_ohlcv["open"]
        self.highs = raw_ohlcv["high"]
        self.lows = raw_ohlcv["low"]
        self.closes = raw_ohlcv["close"]
        self.timestamps = normalized_df.index.to_pydatetime()

        # Calculate lookahead needed (max_candles * candle_agg_minutes)
        self.lookahead = max_candles * candle_agg_minutes

        # Valid indices need enough room for context and lookahead
        self.valid_indices = []
        for i in range(context_len, len(self.df) - self.lookahead):
            self.valid_indices.append(i)

        self.labels = self._compute_labels()
        self.scores = self._compute_scores()

    def _aggregate_candles(self, start_idx: int, num_candles: int) -> List[Dict]:
        """Aggregate 1-minute bars into larger candles.

        Args:
            start_idx: Starting index in the 1-minute data
            num_candles: Number of aggregated candles to create

        Returns:
            List of dicts with aggregated OHLC and wick metrics
        """
        candles = []
        for i in range(num_candles):
            candle_start = start_idx + i * self.candle_agg_minutes
            candle_end = candle_start + self.candle_agg_minutes

            if candle_end > len(self.opens):
                break

            o = self.opens[candle_start]
            h = np.max(self.highs[candle_start:candle_end])
            l = np.min(self.lows[candle_start:candle_end])
            c = self.closes[candle_end - 1]

            # Calculate wick ratio (smaller = cleaner candle)
            body_size = abs(c - o)
            candle_range = h - l if h > l else 0.0001  # Avoid division by zero
            upper_wick = h - max(o, c)
            lower_wick = min(o, c) - l
            wick_ratio = (upper_wick + lower_wick) / candle_range if candle_range > 0 else 1.0

            candles.append({
                "open": o,
                "high": h,
                "low": l,
                "close": c,
                "is_green": c > o,
                "is_red": c < o,
                "body_pct": body_size / o * 100 if o > 0 else 0,
                "wick_ratio": wick_ratio,
                "move_pct": (c - o) / o * 100 if o > 0 else 0,
            })

        return candles

    def _evaluate_direction(
        self,
        candles: List[Dict],
        is_call: bool,
    ) -> Tuple[bool, float, float, float]:
        """Evaluate if candles show a clean directional move.

        Args:
            candles: List of aggregated candles
            is_call: True for bullish (call), False for bearish (put)

        Returns:
            Tuple of:
            - is_valid: True if move meets all criteria
            - consistency_score: Fraction of candles in right direction
            - wick_score: Average wick quality (1 = no wicks, 0 = all wicks)
            - move_size: Total price move percentage
        """
        if not candles:
            return False, 0.0, 0.0, 0.0

        num_candles = len(candles)

        # Count candles in expected direction
        if is_call:
            consistent_candles = sum(1 for c in candles if c["is_green"])
        else:
            consistent_candles = sum(1 for c in candles if c["is_red"])

        consistency_score = consistent_candles / num_candles

        # Check if ALL required candles are consistent
        if consistency_score < self.min_consistency_pct:
            return False, consistency_score, 0.0, 0.0

        # Check first 50% are immediately in the right direction
        half_idx = max(1, num_candles // 2)
        first_half = candles[:half_idx]

        if is_call:
            first_half_consistent = all(c["is_green"] for c in first_half)
        else:
            first_half_consistent = all(c["is_red"] for c in first_half)

        if not first_half_consistent:
            return False, consistency_score, 0.0, 0.0

        # Calculate average wick score (1 - wick_ratio, so lower wicks = higher score)
        avg_wick_ratio = np.mean([c["wick_ratio"] for c in candles])
        wick_score = 1.0 - avg_wick_ratio  # Invert so cleaner candles = higher score

        # Calculate total move size
        start_price = candles[0]["open"]
        end_price = candles[-1]["close"]
        if is_call:
            move_size = (end_price - start_price) / start_price * 100
        else:
            move_size = (start_price - end_price) / start_price * 100

        # Move must be positive (in the expected direction)
        if move_size < self.min_move_pct:
            return False, consistency_score, wick_score, move_size

        return True, consistency_score, wick_score, move_size

    def _compute_labels(self) -> np.ndarray:
        """Compute labels based on directional move criteria.

        For each decision point:
        1. Aggregate next N 5-minute candles (try 6, then 5, 4, 3)
        2. Check if ALL candles trend in the direction
        3. Verify first 50% move immediately in direction
        4. Calculate combined score from move size and wick quality
        5. Assign label based on best direction with sufficient score
        """
        labels = []

        for idx in tqdm(self.valid_indices, desc="Computing directional move labels"):
            spot_now = self.closes[idx]
            timestamp_decision = self.timestamps[idx]

            # Track best scores for each direction
            best_call_score = 0.0
            best_put_score = 0.0
            best_call_candles = 0
            best_put_candles = 0
            call_roi = 0.0
            put_roi = 0.0

            # Try different lookahead windows (6, 5, 4, 3 candles)
            for num_candles in range(self.max_candles, self.min_candles - 1, -1):
                candles = self._aggregate_candles(idx + 1, num_candles)

                if len(candles) < num_candles:
                    continue

                # Evaluate call (bullish) direction
                is_valid_call, call_consistency, call_wick, call_move = self._evaluate_direction(
                    candles, is_call=True
                )
                if is_valid_call:
                    # Calculate combined score
                    call_score = (
                        self.move_size_weight * min(call_move / 1.0, 1.0) +  # Normalize move to 1%
                        (1 - self.move_size_weight - self.wick_penalty_weight) * call_consistency +
                        self.wick_penalty_weight * call_wick
                    )
                    # Bonus for more candles being consistent
                    call_score *= (1 + 0.1 * (num_candles - self.min_candles))

                    if call_score > best_call_score:
                        best_call_score = call_score
                        best_call_candles = num_candles

                # Evaluate put (bearish) direction
                is_valid_put, put_consistency, put_wick, put_move = self._evaluate_direction(
                    candles, is_call=False
                )
                if is_valid_put:
                    put_score = (
                        self.move_size_weight * min(put_move / 1.0, 1.0) +
                        (1 - self.move_size_weight - self.wick_penalty_weight) * put_consistency +
                        self.wick_penalty_weight * put_wick
                    )
                    put_score *= (1 + 0.1 * (num_candles - self.min_candles))

                    if put_score > best_put_score:
                        best_put_score = put_score
                        best_put_candles = num_candles

            # Calculate actual ROI if we have valid signals
            if best_call_score > 0 or best_put_score > 0:
                # Use the best candle count for ROI calculation
                exit_minutes = max(best_call_candles, best_put_candles, self.min_candles) * self.candle_agg_minutes
                spot_future = self.closes[min(idx + exit_minutes, len(self.closes) - 1)]
                timestamp_exit = self.timestamps[min(idx + exit_minutes, len(self.timestamps) - 1)]

                call_atm_strike, call_otm_strike = self.roi_calc.get_atm_otm_strikes(
                    spot_now, is_call=True, atm_offset=self.atm_offset, otm_offset=self.otm_offset
                )
                put_atm_strike, put_otm_strike = self.roi_calc.get_atm_otm_strikes(
                    spot_now, is_call=False, atm_offset=self.atm_offset, otm_offset=self.otm_offset
                )

                if best_call_score > 0:
                    call_atm_roi, _ = self.roi_calc.calculate_roi(
                        timestamp_decision, timestamp_exit, spot_now, spot_future, call_atm_strike, is_call=True
                    )
                    call_otm_roi, _ = self.roi_calc.calculate_roi(
                        timestamp_decision, timestamp_exit, spot_now, spot_future, call_otm_strike, is_call=True
                    )
                    call_roi = max(call_atm_roi, call_otm_roi)
                    call_prefers_otm = call_otm_roi > call_atm_roi * self.otm_buffer

                if best_put_score > 0:
                    put_atm_roi, _ = self.roi_calc.calculate_roi(
                        timestamp_decision, timestamp_exit, spot_now, spot_future, put_atm_strike, is_call=False
                    )
                    put_otm_roi, _ = self.roi_calc.calculate_roi(
                        timestamp_decision, timestamp_exit, spot_now, spot_future, put_otm_strike, is_call=False
                    )
                    put_roi = max(put_atm_roi, put_otm_roi)
                    put_prefers_otm = put_otm_roi > put_atm_roi * self.otm_buffer

            # Determine label based on score and ROI threshold
            if best_call_score == 0 and best_put_score == 0:
                label = EntryAction.HOLD
            elif best_call_score >= best_put_score:
                if call_roi >= self.min_roi_threshold:
                    label = EntryAction.BUY_CALL_OTM if call_prefers_otm else EntryAction.BUY_CALL_ATM
                else:
                    label = EntryAction.HOLD
            else:
                if put_roi >= self.min_roi_threshold:
                    label = EntryAction.BUY_PUT_OTM if put_prefers_otm else EntryAction.BUY_PUT_ATM
                else:
                    label = EntryAction.HOLD

            labels.append(label)

        return np.array(labels)

    def _compute_scores(self) -> np.ndarray:
        """Compute continuous scores for each sample (useful for regression)."""
        # Placeholder - scores computed during label computation could be stored here
        return np.zeros(len(self.valid_indices))

    def __len__(self) -> int:
        return len(self.valid_indices)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        data_idx = self.valid_indices[idx]
        start_idx = data_idx - self.context_len
        patch = self.patcher.create_patch(self.df, start_idx)
        label = self.labels[idx]
        return patch, label

    def get_class_distribution(self) -> Dict[str, int]:
        unique, counts = np.unique(self.labels, return_counts=True)
        dist = {}
        for action in EntryAction:
            count = counts[unique == action.value][0] if action.value in unique else 0
            dist[action.name] = count
        return dist


class EmbeddingDataset(Dataset):
    """Dataset wrapper that pre-computes LeJEPA embeddings."""

    def __init__(
        self,
        base_dataset: Dataset,
        lejepa: LeJEPA,
        device: torch.device,
        batch_size: int = 512,
    ):
        self.labels = torch.tensor(base_dataset.labels, dtype=torch.long)

        print(f"  Pre-computing {len(base_dataset)} embeddings...")

        embeddings = []
        loader = DataLoader(base_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

        lejepa.eval()
        with torch.no_grad():
            for patches, _ in tqdm(loader, desc="  Computing embeddings"):
                patches = patches.to(device)
                with torch.amp.autocast(device_type=device.type, dtype=torch.bfloat16):
                    emb = lejepa.encode(patches)
                embeddings.append(emb.float().cpu())

        self.embeddings = torch.cat(embeddings, dim=0)
        print(f"  Embeddings shape: {self.embeddings.shape}")

    def __len__(self) -> int:
        return len(self.embeddings)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        return self.embeddings[idx], self.labels[idx].item()


class RegressionEmbeddingDataset(Dataset):
    """Dataset wrapper that pre-computes LeJEPA embeddings for regression."""

    def __init__(
        self,
        base_dataset: EntryRegressionDataset,
        lejepa: LeJEPA,
        device: torch.device,
        batch_size: int = 512,
    ):
        self.targets = torch.tensor(base_dataset.targets, dtype=torch.float32)

        print(f"  Pre-computing {len(base_dataset)} embeddings...")

        embeddings = []
        loader = DataLoader(base_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

        lejepa.eval()
        with torch.no_grad():
            for patches, _ in tqdm(loader, desc="  Computing embeddings"):
                patches = patches.to(device)
                with torch.amp.autocast(device_type=device.type, dtype=torch.bfloat16):
                    emb = lejepa.encode(patches)
                embeddings.append(emb.float().cpu())

        self.embeddings = torch.cat(embeddings, dim=0)
        print(f"  Embeddings shape: {self.embeddings.shape}")

    def __len__(self) -> int:
        return len(self.embeddings)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, float]:
        return self.embeddings[idx], self.targets[idx].item()


# =============================================================================
# Entry Policy Training (Classification)
# =============================================================================


def train_entry_epoch(
    policy: EntryPolicy,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    num_classes: int,
    max_grad_norm: float = 1.0,
) -> Dict[str, float]:
    """Train entry policy for one epoch."""
    policy.train()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    action_correct = {a: 0 for a in EntryAction if a.value < num_classes}
    action_total = {a: 0 for a in EntryAction if a.value < num_classes}

    for embeddings, labels in loader:
        embeddings = embeddings.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        output = policy(embeddings)
        logits = output["action_logits"]
        loss = criterion(logits, labels)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), max_grad_norm)
        optimizer.step()

        total_loss += loss.item() * len(labels)
        preds = logits.argmax(dim=-1)
        total_correct += (preds == labels).sum().item()
        total_samples += len(labels)

        for action in action_correct.keys():
            mask = labels == action.value
            action_total[action] += mask.sum().item()
            action_correct[action] += ((preds == labels) & mask).sum().item()

    avg_loss = total_loss / total_samples
    accuracy = total_correct / total_samples

    metrics = {
        "train_loss": avg_loss,
        "train_accuracy": accuracy,
    }

    for action in action_correct.keys():
        if action_total[action] > 0:
            acc = action_correct[action] / action_total[action]
            metrics[f"train_acc_{action.name.lower()}"] = acc

    return metrics


def validate_entry(
    policy: EntryPolicy,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    num_classes: int,
) -> Dict[str, float]:
    """Validate entry policy."""
    policy.eval()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    action_correct = {a: 0 for a in EntryAction if a.value < num_classes}
    action_total = {a: 0 for a in EntryAction if a.value < num_classes}
    action_predicted = {a: 0 for a in EntryAction if a.value < num_classes}

    with torch.no_grad():
        for embeddings, labels in loader:
            embeddings = embeddings.to(device)
            labels = labels.to(device)

            output = policy(embeddings)
            logits = output["action_logits"]

            loss = criterion(logits, labels)
            total_loss += loss.item() * len(labels)

            preds = logits.argmax(dim=-1)
            total_correct += (preds == labels).sum().item()
            total_samples += len(labels)

            for action in action_correct.keys():
                mask = labels == action.value
                pred_mask = preds == action.value
                action_total[action] += mask.sum().item()
                action_correct[action] += ((preds == labels) & mask).sum().item()
                action_predicted[action] += pred_mask.sum().item()

    avg_loss = total_loss / total_samples
    accuracy = total_correct / total_samples

    metrics = {
        "val_loss": avg_loss,
        "val_accuracy": accuracy,
    }

    for action in action_correct.keys():
        if action_total[action] > 0:
            recall = action_correct[action] / action_total[action]
            metrics[f"val_recall_{action.name.lower()}"] = recall

        if action_predicted[action] > 0:
            precision = action_correct[action] / action_predicted[action]
            metrics[f"val_precision_{action.name.lower()}"] = precision

        metrics[f"val_pred_{action.name.lower()}"] = action_predicted[action]

    return metrics


def save_entry_checkpoint(
    policy: EntryPolicy,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    metrics: Dict[str, float],
    path: Path,
    embedding_dim: int,
    args: argparse.Namespace,
) -> None:
    """Save entry policy checkpoint."""
    path.parent.mkdir(parents=True, exist_ok=True)

    num_classes = 5 if args.model_type == "entry-5class" else 3

    checkpoint = {
        "epoch": epoch,
        "policy_state_dict": policy.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "metrics": metrics,
        "config": {
            "embedding_dim": embedding_dim,
            "hidden_dim": policy.hidden_dim,
            "num_actions": policy.num_actions,
            "num_layers": policy.num_layers,
            "num_classes": num_classes,
            "model_type": args.model_type,
            "lookahead": args.lookahead,
        },
    }

    if args.model_type == "entry-5class":
        checkpoint["config"].update({
            "min_roi_threshold": args.min_roi_threshold,
            "otm_buffer": args.otm_buffer,
        })

    torch.save(checkpoint, path)
    print(f"  Saved checkpoint to {path}")


def train_entry(args: argparse.Namespace) -> None:
    """Main entry policy training function."""
    num_classes = 5 if args.model_type == "entry-5class" else 3
    class_name = "5-Class" if num_classes == 5 else "3-Class"

    print("=" * 70)
    print(f"{class_name} Entry Policy Training")
    print("=" * 70)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    if num_classes == 5:
        print(f"\nClasses: HOLD, CALL_ATM, CALL_OTM, PUT_ATM, PUT_OTM")
        print(f"Label generation: ROI comparison with {args.lookahead}min lookahead")
        print(f"Min ROI threshold: {args.min_roi_threshold}%")
        print(f"OTM buffer: {args.otm_buffer}x")
    else:
        print(f"\nClasses: HOLD, BUY_CALL, BUY_PUT")
        print(f"Label generation: Price threshold with {args.lookahead}min lookahead")
        print(f"Threshold: {args.threshold_pct}%")

    set_seed(args.seed)
    device = get_device(args.device)
    print(f"\nDevice: {device}")

    # Load pre-trained LeJEPA
    print("\n" + "=" * 70)
    print("Loading Pre-trained LeJEPA")
    print("=" * 70)

    lejepa, checkpoint = LeJEPA.load_checkpoint(args.lejepa_checkpoint, device=str(device))
    lejepa = lejepa.to(device)
    lejepa.eval()

    for param in lejepa.parameters():
        param.requires_grad = False

    embedding_dim = lejepa.embedding_dim
    print(f"Embedding dimension: {embedding_dim}")

    # Load data using same pipeline as LeJEPA training
    print("\n" + "=" * 70)
    print("Loading Training Data")
    print("=" * 70)

    data_dir = Path(args.data_dir)
    stocks_dir = data_dir / "stocks-1m"
    options_dir = data_dir / "options-1m"

    print(f"Loading combined stocks + options data")
    print(f"Stocks dir: {stocks_dir}")
    print(f"Options dir: {options_dir}")
    print(f"Underlying: {args.underlying}")
    if args.start_date:
        print(f"Start date: {args.start_date}")
    if args.end_date:
        print(f"End date: {args.end_date}")

    # Load normalized data (same as LeJEPA training input)
    df = load_normalized_data(
        stocks_dir=str(stocks_dir),
        options_dir=str(options_dir),
        underlying=args.underlying,
        start_date=args.start_date,
        end_date=args.end_date,
        max_files=args.max_files,
    )

    print(f"Loaded {len(df):,} bars from {df.index.min()} to {df.index.max()}")

    # Load raw close prices from cache for label computation
    print("Loading raw close prices for label computation...")
    start = parse_date(args.start_date)
    end = parse_date(args.end_date)

    raw_cache_dir = Path(RAW_CACHE_DIR)
    raw_files = sorted(raw_cache_dir.glob(f"{args.underlying}_raw_*.parquet"))
    raw_closes_list = []

    for rf in raw_files:
        date_str = rf.stem.split("_")[-1]  # SPY_raw_2021-01-04 -> 2021-01-04
        try:
            file_date = datetime.strptime(date_str, "%Y-%m-%d").date()
            if start and file_date < start:
                continue
            if end and file_date > end:
                continue
            raw_df = pd.read_parquet(rf)
            raw_closes_list.append(raw_df["close"].values)
        except (ValueError, KeyError):
            continue

    raw_closes = np.concatenate(raw_closes_list) if raw_closes_list else np.zeros(len(df))

    # Verify alignment
    if len(raw_closes) != len(df):
        print(f"Warning: raw closes ({len(raw_closes)}) != normalized ({len(df)}), truncating...")
        min_len = min(len(raw_closes), len(df))
        raw_closes = raw_closes[:min_len]
        df = df.iloc[:min_len]

    # Filter to market hours (UTC: 14:30-21:00)
    market_open_utc = time(14, 30)
    market_close_utc = time(21, 0)

    # Apply market hours filter and track indices
    mask = (df.index.time >= market_open_utc) & (df.index.time < market_close_utc)
    mask_array = np.array(mask)  # Convert to numpy for indexing raw_closes
    df = df[mask]
    raw_closes = raw_closes[mask_array]

    print(f"After market hours filter (UTC): {len(df):,} bars")

    # Split into days
    df["date"] = df.index.date
    days = df["date"].unique()

    n_days = len(days)
    n_train_days = int(n_days * args.train_split)
    train_days = days[:n_train_days]
    val_days = days[n_train_days:]

    print(f"Total days: {n_days}")
    print(f"Training days: {len(train_days)}")
    print(f"Validation days: {len(val_days)}")

    # Create train/val masks
    train_mask = df["date"].isin(train_days)
    val_mask = df["date"].isin(val_days)

    train_df = df[train_mask].drop(columns=["date"])
    val_df = df[val_mask].drop(columns=["date"])
    train_raw_closes = raw_closes[train_mask.values]
    val_raw_closes = raw_closes[val_mask.values]

    print(f"\nContext length: {args.context_len}")
    print(f"Feature dimension: {len(train_df.columns)}")

    # Create datasets
    print("\n" + "=" * 70)
    print(f"Creating {class_name} Datasets")
    print("=" * 70)

    if num_classes == 5:
        # 5-class still uses MarketPatch (needs refactoring to match 3-class)
        base_cols = {"open", "high", "low", "close", "volume"}
        extra_columns = [col for col in train_df.columns if col not in base_cols]
        patcher = MarketPatch(patch_length=args.context_len, extra_columns=extra_columns)

        roi_calculator = RealOptionsROICalculator(
            options_dir=str(options_dir),
            base_iv=args.base_iv,
            execution_delay_minutes=args.execution_delay,
            slippage_pct=args.slippage_pct,
        )

        print(f"Using REAL options prices from: {options_dir}")
        print(f"Execution: OPEN price of T+{args.execution_delay} candle")
        print(f"Slippage: {args.slippage_pct}%")

        train_base_dataset = Entry5ClassDataset(
            df=train_df,
            patcher=patcher,
            roi_calculator=roi_calculator,
            context_len=args.context_len,
            lookahead=args.lookahead,
            min_roi_threshold=args.min_roi_threshold,
            otm_buffer=args.otm_buffer,
            atm_offset=args.atm_offset,
            otm_offset=args.otm_offset,
            no_time_decay=args.no_time_decay,
        )

        val_base_dataset = Entry5ClassDataset(
            df=val_df,
            patcher=patcher,
            roi_calculator=roi_calculator,
            context_len=args.context_len,
            lookahead=args.lookahead,
            min_roi_threshold=args.min_roi_threshold,
            otm_buffer=args.otm_buffer,
            atm_offset=args.atm_offset,
            otm_offset=args.otm_offset,
            no_time_decay=args.no_time_decay,
        )

        stats = roi_calculator.get_stats()
        print(f"\nROI Price Lookup Statistics:")
        print(f"  Real options prices: {stats['real_lookups']:,} ({stats['real_pct']:.1f}%)")
        print(f"  Black-Scholes fallback: {stats['fallback_lookups']:,}")
    else:
        train_base_dataset = Entry3ClassDataset(
            normalized_df=train_df,
            raw_closes=train_raw_closes,
            context_len=args.context_len,
            lookahead=args.lookahead,
            threshold_pct=args.threshold_pct,
        )

        val_base_dataset = Entry3ClassDataset(
            normalized_df=val_df,
            raw_closes=val_raw_closes,
            context_len=args.context_len,
            lookahead=args.lookahead,
            threshold_pct=args.threshold_pct,
        )

    print(f"\nTraining samples: {len(train_base_dataset):,}")
    train_dist = train_base_dataset.get_class_distribution()
    for action_name, count in train_dist.items():
        pct = count / len(train_base_dataset) * 100 if len(train_base_dataset) > 0 else 0
        print(f"  {action_name}: {count:,} ({pct:.1f}%)")

    print(f"\nValidation samples: {len(val_base_dataset):,}")
    val_dist = val_base_dataset.get_class_distribution()
    for action_name, count in val_dist.items():
        pct = count / len(val_base_dataset) * 100 if len(val_base_dataset) > 0 else 0
        print(f"  {action_name}: {count:,} ({pct:.1f}%)")

    # Pre-compute embeddings
    print("\n" + "-" * 70)
    print("Pre-computing LeJEPA Embeddings")
    print("-" * 70)

    train_dataset = EmbeddingDataset(train_base_dataset, lejepa, device)
    val_dataset = EmbeddingDataset(val_base_dataset, lejepa, device)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    # Initialize policy
    print("\n" + "=" * 70)
    print(f"Initializing {class_name} Entry Policy")
    print("=" * 70)

    policy = EntryPolicy(
        embedding_dim=embedding_dim,
        hidden_dim=args.policy_hidden_dim,
        num_layers=args.policy_layers,
        dropout=args.dropout,
        num_actions=num_classes,
    )
    policy = policy.to(device)

    total_params = sum(p.numel() for p in policy.parameters())
    print(f"Policy parameters: {total_params:,}")
    print(f"Output classes: {policy.num_actions}")

    # Loss function
    if args.use_focal_loss:
        if num_classes == 5:
            focal_alpha = torch.tensor([
                args.focal_alpha_hold,
                args.focal_alpha_atm,
                args.focal_alpha_otm,
                args.focal_alpha_atm,
                args.focal_alpha_otm,
            ])
        else:
            focal_alpha = torch.tensor([
                args.focal_alpha_hold,
                args.focal_alpha_signal,
                args.focal_alpha_signal,
            ])
        criterion = FocalLoss(alpha=focal_alpha, gamma=args.focal_gamma)
        print(f"Using Focal Loss: gamma={args.focal_gamma}")
    else:
        criterion = nn.CrossEntropyLoss()

    # Optimizer and scheduler
    optimizer = AdamW(
        policy.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=args.epochs - args.warmup_epochs,
        eta_min=args.lr / 100,
    )

    # Register checkpoint
    project_root = Path(__file__).parent
    mgr = CheckpointManager(project_root / "checkpoints", project_root)
    # Only pass parameters that CheckpointManager.create() accepts
    allowed_config_keys = {
        "embedding_dim", "num_layers", "num_heads", "patch_length", "underlying",
        "start_date", "end_date", "epochs", "batch_size", "lr", "lambda_sigreg",
        "lejepa_checkpoint", "lookahead", "min_roi_threshold", "otm_buffer",
        "focal_alpha_hold", "focal_alpha_atm", "focal_alpha_otm", "slippage_pct",
        "execution_delay", "include_partial_patches", "min_partial_length",
    }
    config = {k: v for k, v in vars(args).items() if k in allowed_config_keys}
    entry = mgr.get_or_create(
        args.checkpoint_id,
        model_type=args.model_type,
        config=config,
    )
    checkpoint_dir = Path(entry["path"])
    print(f"\nCheckpoint registered: {entry['name']} (ID: {entry['id'][:8]}...)")
    print(f"Checkpoint directory: {checkpoint_dir}")

    # Save label distribution to config
    mgr.set_label_distribution(args.checkpoint_id, train_dist, val_dist)

    # Training loop
    print("\n" + "=" * 70)
    print(f"Starting {class_name} Training")
    print("=" * 70)
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.lr}")

    best_val_acc = 0.0

    for epoch in range(1, args.epochs + 1):
        epoch_start = datetime.now()

        if epoch <= args.warmup_epochs:
            warmup_factor = epoch / args.warmup_epochs
            for param_group in optimizer.param_groups:
                param_group["lr"] = args.lr * warmup_factor

        train_metrics = train_entry_epoch(
            policy=policy,
            loader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
            num_classes=num_classes,
        )

        val_metrics = validate_entry(
            policy=policy,
            loader=val_loader,
            criterion=criterion,
            device=device,
            num_classes=num_classes,
        )

        if epoch > args.warmup_epochs:
            scheduler.step()

        metrics = {**train_metrics, **val_metrics}
        epoch_time = (datetime.now() - epoch_start).total_seconds()

        if epoch % args.log_every == 0:
            current_lr = optimizer.param_groups[0]["lr"]
            print(f"\nEpoch {epoch}/{args.epochs} ({epoch_time:.1f}s) | LR: {current_lr:.2e}")
            print(f"  Train: Loss={train_metrics['train_loss']:.4f} | Acc={train_metrics['train_accuracy']:.1%}")
            print(f"  Val:   Loss={val_metrics['val_loss']:.4f} | Acc={val_metrics['val_accuracy']:.1%}")

        if epoch % args.save_every == 0:
            save_entry_checkpoint(
                policy, optimizer, epoch, metrics,
                checkpoint_dir / f"entry_policy_epoch_{epoch:04d}.pt",
                embedding_dim, args,
            )

        if val_metrics["val_accuracy"] > best_val_acc:
            best_val_acc = val_metrics["val_accuracy"]
            save_entry_checkpoint(
                policy, optimizer, epoch, metrics,
                checkpoint_dir / "entry_policy_best.pt",
                embedding_dim, args,
            )
            print(f"  New best model! Val Acc: {best_val_acc:.1%}")

    # Save final model
    save_entry_checkpoint(
        policy, optimizer, args.epochs, metrics,
        checkpoint_dir / "entry_policy_final.pt",
        embedding_dim, args,
    )

    # Record metrics using set_training_complete
    mgr.set_training_complete(
        args.checkpoint_id,
        metrics={"best_val_accuracy": float(best_val_acc)},
        status="trained",
    )

    print("\n" + "=" * 70)
    print("Training Complete!")
    print("=" * 70)
    print(f"Best validation accuracy: {best_val_acc:.1%}")
    print(f"Final checkpoint: {checkpoint_dir / 'entry_policy_final.pt'}")
    print(f"Best checkpoint: {checkpoint_dir / 'entry_policy_best.pt'}")
    print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


# =============================================================================
# Entry Regression Policy Training
# =============================================================================


def train_regression_epoch(
    policy: RegressionEntryPolicy,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    max_grad_norm: float = 1.0,
) -> Dict[str, float]:
    """Train regression entry policy for one epoch."""
    policy.train()

    total_loss = 0.0
    total_samples = 0
    all_preds = []
    all_targets = []

    for embeddings, targets in loader:
        embeddings = embeddings.to(device)
        targets = targets.to(device).float().unsqueeze(-1)  # [B, 1]

        optimizer.zero_grad()

        preds = policy(embeddings)  # [B, 1]
        loss = F.mse_loss(preds, targets)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), max_grad_norm)
        optimizer.step()

        total_loss += loss.item() * len(targets)
        total_samples += len(targets)
        all_preds.append(preds.detach().cpu())
        all_targets.append(targets.cpu())

    avg_loss = total_loss / total_samples

    # Compute additional metrics
    all_preds = torch.cat(all_preds).squeeze()
    all_targets = torch.cat(all_targets).squeeze()

    # Correlation between predictions and targets
    correlation = torch.corrcoef(torch.stack([all_preds, all_targets]))[0, 1].item()

    # Direction accuracy (sign agreement)
    direction_correct = ((all_preds > 0) == (all_targets > 0)).float().mean().item()

    # Mean absolute error
    mae = (all_preds - all_targets).abs().mean().item()

    return {
        "train_loss": avg_loss,
        "train_mae": mae,
        "train_correlation": correlation,
        "train_direction_acc": direction_correct,
    }


def validate_regression(
    policy: RegressionEntryPolicy,
    loader: DataLoader,
    device: torch.device,
) -> Dict[str, float]:
    """Validate regression entry policy."""
    policy.eval()

    total_loss = 0.0
    total_samples = 0
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for embeddings, targets in loader:
            embeddings = embeddings.to(device)
            targets = targets.to(device).float().unsqueeze(-1)

            preds = policy(embeddings)
            loss = F.mse_loss(preds, targets)

            total_loss += loss.item() * len(targets)
            total_samples += len(targets)
            all_preds.append(preds.cpu())
            all_targets.append(targets.cpu())

    avg_loss = total_loss / total_samples

    all_preds = torch.cat(all_preds).squeeze()
    all_targets = torch.cat(all_targets).squeeze()

    # Correlation
    correlation = torch.corrcoef(torch.stack([all_preds, all_targets]))[0, 1].item()

    # Direction accuracy
    direction_correct = ((all_preds > 0) == (all_targets > 0)).float().mean().item()

    # MAE
    mae = (all_preds - all_targets).abs().mean().item()

    # Action distribution (for monitoring)
    bullish_pct = (all_preds > 0.3).float().mean().item() * 100
    bearish_pct = (all_preds < -0.3).float().mean().item() * 100
    neutral_pct = ((all_preds >= -0.3) & (all_preds <= 0.3)).float().mean().item() * 100

    return {
        "val_loss": avg_loss,
        "val_mae": mae,
        "val_correlation": correlation,
        "val_direction_acc": direction_correct,
        "val_bullish_pct": bullish_pct,
        "val_bearish_pct": bearish_pct,
        "val_neutral_pct": neutral_pct,
    }


def save_regression_checkpoint(
    policy: RegressionEntryPolicy,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    metrics: Dict[str, float],
    path: Path,
    embedding_dim: int,
    args: argparse.Namespace,
) -> None:
    """Save regression policy checkpoint."""
    path.parent.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        "epoch": epoch,
        "policy_state_dict": policy.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "metrics": metrics,
        "config": {
            "embedding_dim": embedding_dim,
            "hidden_dim": policy.hidden_dim,
            "num_layers": policy.num_layers,
            "model_type": "entry-regression",
            "lookahead": args.lookahead,
            "scale_factor": args.scale_factor,
        },
    }

    torch.save(checkpoint, path)
    print(f"  Saved checkpoint to {path}")


def train_regression(args: argparse.Namespace) -> None:
    """Main regression entry policy training function."""
    print("=" * 70)
    print("Regression Entry Policy Training")
    print("=" * 70)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    print(f"\nOutput: Continuous score in [-1, 1]")
    print(f"  Positive = bullish (buy call), magnitude = conviction")
    print(f"  Negative = bearish (buy put), magnitude = conviction")
    print(f"Lookahead: {args.lookahead} minutes")
    print(f"Scale factor: {args.scale_factor} (1% move → {args.scale_factor/100:.2f} score)")

    set_seed(args.seed)
    device = get_device(args.device)
    print(f"\nDevice: {device}")

    # Load pre-trained LeJEPA
    print("\n" + "=" * 70)
    print("Loading Pre-trained LeJEPA")
    print("=" * 70)

    lejepa, checkpoint = LeJEPA.load_checkpoint(args.lejepa_checkpoint, device=str(device))
    lejepa = lejepa.to(device)
    lejepa.eval()

    for param in lejepa.parameters():
        param.requires_grad = False

    embedding_dim = lejepa.embedding_dim
    print(f"Embedding dimension: {embedding_dim}")

    # Load data using same pipeline as classification training
    print("\n" + "=" * 70)
    print("Loading Training Data")
    print("=" * 70)

    data_dir = Path(args.data_dir)
    stocks_dir = data_dir / "stocks-1m"
    options_dir = data_dir / "options-1m"

    print(f"Loading combined stocks + options data")
    print(f"Stocks dir: {stocks_dir}")
    print(f"Options dir: {options_dir}")
    print(f"Underlying: {args.underlying}")
    if args.start_date:
        print(f"Start date: {args.start_date}")
    if args.end_date:
        print(f"End date: {args.end_date}")

    # Load normalized data
    df = load_normalized_data(
        stocks_dir=str(stocks_dir),
        options_dir=str(options_dir),
        underlying=args.underlying,
        start_date=args.start_date,
        end_date=args.end_date,
        max_files=args.max_files,
    )

    print(f"Loaded {len(df):,} bars from {df.index.min()} to {df.index.max()}")

    # Load raw close prices from cache
    print("Loading raw close prices for target computation...")
    start = parse_date(args.start_date)
    end = parse_date(args.end_date)

    raw_cache_dir = Path(RAW_CACHE_DIR)
    raw_files = sorted(raw_cache_dir.glob(f"{args.underlying}_raw_*.parquet"))
    raw_closes_list = []

    for rf in raw_files:
        date_str = rf.stem.split("_")[-1]
        try:
            file_date = datetime.strptime(date_str, "%Y-%m-%d").date()
            if start and file_date < start:
                continue
            if end and file_date > end:
                continue
            raw_df = pd.read_parquet(rf)
            raw_closes_list.append(raw_df["close"].values)
        except (ValueError, KeyError):
            continue

    raw_closes = np.concatenate(raw_closes_list) if raw_closes_list else np.zeros(len(df))

    # Verify alignment
    if len(raw_closes) != len(df):
        print(f"Warning: raw closes ({len(raw_closes)}) != normalized ({len(df)}), truncating...")
        min_len = min(len(raw_closes), len(df))
        raw_closes = raw_closes[:min_len]
        df = df.iloc[:min_len]

    # Filter to market hours (UTC: 14:30-21:00)
    market_open_utc = time(14, 30)
    market_close_utc = time(21, 0)

    mask = (df.index.time >= market_open_utc) & (df.index.time < market_close_utc)
    mask_array = np.array(mask)
    df = df[mask]
    raw_closes = raw_closes[mask_array]

    print(f"After market hours filter (UTC): {len(df):,} bars")

    # Split into days
    df["date"] = df.index.date
    days = df["date"].unique()

    n_days = len(days)
    n_train_days = int(n_days * args.train_split)
    train_days = days[:n_train_days]
    val_days = days[n_train_days:]

    print(f"Total days: {n_days}")
    print(f"Training days: {len(train_days)}")
    print(f"Validation days: {len(val_days)}")

    # Create train/val masks
    train_mask = df["date"].isin(train_days)
    val_mask = df["date"].isin(val_days)

    train_df = df[train_mask].drop(columns=["date"])
    val_df = df[val_mask].drop(columns=["date"])
    train_raw_closes = raw_closes[train_mask.values]
    val_raw_closes = raw_closes[val_mask.values]

    print(f"\nContext length: {args.context_len}")
    print(f"Feature dimension: {len(train_df.columns)}")

    # Create regression datasets
    print("\n" + "=" * 70)
    print("Creating Regression Datasets")
    print("=" * 70)

    train_base_dataset = EntryRegressionDataset(
        normalized_df=train_df,
        raw_closes=train_raw_closes,
        context_len=args.context_len,
        lookahead=args.lookahead,
        scale_factor=args.scale_factor,
    )

    val_base_dataset = EntryRegressionDataset(
        normalized_df=val_df,
        raw_closes=val_raw_closes,
        context_len=args.context_len,
        lookahead=args.lookahead,
        scale_factor=args.scale_factor,
    )

    print(f"\nTraining samples: {len(train_base_dataset):,}")
    train_stats = train_base_dataset.get_target_stats()
    print(f"  Target mean: {train_stats['mean']:.4f}")
    print(f"  Target std: {train_stats['std']:.4f}")
    print(f"  Bullish (>0.1): {train_stats['bullish_pct']:.1f}%")
    print(f"  Bearish (<-0.1): {train_stats['bearish_pct']:.1f}%")
    print(f"  Neutral: {train_stats['neutral_pct']:.1f}%")

    print(f"\nValidation samples: {len(val_base_dataset):,}")
    val_stats = val_base_dataset.get_target_stats()
    print(f"  Target mean: {val_stats['mean']:.4f}")
    print(f"  Target std: {val_stats['std']:.4f}")
    print(f"  Bullish (>0.1): {val_stats['bullish_pct']:.1f}%")
    print(f"  Bearish (<-0.1): {val_stats['bearish_pct']:.1f}%")
    print(f"  Neutral: {val_stats['neutral_pct']:.1f}%")

    # Pre-compute embeddings
    print("\n" + "-" * 70)
    print("Pre-computing LeJEPA Embeddings")
    print("-" * 70)

    train_dataset = RegressionEmbeddingDataset(train_base_dataset, lejepa, device)
    val_dataset = RegressionEmbeddingDataset(val_base_dataset, lejepa, device)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    # Initialize policy
    print("\n" + "=" * 70)
    print("Initializing Regression Policy")
    print("=" * 70)

    policy = RegressionEntryPolicy(
        embedding_dim=embedding_dim,
        hidden_dim=args.policy_hidden_dim,
        num_layers=args.policy_layers,
        dropout=args.dropout,
    )
    policy = policy.to(device)

    num_params = sum(p.numel() for p in policy.parameters())
    print(f"Policy parameters: {num_params:,}")
    print(f"Hidden dim: {args.policy_hidden_dim}")
    print(f"Num layers: {args.policy_layers}")

    # Optimizer and scheduler
    optimizer = AdamW(
        policy.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=args.epochs,
        eta_min=args.lr * 0.01,
    )

    # Checkpoint management
    project_root = Path(__file__).parent
    mgr = CheckpointManager(project_root / "checkpoints", project_root)
    config = {
        "lejepa_checkpoint": args.lejepa_checkpoint,
        "embedding_dim": embedding_dim,
        "hidden_dim": args.policy_hidden_dim,
        "scale_factor": args.scale_factor,
        "lookahead": args.lookahead,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
    }
    entry = mgr.get_or_create(
        args.checkpoint_id,
        model_type="entry-regression",
        config=config,
    )
    checkpoint_dir = Path(entry["path"])
    print(f"\nCheckpoint registered: {entry['name']} (ID: {entry['id'][:8]}...)")
    print(f"Checkpoint directory: {checkpoint_dir}")

    # Training loop
    print("\n" + "=" * 70)
    print("Starting Training")
    print("=" * 70)

    best_val_corr = -1.0

    for epoch in range(1, args.epochs + 1):
        print(f"\n" + "=" * 70)
        print(f"Epoch {epoch}/{args.epochs}")
        print("=" * 70)

        train_metrics = train_regression_epoch(
            policy, train_loader, optimizer, device
        )

        val_metrics = validate_regression(policy, val_loader, device)

        scheduler.step()

        metrics = {**train_metrics, **val_metrics, "epoch": epoch}

        if epoch % args.log_every == 0:
            print(f"  Train: Loss={train_metrics['train_loss']:.4f} | "
                  f"MAE={train_metrics['train_mae']:.4f} | "
                  f"Corr={train_metrics['train_correlation']:.4f} | "
                  f"Dir={train_metrics['train_direction_acc']:.1%}")
            print(f"  Val:   Loss={val_metrics['val_loss']:.4f} | "
                  f"MAE={val_metrics['val_mae']:.4f} | "
                  f"Corr={val_metrics['val_correlation']:.4f} | "
                  f"Dir={val_metrics['val_direction_acc']:.1%}")
            print(f"  Pred Distribution: Bull={val_metrics['val_bullish_pct']:.1f}% | "
                  f"Bear={val_metrics['val_bearish_pct']:.1f}% | "
                  f"Neutral={val_metrics['val_neutral_pct']:.1f}%")

        if epoch % args.save_every == 0:
            save_regression_checkpoint(
                policy, optimizer, epoch, metrics,
                checkpoint_dir / f"regression_policy_epoch_{epoch:04d}.pt",
                embedding_dim, args,
            )

        # Save best model based on correlation (most important metric for trading)
        if val_metrics["val_correlation"] > best_val_corr:
            best_val_corr = val_metrics["val_correlation"]
            save_regression_checkpoint(
                policy, optimizer, epoch, metrics,
                checkpoint_dir / "regression_policy_best.pt",
                embedding_dim, args,
            )
            print(f"  New best model! Val Correlation: {best_val_corr:.4f}")

    # Save final model
    save_regression_checkpoint(
        policy, optimizer, args.epochs, metrics,
        checkpoint_dir / "regression_policy_final.pt",
        embedding_dim, args,
    )

    # Record metrics using set_training_complete
    mgr.set_training_complete(
        args.checkpoint_id,
        metrics={
            "best_val_correlation": float(best_val_corr),
            "final_val_mae": float(val_metrics["val_mae"]),
            "final_val_direction_acc": float(val_metrics["val_direction_acc"]),
        },
        status="trained",
    )

    print("\n" + "=" * 70)
    print("Training Complete!")
    print("=" * 70)
    print(f"Best validation correlation: {best_val_corr:.4f}")
    print(f"Final checkpoint: {checkpoint_dir / 'regression_policy_final.pt'}")
    print(f"Best checkpoint: {checkpoint_dir / 'regression_policy_best.pt'}")
    print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


# =============================================================================
# Directional Move Entry Policy Training
# =============================================================================


def train_directional(args: argparse.Namespace) -> None:
    """Main directional move entry policy training function.

    This uses DirectionalMoveDataset which labels moments before clean,
    directional price moves using 5-minute smoothed candles.
    """
    num_classes = 5  # HOLD, CALL_ATM, CALL_OTM, PUT_ATM, PUT_OTM

    print("=" * 70)
    print("Directional Move Entry Policy Training")
    print("=" * 70)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    print(f"\nClasses: HOLD, CALL_ATM, CALL_OTM, PUT_ATM, PUT_OTM")
    print(f"Labeling: Directional move with {args.candle_agg_minutes}-minute smoothed candles")
    print(f"Candle range: {args.min_candles}-{args.max_candles} ({args.min_candles * args.candle_agg_minutes}-{args.max_candles * args.candle_agg_minutes} minutes)")
    print(f"Min move: {args.min_move_pct}%, Consistency: {args.min_consistency_pct * 100}%")
    print(f"Weights: Move size={args.move_size_weight}, Wick penalty={args.wick_penalty_weight}")

    set_seed(args.seed)
    device = get_device(args.device)
    print(f"\nDevice: {device}")

    # Load pre-trained LeJEPA
    print("\n" + "=" * 70)
    print("Loading Pre-trained LeJEPA")
    print("=" * 70)

    lejepa, checkpoint = LeJEPA.load_checkpoint(args.lejepa_checkpoint, device=str(device))
    lejepa = lejepa.to(device)
    lejepa.eval()

    for param in lejepa.parameters():
        param.requires_grad = False

    embedding_dim = lejepa.embedding_dim
    print(f"Embedding dimension: {embedding_dim}")

    # Load data using same pipeline as LeJEPA training
    print("\n" + "=" * 70)
    print("Loading Training Data")
    print("=" * 70)

    data_dir = Path(args.data_dir)
    stocks_dir = data_dir / "stocks-1m"
    options_dir = data_dir / "options-1m"
    raw_dir = data_dir / "training-1m-raw" / args.underlying

    print(f"Loading combined stocks + options data")
    print(f"Stocks dir: {stocks_dir}")
    print(f"Options dir: {options_dir}")
    print(f"Raw data dir: {raw_dir}")
    print(f"Underlying: {args.underlying}")
    if args.start_date:
        print(f"Start date: {args.start_date}")
    if args.end_date:
        print(f"End date: {args.end_date}")

    # Load normalized data (for patches/embeddings)
    df = load_normalized_data(
        stocks_dir=str(stocks_dir),
        options_dir=str(options_dir),
        underlying=args.underlying,
        start_date=args.start_date,
        end_date=args.end_date,
        max_files=args.max_files,
    )

    print(f"Loaded {len(df):,} normalized bars from {df.index.min()} to {df.index.max()}")

    # Load raw OHLCV data (for labeling)
    print(f"Loading raw OHLCV data for labeling...")
    raw_files = sorted(raw_dir.glob("**/*.parquet"))
    start = parse_date(args.start_date)
    end = parse_date(args.end_date)

    raw_dfs = []
    for rf in raw_files:
        # Extract date from path (e.g., 2024-12/02.parquet -> 2024-12-02)
        try:
            month_dir = rf.parent.name  # 2024-12
            day = rf.stem  # 02
            file_date = datetime.strptime(f"{month_dir}-{day}", "%Y-%m-%d").date()
            if start and file_date < start:
                continue
            if end and file_date > end:
                continue
            raw_dfs.append(pd.read_parquet(rf))
        except (ValueError, KeyError):
            continue

    if raw_dfs:
        raw_df = pd.concat(raw_dfs, ignore_index=False).sort_index()
        print(f"Loaded {len(raw_df):,} raw bars")
    else:
        raise ValueError(f"No raw OHLCV data found in {raw_dir}")

    # Filter to market hours (UTC: 14:30-21:00)
    market_open_utc = time(14, 30)
    market_close_utc = time(21, 0)

    mask = (df.index.time >= market_open_utc) & (df.index.time < market_close_utc)
    df = df[mask]
    raw_mask = (raw_df.index.time >= market_open_utc) & (raw_df.index.time < market_close_utc)
    raw_df = raw_df[raw_mask]

    print(f"After market hours filter (UTC): {len(df):,} normalized, {len(raw_df):,} raw bars")

    # Verify alignment
    if len(df) != len(raw_df):
        print(f"Warning: normalized ({len(df)}) != raw ({len(raw_df)}), aligning by index...")
        common_idx = df.index.intersection(raw_df.index)
        df = df.loc[common_idx]
        raw_df = raw_df.loc[common_idx]
        print(f"After alignment: {len(df):,} bars")

    # Split into days
    df["date"] = df.index.date
    raw_df["date"] = raw_df.index.date
    days = df["date"].unique()

    n_days = len(days)
    n_train_days = int(n_days * args.train_split)
    train_days = days[:n_train_days]
    val_days = days[n_train_days:]

    print(f"Total days: {n_days}")
    print(f"Training days: {len(train_days)}")
    print(f"Validation days: {len(val_days)}")

    # Create train/val masks
    train_mask = df["date"].isin(train_days)
    val_mask = df["date"].isin(val_days)

    train_df = df[train_mask].drop(columns=["date"])
    val_df = df[val_mask].drop(columns=["date"])

    # Raw OHLCV for labeling
    train_raw_df = raw_df[raw_df["date"].isin(train_days)]
    val_raw_df = raw_df[raw_df["date"].isin(val_days)]

    train_raw_ohlcv = {
        "open": train_raw_df["open"].values,
        "high": train_raw_df["high"].values,
        "low": train_raw_df["low"].values,
        "close": train_raw_df["close"].values,
    }
    val_raw_ohlcv = {
        "open": val_raw_df["open"].values,
        "high": val_raw_df["high"].values,
        "low": val_raw_df["low"].values,
        "close": val_raw_df["close"].values,
    }

    print(f"\nContext length: {args.context_len}")
    print(f"Feature dimension: {len(train_df.columns)}")

    # Create datasets
    print("\n" + "=" * 70)
    print("Creating Directional Move Datasets")
    print("=" * 70)

    # Setup patcher and ROI calculator
    base_cols = {"open", "high", "low", "close", "volume"}
    extra_columns = [col for col in train_df.columns if col not in base_cols]
    patcher = MarketPatch(patch_length=args.context_len, extra_columns=extra_columns)

    roi_calculator = RealOptionsROICalculator(
        options_dir=str(options_dir),
        base_iv=args.base_iv,
        execution_delay_minutes=args.execution_delay,
        slippage_pct=args.slippage_pct,
    )

    print(f"Using REAL options prices from: {options_dir}")
    print(f"Execution: OPEN price of T+{args.execution_delay} candle")
    print(f"Slippage: {args.slippage_pct}%")

    # Create DirectionalMoveDataset instances
    train_base_dataset = DirectionalMoveDataset(
        normalized_df=train_df,
        raw_ohlcv=train_raw_ohlcv,
        patcher=patcher,
        roi_calculator=roi_calculator,
        context_len=args.context_len,
        candle_agg_minutes=args.candle_agg_minutes,
        min_candles=args.min_candles,
        max_candles=args.max_candles,
        min_move_pct=args.min_move_pct,
        min_consistency_pct=args.min_consistency_pct,
        wick_penalty_weight=args.wick_penalty_weight,
        move_size_weight=args.move_size_weight,
        atm_offset=args.atm_offset,
        otm_offset=args.otm_offset,
        otm_buffer=args.otm_buffer,
        min_roi_threshold=args.min_roi_threshold,
    )

    val_base_dataset = DirectionalMoveDataset(
        normalized_df=val_df,
        raw_ohlcv=val_raw_ohlcv,
        patcher=patcher,
        roi_calculator=roi_calculator,
        context_len=args.context_len,
        candle_agg_minutes=args.candle_agg_minutes,
        min_candles=args.min_candles,
        max_candles=args.max_candles,
        min_move_pct=args.min_move_pct,
        min_consistency_pct=args.min_consistency_pct,
        wick_penalty_weight=args.wick_penalty_weight,
        move_size_weight=args.move_size_weight,
        atm_offset=args.atm_offset,
        otm_offset=args.otm_offset,
        otm_buffer=args.otm_buffer,
        min_roi_threshold=args.min_roi_threshold,
    )

    stats = roi_calculator.get_stats()
    print(f"\nROI Price Lookup Statistics:")
    print(f"  Real options prices: {stats['real_lookups']:,} ({stats['real_pct']:.1f}%)")
    print(f"  Black-Scholes fallback: {stats['fallback_lookups']:,}")

    print(f"\nTraining samples: {len(train_base_dataset):,}")
    train_dist = train_base_dataset.get_class_distribution()
    for action_name, count in train_dist.items():
        pct = count / len(train_base_dataset) * 100 if len(train_base_dataset) > 0 else 0
        print(f"  {action_name}: {count:,} ({pct:.1f}%)")

    print(f"\nValidation samples: {len(val_base_dataset):,}")
    val_dist = val_base_dataset.get_class_distribution()
    for action_name, count in val_dist.items():
        pct = count / len(val_base_dataset) * 100 if len(val_base_dataset) > 0 else 0
        print(f"  {action_name}: {count:,} ({pct:.1f}%)")

    # Pre-compute embeddings
    print("\n" + "-" * 70)
    print("Pre-computing LeJEPA Embeddings")
    print("-" * 70)

    train_dataset = EmbeddingDataset(train_base_dataset, lejepa, device)
    val_dataset = EmbeddingDataset(val_base_dataset, lejepa, device)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    # Initialize policy
    print("\n" + "=" * 70)
    print("Initializing Directional Entry Policy")
    print("=" * 70)

    policy = EntryPolicy(
        embedding_dim=embedding_dim,
        hidden_dim=args.policy_hidden_dim,
        num_layers=args.policy_layers,
        dropout=args.dropout,
        num_actions=num_classes,
    )
    policy = policy.to(device)

    total_params = sum(p.numel() for p in policy.parameters())
    print(f"Policy parameters: {total_params:,}")
    print(f"Output classes: {policy.num_actions}")

    # Loss function
    if args.use_focal_loss:
        focal_alpha = torch.tensor([
            args.focal_alpha_hold,
            args.focal_alpha_atm,
            args.focal_alpha_otm,
            args.focal_alpha_atm,
            args.focal_alpha_otm,
        ])
        criterion = FocalLoss(alpha=focal_alpha, gamma=args.focal_gamma)
        print(f"Using Focal Loss: gamma={args.focal_gamma}")
    else:
        criterion = nn.CrossEntropyLoss()

    # Optimizer and scheduler
    optimizer = AdamW(
        policy.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=args.epochs - args.warmup_epochs,
        eta_min=args.lr / 100,
    )

    # Register checkpoint
    project_root = Path(__file__).parent
    mgr = CheckpointManager(project_root / "checkpoints", project_root)
    allowed_config_keys = {
        "embedding_dim", "num_layers", "num_heads", "patch_length", "underlying",
        "start_date", "end_date", "epochs", "batch_size", "lr", "lambda_sigreg",
        "lejepa_checkpoint", "lookahead", "min_roi_threshold", "otm_buffer",
        "focal_alpha_hold", "focal_alpha_atm", "focal_alpha_otm", "slippage_pct",
        "execution_delay", "include_partial_patches", "min_partial_length",
        "candle_agg_minutes", "min_candles", "max_candles", "min_move_pct",
        "min_consistency_pct", "wick_penalty_weight", "move_size_weight",
    }
    config = {k: v for k, v in vars(args).items() if k in allowed_config_keys}
    entry = mgr.get_or_create(
        args.checkpoint_id,
        model_type=args.model_type,
        config=config,
    )
    checkpoint_dir = Path(entry["path"])
    print(f"\nCheckpoint registered: {entry['name']} (ID: {entry['id'][:8]}...)")
    print(f"Checkpoint directory: {checkpoint_dir}")

    # Save label distribution to config
    mgr.set_label_distribution(args.checkpoint_id, train_dist, val_dist)

    # Training loop
    print("\n" + "=" * 70)
    print("Starting Directional Move Training")
    print("=" * 70)
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.lr}")

    best_val_acc = 0.0

    for epoch in range(1, args.epochs + 1):
        epoch_start = datetime.now()

        if epoch <= args.warmup_epochs:
            warmup_factor = epoch / args.warmup_epochs
            for param_group in optimizer.param_groups:
                param_group["lr"] = args.lr * warmup_factor

        train_metrics = train_entry_epoch(
            policy=policy,
            loader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
            num_classes=num_classes,
        )

        val_metrics = validate_entry(
            policy=policy,
            loader=val_loader,
            criterion=criterion,
            device=device,
            num_classes=num_classes,
        )

        if epoch > args.warmup_epochs:
            scheduler.step()

        metrics = {**train_metrics, **val_metrics}
        epoch_time = (datetime.now() - epoch_start).total_seconds()

        if epoch % args.log_every == 0:
            current_lr = optimizer.param_groups[0]["lr"]
            print(f"\nEpoch {epoch}/{args.epochs} ({epoch_time:.1f}s) | LR: {current_lr:.2e}")
            print(f"  Train: Loss={train_metrics['train_loss']:.4f} | Acc={train_metrics['train_accuracy']:.1%}")
            print(f"  Val:   Loss={val_metrics['val_loss']:.4f} | Acc={val_metrics['val_accuracy']:.1%}")

        if epoch % args.save_every == 0:
            save_entry_checkpoint(
                policy, optimizer, epoch, metrics,
                checkpoint_dir / f"entry_policy_epoch_{epoch:04d}.pt",
                embedding_dim, args,
            )

        if val_metrics["val_accuracy"] > best_val_acc:
            best_val_acc = val_metrics["val_accuracy"]
            save_entry_checkpoint(
                policy, optimizer, epoch, metrics,
                checkpoint_dir / "entry_policy_best.pt",
                embedding_dim, args,
            )
            print(f"  New best model! Val Acc: {best_val_acc:.1%}")

    # Save final model
    save_entry_checkpoint(
        policy, optimizer, args.epochs, metrics,
        checkpoint_dir / "entry_policy_final.pt",
        embedding_dim, args,
    )

    # Record metrics using set_training_complete
    mgr.set_training_complete(
        args.checkpoint_id,
        metrics={"best_val_accuracy": float(best_val_acc)},
        status="trained",
    )

    print("\n" + "=" * 70)
    print("Training Complete!")
    print("=" * 70)
    print(f"Best validation accuracy: {best_val_acc:.1%}")
    print(f"Final checkpoint: {checkpoint_dir / 'entry_policy_final.pt'}")
    print(f"Best checkpoint: {checkpoint_dir / 'entry_policy_best.pt'}")
    print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


# =============================================================================
# Main Entry Point
# =============================================================================


def main() -> None:
    """Main entry point dispatching to appropriate training function."""
    args = parse_args()

    if args.model_type == "jepa":
        train_jepa(args)
    elif args.model_type in ["entry-3class", "entry-5class"]:
        train_entry(args)
    elif args.model_type == "entry-regression":
        train_regression(args)
    elif args.model_type == "entry-directional":
        train_directional(args)
    else:
        raise ValueError(f"Unknown model type: {args.model_type}")


if __name__ == "__main__":
    main()
