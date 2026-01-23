"""
Common utilities shared across all training modules.

Contains:
- Random seed and device utilities
- Learning rate scheduling
- File filtering utilities
- Embedding dataset wrappers
- Options ROI calculator
"""

import math
import re
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from src.model.lejepa import LeJEPA


# =============================================================================
# Core Utilities
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
    """Filter file list by date range.

    Handles two file formats:
    - Old: SPY_2020-01-01.parquet (date in filename)
    - New: 2020-01/01.parquet (date split between parent dir and filename)
    """
    if not start_date and not end_date:
        return files

    filtered = []
    # Pattern for full date in filename
    full_date_pattern = re.compile(r'(\d{4}-\d{2}-\d{2})')
    # Pattern for YYYY-MM directory
    month_dir_pattern = re.compile(r'(\d{4}-\d{2})')

    for f in files:
        file_date = None

        # Try to find full date in filename first
        match = full_date_pattern.search(f.stem)
        if match:
            file_date = match.group(1)
        else:
            # Try to construct date from parent dir (YYYY-MM) + filename (DD)
            parent_match = month_dir_pattern.search(f.parent.name)
            if parent_match and f.stem.isdigit():
                file_date = f"{parent_match.group(1)}-{f.stem.zfill(2)}"

        if file_date:
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
# Embedding Dataset Wrappers
# =============================================================================


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
        base_dataset,  # EntryRegressionDataset
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
# Options ROI Calculator
# =============================================================================


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
