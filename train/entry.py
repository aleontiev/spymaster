"""
Entry policy training module.

Contains:
- Entry datasets (3-class, 5-class, regression, directional)
- Entry training functions
- Entry-specific argument handling

All entry policies use a pre-trained LeJEPA encoder.
"""

import argparse
from datetime import datetime, time
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

from src.data.processing import MarketPatch
from src.data.dag.loader import (
    load_normalized_data,
    RAW_CACHE_DIR,
    parse_date,
)
from src.model.lejepa import LeJEPA
from src.model.policy import EntryPolicy, EntryAction, EntryActionLegacy, RegressionEntryPolicy
from src.model.loss import FocalLoss
from src.managers.checkpoint_manager import CheckpointManager

from train.common import (
    set_seed,
    get_device,
    EmbeddingDataset,
    RegressionEmbeddingDataset,
    RealOptionsROICalculator,
)


# =============================================================================
# Entry Datasets
# =============================================================================


class Entry3ClassDataset(Dataset):
    """Dataset for 3-class entry policy training (HOLD, BUY_CALL, BUY_PUT)."""

    def __init__(
        self,
        normalized_df: pd.DataFrame,
        raw_closes: np.ndarray,
        context_len: int = 32,
        lookahead: int = 10,
        threshold_pct: float = 0.1,
    ):
        self.context_len = context_len
        self.lookahead = lookahead
        self.threshold_pct = threshold_pct
        self.raw_closes = raw_closes

        self.data = torch.tensor(normalized_df.values, dtype=torch.float32)
        self.data = torch.nan_to_num(self.data, nan=0.0, posinf=10.0, neginf=-10.0)
        self.data = torch.clamp(self.data, -10, 10)

        self.valid_indices = []
        for i in range(context_len, len(normalized_df) - lookahead):
            self.valid_indices.append(i)

        self.labels = self._compute_labels()

    def _compute_labels(self) -> np.ndarray:
        labels = []
        for idx in self.valid_indices:
            current_price = self.raw_closes[idx]
            future_price = self.raw_closes[idx + self.lookahead]

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
        patch = self.data[start_idx:data_idx]
        label = self.labels[idx]
        return patch, label

    def get_class_distribution(self) -> Dict[str, int]:
        unique, counts = np.unique(self.labels, return_counts=True)
        dist = {}
        for action in EntryActionLegacy:
            count = counts[unique == action.value][0] if action.value in unique else 0
            dist[action.name] = int(count)
        return dist


class EntryPercentile3ClassDataset(Dataset):
    """
    Dataset for 3-class entry policy using percentile-based thresholds.

    Supports asymmetric thresholds for long vs short signals based on
    historical percentile analysis of price movements.

    Labels:
        - LONG (BUY_CALL): price increases >= long_threshold_pct
        - SHORT (BUY_PUT): price decreases <= -short_threshold_pct
        - HOLD: everything else

    Args:
        normalized_df: DataFrame with normalized features
        raw_closes: Raw close prices for label computation
        context_len: Length of context window (should match LeJEPA)
        lookahead: Minutes to look ahead for price change
        long_threshold_pct: % threshold for LONG signal (e.g., 0.363 for 99th percentile)
        short_threshold_pct: % threshold for SHORT signal (e.g., 0.400 for 1st percentile)
    """

    def __init__(
        self,
        normalized_df: pd.DataFrame,
        raw_closes: np.ndarray,
        context_len: int = 60,
        lookahead: int = 15,
        long_threshold_pct: float = 0.363,
        short_threshold_pct: float = 0.400,
    ):
        self.context_len = context_len
        self.lookahead = lookahead
        self.long_threshold_pct = long_threshold_pct
        self.short_threshold_pct = short_threshold_pct
        self.raw_closes = raw_closes

        self.data = torch.tensor(normalized_df.values, dtype=torch.float32)
        self.data = torch.nan_to_num(self.data, nan=0.0, posinf=10.0, neginf=-10.0)
        self.data = torch.clamp(self.data, -10, 10)

        self.valid_indices = []
        for i in range(context_len, len(normalized_df) - lookahead):
            self.valid_indices.append(i)

        self.labels = self._compute_labels()

    def _compute_labels(self) -> np.ndarray:
        labels = []
        for idx in self.valid_indices:
            current_price = self.raw_closes[idx]
            future_price = self.raw_closes[idx + self.lookahead]

            if current_price == 0:
                labels.append(EntryActionLegacy.HOLD)
                continue

            pct_change = (future_price - current_price) / current_price * 100

            if pct_change >= self.long_threshold_pct:
                labels.append(EntryActionLegacy.BUY_CALL)
            elif pct_change <= -self.short_threshold_pct:
                labels.append(EntryActionLegacy.BUY_PUT)
            else:
                labels.append(EntryActionLegacy.HOLD)

        return np.array(labels)

    def __len__(self) -> int:
        return len(self.valid_indices)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        data_idx = self.valid_indices[idx]
        start_idx = data_idx - self.context_len
        patch = self.data[start_idx:data_idx]
        label = self.labels[idx]
        return patch, label

    def get_class_distribution(self) -> Dict[str, int]:
        unique, counts = np.unique(self.labels, return_counts=True)
        dist = {}
        for action in EntryActionLegacy:
            count = counts[unique == action.value][0] if action.value in unique else 0
            dist[action.name] = int(count)
        return dist

    def get_label_stats(self) -> Dict[str, float]:
        """Get detailed label statistics."""
        total = len(self.labels)
        hold_count = (self.labels == EntryActionLegacy.HOLD).sum()
        long_count = (self.labels == EntryActionLegacy.BUY_CALL).sum()
        short_count = (self.labels == EntryActionLegacy.BUY_PUT).sum()

        return {
            "total": total,
            "hold_count": int(hold_count),
            "long_count": int(long_count),
            "short_count": int(short_count),
            "hold_pct": hold_count / total * 100,
            "long_pct": long_count / total * 100,
            "short_pct": short_count / total * 100,
            "signal_pct": (long_count + short_count) / total * 100,
        }


class EntryRegressionDataset(Dataset):
    """Dataset for continuous regression entry policy training."""

    def __init__(
        self,
        normalized_df: pd.DataFrame,
        raw_closes: np.ndarray,
        context_len: int = 32,
        lookahead: int = 10,
        scale_factor: float = 100.0,
    ):
        self.context_len = context_len
        self.lookahead = lookahead
        self.scale_factor = scale_factor
        self.raw_closes = raw_closes

        self.data = torch.tensor(normalized_df.values, dtype=torch.float32)
        self.data = torch.nan_to_num(self.data, nan=0.0, posinf=10.0, neginf=-10.0)
        self.data = torch.clamp(self.data, -10, 10)

        self.valid_indices = []
        for i in range(context_len, len(normalized_df) - lookahead):
            self.valid_indices.append(i)

        self.targets = self._compute_targets()

    def _compute_targets(self) -> np.ndarray:
        targets = []
        for idx in self.valid_indices:
            current_price = self.raw_closes[idx]
            future_price = self.raw_closes[idx + self.lookahead]

            if current_price == 0:
                targets.append(0.0)
                continue

            pct_return = (future_price - current_price) / current_price
            scaled_return = pct_return * self.scale_factor
            target = np.clip(scaled_return, -1.0, 1.0)
            targets.append(target)

        return np.array(targets, dtype=np.float32)

    def __len__(self) -> int:
        return len(self.valid_indices)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, float]:
        data_idx = self.valid_indices[idx]
        start_idx = data_idx - self.context_len
        patch = self.data[start_idx:data_idx]
        target = self.targets[idx]
        return patch, target

    def get_target_stats(self) -> Dict[str, float]:
        return {
            "mean": float(np.mean(self.targets)),
            "std": float(np.std(self.targets)),
            "min": float(np.min(self.targets)),
            "max": float(np.max(self.targets)),
            "bullish_pct": float((self.targets > 0.1).mean() * 100),
            "bearish_pct": float((self.targets < -0.1).mean() * 100),
            "neutral_pct": float((np.abs(self.targets) <= 0.1).mean() * 100),
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

        self.valid_indices = []
        for i in range(context_len, len(df) - context_len):
            self.valid_indices.append(i)

        self.labels = self._compute_labels()

    def _compute_labels(self) -> np.ndarray:
        labels = []

        for idx in tqdm(self.valid_indices, desc="Computing ROI labels"):
            spot_now = self.closes[idx]
            timestamp_decision = self.timestamps[idx]

            call_atm_strike, call_otm_strike = self.roi_calc.get_atm_otm_strikes(
                spot_now, is_call=True, atm_offset=self.atm_offset, otm_offset=self.otm_offset
            )
            put_atm_strike, put_otm_strike = self.roi_calc.get_atm_otm_strikes(
                spot_now, is_call=False, atm_offset=self.atm_offset, otm_offset=self.otm_offset
            )

            best_call_atm_weighted = -float('inf')
            best_call_otm_weighted = -float('inf')
            best_put_atm_weighted = -float('inf')
            best_put_otm_weighted = -float('inf')

            for exit_offset in range(1, self.context_len + 1):
                spot_future = self.closes[idx + exit_offset]
                timestamp_exit = self.timestamps[idx + exit_offset]

                if self.no_time_decay:
                    time_weight = 1.0
                    call_consistency = 1.0
                    put_consistency = 1.0
                else:
                    time_weight = 1.0 - 0.5 * (exit_offset - 1) / (self.context_len - 1)
                    window_opens = self.opens[idx + 1 : idx + exit_offset + 1]
                    window_closes = self.closes[idx + 1 : idx + exit_offset + 1]
                    green_candles = np.sum(window_closes > window_opens)
                    red_candles = np.sum(window_closes < window_opens)
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

                call_atm_weighted = call_atm_roi * time_weight * call_consistency
                call_otm_weighted = call_otm_roi * time_weight * call_consistency
                put_atm_weighted = put_atm_roi * time_weight * put_consistency
                put_otm_weighted = put_otm_roi * time_weight * put_consistency

                if call_atm_weighted > best_call_atm_weighted:
                    best_call_atm_weighted = call_atm_weighted
                if call_otm_weighted > best_call_otm_weighted:
                    best_call_otm_weighted = call_otm_weighted
                if put_atm_weighted > best_put_atm_weighted:
                    best_put_atm_weighted = put_atm_weighted
                if put_otm_weighted > best_put_otm_weighted:
                    best_put_otm_weighted = put_otm_weighted

            best_call_weighted = max(best_call_atm_weighted, best_call_otm_weighted)
            call_prefers_otm = best_call_otm_weighted > best_call_atm_weighted * self.otm_buffer

            best_put_weighted = max(best_put_atm_weighted, best_put_otm_weighted)
            put_prefers_otm = best_put_otm_weighted > best_put_atm_weighted * self.otm_buffer

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


class EntryMAE3ClassDataset(Dataset):
    """
    Dataset for 3-class entry policy training using MAE (Maximum Adverse Excursion) labeling.

    MAE ensures only sharp, one-directional moves are labeled as LONG/SHORT.
    A move is labeled LONG only if:
      1. Price increases by at least `min_move_pct` over the lookahead window
      2. Maximum downward pullback (adverse excursion) is less than `max_mae_pct`

    A move is labeled SHORT only if:
      1. Price decreases by at least `min_move_pct` over the lookahead window
      2. Maximum upward pullback (adverse excursion) is less than `max_mae_pct`

    Everything else is labeled HOLD.

    Args:
        normalized_df: DataFrame with normalized features for patches
        raw_ohlcv: Dict with raw OHLC arrays for label computation
        context_len: Length of context window
        lookahead: Bars to look ahead for labeling
        min_move_pct: Minimum price move % for a directional signal
        max_mae_pct: Maximum adverse excursion % allowed (lower = stricter)
        use_close_to_close: If True, compute move from close to close. If False, use high/low extremes.
    """

    def __init__(
        self,
        normalized_df: pd.DataFrame,
        raw_ohlcv: Dict[str, np.ndarray],
        context_len: int = 15,
        lookahead: int = 5,
        min_move_pct: float = 0.10,
        max_mae_pct: float = 0.05,
        use_close_to_close: bool = False,
    ):
        self.context_len = context_len
        self.lookahead = lookahead
        self.min_move_pct = min_move_pct
        self.max_mae_pct = max_mae_pct
        self.use_close_to_close = use_close_to_close

        # Store raw OHLC for label computation
        self.opens = raw_ohlcv["open"]
        self.highs = raw_ohlcv["high"]
        self.lows = raw_ohlcv["low"]
        self.closes = raw_ohlcv["close"]

        # Prepare normalized data for patches
        self.data = torch.tensor(normalized_df.values, dtype=torch.float32)
        self.data = torch.nan_to_num(self.data, nan=0.0, posinf=10.0, neginf=-10.0)
        self.data = torch.clamp(self.data, -10, 10)

        # Valid indices: need context behind and lookahead ahead
        self.valid_indices = []
        for i in range(context_len, len(normalized_df) - lookahead):
            self.valid_indices.append(i)

        self.labels = self._compute_mae_labels()

    def _compute_mae_labels(self) -> np.ndarray:
        """Compute labels using Maximum Adverse Excursion filtering."""
        labels = []

        for idx in self.valid_indices:
            entry_price = self.closes[idx]

            if entry_price == 0:
                labels.append(EntryActionLegacy.HOLD)
                continue

            # Get the lookahead window
            window_start = idx + 1
            window_end = idx + self.lookahead + 1

            window_highs = self.highs[window_start:window_end]
            window_lows = self.lows[window_start:window_end]
            window_closes = self.closes[window_start:window_end]

            if len(window_closes) == 0:
                labels.append(EntryActionLegacy.HOLD)
                continue

            # Compute move and MAE
            if self.use_close_to_close:
                # Close-to-close: final close vs entry
                exit_price = window_closes[-1]
                move_pct = (exit_price - entry_price) / entry_price * 100

                # MAE for longs: max downward move from entry
                max_adverse_low = np.min(window_lows)
                mae_long = (entry_price - max_adverse_low) / entry_price * 100

                # MAE for shorts: max upward move from entry
                max_adverse_high = np.max(window_highs)
                mae_short = (max_adverse_high - entry_price) / entry_price * 100
            else:
                # Use extremes: best possible exit during window
                max_high = np.max(window_highs)
                min_low = np.min(window_lows)

                # For LONG: move to max high, MAE is worst pullback before that
                move_long_pct = (max_high - entry_price) / entry_price * 100
                # Find where max high occurred
                max_high_idx = np.argmax(window_highs)
                # MAE for long is worst low BEFORE hitting max high
                if max_high_idx > 0:
                    mae_long = (entry_price - np.min(window_lows[:max_high_idx + 1])) / entry_price * 100
                else:
                    mae_long = (entry_price - window_lows[0]) / entry_price * 100
                mae_long = max(0, mae_long)  # Can't be negative

                # For SHORT: move to min low, MAE is worst rally before that
                move_short_pct = (entry_price - min_low) / entry_price * 100
                # Find where min low occurred
                min_low_idx = np.argmin(window_lows)
                # MAE for short is worst high BEFORE hitting min low
                if min_low_idx > 0:
                    mae_short = (np.max(window_highs[:min_low_idx + 1]) - entry_price) / entry_price * 100
                else:
                    mae_short = (window_highs[0] - entry_price) / entry_price * 100
                mae_short = max(0, mae_short)  # Can't be negative

            # Label assignment with MAE filtering
            if self.use_close_to_close:
                # Simple close-to-close logic
                if move_pct >= self.min_move_pct and mae_long <= self.max_mae_pct:
                    labels.append(EntryActionLegacy.BUY_CALL)  # LONG
                elif move_pct <= -self.min_move_pct and mae_short <= self.max_mae_pct:
                    labels.append(EntryActionLegacy.BUY_PUT)  # SHORT
                else:
                    labels.append(EntryActionLegacy.HOLD)
            else:
                # Extreme-based logic: compare quality of long vs short
                long_valid = move_long_pct >= self.min_move_pct and mae_long <= self.max_mae_pct
                short_valid = move_short_pct >= self.min_move_pct and mae_short <= self.max_mae_pct

                if long_valid and short_valid:
                    # Both valid - pick the one with better move/MAE ratio
                    long_ratio = move_long_pct / (mae_long + 0.01)
                    short_ratio = move_short_pct / (mae_short + 0.01)
                    if long_ratio >= short_ratio:
                        labels.append(EntryActionLegacy.BUY_CALL)
                    else:
                        labels.append(EntryActionLegacy.BUY_PUT)
                elif long_valid:
                    labels.append(EntryActionLegacy.BUY_CALL)
                elif short_valid:
                    labels.append(EntryActionLegacy.BUY_PUT)
                else:
                    labels.append(EntryActionLegacy.HOLD)

        return np.array(labels)

    def __len__(self) -> int:
        return len(self.valid_indices)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        data_idx = self.valid_indices[idx]
        start_idx = data_idx - self.context_len
        patch = self.data[start_idx:data_idx]
        label = self.labels[idx]
        return patch, label

    def get_class_distribution(self) -> Dict[str, int]:
        unique, counts = np.unique(self.labels, return_counts=True)
        dist = {}
        for action in EntryActionLegacy:
            count = counts[unique == action.value][0] if action.value in unique else 0
            dist[action.name] = int(count)
        return dist

    def get_label_stats(self) -> Dict[str, float]:
        """Get statistics about the labeling."""
        dist = self.get_class_distribution()
        total = sum(dist.values())
        return {
            "total_samples": total,
            "hold_pct": dist.get("HOLD", 0) / total * 100 if total > 0 else 0,
            "long_pct": dist.get("BUY_CALL", 0) / total * 100 if total > 0 else 0,
            "short_pct": dist.get("BUY_PUT", 0) / total * 100 if total > 0 else 0,
            "signal_pct": (dist.get("BUY_CALL", 0) + dist.get("BUY_PUT", 0)) / total * 100 if total > 0 else 0,
        }


class DirectionalMoveDataset(Dataset):
    """Dataset for entry policy training using directional move labeling."""

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
        min_move_pct: float = 0.05,
        min_consistency_pct: float = 1.0,
        wick_penalty_weight: float = 0.3,
        move_size_weight: float = 0.4,
        atm_offset: float = 0.0,
        otm_offset: float = 2.0,
        otm_buffer: float = 1.2,
        min_roi_threshold: float = 10.0,
    ):
        self.df = normalized_df
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

        self.opens = raw_ohlcv["open"]
        self.highs = raw_ohlcv["high"]
        self.lows = raw_ohlcv["low"]
        self.closes = raw_ohlcv["close"]
        self.timestamps = normalized_df.index.to_pydatetime()

        self.lookahead = max_candles * candle_agg_minutes

        self.valid_indices = []
        for i in range(context_len, len(self.df) - self.lookahead):
            self.valid_indices.append(i)

        self.labels = self._compute_labels()

    def _aggregate_candles(self, start_idx: int, num_candles: int) -> List[Dict]:
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

            body_size = abs(c - o)
            candle_range = h - l if h > l else 0.0001
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
        if not candles:
            return False, 0.0, 0.0, 0.0

        num_candles = len(candles)

        if is_call:
            consistent_candles = sum(1 for c in candles if c["is_green"])
        else:
            consistent_candles = sum(1 for c in candles if c["is_red"])

        consistency_score = consistent_candles / num_candles

        if consistency_score < self.min_consistency_pct:
            return False, consistency_score, 0.0, 0.0

        half_idx = max(1, num_candles // 2)
        first_half = candles[:half_idx]

        if is_call:
            first_half_consistent = all(c["is_green"] for c in first_half)
        else:
            first_half_consistent = all(c["is_red"] for c in first_half)

        if not first_half_consistent:
            return False, consistency_score, 0.0, 0.0

        avg_wick_ratio = np.mean([c["wick_ratio"] for c in candles])
        wick_score = 1.0 - avg_wick_ratio

        start_price = candles[0]["open"]
        end_price = candles[-1]["close"]
        if is_call:
            move_size = (end_price - start_price) / start_price * 100
        else:
            move_size = (start_price - end_price) / start_price * 100

        if move_size < self.min_move_pct:
            return False, consistency_score, wick_score, move_size

        return True, consistency_score, wick_score, move_size

    def _compute_labels(self) -> np.ndarray:
        labels = []

        for idx in tqdm(self.valid_indices, desc="Computing directional labels"):
            spot_now = self.closes[idx]
            timestamp_decision = self.timestamps[idx]

            best_call_score = 0.0
            best_put_score = 0.0
            best_call_candles = 0
            best_put_candles = 0
            call_roi = 0.0
            put_roi = 0.0
            call_prefers_otm = False
            put_prefers_otm = False

            for num_candles in range(self.max_candles, self.min_candles - 1, -1):
                candles = self._aggregate_candles(idx + 1, num_candles)

                if len(candles) < num_candles:
                    continue

                is_valid_call, call_consistency, call_wick, call_move = self._evaluate_direction(
                    candles, is_call=True
                )
                if is_valid_call:
                    call_score = (
                        self.move_size_weight * min(call_move / 1.0, 1.0) +
                        (1 - self.move_size_weight - self.wick_penalty_weight) * call_consistency +
                        self.wick_penalty_weight * call_wick
                    )
                    call_score *= (1 + 0.1 * (num_candles - self.min_candles))

                    if call_score > best_call_score:
                        best_call_score = call_score
                        best_call_candles = num_candles

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

            if best_call_score > 0 or best_put_score > 0:
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


# =============================================================================
# Entry Training Functions
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

    num_classes = 5 if args.model_type in ("entry-5class", "entry-directional") else 3

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
            "lookahead": getattr(args, 'lookahead', 15),
        },
    }

    if args.model_type == "entry-5class":
        checkpoint["config"].update({
            "min_roi_threshold": args.min_roi_threshold,
            "otm_buffer": args.otm_buffer,
        })

    torch.save(checkpoint, path)
    print(f"  Saved checkpoint to {path}")


# =============================================================================
# Regression Training Functions
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
        targets = targets.to(device).float().unsqueeze(-1)

        optimizer.zero_grad()

        preds = policy(embeddings)
        loss = F.mse_loss(preds, targets)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), max_grad_norm)
        optimizer.step()

        total_loss += loss.item() * len(targets)
        total_samples += len(targets)
        all_preds.append(preds.detach().cpu())
        all_targets.append(targets.cpu())

    avg_loss = total_loss / total_samples

    all_preds = torch.cat(all_preds).squeeze()
    all_targets = torch.cat(all_targets).squeeze()

    correlation = torch.corrcoef(torch.stack([all_preds, all_targets]))[0, 1].item()
    direction_correct = ((all_preds > 0) == (all_targets > 0)).float().mean().item()
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

    correlation = torch.corrcoef(torch.stack([all_preds, all_targets]))[0, 1].item()
    direction_correct = ((all_preds > 0) == (all_targets > 0)).float().mean().item()
    mae = (all_preds - all_targets).abs().mean().item()

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


# =============================================================================
# Entry Argument Handling
# =============================================================================


def add_entry_args(parser: argparse.ArgumentParser) -> None:
    """Add entry policy-specific arguments to parser."""
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
    parser.add_argument(
        "--dropout",
        type=float,
        default=0.1,
        help="Dropout probability for policy network",
    )

    # Label generation parameters
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
        "--long-threshold-pct",
        type=float,
        default=0.363,
        help="LONG threshold %% for percentile-based labeling (99th pctl 2023-2025)",
    )
    parser.add_argument(
        "--short-threshold-pct",
        type=float,
        default=0.400,
        help="SHORT threshold %% for percentile-based labeling (1st pctl 2023-2025)",
    )
    parser.add_argument(
        "--scale-factor",
        type=float,
        default=100.0,
        help="Scale factor for regression targets",
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
        help="OTM must beat ATM by this factor",
    )
    parser.add_argument(
        "--atm-offset",
        type=float,
        default=0.0,
        help="ATM strike offset from spot",
    )
    parser.add_argument(
        "--otm-offset",
        type=float,
        default=2.0,
        help="OTM strike offset from ATM in dollars",
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
        help="Execution delay in minutes",
    )
    parser.add_argument(
        "--slippage-pct",
        type=float,
        default=0.5,
        help="Slippage as %% of option price",
    )
    parser.add_argument(
        "--no-time-decay",
        action="store_true",
        default=False,
        help="Disable time-decay weighting in ROI labeling",
    )

    # Directional move parameters
    parser.add_argument(
        "--candle-agg-minutes",
        type=int,
        default=5,
        help="Minutes to aggregate into each smoothed candle",
    )
    parser.add_argument(
        "--min-candles",
        type=int,
        default=3,
        help="Minimum smoothed candles to check",
    )
    parser.add_argument(
        "--max-candles",
        type=int,
        default=6,
        help="Maximum smoothed candles to check",
    )
    parser.add_argument(
        "--min-move-pct",
        type=float,
        default=0.05,
        help="Minimum price move %% for directional signal",
    )
    parser.add_argument(
        "--min-consistency-pct",
        type=float,
        default=1.0,
        help="Required %% of candles in right direction",
    )
    parser.add_argument(
        "--wick-penalty-weight",
        type=float,
        default=0.3,
        help="Weight for wick penalty in score",
    )
    parser.add_argument(
        "--move-size-weight",
        type=float,
        default=0.4,
        help="Weight for move size in score",
    )

    # MAE (Maximum Adverse Excursion) labeling parameters
    parser.add_argument(
        "--max-mae-pct",
        type=float,
        default=0.05,
        help="Maximum adverse excursion %% allowed for MAE labeling (lower = stricter)",
    )
    parser.add_argument(
        "--use-close-to-close",
        action="store_true",
        default=False,
        help="Use close-to-close for MAE move calculation instead of high/low extremes",
    )

    # Focal Loss parameters
    parser.add_argument(
        "--use-focal-loss",
        action="store_true",
        default=True,
        help="Use Focal Loss",
    )
    parser.add_argument(
        "--focal-gamma",
        type=float,
        default=2.0,
        help="Focal Loss gamma",
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
        help="Focal Loss alpha weight for BUY_CALL/BUY_PUT",
    )
    parser.add_argument(
        "--focal-alpha-atm",
        type=float,
        default=1.0,
        help="Focal Loss alpha weight for ATM classes",
    )
    parser.add_argument(
        "--focal-alpha-otm",
        type=float,
        default=1.2,
        help="Focal Loss alpha weight for OTM classes",
    )
    parser.add_argument(
        "--balanced-sampling",
        action="store_true",
        help="Use class-balanced sampling with WeightedRandomSampler",
    )
    parser.add_argument(
        "--use-class-weights",
        action="store_true",
        help="Use inverse-frequency class weights in loss function (helps balance LONG/SHORT predictions)",
    )
