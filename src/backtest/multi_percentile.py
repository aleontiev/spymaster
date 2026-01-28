"""
Multi-scale percentile policy fusion backtester.

Combines multiple independently-trained percentile policy models into a single
unified trading bot using voting fusion.
"""
import random
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Tuple
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from src.backtest.types import (
    BacktestConfig,
    PendingEntry,
    Position,
    PositionType,
    Trade,
)
from src.execution.heuristic import HeuristicConfig, HeuristicModel
from src.execution.options_provider import RealOptionsProvider
from src.strategy.fusion_config import FusedSignal, Signal


class VWAPBreachType(Enum):
    """Type of VWAP breach signal."""
    NONE = 0
    BULLISH = 1  # Price crossed above VWAP with volume confirmation
    BEARISH = 2  # Price crossed below VWAP with volume confirmation
    BULLISH_POTENTIAL = 3  # Price crossed above VWAP but lacks volume (needs continuation)
    BEARISH_POTENTIAL = 4  # Price crossed below VWAP but lacks volume (needs continuation)


class ORBBreachType(Enum):
    """Type of Opening Range Breakout signal."""
    NONE = 0
    BULLISH = 1  # Strong breakout above OR-high, enter immediately
    BEARISH = 2  # Strong breakout below OR-low, enter immediately
    BULLISH_POTENTIAL = 3  # Price broke above OR-high, needs continuation
    BEARISH_POTENTIAL = 4  # Price broke below OR-low, needs continuation


@dataclass
class VWAPBreachSignal:
    """VWAP breach signal details."""
    breach_type: VWAPBreachType
    vwap_value: float
    candle_close: float
    candle_open: float
    candle_high: float
    candle_low: float
    volume: float
    continuation_volume: float  # Total volume including continuation candles
    volume_ratio: float  # Ratio vs recent average
    is_potential: bool = False  # True if breach needs continuation confirmation
    band_level: Optional[float] = None  # The band level that was breached (for +-2σ breaches)


@dataclass
class ORBBreachSignal:
    """Opening Range Breakout signal details."""
    breach_type: ORBBreachType
    or_high: float  # Opening range high
    or_low: float   # Opening range low
    breach_level: float  # The level that was breached (OR-high or OR-low)
    candle_close: float
    candle_open: float
    candle_high: float
    candle_low: float
    volume: float
    volume_ratio: float  # Ratio vs recent average


def calculate_opening_range(
    ohlcv_df: pd.DataFrame,
    or_minutes: int = 15,
) -> Tuple[Optional[float], Optional[float]]:
    """
    Calculate the Opening Range (OR) high and low for the first N minutes of the trading day.

    The opening range is defined as the high and low of the first `or_minutes` minutes
    after market open (9:30 AM ET).

    Args:
        ohlcv_df: OHLCV DataFrame with datetime index
        or_minutes: Number of minutes for opening range (default 15)

    Returns:
        Tuple of (or_high, or_low) or (None, None) if insufficient data
    """
    if len(ohlcv_df) < or_minutes:
        return None, None

    # Get the first N minutes of data
    or_data = ohlcv_df.iloc[:or_minutes]

    or_high = or_data['high'].max()
    or_low = or_data['low'].min()

    return or_high, or_low


def detect_orb_breach(
    ohlcv_df: pd.DataFrame,
    idx: int,
    or_high: float,
    or_low: float,
    or_margin: float = 0.02,  # 2 cents margin for breach detection
    lookback_minutes: int = 20,
) -> ORBBreachSignal:
    """
    Detect if current candle breaches the Opening Range levels.

    For ORB, we use two entry modes:
    - Immediate entry: Strong breakout candle that opens near OR level and closes well beyond
    - Continuation entry: First candle breaches OR, then continuation candle confirms the move

    Args:
        ohlcv_df: OHLCV DataFrame
        idx: Current index in the DataFrame
        or_high: Opening range high
        or_low: Opening range low
        or_margin: Margin for breach detection (default 2 cents)
        lookback_minutes: Lookback for calculating average volume

    Returns:
        ORBBreachSignal with breach details
    """
    # ORB trading disabled - always return NONE
    return ORBBreachSignal(
        breach_type=ORBBreachType.NONE,
        or_high=or_high, or_low=or_low, breach_level=0,
        candle_close=0, candle_open=0, candle_high=0, candle_low=0,
        volume=0, volume_ratio=0,
    )

    if idx < 1:
        return ORBBreachSignal(
            breach_type=ORBBreachType.NONE,
            or_high=or_high, or_low=or_low, breach_level=0,
            candle_close=0, candle_open=0, candle_high=0, candle_low=0,
            volume=0, volume_ratio=0,
        )

    current = ohlcv_df.iloc[idx]
    close = current['close']
    open_price = current['open']
    high = current['high']
    low = current['low']
    volume = current.get('volume', 0)

    # Calculate average volume over lookback period
    lookback_start = max(0, idx - lookback_minutes)
    avg_volume = ohlcv_df.iloc[lookback_start:idx]['volume'].mean()
    if avg_volume <= 0:
        avg_volume = volume
    volume_ratio = volume / avg_volume if avg_volume > 0 else 0

    body = abs(close - open_price)
    candle_range = high - low

    # Require strong body (at least 40% of range - not indecisive)
    if candle_range <= 0 or body / candle_range < 0.4:
        return ORBBreachSignal(
            breach_type=ORBBreachType.NONE,
            or_high=or_high, or_low=or_low, breach_level=0,
            candle_close=close, candle_open=open_price, candle_high=high, candle_low=low,
            volume=volume, volume_ratio=volume_ratio,
        )

    # Require decent volume (at least 1.5x average)
    if volume_ratio < 1.5:
        return ORBBreachSignal(
            breach_type=ORBBreachType.NONE,
            or_high=or_high, or_low=or_low, breach_level=0,
            candle_close=close, candle_open=open_price, candle_high=high, candle_low=low,
            volume=volume, volume_ratio=volume_ratio,
        )

    # Calculate average body for comparison
    recent_bodies = (ohlcv_df.iloc[lookback_start:idx]['close'] - ohlcv_df.iloc[lookback_start:idx]['open']).abs()
    avg_body = recent_bodies.mean() if len(recent_bodies) > 0 else body

    # Require body to be at least 1.5x average body
    if body < avg_body * 1.5:
        return ORBBreachSignal(
            breach_type=ORBBreachType.NONE,
            or_high=or_high, or_low=or_low, breach_level=0,
            candle_close=close, candle_open=open_price, candle_high=high, candle_low=low,
            volume=volume, volume_ratio=volume_ratio,
        )

    # Check for BULLISH breach (price breaks above OR-high)
    # Candle must OPEN within/near OR range and CLOSE above OR-high (actual breakout)
    is_green = close > open_price
    opens_within_or_near_range = open_price <= or_high + 0.10  # Must open at or below OR-high (10c tolerance)
    closes_above_or_high = close > or_high + or_margin

    if is_green and opens_within_or_near_range and closes_above_or_high:
        # Check body-to-wick ratio (upper wick should be small)
        upper_wick = high - close
        body_to_wick = body / upper_wick if upper_wick > 0.001 else 100.0
        if body_to_wick >= 3.0:  # Body at least 3x opposing wick
            # BREAKOUT CANDLE: Opens near OR-high and closes well above
            opens_near_or = abs(open_price - or_high) <= 0.03  # Within 3 cents of OR-high
            closes_well_above = close >= or_high + max(0.15, avg_body * 1.5)  # At least 15c or 1.5x avg body above
            is_breakout_candle = opens_near_or and closes_well_above

            if is_breakout_candle:
                # Immediate entry on strong breakout candle
                return ORBBreachSignal(
                    breach_type=ORBBreachType.BULLISH,
                    or_high=or_high,
                    or_low=or_low,
                    breach_level=or_high,
                    candle_close=close,
                    candle_open=open_price,
                    candle_high=high,
                    candle_low=low,
                    volume=volume,
                    volume_ratio=volume_ratio,
                )
            else:
                # Potential - wait for continuation
                return ORBBreachSignal(
                    breach_type=ORBBreachType.BULLISH_POTENTIAL,
                    or_high=or_high,
                    or_low=or_low,
                    breach_level=or_high,
                    candle_close=close,
                    candle_open=open_price,
                    candle_high=high,
                    candle_low=low,
                    volume=volume,
                    volume_ratio=volume_ratio,
                )

    # Check for BEARISH breach (price breaks below OR-low)
    # Candle must OPEN within/near OR range and CLOSE below OR-low (actual breakout)
    is_red = close < open_price
    opens_within_or_near_range = open_price >= or_low - 0.10  # Must open at or above OR-low (10c tolerance)
    closes_below_or_low = close < or_low - or_margin

    if is_red and opens_within_or_near_range and closes_below_or_low:
        # Check body-to-wick ratio (lower wick should be small)
        lower_wick = close - low
        body_to_wick = body / lower_wick if lower_wick > 0.001 else 100.0
        if body_to_wick >= 3.0:  # Body at least 3x opposing wick
            # BREAKOUT CANDLE: Opens near OR-low and closes well below
            opens_near_or = abs(open_price - or_low) <= 0.03  # Within 3 cents of OR-low
            closes_well_below = close <= or_low - max(0.15, avg_body * 1.5)  # At least 15c or 1.5x avg body below
            is_breakout_candle = opens_near_or and closes_well_below

            if is_breakout_candle:
                # Immediate entry on strong breakout candle
                return ORBBreachSignal(
                    breach_type=ORBBreachType.BEARISH,
                    or_high=or_high,
                    or_low=or_low,
                    breach_level=or_low,
                    candle_close=close,
                    candle_open=open_price,
                    candle_high=high,
                    candle_low=low,
                    volume=volume,
                    volume_ratio=volume_ratio,
                )
            else:
                # Potential - wait for continuation
                return ORBBreachSignal(
                    breach_type=ORBBreachType.BEARISH_POTENTIAL,
                    or_high=or_high,
                    or_low=or_low,
                    breach_level=or_low,
                    candle_close=close,
                    candle_open=open_price,
                    candle_high=high,
                    candle_low=low,
                    volume=volume,
                    volume_ratio=volume_ratio,
                )

    return ORBBreachSignal(
        breach_type=ORBBreachType.NONE,
        or_high=or_high, or_low=or_low, breach_level=0,
        candle_close=close, candle_open=open_price, candle_high=high, candle_low=low,
        volume=volume, volume_ratio=volume_ratio,
    )


def check_orb_continuation_confirmation(
    ohlcv_df: pd.DataFrame,
    idx: int,
    potential_breach: ORBBreachSignal,
    min_body_ratio: float = 0.3,  # Min body/range ratio for continuation candle
) -> bool:
    """
    Check if current candle confirms an ORB breach from the previous candle.

    For bullish ORB breach: continuation candle must be green and close above OR-high
    For bearish ORB breach: continuation candle must be red and close below OR-low

    Args:
        ohlcv_df: OHLCV DataFrame
        idx: Current index (the candle after the potential breach)
        potential_breach: The potential breach signal from previous candle
        min_body_ratio: Minimum body/range ratio for the continuation candle

    Returns:
        True if continuation confirms the breach
    """
    if idx < 1:
        return False

    current = ohlcv_df.iloc[idx]
    current_close = current['close']
    current_open = current['open']
    current_high = current['high']
    current_low = current['low']

    current_range = current_high - current_low
    current_body = abs(current_close - current_open)

    # Check that candle has some body (not indecisive/doji)
    has_body = current_range > 0 and (current_body / current_range) >= min_body_ratio

    if potential_breach.breach_type == ORBBreachType.BULLISH_POTENTIAL:
        # Bullish confirmation: green candle with body fully above OR-high
        is_green = current_close > current_open
        body_low = min(current_open, current_close)
        body_above_or = body_low > potential_breach.or_high  # No tolerance - body must be fully above
        closes_above_or = current_close > potential_breach.or_high
        return is_green and has_body and body_above_or and closes_above_or

    elif potential_breach.breach_type == ORBBreachType.BEARISH_POTENTIAL:
        # Bearish confirmation: red candle with body fully below OR-low
        is_red = current_close < current_open
        body_high = max(current_open, current_close)
        body_below_or = body_high < potential_breach.or_low  # No tolerance - body must be fully below
        closes_below_or = current_close < potential_breach.or_low
        return is_red and has_body and body_below_or and closes_below_or

    return False


def calculate_vwap_series(ohlcv_df: pd.DataFrame) -> pd.Series:
    """
    Calculate VWAP anchored at start of each trading day.

    VWAP = cumulative(typical_price * volume) / cumulative(volume)
    Resets at start of each new trading day.
    """
    vwap, _, _, _, _ = calculate_vwap_with_bands(ohlcv_df)
    return vwap


def calculate_vwap_with_bands(ohlcv_df: pd.DataFrame) -> Tuple[pd.Series, pd.Series, pd.Series, pd.Series, pd.Series]:
    """
    Calculate VWAP with standard deviation bands anchored at start of each trading day.

    VWAP = cumulative(typical_price * volume) / cumulative(volume)
    Standard deviation bands at +-1 and +-2 standard deviations.
    Resets at start of each new trading day.

    Returns:
        Tuple of (vwap, upper_1sd, lower_1sd, upper_2sd, lower_2sd) Series
    """
    typical_price = (ohlcv_df['high'] + ohlcv_df['low'] + ohlcv_df['close']) / 3
    volume = ohlcv_df['volume']

    vwap_values = []
    upper_1sd_values = []
    lower_1sd_values = []
    upper_2sd_values = []
    lower_2sd_values = []

    current_date = None
    cum_tp_vol = 0.0
    cum_vol = 0.0
    cum_tp_sq_vol = 0.0  # For variance calculation

    for i in range(len(ohlcv_df)):
        idx = ohlcv_df.index[i]
        if hasattr(idx, 'date'):
            row_date = idx.date()
        else:
            row_date = pd.Timestamp(idx).date()

        # Reset at new day
        if row_date != current_date:
            current_date = row_date
            cum_tp_vol = 0.0
            cum_vol = 0.0
            cum_tp_sq_vol = 0.0

        tp = typical_price.iloc[i]
        vol = volume.iloc[i]
        cum_tp_vol += tp * vol
        cum_vol += vol
        cum_tp_sq_vol += (tp ** 2) * vol

        if cum_vol > 0:
            vwap = cum_tp_vol / cum_vol
            # Variance = E[X^2] - E[X]^2 (volume-weighted)
            variance = (cum_tp_sq_vol / cum_vol) - (vwap ** 2)
            std_dev = np.sqrt(max(0, variance))  # Ensure non-negative

            vwap_values.append(vwap)
            upper_1sd_values.append(vwap + std_dev)
            lower_1sd_values.append(vwap - std_dev)
            upper_2sd_values.append(vwap + 2 * std_dev)
            lower_2sd_values.append(vwap - 2 * std_dev)
        else:
            tp_val = typical_price.iloc[i]
            vwap_values.append(tp_val)
            upper_1sd_values.append(tp_val)
            lower_1sd_values.append(tp_val)
            upper_2sd_values.append(tp_val)
            lower_2sd_values.append(tp_val)

    return (
        pd.Series(vwap_values, index=ohlcv_df.index),
        pd.Series(upper_1sd_values, index=ohlcv_df.index),
        pd.Series(lower_1sd_values, index=ohlcv_df.index),
        pd.Series(upper_2sd_values, index=ohlcv_df.index),
        pd.Series(lower_2sd_values, index=ohlcv_df.index),
    )


def detect_vwap_breach(
    ohlcv_df: pd.DataFrame,
    vwap_series: pd.Series,
    idx: int,
    lookback_minutes: int = 5,
    vwap_margin: float = 0.02,  # Margin for open/close near VWAP (2 cents)
    upper_1sd_series: Optional[pd.Series] = None,  # +1σ band for immediate entry threshold
    lower_1sd_series: Optional[pd.Series] = None,  # -1σ band for immediate entry threshold
    upper_2sd_series: Optional[pd.Series] = None,  # +2σ band for band reversion detection
    lower_2sd_series: Optional[pd.Series] = None,  # -2σ band for band reversion detection
) -> VWAPBreachSignal:
    """
    Detect VWAP breach - strictly requires body to cross VWAP.
    Also detects +-2σ band reversion breaches (mean reversion plays).

    VWAP Bullish breach:
    - Candle opens at or below VWAP (within margin)
    - Candle closes at or above VWAP (within margin)
    - Body must cross VWAP

    VWAP Bearish breach:
    - Candle opens at or above VWAP (within margin)
    - Candle closes at or below VWAP (within margin)
    - Body must cross VWAP

    Upper band reversion (bearish - SHORT toward VWAP):
    - Candle opens at or above +2σ band
    - Candle closes at or below +2σ band
    - Body crosses the band back toward VWAP

    Lower band reversion (bullish - LONG toward VWAP):
    - Candle opens at or below -2σ band
    - Candle closes at or above -2σ band
    - Body crosses the band back toward VWAP

    Args:
        ohlcv_df: OHLCV DataFrame
        vwap_series: Pre-calculated VWAP series
        idx: Current index in DataFrame
        lookback_minutes: Minutes to look back for average volume
        vwap_margin: Tolerance for open/close near VWAP (default 0.02 = 2 cents)
        upper_2sd_series: Optional +2σ band series for band reversion detection
        lower_2sd_series: Optional -2σ band series for band reversion detection

    Returns:
        VWAPBreachSignal with breach details
    """
    if idx < lookback_minutes:
        return VWAPBreachSignal(
            breach_type=VWAPBreachType.NONE,
            vwap_value=0, candle_close=0, candle_open=0,
            candle_high=0, candle_low=0, volume=0,
            continuation_volume=0, volume_ratio=0,
        )

    # Get current candle data
    current = ohlcv_df.iloc[idx]
    vwap = vwap_series.iloc[idx]

    close = current['close']
    open_price = current['open']
    high = current['high']
    low = current['low']
    volume = current['volume']

    candle_range = high - low
    if candle_range <= 0:
        return VWAPBreachSignal(
            breach_type=VWAPBreachType.NONE,
            vwap_value=vwap, candle_close=close, candle_open=open_price,
            candle_high=high, candle_low=low, volume=volume,
            continuation_volume=volume, volume_ratio=0,
        )

    # Calculate candle body
    body = abs(close - open_price)
    if body <= 0:
        body = 0.0001  # Avoid division by zero for doji candles

    # Calculate average volume over lookback period
    lookback_start = max(0, idx - lookback_minutes)
    avg_volume = ohlcv_df.iloc[lookback_start:idx]['volume'].mean()
    if avg_volume <= 0:
        avg_volume = volume

    # Count continuation candles and their volume
    continuation_volume = volume
    is_bullish_candle = close > open_price
    is_bearish_candle = close < open_price

    # Look back for continuation candles (same direction)
    for j in range(idx - 1, max(0, idx - lookback_minutes) - 1, -1):
        prev = ohlcv_df.iloc[j]
        prev_close = prev['close']
        prev_open = prev['open']
        prev_bullish = prev_close > prev_open
        prev_bearish = prev_close < prev_open

        if is_bullish_candle and prev_bullish:
            continuation_volume += prev['volume']
        elif is_bearish_candle and prev_bearish:
            continuation_volume += prev['volume']
        else:
            break  # Stop at first non-continuation candle

    # Volume ratio: compare current candle's volume to average
    volume_ratio = volume / avg_volume if avg_volume > 0 else 0

    # Check for BULLISH breach (body crosses VWAP upward)
    # Open at or below VWAP (with margin), close at or above VWAP (with margin)
    # Must be a green candle (close > open)
    open_at_or_below_vwap = open_price <= vwap + vwap_margin
    close_at_or_above_vwap = close >= vwap - vwap_margin
    is_bullish_breach = open_at_or_below_vwap and close_at_or_above_vwap and close > open_price

    if is_bullish_breach:
        # Check for immediate entry conditions:
        # 1. Body much longer than opposing wick (upper wick for bullish)
        upper_wick = high - close  # Opposing wick for bullish
        body_to_wick_ratio = body / upper_wick if upper_wick > 0.001 else 100.0
        strong_body = body_to_wick_ratio >= 3.5  # Body at least 3.5x opposing wick

        # Check volume conditions - require current candle to have high volume
        high_volume_current = volume_ratio >= 2.5  # Current candle has 2.5x+ avg volume

        # Check minimum body size (body >= 1.5x average body)
        lookback_start = max(0, idx - lookback_minutes)
        recent_bodies = (ohlcv_df.iloc[lookback_start:idx]['close'] - ohlcv_df.iloc[lookback_start:idx]['open']).abs()
        avg_body = recent_bodies.mean() if len(recent_bodies) > 0 else body
        large_body = body >= avg_body * 1.5

        # BREAKOUT CANDLE: Opens near VWAP and closes well above
        # This is a strong breakout signal - enter on this candle directly
        opens_near_vwap = abs(open_price - vwap) <= 0.03  # Within 3 cents of VWAP
        closes_well_above = close >= vwap + max(0.15, avg_body * 1.5)  # At least 15c or 1.5x avg body above
        is_breakout_candle = opens_near_vwap and closes_well_above and strong_body and large_body

        # Check 1σ band threshold for immediate entry
        # Enter if close moved at least 25% of the way from VWAP to +1σ
        # But don't enter if close exceeds +1σ (overextended)
        upper_1sd = upper_1sd_series.iloc[idx] if upper_1sd_series is not None else None
        meets_1sd_threshold = False
        is_overextended = False
        if upper_1sd is not None and upper_1sd > vwap:
            band_distance = upper_1sd - vwap
            move_from_vwap = close - vwap
            move_pct = move_from_vwap / band_distance if band_distance > 0 else 0
            meets_1sd_threshold = move_pct >= 0.25  # At least 25% toward +1σ
            is_overextended = close > upper_1sd  # Beyond +1σ is overextended

        # Immediate entry if:
        # - Meets 25% threshold toward +1σ AND not overextended, OR
        # - (breakout candle) OR (strong body AND high volume AND large body) - but still check overextension
        can_enter_immediate = (
            (meets_1sd_threshold and not is_overextended) or
            ((is_breakout_candle or (strong_body and high_volume_current and large_body)) and not is_overextended)
        )

        if can_enter_immediate:
            return VWAPBreachSignal(
                breach_type=VWAPBreachType.BULLISH,  # Confirmed - enter immediately
                vwap_value=vwap,
                candle_close=close,
                candle_open=open_price,
                candle_high=high,
                candle_low=low,
                volume=volume,
                continuation_volume=continuation_volume,
                volume_ratio=volume_ratio,
                is_potential=False,
            )
        else:
            # Return POTENTIAL - wait for confirmation candle
            return VWAPBreachSignal(
                breach_type=VWAPBreachType.BULLISH_POTENTIAL,
                vwap_value=vwap,
                candle_close=close,
                candle_open=open_price,
                candle_high=high,
                candle_low=low,
                volume=volume,
                continuation_volume=continuation_volume,
                volume_ratio=volume_ratio,
                is_potential=True,
            )

    # Check for BEARISH breach (body crosses VWAP downward)
    # Open at or above VWAP (with margin), close at or below VWAP (with margin)
    # Must be a red candle (close < open)
    open_at_or_above_vwap = open_price >= vwap - vwap_margin
    close_at_or_below_vwap = close <= vwap + vwap_margin
    is_bearish_breach = open_at_or_above_vwap and close_at_or_below_vwap and close < open_price

    if is_bearish_breach:
        # Check for immediate entry conditions:
        # 1. Body much longer than opposing wick (lower wick for bearish)
        lower_wick = close - low  # Opposing wick for bearish (close is lower than open)
        body_to_wick_ratio = body / lower_wick if lower_wick > 0.001 else 100.0
        strong_body = body_to_wick_ratio >= 3.5  # Body at least 3.5x opposing wick

        # Check volume conditions - require current candle to have high volume
        high_volume_current = volume_ratio >= 2.5  # Current candle has 2.5x+ avg volume

        # Check minimum body size (body >= 1.5x average body)
        lookback_start = max(0, idx - lookback_minutes)
        recent_bodies = (ohlcv_df.iloc[lookback_start:idx]['close'] - ohlcv_df.iloc[lookback_start:idx]['open']).abs()
        avg_body = recent_bodies.mean() if len(recent_bodies) > 0 else body
        large_body = body >= avg_body * 1.5

        # BREAKOUT CANDLE: Opens near VWAP and closes well below
        # This is a strong breakout signal - enter on this candle directly
        opens_near_vwap = abs(open_price - vwap) <= 0.03  # Within 3 cents of VWAP
        closes_well_below = close <= vwap - max(0.15, avg_body * 1.5)  # At least 15c or 1.5x avg body below
        is_breakout_candle = opens_near_vwap and closes_well_below and strong_body and large_body

        # Check 1σ band threshold for immediate entry
        # Enter if close moved at least 25% of the way from VWAP to -1σ
        # But don't enter if close goes below -1σ (overextended)
        lower_1sd = lower_1sd_series.iloc[idx] if lower_1sd_series is not None else None
        meets_1sd_threshold = False
        is_overextended = False
        if lower_1sd is not None and lower_1sd < vwap:
            band_distance = vwap - lower_1sd
            move_from_vwap = vwap - close
            move_pct = move_from_vwap / band_distance if band_distance > 0 else 0
            meets_1sd_threshold = move_pct >= 0.25  # At least 25% toward -1σ
            is_overextended = close < lower_1sd  # Beyond -1σ is overextended

        # Immediate entry if:
        # - Meets 25% threshold toward -1σ AND not overextended, OR
        # - (breakout candle) OR (strong body AND high volume AND large body) - but still check overextension
        can_enter_immediate = (
            (meets_1sd_threshold and not is_overextended) or
            ((is_breakout_candle or (strong_body and high_volume_current and large_body)) and not is_overextended)
        )

        if can_enter_immediate:
            return VWAPBreachSignal(
                breach_type=VWAPBreachType.BEARISH,  # Confirmed - enter immediately
                vwap_value=vwap,
                candle_close=close,
                candle_open=open_price,
                candle_high=high,
                candle_low=low,
                volume=volume,
                continuation_volume=continuation_volume,
                volume_ratio=volume_ratio,
                is_potential=False,
            )
        else:
            # Return POTENTIAL - wait for confirmation candle
            return VWAPBreachSignal(
                breach_type=VWAPBreachType.BEARISH_POTENTIAL,
                vwap_value=vwap,
                candle_close=close,
                candle_open=open_price,
                candle_high=high,
                candle_low=low,
                volume=volume,
                continuation_volume=continuation_volume,
                volume_ratio=volume_ratio,
                is_potential=True,
            )

    # No breach detected
    return VWAPBreachSignal(
        breach_type=VWAPBreachType.NONE,
        vwap_value=vwap,
        candle_close=close,
        candle_open=open_price,
        candle_high=high,
        candle_low=low,
        volume=volume,
        continuation_volume=continuation_volume,
        volume_ratio=volume_ratio,
        is_potential=False,
    )


def check_continuation_confirmation(
    ohlcv_df: pd.DataFrame,
    idx: int,
    potential_breach: VWAPBreachSignal,
    min_body_ratio: float = 0.3,  # Min body/range ratio for continuation candle
    lookback_minutes: int = 20,  # Lookback for calculating averages
    abnormal_move_threshold: float = 5.0,  # 5x average body = abnormal
    abnormal_volume_required: float = 5.0,  # Require 5x volume if abnormal move
) -> bool:
    """
    Check if current candle confirms a potential VWAP breach from the previous candle.

    For bullish potential breach: current candle must be entirely above VWAP
    - Must be green (close > open)
    - Must have some body (not indecisive)
    - Must OPEN above VWAP (candle starts on bullish side)
    - Must CLOSE above VWAP (candle ends on bullish side)
    - If body is 5x+ average, must have 5x+ volume (filter out fake moves)

    For bearish potential breach: current candle must be entirely below VWAP
    - Must be red (close < open)
    - Must have some body (not indecisive)
    - Must OPEN below VWAP (candle starts on bearish side)
    - Must CLOSE below VWAP (candle ends on bearish side)
    - If body is 5x+ average, must have 5x+ volume (filter out fake moves)

    Args:
        ohlcv_df: OHLCV DataFrame
        idx: Current index (the candle after the potential breach)
        potential_breach: The potential breach signal from previous candle
        min_body_ratio: Minimum body/range ratio for the continuation candle
        lookback_minutes: Lookback period for calculating average body and volume
        abnormal_move_threshold: Body must be this many times average to be "abnormal"
        abnormal_volume_required: If abnormal move, require this many times average volume

    Returns:
        True if continuation confirms the breach
    """
    if idx < 1:
        return False

    current = ohlcv_df.iloc[idx]
    current_close = current['close']
    current_open = current['open']
    current_high = current['high']
    current_low = current['low']
    current_volume = current.get('volume', 0)

    current_range = current_high - current_low
    current_body = abs(current_close - current_open)

    # Check that candle has some body (not indecisive/doji)
    has_body = current_range > 0 and (current_body / current_range) >= min_body_ratio

    # Calculate average body and volume over lookback period
    lookback_start = max(0, idx - lookback_minutes)
    if lookback_start < idx:
        recent_data = ohlcv_df.iloc[lookback_start:idx]
        avg_body = (recent_data['close'] - recent_data['open']).abs().mean()
        avg_volume = recent_data['volume'].mean() if 'volume' in recent_data.columns else 0
    else:
        avg_body = current_body
        avg_volume = current_volume

    # Check for abnormal move without corresponding volume
    # If candle body is 5x+ average, require 5x+ volume
    is_abnormal_move = avg_body > 0 and current_body >= abnormal_move_threshold * avg_body
    has_abnormal_volume = avg_volume > 0 and current_volume >= abnormal_volume_required * avg_volume

    if is_abnormal_move and not has_abnormal_volume:
        # Abnormal price move without volume - reject as likely fake move
        return False

    if potential_breach.breach_type == VWAPBreachType.BULLISH_POTENTIAL:
        # Bullish confirmation: green candle with body fully above VWAP
        is_green = current_close > current_open
        # Body must be entirely above VWAP (no tolerance)
        body_low = min(current_open, current_close)
        body_above_vwap = body_low > potential_breach.vwap_value
        closes_above_vwap = current_close > potential_breach.vwap_value
        return is_green and has_body and body_above_vwap and closes_above_vwap

    elif potential_breach.breach_type == VWAPBreachType.BEARISH_POTENTIAL:
        # Bearish confirmation: red candle with body fully below VWAP
        is_red = current_close < current_open
        # Body must be entirely below VWAP (no tolerance)
        body_high = max(current_open, current_close)
        body_below_vwap = body_high < potential_breach.vwap_value
        closes_below_vwap = current_close < potential_breach.vwap_value
        return is_red and has_body and body_below_vwap and closes_below_vwap

    return False


class MultiPercentileBacktester:
    """Backtest multi-scale percentile policy fusion."""

    def __init__(
        self,
        executor,  # Can be MultiPercentileExecutor or FilteredExecutor
        config: BacktestConfig,
        device: torch.device,
        use_heuristic: bool = True,
        heuristic_config: Optional[HeuristicConfig] = None,
    ):
        self.executor = executor
        self.config = config
        self.device = device
        self.options = RealOptionsProvider()
        self.use_heuristic = use_heuristic
        self.heuristic_config = heuristic_config or HeuristicConfig()
        self.heuristic_model = HeuristicModel(
            lookback_minutes=30,
            config=self.heuristic_config,
        ) if use_heuristic else None
        self.et_tz = ZoneInfo("America/New_York")

        model_info = executor.get_model_info()
        self.max_context_len = max(info["context_len"] for info in model_info.values()) if model_info else 60
        self.context_lens = {name: info["context_len"] for name, info in model_info.items()}

        self.position: Optional[Position] = None
        self.trades: List[Trade] = []
        self.capital = config.initial_capital
        self.peak_capital = config.initial_capital
        self.lowest_capital = config.initial_capital  # Track lowest for drawdown
        self.model_contributions: Dict[str, int] = {"15m": 0, "5m": 0, "1m": 0}

        # Profit protection tracking (per day)
        self.daily_cost_basis = 0.0  # Total invested today
        self.daily_realized_profit = 0.0  # Total profit realized today
        self.daily_protected_capital = 0.0  # Capital "set aside" after 100% profit
        self.profit_protection_active = False  # Trading with profits only (activates once)
        self.max_daily_cost_basis = 0.0  # Max we can invest today (= starting balance)

        # Track heuristic rejections for stats
        self.heuristic_rejections = 0
        self.timeframe_rejections = 0
        self.vwap_direction_rejections = 0

        # Pending entry (to be executed on next minute's open)
        self.pending_entry: Optional[PendingEntry] = None

        # Track potential VWAP breach from previous candle (for continuation confirmation)
        self.pending_vwap_breach: Optional[VWAPBreachSignal] = None

        # VWAP-specific metrics tracking
        self.vwap_immediate_entries = 0  # Strong breach entries
        self.vwap_continuation_entries = 0  # Confirmed breach entries

        # ORB-specific tracking
        self.pending_orb_breach: Optional[ORBBreachSignal] = None
        self.orb_entries = 0  # ORB continuation entries
        self.orb_immediate_entries = 0  # ORB immediate entries (strong breakout candles)
        self.current_or_high: Optional[float] = None  # Current day's OR high
        self.current_or_low: Optional[float] = None   # Current day's OR low

        # Daily/Weekly drawdown tracking
        self.max_daily_drawdown_pct = 0.0  # Worst single-day drawdown
        self.max_weekly_drawdown_pct = 0.0  # Worst single-week drawdown
        self.daily_peak_capital = config.initial_capital  # Highest capital within the day
        self.daily_low_capital = config.initial_capital  # Lowest capital within the day
        self.weekly_start_capital = config.initial_capital  # Capital at start of week
        self.weekly_peak_capital = config.initial_capital  # Highest capital within the week
        self.weekly_low_capital = config.initial_capital  # Lowest capital within the week
        self.current_week: Optional[tuple] = None  # (year, week_number)

        # Daily/Weekly gain rate tracking
        self.daily_results: List[float] = []  # List of daily P&L percentages
        self.weekly_results: List[float] = []  # List of weekly P&L percentages
        self.prev_day_capital: float = config.initial_capital  # Capital at start of current day
        self.prev_week_capital: float = config.initial_capital  # Capital at start of current week
        self.last_trade_date: Optional[date] = None  # Date of last trade exit (for day change detection)
        self.last_trade_week: Optional[tuple] = None  # Week of last trade exit (for week change detection)

        # Best/Worst Daily/Weekly P&L tracking (intraday/intraweek peak and trough returns)
        self.best_daily_pnl_pct: float = 0.0  # Best intraday peak return (all-time)
        self.worst_daily_pnl_pct: float = 0.0  # Worst intraday trough return (all-time)
        self.best_weekly_pnl_pct: float = 0.0  # Best intraweek peak return (all-time)
        self.worst_weekly_pnl_pct: float = 0.0  # Worst intraweek trough return (all-time)
        self.day_start_capital: float = config.initial_capital  # Capital at day open
        self.week_start_capital: float = config.initial_capital  # Capital at week open

    def reset(self, starting_capital: Optional[float] = None):
        """Reset state for new backtest."""
        self.position = None
        self.trades = []
        capital = starting_capital if starting_capital is not None else self.config.initial_capital
        self.capital = capital
        self.peak_capital = capital
        self.lowest_capital = capital  # Track lowest for drawdown
        self.daily_trades = 0
        self.daily_allocation = 0.0
        self.daily_pnl = 0.0
        self.daily_start_capital = capital
        self.current_day = None
        self.model_contributions = {"15m": 0, "5m": 0, "1m": 0}

        # Profit protection reset
        self.daily_cost_basis = 0.0
        self.daily_realized_profit = 0.0
        self.daily_protected_capital = 0.0
        self.profit_protection_active = False
        self.max_daily_cost_basis = capital  # Can only invest up to starting balance

        # Heuristic tracking
        self.heuristic_rejections = 0
        self.timeframe_rejections = 0
        self.vwap_direction_rejections = 0

        # Pending entry
        self.pending_entry = None

        # Pending VWAP breach (for continuation confirmation)
        self.pending_vwap_breach = None

        # VWAP-specific metrics tracking
        self.vwap_immediate_entries = 0
        self.vwap_continuation_entries = 0

        # ORB-specific tracking
        self.pending_orb_breach = None
        self.orb_entries = 0
        self.orb_immediate_entries = 0
        self.current_or_high = None
        self.current_or_low = None

        # Daily/Weekly drawdown tracking
        self.max_daily_drawdown_pct = 0.0
        self.max_weekly_drawdown_pct = 0.0
        self.daily_peak_capital = capital
        self.daily_low_capital = capital
        self.weekly_start_capital = capital
        self.weekly_peak_capital = capital
        self.weekly_low_capital = capital
        self.current_week = None

        # Daily/Weekly gain rate tracking
        self.daily_results = []
        self.weekly_results = []
        self.prev_day_capital = capital
        self.prev_week_capital = capital
        self.last_trade_date = None
        self.last_trade_week = None

        # Best/Worst Daily/Weekly P&L tracking
        self.best_daily_pnl_pct = 0.0
        self.worst_daily_pnl_pct = 0.0
        self.best_weekly_pnl_pct = 0.0
        self.worst_weekly_pnl_pct = 0.0
        self.day_start_capital = capital
        self.week_start_capital = capital

    def _get_minutes_elapsed(self, timestamp: datetime) -> int:
        et_time = timestamp.tz_convert("America/New_York")
        minutes_since_midnight = et_time.hour * 60 + et_time.minute
        market_open = 9 * 60 + 30
        return minutes_since_midnight - market_open

    def _update_drawdown_tracking(self, trade_time: datetime):
        """Update daily and weekly drawdown tracking after capital changes."""
        trade_date = trade_time.date()
        trade_week = trade_date.isocalendar()[:2]  # (year, week_number)

        # Check for new day - finalize previous day's drawdown and P&L
        # Use last_trade_date (not current_day) to properly detect day changes
        if self.last_trade_date is not None and self.last_trade_date != trade_date:
            # Calculate previous day's drawdown (peak to trough within day)
            if self.daily_peak_capital > 0:
                daily_dd = (self.daily_peak_capital - self.daily_low_capital) / self.daily_peak_capital * 100
                self.max_daily_drawdown_pct = max(self.max_daily_drawdown_pct, daily_dd)
            # Record daily P&L percentage (end of previous day vs start of previous day)
            if self.prev_day_capital > 0:
                daily_pnl_pct = (self.capital - self.prev_day_capital) / self.prev_day_capital * 100
                self.daily_results.append(daily_pnl_pct)
            self.prev_day_capital = self.capital  # New day starts with current capital
            # Reset for new day
            self.daily_peak_capital = self.capital
            self.daily_low_capital = self.capital
            self.day_start_capital = self.capital  # Reset day start capital

        # Check for new week - finalize previous week's drawdown and P&L
        if self.last_trade_week is not None and self.last_trade_week != trade_week:
            # Calculate previous week's drawdown (peak to trough within week)
            if self.weekly_peak_capital > 0:
                weekly_dd = (self.weekly_peak_capital - self.weekly_low_capital) / self.weekly_peak_capital * 100
                self.max_weekly_drawdown_pct = max(self.max_weekly_drawdown_pct, weekly_dd)
            # Record weekly P&L percentage
            if self.prev_week_capital > 0:
                weekly_pnl_pct = (self.capital - self.prev_week_capital) / self.prev_week_capital * 100
                self.weekly_results.append(weekly_pnl_pct)
            self.prev_week_capital = self.capital
            # Reset for new week
            self.weekly_start_capital = self.capital
            self.weekly_peak_capital = self.capital
            self.weekly_low_capital = self.capital
            self.week_start_capital = self.capital  # Reset week start capital

        # Update tracking variables
        self.last_trade_date = trade_date
        self.last_trade_week = trade_week

        # Update peak and low for current day/week
        self.daily_peak_capital = max(self.daily_peak_capital, self.capital)
        self.daily_low_capital = min(self.daily_low_capital, self.capital)
        self.weekly_peak_capital = max(self.weekly_peak_capital, self.capital)
        self.weekly_low_capital = min(self.weekly_low_capital, self.capital)

        # Track best/worst intraday P&L (relative to day start)
        if self.day_start_capital > 0:
            intraday_peak_pct = (self.daily_peak_capital - self.day_start_capital) / self.day_start_capital * 100
            intraday_trough_pct = (self.daily_low_capital - self.day_start_capital) / self.day_start_capital * 100
            self.best_daily_pnl_pct = max(self.best_daily_pnl_pct, intraday_peak_pct)
            self.worst_daily_pnl_pct = min(self.worst_daily_pnl_pct, intraday_trough_pct)

        # Track best/worst intraweek P&L (relative to week start)
        if self.week_start_capital > 0:
            intraweek_peak_pct = (self.weekly_peak_capital - self.week_start_capital) / self.week_start_capital * 100
            intraweek_trough_pct = (self.weekly_low_capital - self.week_start_capital) / self.week_start_capital * 100
            self.best_weekly_pnl_pct = max(self.best_weekly_pnl_pct, intraweek_peak_pct)
            self.worst_weekly_pnl_pct = min(self.worst_weekly_pnl_pct, intraweek_trough_pct)

    def _finalize_drawdown_tracking(self):
        """Finalize drawdown and P&L tracking for the last day/week at end of backtest."""
        # Finalize last day's drawdown and P&L
        if self.daily_peak_capital > 0:
            daily_dd = (self.daily_peak_capital - self.daily_low_capital) / self.daily_peak_capital * 100
            self.max_daily_drawdown_pct = max(self.max_daily_drawdown_pct, daily_dd)
        # Record last day's P&L
        if self.prev_day_capital > 0 and self.capital != self.prev_day_capital:
            daily_pnl_pct = (self.capital - self.prev_day_capital) / self.prev_day_capital * 100
            self.daily_results.append(daily_pnl_pct)

        # Finalize last week's drawdown and P&L
        if self.weekly_peak_capital > 0:
            weekly_dd = (self.weekly_peak_capital - self.weekly_low_capital) / self.weekly_peak_capital * 100
            self.max_weekly_drawdown_pct = max(self.max_weekly_drawdown_pct, weekly_dd)
        # Record last week's P&L
        if self.prev_week_capital > 0 and self.capital != self.prev_week_capital:
            weekly_pnl_pct = (self.capital - self.prev_week_capital) / self.prev_week_capital * 100
            self.weekly_results.append(weekly_pnl_pct)

        # Finalize best/worst intraday P&L for last day
        if self.day_start_capital > 0:
            intraday_peak_pct = (self.daily_peak_capital - self.day_start_capital) / self.day_start_capital * 100
            intraday_trough_pct = (self.daily_low_capital - self.day_start_capital) / self.day_start_capital * 100
            self.best_daily_pnl_pct = max(self.best_daily_pnl_pct, intraday_peak_pct)
            self.worst_daily_pnl_pct = min(self.worst_daily_pnl_pct, intraday_trough_pct)

        # Finalize best/worst intraweek P&L for last week
        if self.week_start_capital > 0:
            intraweek_peak_pct = (self.weekly_peak_capital - self.week_start_capital) / self.week_start_capital * 100
            intraweek_trough_pct = (self.weekly_low_capital - self.week_start_capital) / self.week_start_capital * 100
            self.best_weekly_pnl_pct = max(self.best_weekly_pnl_pct, intraweek_peak_pct)
            self.worst_weekly_pnl_pct = min(self.worst_weekly_pnl_pct, intraweek_trough_pct)

    def _build_contexts(
        self,
        feature_tensor: torch.Tensor,
        idx: int,
    ) -> Dict[str, torch.Tensor]:
        contexts = {}
        for name, context_len in self.context_lens.items():
            if idx >= context_len:
                context = feature_tensor[idx - context_len:idx].unsqueeze(0)
                contexts[name] = context
        return contexts

    def check_exit_or_renew(
        self,
        current_price: float,
        current_time: datetime,
        fused_signal: FusedSignal,
        upper_2sd: Optional[float] = None,  # +2σ VWAP band for take profit
        lower_2sd: Optional[float] = None,  # -2σ VWAP band for take profit
        upper_1sd: Optional[float] = None,  # +1σ VWAP band for partial take profit
        lower_1sd: Optional[float] = None,  # -1σ VWAP band for partial take profit
        candle_open: Optional[float] = None,  # Current candle open for reversal detection
        candle_close: Optional[float] = None,  # Current candle close for reversal detection
        avg_body: Optional[float] = None,  # Average candle body for "significant" threshold
    ) -> Tuple[Optional[str], bool, int]:
        if self.position is None:
            return None, False, 0

        current_option_price = self.options.get_option_price(
            timestamp=current_time,
            strike=self.position.strike,
            position_type=self.position.position_type,
            underlying_price=current_price,
        )

        if current_option_price is None:
            return None, False, 0

        pnl_pct = (current_option_price - self.position.entry_option_price) / self.position.entry_option_price * 100

        # Track peak P&L
        if not self.position.is_runner:
            if pnl_pct > self.position.peak_pnl_pct:
                self.position.peak_pnl_pct = pnl_pct
        else:
            # Track runner peak separately
            if pnl_pct > self.position.runner_peak_pnl_pct:
                self.position.runner_peak_pnl_pct = pnl_pct

        # === RUNNER EXIT LOGIC ===
        # NOTE: Runner P&L is calculated from ORIGINAL entry price (not split price)
        # runner_entry_pnl_pct = P&L% when runner was created (e.g., +20%)
        # Profit = gained more than split point, Loss = gave back gains since split
        if self.position.is_runner:
            split_pnl = self.position.runner_entry_pnl_pct  # P&L when we split (e.g., +20%)
            gain_since_split = pnl_pct - split_pnl  # How much gained/lost since split

            # RUNNER EXITS:
            # 1. Close if runner falls below the original position's profitability
            if pnl_pct < split_pnl:
                return "runner_below_split", False, 0

            # Runner time barrier (aggressive - 15 mins default)
            runner_time = (current_time - self.position.runner_start_time).total_seconds() / 60
            if runner_time >= self.position.runner_max_hold_minutes:
                # Time barrier exit - compare to split point
                if gain_since_split >= 5.0:
                    return "runner_profit_time", False, 0
                elif gain_since_split >= 0.0:
                    return "runner_breakeven_time", False, 0
                else:
                    return "runner_loss_time", False, 0

            # Runner trailing stop - trail by 10% from runner peak
            # Minimum stop is at split P&L (protect the locked-in gains)
            runner_stop = self.position.runner_peak_pnl_pct - 10.0
            runner_stop = max(runner_stop, split_pnl)  # Never give back more than split gains
            peak_gain_since_split = self.position.runner_peak_pnl_pct - split_pnl
            if pnl_pct <= runner_stop and peak_gain_since_split >= 5.0:
                # Only trigger trailing if we've had some gains since split to protect
                if gain_since_split >= 5.0:
                    return "runner_profit_trailing", False, 0
                elif gain_since_split >= 0.0:
                    return "runner_breakeven_trailing", False, 0
                else:
                    return "runner_loss_trailing", False, 0

            # Fixed stop loss for runner - if we give back 10% from split point
            if gain_since_split <= -10.0:
                return "runner_stop_loss", False, 0

            # Signal reversal closes runner
            if self.position.position_type == PositionType.CALL and fused_signal.action == Signal.SHORT:
                if gain_since_split >= 5.0:
                    return "runner_profit_reversal", False, 0
                elif gain_since_split >= 0.0:
                    return "runner_breakeven_reversal", False, 0
                else:
                    return "runner_loss_reversal", False, 0
            if self.position.position_type == PositionType.PUT and fused_signal.action == Signal.LONG:
                if gain_since_split >= 5.0:
                    return "runner_profit_reversal", False, 0
                elif gain_since_split >= 0.0:
                    return "runner_breakeven_reversal", False, 0
                else:
                    return "runner_loss_reversal", False, 0

            # Runner doesn't renew, just hold
            return None, False, 0

        # === NORMAL POSITION EXIT LOGIC ===

        # 1. Fixed stop loss
        if not self.position.breakeven_activated:
            if pnl_pct <= -self.config.stop_loss_pct:
                return "stop_loss", False, 0
            # VWAP/ORB trades: breakeven at 10%, others at 7.5%
            breakeven_threshold = 10.0 if self.position.is_vwap_breach else 7.5
            if pnl_pct >= breakeven_threshold:
                self.position.breakeven_activated = True

        # 2. Breakeven stop (trailing from peak)
        # Once breakeven is activated, we protect profits with trailing stop
        # VWAP trades: hard floor at +2% (lock in profit when hitting 10%)
        # Other trades: floor at +2.5%
        if self.position.breakeven_activated:
            # VWAP trades have a hard breakeven floor at +2%
            if self.position.is_vwap_breach and pnl_pct <= 2.0:
                return "vwap_breakeven_stop", False, 0

            trail_distance = 5.0  # Same for all trades - aggressive trailing
            trailing_stop = self.position.peak_pnl_pct - trail_distance
            if pnl_pct <= trailing_stop:
                # Check if we should convert to runner (+20% or more)
                if pnl_pct >= 20.0:
                    return "convert_to_runner", False, 0
                elif pnl_pct <= 2.5:
                    # Breakeven stop with auto-stop - guaranteed to close at 0% minimum
                    return "breakeven_stop", False, 0
                else:
                    return "trailing_stop", False, 0

        # 3. Signal reversal
        if self.position.position_type == PositionType.CALL and fused_signal.action == Signal.SHORT:
            # Check if profitable enough for runner
            if pnl_pct >= 20.0:
                return "convert_to_runner", False, 0
            return "signal_reversal", False, 0
        if self.position.position_type == PositionType.PUT and fused_signal.action == Signal.LONG:
            if pnl_pct >= 20.0:
                return "convert_to_runner", False, 0
            return "signal_reversal", False, 0

        # 4. Signal confirmation - reset barriers
        if self.position.position_type == PositionType.CALL and fused_signal.action == Signal.LONG:
            return None, True, fused_signal.exit_horizon_minutes
        if self.position.position_type == PositionType.PUT and fused_signal.action == Signal.SHORT:
            return None, True, fused_signal.exit_horizon_minutes

        # 5. Time barrier - check if profitable enough for runner
        time_since_barrier_start = (current_time - self.position.barrier_start_time).total_seconds() / 60
        if time_since_barrier_start >= self.position.max_hold_minutes:
            if pnl_pct >= 20.0:
                return "convert_to_runner", False, 0
            return "time_barrier", False, 0

        return None, False, 0

    def run_backtest(
        self,
        df: pd.DataFrame,
        raw_closes: np.ndarray,
        year: int,
        month: Optional[int] = None,
        starting_capital: Optional[float] = None,
        ohlcv_df: Optional[pd.DataFrame] = None,
    ) -> Dict:
        """Run backtest for a specific year and optionally month."""
        self.reset(starting_capital)

        if month is not None:
            mask = (df.index.year == year) & (df.index.month == month)
        else:
            mask = df.index.year == year

        period_df = df[mask].copy()
        period_closes = raw_closes[mask]
        period_ohlcv = ohlcv_df[mask].copy() if ohlcv_df is not None else None

        if len(period_df) < self.max_context_len + 1:
            return {"year": year, "month": month, "trades": 0, "error": "Insufficient data"}

        features = period_df.values
        feature_tensor = torch.tensor(features, dtype=torch.float32, device=self.device)

        # Calculate VWAP with bands for the period (for VWAP breach detection)
        vwap_series = None
        upper_1sd_series = None
        lower_1sd_series = None
        upper_2sd_series = None
        lower_2sd_series = None
        if period_ohlcv is not None:
            vwap_series, upper_1sd_series, lower_1sd_series, upper_2sd_series, lower_2sd_series = calculate_vwap_with_bands(period_ohlcv)

        current_day = None
        desc = f"Year {year}" if month is None else f"{year}-{month:02d}"

        for i in tqdm(range(self.max_context_len, len(period_df)), desc=desc, leave=False):
            current_time = period_df.index[i]
            current_price = period_closes[i]

            et_time = current_time.tz_convert("America/New_York")
            et_hour = et_time.hour
            et_minute = et_time.minute

            if et_hour < 9 or (et_hour == 9 and et_minute < 30):
                continue

            # EOD handling: close positions at 4pm, but 15-min rule only applies to OPENING
            if et_hour >= 16:
                self.pending_entry = None  # Cancel any pending entries
                if self.position is not None:
                    self._close_position(current_time, current_price, "eod_close")
                continue

            # Check if we're in the first 10 minutes (no new positions - too volatile)
            is_first_10_minutes = (et_hour == 9 and et_minute < 40)

            # Check if we're in the last 15 minutes (no new positions, but can hold existing)
            is_last_15_minutes = (et_hour == 15 and et_minute >= 45)

            trading_date = et_time.date()
            if current_day != trading_date:
                current_day = trading_date
                self.options.load_date(trading_date)
                # Reset ORB tracking for new day
                self.current_or_high = None
                self.current_or_low = None
                self.pending_orb_breach = None

            minutes_elapsed = self._get_minutes_elapsed(current_time)

            # Calculate Opening Range after first 15 minutes (9:45 AM)
            if self.current_or_high is None and minutes_elapsed >= 15 and period_ohlcv is not None:
                # Find today's data start in period_ohlcv
                day_start_idx = None
                for j in range(max(0, i - 20), i + 1):
                    idx_time = period_ohlcv.index[j]
                    if hasattr(idx_time, 'tz_convert'):
                        idx_et = idx_time.tz_convert("America/New_York")
                    else:
                        idx_et = pd.Timestamp(idx_time).tz_convert("America/New_York")
                    if idx_et.date() == trading_date and idx_et.hour == 9 and idx_et.minute == 30:
                        day_start_idx = j
                        break
                if day_start_idx is not None and i >= day_start_idx + 14:
                    or_data = period_ohlcv.iloc[day_start_idx:day_start_idx + 15]
                    self.current_or_high = or_data['high'].max()
                    self.current_or_low = or_data['low'].min()

            contexts = self._build_contexts(feature_tensor, i)
            fused_signal = self.executor.get_signal(contexts, minutes_elapsed)

            # Execute pending entry at OPEN of this minute (signal was from previous minute)
            if self.pending_entry is not None and self.position is None:
                pe = self.pending_entry
                self.pending_entry = None

                # VWAP direction filter at execution time: no longs below VWAP, no shorts above VWAP
                current_vwap = vwap_series.iloc[i] if vwap_series is not None else None
                vwap_direction_ok = True
                if current_vwap is not None:
                    if pe.position_type == PositionType.CALL and current_price < current_vwap:
                        vwap_direction_ok = False  # Block long below VWAP
                    elif pe.position_type == PositionType.PUT and current_price > current_vwap:
                        vwap_direction_ok = False  # Block short above VWAP

                if vwap_direction_ok:
                    # Execute at this minute's open price
                    self._open_position(
                        pe.position_type,
                        current_time,  # Execute at current time (open of this minute)
                        current_price,
                        i,
                        pe.dominant_model,
                        pe.max_hold_minutes,
                        pe.confidence,
                        pe.confluence_count,
                        pe.agreeing_models,
                        pe.is_early_trade,
                        pe.is_vwap_breach,
                        pe.position_size_bonus,
                    )

            if self.position is not None:
                # Get current band values for VWAP take profit checks
                curr_upper_2sd = upper_2sd_series.iloc[i] if upper_2sd_series is not None else None
                curr_lower_2sd = lower_2sd_series.iloc[i] if lower_2sd_series is not None else None
                curr_upper_1sd = upper_1sd_series.iloc[i] if upper_1sd_series is not None else None
                curr_lower_1sd = lower_1sd_series.iloc[i] if lower_1sd_series is not None else None

                # Get candle data for opposite direction exit check
                candle_open = None
                candle_close = None
                avg_body = None
                if period_ohlcv is not None:
                    candle_open = period_ohlcv.iloc[i]['open']
                    candle_close = period_ohlcv.iloc[i]['close']
                    # Calculate average body over last 20 candles
                    lookback = min(20, i)
                    if lookback > 0:
                        recent = period_ohlcv.iloc[i - lookback:i]
                        avg_body = (recent['close'] - recent['open']).abs().mean()

                exit_reason, should_renew, new_max_hold = self.check_exit_or_renew(
                    current_price, current_time, fused_signal,
                    upper_2sd=curr_upper_2sd, lower_2sd=curr_lower_2sd,
                    upper_1sd=curr_upper_1sd, lower_1sd=curr_lower_1sd,
                    candle_open=candle_open, candle_close=candle_close, avg_body=avg_body
                )
                if exit_reason:
                    if exit_reason == "convert_to_runner":
                        # Convert position to runner instead of closing
                        self._convert_to_runner(current_time, current_price)
                    else:
                        # Calculate stop loss trigger level for realistic fills
                        stop_trigger_pct = None
                        if "stop_loss" in exit_reason:
                            stop_trigger_pct = -self.config.stop_loss_pct
                        elif "trailing" in exit_reason or "breakeven" in exit_reason:
                            if self.position.is_runner:
                                # Runner trailing: 10% from runner peak
                                trail_dist = 10.0
                                stop_trigger_pct = self.position.runner_peak_pnl_pct - trail_dist
                            else:
                                trail_dist = 5.0  # Same for all trades - aggressive trailing
                                stop_trigger_pct = self.position.peak_pnl_pct - trail_dist
                        self._close_position(current_time, current_price, exit_reason, stop_trigger_pct)
                elif should_renew:
                    self.position.barrier_start_time = current_time
                    self.position.max_hold_minutes = new_max_hold
                    self.position.renewals += 1

            # Create pending entry (to be executed on NEXT minute's open)
            # Check for early window (9:45-10:00) - requires 1m+5m+heuristic agreement (no 15m)
            trading_window = self.executor.fusion_config.trading_window
            is_early_window = trading_window.is_early_window(minutes_elapsed)

            if self.position is None and self.pending_entry is None and not is_first_10_minutes and not is_last_15_minutes:
                final_action = Signal.HOLD
                is_early_trade = False
                is_vwap_breach = False
                agreeing_models = ()
                dominant_model = ""
                exit_horizon = 10  # Default for early trades
                confidence = 0.5

                # PRIORITY 1: Check for VWAP breach (always takes priority)
                if vwap_series is not None and period_ohlcv is not None:
                    vwap_breach = detect_vwap_breach(period_ohlcv, vwap_series, i,
                                                       upper_1sd_series=upper_1sd_series,
                                                       lower_1sd_series=lower_1sd_series,
                                                       upper_2sd_series=upper_2sd_series,
                                                       lower_2sd_series=lower_2sd_series)

                    # First, check if we have a pending potential breach that gets confirmed
                    if self.pending_vwap_breach is not None:
                        if check_continuation_confirmation(period_ohlcv, i, self.pending_vwap_breach):
                            # Continuation confirms the potential breach
                            if self.pending_vwap_breach.breach_type == VWAPBreachType.BULLISH_POTENTIAL:
                                final_action = Signal.LONG
                                is_vwap_breach = True
                                agreeing_models = ("vwap_breach_continuation",)
                                dominant_model = "vwap"
                                exit_horizon = 15
                                confidence = 0.7  # Continuation confirmation
                                self.vwap_continuation_entries += 1
                            elif self.pending_vwap_breach.breach_type == VWAPBreachType.BEARISH_POTENTIAL:
                                final_action = Signal.SHORT
                                is_vwap_breach = True
                                agreeing_models = ("vwap_breach_continuation",)
                                dominant_model = "vwap"
                                exit_horizon = 15
                                confidence = 0.7
                                self.vwap_continuation_entries += 1
                        # Clear pending breach after checking (whether confirmed or not)
                        self.pending_vwap_breach = None

                    # If no continuation entry, check for new VWAP breach
                    if not is_vwap_breach:
                        # Check for CONFIRMED breaches (strong body + volume) - enter immediately
                        if vwap_breach.breach_type == VWAPBreachType.BULLISH:
                            final_action = Signal.LONG
                            is_vwap_breach = True
                            agreeing_models = ("vwap_breach_immediate",)
                            dominant_model = "vwap"
                            exit_horizon = 15
                            confidence = 0.8  # High confidence for strong breach
                            self.vwap_immediate_entries += 1
                        elif vwap_breach.breach_type == VWAPBreachType.BEARISH:
                            final_action = Signal.SHORT
                            is_vwap_breach = True
                            agreeing_models = ("vwap_breach_immediate",)
                            dominant_model = "vwap"
                            exit_horizon = 15
                            confidence = 0.8  # High confidence for strong breach
                            self.vwap_immediate_entries += 1
                        # Check for POTENTIAL breaches - wait for confirmation candle
                        elif vwap_breach.breach_type in (
                            VWAPBreachType.BULLISH_POTENTIAL,
                            VWAPBreachType.BEARISH_POTENTIAL,
                        ):
                            # Track potential breach for next candle's continuation check
                            self.pending_vwap_breach = vwap_breach

                # PRIORITY 1.5: Check for ORB breach (after opening range is established)
                is_orb_breach = False
                if not is_vwap_breach and self.current_or_high is not None and period_ohlcv is not None:
                    # First, check if we have a pending ORB breach that gets confirmed
                    if self.pending_orb_breach is not None:
                        if check_orb_continuation_confirmation(period_ohlcv, i, self.pending_orb_breach):
                            # Continuation confirms the ORB breach
                            if self.pending_orb_breach.breach_type == ORBBreachType.BULLISH_POTENTIAL:
                                final_action = Signal.LONG
                                is_orb_breach = True
                                agreeing_models = ("orb_breakout",)
                                dominant_model = "orb"
                                exit_horizon = 15
                                confidence = 0.7
                                self.orb_entries += 1
                            elif self.pending_orb_breach.breach_type == ORBBreachType.BEARISH_POTENTIAL:
                                final_action = Signal.SHORT
                                is_orb_breach = True
                                agreeing_models = ("orb_breakout",)
                                dominant_model = "orb"
                                exit_horizon = 15
                                confidence = 0.7
                                self.orb_entries += 1
                        # Clear pending ORB breach after checking (whether confirmed or not)
                        self.pending_orb_breach = None

                    # If no ORB entry yet, check for new ORB breach
                    if not is_orb_breach:
                        orb_breach = detect_orb_breach(
                            period_ohlcv, i,
                            self.current_or_high, self.current_or_low
                        )
                        # Check for CONFIRMED breaches (strong breakout candles) - enter immediately
                        if orb_breach.breach_type == ORBBreachType.BULLISH:
                            final_action = Signal.LONG
                            is_orb_breach = True
                            agreeing_models = ("orb_breakout_immediate",)
                            dominant_model = "orb"
                            exit_horizon = 15
                            confidence = 0.8  # High confidence for strong breach
                            self.orb_immediate_entries += 1
                        elif orb_breach.breach_type == ORBBreachType.BEARISH:
                            final_action = Signal.SHORT
                            is_orb_breach = True
                            agreeing_models = ("orb_breakout_immediate",)
                            dominant_model = "orb"
                            exit_horizon = 15
                            confidence = 0.8  # High confidence for strong breach
                            self.orb_immediate_entries += 1
                        # Check for POTENTIAL breaches - wait for continuation candle
                        elif orb_breach.breach_type in (
                            ORBBreachType.BULLISH_POTENTIAL,
                            ORBBreachType.BEARISH_POTENTIAL,
                        ):
                            self.pending_orb_breach = orb_breach

                # PRIORITY 2: Early window (9:45-10:00) - only if no VWAP or ORB breach
                if not is_vwap_breach and not is_orb_breach and is_early_window:
                    # Early window (9:45-10:00): Require 1m+5m confluence + heuristic
                    # Get 1m and 5m signals only (ignore 15m)
                    early_agreeing = []
                    early_action = Signal.HOLD
                    early_confidence = 0.0

                    for model_name in ["1m", "5m"]:
                        if model_name in fused_signal.individual_signals:
                            sig, prob = fused_signal.individual_signals[model_name]
                            if sig != Signal.HOLD:
                                if early_action == Signal.HOLD:
                                    early_action = sig
                                    early_agreeing.append(model_name)
                                    early_confidence = prob
                                elif sig == early_action:
                                    early_agreeing.append(model_name)
                                    early_confidence = max(early_confidence, prob)
                                else:
                                    # 1m and 5m disagree
                                    early_action = Signal.HOLD
                                    early_agreeing = []
                                    break

                    # Need both 1m and 5m to agree
                    if len(early_agreeing) == 2 and early_action != Signal.HOLD:
                        # Apply heuristic filter for early trades
                        if self.use_heuristic and self.heuristic_model is not None and period_ohlcv is not None:
                            lookback = min(30, i)
                            ohlcv_window = period_ohlcv.iloc[i - lookback:i + 1]
                            heuristic_action = self.heuristic_model.get_signal(ohlcv_window, early_action)
                            if heuristic_action == early_action:
                                # Early trade: 1m+5m+heuristic all agree
                                final_action = early_action
                                is_early_trade = True
                                agreeing_models = tuple(early_agreeing) + ("heuristic",)
                                dominant_model = "5m"  # 5m is dominant for early trades
                                exit_horizon = 10  # 10-minute exit horizon for early trades
                                confidence = early_confidence

                # PRIORITY 3: Normal trading window (only if no VWAP/ORB breach and no early trade)
                elif not is_vwap_breach and not is_orb_breach and self.executor.is_trading_allowed(minutes_elapsed):
                    # Normal trading window: Use standard fusion + heuristic
                    final_action = fused_signal.action
                    if self.use_heuristic and self.heuristic_model is not None and period_ohlcv is not None:
                        if fused_signal.action != Signal.HOLD:
                            lookback = min(30, i)
                            ohlcv_window = period_ohlcv.iloc[i - lookback:i + 1]
                            final_action = self.heuristic_model.get_signal(ohlcv_window, fused_signal.action)
                            if final_action == Signal.HOLD and fused_signal.action != Signal.HOLD:
                                self.heuristic_rejections += 1

                    agreeing_models = fused_signal.agreeing_models
                    dominant_model = fused_signal.dominant_model
                    exit_horizon = fused_signal.exit_horizon_minutes
                    confidence = fused_signal.confidence

                # Get daily VWAP for timeframe support and position sizing
                daily_vwap = vwap_series.iloc[i] if vwap_series is not None else None
                position_size_bonus = 0.0

                # Check VWAP direction, timeframe support, and calculate position size bonus
                # (Skip for VWAP breach trades which have their own logic)
                if final_action != Signal.HOLD and not is_vwap_breach:
                    if self.use_heuristic and self.heuristic_model is not None and period_ohlcv is not None:
                        lookback = min(30, i)
                        ohlcv_window = period_ohlcv.iloc[i - lookback:i + 1]

                        # Check VWAP direction (LONG only above VWAP, SHORT only below VWAP)
                        # Note: VWAP breach trades skip this check (handled by `not is_vwap_breach` above)
                        vwap_direction_ok, _ = self.heuristic_model.check_vwap_direction(
                            final_action, current_price, daily_vwap
                        )
                        if not vwap_direction_ok:
                            self.vwap_direction_rejections += 1
                            final_action = Signal.HOLD

                        # Check timeframe support (5m or 15m VWAP)
                        if final_action != Signal.HOLD:
                            is_supported, _ = self.heuristic_model.check_timeframe_support(
                                ohlcv_window, final_action, current_price, daily_vwap
                            )
                            if not is_supported:
                                self.timeframe_rejections += 1
                                final_action = Signal.HOLD
                            else:
                                # Calculate position size bonus
                                position_size_bonus, _ = self.heuristic_model.get_position_size_bonus(
                                    ohlcv_window, final_action, current_price, daily_vwap
                                )

                confluence_count = len(agreeing_models)
                # Combine VWAP and ORB breach flags for position sizing and strike selection
                is_breakout_trade = is_vwap_breach or is_orb_breach

                # VWAP direction filter: no longs below VWAP, no shorts above VWAP
                if daily_vwap is not None:
                    if final_action == Signal.LONG and current_price < daily_vwap:
                        final_action = Signal.HOLD  # Block long below VWAP
                    elif final_action == Signal.SHORT and current_price > daily_vwap:
                        final_action = Signal.HOLD  # Block short above VWAP

                if final_action == Signal.LONG:
                    # Queue entry for next minute's open
                    self.pending_entry = PendingEntry(
                        position_type=PositionType.CALL,
                        signal_time=current_time,
                        underlying_price=current_price,
                        signal_idx=i,
                        dominant_model=dominant_model,
                        max_hold_minutes=exit_horizon,
                        confidence=confidence,
                        confluence_count=confluence_count,
                        agreeing_models=agreeing_models,
                        is_early_trade=is_early_trade,
                        is_vwap_breach=is_breakout_trade,
                        position_size_bonus=position_size_bonus,
                    )
                elif final_action == Signal.SHORT:
                    # Queue entry for next minute's open
                    self.pending_entry = PendingEntry(
                        position_type=PositionType.PUT,
                        signal_time=current_time,
                        underlying_price=current_price,
                        signal_idx=i,
                        dominant_model=dominant_model,
                        max_hold_minutes=exit_horizon,
                        confidence=confidence,
                        confluence_count=confluence_count,
                        agreeing_models=agreeing_models,
                        is_early_trade=is_early_trade,
                        is_vwap_breach=is_breakout_trade,
                        position_size_bonus=position_size_bonus,
                    )

        if self.position is not None:
            self._close_position(
                period_df.index[-1],
                period_closes[-1],
                "backtest_end"
            )

        return self._compute_metrics(year, month)

    def run_backtest_with_signals(
        self,
        df: pd.DataFrame,
        raw_closes: np.ndarray,
        year: int,
        month: Optional[int],
        full_executor,  # MultiPercentileExecutor
        starting_capital: Optional[float] = None,
        ohlcv_df: Optional[pd.DataFrame] = None,
    ) -> Tuple[Dict, Dict[str, List]]:
        """Run backtest and track individual model signals for HTML report."""
        from src.backtest.html_report import SignalMarker

        self.reset(starting_capital)

        if month is not None:
            mask = (df.index.year == year) & (df.index.month == month)
        else:
            mask = df.index.year == year

        period_df = df[mask].copy()
        period_closes = raw_closes[mask]
        period_ohlcv = ohlcv_df[mask].copy() if ohlcv_df is not None else None

        if len(period_df) < self.max_context_len + 1:
            return {"year": year, "month": month, "trades": 0, "error": "Insufficient data"}, {}

        features = period_df.values
        feature_tensor = torch.tensor(features, dtype=torch.float32, device=self.device)

        # Calculate VWAP with bands for the period (for VWAP breach detection)
        vwap_series = None
        upper_1sd_series = None
        lower_1sd_series = None
        upper_2sd_series = None
        lower_2sd_series = None
        if period_ohlcv is not None:
            vwap_series, upper_1sd_series, lower_1sd_series, upper_2sd_series, lower_2sd_series = calculate_vwap_with_bands(period_ohlcv)

        # Track signals from all models
        signals = {"1m": [], "5m": [], "15m": [], "vwap": [], "orb": []}
        # Track missed opportunities (triple confluence but no trade)
        missed_signals = []
        current_day = None
        desc = f"{year}-{month:02d}" if month else f"Year {year}"

        # Context lengths for each model
        all_context_lens = {"1m": 5, "5m": 15, "15m": 60}

        for i in tqdm(range(self.max_context_len, len(period_df)), desc=desc, leave=False):
            current_time = period_df.index[i]
            current_price = period_closes[i]

            et_time = current_time.tz_convert("America/New_York")
            et_hour = et_time.hour
            et_minute = et_time.minute

            if et_hour < 9 or (et_hour == 9 and et_minute < 30):
                continue

            # EOD handling: close positions at 4pm, but 15-min rule only applies to OPENING
            if et_hour >= 16:
                self.pending_entry = None  # Cancel any pending entries
                if self.position is not None:
                    self._close_position(current_time, current_price, "eod_close")
                continue

            # Check if we're in the first 10 minutes (no new positions - too volatile)
            is_first_10_minutes = (et_hour == 9 and et_minute < 40)

            # Check if we're in the last 15 minutes (no new positions, but can hold existing)
            is_last_15_minutes = (et_hour == 15 and et_minute >= 45)

            trading_date = et_time.date()
            if current_day != trading_date:
                current_day = trading_date
                self.options.load_date(trading_date)
                # Reset ORB tracking for new day
                self.current_or_high = None
                self.current_or_low = None
                self.pending_orb_breach = None

            minutes_elapsed = self._get_minutes_elapsed(current_time)

            # Calculate Opening Range after first 15 minutes (9:45 AM)
            if self.current_or_high is None and minutes_elapsed >= 15 and period_ohlcv is not None:
                # Find today's data start in period_ohlcv
                day_start_idx = None
                for j in range(max(0, i - 20), i + 1):
                    idx_time = period_ohlcv.index[j]
                    if hasattr(idx_time, 'tz_convert'):
                        idx_et = idx_time.tz_convert("America/New_York")
                    else:
                        idx_et = pd.Timestamp(idx_time).tz_convert("America/New_York")
                    if idx_et.date() == trading_date and idx_et.hour == 9 and idx_et.minute == 30:
                        day_start_idx = j
                        break
                if day_start_idx is not None and i >= day_start_idx + 14:
                    or_data = period_ohlcv.iloc[day_start_idx:day_start_idx + 15]
                    self.current_or_high = or_data['high'].max()
                    self.current_or_low = or_data['low'].min()

            contexts = self._build_contexts(feature_tensor, i)
            fused_signal = self.executor.get_signal(contexts, minutes_elapsed)

            # Get individual signals from all models for HTML report
            for model_name in ["1m", "5m", "15m"]:
                ctx_len = all_context_lens[model_name]
                if i >= ctx_len:
                    model_context = feature_tensor[i - ctx_len:i].unsqueeze(0)
                    try:
                        ind_signal, ind_probs = full_executor.get_individual_signal(model_name, model_context)
                        if ind_signal != Signal.HOLD:
                            signals[model_name].append(SignalMarker(
                                timestamp=int(current_time.timestamp()),
                                signal_type="long" if ind_signal == Signal.LONG else "short",
                                model=model_name,
                                price=current_price,
                            ))
                    except Exception:
                        pass  # Skip if model unavailable

            # Track VWAP breach signals for HTML report (potential breaches only - we always wait for confirmation)
            if vwap_series is not None and period_ohlcv is not None:
                vwap_breach_for_chart = detect_vwap_breach(period_ohlcv, vwap_series, i,
                                                            upper_1sd_series=upper_1sd_series,
                                                            lower_1sd_series=lower_1sd_series,
                                                            upper_2sd_series=upper_2sd_series,
                                                            lower_2sd_series=lower_2sd_series)
                if vwap_breach_for_chart.breach_type in (VWAPBreachType.BULLISH, VWAPBreachType.BULLISH_POTENTIAL):
                    signals["vwap"].append(SignalMarker(
                        timestamp=int(current_time.timestamp()),
                        signal_type="long",
                        model="vwap",
                        price=current_price,
                    ))
                elif vwap_breach_for_chart.breach_type in (VWAPBreachType.BEARISH, VWAPBreachType.BEARISH_POTENTIAL):
                    signals["vwap"].append(SignalMarker(
                        timestamp=int(current_time.timestamp()),
                        signal_type="short",
                        model="vwap",
                        price=current_price,
                    ))

            # Track ORB breach signals for HTML report
            if self.current_or_high is not None and period_ohlcv is not None:
                orb_breach_for_chart = detect_orb_breach(
                    period_ohlcv, i,
                    self.current_or_high, self.current_or_low
                )
                if orb_breach_for_chart.breach_type in (ORBBreachType.BULLISH, ORBBreachType.BULLISH_POTENTIAL):
                    signals["orb"].append(SignalMarker(
                        timestamp=int(current_time.timestamp()),
                        signal_type="long",
                        model="orb",
                        price=current_price,
                    ))
                elif orb_breach_for_chart.breach_type in (ORBBreachType.BEARISH, ORBBreachType.BEARISH_POTENTIAL):
                    signals["orb"].append(SignalMarker(
                        timestamp=int(current_time.timestamp()),
                        signal_type="short",
                        model="orb",
                        price=current_price,
                    ))

            # Execute pending entry at OPEN of this minute (signal was from previous minute)
            if self.pending_entry is not None and self.position is None:
                pe = self.pending_entry
                self.pending_entry = None

                # VWAP direction filter at execution time: no longs below VWAP, no shorts above VWAP
                current_vwap = vwap_series.iloc[i] if vwap_series is not None else None
                vwap_direction_ok = True
                if current_vwap is not None:
                    if pe.position_type == PositionType.CALL and current_price < current_vwap:
                        vwap_direction_ok = False  # Block long below VWAP
                    elif pe.position_type == PositionType.PUT and current_price > current_vwap:
                        vwap_direction_ok = False  # Block short above VWAP

                if vwap_direction_ok:
                    # Execute at this minute's open price
                    self._open_position(
                        pe.position_type,
                        current_time,  # Execute at current time (open of this minute)
                        current_price,
                        i,
                        pe.dominant_model,
                        pe.max_hold_minutes,
                        pe.confidence,
                        pe.confluence_count,
                        pe.agreeing_models,
                        pe.is_early_trade,
                        pe.is_vwap_breach,
                        pe.position_size_bonus,
                    )

            # Normal backtest logic
            if self.position is not None:
                # Get current band values for VWAP take profit checks
                curr_upper_2sd = upper_2sd_series.iloc[i] if upper_2sd_series is not None else None
                curr_lower_2sd = lower_2sd_series.iloc[i] if lower_2sd_series is not None else None
                curr_upper_1sd = upper_1sd_series.iloc[i] if upper_1sd_series is not None else None
                curr_lower_1sd = lower_1sd_series.iloc[i] if lower_1sd_series is not None else None

                # Get candle data for opposite direction exit check
                candle_open = None
                candle_close = None
                avg_body = None
                if period_ohlcv is not None:
                    candle_open = period_ohlcv.iloc[i]['open']
                    candle_close = period_ohlcv.iloc[i]['close']
                    # Calculate average body over last 20 candles
                    lookback = min(20, i)
                    if lookback > 0:
                        recent = period_ohlcv.iloc[i - lookback:i]
                        avg_body = (recent['close'] - recent['open']).abs().mean()

                exit_reason, should_renew, new_max_hold = self.check_exit_or_renew(
                    current_price, current_time, fused_signal,
                    upper_2sd=curr_upper_2sd, lower_2sd=curr_lower_2sd,
                    upper_1sd=curr_upper_1sd, lower_1sd=curr_lower_1sd,
                    candle_open=candle_open, candle_close=candle_close, avg_body=avg_body
                )
                if exit_reason:
                    if exit_reason == "convert_to_runner":
                        # Convert position to runner instead of closing
                        self._convert_to_runner(current_time, current_price)
                    else:
                        # Calculate stop loss trigger level for realistic fills
                        stop_trigger_pct = None
                        if "stop_loss" in exit_reason:
                            stop_trigger_pct = -self.config.stop_loss_pct
                        elif "trailing" in exit_reason or "breakeven" in exit_reason:
                            if self.position.is_runner:
                                # Runner trailing: 10% from runner peak
                                trail_dist = 10.0
                                stop_trigger_pct = self.position.runner_peak_pnl_pct - trail_dist
                            else:
                                trail_dist = 5.0  # Same for all trades - aggressive trailing
                                stop_trigger_pct = self.position.peak_pnl_pct - trail_dist
                        self._close_position(current_time, current_price, exit_reason, stop_trigger_pct)
                elif should_renew:
                    self.position.barrier_start_time = current_time
                    self.position.max_hold_minutes = new_max_hold
                    self.position.renewals += 1

            # Create pending entry (to be executed on NEXT minute's open)
            # Track why confluence signals don't result in trades
            has_confluence = len(fused_signal.agreeing_models) >= 3 and fused_signal.action != Signal.HOLD
            rejection_reason = None

            # Check for early window (9:45-10:00)
            trading_window = self.executor.fusion_config.trading_window
            is_early_window = trading_window.is_early_window(minutes_elapsed)

            if has_confluence:
                # Determine why we can't/won't trade
                if self.position is not None:
                    rejection_reason = "already_in_position"
                elif self.pending_entry is not None:
                    rejection_reason = "pending_entry_exists"
                elif not self.executor.is_trading_allowed(minutes_elapsed) and not is_early_window:
                    rejection_reason = f"time_restricted_min_{minutes_elapsed}"
                elif is_first_10_minutes:
                    rejection_reason = "first_10_minutes"
                elif is_last_15_minutes:
                    rejection_reason = "last_15_minutes"

            if self.position is None and self.pending_entry is None and not is_first_10_minutes and not is_last_15_minutes:
                final_action = Signal.HOLD
                is_early_trade = False
                is_vwap_breach = False
                agreeing_models = ()
                dominant_model = ""
                exit_horizon = 10  # Default for early trades
                confidence = 0.5

                # PRIORITY 1: Check for VWAP breach (always takes priority)
                if vwap_series is not None and period_ohlcv is not None:
                    vwap_breach = detect_vwap_breach(period_ohlcv, vwap_series, i,
                                                       upper_1sd_series=upper_1sd_series,
                                                       lower_1sd_series=lower_1sd_series,
                                                       upper_2sd_series=upper_2sd_series,
                                                       lower_2sd_series=lower_2sd_series)

                    # First, check if we have a pending potential breach that gets confirmed
                    if self.pending_vwap_breach is not None:
                        if check_continuation_confirmation(period_ohlcv, i, self.pending_vwap_breach):
                            # Continuation confirms the potential breach
                            if self.pending_vwap_breach.breach_type == VWAPBreachType.BULLISH_POTENTIAL:
                                final_action = Signal.LONG
                                is_vwap_breach = True
                                agreeing_models = ("vwap_breach_continuation",)
                                dominant_model = "vwap"
                                exit_horizon = 15
                                confidence = 0.7
                                self.vwap_continuation_entries += 1
                            elif self.pending_vwap_breach.breach_type == VWAPBreachType.BEARISH_POTENTIAL:
                                final_action = Signal.SHORT
                                is_vwap_breach = True
                                agreeing_models = ("vwap_breach_continuation",)
                                dominant_model = "vwap"
                                exit_horizon = 15
                                confidence = 0.7
                                self.vwap_continuation_entries += 1
                        # Clear pending breach after checking
                        self.pending_vwap_breach = None

                    # If no continuation entry, check for new VWAP breach
                    if not is_vwap_breach:
                        # Check for CONFIRMED breaches (strong body + volume) - enter immediately
                        if vwap_breach.breach_type == VWAPBreachType.BULLISH:
                            final_action = Signal.LONG
                            is_vwap_breach = True
                            agreeing_models = ("vwap_breach_immediate",)
                            dominant_model = "vwap"
                            exit_horizon = 15
                            confidence = 0.8  # High confidence for strong breach
                            self.vwap_immediate_entries += 1
                        elif vwap_breach.breach_type == VWAPBreachType.BEARISH:
                            final_action = Signal.SHORT
                            is_vwap_breach = True
                            agreeing_models = ("vwap_breach_immediate",)
                            dominant_model = "vwap"
                            exit_horizon = 15
                            confidence = 0.8  # High confidence for strong breach
                            self.vwap_immediate_entries += 1
                        # Check for POTENTIAL breaches - wait for confirmation candle
                        elif vwap_breach.breach_type in (
                            VWAPBreachType.BULLISH_POTENTIAL,
                            VWAPBreachType.BEARISH_POTENTIAL,
                        ):
                            # Track potential breach for next candle's continuation check
                            self.pending_vwap_breach = vwap_breach

                # PRIORITY 1.5: Check for ORB breach (after opening range is established)
                is_orb_breach = False
                if not is_vwap_breach and self.current_or_high is not None and period_ohlcv is not None:
                    # First, check if we have a pending ORB breach that gets confirmed
                    if self.pending_orb_breach is not None:
                        if check_orb_continuation_confirmation(period_ohlcv, i, self.pending_orb_breach):
                            # Continuation confirms the ORB breach
                            if self.pending_orb_breach.breach_type == ORBBreachType.BULLISH_POTENTIAL:
                                final_action = Signal.LONG
                                is_orb_breach = True
                                agreeing_models = ("orb_breakout",)
                                dominant_model = "orb"
                                exit_horizon = 15
                                confidence = 0.7
                                self.orb_entries += 1
                            elif self.pending_orb_breach.breach_type == ORBBreachType.BEARISH_POTENTIAL:
                                final_action = Signal.SHORT
                                is_orb_breach = True
                                agreeing_models = ("orb_breakout",)
                                dominant_model = "orb"
                                exit_horizon = 15
                                confidence = 0.7
                                self.orb_entries += 1
                        # Clear pending ORB breach after checking (whether confirmed or not)
                        self.pending_orb_breach = None

                    # If no ORB entry yet, check for new ORB breach
                    if not is_orb_breach:
                        orb_breach = detect_orb_breach(
                            period_ohlcv, i,
                            self.current_or_high, self.current_or_low
                        )
                        # Check for CONFIRMED breaches (strong breakout candles) - enter immediately
                        if orb_breach.breach_type == ORBBreachType.BULLISH:
                            final_action = Signal.LONG
                            is_orb_breach = True
                            agreeing_models = ("orb_breakout_immediate",)
                            dominant_model = "orb"
                            exit_horizon = 15
                            confidence = 0.8  # High confidence for strong breach
                            self.orb_immediate_entries += 1
                        elif orb_breach.breach_type == ORBBreachType.BEARISH:
                            final_action = Signal.SHORT
                            is_orb_breach = True
                            agreeing_models = ("orb_breakout_immediate",)
                            dominant_model = "orb"
                            exit_horizon = 15
                            confidence = 0.8  # High confidence for strong breach
                            self.orb_immediate_entries += 1
                        # Check for POTENTIAL breaches - wait for continuation candle
                        elif orb_breach.breach_type in (
                            ORBBreachType.BULLISH_POTENTIAL,
                            ORBBreachType.BEARISH_POTENTIAL,
                        ):
                            self.pending_orb_breach = orb_breach

                # PRIORITY 2: Early window (9:45-10:00) - only if no VWAP or ORB breach
                if not is_vwap_breach and not is_orb_breach and is_early_window:
                    # Early window (9:45-10:00): Require 1m+5m confluence + heuristic
                    early_agreeing = []
                    early_action = Signal.HOLD
                    early_confidence = 0.0

                    for model_name in ["1m", "5m"]:
                        if model_name in fused_signal.individual_signals:
                            sig, prob = fused_signal.individual_signals[model_name]
                            if sig != Signal.HOLD:
                                if early_action == Signal.HOLD:
                                    early_action = sig
                                    early_agreeing.append(model_name)
                                    early_confidence = prob
                                elif sig == early_action:
                                    early_agreeing.append(model_name)
                                    early_confidence = max(early_confidence, prob)
                                else:
                                    early_action = Signal.HOLD
                                    early_agreeing = []
                                    break

                    # Need both 1m and 5m to agree
                    if len(early_agreeing) == 2 and early_action != Signal.HOLD:
                        if self.use_heuristic and self.heuristic_model is not None and period_ohlcv is not None:
                            lookback = min(30, i)
                            ohlcv_window = period_ohlcv.iloc[i - lookback:i + 1]
                            heuristic_action = self.heuristic_model.get_signal(ohlcv_window, early_action)
                            if heuristic_action == early_action:
                                final_action = early_action
                                is_early_trade = True
                                agreeing_models = tuple(early_agreeing) + ("heuristic",)
                                dominant_model = "5m"
                                exit_horizon = 10
                                confidence = early_confidence

                # PRIORITY 3: Normal trading window (only if no VWAP/ORB breach and no early trade)
                elif not is_vwap_breach and not is_orb_breach and self.executor.is_trading_allowed(minutes_elapsed):
                    # Normal trading window
                    final_action = fused_signal.action
                    if self.use_heuristic and self.heuristic_model is not None and period_ohlcv is not None:
                        if fused_signal.action != Signal.HOLD:
                            lookback = min(30, i)
                            ohlcv_window = period_ohlcv.iloc[i - lookback:i + 1]
                            final_action = self.heuristic_model.get_signal(ohlcv_window, fused_signal.action)
                            if final_action == Signal.HOLD and fused_signal.action != Signal.HOLD:
                                self.heuristic_rejections += 1
                                if has_confluence:
                                    rejection_reason = "heuristic_rejected"

                    agreeing_models = fused_signal.agreeing_models
                    dominant_model = fused_signal.dominant_model
                    exit_horizon = fused_signal.exit_horizon_minutes
                    confidence = fused_signal.confidence

                # Get daily VWAP for timeframe support and position sizing
                daily_vwap = vwap_series.iloc[i] if vwap_series is not None else None
                position_size_bonus = 0.0

                # Check VWAP direction, timeframe support, and calculate position size bonus
                # (Skip for VWAP breach trades which have their own logic)
                if final_action != Signal.HOLD and not is_vwap_breach:
                    if self.use_heuristic and self.heuristic_model is not None and period_ohlcv is not None:
                        lookback = min(30, i)
                        ohlcv_window = period_ohlcv.iloc[i - lookback:i + 1]

                        # Check VWAP direction (LONG only above VWAP, SHORT only below VWAP)
                        # Note: VWAP breach trades skip this check (handled by `not is_vwap_breach` above)
                        vwap_direction_ok, _ = self.heuristic_model.check_vwap_direction(
                            final_action, current_price, daily_vwap
                        )
                        if not vwap_direction_ok:
                            self.vwap_direction_rejections += 1
                            if has_confluence:
                                rejection_reason = "vwap_direction_rejected"
                            final_action = Signal.HOLD

                        # Check timeframe support (5m or 15m VWAP)
                        if final_action != Signal.HOLD:
                            is_supported, _ = self.heuristic_model.check_timeframe_support(
                                ohlcv_window, final_action, current_price, daily_vwap
                            )
                            if not is_supported:
                                self.timeframe_rejections += 1
                                if has_confluence:
                                    rejection_reason = "timeframe_rejected"
                                final_action = Signal.HOLD
                            else:
                                # Calculate position size bonus
                                position_size_bonus, _ = self.heuristic_model.get_position_size_bonus(
                                    ohlcv_window, final_action, current_price, daily_vwap
                                )

                confluence_count = len(agreeing_models)
                # Combine VWAP and ORB breach flags for position sizing and strike selection
                is_breakout_trade = is_vwap_breach or is_orb_breach

                # VWAP direction filter: no longs below VWAP, no shorts above VWAP
                if daily_vwap is not None:
                    if final_action == Signal.LONG and current_price < daily_vwap:
                        rejection_reason = "long_below_vwap"
                        final_action = Signal.HOLD  # Block long below VWAP
                    elif final_action == Signal.SHORT and current_price > daily_vwap:
                        rejection_reason = "short_above_vwap"
                        final_action = Signal.HOLD  # Block short above VWAP

                if final_action == Signal.LONG:
                    rejection_reason = None  # Trade will be taken
                    self.pending_entry = PendingEntry(
                        position_type=PositionType.CALL,
                        signal_time=current_time,
                        underlying_price=current_price,
                        signal_idx=i,
                        dominant_model=dominant_model,
                        max_hold_minutes=exit_horizon,
                        confidence=confidence,
                        confluence_count=confluence_count,
                        agreeing_models=agreeing_models,
                        is_early_trade=is_early_trade,
                        is_vwap_breach=is_breakout_trade,
                        position_size_bonus=position_size_bonus,
                    )
                elif final_action == Signal.SHORT:
                    rejection_reason = None  # Trade will be taken
                    self.pending_entry = PendingEntry(
                        position_type=PositionType.PUT,
                        signal_time=current_time,
                        underlying_price=current_price,
                        signal_idx=i,
                        dominant_model=dominant_model,
                        max_hold_minutes=exit_horizon,
                        confidence=confidence,
                        confluence_count=confluence_count,
                        agreeing_models=agreeing_models,
                        is_early_trade=is_early_trade,
                        is_vwap_breach=is_breakout_trade,
                        position_size_bonus=position_size_bonus,
                    )

            # Track missed confluence signals
            if has_confluence and rejection_reason is not None:
                et_str = et_time.strftime("%Y-%m-%d %H:%M")
                missed_signals.append({
                    "timestamp": int(current_time.timestamp()),
                    "et_time": et_str,
                    "signal_type": "long" if fused_signal.action == Signal.LONG else "short",
                    "agreeing_models": fused_signal.agreeing_models,
                    "rejection_reason": rejection_reason,
                    "price": current_price,
                    "minutes_elapsed": minutes_elapsed,
                })

        # Clear pending entry at end of backtest
        self.pending_entry = None

        if self.position is not None:
            self._close_position(
                period_df.index[-1],
                period_closes[-1],
                "backtest_end"
            )

        # Print summary of missed signals
        if missed_signals:
            print(f"\n  Missed {len(missed_signals)} triple-confluence signals:")
            reason_counts = {}
            for ms in missed_signals:
                reason = ms["rejection_reason"]
                reason_counts[reason] = reason_counts.get(reason, 0) + 1
            for reason, count in sorted(reason_counts.items(), key=lambda x: -x[1]):
                print(f"    {reason}: {count}")
            # Show first 5 examples with details
            print(f"\n  First 5 missed signals (ET time):")
            for ms in missed_signals[:5]:
                print(f"    {ms['et_time']} - {ms['signal_type'].upper()} @ ${ms['price']:.2f} - {ms['rejection_reason']} (min={ms['minutes_elapsed']})")

        metrics = self._compute_metrics(year, month)
        metrics["missed_signals"] = missed_signals
        return metrics, signals

    def _convert_to_runner(self, current_time: datetime, current_price: float) -> None:
        """
        Convert current position to a runner (leave profits running).

        This records TWO trades:
        1. Main position exit at "runner_split" - captures the +20%+ profit
        2. Runner position that continues from here - starts at 0% P&L
        """
        if self.position is None:
            return

        current_option_price = self.options.get_option_price(
            timestamp=current_time,
            strike=self.position.strike,
            position_type=self.position.position_type,
            underlying_price=current_price,
        )

        if current_option_price is None:
            return

        # Calculate P&L for the main position
        pnl_pct = (current_option_price - self.position.entry_option_price) / self.position.entry_option_price * 100

        # Apply exit slippage to the split exit
        exit_option_price = current_option_price * (1 - self.config.exit_slippage_pct / 100)
        pnl_pct_with_slippage = (exit_option_price - self.position.entry_option_price) / self.position.entry_option_price * 100

        position_value = self.position.position_value

        # Determine runner percentage based on gain level:
        # - 100%+ gain: runner is 50% of position
        # - 50-100% gain: runner is 30% of position
        # - 20-50% gain: runner is 20% of position
        if pnl_pct >= 100.0:
            runner_pct = 0.50
        elif pnl_pct >= 50.0:
            runner_pct = 0.30
        else:  # 20-50%
            runner_pct = 0.20

        exit_pct = 1.0 - runner_pct  # Percentage we're exiting now
        exit_position_value = position_value * exit_pct

        full_pnl_dollars = position_value * (pnl_pct_with_slippage / 100)

        # Costs for the main position exit (exit portion)
        spread_cost = exit_position_value * (self.config.spread_cost_pct / 100)
        exit_contracts = int(self.position.num_contracts * exit_pct)
        if exit_contracts == 0:
            exit_contracts = 1
        commission_cost = self.config.commission_per_contract * exit_contracts
        total_costs = spread_cost + commission_cost
        pnl_dollars = (full_pnl_dollars * exit_pct) - total_costs

        # Update capital for main position exit (exit portion)
        self.capital += pnl_dollars
        self.peak_capital = max(self.peak_capital, self.capital)
        self.lowest_capital = min(self.lowest_capital, self.capital)
        self.daily_pnl += pnl_dollars
        self.daily_allocation -= exit_position_value  # Exit portion exits
        self._update_drawdown_tracking(current_time)

        # Track realized profit (only from the exited portion)
        self.daily_realized_profit += exit_position_value + pnl_dollars

        holding_minutes = int((current_time - self.position.entry_time).total_seconds() / 60)

        # Build contract name
        opt_type = "C" if self.position.position_type == PositionType.CALL else "P"
        contract = f"SPY {int(self.position.strike)}{opt_type}"

        # Store original entry time and price for the runner (so chart and table show original entry)
        original_entry_time = self.position.parent_entry_time or self.position.entry_time
        original_entry_option_price = self.position.parent_entry_option_price or self.position.entry_option_price

        # Split contracts based on gain-dependent runner percentage
        runner_contracts = int(self.position.num_contracts * runner_pct)
        if runner_contracts == 0:
            runner_contracts = 1
        main_contracts = self.position.num_contracts - runner_contracts
        if main_contracts == 0:
            main_contracts = 1
            runner_contracts = max(1, self.position.num_contracts - 1)

        # Record the main position exit as "runner_split" (half the position)
        # pnl_dollars already accounts for half position + costs from above
        main_trade = Trade(
            position_type=self.position.position_type,
            entry_time=self.position.entry_time,
            exit_time=current_time,
            entry_underlying=self.position.entry_underlying_price,
            exit_underlying=current_price,
            entry_option_price=self.position.entry_option_price,
            exit_option_price=exit_option_price,
            pnl_pct=pnl_pct_with_slippage,
            pnl_dollars=pnl_dollars,  # Already half from capital calculation
            capital_after=self.capital,
            exit_reason="runner_split",
            holding_minutes=holding_minutes,
            dominant_model=self.position.dominant_model,
            confluence_count=self.position.confluence_count,
            position_size_multiplier=self.position.position_size_multiplier,
            strike=self.position.strike,
            contract=contract,
            num_contracts=main_contracts,
            is_runner=False,
            parent_entry_time=original_entry_time,
        )
        self.trades.append(main_trade)

        # Now create a NEW runner position starting from here
        # Runner uses ORIGINAL entry price for P&L calculation (not split price)
        # Runner gets the percentage based on gain level (50%, 30%, or 20%)
        runner_position_value = position_value * runner_pct

        # Update daily allocation for runner entry
        self.daily_allocation += runner_position_value
        self.daily_cost_basis += runner_position_value

        # Save position attributes before overwriting
        old_position = self.position

        # Calculate current P&L% from original entry (for runner peak tracking)
        current_pnl_from_original = (current_option_price - original_entry_option_price) / original_entry_option_price * 100

        self.position = Position(
            position_type=old_position.position_type,
            entry_time=original_entry_time,  # Use original entry time
            entry_underlying_price=old_position.entry_underlying_price,  # Original underlying price
            entry_option_price=original_entry_option_price,  # Use ORIGINAL entry price for P&L calc
            strike=old_position.strike,
            entry_idx=old_position.entry_idx,  # Keep for reference
            position_value=runner_position_value,
            dominant_model=old_position.dominant_model,
            max_hold_minutes=15,  # Runner time barrier
            barrier_start_time=current_time,  # But barrier starts from split time
            confluence_count=old_position.confluence_count,
            position_size_multiplier=old_position.position_size_multiplier,
            is_runner=True,
            runner_start_time=current_time,
            runner_max_hold_minutes=15,
            runner_peak_pnl_pct=current_pnl_from_original,  # Start tracking from current P&L
            runner_entry_pnl_pct=current_pnl_from_original,  # P&L when runner started
            parent_entry_time=original_entry_time,  # Track original entry for chart
            parent_entry_option_price=original_entry_option_price,  # Track original entry price
            num_contracts=runner_contracts,  # Half the contracts go to runner
        )

    def _take_partial_profit(
        self,
        current_time: datetime,
        current_price: float,
        exit_reason: str,
        partial_pct: float = 0.20,
    ) -> None:
        """
        Take partial profit on a position (e.g., 20% at +1σ for VWAP trades).

        Records a trade for the partial exit and reduces position size accordingly.
        """
        if self.position is None:
            return

        current_option_price = self.options.get_option_price(
            timestamp=current_time,
            strike=self.position.strike,
            position_type=self.position.position_type,
            underlying_price=current_price,
        )

        if current_option_price is None:
            return

        # Calculate P&L for the full position (before split)
        pnl_pct = (current_option_price - self.position.entry_option_price) / self.position.entry_option_price * 100

        # Apply exit slippage
        exit_option_price = current_option_price * (1 - self.config.exit_slippage_pct / 100)
        pnl_pct_with_slippage = (exit_option_price - self.position.entry_option_price) / self.position.entry_option_price * 100

        position_value = self.position.position_value
        partial_position_value = position_value * partial_pct
        remaining_position_value = position_value * (1 - partial_pct)

        # Calculate P&L dollars for the partial exit
        full_pnl_dollars = position_value * (pnl_pct_with_slippage / 100)
        partial_pnl_dollars = full_pnl_dollars * partial_pct

        # Costs for the partial exit
        spread_cost = partial_position_value * (self.config.spread_cost_pct / 100)
        partial_contracts = int(self.position.num_contracts * partial_pct)
        if partial_contracts == 0:
            partial_contracts = 1
        commission_cost = self.config.commission_per_contract * partial_contracts
        total_costs = spread_cost + commission_cost
        pnl_dollars = partial_pnl_dollars - total_costs

        # Update capital for partial exit
        self.capital += pnl_dollars
        self.peak_capital = max(self.peak_capital, self.capital)
        self.lowest_capital = min(self.lowest_capital, self.capital)
        self.daily_pnl += pnl_dollars
        self.daily_allocation -= partial_position_value  # Partial exit reduces allocation
        self._update_drawdown_tracking(current_time)

        # Track realized profit
        self.daily_realized_profit += partial_position_value + pnl_dollars

        holding_minutes = int((current_time - self.position.entry_time).total_seconds() / 60)

        # Build contract name
        opt_type = "C" if self.position.position_type == PositionType.CALL else "P"
        contract = f"SPY {int(self.position.strike)}{opt_type}"

        # Record the partial exit trade
        partial_trade = Trade(
            position_type=self.position.position_type,
            entry_time=self.position.entry_time,
            exit_time=current_time,
            entry_underlying=self.position.entry_underlying_price,
            exit_underlying=current_price,
            entry_option_price=self.position.entry_option_price,
            exit_option_price=exit_option_price,
            pnl_pct=pnl_pct_with_slippage,
            pnl_dollars=pnl_dollars,
            capital_after=self.capital,
            exit_reason=exit_reason,
            holding_minutes=holding_minutes,
            dominant_model=self.position.dominant_model,
            confluence_count=self.position.confluence_count,
            position_size_multiplier=self.position.position_size_multiplier,
            strike=self.position.strike,
            contract=contract,
            num_contracts=partial_contracts,
            is_runner=False,
            parent_entry_time=self.position.parent_entry_time or self.position.entry_time,
        )
        self.trades.append(partial_trade)

        # Update position to reflect remaining size
        remaining_contracts = self.position.num_contracts - partial_contracts
        if remaining_contracts < 1:
            remaining_contracts = 1  # Keep at least 1 contract

        self.position.position_value = remaining_position_value
        self.position.num_contracts = remaining_contracts
        self.position.took_1sd_partial = True  # Mark that we've taken the +1σ partial

    def _open_position(
        self,
        position_type: PositionType,
        entry_time: datetime,
        underlying_price: float,
        entry_idx: int,
        dominant_model: str,
        max_hold_minutes: int,
        confidence: float = 0.5,
        confluence_count: int = 2,
        agreeing_models: Tuple[str, ...] = (),
        is_early_trade: bool = False,
        is_vwap_breach: bool = False,
        position_size_bonus: float = 0.0,
    ) -> bool:
        entry_date = entry_time.date()
        if self.current_day != entry_date:
            self.current_day = entry_date
            self.daily_trades = 0
            self.daily_allocation = 0.0
            self.daily_pnl = 0.0
            self.daily_start_capital = self.capital
            # Reset profit protection for new day
            self.daily_cost_basis = 0.0
            self.daily_realized_profit = 0.0
            self.daily_protected_capital = 0.0
            self.profit_protection_active = False
            self.max_daily_cost_basis = self.capital  # Can only invest up to starting balance

        if self.daily_trades >= self.config.max_trades_per_day:
            return False

        daily_loss_pct = -self.daily_pnl / self.daily_start_capital * 100
        if daily_loss_pct >= self.config.daily_loss_limit_pct:
            return False

        # Strike selection
        # ORB breakout trades: Always ITM (conservative)
        # VWAP breakout trades: Time-based strike selection
        #   - Before 10:30am or after 3pm: OTM (1 strike away from ITM)
        #   - 10:30am to 3pm: ATM
        # Early trades (9:45-10:00): Always ATM
        # Normal trades: Based on confidence (ITM default, OTM for high confidence)
        et_time = entry_time.astimezone(self.et_tz) if entry_time.tzinfo else entry_time
        hour, minute = et_time.hour, et_time.minute
        time_minutes = hour * 60 + minute
        is_otm_time = time_minutes < 10 * 60 + 30 or time_minutes >= 15 * 60  # Before 10:30am or after 3pm
        is_orb_trade = dominant_model == "orb"

        if is_orb_trade:
            # ORB breakout trades: Always ITM (conservative)
            if position_type == PositionType.CALL:
                strike = int(underlying_price)  # ITM for calls (floor)
            else:
                strike = int(underlying_price) + 1  # ITM for puts (ceiling)
        elif is_vwap_breach:
            # VWAP breakout trades: time-based strike selection
            if is_otm_time:
                # Before 10:30am or after 3pm: OTM (1 strike away from ITM)
                if position_type == PositionType.CALL:
                    strike = int(underlying_price) + 1  # 1 strike OTM for calls
                else:
                    strike = int(underlying_price)  # 1 strike OTM for puts (floor)
            else:
                # 10:30am to 3pm: ATM (round to nearest strike)
                strike = round(underlying_price)
        elif is_early_trade:
            # Early window trades are always ATM (round to nearest strike)
            strike = round(underlying_price)
        elif position_type == PositionType.CALL:
            # For calls: ITM = lower strike, OTM = higher strike
            if confidence >= 0.90:
                strike = int(underlying_price) + 2  # 2 strikes OTM
            elif confidence >= 0.60:
                strike = int(underlying_price) + 1  # 1 strike OTM
            else:
                strike = int(underlying_price)  # ITM (floor)
        else:
            # For puts: ITM = higher strike, OTM = lower strike
            if confidence >= 0.90:
                strike = int(underlying_price) - 1  # 2 strikes OTM (round up then -2)
            elif confidence >= 0.60:
                strike = int(underlying_price)  # 1 strike OTM
            else:
                strike = int(underlying_price) + 1  # ITM (ceiling)

        # Get option price at OPEN of execution minute + slippage
        entry_option_price = self.options.get_option_price(
            timestamp=entry_time,
            strike=strike,
            position_type=position_type,
            underlying_price=underlying_price,
            price_type="open",  # Fill at open price
        )

        if entry_option_price is None:
            return False

        # Add slippage (we pay more to enter)
        entry_option_price *= (1 + self.config.entry_slippage_pct / 100)

        # Position size multiplier: 2x when portfolio is under $15k (to recover faster)
        size_multiplier = 2.0 if self.capital < 15000 else 1.0

        # Power hour boost: max position size during configured window (default 2-4pm ET)
        if self.config.power_hour_boost_enabled:
            if self.config.power_hour_start_minutes <= time_minutes < self.config.power_hour_end_minutes:
                position_size_bonus = 0.02  # Max bonus during power hour

        # Dynamic position sizing
        # ORB breakout trades: 2% of current balance (very conservative)
        # VWAP breakout trades: 3% of current balance (conservative)
        # Normal trades: 3% of current balance + bonus from heuristic confluence
        if is_orb_trade:
            target_position_value = self.capital * 0.02 * size_multiplier  # 2% (4% if under 15k)
        elif is_vwap_breach:
            # VWAP trades: 3% + heuristic bonus, capped at 5% (10% if under 15k)
            total_pct = min(0.03 + position_size_bonus, 0.05)
            target_position_value = self.capital * total_pct * size_multiplier
        else:
            base_pct = self.heuristic_config.base_position_pct if self.heuristic_config else 0.03
            total_pct = base_pct + position_size_bonus
            target_position_value = self.capital * total_pct * size_multiplier

        # Calculate number of contracts based on option price
        # Option price is per share, contracts are 100 shares
        cost_per_contract = entry_option_price * 100
        if cost_per_contract <= 0:
            # Skip if option price is zero or negative (bad data)
            return False
        num_contracts = max(1, int(target_position_value / cost_per_contract))

        # Actual position value is contracts * cost
        position_value = num_contracts * cost_per_contract

        # Check max daily cost basis limit (can only invest up to starting balance)
        if self.daily_cost_basis + position_value > self.max_daily_cost_basis:
            # Cap to remaining allowed cost basis
            remaining_allowed = self.max_daily_cost_basis - self.daily_cost_basis
            if remaining_allowed <= 0:
                return False
            # Reduce contracts to fit
            num_contracts = max(1, int(remaining_allowed / cost_per_contract))
            position_value = num_contracts * cost_per_contract

        max_daily = self.capital * (self.config.max_daily_allocation_pct / 100)
        if self.daily_allocation + position_value > max_daily:
            return False

        self.position = Position(
            position_type=position_type,
            entry_time=entry_time,
            entry_underlying_price=underlying_price,
            entry_option_price=entry_option_price,
            strike=strike,
            entry_idx=entry_idx,
            position_value=position_value,
            dominant_model=dominant_model,
            max_hold_minutes=max_hold_minutes,
            confluence_count=confluence_count,
            position_size_multiplier=1.0,  # No longer using confidence multiplier
            num_contracts=num_contracts,
            is_vwap_breach=is_vwap_breach,
        )

        if dominant_model in self.model_contributions:
            self.model_contributions[dominant_model] += 1

        self.daily_trades += 1
        self.daily_allocation += position_value

        # Track cost basis for profit protection
        self.daily_cost_basis += position_value

        return True

    def _close_position(
        self,
        exit_time: datetime,
        exit_underlying_price: float,
        exit_reason: str,
        stop_loss_trigger_pct: Optional[float] = None,  # For stop loss fills
    ):
        if self.position is None:
            return

        # Determine exit price based on exit type
        is_stop_loss = "stop" in exit_reason.lower() or "trailing" in exit_reason.lower()

        if is_stop_loss and stop_loss_trigger_pct is not None:
            # For stop losses: add variable slippage (0-10% of stop loss amount)
            # e.g., 10% stop loss exits between 10-11% loss
            additional_slippage_pct = random.uniform(0, 0.10) * abs(stop_loss_trigger_pct)
            effective_stop_pct = stop_loss_trigger_pct - additional_slippage_pct  # More negative = worse fill
            stop_trigger_price = self.position.entry_option_price * (1 + effective_stop_pct / 100)
            # Apply normal slippage from stop price
            exit_option_price = stop_trigger_price * (1 - self.config.exit_slippage_pct / 100)
        else:
            # Normal exit: use close price
            exit_option_price = self.options.get_option_price(
                timestamp=exit_time,
                strike=self.position.strike,
                position_type=self.position.position_type,
                underlying_price=exit_underlying_price,
                price_type="close",
            )
            if exit_option_price is not None:
                # Add exit slippage
                exit_option_price *= (1 - self.config.exit_slippage_pct / 100)

        if exit_option_price is None:
            exit_option_price = self.position.entry_option_price
            exit_reason = f"{exit_reason}_no_data"

        pnl_pct = (exit_option_price - self.position.entry_option_price) / self.position.entry_option_price * 100

        # Breakeven stops have a tight auto-stop that guarantees 0% minimum
        # (simulates a stop-limit order placed at entry price when breakeven activates)
        if exit_reason == "breakeven_stop" and pnl_pct < 0:
            pnl_pct = 0.0
            exit_option_price = self.position.entry_option_price

        if self.config.max_trade_gain_pct is not None:
            pnl_pct = min(pnl_pct, self.config.max_trade_gain_pct)

        position_value = self.position.position_value
        pnl_dollars = position_value * (pnl_pct / 100)

        spread_cost = position_value * (self.config.spread_cost_pct / 100)
        commission_cost = 2 * self.config.commission_per_contract * self.config.contracts_per_trade
        total_costs = spread_cost + commission_cost

        pnl_dollars -= total_costs

        self.capital += pnl_dollars
        self.peak_capital = max(self.peak_capital, self.capital)
        self.lowest_capital = min(self.lowest_capital, self.capital)
        self.daily_pnl += pnl_dollars
        self.daily_allocation -= position_value
        self._update_drawdown_tracking(exit_time)

        # Track realized profit for profit protection
        self.daily_realized_profit += position_value + pnl_dollars  # Return of capital + profit

        # Check if we've hit 100% profit threshold (only activates once per day)
        # When realized profit >= 2x cost basis, we've doubled our money
        if not self.profit_protection_active and self.daily_cost_basis > 0:
            if self.daily_realized_profit >= 2 * self.daily_cost_basis:
                # Activate profit protection ONCE - set aside original cost basis
                self.profit_protection_active = True
                self.daily_protected_capital = self.daily_cost_basis
                # Continue trading with remaining profits (no further protection)

        holding_minutes = int((exit_time - self.position.entry_time).total_seconds() / 60)

        # Mark exit reason if this was a runner (for unexpected exits like EOD)
        # Runner P&L is from original entry, compare gain since split point
        if self.position.is_runner and not exit_reason.startswith("runner"):
            gain_since_split = pnl_pct - self.position.runner_entry_pnl_pct
            if gain_since_split >= 5.0:
                exit_reason = f"runner_profit_{exit_reason}"
            elif gain_since_split >= 0.0:
                exit_reason = f"runner_breakeven_{exit_reason}"
            else:
                exit_reason = f"runner_loss_{exit_reason}"

        # Build contract name (e.g., "SPY 600C")
        opt_type = "C" if self.position.position_type == PositionType.CALL else "P"
        contract = f"SPY {int(self.position.strike)}{opt_type}"

        # Position entry_time and entry_option_price are already original values for runners
        trade = Trade(
            position_type=self.position.position_type,
            entry_time=self.position.entry_time,
            exit_time=exit_time,
            entry_underlying=self.position.entry_underlying_price,
            exit_underlying=exit_underlying_price,
            entry_option_price=self.position.entry_option_price,
            exit_option_price=exit_option_price,
            pnl_pct=pnl_pct,
            pnl_dollars=pnl_dollars,
            capital_after=self.capital,
            exit_reason=exit_reason,
            holding_minutes=holding_minutes,
            dominant_model=self.position.dominant_model,
            confluence_count=self.position.confluence_count,
            position_size_multiplier=self.position.position_size_multiplier,
            strike=self.position.strike,
            contract=contract,
            num_contracts=self.position.num_contracts,
            is_runner=self.position.is_runner,
            parent_entry_time=self.position.parent_entry_time,
            parent_entry_option_price=self.position.parent_entry_option_price,
        )
        self.trades.append(trade)
        self.position = None

    def _compute_metrics(self, year: int, month: Optional[int] = None) -> Dict:
        initial = self.config.initial_capital

        # Finalize drawdown tracking at end of backtest
        self._finalize_drawdown_tracking()

        if not self.trades:
            return {
                "year": year,
                "month": month,
                "trades": 0,
                "trades_per_day": 0.0,
                "trading_days": 0,
                "win_rate": 0.0,
                "avg_pnl_pct": 0.0,
                "total_return_pct": 0.0,
                "total_pnl_dollars": 0.0,
                "final_capital": initial,
                "max_drawdown_pct": 0.0,
                "avg_daily_pnl_pct": 0.0,
                "avg_weekly_pnl_pct": 0.0,
                "daily_win_rate": 0.0,
                "weekly_win_rate": 0.0,
                "green_days": 0,
                "red_days": 0,
                "total_days": 0,
                "green_weeks": 0,
                "red_weeks": 0,
                "total_weeks": 0,
                "avg_win_pct": 0.0,
                "avg_loss_pct": 0.0,
                "long_trades": 0,
                "short_trades": 0,
                "avg_hold_minutes": 0.0,
                "exit_reasons": {},
                "model_contributions": self.model_contributions.copy(),
            }

        pnls = [t.pnl_pct for t in self.trades]
        wins = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p <= 0]

        exit_reasons = {}
        for t in self.trades:
            exit_reasons[t.exit_reason] = exit_reasons.get(t.exit_reason, 0) + 1

        long_trades = [t for t in self.trades if t.position_type == PositionType.CALL]
        short_trades = [t for t in self.trades if t.position_type == PositionType.PUT]

        trading_days = len(set(t.entry_time.date() for t in self.trades))

        final_capital = self.trades[-1].capital_after
        total_return_pct = (final_capital - initial) / initial * 100
        total_pnl_dollars = sum(t.pnl_dollars for t in self.trades)

        # Max drawdown = how much below initial capital did we ever go
        # If lowest_capital >= initial, drawdown is 0
        if self.lowest_capital < initial:
            max_dd = (initial - self.lowest_capital) / initial * 100
        else:
            max_dd = 0.0

        model_trades = {"15m": 0, "5m": 0, "1m": 0}
        model_pnl = {"15m": 0.0, "5m": 0.0, "1m": 0.0}
        for t in self.trades:
            if t.dominant_model in model_trades:
                model_trades[t.dominant_model] += 1
                model_pnl[t.dominant_model] += t.pnl_dollars

        # Daily/Weekly win rate and average P/L calculations
        green_days = sum(1 for d in self.daily_results if d > 0)
        red_days = sum(1 for d in self.daily_results if d <= 0)
        total_days = len(self.daily_results)
        daily_win_rate = (green_days / total_days * 100) if total_days > 0 else 0.0
        avg_daily_pnl_pct = np.mean(self.daily_results) if self.daily_results else 0.0

        green_weeks = sum(1 for w in self.weekly_results if w > 0)
        red_weeks = sum(1 for w in self.weekly_results if w <= 0)
        total_weeks = len(self.weekly_results)
        weekly_win_rate = (green_weeks / total_weeks * 100) if total_weeks > 0 else 0.0
        avg_weekly_pnl_pct = np.mean(self.weekly_results) if self.weekly_results else 0.0

        return {
            "year": year,
            "month": month,
            "trades": len(self.trades),
            "trades_per_day": len(self.trades) / trading_days if trading_days > 0 else 0,
            "trading_days": trading_days,
            "win_rate": len(wins) / len(self.trades) * 100 if self.trades else 0,
            "avg_pnl_pct": np.mean(pnls),
            "total_return_pct": total_return_pct,
            "total_pnl_dollars": total_pnl_dollars,
            "final_capital": final_capital,
            "max_drawdown_pct": max_dd,
            "avg_daily_pnl_pct": avg_daily_pnl_pct,
            "avg_weekly_pnl_pct": avg_weekly_pnl_pct,
            "daily_win_rate": daily_win_rate,
            "weekly_win_rate": weekly_win_rate,
            "green_days": green_days,
            "red_days": red_days,
            "total_days": total_days,
            "green_weeks": green_weeks,
            "red_weeks": red_weeks,
            "total_weeks": total_weeks,
            "avg_win_pct": np.mean(wins) if wins else 0,
            "avg_loss_pct": np.mean(losses) if losses else 0,
            "long_trades": len(long_trades),
            "short_trades": len(short_trades),
            "long_pnl": sum(t.pnl_dollars for t in long_trades),
            "short_pnl": sum(t.pnl_dollars for t in short_trades),
            "avg_hold_minutes": np.mean([t.holding_minutes for t in self.trades]),
            "exit_reasons": exit_reasons,
            "model_contributions": self.model_contributions.copy(),
            "model_trades": model_trades,
            "model_pnl": model_pnl,
        }
