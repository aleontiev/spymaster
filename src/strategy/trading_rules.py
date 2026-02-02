"""
Trading rules for entry signals.

Contains breach detection, reversal detection,
and continuation confirmation logic extracted from the backtester for reuse
in live trading.
"""
from dataclasses import dataclass
from enum import Enum, IntEnum
from typing import Optional, Tuple

import numpy as np
import pandas as pd


class BreachType(Enum):
    """Type of breach signal."""
    NONE = 0
    BULLISH = 1  # Price crossed above VWAP with volume confirmation
    BEARISH = 2  # Price crossed below VWAP with volume confirmation
    BULLISH_POTENTIAL = 3  # Price crossed above VWAP but lacks volume (needs continuation)
    BEARISH_POTENTIAL = 4  # Price crossed below VWAP but lacks volume (needs continuation)


@dataclass
class BreachSignal:
    """Breach signal details."""
    breach_type: BreachType
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


def calculate_vwap_with_bands(
    ohlcv_df: pd.DataFrame
) -> Tuple[pd.Series, pd.Series, pd.Series, pd.Series, pd.Series]:
    """
    Calculate VWAP with standard deviation bands anchored at start of each trading day.

    VWAP = cumulative(typical_price * volume) / cumulative(volume)
    Standard deviation bands at +-1 and +-2 standard deviations.
    Resets at start of each new trading day.

    Args:
        ohlcv_df: OHLCV DataFrame with open, high, low, close, volume columns

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


def calculate_vwap_series(ohlcv_df: pd.DataFrame) -> pd.Series:
    """
    Calculate VWAP anchored at start of each trading day.

    VWAP = cumulative(typical_price * volume) / cumulative(volume)
    Resets at start of each new trading day.

    Args:
        ohlcv_df: OHLCV DataFrame

    Returns:
        Series with VWAP values
    """
    vwap, _, _, _, _ = calculate_vwap_with_bands(ohlcv_df)
    return vwap


def detect_breach(
    ohlcv_df: pd.DataFrame,
    vwap_series: pd.Series,
    idx: int,
    lookback_minutes: int = 5,
    vwap_margin: float = 0.02,  # Margin for open/close near VWAP (2 cents)
    upper_1sd_series: Optional[pd.Series] = None,
    lower_1sd_series: Optional[pd.Series] = None,
    upper_2sd_series: Optional[pd.Series] = None,
    lower_2sd_series: Optional[pd.Series] = None,
) -> BreachSignal:
    """
    Detect VWAP breach - strictly requires body to cross VWAP.

    VWAP Bullish breach:
    - Candle opens at or below VWAP (within margin)
    - Candle closes at or above VWAP (within margin)
    - Body must cross VWAP

    VWAP Bearish breach:
    - Candle opens at or above VWAP (within margin)
    - Candle closes at or below VWAP (within margin)
    - Body must cross VWAP

    Args:
        ohlcv_df: OHLCV DataFrame
        vwap_series: Pre-calculated VWAP series
        idx: Current index in DataFrame
        lookback_minutes: Minutes to look back for average volume
        vwap_margin: Tolerance for open/close near VWAP
        upper_1sd_series: Optional +1σ band series for immediate entry threshold
        lower_1sd_series: Optional -1σ band series for immediate entry threshold
        upper_2sd_series: Optional +2σ band series for band reversion detection
        lower_2sd_series: Optional -2σ band series for band reversion detection

    Returns:
        BreachSignal with breach details
    """
    if idx < lookback_minutes:
        return BreachSignal(
            breach_type=BreachType.NONE,
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
        return BreachSignal(
            breach_type=BreachType.NONE,
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
    open_at_or_below_vwap = open_price <= vwap + vwap_margin
    close_at_or_above_vwap = close >= vwap - vwap_margin
    is_bullish_breach = open_at_or_below_vwap and close_at_or_above_vwap and close > open_price

    # Prior-close check: for a bullish breakout, at least one of the preceding
    # candles must have *closed* below VWAP (proving price was established below).
    # Without this, wicks above VWAP followed by a dip look like false breakdowns.
    if is_bullish_breach:
        has_prior_close_below = False
        for j in range(idx - 1, max(0, idx - lookback_minutes) - 1, -1):
            if ohlcv_df.iloc[j]['close'] < vwap_series.iloc[j]:
                has_prior_close_below = True
                break
        if not has_prior_close_below:
            is_bullish_breach = False

    if is_bullish_breach:
        # Check for immediate entry conditions
        upper_wick = high - close
        body_to_wick_ratio = body / upper_wick if upper_wick > 0.001 else 100.0
        strong_body = body_to_wick_ratio >= 3.5

        high_volume_current = volume_ratio >= 2.5

        lookback_start = max(0, idx - lookback_minutes)
        recent_bodies = (ohlcv_df.iloc[lookback_start:idx]['close'] - ohlcv_df.iloc[lookback_start:idx]['open']).abs()
        avg_body = recent_bodies.mean() if len(recent_bodies) > 0 else body
        large_body = body >= avg_body * 1.5

        opens_near_vwap = abs(open_price - vwap) <= 0.03
        closes_well_above = close >= vwap + max(0.15, avg_body * 1.5)
        is_breakout_candle = opens_near_vwap and closes_well_above and strong_body and large_body

        # Check 1σ band threshold
        upper_1sd = upper_1sd_series.iloc[idx] if upper_1sd_series is not None else None
        meets_1sd_threshold = False
        is_overextended = False
        if upper_1sd is not None and upper_1sd > vwap:
            band_distance = upper_1sd - vwap
            move_from_vwap = close - vwap
            move_pct = move_from_vwap / band_distance if band_distance > 0 else 0
            meets_1sd_threshold = move_pct >= 0.25
            is_overextended = close > upper_1sd

        can_enter_immediate = (
            (meets_1sd_threshold and not is_overextended) or
            ((is_breakout_candle or (strong_body and high_volume_current and large_body)) and not is_overextended)
        )

        if can_enter_immediate:
            return BreachSignal(
                breach_type=BreachType.BULLISH,
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
            return BreachSignal(
                breach_type=BreachType.BULLISH_POTENTIAL,
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
    open_at_or_above_vwap = open_price >= vwap - vwap_margin
    close_at_or_below_vwap = close <= vwap + vwap_margin
    is_bearish_breach = open_at_or_above_vwap and close_at_or_below_vwap and close < open_price

    # Prior-close check: for a bearish breakdown, at least one of the preceding
    # candles must have *closed* above VWAP (proving price was established above).
    if is_bearish_breach:
        has_prior_close_above = False
        for j in range(idx - 1, max(0, idx - lookback_minutes) - 1, -1):
            if ohlcv_df.iloc[j]['close'] > vwap_series.iloc[j]:
                has_prior_close_above = True
                break
        if not has_prior_close_above:
            is_bearish_breach = False

    if is_bearish_breach:
        lower_wick = close - low
        body_to_wick_ratio = body / lower_wick if lower_wick > 0.001 else 100.0
        strong_body = body_to_wick_ratio >= 3.5

        high_volume_current = volume_ratio >= 2.5

        lookback_start = max(0, idx - lookback_minutes)
        recent_bodies = (ohlcv_df.iloc[lookback_start:idx]['close'] - ohlcv_df.iloc[lookback_start:idx]['open']).abs()
        avg_body = recent_bodies.mean() if len(recent_bodies) > 0 else body
        large_body = body >= avg_body * 1.5

        opens_near_vwap = abs(open_price - vwap) <= 0.03
        closes_well_below = close <= vwap - max(0.15, avg_body * 1.5)
        is_breakout_candle = opens_near_vwap and closes_well_below and strong_body and large_body

        lower_1sd = lower_1sd_series.iloc[idx] if lower_1sd_series is not None else None
        meets_1sd_threshold = False
        is_overextended = False
        if lower_1sd is not None and lower_1sd < vwap:
            band_distance = vwap - lower_1sd
            move_from_vwap = vwap - close
            move_pct = move_from_vwap / band_distance if band_distance > 0 else 0
            meets_1sd_threshold = move_pct >= 0.25
            is_overextended = close < lower_1sd

        can_enter_immediate = (
            (meets_1sd_threshold and not is_overextended) or
            ((is_breakout_candle or (strong_body and high_volume_current and large_body)) and not is_overextended)
        )

        if can_enter_immediate:
            return BreachSignal(
                breach_type=BreachType.BEARISH,
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
            return BreachSignal(
                breach_type=BreachType.BEARISH_POTENTIAL,
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
    return BreachSignal(
        breach_type=BreachType.NONE,
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


def check_breach_continuation_confirmation(
    ohlcv_df: pd.DataFrame,
    idx: int,
    potential_breach: BreachSignal,
    min_body_ratio: float = 0.3,
    lookback_minutes: int = 20,
    abnormal_move_threshold: float = 5.0,
    abnormal_volume_required: float = 5.0,
) -> bool:
    """
    Check if current candle confirms a potential VWAP breach from the previous candle.

    For bullish potential breach: current candle must be entirely above VWAP
    - Must be green (close > open)
    - Must have some body (not indecisive)
    - Must OPEN above VWAP (candle starts on bullish side)
    - Must CLOSE above VWAP (candle ends on bullish side)

    For bearish potential breach: current candle must be entirely below VWAP
    - Must be red (close < open)
    - Must have some body (not indecisive)
    - Must OPEN below VWAP (candle starts on bearish side)
    - Must CLOSE below VWAP (candle ends on bearish side)

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
    is_abnormal_move = avg_body > 0 and current_body >= abnormal_move_threshold * avg_body
    has_abnormal_volume = avg_volume > 0 and current_volume >= abnormal_volume_required * avg_volume

    if is_abnormal_move and not has_abnormal_volume:
        return False

    if potential_breach.breach_type == BreachType.BULLISH_POTENTIAL:
        is_green = current_close > current_open
        body_low = min(current_open, current_close)
        body_above_vwap = body_low > potential_breach.vwap_value
        closes_above_vwap = current_close > potential_breach.vwap_value
        # Wick must not dip below breakout candle's low (fake breakout rejection)
        wick_ok = current_low >= potential_breach.candle_low
        return is_green and has_body and body_above_vwap and closes_above_vwap and wick_ok

    elif potential_breach.breach_type == BreachType.BEARISH_POTENTIAL:
        is_red = current_close < current_open
        body_high = max(current_open, current_close)
        body_below_vwap = body_high < potential_breach.vwap_value
        closes_below_vwap = current_close < potential_breach.vwap_value
        # Wick must not rise above breakout candle's high (fake breakout rejection)
        wick_ok = current_high <= potential_breach.candle_high
        return is_red and has_body and body_below_vwap and closes_below_vwap and wick_ok

    return False


class ReversalType(IntEnum):
    """Type of reversal candle pattern."""
    NONE = 0
    BULLISH_HAMMER = 1      # Hammer at -2σ band
    BULLISH_ENGULFING = 2   # Bullish engulfing at -2σ band
    BEARISH_SHOOTING = 3    # Shooting star at +2σ band
    BEARISH_ENGULFING = 4   # Bearish engulfing at +2σ band


@dataclass
class ReversalSignal:
    """Reversal signal details."""
    reversal_type: ReversalType
    zone: str  # "lower_band", "upper_band"
    candle_open: float
    candle_high: float
    candle_low: float
    candle_close: float
    volume: float
    volume_ratio: float
    band_value: float  # The band value that triggered the reversal
    vwap_value: float
    is_engulfing: bool = False


class BounceType(IntEnum):
    """Type of VWAP bounce signal."""
    NONE = 0
    BULLISH_HAMMER = 1      # Hammer at VWAP support
    BULLISH_ENGULFING = 2   # Bullish engulfing at VWAP support
    BEARISH_SHOOTING = 3    # Shooting star at VWAP resistance
    BEARISH_ENGULFING = 4   # Bearish engulfing at VWAP resistance


@dataclass
class BounceSignal:
    """VWAP bounce signal details."""
    bounce_type: BounceType
    candle_open: float
    candle_high: float
    candle_low: float
    candle_close: float
    volume: float
    volume_ratio: float
    vwap_value: float
    is_engulfing: bool = False


class DoubleBreakoutType(IntEnum):
    """Type of double breakout signal."""
    NONE = 0
    BULLISH = 1  # Bullish double breakout (break above → retrace → re-break → confirm)
    BEARISH = 2  # Bearish double breakout (break below → retrace → re-break → confirm)


@dataclass
class DoubleBreakoutSignal:
    """Double breakout signal details."""
    breakout_type: DoubleBreakoutType
    vwap_value: float
    candle_a_close: float  # Breakout candle close
    candle_d_close: float  # Confirmation candle close
    candle_d_open: float
    candle_d_high: float
    candle_d_low: float
    volume: float


def detect_double_breakout(
    ohlcv_df: pd.DataFrame,
    vwap_series: pd.Series,
    idx: int,
) -> DoubleBreakoutSignal:
    """
    Detect a 4-candle double breakout pattern around VWAP.

    Pattern (bullish):
    - Candle A (idx-3): Breakout — crosses above VWAP (open < VWAP, close > VWAP)
    - Candle B (idx-2): Retrace — closes lower than A (pulls back toward VWAP)
    - Candle C (idx-1): Re-break — closes higher than B and above VWAP
    - Candle D (idx):   Confirm — bullish body (close > open) OR doji near high

    Pattern (bearish):
    - Candle A (idx-3): Breakout — crosses below VWAP (open > VWAP, close < VWAP)
    - Candle B (idx-2): Retrace — closes higher than A (pulls back toward VWAP)
    - Candle C (idx-1): Re-break — closes lower than B and below VWAP
    - Candle D (idx):   Confirm — bearish body (close < open) OR doji near low

    Args:
        ohlcv_df: OHLCV DataFrame
        vwap_series: Pre-calculated VWAP series
        idx: Current index (candle D)

    Returns:
        DoubleBreakoutSignal with detection results
    """
    no_signal = DoubleBreakoutSignal(
        breakout_type=DoubleBreakoutType.NONE,
        vwap_value=0, candle_a_close=0, candle_d_close=0,
        candle_d_open=0, candle_d_high=0, candle_d_low=0, volume=0,
    )

    if idx < 3 or idx >= len(ohlcv_df):
        return no_signal

    # Get candles A through D
    a = ohlcv_df.iloc[idx - 3]
    b = ohlcv_df.iloc[idx - 2]
    c = ohlcv_df.iloc[idx - 1]
    d = ohlcv_df.iloc[idx]

    vwap_a = vwap_series.iloc[idx - 3]
    vwap_c = vwap_series.iloc[idx - 1]
    vwap_d = vwap_series.iloc[idx]

    a_open, a_close = a['open'], a['close']
    b_close = b['close']
    c_close = c['close']
    d_open, d_high, d_low, d_close = d['open'], d['high'], d['low'], d['close']
    d_volume = d['volume']

    d_range = d_high - d_low
    d_body = abs(d_close - d_open)
    is_doji_d = d_range > 0 and (d_body / d_range) < 0.20

    # --- BULLISH DOUBLE BREAKOUT ---
    # A: crosses above VWAP
    a_crosses_above = a_open < vwap_a and a_close > vwap_a
    # B: retraces (closes lower than A)
    b_retraces = b_close < a_close
    # C: re-breaks above (closes higher than B and above VWAP)
    c_rebreaks = c_close > b_close and c_close > vwap_c
    # D: confirms bullish (green candle or doji near high)
    d_bullish = d_close > d_open
    d_doji_near_high = is_doji_d and d_range > 0 and (d_high - max(d_open, d_close)) < 0.3 * d_range

    if a_crosses_above and b_retraces and c_rebreaks and (d_bullish or d_doji_near_high):
        return DoubleBreakoutSignal(
            breakout_type=DoubleBreakoutType.BULLISH,
            vwap_value=vwap_d,
            candle_a_close=a_close,
            candle_d_close=d_close,
            candle_d_open=d_open,
            candle_d_high=d_high,
            candle_d_low=d_low,
            volume=d_volume,
        )

    # --- BEARISH DOUBLE BREAKOUT ---
    # A: crosses below VWAP
    a_crosses_below = a_open > vwap_a and a_close < vwap_a
    # B: retraces (closes higher than A)
    b_retraces_bear = b_close > a_close
    # C: re-breaks below (closes lower than B and below VWAP)
    c_rebreaks_bear = c_close < b_close and c_close < vwap_c
    # D: confirms bearish (red candle or doji near low)
    d_bearish = d_close < d_open
    d_doji_near_low = is_doji_d and d_range > 0 and (min(d_open, d_close) - d_low) < 0.3 * d_range

    if a_crosses_below and b_retraces_bear and c_rebreaks_bear and (d_bearish or d_doji_near_low):
        return DoubleBreakoutSignal(
            breakout_type=DoubleBreakoutType.BEARISH,
            vwap_value=vwap_d,
            candle_a_close=a_close,
            candle_d_close=d_close,
            candle_d_open=d_open,
            candle_d_high=d_high,
            candle_d_low=d_low,
            volume=d_volume,
        )

    return no_signal


def _is_doji(candle_open: float, candle_high: float, candle_low: float, candle_close: float) -> bool:
    """Check if a candle is a doji (body < 20% of range)."""
    candle_range = candle_high - candle_low
    if candle_range <= 0:
        return True  # No range = doji
    body = abs(candle_close - candle_open)
    return body / candle_range < 0.20


def detect_reversal(
    ohlcv_df: pd.DataFrame,
    vwap_series: pd.Series,
    upper_2sd_series: pd.Series,
    lower_2sd_series: pd.Series,
    idx: int,
) -> ReversalSignal:
    """
    Detect mean-reversion reversal candle patterns at +-2σ VWAP bands only.

    Checks for:
    1. Bullish reversal at -2σ band (hammer or bullish engulfing)
    2. Bearish reversal at +2σ band (shooting star or bearish engulfing)

    Conditions for all reversals:
    - Green/red candle in the expected direction
    - Volume > prior candle's volume
    - Neither current nor prior candle is a doji
    - Current candle must either move more than prior candle body OR have
      higher volume with a significant body (>= 50% of range)
    - Hammer: bottom wick >= 2x body (bullish) or top wick >= 2x body (bearish)
    - Engulfing: current body fully contains prior candle body

    Args:
        ohlcv_df: OHLCV DataFrame
        vwap_series: VWAP values series
        upper_2sd_series: Upper 2σ band series
        lower_2sd_series: Lower 2σ band series
        idx: Current candle index

    Returns:
        ReversalSignal with detection results (NONE if no reversal)
    """
    no_signal = ReversalSignal(
        reversal_type=ReversalType.NONE,
        zone="none",
        candle_open=0, candle_high=0, candle_low=0, candle_close=0,
        volume=0, volume_ratio=0, band_value=0, vwap_value=0,
    )

    if idx < 1 or idx >= len(ohlcv_df):
        return no_signal

    current = ohlcv_df.iloc[idx]
    prior = ohlcv_df.iloc[idx - 1]

    c_open = current['open']
    c_high = current['high']
    c_low = current['low']
    c_close = current['close']
    c_volume = current['volume']

    p_open = prior['open']
    p_high = prior['high']
    p_low = prior['low']
    p_close = prior['close']
    p_volume = prior['volume']

    vwap = vwap_series.iloc[idx]
    upper_2sd = upper_2sd_series.iloc[idx]
    lower_2sd = lower_2sd_series.iloc[idx]

    # Volume must exceed prior candle
    if c_volume <= p_volume or p_volume <= 0:
        return no_signal

    volume_ratio = c_volume / p_volume

    body = abs(c_close - c_open)
    prior_body = abs(p_close - p_open)
    candle_range = c_high - c_low

    # Anti-doji filter: neither current nor prior candle should be a doji
    if _is_doji(c_open, c_high, c_low, c_close):
        return no_signal
    if _is_doji(p_open, p_high, p_low, p_close):
        return no_signal

    # Reversal candle must move the price: either move more than prior candle
    # body OR have higher volume with significant body (>= 50% of range)
    moves_more = body > prior_body
    significant_body = candle_range > 0 and (body / candle_range) >= 0.50
    has_price_action = moves_more or (volume_ratio >= 1.5 and significant_body)
    if not has_price_action:
        return no_signal

    body = max(body, 0.001)  # Avoid division by zero
    is_green = c_close > c_open
    is_red = c_close < c_open

    # Wick calculations
    bottom_wick = min(c_open, c_close) - c_low
    top_wick = c_high - max(c_open, c_close)

    # Prior candle body bounds
    prior_body_high = max(p_open, p_close)
    prior_body_low = min(p_open, p_close)

    # Current candle body bounds
    curr_body_high = max(c_open, c_close)
    curr_body_low = min(c_open, c_close)

    # Check engulfing: current body fully contains prior body
    is_engulfing_pattern = curr_body_high > prior_body_high and curr_body_low < prior_body_low

    # Determine zone and check reversal patterns
    # BULLISH checks: green candle + (hammer OR engulfing) at -2σ band
    if is_green:
        zone = None
        band_value = 0.0
        if c_low <= lower_2sd:
            zone = "lower_band"
            band_value = lower_2sd

        if zone is not None:
            is_hammer = bottom_wick >= 2.0 * body
            if is_hammer:
                return ReversalSignal(
                    reversal_type=ReversalType.BULLISH_HAMMER,
                    zone=zone,
                    candle_open=c_open, candle_high=c_high,
                    candle_low=c_low, candle_close=c_close,
                    volume=c_volume, volume_ratio=volume_ratio,
                    band_value=band_value, vwap_value=vwap,
                    is_engulfing=False,
                )
            if is_engulfing_pattern:
                return ReversalSignal(
                    reversal_type=ReversalType.BULLISH_ENGULFING,
                    zone=zone,
                    candle_open=c_open, candle_high=c_high,
                    candle_low=c_low, candle_close=c_close,
                    volume=c_volume, volume_ratio=volume_ratio,
                    band_value=band_value, vwap_value=vwap,
                    is_engulfing=True,
                )

    # BEARISH checks: red candle + (shooting star OR engulfing) at +2σ band
    if is_red:
        zone = None
        band_value = 0.0
        if c_high >= upper_2sd:
            zone = "upper_band"
            band_value = upper_2sd

        if zone is not None:
            is_shooting_star = top_wick >= 2.0 * body
            if is_shooting_star:
                return ReversalSignal(
                    reversal_type=ReversalType.BEARISH_SHOOTING,
                    zone=zone,
                    candle_open=c_open, candle_high=c_high,
                    candle_low=c_low, candle_close=c_close,
                    volume=c_volume, volume_ratio=volume_ratio,
                    band_value=band_value, vwap_value=vwap,
                    is_engulfing=False,
                )
            if is_engulfing_pattern:
                return ReversalSignal(
                    reversal_type=ReversalType.BEARISH_ENGULFING,
                    zone=zone,
                    candle_open=c_open, candle_high=c_high,
                    candle_low=c_low, candle_close=c_close,
                    volume=c_volume, volume_ratio=volume_ratio,
                    band_value=band_value, vwap_value=vwap,
                    is_engulfing=True,
                )

    return no_signal


def detect_vwap_bounce(
    ohlcv_df: pd.DataFrame,
    vwap_series: pd.Series,
    idx: int,
) -> BounceSignal:
    """
    Detect VWAP support/resistance bounce with approach direction check and stickiness filter.

    Requirements:
    1. Only triggers when price is moving towards VWAP and cleanly bounces off
    2. If price has been near VWAP recently (last 5 candles), do not trade (stickiness)
    3. Same candle pattern requirements as reversal: volume > prior, hammer/engulfing

    Anti-doji filter: neither current nor prior candle should be a doji.
    Current candle must either move more than prior candle body OR have higher
    volume with a significant body.

    Args:
        ohlcv_df: OHLCV DataFrame
        vwap_series: VWAP values series
        idx: Current candle index

    Returns:
        BounceSignal with detection results (NONE if no bounce)
    """
    no_signal = BounceSignal(
        bounce_type=BounceType.NONE,
        candle_open=0, candle_high=0, candle_low=0, candle_close=0,
        volume=0, volume_ratio=0, vwap_value=0,
    )

    # Need at least 6 candles of history (5 lookback + current)
    if idx < 6 or idx >= len(ohlcv_df):
        return no_signal

    current = ohlcv_df.iloc[idx]
    prior = ohlcv_df.iloc[idx - 1]

    c_open = current['open']
    c_high = current['high']
    c_low = current['low']
    c_close = current['close']
    c_volume = current['volume']

    p_open = prior['open']
    p_high = prior['high']
    p_low = prior['low']
    p_close = prior['close']
    p_volume = prior['volume']

    vwap = vwap_series.iloc[idx]

    # Volume must exceed prior candle
    if c_volume <= p_volume or p_volume <= 0:
        return no_signal

    volume_ratio = c_volume / p_volume

    body = abs(c_close - c_open)
    prior_body = abs(p_close - p_open)
    candle_range = c_high - c_low

    # Anti-doji filter
    if _is_doji(c_open, c_high, c_low, c_close):
        return no_signal
    if _is_doji(p_open, p_high, p_low, p_close):
        return no_signal

    # Candle must move price meaningfully
    moves_more = body > prior_body
    significant_body = candle_range > 0 and (body / candle_range) >= 0.50
    has_price_action = moves_more or (volume_ratio >= 1.5 and significant_body)
    if not has_price_action:
        return no_signal

    body = max(body, 0.001)  # Avoid division by zero
    is_green = c_close > c_open
    is_red = c_close < c_open

    # --- STICKINESS FILTER ---
    # Check last 5 candles (idx-5 to idx-1). If ANY close is within 0.05% of VWAP,
    # price is hovering near VWAP — not a clean bounce.
    vwap_threshold = vwap * 0.0005  # 0.05% of VWAP
    for j in range(max(0, idx - 5), idx):
        prev_close = ohlcv_df.iloc[j]['close']
        prev_vwap = vwap_series.iloc[j]
        if abs(prev_close - prev_vwap) <= prev_vwap * 0.0005:
            return no_signal

    # --- APPROACH DIRECTION CHECK ---
    # Bullish bounce: at least 2 of prior 3 candles must have declining closes
    # Bearish bounce: at least 2 of prior 3 candles must have rising closes
    prior_closes = [ohlcv_df.iloc[idx - k]['close'] for k in range(3, 0, -1)]  # [idx-3, idx-2, idx-1]
    declining_count = sum(1 for k in range(1, len(prior_closes)) if prior_closes[k] < prior_closes[k - 1])
    rising_count = sum(1 for k in range(1, len(prior_closes)) if prior_closes[k] > prior_closes[k - 1])

    # Wick calculations
    bottom_wick = min(c_open, c_close) - c_low
    top_wick = c_high - max(c_open, c_close)

    # Prior candle body bounds (for engulfing)
    prior_body_high = max(p_open, p_close)
    prior_body_low = min(p_open, p_close)
    curr_body_high = max(c_open, c_close)
    curr_body_low = min(c_open, c_close)
    is_engulfing_pattern = curr_body_high > prior_body_high and curr_body_low < prior_body_low

    # BULLISH VWAP bounce: low dips to/below VWAP, close above VWAP, green candle
    # Price must have been approaching from above (declining closes)
    if is_green and c_low <= vwap and c_close > vwap and declining_count >= 2:
        is_hammer = bottom_wick >= 2.0 * body
        if is_hammer:
            return BounceSignal(
                bounce_type=BounceType.BULLISH_HAMMER,
                candle_open=c_open, candle_high=c_high,
                candle_low=c_low, candle_close=c_close,
                volume=c_volume, volume_ratio=volume_ratio,
                vwap_value=vwap, is_engulfing=False,
            )
        if is_engulfing_pattern:
            return BounceSignal(
                bounce_type=BounceType.BULLISH_ENGULFING,
                candle_open=c_open, candle_high=c_high,
                candle_low=c_low, candle_close=c_close,
                volume=c_volume, volume_ratio=volume_ratio,
                vwap_value=vwap, is_engulfing=True,
            )

    # BEARISH VWAP bounce: high rises to/above VWAP, close below VWAP, red candle
    # Price must have been approaching from below (rising closes)
    if is_red and c_high >= vwap and c_close < vwap and rising_count >= 2:
        is_shooting_star = top_wick >= 2.0 * body
        if is_shooting_star:
            return BounceSignal(
                bounce_type=BounceType.BEARISH_SHOOTING,
                candle_open=c_open, candle_high=c_high,
                candle_low=c_low, candle_close=c_close,
                volume=c_volume, volume_ratio=volume_ratio,
                vwap_value=vwap, is_engulfing=False,
            )
        if is_engulfing_pattern:
            return BounceSignal(
                bounce_type=BounceType.BEARISH_ENGULFING,
                candle_open=c_open, candle_high=c_high,
                candle_low=c_low, candle_close=c_close,
                volume=c_volume, volume_ratio=volume_ratio,
                vwap_value=vwap, is_engulfing=True,
            )

    return no_signal


class NewsEventType(IntEnum):
    """Type of news event signal."""
    NONE = 0
    BULLISH = 1  # Large green candle → CALL continuation
    BEARISH = 2  # Large red candle → PUT continuation


@dataclass
class NewsEventSignal:
    """News event signal details."""
    event_type: NewsEventType
    volume_percentile: float  # Percentile rank of volume (0-1)
    body_rank: int  # Rank of body size among today's candles (1 = largest)
    candle_open: float
    candle_high: float
    candle_low: float
    candle_close: float
    volume: float


def detect_news_event(
    ohlcv_df: pd.DataFrame,
    idx: int,
    upper_1sd: Optional[float] = None,
    lower_1sd: Optional[float] = None,
    upper_2sd: Optional[float] = None,
    lower_2sd: Optional[float] = None,
) -> NewsEventSignal:
    """
    Detect a high-conviction news event candle (big volume + big body + minimal opposing wick).

    Criteria:
    1. Volume in top 10% of bars seen so far today
    2. Body size is the largest for the day so far
    3. Opposing wick (upper for red, lower for green) <= 15% of total candle range
    4. Candle close beyond +-2σ VWAP band (bullish: close > upper_2sd, bearish: close < lower_2sd)
    5. Candle open at least 1σ away from the close band:
       - Bullish: open < +1σ (candle must travel from below +1σ to above +2σ)
       - Bearish: open > -1σ (candle must travel from above -1σ to below -2σ)

    Direction: continuation — red candle → BEARISH (PUT), green candle → BULLISH (CALL).

    Args:
        ohlcv_df: OHLCV DataFrame with DatetimeIndex
        idx: Current index in DataFrame
        upper_1sd: +1σ VWAP band value at current bar
        lower_1sd: -1σ VWAP band value at current bar
        upper_2sd: +2σ VWAP band value at current bar (required for signal)
        lower_2sd: -2σ VWAP band value at current bar (required for signal)

    Returns:
        NewsEventSignal with detection results (NONE if no event)
    """
    no_signal = NewsEventSignal(
        event_type=NewsEventType.NONE,
        volume_percentile=0.0, body_rank=0,
        candle_open=0, candle_high=0, candle_low=0, candle_close=0, volume=0,
    )

    if idx < 1 or idx >= len(ohlcv_df):
        return no_signal

    current = ohlcv_df.iloc[idx]
    c_open = current['open']
    c_high = current['high']
    c_low = current['low']
    c_close = current['close']
    c_volume = current['volume']

    candle_range = c_high - c_low
    if candle_range <= 0:
        return no_signal

    body = abs(c_close - c_open)
    is_green = c_close > c_open
    is_red = c_close < c_open

    if not is_green and not is_red:
        return no_signal  # Doji — no direction

    # Opposing wick check: must be <= 15% of total range
    if is_green:
        opposing_wick = c_high - c_close  # Upper wick for green candle
    else:
        opposing_wick = c_close - c_low  # Lower wick for red candle (close is at bottom of body)

    if opposing_wick > 0.15 * candle_range:
        return no_signal

    # Criterion 4: Close must be beyond +-2σ VWAP band
    # Criterion 5: Open must be at least 1σ away from the close band
    #   Bullish: close > +2σ AND open < +1σ
    #   Bearish: close < -2σ AND open > -1σ
    if is_green:
        if upper_2sd is None or upper_1sd is None:
            return no_signal
        if c_close <= upper_2sd:
            return no_signal
        if c_open >= upper_1sd:
            return no_signal  # Open too close to +2σ — not a big enough move
    else:  # is_red
        if lower_2sd is None or lower_1sd is None:
            return no_signal
        if c_close >= lower_2sd:
            return no_signal
        if c_open <= lower_1sd:
            return no_signal  # Open too close to -2σ — not a big enough move

    # Collect today's data up to and including current bar
    current_ts = ohlcv_df.index[idx]
    if hasattr(current_ts, 'date'):
        current_date = current_ts.date()
    else:
        current_date = pd.Timestamp(current_ts).date()

    # Find start of today
    day_start_idx = idx
    for j in range(idx - 1, -1, -1):
        ts_j = ohlcv_df.index[j]
        if hasattr(ts_j, 'date'):
            d_j = ts_j.date()
        else:
            d_j = pd.Timestamp(ts_j).date()
        if d_j != current_date:
            break
        day_start_idx = j

    today_data = ohlcv_df.iloc[day_start_idx:idx + 1]

    if len(today_data) < 5:
        return no_signal  # Need minimum data

    # Criterion 1: Volume in top 10%
    today_volumes = today_data['volume'].values
    volume_threshold = np.percentile(today_volumes, 90)
    if c_volume < volume_threshold:
        return no_signal

    volume_percentile = (today_volumes < c_volume).sum() / len(today_volumes)

    # Criterion 2: Body size must be the largest of the day so far
    today_bodies = (today_data['close'] - today_data['open']).abs().values
    # Rank: count how many bodies are larger than current
    larger_count = (today_bodies > body).sum()
    body_rank = larger_count + 1  # 1 = largest
    if body_rank > 1:
        return no_signal

    event_type = NewsEventType.BULLISH if is_green else NewsEventType.BEARISH

    return NewsEventSignal(
        event_type=event_type,
        volume_percentile=volume_percentile,
        body_rank=body_rank,
        candle_open=c_open,
        candle_high=c_high,
        candle_low=c_low,
        candle_close=c_close,
        volume=c_volume,
    )
