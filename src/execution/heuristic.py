"""
Heuristic trading model for trend confirmation.

Rule-based filter that confirms or rejects ML model signals based on
candle patterns and price action.
"""
from typing import Optional, Tuple

import pandas as pd

from src.strategy.fusion_config import Signal


class HeuristicModel:
    """
    Rule-based trading model for trend confirmation.

    Rules:
    1. Don't go SHORT on a green candle (close > open)
    2. Don't go LONG on a red candle (close < open)
    3. Exception: Reversal candles (wick >= 2x body AND substantial range/volume)
       - Even for reversals, wait for confirmation of movement
    """

    def __init__(self, lookback_minutes: int = 30):
        self.lookback_minutes = lookback_minutes

    def get_signal(
        self,
        ohlcv_data: pd.DataFrame,  # Must have open, high, low, close, volume columns
        proposed_signal: Signal,  # LONG or SHORT from ML models
    ) -> Signal:
        """
        Confirm or reject a proposed signal based on heuristic rules.

        Args:
            ohlcv_data: Recent OHLCV data (at least 2 rows)
            proposed_signal: The signal proposed by ML models

        Returns:
            Confirmed signal or HOLD if rejected
        """
        if proposed_signal == Signal.HOLD:
            return Signal.HOLD

        if len(ohlcv_data) < 2:
            return Signal.HOLD

        # Get current and recent candles
        current = ohlcv_data.iloc[-1]
        recent = ohlcv_data.iloc[-self.lookback_minutes:] if len(ohlcv_data) >= self.lookback_minutes else ohlcv_data

        open_p = current['open']
        close_p = current['close']

        # Determine candle color
        is_green = close_p > open_p
        is_red = close_p < open_p

        # Check for reversal candle
        is_reversal, reversal_type, has_confirmation = self._check_reversal(current, recent)

        if proposed_signal == Signal.LONG:
            if is_green:
                return Signal.LONG  # Green candle confirms long - trend following
            elif is_red:
                # Exception: bullish reversal with confirmation
                if is_reversal and reversal_type == 'bullish' and has_confirmation:
                    return Signal.LONG
                return Signal.HOLD  # Reject long on red candle

        elif proposed_signal == Signal.SHORT:
            if is_red:
                return Signal.SHORT  # Red candle confirms short - trend following
            elif is_green:
                # Exception: bearish reversal with confirmation
                if is_reversal and reversal_type == 'bearish' and has_confirmation:
                    return Signal.SHORT
                return Signal.HOLD  # Reject short on green candle

        return Signal.HOLD

    def _check_reversal(
        self,
        candle: pd.Series,
        recent_data: pd.DataFrame,
    ) -> Tuple[bool, Optional[str], bool]:
        """
        Check if candle is a reversal candle with confirmation.

        Returns:
            (is_reversal, reversal_type, has_confirmation)
            reversal_type is 'bullish' or 'bearish'
        """
        body = abs(candle['close'] - candle['open'])
        if body == 0:
            body = 0.0001  # Avoid division by zero (doji)

        upper_wick = candle['high'] - max(candle['open'], candle['close'])
        lower_wick = min(candle['open'], candle['close']) - candle['low']
        total_range = candle['high'] - candle['low']
        mid_point = (candle['high'] + candle['low']) / 2

        # Check for substantial range/volume (dynamic from recent candles)
        avg_range = (recent_data['high'] - recent_data['low']).mean()
        is_substantial = total_range > 1.25 * avg_range

        if 'volume' in recent_data.columns:
            avg_vol = recent_data['volume'].mean()
            if avg_vol > 0:
                vol = candle.get('volume', 0)
                is_substantial = is_substantial or (vol > 1.5 * avg_vol)

        if not is_substantial:
            return False, None, False

        # Check wick ratio (wick must be >= 1.5x body)
        max_wick = max(upper_wick, lower_wick)
        if max_wick < 1.5 * body:
            return False, None, False

        # Determine reversal type and confirmation
        if lower_wick > upper_wick:
            # Bullish reversal (hammer pattern) - long lower wick
            # Confirmation: close should be in upper half of range (buyers pushing up)
            has_confirmation = candle['close'] > mid_point
            return True, 'bullish', has_confirmation
        else:
            # Bearish reversal (shooting star pattern) - long upper wick
            # Confirmation: close should be in lower half of range (sellers pushing down)
            has_confirmation = candle['close'] < mid_point
            return True, 'bearish', has_confirmation
