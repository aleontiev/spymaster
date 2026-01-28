"""
Heuristic trading model for trend confirmation.

Rule-based filter that confirms or rejects ML model signals based on
candle patterns and price action.
"""
from dataclasses import dataclass
from typing import Optional, Tuple

import pandas as pd

from src.strategy.fusion_config import Signal


__all__ = ['HeuristicModel', 'HeuristicConfig']


@dataclass
class HeuristicConfig:
    """Configuration for heuristic model features."""
    # VWAP direction filter settings
    use_vwap_direction_filter: bool = True  # Reject LONG below VWAP, SHORT above VWAP
    vwap_direction_slippage: float = 0.05  # Small tolerance for VWAP comparison (5 cents)

    # Timeframe filter settings
    use_timeframe_filter: bool = True
    timeframe_slippage: float = 0.01  # Allowed slippage for VWAP comparison
    vwap_5m_periods: int = 5  # Minutes for 5m VWAP calculation
    vwap_15m_periods: int = 15  # Minutes for 15m VWAP calculation

    # Dynamic position sizing settings
    use_dynamic_sizing: bool = True
    base_position_pct: float = 0.03  # Base position size (3%)
    triple_confluence_boost_pct: float = 0.01  # Boost when above all VWAPs (1%)
    volume_candle_boost_pct: float = 0.01  # Boost for high volume + good body (1%)
    max_total_boost_pct: float = 0.02  # Maximum boost (2%)
    volume_threshold_multiplier: float = 2.5  # Volume must be 2.5x average
    min_body_wick_ratio: float = 2.0  # Body must be 2x the opposing wick


class HeuristicModel:
    """
    Rule-based trading model for trend confirmation.

    Rules:
    1. Don't go SHORT on a green candle (close > open)
    2. Don't go LONG on a red candle (close < open)
    3. Exception: Reversal candles (wick >= 2x body AND substantial range/volume)
       - Even for reversals, wait for confirmation of movement
    4. (Optional) Entry must be supported by 5m or 15m VWAP
    5. (Optional) Dynamic position sizing based on confluence
    """

    def __init__(
        self,
        lookback_minutes: int = 30,
        config: Optional[HeuristicConfig] = None,
    ):
        self.lookback_minutes = lookback_minutes
        self.config = config or HeuristicConfig()

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

    def _calculate_period_vwap(
        self,
        ohlcv_data: pd.DataFrame,
        periods: int,
    ) -> Optional[float]:
        """
        Calculate VWAP over the last N periods.

        VWAP = sum(typical_price * volume) / sum(volume)
        where typical_price = (high + low + close) / 3

        Args:
            ohlcv_data: OHLCV DataFrame
            periods: Number of periods to include

        Returns:
            VWAP value or None if insufficient data
        """
        if len(ohlcv_data) < periods:
            return None

        recent = ohlcv_data.iloc[-periods:]
        typical_price = (recent['high'] + recent['low'] + recent['close']) / 3
        volume = recent['volume']

        total_volume = volume.sum()
        if total_volume <= 0:
            return None

        vwap = (typical_price * volume).sum() / total_volume
        return vwap

    def check_timeframe_support(
        self,
        ohlcv_data: pd.DataFrame,
        proposed_signal: Signal,
        entry_price: float,
        daily_vwap: Optional[float] = None,
    ) -> Tuple[bool, dict]:
        """
        Check if entry price is supported by 5m or 15m VWAP.

        For LONG: price should be at/above 5m VWAP OR 15m VWAP (with slippage)
        For SHORT: price should be at/below 5m VWAP OR 15m VWAP (with slippage)

        Args:
            ohlcv_data: Recent OHLCV data
            proposed_signal: LONG or SHORT
            entry_price: The price we're considering entering at
            daily_vwap: Optional daily VWAP for additional confluence check

        Returns:
            (is_supported, details_dict) where details contains VWAP values
        """
        if not self.config.use_timeframe_filter:
            return True, {}

        if proposed_signal == Signal.HOLD:
            return True, {}

        # Calculate 5m and 15m VWAPs
        vwap_5m = self._calculate_period_vwap(
            ohlcv_data, self.config.vwap_5m_periods
        )
        vwap_15m = self._calculate_period_vwap(
            ohlcv_data, self.config.vwap_15m_periods
        )

        details = {
            'vwap_5m': vwap_5m,
            'vwap_15m': vwap_15m,
            'daily_vwap': daily_vwap,
            'entry_price': entry_price,
        }

        slippage = self.config.timeframe_slippage

        if proposed_signal == Signal.LONG:
            # For LONG: price should be at/above 5m VWAP OR 15m VWAP
            above_5m = vwap_5m is not None and entry_price >= (vwap_5m - slippage)
            above_15m = vwap_15m is not None and entry_price >= (vwap_15m - slippage)
            is_supported = above_5m or above_15m
            details['above_5m'] = above_5m
            details['above_15m'] = above_15m
        else:  # SHORT
            # For SHORT: price should be at/below 5m VWAP OR 15m VWAP
            below_5m = vwap_5m is not None and entry_price <= (vwap_5m + slippage)
            below_15m = vwap_15m is not None and entry_price <= (vwap_15m + slippage)
            is_supported = below_5m or below_15m
            details['below_5m'] = below_5m
            details['below_15m'] = below_15m

        return is_supported, details

    def get_position_size_bonus(
        self,
        ohlcv_data: pd.DataFrame,
        proposed_signal: Signal,
        entry_price: float,
        daily_vwap: Optional[float] = None,
    ) -> Tuple[float, dict]:
        """
        Calculate position size bonus based on confluence factors.

        Bonus factors:
        1. Triple VWAP confluence (+1%): entry above/below 5m AND 15m AND daily VWAP
        2. Strong candle (+1%): high volume (2.5x avg) AND body >> opposing wick

        Args:
            ohlcv_data: Recent OHLCV data
            proposed_signal: LONG or SHORT
            entry_price: The price we're considering entering at
            daily_vwap: Daily VWAP value

        Returns:
            (bonus_pct, details_dict) where bonus is 0.0 to max_total_boost_pct
        """
        if not self.config.use_dynamic_sizing:
            return 0.0, {}

        if proposed_signal == Signal.HOLD:
            return 0.0, {}

        bonus = 0.0
        details = {
            'triple_confluence': False,
            'strong_candle': False,
        }

        # Calculate VWAPs
        vwap_5m = self._calculate_period_vwap(
            ohlcv_data, self.config.vwap_5m_periods
        )
        vwap_15m = self._calculate_period_vwap(
            ohlcv_data, self.config.vwap_15m_periods
        )

        slippage = self.config.timeframe_slippage

        # Check triple VWAP confluence
        if vwap_5m is not None and vwap_15m is not None and daily_vwap is not None:
            if proposed_signal == Signal.LONG:
                above_5m = entry_price >= (vwap_5m - slippage)
                above_15m = entry_price >= (vwap_15m - slippage)
                above_daily = entry_price >= (daily_vwap - slippage)
                if above_5m and above_15m and above_daily:
                    bonus += self.config.triple_confluence_boost_pct
                    details['triple_confluence'] = True
            else:  # SHORT
                below_5m = entry_price <= (vwap_5m + slippage)
                below_15m = entry_price <= (vwap_15m + slippage)
                below_daily = entry_price <= (daily_vwap + slippage)
                if below_5m and below_15m and below_daily:
                    bonus += self.config.triple_confluence_boost_pct
                    details['triple_confluence'] = True

        # Check for strong candle (high volume + body >> opposing wick)
        if len(ohlcv_data) >= self.lookback_minutes:
            recent = ohlcv_data.iloc[-self.lookback_minutes:]
            current = ohlcv_data.iloc[-1]

            # Check volume
            avg_volume = recent['volume'].mean()
            current_volume = current['volume']
            high_volume = (
                avg_volume > 0 and
                current_volume >= self.config.volume_threshold_multiplier * avg_volume
            )

            # Check body vs opposing wick
            body = abs(current['close'] - current['open'])
            upper_wick = current['high'] - max(current['open'], current['close'])
            lower_wick = min(current['open'], current['close']) - current['low']

            if proposed_signal == Signal.LONG:
                # For long, opposing wick is upper wick
                opposing_wick = upper_wick
            else:
                # For short, opposing wick is lower wick
                opposing_wick = lower_wick

            # Avoid division by zero
            if opposing_wick <= 0:
                opposing_wick = 0.0001

            good_body = body >= self.config.min_body_wick_ratio * opposing_wick

            if high_volume and good_body:
                bonus += self.config.volume_candle_boost_pct
                details['strong_candle'] = True
                details['volume_ratio'] = current_volume / avg_volume if avg_volume > 0 else 0
                details['body_wick_ratio'] = body / opposing_wick

        # Cap at max boost
        bonus = min(bonus, self.config.max_total_boost_pct)
        details['total_bonus'] = bonus

        return bonus, details

    def check_vwap_direction(
        self,
        proposed_signal: Signal,
        entry_price: float,
        daily_vwap: Optional[float],
    ) -> Tuple[bool, dict]:
        """
        Check if the trade direction is aligned with VWAP.

        - LONG trades: only allowed when price is AT or ABOVE daily VWAP
        - SHORT trades: only allowed when price is AT or BELOW daily VWAP

        This filter prevents "long under VWAP" and "short above VWAP" trades
        which historically perform worse.

        NOTE: This filter is skipped for VWAP breach trades (mean reversion plays)
        which have their own entry logic and are allowed regardless of VWAP position.

        Args:
            proposed_signal: LONG or SHORT
            entry_price: The price we're considering entering at
            daily_vwap: Daily VWAP value

        Returns:
            (is_allowed, details_dict) where is_allowed is True if trade direction matches VWAP
        """
        if not self.config.use_vwap_direction_filter:
            return True, {'filter_disabled': True}

        if proposed_signal == Signal.HOLD:
            return True, {}

        if daily_vwap is None:
            # If no VWAP available, allow the trade
            return True, {'no_vwap': True}

        slippage = self.config.vwap_direction_slippage
        details = {
            'entry_price': entry_price,
            'daily_vwap': daily_vwap,
            'slippage': slippage,
        }

        if proposed_signal == Signal.LONG:
            # LONG only allowed when price is at or above VWAP
            is_allowed = entry_price >= (daily_vwap - slippage)
            details['price_vs_vwap'] = 'above' if is_allowed else 'below'
            details['reason'] = 'long_above_vwap' if is_allowed else 'long_under_vwap_rejected'
        else:  # SHORT
            # SHORT only allowed when price is at or below VWAP
            is_allowed = entry_price <= (daily_vwap + slippage)
            details['price_vs_vwap'] = 'below' if is_allowed else 'above'
            details['reason'] = 'short_below_vwap' if is_allowed else 'short_above_vwap_rejected'

        return is_allowed, details
