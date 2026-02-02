"""
Live data buffer for maintaining rolling window of market data.

Maintains OHLCV bars with real-time VWAP calculation for live trading.
"""
from collections import deque
from dataclasses import dataclass
from datetime import date, datetime
from typing import Any, Deque, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch


@dataclass
class Bar:
    """OHLCV bar data."""
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: int
    vwap: Optional[float] = None  # From data provider
    trade_count: Optional[int] = None


class LiveDataBuffer:
    """
    Maintains a rolling window of OHLCV bars with real-time VWAP calculation.

    Features:
    - Rolling window of configurable size
    - Day-anchored VWAP with standard deviation bands
    - Context tensors for model inference
    - Automatic day reset
    """

    def __init__(self, max_bars: int = 390):
        """
        Initialize the data buffer.

        Args:
            max_bars: Maximum bars to retain (390 = full trading day)
        """
        self.max_bars = max_bars
        self._bars: Deque[Bar] = deque(maxlen=max_bars)

        # VWAP tracking (cumulative for day-anchored calculation)
        self._vwap_cumulative_tp_vol: float = 0.0
        self._vwap_cumulative_vol: float = 0.0
        self._vwap_cumulative_tp_sq_vol: float = 0.0

        # Current day tracking
        self._current_date: Optional[date] = None

        # Opening Range tracking
        self._or_high: Optional[float] = None
        self._or_low: Optional[float] = None
        self._or_calculated: bool = False

    def add_bar(self, bar: Dict) -> None:
        """
        Add a new bar to the buffer.

        Args:
            bar: Bar data dict with keys:
                - timestamp: datetime
                - open, high, low, close: prices
                - volume: int
                - trade_count: optional int
                - vwap: optional float (from provider)
        """
        timestamp = bar.get("timestamp")
        if isinstance(timestamp, str):
            timestamp = pd.Timestamp(timestamp)

        bar_date = timestamp.date() if hasattr(timestamp, "date") else pd.Timestamp(timestamp).date()

        # Check for new day
        if self._current_date != bar_date:
            self.reset_day()
            self._current_date = bar_date

        # Create Bar object
        new_bar = Bar(
            timestamp=timestamp,
            open=bar["open"],
            high=bar["high"],
            low=bar["low"],
            close=bar["close"],
            volume=bar.get("volume", 0),
            vwap=bar.get("vwap"),
            trade_count=bar.get("trade_count"),
        )

        self._bars.append(new_bar)

        # Update VWAP calculations
        self._update_vwap(new_bar)

        # Calculate Opening Range after 15 minutes (bars 0-14)
        if not self._or_calculated and len(self._bars) >= 15:
            self._calculate_opening_range()

    def _update_vwap(self, bar: Bar) -> None:
        """Update cumulative VWAP calculations."""
        typical_price = (bar.high + bar.low + bar.close) / 3
        self._vwap_cumulative_tp_vol += typical_price * bar.volume
        self._vwap_cumulative_vol += bar.volume
        self._vwap_cumulative_tp_sq_vol += (typical_price ** 2) * bar.volume

    def _calculate_opening_range(self) -> None:
        """Calculate the Opening Range from first 15 bars."""
        if len(self._bars) < 15:
            return

        first_15 = list(self._bars)[:15]
        self._or_high = max(b.high for b in first_15)
        self._or_low = min(b.low for b in first_15)
        self._or_calculated = True

    def reset_day(self) -> None:
        """Reset buffer for a new trading day."""
        self._bars.clear()
        self._vwap_cumulative_tp_vol = 0.0
        self._vwap_cumulative_vol = 0.0
        self._vwap_cumulative_tp_sq_vol = 0.0
        self._or_high = None
        self._or_low = None
        self._or_calculated = False
        self._current_date = None

    def get_current_vwap(self) -> Optional[float]:
        """Get current VWAP value."""
        if self._vwap_cumulative_vol <= 0:
            return None
        return self._vwap_cumulative_tp_vol / self._vwap_cumulative_vol

    def get_vwap_bands(self) -> Tuple[Optional[float], Optional[float], Optional[float], Optional[float], Optional[float]]:
        """
        Get current VWAP with standard deviation bands.

        Returns:
            Tuple of (vwap, upper_1sd, lower_1sd, upper_2sd, lower_2sd)
            All values are None if insufficient data.
        """
        if self._vwap_cumulative_vol <= 0:
            return None, None, None, None, None

        vwap = self._vwap_cumulative_tp_vol / self._vwap_cumulative_vol

        # Calculate standard deviation
        # Variance = E[X^2] - E[X]^2 (volume-weighted)
        mean_sq = self._vwap_cumulative_tp_sq_vol / self._vwap_cumulative_vol
        variance = mean_sq - (vwap ** 2)
        std_dev = np.sqrt(max(0, variance))

        upper_1sd = vwap + std_dev
        lower_1sd = vwap - std_dev
        upper_2sd = vwap + 2 * std_dev
        lower_2sd = vwap - 2 * std_dev

        return vwap, upper_1sd, lower_1sd, upper_2sd, lower_2sd

    def get_opening_range(self) -> Tuple[Optional[float], Optional[float]]:
        """
        Get the Opening Range high and low.

        Returns:
            Tuple of (or_high, or_low) or (None, None) if not yet calculated.
        """
        return self._or_high, self._or_low

    def get_ohlcv_df(self) -> pd.DataFrame:
        """
        Get bars as a pandas DataFrame.

        Returns:
            DataFrame with columns: open, high, low, close, volume
            Index is the timestamp.
        """
        if not self._bars:
            return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])

        data = [
            {
                "open": b.open,
                "high": b.high,
                "low": b.low,
                "close": b.close,
                "volume": b.volume,
            }
            for b in self._bars
        ]

        df = pd.DataFrame(data)
        df.index = pd.DatetimeIndex([b.timestamp for b in self._bars])

        return df

    def get_latest_bar(self) -> Optional[Bar]:
        """Get the most recent bar."""
        if not self._bars:
            return None
        return self._bars[-1]

    def get_latest_price(self) -> Optional[float]:
        """Get the most recent close price."""
        if not self._bars:
            return None
        return self._bars[-1].close

    def get_context_for_model(
        self,
        context_length: int,
        feature_columns: Optional[List[str]] = None,
        device: Optional[torch.device] = None,
    ) -> Optional[torch.Tensor]:
        """
        Get context tensor for model inference.

        Args:
            context_length: Number of bars required
            feature_columns: Columns to include (default: all OHLCV)
            device: Torch device for tensor

        Returns:
            Tensor of shape (1, context_length, num_features) or None if insufficient data.
        """
        if len(self._bars) < context_length:
            return None

        if feature_columns is None:
            feature_columns = ["open", "high", "low", "close", "volume"]

        # Get last context_length bars
        bars = list(self._bars)[-context_length:]

        # Build feature matrix
        features = []
        for bar in bars:
            row = []
            for col in feature_columns:
                if col == "open":
                    row.append(bar.open)
                elif col == "high":
                    row.append(bar.high)
                elif col == "low":
                    row.append(bar.low)
                elif col == "close":
                    row.append(bar.close)
                elif col == "volume":
                    row.append(bar.volume)
            features.append(row)

        tensor = torch.tensor(features, dtype=torch.float32)

        if device is not None:
            tensor = tensor.to(device)

        # Add batch dimension
        return tensor.unsqueeze(0)

    def get_average_volume(self, lookback: int = 20) -> float:
        """
        Get average volume over lookback period.

        Args:
            lookback: Number of bars to average

        Returns:
            Average volume (0 if no data)
        """
        if not self._bars:
            return 0.0

        bars = list(self._bars)[-lookback:]
        if not bars:
            return 0.0

        return sum(b.volume for b in bars) / len(bars)

    def get_average_body(self, lookback: int = 20) -> float:
        """
        Get average candle body size over lookback period.

        Args:
            lookback: Number of bars to average

        Returns:
            Average body size (0 if no data)
        """
        if not self._bars:
            return 0.0

        bars = list(self._bars)[-lookback:]
        if not bars:
            return 0.0

        bodies = [abs(b.close - b.open) for b in bars]
        return sum(bodies) / len(bodies)

    def get_all_candles_for_chart(self) -> List[Dict]:
        """
        Return all buffered bars formatted for Lightweight Charts.

        Returns:
            List of dicts with {time, open, high, low, close} where time
            is Unix seconds in ET (business-day local time for Lightweight Charts).
        """
        if not self._bars:
            return []

        candles = []
        for bar in self._bars:
            ts = bar.timestamp
            if hasattr(ts, 'timestamp'):
                # Convert to Unix seconds. Lightweight Charts expects
                # business-day time in local (ET) seconds.
                unix_sec = int(ts.timestamp())
            else:
                unix_sec = int(pd.Timestamp(ts).timestamp())
            candles.append({
                "time": unix_sec,
                "open": bar.open,
                "high": bar.high,
                "low": bar.low,
                "close": bar.close,
            })
        return candles

    def __len__(self) -> int:
        """Return number of bars in buffer."""
        return len(self._bars)

    def get_full_chart_data(self) -> Dict[str, Any]:
        """Return all buffered data formatted for TrainingChartView."""
        if not self._bars:
            return {}
        bar_dicts = [
            {
                "timestamp": b.timestamp,
                "open": b.open,
                "high": b.high,
                "low": b.low,
                "close": b.close,
                "volume": b.volume,
            }
            for b in self._bars
        ]
        return compute_chart_data(bar_dicts)

    @property
    def is_empty(self) -> bool:
        """Check if buffer is empty."""
        return len(self._bars) == 0

    @property
    def has_opening_range(self) -> bool:
        """Check if opening range has been calculated."""
        return self._or_calculated


def compute_chart_data(bars: List[Dict]) -> Dict[str, Any]:
    """
    Compute TrainingChartView-compatible chart data from a list of bar dicts.

    Each bar dict must have: timestamp, open, high, low, close, volume.
    Returns dict with keys: ohlcv, volume, vwap, vwap_upper_1, vwap_lower_1,
    vwap_upper_2, vwap_lower_2, or_high, or_low.
    """
    if not bars:
        return {}

    ohlcv = []
    volume = []
    vwap_series = []
    vwap_u1 = []
    vwap_l1 = []
    vwap_u2 = []
    vwap_l2 = []
    or_high_series = []
    or_low_series = []

    # Cumulative VWAP state (resets per day)
    cum_tp_vol = 0.0
    cum_vol = 0.0
    cum_tp_sq_vol = 0.0
    current_date = None
    or_h: Optional[float] = None
    or_l: Optional[float] = None
    bar_idx_today = 0

    for bar in bars:
        ts = bar["timestamp"]
        if hasattr(ts, "timestamp"):
            unix_sec = int(ts.timestamp())
        else:
            unix_sec = int(pd.Timestamp(ts).timestamp())

        bar_date = ts.date() if hasattr(ts, "date") else pd.Timestamp(ts).date()

        # Day boundary reset
        if bar_date != current_date:
            cum_tp_vol = 0.0
            cum_vol = 0.0
            cum_tp_sq_vol = 0.0
            current_date = bar_date
            or_h = None
            or_l = None
            bar_idx_today = 0

        bar_idx_today += 1

        o, h, l, c = bar["open"], bar["high"], bar["low"], bar["close"]
        vol = bar.get("volume", 0)

        # OHLCV
        ohlcv.append({"time": unix_sec, "open": o, "high": h, "low": l, "close": c})

        # Volume with color
        vol_color = (
            "rgba(38, 166, 154, 0.5)" if c >= o else "rgba(239, 83, 80, 0.5)"
        )
        volume.append({"time": unix_sec, "value": vol, "color": vol_color})

        # VWAP calculation
        tp = (h + l + c) / 3
        cum_tp_vol += tp * vol
        cum_vol += vol
        cum_tp_sq_vol += (tp**2) * vol

        if cum_vol > 0:
            vwap = cum_tp_vol / cum_vol
            var = (cum_tp_sq_vol / cum_vol) - vwap**2
            std = max(0, var) ** 0.5

            vwap_series.append({"time": unix_sec, "value": round(vwap, 4)})
            vwap_u1.append({"time": unix_sec, "value": round(vwap + std, 4)})
            vwap_l1.append({"time": unix_sec, "value": round(vwap - std, 4)})
            vwap_u2.append({"time": unix_sec, "value": round(vwap + 2 * std, 4)})
            vwap_l2.append({"time": unix_sec, "value": round(vwap - 2 * std, 4)})

        # Opening Range (first 15 bars of each day)
        if bar_idx_today <= 15:
            if or_h is None or h > or_h:
                or_h = h
            if or_l is None or l < or_l:
                or_l = l

        if bar_idx_today >= 15 and or_h is not None:
            or_high_series.append({"time": unix_sec, "value": or_h})
            or_low_series.append({"time": unix_sec, "value": or_l})

    result: Dict[str, Any] = {"ohlcv": ohlcv, "volume": volume}

    if vwap_series:
        result["vwap"] = vwap_series
        result["vwap_upper_1"] = vwap_u1
        result["vwap_lower_1"] = vwap_l1
        result["vwap_upper_2"] = vwap_u2
        result["vwap_lower_2"] = vwap_l2

    if or_high_series:
        result["or_high"] = or_high_series
        result["or_low"] = or_low_series

    return result
