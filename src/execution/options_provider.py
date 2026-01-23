"""
Options data provider for backtesting with real options prices.

Loads real options prices from parquet files and provides price lookups.
"""
import re
from datetime import date, datetime
from pathlib import Path
from typing import Optional, Tuple

import pandas as pd

from src.backtest.types import PositionType


class RealOptionsProvider:
    """Load real options prices from options-1m dataset. No simulation fallback."""

    def __init__(self, options_dir: str = "data/options-1m/SPY"):
        self.options_dir = Path(options_dir)
        self._current_date: Optional[date] = None
        self._day_data: Optional[pd.DataFrame] = None
        self._price_cache: dict = {}

    def _parse_ticker(self, ticker: str) -> Tuple[Optional[date], Optional[str], Optional[float]]:
        """Parse options ticker to extract expiration, type, strike."""
        match = re.match(r'O:SPY(\d{6})([CP])(\d{8})', ticker)
        if not match:
            return None, None, None

        exp_str = match.group(1)
        opt_type = match.group(2)
        strike_cents = int(match.group(3))

        year = 2000 + int(exp_str[:2])
        month = int(exp_str[2:4])
        day = int(exp_str[4:6])
        try:
            exp_date = date(year, month, day)
        except ValueError:
            return None, None, None

        strike = strike_cents / 1000.0
        return exp_date, opt_type, strike

    def load_date(self, trading_date: date) -> bool:
        """Load options data for a specific trading date."""
        if self._current_date == trading_date:
            return self._day_data is not None

        self._current_date = trading_date
        self._day_data = None
        self._price_cache = {}

        month_str = trading_date.strftime("%Y-%m")
        day_str = f"{trading_date.day:02d}"
        file_path = self.options_dir / month_str / f"{day_str}.parquet"

        if not file_path.exists():
            return False

        try:
            df = pd.read_parquet(file_path)
            parsed = df['ticker'].apply(self._parse_ticker)
            df['expiration'] = parsed.apply(lambda x: x[0])
            df['option_type'] = parsed.apply(lambda x: x[1])
            df['strike'] = parsed.apply(lambda x: x[2])
            df = df[df['expiration'] == trading_date].copy()

            if len(df) == 0:
                return False

            df['window_start'] = pd.to_datetime(df['window_start'])
            df['minute'] = df['window_start'].dt.floor('min')
            self._day_data = df
            return True

        except Exception as e:
            print(f"Error loading options data for {trading_date}: {e}")
            return False

    def get_option_price(
        self,
        timestamp: datetime,
        strike: float,
        position_type: PositionType,
        underlying_price: float,
        price_type: str = "close",  # "close", "open", "low", "high"
    ) -> Optional[float]:
        """Get option price from real data only. Returns None if no data available."""
        if hasattr(timestamp, 'tz_localize'):
            ts_minute = timestamp.replace(second=0, microsecond=0)
        else:
            ts_minute = pd.Timestamp(timestamp).floor('min')

        opt_type = 'C' if position_type == PositionType.CALL else 'P'
        cache_key = (ts_minute, strike, opt_type, price_type)

        if cache_key in self._price_cache:
            return self._price_cache[cache_key]

        price = self._lookup_price(ts_minute, strike, opt_type, underlying_price, price_type)
        self._price_cache[cache_key] = price
        return price

    def _lookup_price(
        self,
        ts_minute: pd.Timestamp,
        strike: float,
        opt_type: str,
        underlying_price: float,
        price_type: str = "close",
    ) -> Optional[float]:
        """Look up price from real data only. Returns None if no data available."""
        if self._day_data is None:
            return None

        df = self._day_data
        mask = (df['strike'] == strike) & (df['option_type'] == opt_type)
        subset = df[mask]

        if len(subset) == 0:
            mask = (abs(df['strike'] - strike) <= 1) & (df['option_type'] == opt_type)
            subset = df[mask]
            if len(subset) == 0:
                return None

        ts_naive = ts_minute.tz_localize(None) if ts_minute.tzinfo else ts_minute
        subset = subset.copy()
        subset['time_diff'] = abs((subset['minute'].dt.tz_localize(None) - ts_naive).dt.total_seconds())
        close_rows = subset[subset['time_diff'] <= 300]

        if len(close_rows) == 0:
            if len(subset) > 0:
                return subset[price_type].median()
            return None

        closest = close_rows.loc[close_rows['time_diff'].idxmin()]
        return float(closest[price_type])
