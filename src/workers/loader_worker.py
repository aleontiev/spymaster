"""
LoaderWorker for real-time data loading into training cache.

Responsible for:
- Listening for "minute_synced" events from Syncer
- Loading source files (stocks, options, gex_flow)
- Computing derived features (VWAP, SMA distances, etc.)
- Updating raw and normalized cache files
- Emitting "cache_updated" event for StrategyRunner

This worker bridges the gap between raw synced data and training-ready cache.
"""
import asyncio
import logging
from datetime import datetime, date, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import pytz

from src.workers.event_bus import EventBus, Event
from src.data.loader import (
    CACHE_VERSION,
    GEX_FLOW_FEATURES,
    NORMALIZED_CACHE_DIR,
    RAW_CACHE_DIR,
    DAILY_CACHE_DIR,
    get_market_hours_utc,
    normalize_features,
    load_gex_flow_for_date,
    compute_options_features_vectorized,
)

logger = logging.getLogger(__name__)

EASTERN = pytz.timezone("America/New_York")
UTC = pytz.UTC

# Data directories
PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
STOCKS_DIR = DATA_DIR / "stocks"
OPTIONS_DIR = DATA_DIR / "options"
GEX_FLOW_DIR = DATA_DIR / "gex_flow"


class LoaderWorker:
    """
    Worker that loads source data files into training cache.

    Triggered when Syncer emits "minute_synced" event.
    Produces raw and normalized cache files.

    The loader incrementally updates cache files by appending new minute data,
    rather than recomputing the entire day on each update.
    """

    def __init__(
        self,
        underlying: str = "SPY",
        event_bus: Optional[EventBus] = None,
        stocks_dir: Optional[Path] = None,
        options_dir: Optional[Path] = None,
        gex_flow_dir: Optional[Path] = None,
        cache_dir: Optional[Path] = None,
        raw_cache_dir: Optional[Path] = None,
    ) -> None:
        """
        Initialize the LoaderWorker.

        Args:
            underlying: Stock ticker (default: SPY)
            event_bus: Event bus for receiving and emitting events
            stocks_dir: Directory for stocks data
            options_dir: Directory for options data
            gex_flow_dir: Directory for GEX flow data
            cache_dir: Directory for normalized cache
            raw_cache_dir: Directory for raw cache
        """
        self.underlying = underlying
        self.event_bus = event_bus

        # Data directories
        self.stocks_dir = stocks_dir or STOCKS_DIR
        self.options_dir = options_dir or OPTIONS_DIR
        self.gex_flow_dir = gex_flow_dir or GEX_FLOW_DIR
        self.cache_dir = cache_dir or NORMALIZED_CACHE_DIR
        self.raw_cache_dir = raw_cache_dir or RAW_CACHE_DIR

        # Ensure directories exist
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.raw_cache_dir.mkdir(parents=True, exist_ok=True)

        # State
        self._running = False
        self._current_date: Optional[date] = None
        self._daily_df: Optional[pd.DataFrame] = None

        # In-memory cache for today's data (for feature computation)
        self._today_close_history: List[float] = []
        self._today_volume_history: List[float] = []
        self._today_timestamps: List[datetime] = []

    def _get_stocks_path(self, dt: date) -> Path:
        """Get path for stocks parquet file."""
        return self.stocks_dir / f"{self.underlying}_STOCKS-1M_{dt.isoformat()}.parquet"

    def _get_options_path(self, dt: date) -> Path:
        """Get path for options parquet file."""
        return self.options_dir / f"{self.underlying}_OPTIONS-1M_{dt.isoformat()}.parquet"

    def _get_gex_flow_path(self, dt: date) -> Path:
        """Get path for GEX flow parquet file."""
        return self.gex_flow_dir / f"{self.underlying}_thetadata_1m_combined_{dt.isoformat()}.parquet"

    def _get_raw_cache_path(self, dt: date) -> Path:
        """Get path for raw cache file."""
        return self.raw_cache_dir / f"{self.underlying}_raw_{dt.isoformat()}.parquet"

    def _get_normalized_cache_path(self, dt: date) -> Path:
        """Get path for normalized cache file."""
        return self.cache_dir / f"{self.underlying}_{CACHE_VERSION}_{dt.isoformat()}.parquet"

    def _reset_daily_state(self, target_date: date) -> None:
        """Reset state for a new trading day."""
        if self._current_date != target_date:
            logger.info(f"Resetting daily state for {target_date}")
            self._current_date = target_date
            self._today_close_history = []
            self._today_volume_history = []
            self._today_timestamps = []

            # Load daily aggregates for SMA features
            self._load_daily_aggregates()

    def _load_daily_aggregates(self) -> None:
        """Load daily aggregates for SMA distance computations."""
        daily_path = DAILY_CACHE_DIR / f"{self.underlying}-1d.parquet"
        if daily_path.exists():
            self._daily_df = pd.read_parquet(daily_path)
            logger.debug(f"Loaded daily aggregates: {len(self._daily_df)} days")
        else:
            self._daily_df = None
            logger.warning("Daily aggregates not found - SMA features will be zero")

    def _load_minute_data(self, target_minute: datetime, target_date: date) -> Optional[pd.DataFrame]:
        """
        Load source data for a single minute.

        Returns combined DataFrame with stocks, options features, and GEX flow.
        """
        stocks_path = self._get_stocks_path(target_date)
        if not stocks_path.exists():
            logger.warning(f"Stocks file not found: {stocks_path}")
            return None

        # Load stocks data
        stocks_df = pd.read_parquet(stocks_path)
        stocks_df["timestamp"] = pd.to_datetime(stocks_df["window_start"])

        # Normalize target_minute to pandas Timestamp for comparison
        target_ts = pd.Timestamp(target_minute)
        if target_ts.tz is not None:
            target_ts = target_ts.tz_localize(None)  # Make naive for comparison
        if stocks_df["timestamp"].dt.tz is not None:
            stocks_df["timestamp"] = stocks_df["timestamp"].dt.tz_localize(None)

        # Filter to the specific minute
        minute_mask = (
            stocks_df["timestamp"] >= target_ts
        ) & (
            stocks_df["timestamp"] < target_ts + timedelta(minutes=1)
        )
        minute_stocks = stocks_df[minute_mask].copy()

        if minute_stocks.empty:
            logger.debug(f"No stocks data for minute {target_minute}")
            return None

        minute_stocks = minute_stocks.set_index("timestamp")

        # Load options data
        options_path = self._get_options_path(target_date)
        options_feat_df = None

        if options_path.exists():
            options_df = pd.read_parquet(options_path)
            options_df["timestamp"] = pd.to_datetime(options_df["window_start"])
            if options_df["timestamp"].dt.tz is not None:
                options_df["timestamp"] = options_df["timestamp"].dt.tz_localize(None)

            # Filter to the specific minute
            minute_mask = (
                options_df["timestamp"] >= target_ts
            ) & (
                options_df["timestamp"] < target_ts + timedelta(minutes=1)
            )
            minute_options = options_df[minute_mask].copy()

            if not minute_options.empty:
                spy_prices = minute_stocks["close"]
                options_feat_df = compute_options_features_vectorized(minute_options, spy_prices)

        if options_feat_df is None:
            options_feat_df = pd.DataFrame(
                index=minute_stocks.index,
                columns=[
                    "atm_spread", "net_premium_flow", "implied_volatility",
                    "call_strikes_active", "put_strikes_active", "strike_breadth_ratio",
                    "atm_call_volume", "otm_call_volume", "atm_put_volume", "otm_put_volume"
                ]
            ).fillna(0)

        # Load GEX flow features
        gex_flow_path = self._get_gex_flow_path(target_date)
        gex_feat_df = None

        if gex_flow_path.exists():
            gex_df = pd.read_parquet(gex_flow_path)
            if "timestamp" in gex_df.columns:
                gex_df["timestamp"] = pd.to_datetime(gex_df["timestamp"])
                gex_df = gex_df.set_index("timestamp")

            # Handle timezone conversion if needed (ThetaData is Eastern)
            if gex_df.index.tz is None:
                try:
                    # ThetaData timestamps are Eastern - convert to UTC
                    gex_df.index = gex_df.index.tz_localize(EASTERN).tz_convert("UTC")
                    gex_df.index = gex_df.index.tz_localize(None)  # Remove tz for consistency
                except Exception:
                    pass

            # Make index timezone naive for comparison
            if gex_df.index.tz is not None:
                gex_df.index = gex_df.index.tz_localize(None)

            # Filter to the specific minute
            minute_mask = (
                gex_df.index >= target_ts
            ) & (
                gex_df.index < target_ts + timedelta(minutes=1)
            )
            minute_gex = gex_df[minute_mask]

            if not minute_gex.empty:
                gex_feat_df = minute_gex[
                    [col for col in GEX_FLOW_FEATURES if col in minute_gex.columns]
                ]

        if gex_feat_df is None:
            # Create default GEX features
            gex_feat_df = pd.DataFrame({
                col: 0.0 for col in GEX_FLOW_FEATURES
            }, index=minute_stocks.index)

        # Combine all data
        combined = minute_stocks[["open", "high", "low", "close", "volume"]].copy()

        # Update history for VWAP computation
        close_val = combined["close"].iloc[0]
        volume_val = combined["volume"].iloc[0]
        self._today_close_history.append(close_val)
        self._today_volume_history.append(volume_val)
        self._today_timestamps.append(target_minute)

        # Calculate VWAP (cumulative for the day)
        typical_price = (combined["high"] + combined["low"] + combined["close"]) / 3
        if len(self._today_volume_history) > 0:
            # Use numpy for cumulative VWAP calculation
            volumes = np.array(self._today_volume_history)
            typical_prices = np.array([
                (h + l + c) / 3
                for h, l, c in zip(
                    [combined["high"].iloc[0]],
                    [combined["low"].iloc[0]],
                    [combined["close"].iloc[0]]
                )
            ])
            # Simple: just use the current typical price weighted by volume
            all_typical = (
                np.array([h + l + c for h, l, c in zip(
                    self._today_close_history,  # Use close as proxy
                    self._today_close_history,
                    self._today_close_history
                )]) / 3
            )
            cumulative_tp_vol = np.sum(all_typical * volumes)
            cumulative_vol = np.sum(volumes)
            vwap = cumulative_tp_vol / cumulative_vol if cumulative_vol > 0 else close_val
        else:
            vwap = close_val

        combined["vwap"] = vwap

        # Calculate dist_vwap: distance from VWAP as log ratio
        combined["dist_vwap"] = np.log(combined["close"] / combined["vwap"]) * 100

        # Drop implied_volatility from options if GEX flow has it
        if gex_feat_df is not None and "implied_volatility" in gex_feat_df.columns:
            if "implied_volatility" in options_feat_df.columns:
                options_feat_df = options_feat_df.drop(columns=["implied_volatility"])

        combined = combined.join(options_feat_df, how="left")
        combined = combined.join(gex_feat_df, how="left")
        combined = combined.fillna(0)

        # Add time features
        market_open_utc, market_close_utc = get_market_hours_utc(target_date)
        from datetime import timezone as dt_tz
        market_close_unix = market_close_utc.replace(tzinfo=dt_tz.utc).timestamp()
        market_open_unix = market_open_utc.replace(tzinfo=dt_tz.utc).timestamp()
        MARKET_DURATION_SECS = 6.5 * 3600

        # Time to close
        timestamp_unix = target_minute.replace(tzinfo=dt_tz.utc).timestamp()
        bar_end_unix = timestamp_unix + 60
        time_to_close = max(0, market_close_unix - bar_end_unix)
        combined["time_to_close"] = time_to_close

        # Time progress
        elapsed = bar_end_unix - market_open_unix
        progress = elapsed / MARKET_DURATION_SECS
        combined["sin_time"] = np.sin(2 * np.pi * progress)
        combined["cos_time"] = np.cos(2 * np.pi * progress)

        # Day of week (from target_date)
        combined["day_of_week"] = target_date.weekday()

        # SMA distance features (requires daily aggregates)
        if self._daily_df is not None and not self._daily_df.empty:
            combined = self._add_sma_features(combined, target_date)
        else:
            # Default to zero for SMA features
            for col in ["dist_ma20", "dist_ma50", "dist_ma200", "vol_regime"]:
                combined[col] = 0.0

        return combined

    def _add_sma_features(self, df: pd.DataFrame, target_date: date) -> pd.DataFrame:
        """Add SMA distance features using daily aggregates."""
        try:
            daily = self._daily_df
            if daily.index.name != "date":
                if "date" in daily.columns:
                    daily = daily.set_index("date")

            # Get data up to previous day (no lookahead)
            prev_day = target_date - timedelta(days=1)
            daily_mask = daily.index <= pd.Timestamp(prev_day)
            daily_subset = daily[daily_mask]

            if daily_subset.empty or len(daily_subset) < 20:
                # Not enough history
                for col in ["dist_ma20", "dist_ma50", "dist_ma200", "vol_regime"]:
                    df[col] = 0.0
                return df

            # Get SMAs from daily aggregates (already shifted by 1 to prevent lookahead)
            latest = daily_subset.iloc[-1]

            current_price = df["close"].iloc[0]

            # Distance from SMAs (log difference * 100 for percentage)
            sma_20 = latest.get("sma_20", current_price)
            sma_50 = latest.get("sma_50", current_price)
            sma_200 = latest.get("sma_200", current_price)

            df["dist_ma20"] = np.log(current_price / sma_20) * 100 if sma_20 > 0 else 0.0
            df["dist_ma50"] = np.log(current_price / sma_50) * 100 if sma_50 > 0 else 0.0
            df["dist_ma200"] = np.log(current_price / sma_200) * 100 if sma_200 > 0 else 0.0

            # Volatility regime
            current_iv = df.get("implied_volatility", pd.Series([0.0])).iloc[0]
            iv_sma = latest.get("iv_sma_20", current_iv)
            df["vol_regime"] = current_iv - iv_sma if iv_sma > 0 else 0.0

        except Exception as e:
            logger.warning(f"Error computing SMA features: {e}")
            for col in ["dist_ma20", "dist_ma50", "dist_ma200", "vol_regime"]:
                df[col] = 0.0

        return df

    def _append_to_cache(
        self,
        cache_path: Path,
        new_data: pd.DataFrame,
    ) -> None:
        """Append new data to existing cache file or create new one."""
        if new_data.empty:
            return

        if cache_path.exists():
            existing = pd.read_parquet(cache_path)
            # Remove any existing rows with the same timestamp
            if existing.index.name is None and "timestamp" not in existing.columns:
                existing = existing.reset_index()
            if "timestamp" in existing.columns:
                existing = existing.set_index("timestamp")

            # Concatenate and drop duplicates
            combined = pd.concat([existing, new_data])
            combined = combined[~combined.index.duplicated(keep="last")]
            combined = combined.sort_index()
            combined.to_parquet(cache_path)
        else:
            new_data.to_parquet(cache_path)

    async def on_minute_synced(self, event: Event) -> None:
        """
        Handle minute_synced event.

        Loads source data, computes features, and updates cache files.
        """
        event_data = event.data
        target_minute = event_data.get("timestamp")
        target_date = event_data.get("date")

        if target_minute is None or target_date is None:
            logger.error("Invalid minute_synced event: missing timestamp or date")
            return

        # Reset state if new day
        self._reset_daily_state(target_date)

        logger.debug(f"Processing minute: {target_minute}")

        try:
            # Load source data for this minute
            raw_df = self._load_minute_data(target_minute, target_date)

            if raw_df is None or raw_df.empty:
                logger.warning(f"No data loaded for {target_minute}")
                return

            # Save raw data to cache
            raw_cache_path = self._get_raw_cache_path(target_date)
            self._append_to_cache(raw_cache_path, raw_df)

            # Normalize and save to normalized cache
            normalized_df = normalize_features(raw_df)
            normalized_cache_path = self._get_normalized_cache_path(target_date)
            self._append_to_cache(normalized_cache_path, normalized_df)

            logger.info(
                f"Cache updated: {target_minute.strftime('%H:%M')} - "
                f"raw={raw_cache_path.name}, norm={normalized_cache_path.name}"
            )

            # Emit cache_updated event
            if self.event_bus is not None:
                await self.event_bus.emit(
                    "cache_updated",
                    {
                        "timestamp": target_minute,
                        "date": target_date,
                        "raw_cache_path": str(raw_cache_path),
                        "normalized_cache_path": str(normalized_cache_path),
                        "underlying_price": raw_df["close"].iloc[-1] if "close" in raw_df.columns else None,
                    },
                )

        except Exception as e:
            logger.error(f"Error processing minute {target_minute}: {e}")

    async def run(self) -> None:
        """
        Main loop: listen for minute_synced events and process them.
        """
        if self.event_bus is None:
            logger.error("LoaderWorker requires an event bus")
            return

        logger.info("LoaderWorker started")
        self._running = True

        # Subscribe to events
        self.event_bus.subscribe("minute_synced", self.on_minute_synced)

        try:
            # Keep running until stopped
            while self._running:
                await asyncio.sleep(1)

        except asyncio.CancelledError:
            logger.info("LoaderWorker cancelled")
        finally:
            self.event_bus.unsubscribe("minute_synced", self.on_minute_synced)
            logger.info("LoaderWorker stopped")

    def stop(self) -> None:
        """Stop the loader worker."""
        self._running = False

    @property
    def is_running(self) -> bool:
        """Check if the loader is running."""
        return self._running


async def load_minute_for_strategy(
    target_minute: datetime,
    underlying: str = "SPY",
    cache_dir: Path = NORMALIZED_CACHE_DIR,
) -> Optional[pd.DataFrame]:
    """
    Load a specific minute's data from cache for strategy execution.

    This is a helper function for StrategyRunner to quickly get the latest
    cached data without reading the entire day's file.

    Args:
        target_minute: The minute to load
        underlying: Stock ticker
        cache_dir: Normalized cache directory

    Returns:
        DataFrame with normalized features for the minute, or None if not found
    """
    target_date = target_minute.date() if hasattr(target_minute, "date") else target_minute
    if isinstance(target_date, datetime):
        target_date = target_date.date()

    cache_path = cache_dir / f"{underlying}_{CACHE_VERSION}_{target_date.isoformat()}.parquet"

    if not cache_path.exists():
        return None

    try:
        df = pd.read_parquet(cache_path)
        if "timestamp" in df.columns:
            df = df.set_index("timestamp")

        # Normalize target_minute to pandas Timestamp for comparison
        target_ts = pd.Timestamp(target_minute)
        if target_ts.tz is not None:
            target_ts = target_ts.tz_localize(None)
        if df.index.tz is not None:
            df.index = df.index.tz_localize(None)

        # Filter to the specific minute
        minute_mask = (
            df.index >= target_ts
        ) & (
            df.index < target_ts + timedelta(minutes=1)
        )
        minute_df = df[minute_mask]

        return minute_df if not minute_df.empty else None

    except Exception as e:
        logger.error(f"Error loading minute from cache: {e}")
        return None
