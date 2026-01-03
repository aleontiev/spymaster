"""
Syncer worker for real-time data synchronization.

Responsible for:
- Real-time trade-quote fetching from ThetaData (batch per minute)
- 1-minute SPY OHLCV from Polygon/Massive
- 1-minute options aggregates from Polygon/Massive
- 1-minute Greeks from ThetaData
- Daily OI snapshots
- Computing GEX flow features in real-time

Uses REST polling since StrategyRunner only runs once per minute.
"""
import asyncio
import logging
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime, date, timedelta, time as dt_time
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import pytz

from src.data.massive_client import MassiveClient
from src.data.thetadata_client import ThetaDataClient, ThetaDataError
from src.data.gex_flow_engine import GEXFlowEngine, LeeReadyClassifier, GEXFlowFeatures
from src.workers.event_bus import EventBus

logger = logging.getLogger(__name__)

EASTERN = pytz.timezone("America/New_York")
UTC = pytz.UTC

# Market hours (Eastern)
MARKET_OPEN = dt_time(9, 30)
MARKET_CLOSE = dt_time(16, 0)

# Data directories
PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
STOCKS_DIR = DATA_DIR / "stocks"
OPTIONS_DIR = DATA_DIR / "options"
OI_DIR = DATA_DIR / "oi"
GEX_FLOW_DIR = DATA_DIR / "gex_flow"


@dataclass
class SyncResult:
    """Result of a minute sync operation."""

    timestamp: datetime
    stock_data: Optional[Dict[str, Any]] = None
    options_count: int = 0
    gex_flow: Optional[GEXFlowFeatures] = None
    success: bool = True
    error: Optional[str] = None


class Syncer:
    """
    Worker responsible for syncing raw market data.

    Uses REST polling (not WebSocket) since StrategyRunner only runs once per minute.
    Batch-fetches trade-quotes for the completed minute, computes GEX flow features.

    Data sources:
    - ThetaData REST: Trade-quotes (batch per minute), 1m Greeks
    - Polygon/Massive REST: 1m SPY OHLCV, 1m options aggs
    """

    def __init__(
        self,
        underlying: str = "SPY",
        delay_minutes: int = 0,
        strike_range: float = 20.0,
        thetadata_config: Optional[Dict] = None,
        polygon_config: Optional[Dict] = None,
        event_bus: Optional[EventBus] = None,
    ) -> None:
        """
        Initialize the Syncer.

        Args:
            underlying: Stock ticker to sync (default: SPY)
            delay_minutes: Data delay in minutes (0 for Pro plan)
            strike_range: Strike range from ATM for options
            thetadata_config: ThetaData configuration
            polygon_config: Polygon configuration
            event_bus: Event bus for emitting sync events
        """
        self.underlying = underlying
        self.delay_minutes = delay_minutes
        self.strike_range = strike_range
        self.event_bus = event_bus

        # Initialize clients
        polygon_delay = polygon_config.get("delay_minutes", 0) if polygon_config else delay_minutes
        self.massive_client = MassiveClient(delay_minutes=polygon_delay)
        self.stock_client = MassiveClient(delay_minutes=polygon_delay)
        self.options_client = MassiveClient(delay_minutes=polygon_delay)

        self.thetadata_client = ThetaDataClient(
            base_url=thetadata_config.get("terminal_url") if thetadata_config else None,
            timeout=thetadata_config.get("timeout", 60) if thetadata_config else 60,
        )

        # GEX flow computation
        self.gex_flow_engine = GEXFlowEngine()
        self.lee_ready = LeeReadyClassifier()

        # State
        self.last_synced_minute: Optional[datetime] = None
        self.last_reference_price: Optional[float] = None
        self._running = False

        # Thread pool for sync operations
        self._executor = ThreadPoolExecutor(max_workers=4)

        self._current_date: Optional[date] = None

        # Ensure directories exist
        for dir_path in [STOCKS_DIR, OPTIONS_DIR, OI_DIR, GEX_FLOW_DIR]:
            dir_path.mkdir(parents=True, exist_ok=True)

    def is_market_hours(self, dt: Optional[datetime] = None) -> bool:
        """Check if given time is during market hours."""
        if dt is None:
            dt = datetime.now(EASTERN)
        elif dt.tzinfo is None:
            dt = EASTERN.localize(dt)
        else:
            dt = dt.astimezone(EASTERN)

        if dt.weekday() > 4:
            return False

        current_time = dt.time()
        return MARKET_OPEN <= current_time <= MARKET_CLOSE

    def get_target_minute(self) -> datetime:
        """Get the target minute to sync (current time - delay)."""
        now = datetime.now(UTC)
        target = now - timedelta(minutes=self.delay_minutes)
        return target.replace(second=0, microsecond=0)

    def _get_stocks_path(self, dt: date) -> Path:
        """Get path for stocks parquet file."""
        return STOCKS_DIR / f"{self.underlying}_STOCKS-1M_{dt.isoformat()}.parquet"

    def _get_options_path(self, dt: date) -> Path:
        """Get path for options parquet file."""
        return OPTIONS_DIR / f"{self.underlying}_OPTIONS-1M_{dt.isoformat()}.parquet"

    def _get_oi_path(self, dt: date) -> Path:
        """Get path for OI parquet file."""
        return OI_DIR / f"{self.underlying}_OI-0DTE_{dt.isoformat()}.parquet"

    def _get_gex_flow_path(self, dt: date) -> Path:
        """Get path for GEX flow parquet file."""
        return GEX_FLOW_DIR / f"{self.underlying}_thetadata_1m_combined_{dt.isoformat()}.parquet"

    def _sync_minute_stock(self, target_minute: datetime) -> Optional[Dict[str, Any]]:
        """Sync a single minute of stock data (runs in thread pool)."""
        try:
            agg = self.stock_client.fetch_minute_agg(self.underlying, target_minute)
            if not agg:
                return None

            return {
                "ticker": self.underlying,
                "volume": int(agg.get("volume", 0)),
                "open": float(agg["open"]),
                "close": float(agg["close"]),
                "high": float(agg["high"]),
                "low": float(agg["low"]),
                "window_start": agg["timestamp"],
                "transactions": int(agg.get("transactions", 0) or 0),
            }
        except Exception as e:
            logger.error(f"Error syncing stock data: {e}")
            return None

    async def _sync_minute_options_thetadata(
        self,
        target_minute: datetime,
        reference_price: float,
    ) -> pd.DataFrame:
        """
        Sync options minute aggregates from ThetaData (async).

        Uses ThetaData v3 API to fetch 1-minute OHLCV data for 0DTE options,
        returning data in Polygon-compatible format.

        Args:
            target_minute: The minute to fetch data for
            reference_price: Current underlying price for strike filtering

        Returns:
            DataFrame with Polygon-compatible columns:
            - ticker: OCC option symbol (e.g., "O:SPY241213C00500000")
            - volume: Volume traded
            - open: Opening price
            - close: Closing price
            - high: High price
            - low: Low price
            - window_start: Bar timestamp
            - transactions: Number of trades
        """
        try:
            eastern_time = target_minute.astimezone(EASTERN)
            expiration_date = eastern_time.date()

            df = await self.thetadata_client.fetch_option_ohlc_for_minute(
                query_date=expiration_date,
                target_minute=eastern_time,
                root=self.underlying,
                expiration=expiration_date,  # 0DTE
                strike_range=self.strike_range,
                reference_price=reference_price,
                zero_dte=True,
            )

            if df.empty:
                logger.debug(f"No options data from ThetaData for {eastern_time}")
                return pd.DataFrame()

            logger.debug(f"Fetched {len(df)} options bars from ThetaData")
            return df

        except ThetaDataError as e:
            logger.warning(f"ThetaData options fetch failed: {e}")
            return pd.DataFrame()
        except Exception as e:
            logger.error(f"Error syncing options data from ThetaData: {e}")
            return pd.DataFrame()

    async def _fetch_minute_trade_quotes(
        self,
        target_minute: datetime,
    ) -> pd.DataFrame:
        """
        Batch-fetch trade-quotes for a single minute from ThetaData.

        Uses start_time/end_time params to get exactly one minute of data.
        """
        eastern_time = target_minute.astimezone(EASTERN)
        query_date = eastern_time.date()

        start_time = eastern_time.strftime("%H:%M:%S")
        end_minute = eastern_time + timedelta(minutes=1)
        end_time = end_minute.strftime("%H:%M:%S")

        try:
            df = await self.thetadata_client.fetch_trade_quotes(
                query_date=query_date,
                root=self.underlying,
                expiration=query_date,  # 0DTE
                start_time=start_time,
                end_time=end_time,
            )
            return df
        except ThetaDataError as e:
            logger.warning(f"Failed to fetch trade-quotes for {target_minute}: {e}")
            return pd.DataFrame()

    async def _fetch_minute_greeks(
        self,
        target_minute: datetime,
    ) -> pd.DataFrame:
        """Fetch 1-minute Greeks from ThetaData."""
        eastern_time = target_minute.astimezone(EASTERN)
        query_date = eastern_time.date()

        try:
            df = await self.thetadata_client.fetch_greeks_for_date(
                query_date=query_date,
                root=self.underlying,
                expiration=query_date,
                interval="1m",
            )

            if df.empty:
                return df

            # Filter to the specific minute
            if "timestamp" in df.columns:
                df["timestamp"] = pd.to_datetime(df["timestamp"])
                minute_start = eastern_time.replace(second=0, microsecond=0)
                minute_end = minute_start + timedelta(minutes=1)
                df = df[
                    (df["timestamp"] >= minute_start) & (df["timestamp"] < minute_end)
                ]

            return df
        except ThetaDataError as e:
            logger.warning(f"Failed to fetch Greeks for {target_minute}: {e}")
            return pd.DataFrame()

    async def _compute_gex_flow_for_minute(
        self,
        trade_quotes_df: pd.DataFrame,
        greeks_df: pd.DataFrame,
        underlying_price: float,
        timestamp: datetime,
    ) -> Optional[GEXFlowFeatures]:
        """
        Compute GEX flow features for a single minute.

        1. Classify trades with Lee-Ready
        2. Aggregate buy/sell volume per contract
        3. Compute all 11 GEX flow features
        """
        try:
            if trade_quotes_df.empty or greeks_df.empty:
                # Return empty features if no data
                return GEXFlowFeatures(
                    timestamp=timestamp,
                    underlying_price=underlying_price,
                    net_gex=0.0,
                    net_dex=0.0,
                    gamma_flow=0.0,
                    delta_flow=0.0,
                    zero_gex_price=underlying_price,
                    zero_dex_price=underlying_price,
                    positive_gws=underlying_price,
                    negative_gws=underlying_price,
                    negative_dws=underlying_price,
                    positive_dws=underlying_price,
                    gamma_call_wall=underlying_price,
                    gamma_put_wall=underlying_price,
                    dex_call_wall=underlying_price,
                    dex_put_wall=underlying_price,
                    market_velocity=0.0,
                    anchored_vwap_z=0.0,
                    vwap_divergence=0.0,
                    implied_volatility=0.0,
                    call_buy_volume=0.0,
                    call_sell_volume=0.0,
                    put_buy_volume=0.0,
                    put_sell_volume=0.0,
                )

            # Classify trades
            classified_trades = self.lee_ready.classify_trades(trade_quotes_df)

            # Use the GEX flow engine to compute features
            features = self.gex_flow_engine.compute_single_minute(
                greeks_df=greeks_df,
                trades_df=classified_trades,
                underlying_price=underlying_price,
                timestamp=timestamp,
            )

            return features

        except Exception as e:
            logger.error(f"Error computing GEX flow: {e}")
            return None

    def _append_to_parquet(
        self,
        path: Path,
        new_data: pd.DataFrame,
        dedup_cols: Optional[List[str]] = None,
    ) -> None:
        """Append new data to existing parquet file or create new one."""
        if new_data.empty:
            return

        if path.exists():
            existing = pd.read_parquet(path)
            combined = pd.concat([existing, new_data], ignore_index=True)
            if dedup_cols:
                combined = combined.drop_duplicates(subset=dedup_cols, keep="last")
            combined.to_parquet(path, index=False)
        else:
            new_data.to_parquet(path, index=False)

    async def _load_daily_oi_for_gex(self, target_date: date) -> Optional[pd.DataFrame]:
        """
        Load daily OI for GEX baseline calculation.

        First checks for cached OI file, then fetches from ThetaData if needed.

        Args:
            target_date: Date to load OI for

        Returns:
            DataFrame with OI data (strike, right, open_interest) or None
        """
        oi_path = self._get_oi_path(target_date)

        # Try to load from cached file first
        if oi_path.exists():
            try:
                oi_df = pd.read_parquet(oi_path)
                logger.info(f"Loaded cached OI for {target_date}: {len(oi_df)} contracts")
                return oi_df
            except Exception as e:
                logger.warning(f"Failed to load cached OI: {e}")

        # Fetch from ThetaData
        try:
            oi_df = await self.thetadata_client.fetch_open_interest_for_date(
                query_date=target_date,
                root=self.underlying,
                zero_dte=True,
            )

            if not oi_df.empty:
                # Cache for future use
                oi_df.to_parquet(oi_path, index=False)
                logger.info(f"Fetched and cached OI for {target_date}: {len(oi_df)} contracts")
                return oi_df
            else:
                logger.warning(f"No OI data available from ThetaData for {target_date}")
                return None

        except ThetaDataError as e:
            logger.warning(f"Failed to fetch OI from ThetaData for {target_date}: {e}")
            return None

    async def sync_minute_data(self, target_minute: datetime) -> SyncResult:
        """
        Sync all data for a completed minute.

        1. Fetch SPY 1m OHLCV from Polygon/Massive
        2. Fetch Options 1m aggs from ThetaData (migrated from Polygon)
        3. Fetch Trade-quotes for the minute from ThetaData
        4. Fetch 1m Greeks from ThetaData
        5. Classify trades with Lee-Ready
        6. Compute GEX flow features for this minute
        7. Save to source files

        Returns:
            SyncResult with sync status and data
        """
        eastern_time = target_minute.astimezone(EASTERN)
        target_date = eastern_time.date()

        # Reset state at start of new day
        if self._current_date != target_date:
            self._current_date = target_date

            # Load daily OI for GEX baseline (use cached file or fetch)
            daily_oi_df = await self._load_daily_oi_for_gex(target_date)
            self.gex_flow_engine.reset_daily_state(daily_oi_df)

        logger.info(f"Syncing minute: {eastern_time.strftime('%Y-%m-%d %H:%M')} ET")

        loop = asyncio.get_event_loop()

        try:
            # Fetch stock data (thread pool - sync Polygon/Massive client)
            reference_price = self.last_reference_price or 500.0

            stock_future = loop.run_in_executor(
                self._executor,
                self._sync_minute_stock,
                target_minute,
            )

            # Fetch all ThetaData in parallel (async)
            # Options OHLC now comes from ThetaData (migrated from Polygon)
            options_future = self._sync_minute_options_thetadata(
                target_minute,
                reference_price,
            )
            trade_quotes_future = self._fetch_minute_trade_quotes(target_minute)
            greeks_future = self._fetch_minute_greeks(target_minute)

            # Wait for all
            stock_data, options_df, trade_quotes_df, greeks_df = await asyncio.gather(
                stock_future,
                options_future,
                trade_quotes_future,
                greeks_future,
            )

            if not stock_data:
                return SyncResult(
                    timestamp=target_minute,
                    success=False,
                    error="No stock data available",
                )

            # Update reference price
            self.last_reference_price = stock_data["close"]

            # Save stock data
            stock_df = pd.DataFrame([stock_data])
            stock_path = self._get_stocks_path(target_date)
            self._append_to_parquet(
                stock_path,
                stock_df,
                dedup_cols=["ticker", "window_start"],
            )

            # Save options data (now from ThetaData, in Polygon-compatible format)
            options_count = 0
            if not options_df.empty:
                options_path = self._get_options_path(target_date)
                self._append_to_parquet(
                    options_path,
                    options_df,
                    dedup_cols=["ticker", "window_start"],
                )
                options_count = len(options_df)

            # Compute GEX flow features
            gex_flow = await self._compute_gex_flow_for_minute(
                trade_quotes_df=trade_quotes_df,
                greeks_df=greeks_df,
                underlying_price=stock_data["close"],
                timestamp=target_minute,
            )

            # Save GEX flow features
            if gex_flow is not None:
                gex_df = pd.DataFrame([gex_flow.to_dict()])
                gex_path = self._get_gex_flow_path(target_date)
                self._append_to_parquet(
                    gex_path,
                    gex_df,
                    dedup_cols=["timestamp"],
                )

            logger.info(
                f"Sync complete: stock ${stock_data['close']:.2f}, "
                f"{options_count} options, GEX flow computed"
            )

            self.last_synced_minute = target_minute

            return SyncResult(
                timestamp=target_minute,
                stock_data=stock_data,
                options_count=options_count,
                gex_flow=gex_flow,
                success=True,
            )

        except Exception as e:
            logger.error(f"Error in sync_minute_data: {e}")
            return SyncResult(
                timestamp=target_minute,
                success=False,
                error=str(e),
            )

    async def sync_daily_oi(self, target_date: date) -> bool:
        """Sync daily OI snapshot from ThetaData."""
        oi_path = self._get_oi_path(target_date)
        if oi_path.exists():
            logger.debug(f"OI file already exists for {target_date}")
            return True

        try:
            df = await self.thetadata_client.fetch_open_interest_for_date(
                query_date=target_date,
                root=self.underlying,
                zero_dte=True,
            )

            if df.empty:
                logger.warning(f"No OI data from ThetaData for {target_date}")
                return False

            df.to_parquet(oi_path, index=False)
            logger.info(f"Synced OI from ThetaData: {len(df)} contracts for {target_date}")
            return True

        except ThetaDataError as e:
            logger.warning(f"Failed to fetch OI from ThetaData: {e}")
            return False

    async def run(self) -> None:
        """
        Main sync loop.

        1. Wait for minute boundary
        2. Sync all data for completed minute
        3. Emit "minute_synced" event
        4. Repeat

        Also syncs OI once per day after market open.
        """
        logger.info(f"Syncer started (delay: {self.delay_minutes}min)")
        self._running = True

        last_oi_sync_date: Optional[date] = None

        while self._running:
            try:
                target_minute = self.get_target_minute()
                eastern_time = target_minute.astimezone(EASTERN)
                target_date = eastern_time.date()

                # Skip if already synced this minute
                if self.last_synced_minute == target_minute:
                    await asyncio.sleep(1)
                    continue

                # Skip if outside market hours
                if not self.is_market_hours(target_minute):
                    await asyncio.sleep(30)
                    continue

                # Wait for minute boundary
                now = datetime.now(UTC)
                current_second = now.second

                if current_second >= 50:
                    wait_time = 60 - current_second + 2
                    await asyncio.sleep(wait_time)
                elif current_second < 2:
                    await asyncio.sleep(2 - current_second)

                # Sync the minute
                result = await self.sync_minute_data(target_minute)

                # Emit event
                if self.event_bus and result.success:
                    await self.event_bus.emit(
                        "minute_synced",
                        {
                            "timestamp": target_minute,
                            "date": target_date,
                            "underlying_price": result.stock_data.get("close")
                            if result.stock_data
                            else None,
                        },
                    )

                # Sync OI once per day
                if last_oi_sync_date != target_date:
                    market_open_time = datetime.combine(
                        target_date,
                        MARKET_OPEN,
                        tzinfo=EASTERN,
                    ) + timedelta(minutes=5)

                    if eastern_time >= market_open_time:
                        success = await self.sync_daily_oi(target_date)
                        if success:
                            last_oi_sync_date = target_date

            except asyncio.CancelledError:
                logger.info("Syncer cancelled")
                break
            except Exception as e:
                logger.error(f"Error in sync loop: {e}")
                await asyncio.sleep(5)

        logger.info("Syncer stopped")

    def stop(self) -> None:
        """Stop the sync loop."""
        self._running = False

    @property
    def is_running(self) -> bool:
        """Check if the syncer is running."""
        return self._running
