"""
Background worker for real-time data synchronization from Massive (Polygon.io).

This worker syncs minute-level market data with a configurable delay (15 minutes
for Standard plan) and maintains local parquet files and cache.
"""
import asyncio
import logging
import os
import signal
import sys
from datetime import datetime, date, timedelta, time as dt_time
from pathlib import Path
from typing import Optional, Dict, Any, List

import pandas as pd
import pytz
from dotenv import load_dotenv

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.massive_client import MassiveClient
from src.data.thetadata_client import ThetaDataClient, ThetaDataError
from src.data.dag.loader import RAW_CACHE_DIR, NORMALIZED_CACHE_DIR
from src.data.dag.realtime import load_combined_day, normalize_features
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

EASTERN = pytz.timezone("America/New_York")
UTC = pytz.UTC

# Market hours (Eastern)
MARKET_OPEN = dt_time(9, 30)
MARKET_CLOSE = dt_time(16, 0)

# Data directories
DATA_DIR = PROJECT_ROOT / "data"
STOCKS_DIR = DATA_DIR / "stocks"
OPTIONS_DIR = DATA_DIR / "options"
OI_DIR = DATA_DIR / "oi"


class DataSyncWorker:
    """
    Background worker that syncs market data from Massive API.

    Features:
    - Minute-level stock data sync
    - 0DTE options minute aggregates sync
    - Daily OI snapshot sync (from Polygon or ThetaData)
    - Raw and normalized cache updates
    """

    def __init__(
        self,
        underlying: str = "SPY",
        delay_minutes: int = 15,
        strike_range: float = 20.0,
        use_thetadata_oi: bool = False,
    ) -> None:
        """
        Initialize the data sync worker.

        Args:
            underlying: Stock ticker to sync (default: SPY)
            delay_minutes: Data delay in minutes (15 for Standard plan)
            strike_range: Strike range from ATM for options (+/- this value)
            use_thetadata_oi: If True, fetch OI from ThetaData instead of Polygon
        """
        self.underlying = underlying
        self.delay_minutes = delay_minutes
        self.strike_range = strike_range
        self.use_thetadata_oi = use_thetadata_oi

        # Use separate clients for parallel requests (avoids connection pool contention)
        self.client = MassiveClient(delay_minutes=delay_minutes)
        self.stock_client = MassiveClient(delay_minutes=delay_minutes)
        self.options_client = MassiveClient(delay_minutes=delay_minutes)

        # ThetaData client for OI (initialized lazily)
        self._thetadata_client: Optional[ThetaDataClient] = None
        self._thetadata_available: Optional[bool] = None

        # Track last synced minute to avoid duplicates
        self.last_synced_minute: Optional[datetime] = None

        # Cache last known price for parallel options fetching
        self.last_reference_price: Optional[float] = None

        # Running state
        self._running = False

        # Ensure directories exist
        STOCKS_DIR.mkdir(parents=True, exist_ok=True)
        OPTIONS_DIR.mkdir(parents=True, exist_ok=True)
        OI_DIR.mkdir(parents=True, exist_ok=True)
        RAW_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        NORMALIZED_CACHE_DIR.mkdir(parents=True, exist_ok=True)

    def is_market_hours(self, dt: Optional[datetime] = None) -> bool:
        """Check if given time is during market hours."""
        if dt is None:
            dt = datetime.now(EASTERN)
        elif dt.tzinfo is None:
            dt = EASTERN.localize(dt)
        else:
            dt = dt.astimezone(EASTERN)

        # Check if weekday (Mon=0, Fri=4)
        if dt.weekday() > 4:
            return False

        current_time = dt.time()
        return MARKET_OPEN <= current_time <= MARKET_CLOSE

    def get_target_minute(self) -> datetime:
        """Get the target minute to sync (current time - delay)."""
        now = datetime.now(UTC)
        target = now - timedelta(minutes=self.delay_minutes)
        # Truncate to minute boundary
        return target.replace(second=0, microsecond=0)

    def _get_stocks_path(self, dt: date) -> Path:
        """Get path for stocks parquet file for a date."""
        return STOCKS_DIR / f"{self.underlying}_STOCKS-1M_{dt.isoformat()}.parquet"

    def _get_options_path(self, dt: date) -> Path:
        """Get path for options parquet file for a date."""
        return OPTIONS_DIR / f"{self.underlying}_OPTIONS-1M_{dt.isoformat()}.parquet"

    def _get_oi_path(self, dt: date) -> Path:
        """Get path for OI parquet file for a date."""
        return OI_DIR / f"{self.underlying}_OI-0DTE_{dt.isoformat()}.parquet"

    def sync_minute_stock(self, target_minute: datetime) -> Optional[Dict[str, Any]]:
        """
        Sync a single minute of stock data.

        Args:
            target_minute: The minute to sync (UTC)

        Returns:
            Dict with OHLCV data or None if not available
        """
        agg = self.stock_client.fetch_minute_agg(self.underlying, target_minute)
        if not agg:
            return None

        # Convert to expected schema
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

    def sync_minute_options(
        self,
        target_minute: datetime,
        reference_price: float,
    ) -> List[Dict[str, Any]]:
        """
        Sync options minute aggregates for 0DTE contracts.

        Args:
            target_minute: The minute to sync (UTC)
            reference_price: ATM reference price

        Returns:
            List of dicts with options OHLCV data
        """
        eastern_time = target_minute.astimezone(EASTERN)
        expiration_date = eastern_time.date()

        # Fetch options chain to get contract tickers
        chain = self.options_client.fetch_options_chain_snapshot(
            underlying=self.underlying,
            expiration_date=expiration_date,
            strike_range=self.strike_range,
            reference_price=reference_price,
        )

        if chain.empty:
            return []

        # For each contract, fetch the minute aggregate
        results = []
        for ticker in chain["ticker"].dropna().unique():
            try:
                minute_end = target_minute + timedelta(minutes=1)
                from_ms = int(target_minute.timestamp() * 1000)
                to_ms = int(minute_end.timestamp() * 1000)

                aggs = list(self.options_client.client.list_aggs(
                    ticker=ticker,
                    multiplier=1,
                    timespan="minute",
                    from_=from_ms,
                    to=to_ms,
                    limit=1,
                ))

                if aggs:
                    agg = aggs[0]
                    results.append({
                        "ticker": ticker,
                        "volume": int(agg.volume),
                        "open": float(agg.open),
                        "close": float(agg.close),
                        "high": float(agg.high),
                        "low": float(agg.low),
                        "window_start": pd.to_datetime(agg.timestamp, unit="ms"),
                        "transactions": int(getattr(agg, "transactions", 0) or 0),
                    })
            except Exception as e:
                logger.debug(f"Error fetching {ticker}: {e}")
                continue

        return results

    def sync_daily_oi(self, as_of_date: date) -> pd.DataFrame:
        """
        Sync daily OI snapshot for 0DTE contracts.

        Args:
            as_of_date: Date for OI data

        Returns:
            DataFrame with OI data
        """
        try:
            chain = self.client.fetch_options_chain_snapshot(
                underlying=self.underlying,
                expiration_date=as_of_date,
                strike_range=50.0,  # Wide range for OI
            )

            if chain.empty:
                return pd.DataFrame()

            # Transform to OI schema
            rows = []
            for _, row in chain.iterrows():
                # Parse ticker to extract details
                ticker = row.get("ticker", "")
                if not ticker:
                    continue

                rows.append({
                    "symbol": self.underlying,
                    "expiration": pd.to_datetime(as_of_date),
                    "strike": row.get("strike"),
                    "right": "C" if row.get("option_type") == "call" else "P",
                    "timestamp": pd.Timestamp.now(tz=UTC),
                    "open_interest": int(row.get("open_interest", 0) or 0),
                    "date": pd.to_datetime(as_of_date),
                })

            return pd.DataFrame(rows)

        except Exception as e:
            logger.error(f"Error fetching OI: {e}")
            return pd.DataFrame()

    async def _get_thetadata_client(self) -> Optional[ThetaDataClient]:
        """Get ThetaData client, checking if terminal is available."""
        if self._thetadata_available is False:
            return None

        if self._thetadata_client is None:
            self._thetadata_client = ThetaDataClient()

        # Check terminal availability on first call
        if self._thetadata_available is None:
            try:
                self._thetadata_available = await self._thetadata_client._check_terminal_running()
                if self._thetadata_available:
                    logger.info("ThetaData terminal is available for OI sync")
                else:
                    logger.warning("ThetaData terminal not running, falling back to Polygon for OI")
            except Exception as e:
                logger.warning(f"Could not connect to ThetaData terminal: {e}")
                self._thetadata_available = False

        return self._thetadata_client if self._thetadata_available else None

    async def sync_daily_oi_thetadata(self, as_of_date: date) -> pd.DataFrame:
        """
        Sync daily OI snapshot for 0DTE contracts from ThetaData.

        ThetaData provides more reliable historical OI data compared to Polygon's
        realtime snapshot. Requires ThetaData Terminal to be running locally.

        Args:
            as_of_date: Date for OI data

        Returns:
            DataFrame with OI data in standard schema
        """
        client = await self._get_thetadata_client()
        if client is None:
            logger.debug("ThetaData not available, using Polygon for OI")
            return self.sync_daily_oi(as_of_date)

        try:
            # Fetch 0DTE OI from ThetaData
            df = await client.fetch_open_interest_for_date(
                query_date=as_of_date,
                root=self.underlying,
                zero_dte=True,
            )

            if df.empty:
                logger.info(f"No 0DTE OI from ThetaData for {as_of_date}, trying Polygon")
                return self.sync_daily_oi(as_of_date)

            # Transform to our standard OI schema
            # ThetaData returns: symbol, expiration, strike, right, timestamp, open_interest, date
            result_df = df.rename(columns={
                "right": "right_raw",
            })

            # Normalize right column (ThetaData uses "CALL"/"PUT", we use "C"/"P")
            if "right_raw" in result_df.columns:
                result_df["right"] = result_df["right_raw"].map(
                    lambda x: "C" if str(x).upper() in ("CALL", "C") else "P"
                )
                result_df = result_df.drop(columns=["right_raw"])

            logger.info(f"Synced OI from ThetaData: {len(result_df)} contracts for {as_of_date}")
            return result_df

        except ThetaDataError as e:
            logger.warning(f"ThetaData OI fetch failed: {e}, falling back to Polygon")
            return self.sync_daily_oi(as_of_date)
        except Exception as e:
            logger.error(f"Error fetching ThetaData OI: {e}")
            return self.sync_daily_oi(as_of_date)

    async def sync_oi_for_date(self, target_date: date) -> bool:
        """
        Sync OI data for a date using ThetaData (if enabled) or Polygon.

        This is the main entry point for OI sync. Call this method to sync
        OI data for a specific date.

        Args:
            target_date: Date to sync OI for

        Returns:
            True if OI was synced successfully
        """
        oi_path = self._get_oi_path(target_date)
        if oi_path.exists():
            logger.debug(f"OI file already exists for {target_date}")
            return True

        try:
            if self.use_thetadata_oi:
                oi_data = await self.sync_daily_oi_thetadata(target_date)
            else:
                oi_data = self.sync_daily_oi(target_date)

            if oi_data.empty:
                logger.warning(f"No OI data available for {target_date}")
                return False

            self.append_to_parquet(
                oi_path,
                oi_data,
                dedup_cols=["symbol", "strike", "right"],
            )
            source = "ThetaData" if self.use_thetadata_oi else "Polygon"
            logger.info(f"Synced OI from {source}: {len(oi_data)} contracts for {target_date}")
            return True

        except Exception as e:
            logger.error(f"Error syncing OI for {target_date}: {e}")
            return False

    def append_to_parquet(
        self,
        path: Path,
        new_data: pd.DataFrame,
        dedup_cols: Optional[List[str]] = None,
    ) -> None:
        """
        Append new data to existing parquet file or create new one.

        Args:
            path: Path to parquet file
            new_data: DataFrame to append
            dedup_cols: Columns to use for deduplication
        """
        if new_data.empty:
            return

        if path.exists():
            existing = pd.read_parquet(path)
            combined = pd.concat([existing, new_data], ignore_index=True)

            # Deduplicate if columns specified
            if dedup_cols:
                combined = combined.drop_duplicates(
                    subset=dedup_cols,
                    keep="last",
                )

            combined.to_parquet(path, index=False)
        else:
            new_data.to_parquet(path, index=False)

    def _get_oi_path(self, target_date: date) -> Path:
        """Get path to OI file for a given date."""
        return OI_DIR / f"{self.underlying}_OI-0DTE_{target_date.isoformat()}.parquet"

    def update_cache(self, target_date: date) -> None:
        """
        Update raw and normalized cache for a date.

        Args:
            target_date: Date to update cache for
        """
        try:
            # Get file paths for this date
            stocks_file = self._get_stocks_path(target_date)
            options_file = self._get_options_path(target_date)
            oi_file = self._get_oi_path(target_date)

            if not stocks_file.exists():
                logger.debug(f"No stocks file for {target_date}, skipping cache update")
                return

            # Load combined day data (GEX features from pre-computed flow files)
            df = load_combined_day(
                stocks_file=stocks_file,
                options_file=options_file if options_file.exists() else None,
            )

            if df.empty:
                return

            # Ensure cache directories exist
            RAW_CACHE_DIR.mkdir(parents=True, exist_ok=True)
            NORMALIZED_CACHE_DIR.mkdir(parents=True, exist_ok=True)

            # Name the index so it shows as 'timestamp' in parquet browsers
            df.index.name = "timestamp"

            # Save raw cache (index=True to preserve timestamp)
            raw_path = RAW_CACHE_DIR / f"{self.underlying}_raw_{target_date.isoformat()}.parquet"
            df.to_parquet(raw_path, index=True)

            # Normalize and save (index=True to preserve timestamp)
            normalized = normalize_features(df)
            if not normalized.empty:
                normalized.index.name = "timestamp"
                norm_path = NORMALIZED_CACHE_DIR / f"{self.underlying}_v2_{target_date.isoformat()}.parquet"
                normalized.to_parquet(norm_path, index=True)

            logger.info(f"Updated cache for {target_date}")

        except Exception as e:
            logger.error(f"Error updating cache for {target_date}: {e}")

    def sync_minute(self, target_minute: datetime) -> bool:
        """
        Sync all data for a specific minute.

        Args:
            target_minute: The minute to sync (UTC)

        Returns:
            True if sync was successful
        """
        from concurrent.futures import ThreadPoolExecutor, as_completed

        eastern_time = target_minute.astimezone(EASTERN)
        target_date = eastern_time.date()

        logger.info(f"Syncing minute: {eastern_time.strftime('%Y-%m-%d %H:%M')} ET")

        # Use cached reference price for parallel fetching (None on first run)
        reference_price = self.last_reference_price

        # Fetch stock and options data in parallel
        def fetch_stock():
            data = self.sync_minute_stock(target_minute)
            if data:
                logger.info(f"Stock data received: ${data['close']:.2f}")
            return data

        def fetch_options():
            # Use cached price or let the API fetch it internally
            data = self.sync_minute_options(target_minute, reference_price)
            logger.info(f"Options data received: {len(data) if data else 0} contracts")
            return data

        with ThreadPoolExecutor(max_workers=2) as executor:
            stock_future = executor.submit(fetch_stock)
            options_future = executor.submit(fetch_options)

            # Wait for both API calls to complete
            stock_data = stock_future.result()
            options_data = options_future.result()

        if not stock_data:
            logger.warning("No stock data available")
            return False

        # Update cached reference price for next sync
        self.last_reference_price = stock_data["close"]

        # Save both to parquet
        stock_df = pd.DataFrame([stock_data])
        stock_path = self._get_stocks_path(target_date)
        self.append_to_parquet(
            stock_path,
            stock_df,
            dedup_cols=["ticker", "window_start"],
        )

        options_count = 0
        if options_data:
            options_df = pd.DataFrame(options_data)
            options_path = self._get_options_path(target_date)
            self.append_to_parquet(
                options_path,
                options_df,
                dedup_cols=["ticker", "window_start"],
            )
            options_count = len(options_data)

        logger.info(f"Sync complete: stock ${stock_data['close']:.2f}, {options_count} options")

        # 3. OI sync is handled separately via sync_oi_for_date (async method)
        # When not using ThetaData, sync OI via Polygon here synchronously
        oi_path = self._get_oi_path(target_date)
        if not oi_path.exists() and not self.use_thetadata_oi:
            # Sync OI after first 15 minutes of market
            market_open_delayed = datetime.combine(
                target_date,
                MARKET_OPEN,
                tzinfo=EASTERN,
            ) + timedelta(minutes=self.delay_minutes + 5)

            if eastern_time >= market_open_delayed:
                # Use Polygon OI (sync method)
                oi_data = self.sync_daily_oi(target_date)
                if not oi_data.empty:
                    self.append_to_parquet(
                        oi_path,
                        oi_data,
                        dedup_cols=["symbol", "strike", "right"],
                    )
                    logger.info(f"Synced OI from Polygon: {len(oi_data)} contracts")
        elif oi_path.exists():
            logger.debug(f"OI file already exists for {target_date}, skipping")

        # 4. Update cache
        self.update_cache(target_date)

        self.last_synced_minute = target_minute
        return True

    async def run_loop(self) -> None:
        """Main sync loop - runs every minute."""
        logger.info(f"Starting data sync worker (delay: {self.delay_minutes}min)")
        if self.use_thetadata_oi:
            logger.info("ThetaData OI sync enabled")
        self._running = True

        # Track last OI sync date to avoid repeated attempts
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
                    logger.debug("Outside market hours, waiting...")
                    await asyncio.sleep(30)
                    continue

                # Ensure we sync at the start of a minute (0X seconds, not :5X like :59)
                # Wait until seconds are in the 0-9 range for the NEXT minute
                now = datetime.now(UTC)
                current_second = now.second

                if current_second >= 50:
                    # If we're at :5X seconds, wait until :0X of next minute
                    wait_until_next_minute = 60 - current_second + 2  # +2 for buffer
                    logger.debug(f"Waiting {wait_until_next_minute}s for next minute boundary")
                    await asyncio.sleep(wait_until_next_minute)
                elif current_second < 2:
                    # Small buffer to ensure data is available
                    await asyncio.sleep(2 - current_second)

                # Sync the minute (stock + options data)
                self.sync_minute(target_minute)

                # If ThetaData OI is enabled, sync OI asynchronously once per day
                if self.use_thetadata_oi and last_oi_sync_date != target_date:
                    # Check if OI file already exists
                    oi_path = self._get_oi_path(target_date)
                    if not oi_path.exists():
                        # Sync OI after first 15 minutes of market
                        market_open_delayed = datetime.combine(
                            target_date,
                            MARKET_OPEN,
                            tzinfo=EASTERN,
                        ) + timedelta(minutes=self.delay_minutes + 5)

                        if eastern_time >= market_open_delayed:
                            logger.info(f"Attempting ThetaData OI sync for {target_date}")
                            success = await self.sync_oi_for_date(target_date)
                            if success:
                                last_oi_sync_date = target_date
                                # Regenerate cache with new OI data
                                self.update_cache(target_date)
                    else:
                        last_oi_sync_date = target_date

            except asyncio.CancelledError:
                logger.info("Sync loop cancelled")
                break
            except Exception as e:
                logger.error(f"Error in sync loop: {e}")
                await asyncio.sleep(5)

        logger.info("Data sync worker stopped")

    def stop(self) -> None:
        """Stop the sync loop."""
        self._running = False

    # =========================================================================
    # Backfill Methods
    # =========================================================================

    def backfill_day_stocks(self, target_date: date, force: bool = False) -> bool:
        """
        Backfill stock minute data for a full trading day.

        Uses the Polygon aggregates endpoint to fetch all minute bars at once.

        Args:
            target_date: Date to backfill
            force: If True, overwrite existing data

        Returns:
            True if successful
        """
        stocks_path = self._get_stocks_path(target_date)

        if stocks_path.exists() and not force:
            logger.debug(f"Stocks file already exists for {target_date}, skipping")
            return True

        try:
            # Fetch full day minute aggregates
            from_ts = datetime.combine(target_date, MARKET_OPEN, tzinfo=EASTERN)
            to_ts = datetime.combine(target_date, MARKET_CLOSE, tzinfo=EASTERN)

            # Convert to milliseconds
            from_ms = int(from_ts.timestamp() * 1000)
            to_ms = int(to_ts.timestamp() * 1000)

            aggs = list(self.stock_client.client.list_aggs(
                ticker=self.underlying,
                multiplier=1,
                timespan="minute",
                from_=from_ms,
                to=to_ms,
                limit=50000,
            ))

            if not aggs:
                logger.warning(f"No stock data available for {target_date}")
                return False

            # Convert to DataFrame
            rows = []
            for agg in aggs:
                rows.append({
                    "ticker": self.underlying,
                    "volume": int(agg.volume),
                    "open": float(agg.open),
                    "close": float(agg.close),
                    "high": float(agg.high),
                    "low": float(agg.low),
                    "window_start": pd.to_datetime(agg.timestamp, unit="ms"),
                    "transactions": int(getattr(agg, "transactions", 0) or 0),
                })

            df = pd.DataFrame(rows)
            df.to_parquet(stocks_path, index=False)
            logger.info(f"Backfilled stocks for {target_date}: {len(df)} bars")
            return True

        except Exception as e:
            logger.error(f"Error backfilling stocks for {target_date}: {e}")
            return False

    def backfill_day_options(
        self,
        target_date: date,
        force: bool = False,
        reference_price: Optional[float] = None,
    ) -> bool:
        """
        Backfill options minute data for 0DTE contracts.

        This fetches minute aggregates for all 0DTE option contracts within
        the strike range.

        Args:
            target_date: Date to backfill
            force: If True, overwrite existing data
            reference_price: ATM reference price (fetched if not provided)

        Returns:
            True if successful
        """
        options_path = self._get_options_path(target_date)

        if options_path.exists() and not force:
            logger.debug(f"Options file already exists for {target_date}, skipping")
            return True

        try:
            # Get reference price from stock data or fetch it
            if reference_price is None:
                stocks_path = self._get_stocks_path(target_date)
                if stocks_path.exists():
                    stocks_df = pd.read_parquet(stocks_path)
                    if not stocks_df.empty:
                        reference_price = float(stocks_df["close"].iloc[-1])

            if reference_price is None:
                # Fetch from API
                ref_agg = self.stock_client.fetch_minute_agg(
                    self.underlying,
                    datetime.combine(target_date, MARKET_OPEN, tzinfo=EASTERN),
                )
                if ref_agg:
                    reference_price = ref_agg["close"]
                else:
                    logger.warning(f"Could not get reference price for {target_date}")
                    return False

            # Get options chain for this day
            chain = self.options_client.fetch_options_chain_snapshot(
                underlying=self.underlying,
                expiration_date=target_date,
                strike_range=self.strike_range,
                reference_price=reference_price,
            )

            if chain.empty:
                logger.warning(f"No options chain for {target_date}")
                return False

            # Fetch minute data for each contract
            from_ts = datetime.combine(target_date, MARKET_OPEN, tzinfo=EASTERN)
            to_ts = datetime.combine(target_date, MARKET_CLOSE, tzinfo=EASTERN)
            from_ms = int(from_ts.timestamp() * 1000)
            to_ms = int(to_ts.timestamp() * 1000)

            all_rows = []
            contract_tickers = chain["ticker"].dropna().unique()
            logger.info(f"Backfilling {len(contract_tickers)} options contracts for {target_date}")

            for ticker in contract_tickers:
                try:
                    aggs = list(self.options_client.client.list_aggs(
                        ticker=ticker,
                        multiplier=1,
                        timespan="minute",
                        from_=from_ms,
                        to=to_ms,
                        limit=50000,
                    ))

                    for agg in aggs:
                        all_rows.append({
                            "ticker": ticker,
                            "volume": int(agg.volume),
                            "open": float(agg.open),
                            "close": float(agg.close),
                            "high": float(agg.high),
                            "low": float(agg.low),
                            "window_start": pd.to_datetime(agg.timestamp, unit="ms"),
                            "transactions": int(getattr(agg, "transactions", 0) or 0),
                        })
                except Exception as e:
                    logger.debug(f"Error fetching {ticker}: {e}")
                    continue

            if not all_rows:
                logger.warning(f"No options data available for {target_date}")
                return False

            df = pd.DataFrame(all_rows)
            df.to_parquet(options_path, index=False)
            logger.info(f"Backfilled options for {target_date}: {len(df)} bars across {len(contract_tickers)} contracts")
            return True

        except Exception as e:
            logger.error(f"Error backfilling options for {target_date}: {e}")
            return False

    async def backfill_date_range(
        self,
        start_date: date,
        end_date: date,
        mode: str = "all",
        force: bool = False,
        update_cache: bool = True,
    ) -> Dict[str, int]:
        """
        Backfill data for a date range.

        Args:
            start_date: Start date (inclusive)
            end_date: End date (inclusive)
            mode: What to backfill - "stocks", "options", "oi", "cache", or "all"
            force: If True, overwrite existing data
            update_cache: If True, update raw/normalized cache after backfill

        Returns:
            Dict with counts of successful backfills by type
        """
        results = {"stocks": 0, "options": 0, "oi": 0, "cache": 0, "skipped": 0}

        current_date = start_date
        total_days = (end_date - start_date).days + 1

        logger.info(f"Starting backfill from {start_date} to {end_date} ({total_days} days)")
        logger.info(f"Mode: {mode}, Force: {force}")

        while current_date <= end_date:
            # Skip weekends
            if current_date.weekday() > 4:
                current_date += timedelta(days=1)
                continue

            logger.info(f"Processing {current_date}...")

            try:
                # Stocks
                if mode in ("stocks", "all"):
                    if self.backfill_day_stocks(current_date, force=force):
                        results["stocks"] += 1

                # Options
                if mode in ("options", "all"):
                    if self.backfill_day_options(current_date, force=force):
                        results["options"] += 1

                # OI
                if mode in ("oi", "all"):
                    if await self.sync_oi_for_date(current_date):
                        results["oi"] += 1

                # Cache update
                if update_cache and mode in ("cache", "all"):
                    self.update_cache(current_date)
                    results["cache"] += 1

            except Exception as e:
                logger.error(f"Error processing {current_date}: {e}")
                results["skipped"] += 1

            current_date += timedelta(days=1)

        logger.info(f"Backfill complete: {results}")
        return results

    def get_missing_dates(
        self,
        start_date: date,
        end_date: date,
        data_type: str = "stocks",
    ) -> List[date]:
        """
        Find dates that are missing data files.

        Args:
            start_date: Start date
            end_date: End date
            data_type: Type of data to check - "stocks", "options", "oi", or "cache"

        Returns:
            List of dates missing the specified data
        """
        missing = []
        current_date = start_date

        while current_date <= end_date:
            # Skip weekends
            if current_date.weekday() > 4:
                current_date += timedelta(days=1)
                continue

            # Check if file exists
            if data_type == "stocks":
                path = self._get_stocks_path(current_date)
            elif data_type == "options":
                path = self._get_options_path(current_date)
            elif data_type == "oi":
                path = self._get_oi_path(current_date)
            elif data_type == "cache":
                path = RAW_CACHE_DIR / f"{self.underlying}_raw_{current_date.isoformat()}.parquet"
            else:
                raise ValueError(f"Unknown data type: {data_type}")

            if not path.exists():
                missing.append(current_date)

            current_date += timedelta(days=1)

        return missing


async def main() -> None:
    """Run the data sync worker."""
    import argparse

    parser = argparse.ArgumentParser(description="Data sync worker")
    parser.add_argument(
        "--delay",
        type=int,
        default=15,
        help="Data delay in minutes (default: 15 for Standard plan)",
    )
    parser.add_argument(
        "--underlying",
        type=str,
        default="SPY",
        help="Underlying ticker (default: SPY)",
    )
    parser.add_argument(
        "--strike-range",
        type=float,
        default=20.0,
        help="Strike range from ATM (default: 20)",
    )
    parser.add_argument(
        "--backfill-minutes",
        type=int,
        default=0,
        help="Backfill N minutes of data before starting live sync",
    )
    parser.add_argument(
        "--use-thetadata-oi",
        action="store_true",
        help="Use ThetaData for OI sync (requires ThetaData Terminal running)",
    )

    # Backfill arguments
    parser.add_argument(
        "--backfill-start",
        type=str,
        default=None,
        help="Start date for backfill (YYYY-MM-DD). Enables backfill mode.",
    )
    parser.add_argument(
        "--backfill-end",
        type=str,
        default=None,
        help="End date for backfill (YYYY-MM-DD). Defaults to yesterday.",
    )
    parser.add_argument(
        "--backfill-mode",
        type=str,
        default="all",
        choices=["stocks", "options", "oi", "cache", "all"],
        help="What to backfill: stocks, options, oi, cache, or all (default: all)",
    )
    parser.add_argument(
        "--backfill-force",
        action="store_true",
        help="Force overwrite existing data during backfill",
    )
    parser.add_argument(
        "--list-missing",
        type=str,
        default=None,
        choices=["stocks", "options", "oi", "cache"],
        help="List dates missing the specified data type and exit",
    )

    args = parser.parse_args()

    worker = DataSyncWorker(
        underlying=args.underlying,
        delay_minutes=args.delay,
        strike_range=args.strike_range,
        use_thetadata_oi=args.use_thetadata_oi,
    )

    # Set up signal handlers
    loop = asyncio.get_event_loop()

    def signal_handler():
        logger.info("Received shutdown signal")
        worker.stop()

    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, signal_handler)

    # List missing dates if requested
    if args.list_missing:
        start = date.fromisoformat(args.backfill_start) if args.backfill_start else date(2020, 1, 1)
        end = date.fromisoformat(args.backfill_end) if args.backfill_end else date.today() - timedelta(days=1)
        missing = worker.get_missing_dates(start, end, args.list_missing)
        logger.info(f"Missing {args.list_missing} data for {len(missing)} dates:")
        for d in missing[:50]:  # Show first 50
            print(d.isoformat())
        if len(missing) > 50:
            print(f"... and {len(missing) - 50} more")
        return

    # Historical date range backfill mode
    if args.backfill_start:
        start_date = date.fromisoformat(args.backfill_start)
        end_date = date.fromisoformat(args.backfill_end) if args.backfill_end else date.today() - timedelta(days=1)

        logger.info(f"Running backfill from {start_date} to {end_date}")
        logger.info(f"Mode: {args.backfill_mode}, Force: {args.backfill_force}")

        results = await worker.backfill_date_range(
            start_date=start_date,
            end_date=end_date,
            mode=args.backfill_mode,
            force=args.backfill_force,
        )

        logger.info(f"Backfill complete: {results}")
        return

    # Legacy minute-based backfill
    if args.backfill_minutes > 0:
        logger.info(f"Backfilling {args.backfill_minutes} minutes...")
        now = datetime.now(UTC)
        for i in range(args.backfill_minutes, 0, -1):
            target = (now - timedelta(minutes=i + args.delay)).replace(
                second=0, microsecond=0
            )
            if worker.is_market_hours(target):
                worker.sync_minute(target)

    # Run the main loop
    await worker.run_loop()


if __name__ == "__main__":
    asyncio.run(main())
