"""
Simulation service for replaying historical 1m data through the trading engine.

Loads parquet data from data/stocks-1m/{underlying}/{YYYY-MM}/{DD}.parquet,
replays bars at configurable speed, and provides the same snapshot API as
TradingService so the frontend is mode-agnostic.
"""
import logging
import threading
import time as time_module
from datetime import date, datetime, time, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional
from zoneinfo import ZoneInfo

import pandas as pd

from src.execution.data_buffer import LiveDataBuffer, compute_chart_data

logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).parent.parent.parent
ET = ZoneInfo("America/New_York")

# Market hours in ET
MARKET_OPEN = time(9, 30)
MARKET_CLOSE = time(16, 0)


class SimulationService:
    """
    Historical simulation engine that replays 1m bars through the trading engine.

    Provides the same snapshot/trades/candles API shape as TradingService so the
    frontend can operate identically in both modes.
    """

    _instance: Optional["SimulationService"] = None

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._thread: Optional[threading.Thread] = None
        self._running: bool = False
        self._paused: bool = False
        self._speed: float = 30.0
        self._error: Optional[str] = None
        self._started_at: Optional[datetime] = None
        self._config: Dict[str, Any] = {}

        # Playback state
        self._bars: List[Dict] = []
        self._current_bar_index: int = 0
        self._seek_target: Optional[int] = None

        # Data buffer for VWAP / OR calculations
        self._data_buffer: Optional[LiveDataBuffer] = None

        # Candle tracking
        self._candles: List[Dict] = []

        # Trade tracking (mock trades)
        self._trades: List[Dict] = []
        self._daily_pnl: float = 0.0
        self._capital: float = 25000.0

        # Signal state (mock)
        self._last_signal_action: str = "HOLD"
        self._last_signal_source: str = "none"
        self._last_signal_confidence: float = 0.0

    @classmethod
    def instance(cls) -> "SimulationService":
        """Get or create the singleton instance."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    @property
    def is_running(self) -> bool:
        return self._running

    @property
    def error(self) -> Optional[str]:
        return self._error

    def _load_bars(
        self, underlying: str, start_date: date, end_date: date
    ) -> List[Dict]:
        """
        Load 1m bars from parquet files for the given date range.

        Reads data/stocks-1m/{underlying}/{YYYY-MM}/{DD}.parquet, filters to
        market hours (9:30-16:00 ET), and concatenates into a single bar list.
        """
        data_dir = BASE_DIR / "data" / "stocks-1m" / underlying
        if not data_dir.exists():
            raise FileNotFoundError(f"No data directory: {data_dir}")

        all_bars = []
        current = start_date
        while current <= end_date:
            month_dir = data_dir / current.strftime("%Y-%m")
            file_path = month_dir / f"{current.strftime('%d')}.parquet"

            if file_path.exists():
                df = pd.read_parquet(file_path)

                # Convert timestamp to ET
                if df["timestamp"].dt.tz is not None:
                    df["timestamp"] = df["timestamp"].dt.tz_convert(ET)
                else:
                    df["timestamp"] = df["timestamp"].dt.tz_localize("UTC").dt.tz_convert(ET)

                # Filter to market hours
                mask = (
                    (df["timestamp"].dt.time >= MARKET_OPEN)
                    & (df["timestamp"].dt.time < MARKET_CLOSE)
                )
                df = df[mask].sort_values("timestamp").reset_index(drop=True)

                for _, row in df.iterrows():
                    all_bars.append({
                        "timestamp": row["timestamp"],
                        "open": float(row["open"]),
                        "high": float(row["high"]),
                        "low": float(row["low"]),
                        "close": float(row["close"]),
                        "volume": int(row["volume"]),
                        "trade_count": int(row.get("transactions", 0)),
                    })

            current += timedelta(days=1)

        if not all_bars:
            raise ValueError(
                f"No bars found for {underlying} between {start_date} and {end_date}"
            )

        logger.info(f"Loaded {len(all_bars)} bars for {underlying} ({start_date} to {end_date})")
        return all_bars

    def start(
        self,
        underlying: str = "SPY",
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        capital: float = 25000.0,
        speed: float = 30.0,
        use_heuristic: bool = True,
        dry_run: bool = True,
    ) -> None:
        """Start simulation playback."""
        with self._lock:
            if self._running:
                raise RuntimeError("Simulation is already running")

            self._error = None
            self._config = {
                "underlying": underlying,
                "start_date": start_date,
                "end_date": end_date,
                "capital": capital,
                "speed": speed,
                "use_heuristic": use_heuristic,
                "dry_run": dry_run,
                "mode": "sim",
            }

            try:
                sd = date.fromisoformat(start_date) if start_date else date(2025, 1, 2)
                ed = date.fromisoformat(end_date) if end_date else sd
                self._bars = self._load_bars(underlying, sd, ed)
            except Exception as e:
                self._error = f"Failed to load data: {e}"
                logger.error(self._error, exc_info=True)
                raise RuntimeError(self._error) from e

            self._speed = speed
            self._capital = capital
            self._paused = False
            self._current_bar_index = 0
            self._seek_target = None
            self._candles = []
            self._trades = []
            self._daily_pnl = 0.0
            self._data_buffer = LiveDataBuffer()

            self._last_signal_action = "HOLD"
            self._last_signal_source = "none"
            self._last_signal_confidence = 0.0

            # Set running before starting thread so the loop condition works
            self._running = True
            self._started_at = datetime.now(ET)

            self._thread = threading.Thread(
                target=self._playback_loop,
                name="SimulationService",
                daemon=True,
            )
            self._thread.start()

            logger.info(
                f"Simulation started: {underlying} {sd} to {ed}, "
                f"{len(self._bars)} bars, speed={speed}x"
            )

    def _playback_loop(self) -> None:
        """Background thread that iterates through bars at configured speed."""
        try:
            while self._current_bar_index < len(self._bars) and self._running:
                # Handle seek
                if self._seek_target is not None:
                    target = self._seek_target
                    self._seek_target = None
                    self._do_seek(target)
                    continue

                # Handle pause
                if self._paused:
                    time_module.sleep(0.1)
                    continue

                # Process current bar
                bar = self._bars[self._current_bar_index]
                self._process_bar(bar)
                self._current_bar_index += 1

                # Sleep based on speed (60 real seconds / speed factor)
                # Use short intervals for responsive speed/pause changes
                sleep_total = 60.0 / self._speed
                elapsed = 0.0
                while elapsed < sleep_total and self._running and not self._paused and self._seek_target is None:
                    step = min(0.1, sleep_total - elapsed)
                    time_module.sleep(step)
                    elapsed += step

        except Exception as e:
            self._error = f"Simulation error: {e}"
            logger.error(self._error, exc_info=True)
        finally:
            self._running = False
            logger.info("Simulation playback finished")

    def _process_bar(self, bar: Dict) -> None:
        """Process a single bar: feed to data buffer, update candles."""
        self._data_buffer.add_bar(bar)

        # Add candle for chart
        ts = bar["timestamp"]
        if hasattr(ts, "timestamp"):
            unix_sec = int(ts.timestamp())
        else:
            unix_sec = int(pd.Timestamp(ts).timestamp())

        self._candles.append({
            "time": unix_sec,
            "open": bar["open"],
            "high": bar["high"],
            "low": bar["low"],
            "close": bar["close"],
        })

    def _do_seek(self, target_index: int) -> None:
        """
        Seek to a specific bar index.

        Backward seek replays from start (necessary because LiveDataBuffer
        VWAP is cumulative and can't be truncated).
        """
        target_index = max(0, min(target_index, len(self._bars) - 1))

        if target_index <= self._current_bar_index:
            # Backward seek: replay from start
            self._data_buffer = LiveDataBuffer()
            self._candles = []
            for i in range(target_index + 1):
                self._process_bar(self._bars[i])
            self._current_bar_index = target_index + 1
        else:
            # Forward seek: fast-forward without sleep
            while self._current_bar_index <= target_index:
                self._process_bar(self._bars[self._current_bar_index])
                self._current_bar_index += 1

    def stop(self) -> None:
        """Stop the simulation."""
        with self._lock:
            if not self._running:
                return
            self._running = False
            if self._thread is not None:
                self._thread.join(timeout=5.0)
                self._thread = None
            self._started_at = None
            logger.info("Simulation stopped")

    def set_speed(self, speed: float) -> None:
        """Change playback speed."""
        self._speed = max(1.0, min(600.0, speed))

    def seek(self, bar_index: int) -> None:
        """Request seek to a specific bar index."""
        self._seek_target = bar_index

    def pause(self) -> None:
        """Pause playback."""
        self._paused = True

    def resume(self) -> None:
        """Resume playback."""
        self._paused = False

    def get_snapshot(self) -> Dict[str, Any]:
        """
        Build a snapshot dict matching TradingService shape, plus sim fields.
        """
        now = datetime.now(ET)

        # Get current simulated time from latest bar
        sim_time = None
        latest_bar = None
        if self._candles:
            latest_bar_data = self._data_buffer.get_latest_bar() if self._data_buffer else None
            if latest_bar_data:
                sim_time = str(latest_bar_data.timestamp)
                latest_bar = {
                    "timestamp": str(latest_bar_data.timestamp),
                    "open": latest_bar_data.open,
                    "high": latest_bar_data.high,
                    "low": latest_bar_data.low,
                    "close": latest_bar_data.close,
                    "volume": latest_bar_data.volume,
                }

        # Current price
        current_price = self._data_buffer.get_latest_price() if self._data_buffer else None

        # VWAP bands
        vwap = upper_1sd = lower_1sd = upper_2sd = lower_2sd = None
        if self._data_buffer:
            vwap, upper_1sd, lower_1sd, upper_2sd, lower_2sd = self._data_buffer.get_vwap_bands()

        # Opening range
        or_high = or_low = None
        if self._data_buffer:
            or_high, or_low = self._data_buffer.get_opening_range()

        # Minutes elapsed (from simulated time)
        minutes_elapsed = 0
        if self._data_buffer and self._data_buffer.get_latest_bar():
            bar_ts = self._data_buffer.get_latest_bar().timestamp
            if hasattr(bar_ts, 'hour'):
                market_open_dt = bar_ts.replace(hour=9, minute=30, second=0, microsecond=0)
                minutes_elapsed = max(0, int((bar_ts - market_open_dt).total_seconds() / 60))

        total_bars = len(self._bars)
        current_bar = min(self._current_bar_index, total_bars)
        pct = (current_bar / total_bars * 100) if total_bars > 0 else 0

        result: Dict[str, Any] = {
            "is_running": self._running,
            "error": self._error,
            "started_at": self._started_at.isoformat() if self._started_at else None,
            "config": self._config,
            "timestamp": now.isoformat(),
            "mode": "sim",
            "sim_progress": {
                "current_bar": current_bar,
                "total_bars": total_bars,
                "pct": round(pct, 1),
            },
            "sim_speed": self._speed,
            "sim_paused": self._paused,
            "sim_time": sim_time,
            "minutes_elapsed": minutes_elapsed,
            "is_market_hours": True,
            "current_price": current_price,
            "latest_bar": latest_bar,
            "vwap": vwap,
            "vwap_upper_1sd": upper_1sd,
            "vwap_lower_1sd": lower_1sd,
            "vwap_upper_2sd": upper_2sd,
            "vwap_lower_2sd": lower_2sd,
            "or_high": or_high,
            "or_low": or_low,
            "signal": {
                "action": self._last_signal_action,
                "source": self._last_signal_source,
                "confidence": self._last_signal_confidence,
                "agreeing_models": [],
            },
            "pending": {
                "has_pending_entry": False,
                "pending_entry": None,
                "has_pending_breach": False,
                "pending_breach_type": None,
            },
            "position": None,
            "stats": {
                "breach_immediate_entries": 0,
                "breach_continuation_entries": 0,
                "vwap_direction_rejections": 0,
                "timeframe_rejections": 0,
                "bars_in_buffer": len(self._data_buffer) if self._data_buffer else 0,
            },
            "session_id": "simulation",
        }

        return result

    def get_trades(self) -> Dict[str, Any]:
        """Get trade data (mock for now)."""
        return {
            "trades": self._trades,
            "daily_trades": len(self._trades),
            "daily_pnl": self._daily_pnl,
            "capital": self._capital,
            "win_rate": 0.0,
        }

    def get_candles(self) -> List[Dict]:
        """Return all candles for the chart."""
        return list(self._candles)

    def get_chart_data(self) -> Dict[str, Any]:
        """Build full chart data for TrainingChartView from all processed bars."""
        bar_count = min(self._current_bar_index, len(self._bars))
        if bar_count == 0:
            return {}
        return compute_chart_data(self._bars[:bar_count])
