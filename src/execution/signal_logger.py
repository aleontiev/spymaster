"""
Signal logger for recording trading signals to parquet files.

Logs candle data with signal annotations for analysis and replay.
Creates a signals-1m dataset that mirrors stocks-1m but with additional
signal annotation columns.
"""
import logging
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path
from typing import Dict, List, Optional, Any
from zoneinfo import ZoneInfo

import pandas as pd

from src.backtest.types import PositionType


logger = logging.getLogger(__name__)


@dataclass
class SignalAnnotation:
    """Annotation for a single candle's signal state."""
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: int

    # Entry annotations
    is_entry: bool = False
    entry_reason: Optional[str] = None  # "vwap_breach", "model_signal"
    entry_direction: Optional[str] = None  # "call" or "put"
    entry_confidence: Optional[float] = None

    # Exit annotations
    is_exit: bool = False
    exit_reason: Optional[str] = None  # "stop_loss", "time_barrier", "trailing_stop", etc.

    # P&L (only set on exit candle)
    pnl_pct: Optional[float] = None
    pnl_dollars: Optional[float] = None

    # Position state
    has_position: bool = False
    position_type: Optional[str] = None  # "call" or "put"
    position_pnl_pct: Optional[float] = None  # Running P&L while in position

    # Model signals (for analysis)
    model_1m_signal: Optional[str] = None
    model_5m_signal: Optional[str] = None
    model_15m_signal: Optional[str] = None
    fused_signal: Optional[str] = None
    fused_confidence: Optional[float] = None

    # Market context
    vwap: Optional[float] = None
    vwap_upper_1sd: Optional[float] = None
    vwap_lower_1sd: Optional[float] = None
    or_high: Optional[float] = None
    or_low: Optional[float] = None


class SignalLogger:
    """
    Logs trading signals with OHLCV data for each candle.

    Writes to parquet files following the signals-1m dataset path pattern:
    data/signals-1m/{underlying}/{date:%Y-%m}/{date:%d}.parquet
    """

    def __init__(
        self,
        underlying: str = "SPY",
        base_path: Path = Path("data/signals-1m"),
    ):
        """
        Initialize the signal logger.

        Args:
            underlying: Underlying symbol (default: "SPY")
            base_path: Base path for signal files
        """
        self.underlying = underlying
        self.base_path = Path(base_path)
        self.et_tz = ZoneInfo("America/New_York")

        # Buffer for current day's signals
        self._signals: List[SignalAnnotation] = []
        self._current_date: Optional[date] = None

    def log_candle(
        self,
        timestamp: datetime,
        open_: float,
        high: float,
        low: float,
        close: float,
        volume: int,
        is_entry: bool = False,
        entry_reason: Optional[str] = None,
        entry_direction: Optional[str] = None,
        entry_confidence: Optional[float] = None,
        is_exit: bool = False,
        exit_reason: Optional[str] = None,
        pnl_pct: Optional[float] = None,
        pnl_dollars: Optional[float] = None,
        has_position: bool = False,
        position_type: Optional[str] = None,
        position_pnl_pct: Optional[float] = None,
        model_1m_signal: Optional[str] = None,
        model_5m_signal: Optional[str] = None,
        model_15m_signal: Optional[str] = None,
        fused_signal: Optional[str] = None,
        fused_confidence: Optional[float] = None,
        vwap: Optional[float] = None,
        vwap_upper_1sd: Optional[float] = None,
        vwap_lower_1sd: Optional[float] = None,
        or_high: Optional[float] = None,
        or_low: Optional[float] = None,
    ) -> None:
        """
        Log a candle with its signal annotations.

        Args:
            timestamp: Candle timestamp
            open_: Open price
            high: High price
            low: Low price
            close: Close price
            volume: Volume
            is_entry: Whether this candle triggered an entry
            entry_reason: Reason for entry
            entry_direction: "call" or "put"
            entry_confidence: Entry confidence score
            is_exit: Whether this candle triggered an exit
            exit_reason: Reason for exit
            pnl_pct: P&L percentage (on exit)
            pnl_dollars: P&L in dollars (on exit)
            has_position: Whether we have a position
            position_type: Type of position ("call" or "put")
            position_pnl_pct: Current P&L while in position
            model_1m_signal: 1-minute model signal
            model_5m_signal: 5-minute model signal
            model_15m_signal: 15-minute model signal
            fused_signal: Fused signal from all models
            fused_confidence: Fused signal confidence
            vwap: Current VWAP
            vwap_upper_1sd: VWAP + 1 standard deviation
            vwap_lower_1sd: VWAP - 1 standard deviation
            or_high: Opening Range high
            or_low: Opening Range low
        """
        # Check for new day
        candle_date = timestamp.date() if hasattr(timestamp, 'date') else pd.Timestamp(timestamp).date()

        if self._current_date is not None and candle_date != self._current_date:
            # Flush previous day's data
            self._flush_to_disk()

        self._current_date = candle_date

        signal = SignalAnnotation(
            timestamp=timestamp,
            open=open_,
            high=high,
            low=low,
            close=close,
            volume=volume,
            is_entry=is_entry,
            entry_reason=entry_reason,
            entry_direction=entry_direction,
            entry_confidence=entry_confidence,
            is_exit=is_exit,
            exit_reason=exit_reason,
            pnl_pct=pnl_pct,
            pnl_dollars=pnl_dollars,
            has_position=has_position,
            position_type=position_type,
            position_pnl_pct=position_pnl_pct,
            model_1m_signal=model_1m_signal,
            model_5m_signal=model_5m_signal,
            model_15m_signal=model_15m_signal,
            fused_signal=fused_signal,
            fused_confidence=fused_confidence,
            vwap=vwap,
            vwap_upper_1sd=vwap_upper_1sd,
            vwap_lower_1sd=vwap_lower_1sd,
            or_high=or_high,
            or_low=or_low,
        )

        self._signals.append(signal)

    def log_entry(
        self,
        timestamp: datetime,
        bar: Dict[str, Any],
        reason: str,
        direction: PositionType,
        confidence: float,
        vwap_info: Optional[Dict[str, float]] = None,
        or_info: Optional[Dict[str, float]] = None,
        model_signals: Optional[Dict[str, str]] = None,
    ) -> None:
        """
        Convenience method to log an entry signal.

        Args:
            timestamp: Entry timestamp
            bar: OHLCV bar data
            reason: Entry reason
            direction: Position type (CALL or PUT)
            confidence: Entry confidence
            vwap_info: VWAP bands info
            or_info: Opening Range info
            model_signals: Individual model signals
        """
        self.log_candle(
            timestamp=timestamp,
            open_=bar.get("open", 0),
            high=bar.get("high", 0),
            low=bar.get("low", 0),
            close=bar.get("close", 0),
            volume=bar.get("volume", 0),
            is_entry=True,
            entry_reason=reason,
            entry_direction=direction.value if isinstance(direction, PositionType) else str(direction).lower(),
            entry_confidence=confidence,
            has_position=True,
            position_type=direction.value if isinstance(direction, PositionType) else str(direction).lower(),
            position_pnl_pct=0.0,
            vwap=vwap_info.get("vwap") if vwap_info else None,
            vwap_upper_1sd=vwap_info.get("upper_1sd") if vwap_info else None,
            vwap_lower_1sd=vwap_info.get("lower_1sd") if vwap_info else None,
            or_high=or_info.get("high") if or_info else None,
            or_low=or_info.get("low") if or_info else None,
            model_1m_signal=model_signals.get("1m") if model_signals else None,
            model_5m_signal=model_signals.get("5m") if model_signals else None,
            model_15m_signal=model_signals.get("15m") if model_signals else None,
            fused_signal=model_signals.get("fused") if model_signals else None,
            fused_confidence=confidence,
        )

    def log_exit(
        self,
        timestamp: datetime,
        bar: Dict[str, Any],
        reason: str,
        pnl_pct: float,
        pnl_dollars: float,
        position_type: PositionType,
    ) -> None:
        """
        Convenience method to log an exit signal.

        Args:
            timestamp: Exit timestamp
            bar: OHLCV bar data
            reason: Exit reason
            pnl_pct: P&L percentage
            pnl_dollars: P&L in dollars
            position_type: Type of position being closed
        """
        self.log_candle(
            timestamp=timestamp,
            open_=bar.get("open", 0),
            high=bar.get("high", 0),
            low=bar.get("low", 0),
            close=bar.get("close", 0),
            volume=bar.get("volume", 0),
            is_exit=True,
            exit_reason=reason,
            pnl_pct=pnl_pct,
            pnl_dollars=pnl_dollars,
            has_position=False,
            position_type=position_type.value if isinstance(position_type, PositionType) else str(position_type).lower(),
        )

    def _flush_to_disk(self) -> None:
        """Write buffered signals to disk."""
        if not self._signals or self._current_date is None:
            return

        # Convert to DataFrame
        records = []
        for s in self._signals:
            records.append({
                "timestamp": s.timestamp,
                "open": s.open,
                "high": s.high,
                "low": s.low,
                "close": s.close,
                "volume": s.volume,
                "is_entry": s.is_entry,
                "entry_reason": s.entry_reason,
                "entry_direction": s.entry_direction,
                "entry_confidence": s.entry_confidence,
                "is_exit": s.is_exit,
                "exit_reason": s.exit_reason,
                "pnl_pct": s.pnl_pct,
                "pnl_dollars": s.pnl_dollars,
                "has_position": s.has_position,
                "position_type": s.position_type,
                "position_pnl_pct": s.position_pnl_pct,
                "model_1m_signal": s.model_1m_signal,
                "model_5m_signal": s.model_5m_signal,
                "model_15m_signal": s.model_15m_signal,
                "fused_signal": s.fused_signal,
                "fused_confidence": s.fused_confidence,
                "vwap": s.vwap,
                "vwap_upper_1sd": s.vwap_upper_1sd,
                "vwap_lower_1sd": s.vwap_lower_1sd,
                "or_high": s.or_high,
                "or_low": s.or_low,
            })

        df = pd.DataFrame(records)

        # Build output path
        output_path = (
            self.base_path
            / self.underlying
            / self._current_date.strftime("%Y-%m")
            / f"{self._current_date.strftime('%d')}.parquet"
        )
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Check for existing file and merge if needed
        if output_path.exists():
            existing_df = pd.read_parquet(output_path)
            df = pd.concat([existing_df, df]).drop_duplicates(
                subset=["timestamp"], keep="last"
            ).sort_values("timestamp")

        # Write to parquet
        df.to_parquet(output_path, index=False)

        logger.info(f"Wrote {len(df)} signals to {output_path}")

        # Clear buffer
        self._signals.clear()

    def flush(self) -> None:
        """Force flush any buffered signals to disk."""
        self._flush_to_disk()

    def close(self) -> None:
        """Flush and close the logger."""
        self._flush_to_disk()

    def __enter__(self) -> "SignalLogger":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()


def register_signals_dataset() -> None:
    """
    Register the signals-1m dataset in the database.

    This should be called once to set up the dataset configuration.
    """
    from src.db.database import get_db

    db = get_db()

    # Check if already exists
    if db.dataset_exists("signals-1m"):
        logger.info("signals-1m dataset already registered")
        return

    # Create the dataset
    db.create_dataset(
        name="signals-1m",
        type="computed",
        path_pattern="data/signals-1m/{underlying}/{date:%Y-%m}/{date:%d}.parquet",
        description="1-minute candles with trading signal annotations",
        granularity="daily",
        computation="src.execution.signal_logger:compute",
        field_map={
            "timestamp": {"type": "datetime", "description": "Candle timestamp"},
            "open": {"type": "float64", "description": "Opening price"},
            "high": {"type": "float64", "description": "High price"},
            "low": {"type": "float64", "description": "Low price"},
            "close": {"type": "float64", "description": "Closing price"},
            "volume": {"type": "int64", "description": "Volume"},
            "is_entry": {"type": "bool", "description": "Whether this candle triggered an entry"},
            "entry_reason": {"type": "string", "description": "Reason for entry (vwap_breach, model_signal)"},
            "entry_direction": {"type": "string", "description": "Entry direction (call or put)"},
            "entry_confidence": {"type": "float64", "description": "Entry confidence score"},
            "is_exit": {"type": "bool", "description": "Whether this candle triggered an exit"},
            "exit_reason": {"type": "string", "description": "Reason for exit"},
            "pnl_pct": {"type": "float64", "description": "P&L percentage (on exit)"},
            "pnl_dollars": {"type": "float64", "description": "P&L in dollars (on exit)"},
            "has_position": {"type": "bool", "description": "Whether we have a position"},
            "position_type": {"type": "string", "description": "Position type (call or put)"},
            "position_pnl_pct": {"type": "float64", "description": "Current P&L while in position"},
            "model_1m_signal": {"type": "string", "description": "1-minute model signal"},
            "model_5m_signal": {"type": "string", "description": "5-minute model signal"},
            "model_15m_signal": {"type": "string", "description": "15-minute model signal"},
            "fused_signal": {"type": "string", "description": "Fused signal from all models"},
            "fused_confidence": {"type": "float64", "description": "Fused signal confidence"},
            "vwap": {"type": "float64", "description": "Current VWAP"},
            "vwap_upper_1sd": {"type": "float64", "description": "VWAP + 1 standard deviation"},
            "vwap_lower_1sd": {"type": "float64", "description": "VWAP - 1 standard deviation"},
            "or_high": {"type": "float64", "description": "Opening Range high"},
            "or_low": {"type": "float64", "description": "Opening Range low"},
        },
        dependencies={
            "stocks-1m": {"relation": "same_day"},
        },
    )

    logger.info("Registered signals-1m dataset")


# Computation function for DAG (placeholder - signals are logged live)
def compute(
    data: Dict[str, pd.DataFrame],
    target_date: date,
    underlying: str = "SPY",
) -> pd.DataFrame:
    """
    Computation function for signals-1m dataset.

    This is a placeholder - signals are actually logged live by SignalLogger.
    If called, it reads any existing signal file.
    """
    signal_path = Path(f"data/signals-1m/{underlying}/{target_date:%Y-%m}/{target_date:%d}.parquet")

    if signal_path.exists():
        return pd.read_parquet(signal_path)

    # Return empty DataFrame with correct schema if no signals exist
    return pd.DataFrame({
        "timestamp": pd.Series(dtype="datetime64[ns]"),
        "open": pd.Series(dtype="float64"),
        "high": pd.Series(dtype="float64"),
        "low": pd.Series(dtype="float64"),
        "close": pd.Series(dtype="float64"),
        "volume": pd.Series(dtype="int64"),
        "is_entry": pd.Series(dtype="bool"),
        "entry_reason": pd.Series(dtype="object"),
        "entry_direction": pd.Series(dtype="object"),
        "entry_confidence": pd.Series(dtype="float64"),
        "is_exit": pd.Series(dtype="bool"),
        "exit_reason": pd.Series(dtype="object"),
        "pnl_pct": pd.Series(dtype="float64"),
        "pnl_dollars": pd.Series(dtype="float64"),
        "has_position": pd.Series(dtype="bool"),
        "position_type": pd.Series(dtype="object"),
        "position_pnl_pct": pd.Series(dtype="float64"),
        "model_1m_signal": pd.Series(dtype="object"),
        "model_5m_signal": pd.Series(dtype="object"),
        "model_15m_signal": pd.Series(dtype="object"),
        "fused_signal": pd.Series(dtype="object"),
        "fused_confidence": pd.Series(dtype="float64"),
        "vwap": pd.Series(dtype="float64"),
        "vwap_upper_1sd": pd.Series(dtype="float64"),
        "vwap_lower_1sd": pd.Series(dtype="float64"),
        "or_high": pd.Series(dtype="float64"),
        "or_low": pd.Series(dtype="float64"),
    })
