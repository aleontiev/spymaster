"""
Backtest data types and configuration.

Contains Position, Trade, and config dataclasses used by the backtesting engine.
"""
from dataclasses import dataclass, field
from datetime import datetime
from enum import IntEnum
from typing import Optional, Tuple


class PositionType(IntEnum):
    """Position type for options."""
    CALL = 1
    PUT = 2


@dataclass
class Position:
    """Active position."""
    position_type: PositionType
    entry_time: datetime
    entry_underlying_price: float
    entry_option_price: float
    strike: float
    entry_idx: int
    position_value: float
    dominant_model: str
    max_hold_minutes: int
    barrier_start_time: datetime = None
    renewals: int = 0
    confluence_count: int = 2
    position_size_multiplier: float = 1.0
    peak_pnl_pct: float = 0.0
    breakeven_activated: bool = False
    # Runner state - runner is left when closing first half at +20%+
    is_runner: bool = False
    runner_start_time: datetime = None
    runner_max_hold_minutes: int = 15  # Aggressive time barrier for runners
    runner_peak_pnl_pct: float = 0.0
    runner_entry_pnl_pct: float = 0.0  # P&L when runner started
    parent_entry_time: datetime = None  # Original entry time (for runners, tracks the first entry)
    parent_entry_option_price: float = None  # Original entry option price (for runners)
    num_contracts: int = 10  # Number of contracts in this position
    is_breach: bool = False  # True if this is a breakout/reversal trade (uses wider trailing stop)
    is_reversal: bool = False  # True if this is a reversal trade (+-2σ band)
    is_bounce: bool = False  # True if this is a VWAP bounce trade
    is_double_breakout: bool = False  # True if this is a double breakout trade
    is_news_event: bool = False  # True if this is a news event trade
    took_1sd_partial: bool = False  # True if we've already taken 20% partial profit at +1σ
    quality: str = ""  # Trade quality grade: A (5%), B (4%), C (3%), D (2%)

    def __post_init__(self):
        if self.barrier_start_time is None:
            self.barrier_start_time = self.entry_time
        if self.parent_entry_time is None:
            self.parent_entry_time = self.entry_time
        if self.parent_entry_option_price is None:
            self.parent_entry_option_price = self.entry_option_price


@dataclass
class Trade:
    """Completed trade."""
    position_type: PositionType
    entry_time: datetime
    exit_time: datetime
    entry_underlying: float
    exit_underlying: float
    entry_option_price: float
    exit_option_price: float
    pnl_pct: float
    pnl_dollars: float
    capital_after: float
    exit_reason: str
    holding_minutes: int
    dominant_model: str
    confluence_count: int = 2
    position_size_multiplier: float = 1.0
    agreeing_models: Tuple[str, ...] = ()
    strike: float = 0.0  # Strike price
    contract: str = ""  # e.g., "SPY 600C"
    num_contracts: int = 10  # Number of contracts traded
    is_runner: bool = False  # True if this is a runner continuation trade
    parent_entry_time: Optional[datetime] = None  # Original entry time for runners
    parent_entry_option_price: Optional[float] = None  # Original entry option price for runners
    quality: str = ""  # Trade quality grade: A (5%), B (4%), C (3%), D (2%)


@dataclass
class TradingConfig:
    """Trading configuration (used by both backtest and live trading)."""
    stop_loss_pct: float = 10.0
    position_size_pct: float = 5.0
    initial_capital: float = 25_000.0
    spread_cost_pct: float = 0.5
    commission_per_contract: float = 0.65
    contracts_per_trade: int = 10
    max_daily_allocation_pct: float = 1000.0
    max_trades_per_day: int = 1000
    daily_loss_limit_pct: float = 100.0
    max_trade_gain_pct: float = None
    trailing_stop_activation_pct: float = 5.0
    trailing_stop_distance_pct: float = 5.0
    entry_slippage_pct: float = 0.5  # Slippage on entry fills
    exit_slippage_pct: float = 0.5  # Slippage on exit fills
    # Power hour boost: max position size during 2-4pm ET
    power_hour_boost_enabled: bool = True
    power_hour_start_minutes: int = 14 * 60  # 2:00 PM ET (minutes from midnight)
    power_hour_end_minutes: int = 16 * 60    # 4:00 PM ET (minutes from midnight)


# Backward-compat alias
BacktestConfig = TradingConfig


@dataclass
class PendingEntry:
    """A pending entry to be executed on the next minute's open."""
    position_type: PositionType
    signal_time: datetime  # When signal was generated
    underlying_price: float  # Price when signal was generated
    signal_idx: int  # Index in the dataframe
    dominant_model: str
    max_hold_minutes: int
    confidence: float
    confluence_count: int
    agreeing_models: Tuple[str, ...]
    is_early_trade: bool = False  # True if this is an early window trade (9:45-10:00, ATM)
    is_breach: bool = False  # True if this is a breakout/reversal trade
    is_reversal: bool = False  # True if this is a reversal trade
    is_reversal_engulfing: bool = False  # True if reversal uses engulfing pattern (5% sizing)
    is_bounce: bool = False  # True if this is a VWAP bounce trade
    is_bounce_engulfing: bool = False  # True if bounce uses engulfing pattern (5% sizing)
    is_double_breakout: bool = False  # True if this is a double breakout trade
    is_news_event: bool = False  # True if this is a news event trade
    position_size_bonus: float = 0.0  # Dynamic position size bonus (0.0 to 0.02)


@dataclass
class MonthlyMetrics:
    """Monthly performance metrics."""
    year: int
    month: int
    trades: int = 0
    wins: int = 0
    win_rate: float = 0.0
    total_pnl_dollars: float = 0.0
    return_pct: float = 0.0
    long_trades: int = 0
    short_trades: int = 0
    long_pnl: float = 0.0
    short_pnl: float = 0.0
    avg_pnl_pct: float = 0.0
    max_drawdown_pct: float = 0.0
