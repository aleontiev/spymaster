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


@dataclass
class BacktestConfig:
    """Backtest configuration."""
    stop_loss_pct: float = 10.0
    position_size_pct: float = 5.0
    initial_capital: float = 100_000.0
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
