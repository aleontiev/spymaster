"""
Position manager.

Owns all position, capital, and drawdown state for trading (backtest and live).
Extracted from MultiPercentileBacktester to separate position management
from signal generation and backtesting orchestration.
"""
import random
from dataclasses import dataclass
from datetime import date, datetime
from typing import Dict, List, Optional, Tuple
from zoneinfo import ZoneInfo

from src.backtest.types import (
    TradingConfig,
    Position,
    PositionType,
    Trade,
)
from src.execution.heuristic import HeuristicConfig
from src.execution.options_provider import RealOptionsProvider
from src.strategy.fusion_config import FusedSignal, Signal


@dataclass
class LivePosition:
    """Active position for live trading (mirrors backtest Position with live-specific fields)."""
    position_type: PositionType
    entry_time: datetime
    entry_option_price: float
    entry_underlying_price: float
    option_symbol: str
    strike: float
    num_contracts: int
    position_value: float
    dominant_model: str
    max_hold_minutes: int
    barrier_start_time: Optional[datetime] = None
    confluence_count: int = 2
    peak_pnl_pct: float = 0.0
    breakeven_activated: bool = False
    is_runner: bool = False
    runner_start_time: Optional[datetime] = None
    runner_max_hold_minutes: int = 15
    runner_peak_pnl_pct: float = 0.0
    runner_entry_pnl_pct: float = 0.0
    is_breach: bool = False
    is_reversal: bool = False
    is_bounce: bool = False
    is_double_breakout: bool = False
    is_news_event: bool = False

    def __post_init__(self) -> None:
        if self.barrier_start_time is None:
            self.barrier_start_time = self.entry_time

    def to_dict(self) -> dict:
        """Serialize to dict for the live dashboard."""
        return {
            "position_type": self.position_type.value,
            "entry_time": self.entry_time.isoformat(),
            "entry_option_price": self.entry_option_price,
            "entry_underlying_price": self.entry_underlying_price,
            "option_symbol": self.option_symbol,
            "strike": self.strike,
            "num_contracts": self.num_contracts,
            "position_value": self.position_value,
            "dominant_model": self.dominant_model,
            "max_hold_minutes": self.max_hold_minutes,
            "peak_pnl_pct": self.peak_pnl_pct,
            "breakeven_activated": self.breakeven_activated,
            "is_runner": self.is_runner,
            "is_breach": self.is_breach,
            "is_reversal": self.is_reversal,
            "is_bounce": self.is_bounce,
            "is_news_event": self.is_news_event,
            "confluence_count": self.confluence_count,
        }


class PositionManager:
    """Manages position state, capital tracking, and trade execution."""

    def __init__(
        self,
        config: TradingConfig,
        options: RealOptionsProvider,
        heuristic_config: Optional[HeuristicConfig] = None,
    ):
        self.config = config
        self.options = options
        self.heuristic_config = heuristic_config or HeuristicConfig()
        self.et_tz = ZoneInfo("America/New_York")

        self.position: Optional[Position] = None
        self.trades: List[Trade] = []
        self.capital = config.initial_capital
        self.peak_capital = config.initial_capital
        self.lowest_capital = config.initial_capital
        self.model_contributions: Dict[str, int] = {"15m": 0, "5m": 0, "1m": 0, "bounce": 0, "news_event": 0}

        # Daily accounting
        self.daily_trades = 0
        self.daily_allocation = 0.0
        self.daily_pnl = 0.0
        self.daily_start_capital = config.initial_capital
        self.current_day: Optional[date] = None

        # Profit protection tracking (per day)
        self.daily_cost_basis = 0.0
        self.daily_realized_profit = 0.0
        self.daily_protected_capital = 0.0
        self.profit_protection_active = False
        self.max_daily_cost_basis = config.initial_capital

        # Daily/Weekly drawdown tracking
        self.max_daily_drawdown_pct = 0.0
        self.max_weekly_drawdown_pct = 0.0
        self.daily_peak_capital = config.initial_capital
        self.daily_low_capital = config.initial_capital
        self.weekly_start_capital = config.initial_capital
        self.weekly_peak_capital = config.initial_capital
        self.weekly_low_capital = config.initial_capital
        self.current_week: Optional[tuple] = None

        # Daily/Weekly gain rate tracking
        self.daily_results: List[float] = []
        self.weekly_results: List[float] = []
        self.prev_day_capital: float = config.initial_capital
        self.prev_week_capital: float = config.initial_capital
        self.last_trade_date: Optional[date] = None
        self.last_trade_week: Optional[tuple] = None

        # Best/Worst Daily/Weekly P&L tracking
        self.best_daily_pnl_pct: float = 0.0
        self.worst_daily_pnl_pct: float = 0.0
        self.best_weekly_pnl_pct: float = 0.0
        self.worst_weekly_pnl_pct: float = 0.0
        self.day_start_capital: float = config.initial_capital
        self.week_start_capital: float = config.initial_capital

    def reset(self, starting_capital: Optional[float] = None) -> None:
        """Reset all position manager state for a new backtest."""
        capital = starting_capital if starting_capital is not None else self.config.initial_capital
        self.position = None
        self.trades = []
        self.capital = capital
        self.peak_capital = capital
        self.lowest_capital = capital
        self.daily_trades = 0
        self.daily_allocation = 0.0
        self.daily_pnl = 0.0
        self.daily_start_capital = capital
        self.current_day = None
        self.model_contributions = {"15m": 0, "5m": 0, "1m": 0, "bounce": 0, "news_event": 0}

        # Profit protection reset
        self.daily_cost_basis = 0.0
        self.daily_realized_profit = 0.0
        self.daily_protected_capital = 0.0
        self.profit_protection_active = False
        self.max_daily_cost_basis = capital

        # Daily/Weekly drawdown tracking
        self.max_daily_drawdown_pct = 0.0
        self.max_weekly_drawdown_pct = 0.0
        self.daily_peak_capital = capital
        self.daily_low_capital = capital
        self.weekly_start_capital = capital
        self.weekly_peak_capital = capital
        self.weekly_low_capital = capital
        self.current_week = None

        # Daily/Weekly gain rate tracking
        self.daily_results = []
        self.weekly_results = []
        self.prev_day_capital = capital
        self.prev_week_capital = capital
        self.last_trade_date = None
        self.last_trade_week = None

        # Best/Worst Daily/Weekly P&L tracking
        self.best_daily_pnl_pct = 0.0
        self.worst_daily_pnl_pct = 0.0
        self.best_weekly_pnl_pct = 0.0
        self.worst_weekly_pnl_pct = 0.0
        self.day_start_capital = capital
        self.week_start_capital = capital

    def update_drawdown_tracking(self, trade_time: datetime) -> None:
        """Update daily and weekly drawdown tracking after capital changes."""
        trade_date = trade_time.date()
        trade_week = trade_date.isocalendar()[:2]  # (year, week_number)

        # Check for new day - finalize previous day's drawdown and P&L
        # Use last_trade_date (not current_day) to properly detect day changes
        if self.last_trade_date is not None and self.last_trade_date != trade_date:
            # Calculate previous day's drawdown (peak to trough within day)
            if self.daily_peak_capital > 0:
                daily_dd = (self.daily_peak_capital - self.daily_low_capital) / self.daily_peak_capital * 100
                self.max_daily_drawdown_pct = max(self.max_daily_drawdown_pct, daily_dd)
            # Record daily P&L percentage (end of previous day vs start of previous day)
            if self.prev_day_capital > 0:
                daily_pnl_pct = (self.capital - self.prev_day_capital) / self.prev_day_capital * 100
                self.daily_results.append(daily_pnl_pct)
            self.prev_day_capital = self.capital  # New day starts with current capital
            # Reset for new day
            self.daily_peak_capital = self.capital
            self.daily_low_capital = self.capital
            self.day_start_capital = self.capital  # Reset day start capital

        # Check for new week - finalize previous week's drawdown and P&L
        if self.last_trade_week is not None and self.last_trade_week != trade_week:
            # Calculate previous week's drawdown (peak to trough within week)
            if self.weekly_peak_capital > 0:
                weekly_dd = (self.weekly_peak_capital - self.weekly_low_capital) / self.weekly_peak_capital * 100
                self.max_weekly_drawdown_pct = max(self.max_weekly_drawdown_pct, weekly_dd)
            # Record weekly P&L percentage
            if self.prev_week_capital > 0:
                weekly_pnl_pct = (self.capital - self.prev_week_capital) / self.prev_week_capital * 100
                self.weekly_results.append(weekly_pnl_pct)
            self.prev_week_capital = self.capital
            # Reset for new week
            self.weekly_start_capital = self.capital
            self.weekly_peak_capital = self.capital
            self.weekly_low_capital = self.capital
            self.week_start_capital = self.capital  # Reset week start capital

        # Update tracking variables
        self.last_trade_date = trade_date
        self.last_trade_week = trade_week

        # Update peak and low for current day/week
        self.daily_peak_capital = max(self.daily_peak_capital, self.capital)
        self.daily_low_capital = min(self.daily_low_capital, self.capital)
        self.weekly_peak_capital = max(self.weekly_peak_capital, self.capital)
        self.weekly_low_capital = min(self.weekly_low_capital, self.capital)

        # Track best/worst intraday P&L (relative to day start)
        if self.day_start_capital > 0:
            intraday_peak_pct = (self.daily_peak_capital - self.day_start_capital) / self.day_start_capital * 100
            intraday_trough_pct = (self.daily_low_capital - self.day_start_capital) / self.day_start_capital * 100
            self.best_daily_pnl_pct = max(self.best_daily_pnl_pct, intraday_peak_pct)
            self.worst_daily_pnl_pct = min(self.worst_daily_pnl_pct, intraday_trough_pct)

        # Track best/worst intraweek P&L (relative to week start)
        if self.week_start_capital > 0:
            intraweek_peak_pct = (self.weekly_peak_capital - self.week_start_capital) / self.week_start_capital * 100
            intraweek_trough_pct = (self.weekly_low_capital - self.week_start_capital) / self.week_start_capital * 100
            self.best_weekly_pnl_pct = max(self.best_weekly_pnl_pct, intraweek_peak_pct)
            self.worst_weekly_pnl_pct = min(self.worst_weekly_pnl_pct, intraweek_trough_pct)

    def finalize_drawdown_tracking(self) -> None:
        """Finalize drawdown and P&L tracking for the last day/week at end of backtest."""
        # Finalize last day's drawdown and P&L
        if self.daily_peak_capital > 0:
            daily_dd = (self.daily_peak_capital - self.daily_low_capital) / self.daily_peak_capital * 100
            self.max_daily_drawdown_pct = max(self.max_daily_drawdown_pct, daily_dd)
        # Record last day's P&L
        if self.prev_day_capital > 0 and self.capital != self.prev_day_capital:
            daily_pnl_pct = (self.capital - self.prev_day_capital) / self.prev_day_capital * 100
            self.daily_results.append(daily_pnl_pct)

        # Finalize last week's drawdown and P&L
        if self.weekly_peak_capital > 0:
            weekly_dd = (self.weekly_peak_capital - self.weekly_low_capital) / self.weekly_peak_capital * 100
            self.max_weekly_drawdown_pct = max(self.max_weekly_drawdown_pct, weekly_dd)
        # Record last week's P&L
        if self.prev_week_capital > 0 and self.capital != self.prev_week_capital:
            weekly_pnl_pct = (self.capital - self.prev_week_capital) / self.prev_week_capital * 100
            self.weekly_results.append(weekly_pnl_pct)

        # Finalize best/worst intraday P&L for last day
        if self.day_start_capital > 0:
            intraday_peak_pct = (self.daily_peak_capital - self.day_start_capital) / self.day_start_capital * 100
            intraday_trough_pct = (self.daily_low_capital - self.day_start_capital) / self.day_start_capital * 100
            self.best_daily_pnl_pct = max(self.best_daily_pnl_pct, intraday_peak_pct)
            self.worst_daily_pnl_pct = min(self.worst_daily_pnl_pct, intraday_trough_pct)

        # Finalize best/worst intraweek P&L for last week
        if self.week_start_capital > 0:
            intraweek_peak_pct = (self.weekly_peak_capital - self.week_start_capital) / self.week_start_capital * 100
            intraweek_trough_pct = (self.weekly_low_capital - self.week_start_capital) / self.week_start_capital * 100
            self.best_weekly_pnl_pct = max(self.best_weekly_pnl_pct, intraweek_peak_pct)
            self.worst_weekly_pnl_pct = min(self.worst_weekly_pnl_pct, intraweek_trough_pct)

    def check_exit_or_renew(
        self,
        current_price: float,
        current_time: datetime,
        fused_signal: FusedSignal,
        upper_2sd: Optional[float] = None,
        lower_2sd: Optional[float] = None,
        upper_1sd: Optional[float] = None,
        lower_1sd: Optional[float] = None,
        candle_open: Optional[float] = None,
        candle_close: Optional[float] = None,
        avg_body: Optional[float] = None,
    ) -> Tuple[Optional[str], bool, int]:
        if self.position is None:
            return None, False, 0

        current_option_price = self.options.get_option_price(
            timestamp=current_time,
            strike=self.position.strike,
            position_type=self.position.position_type,
            underlying_price=current_price,
        )

        if current_option_price is None:
            return None, False, 0

        pnl_pct = (current_option_price - self.position.entry_option_price) / self.position.entry_option_price * 100

        # Track peak P&L
        if not self.position.is_runner:
            if pnl_pct > self.position.peak_pnl_pct:
                self.position.peak_pnl_pct = pnl_pct
        else:
            # Track runner peak separately
            if pnl_pct > self.position.runner_peak_pnl_pct:
                self.position.runner_peak_pnl_pct = pnl_pct

        # === RUNNER EXIT LOGIC ===
        # NOTE: Runner P&L is calculated from ORIGINAL entry price (not split price)
        # runner_entry_pnl_pct = P&L% when runner was created (e.g., +20%)
        # Profit = gained more than split point, Loss = gave back gains since split
        if self.position.is_runner:
            split_pnl = self.position.runner_entry_pnl_pct  # P&L when we split (e.g., +20%)
            gain_since_split = pnl_pct - split_pnl  # How much gained/lost since split

            # RUNNER EXITS:
            # 1. Close if runner falls below the original position's profitability
            if pnl_pct < split_pnl:
                return "runner_below_split", False, 0

            # Runner time barrier (aggressive - 15 mins default)
            runner_time = (current_time - self.position.runner_start_time).total_seconds() / 60
            if runner_time >= self.position.runner_max_hold_minutes:
                # Time barrier exit - compare to split point
                if gain_since_split >= 5.0:
                    return "runner_profit_time", False, 0
                elif gain_since_split >= 0.0:
                    return "runner_breakeven_time", False, 0
                else:
                    return "runner_loss_time", False, 0

            # Runner trailing stop - trail by 10% from runner peak
            # Minimum stop is at split P&L (protect the locked-in gains)
            runner_stop = self.position.runner_peak_pnl_pct - 10.0
            runner_stop = max(runner_stop, split_pnl)  # Never give back more than split gains
            peak_gain_since_split = self.position.runner_peak_pnl_pct - split_pnl
            if pnl_pct <= runner_stop and peak_gain_since_split >= 5.0:
                # Only trigger trailing if we've had some gains since split to protect
                if gain_since_split >= 5.0:
                    return "runner_profit_trailing", False, 0
                elif gain_since_split >= 0.0:
                    return "runner_breakeven_trailing", False, 0
                else:
                    return "runner_loss_trailing", False, 0

            # Fixed stop loss for runner - if we give back 10% from split point
            if gain_since_split <= -10.0:
                return "runner_stop_loss", False, 0

            # Signal reversal closes runner
            if self.position.position_type == PositionType.CALL and fused_signal.action == Signal.SHORT:
                if gain_since_split >= 5.0:
                    return "runner_profit_reversal", False, 0
                elif gain_since_split >= 0.0:
                    return "runner_breakeven_reversal", False, 0
                else:
                    return "runner_loss_reversal", False, 0
            if self.position.position_type == PositionType.PUT and fused_signal.action == Signal.LONG:
                if gain_since_split >= 5.0:
                    return "runner_profit_reversal", False, 0
                elif gain_since_split >= 0.0:
                    return "runner_breakeven_reversal", False, 0
                else:
                    return "runner_loss_reversal", False, 0

            # Runner doesn't renew, just hold
            return None, False, 0

        # === NORMAL POSITION EXIT LOGIC ===

        # 1. Fixed stop loss
        if not self.position.breakeven_activated:
            if pnl_pct <= -self.config.stop_loss_pct:
                return "stop_loss", False, 0
            # VWAP trades: breakeven at 10%, others at 7.5%
            breakeven_threshold = 10.0 if self.position.is_breach else 7.5
            if pnl_pct >= breakeven_threshold:
                self.position.breakeven_activated = True

        # 2. Breakeven stop (trailing from peak)
        # Once breakeven is activated, we protect profits with trailing stop
        # VWAP trades: hard floor at +2% (lock in profit when hitting 10%)
        # Other trades: floor at +2.5%
        if self.position.breakeven_activated:
            # VWAP trades have a hard breakeven floor at +2%
            if self.position.is_breach and pnl_pct <= 2.0:
                return "vwap_breakeven_stop", False, 0

            trail_distance = 5.0  # Same for all trades - aggressive trailing
            trailing_stop = self.position.peak_pnl_pct - trail_distance
            if pnl_pct <= trailing_stop:
                # Check if we should convert to runner (+20% or more)
                if pnl_pct >= 20.0:
                    return "convert_to_runner", False, 0
                elif pnl_pct <= 2.5:
                    # Breakeven stop with auto-stop - guaranteed to close at 0% minimum
                    return "breakeven_stop", False, 0
                else:
                    return "trailing_stop", False, 0

        # 3. Signal reversal
        if self.position.position_type == PositionType.CALL and fused_signal.action == Signal.SHORT:
            # Check if profitable enough for runner
            if pnl_pct >= 20.0:
                return "convert_to_runner", False, 0
            return "signal_reversal", False, 0
        if self.position.position_type == PositionType.PUT and fused_signal.action == Signal.LONG:
            if pnl_pct >= 20.0:
                return "convert_to_runner", False, 0
            return "signal_reversal", False, 0

        # 4. Signal confirmation - reset barriers
        if self.position.position_type == PositionType.CALL and fused_signal.action == Signal.LONG:
            return None, True, fused_signal.exit_horizon_minutes
        if self.position.position_type == PositionType.PUT and fused_signal.action == Signal.SHORT:
            return None, True, fused_signal.exit_horizon_minutes

        # 5. Time barrier - check if profitable enough for runner
        time_since_barrier_start = (current_time - self.position.barrier_start_time).total_seconds() / 60
        if time_since_barrier_start >= self.position.max_hold_minutes:
            if pnl_pct >= 20.0:
                return "convert_to_runner", False, 0
            return "time_barrier", False, 0

        return None, False, 0

    def open_position(
        self,
        position_type: PositionType,
        entry_time: datetime,
        underlying_price: float,
        entry_idx: int,
        dominant_model: str,
        max_hold_minutes: int,
        confidence: float = 0.5,
        confluence_count: int = 2,
        agreeing_models: Tuple[str, ...] = (),
        is_early_trade: bool = False,
        is_breach: bool = False,
        position_size_bonus: float = 0.0,
        is_bounce: bool = False,
        is_bounce_engulfing: bool = False,
        is_news_event: bool = False,
    ) -> bool:
        entry_date = entry_time.date()
        if self.current_day != entry_date:
            self.current_day = entry_date
            self.daily_trades = 0
            self.daily_allocation = 0.0
            self.daily_pnl = 0.0
            self.daily_start_capital = self.capital
            # Reset profit protection for new day
            self.daily_cost_basis = 0.0
            self.daily_realized_profit = 0.0
            self.daily_protected_capital = 0.0
            self.profit_protection_active = False
            self.max_daily_cost_basis = self.capital  # Can only invest up to starting balance

        if self.daily_trades >= self.config.max_trades_per_day:
            return False

        daily_loss_pct = -self.daily_pnl / self.daily_start_capital * 100
        if daily_loss_pct >= self.config.daily_loss_limit_pct:
            return False

        # Strike selection
        # VWAP breakout trades: Time-based strike selection
        #   - Before 10:30am or after 3pm: OTM (1 strike away from ITM)
        #   - 10:30am to 3pm: ATM
        # Early trades (9:45-10:00): Always ATM
        # Normal trades: Based on confidence (ITM default, OTM for high confidence)
        et_time = entry_time.astimezone(self.et_tz) if entry_time.tzinfo else entry_time
        hour, minute = et_time.hour, et_time.minute
        time_minutes = hour * 60 + minute
        is_otm_time = time_minutes < 10 * 60 + 30 or time_minutes >= 15 * 60  # Before 10:30am or after 3pm

        if is_news_event:
            # News event: Always ATM
            strike = round(underlying_price)
        elif is_bounce_engulfing:
            # Strong bounce (engulfing): OTM
            if position_type == PositionType.CALL:
                strike = int(underlying_price) + 1  # 1 strike OTM for calls
            else:
                strike = int(underlying_price)  # 1 strike OTM for puts
        elif is_bounce:
            # Normal bounce: ATM
            strike = round(underlying_price)
        elif is_breach:
            # VWAP breakout trades: Always ATM
            strike = round(underlying_price)
        elif is_early_trade:
            # Early window trades are always ATM (round to nearest strike)
            strike = round(underlying_price)
        elif position_type == PositionType.CALL:
            # For calls: ITM = lower strike, OTM = higher strike
            if confidence >= 0.90:
                strike = int(underlying_price) + 2  # 2 strikes OTM
            elif confidence >= 0.60:
                strike = int(underlying_price) + 1  # 1 strike OTM
            else:
                strike = int(underlying_price)  # ITM (floor)
        else:
            # For puts: ITM = higher strike, OTM = lower strike
            if confidence >= 0.90:
                strike = int(underlying_price) - 1  # 2 strikes OTM (round up then -2)
            elif confidence >= 0.60:
                strike = int(underlying_price)  # 1 strike OTM
            else:
                strike = int(underlying_price) + 1  # ITM (ceiling)

        # Get option price at OPEN of execution minute + slippage
        entry_option_price = self.options.get_option_price(
            timestamp=entry_time,
            strike=strike,
            position_type=position_type,
            underlying_price=underlying_price,
            price_type="open",  # Fill at open price
        )

        if entry_option_price is None:
            return False

        # Add slippage (we pay more to enter)
        entry_option_price *= (1 + self.config.entry_slippage_pct / 100)

        # Position size multiplier: 2x when portfolio is under $50k (to recover faster)
        size_multiplier = 2.0 if self.capital < 50000 else 1.0

        # Power hour boost: max position size during configured window (default 2-4pm ET)
        if self.config.power_hour_boost_enabled:
            if self.config.power_hour_start_minutes <= time_minutes < self.config.power_hour_end_minutes:
                position_size_bonus = 0.01  # 1% bonus during power hour

        # Dynamic position sizing
        # VWAP breakout trades: 3% of current balance (conservative)
        # Bounce trades: 3% base, 5% for engulfing patterns
        # Normal trades: 3% of current balance + bonus from heuristic confluence
        if is_news_event:
            # News event: 5% (high conviction)
            total_pct = min(0.05 + position_size_bonus, 0.05)
            target_position_value = self.capital * total_pct * size_multiplier
        elif is_bounce_engulfing:
            # Engulfing bounce: 5% + bonus, capped at 5%
            total_pct = min(0.05 + position_size_bonus, 0.05)
            target_position_value = self.capital * total_pct * size_multiplier
        elif is_bounce:
            # Bounce trades: 3% + bonus, capped at 5%
            total_pct = min(0.03 + position_size_bonus, 0.05)
            target_position_value = self.capital * total_pct * size_multiplier
        elif is_breach:
            # VWAP trades: 3% + heuristic bonus, capped at 5% (10% if under 15k)
            total_pct = min(0.03 + position_size_bonus, 0.05)
            target_position_value = self.capital * total_pct * size_multiplier
        else:
            base_pct = self.heuristic_config.base_position_pct if self.heuristic_config else 0.03
            total_pct = base_pct + position_size_bonus
            target_position_value = self.capital * total_pct * size_multiplier

        # Determine quality grade based on total_pct (pre-doubling)
        if total_pct >= 0.05:
            quality = "A"
        elif total_pct >= 0.04:
            quality = "B"
        elif total_pct >= 0.03:
            quality = "C"
        else:
            quality = "D"

        # Calculate number of contracts based on option price
        # Option price is per share, contracts are 100 shares
        cost_per_contract = entry_option_price * 100
        if cost_per_contract <= 0:
            # Skip if option price is zero or negative (bad data)
            return False
        num_contracts = max(1, int(target_position_value / cost_per_contract))

        # Actual position value is contracts * cost
        position_value = num_contracts * cost_per_contract

        # Check max daily cost basis limit (can only invest up to starting balance)
        if self.daily_cost_basis + position_value > self.max_daily_cost_basis:
            # Cap to remaining allowed cost basis
            remaining_allowed = self.max_daily_cost_basis - self.daily_cost_basis
            if remaining_allowed <= 0:
                return False
            # Reduce contracts to fit
            num_contracts = max(1, int(remaining_allowed / cost_per_contract))
            position_value = num_contracts * cost_per_contract

        max_daily = self.capital * (self.config.max_daily_allocation_pct / 100)
        if self.daily_allocation + position_value > max_daily:
            return False

        self.position = Position(
            position_type=position_type,
            entry_time=entry_time,
            entry_underlying_price=underlying_price,
            entry_option_price=entry_option_price,
            strike=strike,
            entry_idx=entry_idx,
            position_value=position_value,
            dominant_model=dominant_model,
            max_hold_minutes=max_hold_minutes,
            confluence_count=confluence_count,
            position_size_multiplier=1.0,  # No longer using confidence multiplier
            num_contracts=num_contracts,
            is_breach=is_breach,
            is_bounce=is_bounce,
            is_news_event=is_news_event,
            quality=quality,
        )

        if dominant_model in self.model_contributions:
            self.model_contributions[dominant_model] += 1

        self.daily_trades += 1
        self.daily_allocation += position_value

        # Track cost basis for profit protection
        self.daily_cost_basis += position_value

        return True

    def close_position(
        self,
        exit_time: datetime,
        exit_underlying_price: float,
        exit_reason: str,
        stop_loss_trigger_pct: Optional[float] = None,
    ) -> None:
        if self.position is None:
            return

        # Determine exit price based on exit type
        is_stop_loss = "stop" in exit_reason.lower() or "trailing" in exit_reason.lower()

        if is_stop_loss and stop_loss_trigger_pct is not None:
            # For stop losses: add variable slippage (0-10% of stop loss amount)
            # e.g., 10% stop loss exits between 10-11% loss
            additional_slippage_pct = random.uniform(0, 0.10) * abs(stop_loss_trigger_pct)
            effective_stop_pct = stop_loss_trigger_pct - additional_slippage_pct  # More negative = worse fill
            stop_trigger_price = self.position.entry_option_price * (1 + effective_stop_pct / 100)
            # Apply normal slippage from stop price
            exit_option_price = stop_trigger_price * (1 - self.config.exit_slippage_pct / 100)
        else:
            # Normal exit: use close price
            exit_option_price = self.options.get_option_price(
                timestamp=exit_time,
                strike=self.position.strike,
                position_type=self.position.position_type,
                underlying_price=exit_underlying_price,
                price_type="close",
            )
            if exit_option_price is not None:
                # Add exit slippage
                exit_option_price *= (1 - self.config.exit_slippage_pct / 100)

        if exit_option_price is None:
            exit_option_price = self.position.entry_option_price
            exit_reason = f"{exit_reason}_no_data"

        pnl_pct = (exit_option_price - self.position.entry_option_price) / self.position.entry_option_price * 100

        # Breakeven stops have a tight auto-stop that guarantees 0% minimum
        # (simulates a stop-limit order placed at entry price when breakeven activates)
        if exit_reason == "breakeven_stop" and pnl_pct < 0:
            pnl_pct = 0.0
            exit_option_price = self.position.entry_option_price

        if self.config.max_trade_gain_pct is not None:
            pnl_pct = min(pnl_pct, self.config.max_trade_gain_pct)

        position_value = self.position.position_value
        pnl_dollars = position_value * (pnl_pct / 100)

        spread_cost = position_value * (self.config.spread_cost_pct / 100)
        commission_cost = 2 * self.config.commission_per_contract * self.config.contracts_per_trade
        total_costs = spread_cost + commission_cost

        pnl_dollars -= total_costs

        self.capital += pnl_dollars
        self.peak_capital = max(self.peak_capital, self.capital)
        self.lowest_capital = min(self.lowest_capital, self.capital)
        self.daily_pnl += pnl_dollars
        self.daily_allocation -= position_value
        self.update_drawdown_tracking(exit_time)

        # Track realized profit for profit protection
        self.daily_realized_profit += position_value + pnl_dollars  # Return of capital + profit

        # Check if we've hit 100% profit threshold (only activates once per day)
        # When realized profit >= 2x cost basis, we've doubled our money
        if not self.profit_protection_active and self.daily_cost_basis > 0:
            if self.daily_realized_profit >= 2 * self.daily_cost_basis:
                # Activate profit protection ONCE - set aside original cost basis
                self.profit_protection_active = True
                self.daily_protected_capital = self.daily_cost_basis
                # Continue trading with remaining profits (no further protection)

        holding_minutes = int((exit_time - self.position.entry_time).total_seconds() / 60)

        # Mark exit reason if this was a runner (for unexpected exits like EOD)
        # Runner P&L is from original entry, compare gain since split point
        if self.position.is_runner and not exit_reason.startswith("runner"):
            gain_since_split = pnl_pct - self.position.runner_entry_pnl_pct
            if gain_since_split >= 5.0:
                exit_reason = f"runner_profit_{exit_reason}"
            elif gain_since_split >= 0.0:
                exit_reason = f"runner_breakeven_{exit_reason}"
            else:
                exit_reason = f"runner_loss_{exit_reason}"

        # Build contract name (e.g., "SPY 600C")
        opt_type = "C" if self.position.position_type == PositionType.CALL else "P"
        contract = f"SPY {int(self.position.strike)}{opt_type}"

        # Position entry_time and entry_option_price are already original values for runners
        trade = Trade(
            position_type=self.position.position_type,
            entry_time=self.position.entry_time,
            exit_time=exit_time,
            entry_underlying=self.position.entry_underlying_price,
            exit_underlying=exit_underlying_price,
            entry_option_price=self.position.entry_option_price,
            exit_option_price=exit_option_price,
            pnl_pct=pnl_pct,
            pnl_dollars=pnl_dollars,
            capital_after=self.capital,
            exit_reason=exit_reason,
            holding_minutes=holding_minutes,
            dominant_model=self.position.dominant_model,
            confluence_count=self.position.confluence_count,
            position_size_multiplier=self.position.position_size_multiplier,
            strike=self.position.strike,
            contract=contract,
            num_contracts=self.position.num_contracts,
            is_runner=self.position.is_runner,
            parent_entry_time=self.position.parent_entry_time,
            parent_entry_option_price=self.position.parent_entry_option_price,
            quality=self.position.quality,
        )
        self.trades.append(trade)
        self.position = None

    def convert_to_runner(self, current_time: datetime, current_price: float) -> None:
        """
        Convert current position to a runner (leave profits running).

        This records TWO trades:
        1. Main position exit at "runner_split" - captures the +20%+ profit
        2. Runner position that continues from here - starts at 0% P&L
        """
        if self.position is None:
            return

        current_option_price = self.options.get_option_price(
            timestamp=current_time,
            strike=self.position.strike,
            position_type=self.position.position_type,
            underlying_price=current_price,
        )

        if current_option_price is None:
            return

        # Calculate P&L for the main position
        pnl_pct = (current_option_price - self.position.entry_option_price) / self.position.entry_option_price * 100

        # Apply exit slippage to the split exit
        exit_option_price = current_option_price * (1 - self.config.exit_slippage_pct / 100)
        pnl_pct_with_slippage = (exit_option_price - self.position.entry_option_price) / self.position.entry_option_price * 100

        position_value = self.position.position_value

        # Determine runner percentage based on gain level:
        # - 100%+ gain: runner is 50% of position
        # - 50-100% gain: runner is 30% of position
        # - 20-50% gain: runner is 20% of position
        if pnl_pct >= 100.0:
            runner_pct = 0.50
        elif pnl_pct >= 50.0:
            runner_pct = 0.30
        else:  # 20-50%
            runner_pct = 0.20

        exit_pct = 1.0 - runner_pct  # Percentage we're exiting now
        exit_position_value = position_value * exit_pct

        full_pnl_dollars = position_value * (pnl_pct_with_slippage / 100)

        # Costs for the main position exit (exit portion)
        spread_cost = exit_position_value * (self.config.spread_cost_pct / 100)
        exit_contracts = int(self.position.num_contracts * exit_pct)
        if exit_contracts == 0:
            exit_contracts = 1
        commission_cost = self.config.commission_per_contract * exit_contracts
        total_costs = spread_cost + commission_cost
        pnl_dollars = (full_pnl_dollars * exit_pct) - total_costs

        # Update capital for main position exit (exit portion)
        self.capital += pnl_dollars
        self.peak_capital = max(self.peak_capital, self.capital)
        self.lowest_capital = min(self.lowest_capital, self.capital)
        self.daily_pnl += pnl_dollars
        self.daily_allocation -= exit_position_value  # Exit portion exits
        self.update_drawdown_tracking(current_time)

        # Track realized profit (only from the exited portion)
        self.daily_realized_profit += exit_position_value + pnl_dollars

        holding_minutes = int((current_time - self.position.entry_time).total_seconds() / 60)

        # Build contract name
        opt_type = "C" if self.position.position_type == PositionType.CALL else "P"
        contract = f"SPY {int(self.position.strike)}{opt_type}"

        # Store original entry time and price for the runner (so chart and table show original entry)
        original_entry_time = self.position.parent_entry_time or self.position.entry_time
        original_entry_option_price = self.position.parent_entry_option_price or self.position.entry_option_price

        # Split contracts based on gain-dependent runner percentage
        runner_contracts = int(self.position.num_contracts * runner_pct)
        if runner_contracts == 0:
            runner_contracts = 1
        main_contracts = self.position.num_contracts - runner_contracts
        if main_contracts == 0:
            main_contracts = 1
            runner_contracts = max(1, self.position.num_contracts - 1)

        # Record the main position exit as "runner_split" (half the position)
        # pnl_dollars already accounts for half position + costs from above
        main_trade = Trade(
            position_type=self.position.position_type,
            entry_time=self.position.entry_time,
            exit_time=current_time,
            entry_underlying=self.position.entry_underlying_price,
            exit_underlying=current_price,
            entry_option_price=self.position.entry_option_price,
            exit_option_price=exit_option_price,
            pnl_pct=pnl_pct_with_slippage,
            pnl_dollars=pnl_dollars,  # Already half from capital calculation
            capital_after=self.capital,
            exit_reason="runner_split",
            holding_minutes=holding_minutes,
            dominant_model=self.position.dominant_model,
            confluence_count=self.position.confluence_count,
            position_size_multiplier=self.position.position_size_multiplier,
            strike=self.position.strike,
            contract=contract,
            num_contracts=main_contracts,
            is_runner=False,
            parent_entry_time=original_entry_time,
            quality=self.position.quality,
        )
        self.trades.append(main_trade)

        # Now create a NEW runner position starting from here
        # Runner uses ORIGINAL entry price for P&L calculation (not split price)
        # Runner gets the percentage based on gain level (50%, 30%, or 20%)
        runner_position_value = position_value * runner_pct

        # Update daily allocation for runner entry
        self.daily_allocation += runner_position_value
        self.daily_cost_basis += runner_position_value

        # Save position attributes before overwriting
        old_position = self.position

        # Calculate current P&L% from original entry (for runner peak tracking)
        current_pnl_from_original = (current_option_price - original_entry_option_price) / original_entry_option_price * 100

        self.position = Position(
            position_type=old_position.position_type,
            entry_time=original_entry_time,  # Use original entry time
            entry_underlying_price=old_position.entry_underlying_price,  # Original underlying price
            entry_option_price=original_entry_option_price,  # Use ORIGINAL entry price for P&L calc
            strike=old_position.strike,
            entry_idx=old_position.entry_idx,  # Keep for reference
            position_value=runner_position_value,
            dominant_model=old_position.dominant_model,
            max_hold_minutes=15,  # Runner time barrier
            barrier_start_time=current_time,  # But barrier starts from split time
            confluence_count=old_position.confluence_count,
            position_size_multiplier=old_position.position_size_multiplier,
            is_runner=True,
            runner_start_time=current_time,
            runner_max_hold_minutes=15,
            runner_peak_pnl_pct=current_pnl_from_original,  # Start tracking from current P&L
            runner_entry_pnl_pct=current_pnl_from_original,  # P&L when runner started
            parent_entry_time=original_entry_time,  # Track original entry for chart
            parent_entry_option_price=original_entry_option_price,  # Track original entry price
            num_contracts=runner_contracts,  # Half the contracts go to runner
            quality=old_position.quality,
        )

    def take_partial_profit(
        self,
        current_time: datetime,
        current_price: float,
        exit_reason: str,
        partial_pct: float = 0.20,
    ) -> None:
        """
        Take partial profit on a position (e.g., 20% at +1sigma for VWAP trades).

        Records a trade for the partial exit and reduces position size accordingly.
        """
        if self.position is None:
            return

        current_option_price = self.options.get_option_price(
            timestamp=current_time,
            strike=self.position.strike,
            position_type=self.position.position_type,
            underlying_price=current_price,
        )

        if current_option_price is None:
            return

        # Calculate P&L for the full position (before split)
        pnl_pct = (current_option_price - self.position.entry_option_price) / self.position.entry_option_price * 100

        # Apply exit slippage
        exit_option_price = current_option_price * (1 - self.config.exit_slippage_pct / 100)
        pnl_pct_with_slippage = (exit_option_price - self.position.entry_option_price) / self.position.entry_option_price * 100

        position_value = self.position.position_value
        partial_position_value = position_value * partial_pct
        remaining_position_value = position_value * (1 - partial_pct)

        # Calculate P&L dollars for the partial exit
        full_pnl_dollars = position_value * (pnl_pct_with_slippage / 100)
        partial_pnl_dollars = full_pnl_dollars * partial_pct

        # Costs for the partial exit
        spread_cost = partial_position_value * (self.config.spread_cost_pct / 100)
        partial_contracts = int(self.position.num_contracts * partial_pct)
        if partial_contracts == 0:
            partial_contracts = 1
        commission_cost = self.config.commission_per_contract * partial_contracts
        total_costs = spread_cost + commission_cost
        pnl_dollars = partial_pnl_dollars - total_costs

        # Update capital for partial exit
        self.capital += pnl_dollars
        self.peak_capital = max(self.peak_capital, self.capital)
        self.lowest_capital = min(self.lowest_capital, self.capital)
        self.daily_pnl += pnl_dollars
        self.daily_allocation -= partial_position_value  # Partial exit reduces allocation
        self.update_drawdown_tracking(current_time)

        # Track realized profit
        self.daily_realized_profit += partial_position_value + pnl_dollars

        holding_minutes = int((current_time - self.position.entry_time).total_seconds() / 60)

        # Build contract name
        opt_type = "C" if self.position.position_type == PositionType.CALL else "P"
        contract = f"SPY {int(self.position.strike)}{opt_type}"

        # Record the partial exit trade
        partial_trade = Trade(
            position_type=self.position.position_type,
            entry_time=self.position.entry_time,
            exit_time=current_time,
            entry_underlying=self.position.entry_underlying_price,
            exit_underlying=current_price,
            entry_option_price=self.position.entry_option_price,
            exit_option_price=exit_option_price,
            pnl_pct=pnl_pct_with_slippage,
            pnl_dollars=pnl_dollars,
            capital_after=self.capital,
            exit_reason=exit_reason,
            holding_minutes=holding_minutes,
            dominant_model=self.position.dominant_model,
            confluence_count=self.position.confluence_count,
            position_size_multiplier=self.position.position_size_multiplier,
            strike=self.position.strike,
            contract=contract,
            num_contracts=partial_contracts,
            is_runner=False,
            parent_entry_time=self.position.parent_entry_time or self.position.entry_time,
            quality=self.position.quality,
        )
        self.trades.append(partial_trade)

        # Update position to reflect remaining size
        remaining_contracts = self.position.num_contracts - partial_contracts
        if remaining_contracts < 1:
            remaining_contracts = 1  # Keep at least 1 contract

        self.position.position_value = remaining_position_value
        self.position.num_contracts = remaining_contracts
        self.position.took_1sd_partial = True  # Mark that we've taken the +1sigma partial
