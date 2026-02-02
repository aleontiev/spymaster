"""
Exit rules for position management.

Contains all exit logic including stop loss, trailing stop, time barrier,
breakeven, and runner conversion - extracted from the backtester for reuse
in live trading.
"""
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Optional, Tuple

from src.backtest.types import Position, PositionType
from src.strategy.fusion_config import FusedSignal, Signal


class ExitReason(Enum):
    """Reasons for exiting a position."""
    # Stop losses
    STOP_LOSS = "stop_loss"
    BREAKEVEN_STOP = "breakeven_stop"
    VWAP_BREAKEVEN_STOP = "vwap_breakeven_stop"
    REVERSAL_BREAKEVEN_STOP = "reversal_breakeven_stop"
    TRAILING_STOP = "trailing_stop"

    # Signal-based
    SIGNAL_REVERSAL = "signal_reversal"

    # Time-based
    TIME_BARRIER = "time_barrier"
    EOD_CLOSE = "eod_close"

    # Runner conversions
    CONVERT_TO_RUNNER = "convert_to_runner"

    # Runner exits
    RUNNER_BELOW_SPLIT = "runner_below_split"
    RUNNER_PROFIT_TIME = "runner_profit_time"
    RUNNER_BREAKEVEN_TIME = "runner_breakeven_time"
    RUNNER_LOSS_TIME = "runner_loss_time"
    RUNNER_PROFIT_TRAILING = "runner_profit_trailing"
    RUNNER_BREAKEVEN_TRAILING = "runner_breakeven_trailing"
    RUNNER_LOSS_TRAILING = "runner_loss_trailing"
    RUNNER_STOP_LOSS = "runner_stop_loss"
    RUNNER_PROFIT_REVERSAL = "runner_profit_reversal"
    RUNNER_BREAKEVEN_REVERSAL = "runner_breakeven_reversal"
    RUNNER_LOSS_REVERSAL = "runner_loss_reversal"


@dataclass
class ExitConfig:
    """Configuration for exit rules."""
    stop_loss_pct: float = 10.0
    breakeven_activation_pct: float = 7.5
    breach_breakeven_activation_pct: float = 10.0
    reversal_breakeven_activation_pct: float = 7.0
    reversal_breakeven_floor_pct: float = 1.0
    trailing_stop_distance_pct: float = 5.0
    runner_trailing_stop_pct: float = 10.0
    runner_max_hold_minutes: int = 15
    min_profit_for_runner_pct: float = 20.0
    breach_breakeven_floor_pct: float = 2.0


@dataclass
class ExitDecision:
    """Result of exit rule evaluation."""
    should_exit: bool
    exit_reason: Optional[str] = None
    should_renew: bool = False
    new_max_hold_minutes: int = 0
    convert_to_runner: bool = False
    stop_trigger_pct: Optional[float] = None  # For calculating realistic fill price


class ExitRuleEngine:
    """
    Evaluates exit rules for open positions.

    Exit rules (in priority order):
    1. Fixed stop loss (10% default)
    2. Breakeven stop (trailing from peak after activation)
    3. Signal reversal (opposite signal from models)
    4. Signal confirmation (reset time barrier)
    5. Time barrier (max hold time reached)

    Runner rules (after conversion):
    1. Below split point
    2. Runner time barrier
    3. Runner trailing stop
    4. Runner fixed stop loss
    5. Signal reversal
    """

    def __init__(self, config: Optional[ExitConfig] = None):
        """
        Initialize the exit rule engine.

        Args:
            config: Exit rule configuration
        """
        self.config = config or ExitConfig()

    def evaluate(
        self,
        position: Position,
        current_option_price: float,
        fused_signal: FusedSignal,
        current_time: datetime,
        upper_2sd: Optional[float] = None,
        lower_2sd: Optional[float] = None,
        upper_1sd: Optional[float] = None,
        lower_1sd: Optional[float] = None,
    ) -> ExitDecision:
        """
        Evaluate all exit rules for the current position.

        Args:
            position: Current open position
            current_option_price: Current option price
            fused_signal: Current fused signal from models
            current_time: Current timestamp
            upper_2sd: +2σ VWAP band (for take profit)
            lower_2sd: -2σ VWAP band (for take profit)
            upper_1sd: +1σ VWAP band (for partial take profit)
            lower_1sd: -1σ VWAP band (for partial take profit)

        Returns:
            ExitDecision with exit details
        """
        if current_option_price is None or current_option_price <= 0:
            return ExitDecision(should_exit=False)

        # Calculate P&L percentage
        pnl_pct = (current_option_price - position.entry_option_price) / position.entry_option_price * 100

        # Update peak P&L tracking (done externally in live trading)
        if not position.is_runner:
            if pnl_pct > position.peak_pnl_pct:
                position.peak_pnl_pct = pnl_pct
        else:
            if pnl_pct > position.runner_peak_pnl_pct:
                position.runner_peak_pnl_pct = pnl_pct

        # Handle runner exits separately
        if position.is_runner:
            return self._evaluate_runner_exit(position, pnl_pct, fused_signal, current_time)

        # Normal position exit logic
        return self._evaluate_normal_exit(position, pnl_pct, fused_signal, current_time)

    def _evaluate_normal_exit(
        self,
        position: Position,
        pnl_pct: float,
        fused_signal: FusedSignal,
        current_time: datetime,
    ) -> ExitDecision:
        """Evaluate exit rules for normal (non-runner) positions."""

        # 1. Fixed stop loss
        if not position.breakeven_activated:
            if pnl_pct <= -self.config.stop_loss_pct:
                return ExitDecision(
                    should_exit=True,
                    exit_reason=ExitReason.STOP_LOSS.value,
                    stop_trigger_pct=-self.config.stop_loss_pct,
                )

            # Check breakeven activation
            if position.is_reversal or position.is_bounce:
                breakeven_threshold = self.config.reversal_breakeven_activation_pct
            elif position.is_breach:
                breakeven_threshold = self.config.breach_breakeven_activation_pct
            else:
                breakeven_threshold = self.config.breakeven_activation_pct
            if pnl_pct >= breakeven_threshold:
                position.breakeven_activated = True

        # 2. Breakeven stop (trailing from peak)
        if position.breakeven_activated:
            # Reversal/bounce trades: tight breakeven floor at 1%
            if (position.is_reversal or position.is_bounce) and pnl_pct <= self.config.reversal_breakeven_floor_pct:
                return ExitDecision(
                    should_exit=True,
                    exit_reason=ExitReason.REVERSAL_BREAKEVEN_STOP.value,
                    stop_trigger_pct=self.config.reversal_breakeven_floor_pct,
                )

            # VWAP breach trades have a hard breakeven floor
            if position.is_breach and not position.is_reversal and not position.is_bounce and pnl_pct <= self.config.breach_breakeven_floor_pct:
                return ExitDecision(
                    should_exit=True,
                    exit_reason=ExitReason.VWAP_BREAKEVEN_STOP.value,
                    stop_trigger_pct=self.config.breach_breakeven_floor_pct,
                )

            trailing_stop = position.peak_pnl_pct - self.config.trailing_stop_distance_pct
            if pnl_pct <= trailing_stop:
                # Check if we should convert to runner (+20% or more)
                if pnl_pct >= self.config.min_profit_for_runner_pct:
                    return ExitDecision(
                        should_exit=False,
                        convert_to_runner=True,
                        exit_reason=ExitReason.CONVERT_TO_RUNNER.value,
                    )
                elif pnl_pct <= 2.5:
                    return ExitDecision(
                        should_exit=True,
                        exit_reason=ExitReason.BREAKEVEN_STOP.value,
                        stop_trigger_pct=trailing_stop,
                    )
                else:
                    return ExitDecision(
                        should_exit=True,
                        exit_reason=ExitReason.TRAILING_STOP.value,
                        stop_trigger_pct=trailing_stop,
                    )

        # 3. Signal reversal
        if self._is_signal_reversal(position, fused_signal):
            if pnl_pct >= self.config.min_profit_for_runner_pct:
                return ExitDecision(
                    should_exit=False,
                    convert_to_runner=True,
                    exit_reason=ExitReason.CONVERT_TO_RUNNER.value,
                )
            return ExitDecision(
                should_exit=True,
                exit_reason=ExitReason.SIGNAL_REVERSAL.value,
            )

        # 4. Signal confirmation - reset barriers
        if self._is_signal_confirmation(position, fused_signal):
            return ExitDecision(
                should_exit=False,
                should_renew=True,
                new_max_hold_minutes=fused_signal.exit_horizon_minutes,
            )

        # 5. Time barrier
        time_since_barrier_start = (current_time - position.barrier_start_time).total_seconds() / 60
        if time_since_barrier_start >= position.max_hold_minutes:
            if pnl_pct >= self.config.min_profit_for_runner_pct:
                return ExitDecision(
                    should_exit=False,
                    convert_to_runner=True,
                    exit_reason=ExitReason.CONVERT_TO_RUNNER.value,
                )
            return ExitDecision(
                should_exit=True,
                exit_reason=ExitReason.TIME_BARRIER.value,
            )

        return ExitDecision(should_exit=False)

    def _evaluate_runner_exit(
        self,
        position: Position,
        pnl_pct: float,
        fused_signal: FusedSignal,
        current_time: datetime,
    ) -> ExitDecision:
        """Evaluate exit rules for runner positions."""
        split_pnl = position.runner_entry_pnl_pct
        gain_since_split = pnl_pct - split_pnl

        # 1. Close if runner falls below the original position's profitability
        if pnl_pct < split_pnl:
            return ExitDecision(
                should_exit=True,
                exit_reason=ExitReason.RUNNER_BELOW_SPLIT.value,
            )

        # 2. Runner time barrier
        runner_time = (current_time - position.runner_start_time).total_seconds() / 60
        if runner_time >= position.runner_max_hold_minutes:
            if gain_since_split >= 5.0:
                reason = ExitReason.RUNNER_PROFIT_TIME.value
            elif gain_since_split >= 0.0:
                reason = ExitReason.RUNNER_BREAKEVEN_TIME.value
            else:
                reason = ExitReason.RUNNER_LOSS_TIME.value
            return ExitDecision(should_exit=True, exit_reason=reason)

        # 3. Runner trailing stop - trail by 10% from runner peak
        runner_stop = position.runner_peak_pnl_pct - self.config.runner_trailing_stop_pct
        runner_stop = max(runner_stop, split_pnl)  # Never give back more than split gains
        peak_gain_since_split = position.runner_peak_pnl_pct - split_pnl

        if pnl_pct <= runner_stop and peak_gain_since_split >= 5.0:
            if gain_since_split >= 5.0:
                reason = ExitReason.RUNNER_PROFIT_TRAILING.value
            elif gain_since_split >= 0.0:
                reason = ExitReason.RUNNER_BREAKEVEN_TRAILING.value
            else:
                reason = ExitReason.RUNNER_LOSS_TRAILING.value
            return ExitDecision(
                should_exit=True,
                exit_reason=reason,
                stop_trigger_pct=runner_stop,
            )

        # 4. Fixed stop loss for runner - if we give back 10% from split point
        if gain_since_split <= -10.0:
            return ExitDecision(
                should_exit=True,
                exit_reason=ExitReason.RUNNER_STOP_LOSS.value,
                stop_trigger_pct=split_pnl - 10.0,
            )

        # 5. Signal reversal closes runner
        if self._is_signal_reversal(position, fused_signal):
            if gain_since_split >= 5.0:
                reason = ExitReason.RUNNER_PROFIT_REVERSAL.value
            elif gain_since_split >= 0.0:
                reason = ExitReason.RUNNER_BREAKEVEN_REVERSAL.value
            else:
                reason = ExitReason.RUNNER_LOSS_REVERSAL.value
            return ExitDecision(should_exit=True, exit_reason=reason)

        return ExitDecision(should_exit=False)

    def _is_signal_reversal(self, position: Position, fused_signal: FusedSignal) -> bool:
        """Check if the fused signal indicates a reversal."""
        if position.position_type == PositionType.CALL and fused_signal.action == Signal.SHORT:
            return True
        if position.position_type == PositionType.PUT and fused_signal.action == Signal.LONG:
            return True
        return False

    def _is_signal_confirmation(self, position: Position, fused_signal: FusedSignal) -> bool:
        """Check if the fused signal confirms the position direction."""
        if position.position_type == PositionType.CALL and fused_signal.action == Signal.LONG:
            return True
        if position.position_type == PositionType.PUT and fused_signal.action == Signal.SHORT:
            return True
        return False

    def check_stop_loss_immediate(
        self,
        position: Position,
        current_option_price: float,
    ) -> Tuple[bool, Optional[str]]:
        """
        Check stop loss without full exit evaluation.

        This is used for real-time price monitoring to quickly detect
        if a stop loss has been breached mid-minute.

        Args:
            position: Current position
            current_option_price: Current option price

        Returns:
            Tuple of (should_stop, exit_reason)
        """
        if current_option_price is None or current_option_price <= 0:
            return False, None

        pnl_pct = (current_option_price - position.entry_option_price) / position.entry_option_price * 100

        if position.is_runner:
            split_pnl = position.runner_entry_pnl_pct
            gain_since_split = pnl_pct - split_pnl

            # Runner below split point
            if pnl_pct < split_pnl:
                return True, ExitReason.RUNNER_BELOW_SPLIT.value

            # Runner fixed stop loss
            if gain_since_split <= -10.0:
                return True, ExitReason.RUNNER_STOP_LOSS.value

            # Runner trailing stop
            runner_stop = position.runner_peak_pnl_pct - self.config.runner_trailing_stop_pct
            runner_stop = max(runner_stop, split_pnl)
            peak_gain_since_split = position.runner_peak_pnl_pct - split_pnl
            if pnl_pct <= runner_stop and peak_gain_since_split >= 5.0:
                return True, ExitReason.RUNNER_PROFIT_TRAILING.value
        else:
            # Fixed stop loss
            if not position.breakeven_activated and pnl_pct <= -self.config.stop_loss_pct:
                return True, ExitReason.STOP_LOSS.value

            # Breakeven/trailing stop
            if position.breakeven_activated:
                if (position.is_reversal or position.is_bounce) and pnl_pct <= self.config.reversal_breakeven_floor_pct:
                    return True, ExitReason.REVERSAL_BREAKEVEN_STOP.value

                if position.is_breach and not position.is_reversal and not position.is_bounce and pnl_pct <= self.config.breach_breakeven_floor_pct:
                    return True, ExitReason.VWAP_BREAKEVEN_STOP.value

                trailing_stop = position.peak_pnl_pct - self.config.trailing_stop_distance_pct
                if pnl_pct <= trailing_stop:
                    return True, ExitReason.TRAILING_STOP.value

        return False, None
