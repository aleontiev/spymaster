"""
Multi-scale percentile policy fusion backtester.

Combines multiple independently-trained percentile policy models into a single
unified trading bot using voting fusion.
"""
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from src.backtest.types import (
    BacktestConfig,
    PendingEntry,
    Position,
    PositionType,
    Trade,
)
from src.execution.heuristic import HeuristicModel
from src.execution.options_provider import RealOptionsProvider
from src.strategy.fusion_config import FusedSignal, Signal


class MultiPercentileBacktester:
    """Backtest multi-scale percentile policy fusion."""

    def __init__(
        self,
        executor,  # Can be MultiPercentileExecutor or FilteredExecutor
        config: BacktestConfig,
        device: torch.device,
        use_heuristic: bool = True,
    ):
        self.executor = executor
        self.config = config
        self.device = device
        self.options = RealOptionsProvider()
        self.use_heuristic = use_heuristic
        self.heuristic_model = HeuristicModel(lookback_minutes=30) if use_heuristic else None

        model_info = executor.get_model_info()
        self.max_context_len = max(info["context_len"] for info in model_info.values()) if model_info else 60
        self.context_lens = {name: info["context_len"] for name, info in model_info.items()}

        self.position: Optional[Position] = None
        self.trades: List[Trade] = []
        self.capital = config.initial_capital
        self.peak_capital = config.initial_capital
        self.model_contributions: Dict[str, int] = {"15m": 0, "5m": 0, "1m": 0}

        # Profit protection tracking (per day)
        self.daily_cost_basis = 0.0  # Total invested today
        self.daily_realized_profit = 0.0  # Total profit realized today
        self.daily_protected_capital = 0.0  # Capital "set aside" after 100% profit
        self.profit_protection_active = False  # Trading with profits only (activates once)
        self.max_daily_cost_basis = 0.0  # Max we can invest today (= starting balance)

        # Track heuristic rejections for stats
        self.heuristic_rejections = 0

        # Pending entry (to be executed on next minute's open)
        self.pending_entry: Optional[PendingEntry] = None

    def reset(self, starting_capital: Optional[float] = None):
        """Reset state for new backtest."""
        self.position = None
        self.trades = []
        capital = starting_capital if starting_capital is not None else self.config.initial_capital
        self.capital = capital
        self.peak_capital = capital
        self.daily_trades = 0
        self.daily_allocation = 0.0
        self.daily_pnl = 0.0
        self.daily_start_capital = capital
        self.current_day = None
        self.model_contributions = {"15m": 0, "5m": 0, "1m": 0}

        # Profit protection reset
        self.daily_cost_basis = 0.0
        self.daily_realized_profit = 0.0
        self.daily_protected_capital = 0.0
        self.profit_protection_active = False
        self.max_daily_cost_basis = capital  # Can only invest up to starting balance

        # Heuristic tracking
        self.heuristic_rejections = 0

        # Pending entry
        self.pending_entry = None

    def _get_minutes_elapsed(self, timestamp: datetime) -> int:
        et_time = timestamp.tz_convert("America/New_York")
        minutes_since_midnight = et_time.hour * 60 + et_time.minute
        market_open = 9 * 60 + 30
        return minutes_since_midnight - market_open

    def _build_contexts(
        self,
        feature_tensor: torch.Tensor,
        idx: int,
    ) -> Dict[str, torch.Tensor]:
        contexts = {}
        for name, context_len in self.context_lens.items():
            if idx >= context_len:
                context = feature_tensor[idx - context_len:idx].unsqueeze(0)
                contexts[name] = context
        return contexts

    def check_exit_or_renew(
        self,
        current_price: float,
        current_time: datetime,
        fused_signal: FusedSignal,
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

            # Runner trailing stop - trail by 5% from runner peak
            # Minimum stop is at split P&L (protect the locked-in gains)
            runner_stop = self.position.runner_peak_pnl_pct - 5.0
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
            if pnl_pct >= 5.0:
                self.position.breakeven_activated = True

        # 2. Breakeven stop (trailing from peak)
        # Breakeven threshold is +2% so "breakeven" trades close slightly positive
        if self.position.breakeven_activated:
            trailing_stop = self.position.peak_pnl_pct - 5.0
            if pnl_pct <= trailing_stop:
                # Check if we should convert to runner (+20% or more)
                if pnl_pct >= 20.0:
                    return "convert_to_runner", False, 0
                elif pnl_pct <= 2.0:
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

    def run_backtest(
        self,
        df: pd.DataFrame,
        raw_closes: np.ndarray,
        year: int,
        month: Optional[int] = None,
        starting_capital: Optional[float] = None,
        ohlcv_df: Optional[pd.DataFrame] = None,
    ) -> Dict:
        """Run backtest for a specific year and optionally month."""
        self.reset(starting_capital)

        if month is not None:
            mask = (df.index.year == year) & (df.index.month == month)
        else:
            mask = df.index.year == year

        period_df = df[mask].copy()
        period_closes = raw_closes[mask]
        period_ohlcv = ohlcv_df[mask].copy() if ohlcv_df is not None else None

        if len(period_df) < self.max_context_len + 1:
            return {"year": year, "month": month, "trades": 0, "error": "Insufficient data"}

        features = period_df.values
        feature_tensor = torch.tensor(features, dtype=torch.float32, device=self.device)

        current_day = None
        desc = f"Year {year}" if month is None else f"{year}-{month:02d}"

        for i in tqdm(range(self.max_context_len, len(period_df)), desc=desc, leave=False):
            current_time = period_df.index[i]
            current_price = period_closes[i]

            et_time = current_time.tz_convert("America/New_York")
            et_hour = et_time.hour
            et_minute = et_time.minute

            if et_hour < 9 or (et_hour == 9 and et_minute < 30):
                continue

            # EOD handling: close positions at 4pm, but 15-min rule only applies to OPENING
            if et_hour >= 16:
                self.pending_entry = None  # Cancel any pending entries
                if self.position is not None:
                    self._close_position(current_time, current_price, "eod_close")
                continue

            # Check if we're in the last 15 minutes (no new positions, but can hold existing)
            is_last_15_minutes = (et_hour == 15 and et_minute >= 45)

            trading_date = et_time.date()
            if current_day != trading_date:
                current_day = trading_date
                self.options.load_date(trading_date)

            minutes_elapsed = self._get_minutes_elapsed(current_time)
            contexts = self._build_contexts(feature_tensor, i)
            fused_signal = self.executor.get_signal(contexts, minutes_elapsed)

            # Execute pending entry at OPEN of this minute (signal was from previous minute)
            if self.pending_entry is not None and self.position is None:
                pe = self.pending_entry
                self.pending_entry = None
                # Execute at this minute's open price
                self._open_position(
                    pe.position_type,
                    current_time,  # Execute at current time (open of this minute)
                    current_price,
                    i,
                    pe.dominant_model,
                    pe.max_hold_minutes,
                    pe.confidence,
                    pe.confluence_count,
                    pe.agreeing_models,
                )

            if self.position is not None:
                exit_reason, should_renew, new_max_hold = self.check_exit_or_renew(
                    current_price, current_time, fused_signal
                )
                if exit_reason:
                    if exit_reason == "convert_to_runner":
                        # Convert position to runner instead of closing
                        self._convert_to_runner(current_time, current_price)
                    else:
                        # Calculate stop loss trigger level for realistic fills
                        stop_trigger_pct = None
                        if "stop_loss" in exit_reason:
                            stop_trigger_pct = -self.config.stop_loss_pct
                        elif "trailing" in exit_reason or "breakeven" in exit_reason:
                            stop_trigger_pct = self.position.peak_pnl_pct - 5.0
                        self._close_position(current_time, current_price, exit_reason, stop_trigger_pct)
                elif should_renew:
                    self.position.barrier_start_time = current_time
                    self.position.max_hold_minutes = new_max_hold
                    self.position.renewals += 1

            # Create pending entry (to be executed on NEXT minute's open)
            if self.position is None and self.pending_entry is None and self.executor.is_trading_allowed(minutes_elapsed) and not is_last_15_minutes:
                # Apply heuristic filter if enabled
                final_action = fused_signal.action
                if self.use_heuristic and self.heuristic_model is not None and period_ohlcv is not None:
                    if fused_signal.action != Signal.HOLD:
                        # Get last 30 minutes of OHLCV for heuristic
                        lookback = min(30, i)
                        ohlcv_window = period_ohlcv.iloc[i - lookback:i + 1]
                        final_action = self.heuristic_model.get_signal(ohlcv_window, fused_signal.action)
                        if final_action == Signal.HOLD and fused_signal.action != Signal.HOLD:
                            self.heuristic_rejections += 1

                confluence_count = len(fused_signal.agreeing_models)
                if final_action == Signal.LONG:
                    # Queue entry for next minute's open
                    self.pending_entry = PendingEntry(
                        position_type=PositionType.CALL,
                        signal_time=current_time,
                        underlying_price=current_price,
                        signal_idx=i,
                        dominant_model=fused_signal.dominant_model,
                        max_hold_minutes=fused_signal.exit_horizon_minutes,
                        confidence=fused_signal.confidence,
                        confluence_count=confluence_count,
                        agreeing_models=fused_signal.agreeing_models,
                    )
                elif final_action == Signal.SHORT:
                    # Queue entry for next minute's open
                    self.pending_entry = PendingEntry(
                        position_type=PositionType.PUT,
                        signal_time=current_time,
                        underlying_price=current_price,
                        signal_idx=i,
                        dominant_model=fused_signal.dominant_model,
                        max_hold_minutes=fused_signal.exit_horizon_minutes,
                        confidence=fused_signal.confidence,
                        confluence_count=confluence_count,
                        agreeing_models=fused_signal.agreeing_models,
                    )

        if self.position is not None:
            self._close_position(
                period_df.index[-1],
                period_closes[-1],
                "backtest_end"
            )

        return self._compute_metrics(year, month)

    def run_backtest_with_signals(
        self,
        df: pd.DataFrame,
        raw_closes: np.ndarray,
        year: int,
        month: Optional[int],
        full_executor,  # MultiPercentileExecutor
        starting_capital: Optional[float] = None,
        ohlcv_df: Optional[pd.DataFrame] = None,
    ) -> Tuple[Dict, Dict[str, List]]:
        """Run backtest and track individual model signals for HTML report."""
        from src.backtest.html_report import SignalMarker

        self.reset(starting_capital)

        if month is not None:
            mask = (df.index.year == year) & (df.index.month == month)
        else:
            mask = df.index.year == year

        period_df = df[mask].copy()
        period_closes = raw_closes[mask]
        period_ohlcv = ohlcv_df[mask].copy() if ohlcv_df is not None else None

        if len(period_df) < self.max_context_len + 1:
            return {"year": year, "month": month, "trades": 0, "error": "Insufficient data"}, {}

        features = period_df.values
        feature_tensor = torch.tensor(features, dtype=torch.float32, device=self.device)

        # Track signals from all models
        signals = {"1m": [], "5m": [], "15m": []}
        current_day = None
        desc = f"{year}-{month:02d}" if month else f"Year {year}"

        # Context lengths for each model
        all_context_lens = {"1m": 5, "5m": 15, "15m": 60}

        for i in tqdm(range(self.max_context_len, len(period_df)), desc=desc, leave=False):
            current_time = period_df.index[i]
            current_price = period_closes[i]

            et_time = current_time.tz_convert("America/New_York")
            et_hour = et_time.hour
            et_minute = et_time.minute

            if et_hour < 9 or (et_hour == 9 and et_minute < 30):
                continue

            # EOD handling: close positions at 4pm, but 15-min rule only applies to OPENING
            if et_hour >= 16:
                self.pending_entry = None  # Cancel any pending entries
                if self.position is not None:
                    self._close_position(current_time, current_price, "eod_close")
                continue

            # Check if we're in the last 15 minutes (no new positions, but can hold existing)
            is_last_15_minutes = (et_hour == 15 and et_minute >= 45)

            trading_date = et_time.date()
            if current_day != trading_date:
                current_day = trading_date
                self.options.load_date(trading_date)

            minutes_elapsed = self._get_minutes_elapsed(current_time)
            contexts = self._build_contexts(feature_tensor, i)
            fused_signal = self.executor.get_signal(contexts, minutes_elapsed)

            # Get individual signals from all models for HTML report
            for model_name in ["1m", "5m", "15m"]:
                ctx_len = all_context_lens[model_name]
                if i >= ctx_len:
                    model_context = feature_tensor[i - ctx_len:i].unsqueeze(0)
                    try:
                        ind_signal, ind_probs = full_executor.get_individual_signal(model_name, model_context)
                        if ind_signal != Signal.HOLD:
                            signals[model_name].append(SignalMarker(
                                timestamp=int(current_time.timestamp()),
                                signal_type="long" if ind_signal == Signal.LONG else "short",
                                model=model_name,
                                price=current_price,
                            ))
                    except Exception:
                        pass  # Skip if model unavailable

            # Execute pending entry at OPEN of this minute (signal was from previous minute)
            if self.pending_entry is not None and self.position is None:
                pe = self.pending_entry
                self.pending_entry = None
                # Execute at this minute's open price
                self._open_position(
                    pe.position_type,
                    current_time,  # Execute at current time (open of this minute)
                    current_price,
                    i,
                    pe.dominant_model,
                    pe.max_hold_minutes,
                    pe.confidence,
                    pe.confluence_count,
                    pe.agreeing_models,
                )

            # Normal backtest logic
            if self.position is not None:
                exit_reason, should_renew, new_max_hold = self.check_exit_or_renew(
                    current_price, current_time, fused_signal
                )
                if exit_reason:
                    if exit_reason == "convert_to_runner":
                        # Convert position to runner instead of closing
                        self._convert_to_runner(current_time, current_price)
                    else:
                        # Calculate stop loss trigger level for realistic fills
                        stop_trigger_pct = None
                        if "stop_loss" in exit_reason:
                            stop_trigger_pct = -self.config.stop_loss_pct
                        elif "trailing" in exit_reason or "breakeven" in exit_reason:
                            stop_trigger_pct = self.position.peak_pnl_pct - 5.0
                        self._close_position(current_time, current_price, exit_reason, stop_trigger_pct)
                elif should_renew:
                    self.position.barrier_start_time = current_time
                    self.position.max_hold_minutes = new_max_hold
                    self.position.renewals += 1

            # Create pending entry (to be executed on NEXT minute's open)
            if self.position is None and self.pending_entry is None and self.executor.is_trading_allowed(minutes_elapsed) and not is_last_15_minutes:
                # Apply heuristic filter if enabled
                final_action = fused_signal.action
                if self.use_heuristic and self.heuristic_model is not None and period_ohlcv is not None:
                    if fused_signal.action != Signal.HOLD:
                        # Get last 30 minutes of OHLCV for heuristic
                        lookback = min(30, i)
                        ohlcv_window = period_ohlcv.iloc[i - lookback:i + 1]
                        final_action = self.heuristic_model.get_signal(ohlcv_window, fused_signal.action)
                        if final_action == Signal.HOLD and fused_signal.action != Signal.HOLD:
                            self.heuristic_rejections += 1

                confluence_count = len(fused_signal.agreeing_models)
                if final_action == Signal.LONG:
                    # Queue entry for next minute's open
                    self.pending_entry = PendingEntry(
                        position_type=PositionType.CALL,
                        signal_time=current_time,
                        underlying_price=current_price,
                        signal_idx=i,
                        dominant_model=fused_signal.dominant_model,
                        max_hold_minutes=fused_signal.exit_horizon_minutes,
                        confidence=fused_signal.confidence,
                        confluence_count=confluence_count,
                        agreeing_models=fused_signal.agreeing_models,
                    )
                elif final_action == Signal.SHORT:
                    # Queue entry for next minute's open
                    self.pending_entry = PendingEntry(
                        position_type=PositionType.PUT,
                        signal_time=current_time,
                        underlying_price=current_price,
                        signal_idx=i,
                        dominant_model=fused_signal.dominant_model,
                        max_hold_minutes=fused_signal.exit_horizon_minutes,
                        confidence=fused_signal.confidence,
                        confluence_count=confluence_count,
                        agreeing_models=fused_signal.agreeing_models,
                    )

        # Clear pending entry at end of backtest
        self.pending_entry = None

        if self.position is not None:
            self._close_position(
                period_df.index[-1],
                period_closes[-1],
                "backtest_end"
            )

        return self._compute_metrics(year, month), signals

    def _convert_to_runner(self, current_time: datetime, current_price: float) -> None:
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
        full_pnl_dollars = position_value * (pnl_pct_with_slippage / 100)

        # We're splitting in half - only realize profit on the exiting half
        half_position_value = position_value / 2

        # Costs for the main position exit (half position)
        spread_cost = half_position_value * (self.config.spread_cost_pct / 100)
        commission_cost = self.config.commission_per_contract * (self.position.num_contracts // 2)
        total_costs = spread_cost + commission_cost
        pnl_dollars = (full_pnl_dollars / 2) - total_costs

        # Update capital for main position exit (only half)
        self.capital += pnl_dollars
        self.peak_capital = max(self.peak_capital, self.capital)
        self.daily_pnl += pnl_dollars
        self.daily_allocation -= half_position_value  # Only half exits

        # Track realized profit (only from the exited half)
        self.daily_realized_profit += half_position_value + pnl_dollars

        holding_minutes = int((current_time - self.position.entry_time).total_seconds() / 60)

        # Build contract name
        opt_type = "C" if self.position.position_type == PositionType.CALL else "P"
        contract = f"SPY {int(self.position.strike)}{opt_type}"

        # Store original entry time and price for the runner (so chart and table show original entry)
        original_entry_time = self.position.parent_entry_time or self.position.entry_time
        original_entry_option_price = self.position.parent_entry_option_price or self.position.entry_option_price

        # Split contracts in half - main exit gets half, runner gets half
        main_contracts = self.position.num_contracts // 2
        runner_contracts = self.position.num_contracts - main_contracts  # Remainder goes to runner
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
        )
        self.trades.append(main_trade)

        # Now create a NEW runner position starting from here
        # Runner uses ORIGINAL entry price for P&L calculation (not split price)
        # Runner gets half the position value
        runner_position_value = half_position_value

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
        )

    def _open_position(
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

        # Scale position size with confidence: 1x at 50%, 5x at 100%
        # Linear interpolation: multiplier = 1 + 8 * (confidence - 0.5)
        confidence_multiplier = max(1.0, 1.0 + 8.0 * (confidence - 0.5))
        position_value = self.capital * (self.config.position_size_pct / 100) * confidence_multiplier

        # Check max daily cost basis limit (can only invest up to starting balance)
        if self.daily_cost_basis + position_value > self.max_daily_cost_basis:
            # Cap to remaining allowed cost basis
            remaining_allowed = self.max_daily_cost_basis - self.daily_cost_basis
            if remaining_allowed <= 0:
                return False  # Already at max daily cost basis
            position_value = remaining_allowed

        max_daily = self.capital * (self.config.max_daily_allocation_pct / 100)
        if self.daily_allocation + position_value > max_daily:
            return False

        # Strike selection based on confidence
        # Default: ITM (conservative)
        # 60%+ confidence: 1 strike OTM
        # 90%+ confidence: 2 strikes OTM
        if position_type == PositionType.CALL:
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
            position_size_multiplier=confidence_multiplier,
            num_contracts=self.config.contracts_per_trade,
        )

        if dominant_model in self.model_contributions:
            self.model_contributions[dominant_model] += 1

        self.daily_trades += 1
        self.daily_allocation += position_value

        # Track cost basis for profit protection
        self.daily_cost_basis += position_value

        return True

    def _close_position(
        self,
        exit_time: datetime,
        exit_underlying_price: float,
        exit_reason: str,
        stop_loss_trigger_pct: Optional[float] = None,  # For stop loss fills
    ):
        if self.position is None:
            return

        # Determine exit price based on exit type
        is_stop_loss = "stop" in exit_reason.lower() or "trailing" in exit_reason.lower()

        if is_stop_loss and stop_loss_trigger_pct is not None:
            # For stop losses: get both close and low prices
            close_price = self.options.get_option_price(
                timestamp=exit_time,
                strike=self.position.strike,
                position_type=self.position.position_type,
                underlying_price=exit_underlying_price,
                price_type="close",
            )
            low_price = self.options.get_option_price(
                timestamp=exit_time,
                strike=self.position.strike,
                position_type=self.position.position_type,
                underlying_price=exit_underlying_price,
                price_type="low",
            )

            if close_price is not None and low_price is not None:
                # Calculate stop trigger price
                stop_trigger_price = self.position.entry_option_price * (1 + stop_loss_trigger_pct / 100)
                # Fill at average of trigger and low (realistic slippage through stop)
                exit_option_price = (stop_trigger_price + low_price) / 2
                # Add exit slippage (we get less when selling)
                exit_option_price *= (1 - self.config.exit_slippage_pct / 100)
            else:
                exit_option_price = close_price
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
        self.daily_pnl += pnl_dollars
        self.daily_allocation -= position_value

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
        )
        self.trades.append(trade)
        self.position = None

    def _compute_metrics(self, year: int, month: Optional[int] = None) -> Dict:
        initial = self.config.initial_capital
        if not self.trades:
            return {
                "year": year,
                "month": month,
                "trades": 0,
                "trades_per_day": 0.0,
                "trading_days": 0,
                "win_rate": 0.0,
                "avg_pnl_pct": 0.0,
                "total_return_pct": 0.0,
                "total_pnl_dollars": 0.0,
                "final_capital": initial,
                "max_drawdown_pct": 0.0,
                "avg_win_pct": 0.0,
                "avg_loss_pct": 0.0,
                "long_trades": 0,
                "short_trades": 0,
                "avg_hold_minutes": 0.0,
                "exit_reasons": {},
                "model_contributions": self.model_contributions.copy(),
            }

        pnls = [t.pnl_pct for t in self.trades]
        wins = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p <= 0]

        exit_reasons = {}
        for t in self.trades:
            exit_reasons[t.exit_reason] = exit_reasons.get(t.exit_reason, 0) + 1

        long_trades = [t for t in self.trades if t.position_type == PositionType.CALL]
        short_trades = [t for t in self.trades if t.position_type == PositionType.PUT]

        trading_days = len(set(t.entry_time.date() for t in self.trades))

        final_capital = self.trades[-1].capital_after
        total_return_pct = (final_capital - initial) / initial * 100
        total_pnl_dollars = sum(t.pnl_dollars for t in self.trades)

        peak = initial
        max_dd = 0
        for t in self.trades:
            peak = max(peak, t.capital_after)
            dd = (peak - t.capital_after) / peak * 100
            max_dd = max(max_dd, dd)

        model_trades = {"15m": 0, "5m": 0, "1m": 0}
        model_pnl = {"15m": 0.0, "5m": 0.0, "1m": 0.0}
        for t in self.trades:
            if t.dominant_model in model_trades:
                model_trades[t.dominant_model] += 1
                model_pnl[t.dominant_model] += t.pnl_dollars

        return {
            "year": year,
            "month": month,
            "trades": len(self.trades),
            "trades_per_day": len(self.trades) / trading_days if trading_days > 0 else 0,
            "trading_days": trading_days,
            "win_rate": len(wins) / len(self.trades) * 100 if self.trades else 0,
            "avg_pnl_pct": np.mean(pnls),
            "total_return_pct": total_return_pct,
            "total_pnl_dollars": total_pnl_dollars,
            "final_capital": final_capital,
            "max_drawdown_pct": max_dd,
            "avg_win_pct": np.mean(wins) if wins else 0,
            "avg_loss_pct": np.mean(losses) if losses else 0,
            "long_trades": len(long_trades),
            "short_trades": len(short_trades),
            "long_pnl": sum(t.pnl_dollars for t in long_trades),
            "short_pnl": sum(t.pnl_dollars for t in short_trades),
            "avg_hold_minutes": np.mean([t.holding_minutes for t in self.trades]),
            "exit_reasons": exit_reasons,
            "model_contributions": self.model_contributions.copy(),
            "model_trades": model_trades,
            "model_pnl": model_pnl,
        }
