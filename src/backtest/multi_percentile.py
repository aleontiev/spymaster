"""
Multi-scale percentile policy fusion backtester.

Combines multiple independently-trained percentile policy models into a single
unified trading bot using voting fusion.
"""
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from zoneinfo import ZoneInfo

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
from src.execution.position_manager import PositionManager
from src.execution.heuristic import HeuristicConfig, HeuristicModel
from src.execution.options_provider import RealOptionsProvider
from src.strategy.fusion_config import FusedSignal, Signal
from src.strategy.trading_rules import (
    BreachType,
    BreachSignal,
    calculate_vwap_series,
    calculate_vwap_with_bands,
    detect_breach,
    check_breach_continuation_confirmation,
    BounceType,
    BounceSignal,
    detect_vwap_bounce,
    NewsEventType,
    NewsEventSignal,
    detect_news_event,
)


def check_model_confluence(
    fused_signal: FusedSignal,
    trade_direction: Signal,
) -> Tuple[bool, float]:
    """
    Check if model signals (1m/5m/15m) confirm or contradict a breakout/reversal trade.

    Returns:
        Tuple of (is_allowed, bonus_pct):
        - is_allowed: False if any model has a contrasting signal → reject trade
        - bonus_pct: Extra position size percentage from confirming models (0.005 per model)
    """
    contrasting = Signal.SHORT if trade_direction == Signal.LONG else Signal.LONG
    confirming_count = 0

    for model_name in ("1m", "5m", "15m"):
        if model_name in fused_signal.individual_signals:
            sig, _prob = fused_signal.individual_signals[model_name]
            if sig == contrasting:
                return False, 0.0  # Contrasting signal → reject
            elif sig == trade_direction:
                confirming_count += 1

    # Bonus: 0.5% per confirming model (max 1.5%)
    bonus = confirming_count * 0.005
    return True, bonus


class MultiPercentileBacktester:
    """Backtest multi-scale percentile policy fusion."""

    def __init__(
        self,
        executor,  # Can be MultiPercentileExecutor or FilteredExecutor
        config: BacktestConfig,
        device: torch.device,
        use_heuristic: bool = True,
        heuristic_config: Optional[HeuristicConfig] = None,
    ):
        self.executor = executor
        self.config = config
        self.device = device
        self.options = RealOptionsProvider()
        self.use_heuristic = use_heuristic
        self.heuristic_config = heuristic_config or HeuristicConfig()
        self.heuristic_model = HeuristicModel(
            lookback_minutes=30,
            config=self.heuristic_config,
        ) if use_heuristic else None
        self.et_tz = ZoneInfo("America/New_York")

        model_info = executor.get_model_info()
        self.max_context_len = max(info["context_len"] for info in model_info.values()) if model_info else 60
        self.context_lens = {name: info["context_len"] for name, info in model_info.items()}

        # Position manager owns all position/capital/drawdown state
        self.pm = PositionManager(config, self.options, self.heuristic_config)

        # Track heuristic rejections for stats
        self.heuristic_rejections = 0
        self.timeframe_rejections = 0
        self.vwap_direction_rejections = 0
        self.confluence_rejections = 0
        self.bounce_entries = 0

        # Pending entry (to be executed on next minute's open)
        self.pending_entry: Optional[PendingEntry] = None

        # Track potential VWAP breach from previous candle (for continuation confirmation)
        self.pending_breach: Optional[BreachSignal] = None

        # VWAP-specific metrics tracking
        self.breach_immediate_entries = 0  # Strong breach entries
        self.breach_continuation_entries = 0  # Confirmed breach entries
        self.news_event_entries = 0

    def reset(self, starting_capital: Optional[float] = None):
        """Reset state for new backtest."""
        self.pm.reset(starting_capital)

        # Heuristic tracking
        self.heuristic_rejections = 0
        self.timeframe_rejections = 0
        self.vwap_direction_rejections = 0
        self.confluence_rejections = 0
        self.bounce_entries = 0

        # Pending entry
        self.pending_entry = None

        # Pending VWAP breach (for continuation confirmation)
        self.pending_breach = None

        # VWAP-specific metrics tracking
        self.breach_immediate_entries = 0
        self.breach_continuation_entries = 0
        self.news_event_entries = 0

    def _get_minutes_elapsed(self, timestamp: datetime) -> int:
        et_time = timestamp.tz_convert("America/New_York")
        minutes_since_midnight = et_time.hour * 60 + et_time.minute
        market_open = 9 * 60 + 30
        return minutes_since_midnight - market_open

    # === Proxy properties for external compatibility ===

    @property
    def position(self) -> Optional[Position]:
        return self.pm.position

    @position.setter
    def position(self, value: Optional[Position]) -> None:
        self.pm.position = value

    @property
    def trades(self) -> List[Trade]:
        return self.pm.trades

    @property
    def capital(self) -> float:
        return self.pm.capital

    @property
    def peak_capital(self) -> float:
        return self.pm.peak_capital

    @property
    def lowest_capital(self) -> float:
        return self.pm.lowest_capital

    @property
    def model_contributions(self) -> Dict[str, int]:
        return self.pm.model_contributions

    @property
    def daily_results(self) -> List[float]:
        return self.pm.daily_results

    @property
    def weekly_results(self) -> List[float]:
        return self.pm.weekly_results

    @property
    def best_daily_pnl_pct(self) -> float:
        return self.pm.best_daily_pnl_pct

    @property
    def worst_daily_pnl_pct(self) -> float:
        return self.pm.worst_daily_pnl_pct

    @property
    def best_weekly_pnl_pct(self) -> float:
        return self.pm.best_weekly_pnl_pct

    @property
    def worst_weekly_pnl_pct(self) -> float:
        return self.pm.worst_weekly_pnl_pct

    @property
    def max_daily_drawdown_pct(self) -> float:
        return self.pm.max_daily_drawdown_pct

    @property
    def max_weekly_drawdown_pct(self) -> float:
        return self.pm.max_weekly_drawdown_pct

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

    def run_backtest(
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

        # Calculate VWAP with bands for the period (for VWAP breach detection)
        vwap_series = None
        upper_1sd_series = None
        lower_1sd_series = None
        upper_2sd_series = None
        lower_2sd_series = None
        if period_ohlcv is not None:
            vwap_series, upper_1sd_series, lower_1sd_series, upper_2sd_series, lower_2sd_series = calculate_vwap_with_bands(period_ohlcv)

        # Track signals from all models
        signals = {"1m": [], "5m": [], "15m": [], "breach": [], "bounce": [], "news_event": []}
        # Track missed opportunities (triple confluence but no trade)
        missed_signals = []
        current_day = None
        desc = f"{year}-{month:02d}" if month else f"Year {year}"

        # Context lengths for each model
        all_context_lens = {"1m": 5, "5m": 15, "15m": 60}

        for i in tqdm(range(self.max_context_len, len(period_df)), desc=desc, leave=False):
            current_time = period_df.index[i]
            minutes_elapsed = self._get_minutes_elapsed(current_time)
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
                    self.pm.close_position(current_time, current_price, "eod_close")
                continue

            # Check if we're in the first 10 minutes (no new positions - too volatile)
            is_first_10_minutes = (et_hour == 9 and et_minute < 40)

            # Check if we're in the last 15 minutes (no new positions, but can hold existing)
            is_last_15_minutes = (et_hour == 15 and et_minute >= 45)

            trading_date = et_time.date()
            if current_day != trading_date:
                current_day = trading_date
                self.options.load_date(trading_date)

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

            # Track VWAP breach signals for HTML report (potential breaches only - we always wait for confirmation)
            if vwap_series is not None and period_ohlcv is not None:
                vwap_breach_for_chart = detect_breach(period_ohlcv, vwap_series, i,
                                                            upper_1sd_series=upper_1sd_series,
                                                            lower_1sd_series=lower_1sd_series,
                                                            upper_2sd_series=upper_2sd_series,
                                                            lower_2sd_series=lower_2sd_series)
                if vwap_breach_for_chart.breach_type in (BreachType.BULLISH, BreachType.BULLISH_POTENTIAL):
                    signals["breach"].append(SignalMarker(
                        timestamp=int(current_time.timestamp()),
                        signal_type="long",
                        model="breach",
                        price=current_price,
                    ))
                elif vwap_breach_for_chart.breach_type in (BreachType.BEARISH, BreachType.BEARISH_POTENTIAL):
                    signals["breach"].append(SignalMarker(
                        timestamp=int(current_time.timestamp()),
                        signal_type="short",
                        model="breach",
                        price=current_price,
                    ))

            # Track bounce signals for HTML report
            if vwap_series is not None and period_ohlcv is not None:
                bounce_for_chart = detect_vwap_bounce(period_ohlcv, vwap_series, i)
                if bounce_for_chart.bounce_type in (BounceType.BULLISH_HAMMER, BounceType.BULLISH_ENGULFING):
                    signals["bounce"].append(SignalMarker(
                        timestamp=int(current_time.timestamp()),
                        signal_type="long",
                        model="bounce",
                        price=current_price,
                    ))
                elif bounce_for_chart.bounce_type in (BounceType.BEARISH_SHOOTING, BounceType.BEARISH_ENGULFING):
                    signals["bounce"].append(SignalMarker(
                        timestamp=int(current_time.timestamp()),
                        signal_type="short",
                        model="bounce",
                        price=current_price,
                    ))

            # Track news event signals for HTML report
            is_structure_ready = et_hour >= 10
            if period_ohlcv is not None and is_structure_ready:
                ne_for_chart = detect_news_event(
                    period_ohlcv, i,
                    upper_1sd=upper_1sd_series.iloc[i] if upper_1sd_series is not None and i < len(upper_1sd_series) else None,
                    lower_1sd=lower_1sd_series.iloc[i] if lower_1sd_series is not None and i < len(lower_1sd_series) else None,
                    upper_2sd=upper_2sd_series.iloc[i] if upper_2sd_series is not None and i < len(upper_2sd_series) else None,
                    lower_2sd=lower_2sd_series.iloc[i] if lower_2sd_series is not None and i < len(lower_2sd_series) else None,
                )
                if ne_for_chart.event_type == NewsEventType.BULLISH:
                    signals["news_event"].append(SignalMarker(
                        timestamp=int(current_time.timestamp()),
                        signal_type="long",
                        model="news_event",
                        price=current_price,
                    ))
                elif ne_for_chart.event_type == NewsEventType.BEARISH:
                    signals["news_event"].append(SignalMarker(
                        timestamp=int(current_time.timestamp()),
                        signal_type="short",
                        model="news_event",
                        price=current_price,
                    ))

            # Execute pending entry at OPEN of this minute (signal was from previous minute)
            if self.pending_entry is not None and self.position is None:
                pe = self.pending_entry
                self.pending_entry = None

                # VWAP direction filter at execution time: no longs below VWAP, no shorts above VWAP
                # (Skip for bounces which trade at VWAP level by definition)
                current_vwap = vwap_series.iloc[i] if vwap_series is not None else None
                vwap_direction_ok = True
                if current_vwap is not None and not pe.is_bounce and not pe.is_news_event:
                    if pe.position_type == PositionType.CALL and current_price < current_vwap:
                        vwap_direction_ok = False  # Block long below VWAP
                    elif pe.position_type == PositionType.PUT and current_price > current_vwap:
                        vwap_direction_ok = False  # Block short above VWAP

                if vwap_direction_ok:
                    # Execute at this minute's open price
                    self.pm.open_position(
                        pe.position_type,
                        current_time,  # Execute at current time (open of this minute)
                        current_price,
                        i,
                        pe.dominant_model,
                        pe.max_hold_minutes,
                        pe.confidence,
                        pe.confluence_count,
                        pe.agreeing_models,
                        pe.is_early_trade,
                        pe.is_breach,
                        pe.position_size_bonus,
                        pe.is_bounce,
                        pe.is_bounce_engulfing,
                        is_news_event=pe.is_news_event,
                    )

            # Normal backtest logic
            if self.position is not None:
                # Get current band values for VWAP take profit checks
                curr_upper_2sd = upper_2sd_series.iloc[i] if upper_2sd_series is not None else None
                curr_lower_2sd = lower_2sd_series.iloc[i] if lower_2sd_series is not None else None
                curr_upper_1sd = upper_1sd_series.iloc[i] if upper_1sd_series is not None else None
                curr_lower_1sd = lower_1sd_series.iloc[i] if lower_1sd_series is not None else None

                # Get candle data for opposite direction exit check
                candle_open = None
                candle_close = None
                avg_body = None
                if period_ohlcv is not None:
                    candle_open = period_ohlcv.iloc[i]['open']
                    candle_close = period_ohlcv.iloc[i]['close']
                    # Calculate average body over last 20 candles
                    lookback = min(20, i)
                    if lookback > 0:
                        recent = period_ohlcv.iloc[i - lookback:i]
                        avg_body = (recent['close'] - recent['open']).abs().mean()

                exit_reason, should_renew, new_max_hold = self.pm.check_exit_or_renew(
                    current_price, current_time, fused_signal,
                    upper_2sd=curr_upper_2sd, lower_2sd=curr_lower_2sd,
                    upper_1sd=curr_upper_1sd, lower_1sd=curr_lower_1sd,
                    candle_open=candle_open, candle_close=candle_close, avg_body=avg_body
                )
                if exit_reason:
                    if exit_reason == "convert_to_runner":
                        # Convert position to runner instead of closing
                        self.pm.convert_to_runner(current_time, current_price)
                    else:
                        # Calculate stop loss trigger level for realistic fills
                        stop_trigger_pct = None
                        if "stop_loss" in exit_reason:
                            stop_trigger_pct = -self.config.stop_loss_pct
                        elif "trailing" in exit_reason or "breakeven" in exit_reason:
                            if self.position.is_runner:
                                # Runner trailing: 10% from runner peak
                                trail_dist = 10.0
                                stop_trigger_pct = self.position.runner_peak_pnl_pct - trail_dist
                            else:
                                trail_dist = 5.0  # Same for all trades - aggressive trailing
                                stop_trigger_pct = self.position.peak_pnl_pct - trail_dist
                        self.pm.close_position(current_time, current_price, exit_reason, stop_trigger_pct)
                elif should_renew:
                    self.position.barrier_start_time = current_time
                    self.position.max_hold_minutes = new_max_hold
                    self.position.renewals += 1

            # Create pending entry (to be executed on NEXT minute's open)
            # Track why confluence signals don't result in trades
            has_confluence = len(fused_signal.agreeing_models) >= 3 and fused_signal.action != Signal.HOLD
            rejection_reason = None

            # Check for early window (9:45-10:00)
            trading_window = self.executor.fusion_config.trading_window
            is_early_window = trading_window.is_early_window(minutes_elapsed)

            if has_confluence:
                # Determine why we can't/won't trade
                if self.position is not None:
                    rejection_reason = "already_in_position"
                elif self.pending_entry is not None:
                    rejection_reason = "pending_entry_exists"
                elif not self.executor.is_trading_allowed(minutes_elapsed) and not is_early_window:
                    rejection_reason = f"time_restricted_min_{minutes_elapsed}"
                elif is_first_10_minutes:
                    rejection_reason = "first_10_minutes"
                elif is_last_15_minutes:
                    rejection_reason = "last_15_minutes"

            if self.position is None and self.pending_entry is None and not is_first_10_minutes and not is_last_15_minutes:
                final_action = Signal.HOLD
                is_early_trade = False
                is_breach_detected = False
                is_bounce = False
                is_bounce_engulfing = False
                is_news_event = False
                agreeing_models = ()
                dominant_model = ""
                exit_horizon = 10  # Default for early trades
                confidence = 0.5

                # No VWAP-based signals before 10:00 AM - wait for structure to develop
                is_structure_ready = et_hour >= 10

                # PRIORITY 0.5: News Event (highest priority — exogenous signal)
                if is_structure_ready and period_ohlcv is not None:
                    ne_signal = detect_news_event(
                        period_ohlcv, i,
                        upper_1sd=upper_1sd_series.iloc[i] if upper_1sd_series is not None and i < len(upper_1sd_series) else None,
                        lower_1sd=lower_1sd_series.iloc[i] if lower_1sd_series is not None and i < len(lower_1sd_series) else None,
                        upper_2sd=upper_2sd_series.iloc[i] if upper_2sd_series is not None and i < len(upper_2sd_series) else None,
                        lower_2sd=lower_2sd_series.iloc[i] if lower_2sd_series is not None and i < len(lower_2sd_series) else None,
                    )
                    if ne_signal.event_type != NewsEventType.NONE:
                        if ne_signal.event_type == NewsEventType.BULLISH:
                            final_action = Signal.LONG
                        else:
                            final_action = Signal.SHORT
                        is_news_event = True
                        is_breach_detected = True  # Inherit breach exit rules
                        dominant_model = "news_event"
                        exit_horizon = 10
                        confidence = 0.90
                        agreeing_models = ("news_event",)
                        position_size_bonus = 0.0
                        self.news_event_entries += 1

                # PRIORITY 1: Check for VWAP breach (always takes priority)
                if not is_news_event and is_structure_ready and vwap_series is not None and period_ohlcv is not None:
                    vwap_breach = detect_breach(period_ohlcv, vwap_series, i,
                                                       upper_1sd_series=upper_1sd_series,
                                                       lower_1sd_series=lower_1sd_series,
                                                       upper_2sd_series=upper_2sd_series,
                                                       lower_2sd_series=lower_2sd_series)

                    # First, check if we have a pending potential breach that gets confirmed
                    if self.pending_breach is not None:
                        if check_breach_continuation_confirmation(period_ohlcv, i, self.pending_breach):
                            # Continuation confirms the potential breach
                            if self.pending_breach.breach_type == BreachType.BULLISH_POTENTIAL:
                                final_action = Signal.LONG
                                is_breach_detected = True
                                agreeing_models = ("vwap_breach_continuation",)
                                dominant_model = "breach"
                                exit_horizon = 15
                                confidence = 0.7
                                self.breach_continuation_entries += 1
                            elif self.pending_breach.breach_type == BreachType.BEARISH_POTENTIAL:
                                final_action = Signal.SHORT
                                is_breach_detected = True
                                agreeing_models = ("vwap_breach_continuation",)
                                dominant_model = "breach"
                                exit_horizon = 15
                                confidence = 0.7
                                self.breach_continuation_entries += 1
                        # Clear pending breach after checking
                        self.pending_breach = None

                    # If no continuation entry, check for new VWAP breach
                    if not is_breach_detected:
                        # Check for CONFIRMED breaches (strong body + volume) - enter immediately
                        if vwap_breach.breach_type == BreachType.BULLISH:
                            final_action = Signal.LONG
                            is_breach_detected = True
                            agreeing_models = ("vwap_breach_immediate",)
                            dominant_model = "breach"
                            exit_horizon = 15
                            confidence = 0.8  # High confidence for strong breach
                            self.breach_immediate_entries += 1
                        elif vwap_breach.breach_type == BreachType.BEARISH:
                            final_action = Signal.SHORT
                            is_breach_detected = True
                            agreeing_models = ("vwap_breach_immediate",)
                            dominant_model = "breach"
                            exit_horizon = 15
                            confidence = 0.8  # High confidence for strong breach
                            self.breach_immediate_entries += 1
                        # Check for POTENTIAL breaches - wait for confirmation candle
                        elif vwap_breach.breach_type in (
                            BreachType.BULLISH_POTENTIAL,
                            BreachType.BEARISH_POTENTIAL,
                        ):
                            # Track potential breach for next candle's continuation check
                            self.pending_breach = vwap_breach

                    # Model confluence check for VWAP breach: reject if contrasting, bonus if confirming
                    # (News events skip confluence check — exogenous signal)
                    if is_breach_detected and not is_news_event and final_action != Signal.HOLD:
                        confluence_ok, confluence_bonus = check_model_confluence(fused_signal, final_action)
                        if not confluence_ok:
                            self.confluence_rejections += 1
                            if has_confluence:
                                rejection_reason = "confluence_rejected"
                            final_action = Signal.HOLD
                            is_breach_detected = False
                        else:
                            position_size_bonus = confluence_bonus

                # PRIORITY 1.8: Check for VWAP bounce
                if not is_news_event and is_structure_ready and not is_breach_detected and vwap_series is not None and period_ohlcv is not None:
                    bounce_signal = detect_vwap_bounce(period_ohlcv, vwap_series, i)
                    if bounce_signal.bounce_type != BounceType.NONE:
                        if bounce_signal.bounce_type in (BounceType.BULLISH_HAMMER, BounceType.BULLISH_ENGULFING):
                            bounce_action = Signal.LONG
                        else:
                            bounce_action = Signal.SHORT
                        confluence_ok, confluence_bonus = check_model_confluence(fused_signal, bounce_action)
                        if confluence_ok:
                            final_action = bounce_action
                            is_bounce = True
                            is_bounce_engulfing = bounce_signal.is_engulfing
                            dominant_model = "bounce"
                            exit_horizon = 15
                            confidence = 0.75
                            agreeing_models = ("bounce",)
                            position_size_bonus = confluence_bonus
                            self.bounce_entries += 1
                        else:
                            self.confluence_rejections += 1
                            if has_confluence:
                                rejection_reason = "confluence_rejected"

                # PRIORITY 2: Early window (9:45-10:00) - only if no VWAP, bounce, or news event
                if not is_news_event and not is_breach_detected and not is_bounce and is_early_window:
                    # Early window (9:45-10:00): Require 1m+5m confluence + heuristic
                    early_agreeing = []
                    early_action = Signal.HOLD
                    early_confidence = 0.0

                    for model_name in ["1m", "5m"]:
                        if model_name in fused_signal.individual_signals:
                            sig, prob = fused_signal.individual_signals[model_name]
                            if sig != Signal.HOLD:
                                if early_action == Signal.HOLD:
                                    early_action = sig
                                    early_agreeing.append(model_name)
                                    early_confidence = prob
                                elif sig == early_action:
                                    early_agreeing.append(model_name)
                                    early_confidence = max(early_confidence, prob)
                                else:
                                    early_action = Signal.HOLD
                                    early_agreeing = []
                                    break

                    # Need both 1m and 5m to agree
                    if len(early_agreeing) == 2 and early_action != Signal.HOLD:
                        if self.use_heuristic and self.heuristic_model is not None and period_ohlcv is not None:
                            lookback = min(30, i)
                            ohlcv_window = period_ohlcv.iloc[i - lookback:i + 1]
                            heuristic_action = self.heuristic_model.get_signal(ohlcv_window, early_action)
                            if heuristic_action == early_action:
                                final_action = early_action
                                is_early_trade = True
                                agreeing_models = tuple(early_agreeing) + ("heuristic",)
                                dominant_model = "5m"
                                exit_horizon = 10
                                confidence = early_confidence

                # PRIORITY 3: Normal trading window (only if no VWAP/bounce and no early trade)
                elif not is_breach_detected and not is_bounce and self.executor.is_trading_allowed(minutes_elapsed):
                    # Normal trading window
                    final_action = fused_signal.action
                    if self.use_heuristic and self.heuristic_model is not None and period_ohlcv is not None:
                        if fused_signal.action != Signal.HOLD:
                            lookback = min(30, i)
                            ohlcv_window = period_ohlcv.iloc[i - lookback:i + 1]
                            final_action = self.heuristic_model.get_signal(ohlcv_window, fused_signal.action)
                            if final_action == Signal.HOLD and fused_signal.action != Signal.HOLD:
                                self.heuristic_rejections += 1
                                if has_confluence:
                                    rejection_reason = "heuristic_rejected"

                    agreeing_models = fused_signal.agreeing_models
                    dominant_model = fused_signal.dominant_model
                    exit_horizon = fused_signal.exit_horizon_minutes
                    confidence = fused_signal.confidence

                # Get daily VWAP for timeframe support and position sizing
                daily_vwap = vwap_series.iloc[i] if vwap_series is not None else None
                if not is_breach_detected and not is_bounce:
                    position_size_bonus = 0.0

                # Check VWAP direction, timeframe support, and calculate position size bonus
                # (Skip for VWAP breach and bounce trades which have their own logic)
                if final_action != Signal.HOLD and not is_breach_detected and not is_bounce:
                    if self.use_heuristic and self.heuristic_model is not None and period_ohlcv is not None:
                        lookback = min(30, i)
                        ohlcv_window = period_ohlcv.iloc[i - lookback:i + 1]

                        # Check VWAP direction (LONG only above VWAP, SHORT only below VWAP)
                        # Note: VWAP breach and bounce trades skip this check
                        vwap_direction_ok, _ = self.heuristic_model.check_vwap_direction(
                            final_action, current_price, daily_vwap
                        )
                        if not vwap_direction_ok:
                            self.vwap_direction_rejections += 1
                            if has_confluence:
                                rejection_reason = "vwap_direction_rejected"
                            final_action = Signal.HOLD

                        # Check timeframe support (5m or 15m VWAP)
                        if final_action != Signal.HOLD:
                            is_supported, _ = self.heuristic_model.check_timeframe_support(
                                ohlcv_window, final_action, current_price, daily_vwap
                            )
                            if not is_supported:
                                self.timeframe_rejections += 1
                                if has_confluence:
                                    rejection_reason = "timeframe_rejected"
                                final_action = Signal.HOLD
                            else:
                                # Calculate position size bonus
                                position_size_bonus, _ = self.heuristic_model.get_position_size_bonus(
                                    ohlcv_window, final_action, current_price, daily_vwap
                                )

                confluence_count = len(agreeing_models)
                # Combine flags for position sizing and exit logic
                is_breakout_trade = is_breach_detected or is_bounce or is_news_event

                # VWAP direction filter: no longs below VWAP, no shorts above VWAP
                # (Skip for bounces which trade at VWAP level by definition, and news events)
                if daily_vwap is not None and not is_bounce and not is_news_event:
                    if final_action == Signal.LONG and current_price < daily_vwap:
                        rejection_reason = "long_below_vwap"
                        final_action = Signal.HOLD  # Block long below VWAP
                    elif final_action == Signal.SHORT and current_price > daily_vwap:
                        rejection_reason = "short_above_vwap"
                        final_action = Signal.HOLD  # Block short above VWAP

                # Volume bonus: +1% position size if signal candle volume >= 1.25x previous candle
                if final_action != Signal.HOLD and period_ohlcv is not None and i > 0:
                    signal_vol = period_ohlcv.iloc[i]['volume']
                    prev_vol = period_ohlcv.iloc[i - 1]['volume']
                    if prev_vol > 0 and signal_vol >= 1.25 * prev_vol:
                        position_size_bonus += 0.01

                if final_action == Signal.LONG:
                    rejection_reason = None  # Trade will be taken
                    self.pending_entry = PendingEntry(
                        position_type=PositionType.CALL,
                        signal_time=current_time,
                        underlying_price=current_price,
                        signal_idx=i,
                        dominant_model=dominant_model,
                        max_hold_minutes=exit_horizon,
                        confidence=confidence,
                        confluence_count=confluence_count,
                        agreeing_models=agreeing_models,
                        is_early_trade=is_early_trade,
                        is_breach=is_breakout_trade,
                        position_size_bonus=position_size_bonus,
                        is_bounce=is_bounce,
                        is_bounce_engulfing=is_bounce_engulfing,
                        is_news_event=is_news_event,
                    )
                elif final_action == Signal.SHORT:
                    rejection_reason = None  # Trade will be taken
                    self.pending_entry = PendingEntry(
                        position_type=PositionType.PUT,
                        signal_time=current_time,
                        underlying_price=current_price,
                        signal_idx=i,
                        dominant_model=dominant_model,
                        max_hold_minutes=exit_horizon,
                        confidence=confidence,
                        confluence_count=confluence_count,
                        agreeing_models=agreeing_models,
                        is_early_trade=is_early_trade,
                        is_breach=is_breakout_trade,
                        position_size_bonus=position_size_bonus,
                        is_bounce=is_bounce,
                        is_bounce_engulfing=is_bounce_engulfing,
                        is_news_event=is_news_event,
                    )

            # Track missed confluence signals
            if has_confluence and rejection_reason is not None:
                et_str = et_time.strftime("%Y-%m-%d %H:%M")
                missed_signals.append({
                    "timestamp": int(current_time.timestamp()),
                    "et_time": et_str,
                    "signal_type": "long" if fused_signal.action == Signal.LONG else "short",
                    "agreeing_models": fused_signal.agreeing_models,
                    "rejection_reason": rejection_reason,
                    "price": current_price,
                    "minutes_elapsed": minutes_elapsed,
                })

        # Clear pending entry at end of backtest
        self.pending_entry = None

        if self.position is not None:
            self.pm.close_position(
                period_df.index[-1],
                period_closes[-1],
                "backtest_end"
            )

        # Print summary of missed signals
        if missed_signals:
            print(f"\n  Missed {len(missed_signals)} triple-confluence signals:")
            reason_counts = {}
            for ms in missed_signals:
                reason = ms["rejection_reason"]
                reason_counts[reason] = reason_counts.get(reason, 0) + 1
            for reason, count in sorted(reason_counts.items(), key=lambda x: -x[1]):
                print(f"    {reason}: {count}")
            # Show first 5 examples with details
            print(f"\n  First 5 missed signals (ET time):")
            for ms in missed_signals[:5]:
                print(f"    {ms['et_time']} - {ms['signal_type'].upper()} @ ${ms['price']:.2f} - {ms['rejection_reason']} (min={ms['minutes_elapsed']})")

        metrics = self._compute_metrics(year, month)
        metrics["missed_signals"] = missed_signals
        return metrics, signals

    def _compute_metrics(self, year: int, month: Optional[int] = None) -> Dict:
        initial = self.config.initial_capital

        # Finalize drawdown tracking at end of backtest
        self.pm.finalize_drawdown_tracking()

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
                "avg_daily_pnl_pct": 0.0,
                "avg_weekly_pnl_pct": 0.0,
                "daily_win_rate": 0.0,
                "weekly_win_rate": 0.0,
                "green_days": 0,
                "red_days": 0,
                "total_days": 0,
                "green_weeks": 0,
                "red_weeks": 0,
                "total_weeks": 0,
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

        # Max drawdown = how much below initial capital did we ever go
        # If lowest_capital >= initial, drawdown is 0
        if self.lowest_capital < initial:
            max_dd = (initial - self.lowest_capital) / initial * 100
        else:
            max_dd = 0.0

        model_trades = {"15m": 0, "5m": 0, "1m": 0}
        model_pnl = {"15m": 0.0, "5m": 0.0, "1m": 0.0}
        for t in self.trades:
            if t.dominant_model in model_trades:
                model_trades[t.dominant_model] += 1
                model_pnl[t.dominant_model] += t.pnl_dollars

        # Daily/Weekly win rate and average P/L calculations
        green_days = sum(1 for d in self.daily_results if d > 0)
        red_days = sum(1 for d in self.daily_results if d <= 0)
        total_days = len(self.daily_results)
        daily_win_rate = (green_days / total_days * 100) if total_days > 0 else 0.0
        avg_daily_pnl_pct = np.mean(self.daily_results) if self.daily_results else 0.0

        green_weeks = sum(1 for w in self.weekly_results if w > 0)
        red_weeks = sum(1 for w in self.weekly_results if w <= 0)
        total_weeks = len(self.weekly_results)
        weekly_win_rate = (green_weeks / total_weeks * 100) if total_weeks > 0 else 0.0
        avg_weekly_pnl_pct = np.mean(self.weekly_results) if self.weekly_results else 0.0

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
            "avg_daily_pnl_pct": avg_daily_pnl_pct,
            "avg_weekly_pnl_pct": avg_weekly_pnl_pct,
            "daily_win_rate": daily_win_rate,
            "weekly_win_rate": weekly_win_rate,
            "green_days": green_days,
            "red_days": red_days,
            "total_days": total_days,
            "green_weeks": green_weeks,
            "red_weeks": red_weeks,
            "total_weeks": total_weeks,
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
