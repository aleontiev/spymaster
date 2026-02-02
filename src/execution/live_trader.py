"""
Live paper trading orchestrator.

Main trading loop that coordinates all components for live paper trading,
maintaining parity with backtested behavior.
"""
import asyncio
import logging
from dataclasses import dataclass
from datetime import datetime, time, timedelta
from pathlib import Path
from typing import Optional, Tuple
from zoneinfo import ZoneInfo

import pandas as pd
import torch

from src.backtest.types import PositionType, Trade
from src.data.providers.alpaca import AlpacaProvider
from src.execution.alpaca_client import AlpacaClient, format_option_symbol
from src.execution.data_buffer import LiveDataBuffer
from src.execution.heuristic import HeuristicConfig, HeuristicModel
from src.execution.order_manager import OrderConfig, OrderManager, OrderStatus
from src.execution.position_manager import LivePosition, PositionManager
from src.strategy.exit_rules import ExitConfig, ExitRuleEngine
from src.strategy.fusion_config import FusedSignal, Signal
from src.strategy.multi_percentile_executor import MultiPercentileExecutor
from src.strategy.position_sizing import PositionSizeConfig, PositionSizer
from src.strategy.trading_rules import (
    BreachSignal,
    BreachType,
    BounceType,
    BounceSignal,
    NewsEventType,
    NewsEventSignal,
    calculate_vwap_with_bands,
    detect_breach,
    check_breach_continuation_confirmation,
    detect_vwap_bounce,
    detect_news_event,
)
from src.backtest.multi_percentile import check_model_confluence


logger = logging.getLogger(__name__)


@dataclass
class LiveTraderConfig:
    """Configuration for live trader."""
    initial_capital: float = 25000.0
    max_trades_per_day: int = 10
    daily_loss_limit_pct: float = 5.0
    session_id: str = "paper_trading"  # Session ID for database state persistence
    use_heuristic: bool = True
    trading_start_minutes: int = 10  # Start trading 10 min after open (9:40)
    trading_end_minutes: int = 375  # Stop opening positions 15 min before close (3:45)
    poll_interval_seconds: float = 60.0  # Main loop interval


@dataclass
class PendingEntry:
    """Pending entry to be executed on next minute's open."""
    position_type: PositionType
    signal_time: datetime
    underlying_price: float
    dominant_model: str
    max_hold_minutes: int
    confidence: float
    confluence_count: int
    agreeing_models: Tuple[str, ...]
    is_early_trade: bool
    is_breach: bool
    position_size_bonus: float
    is_reversal: bool = False
    is_reversal_engulfing: bool = False
    is_bounce: bool = False
    is_bounce_engulfing: bool = False
    is_double_breakout: bool = False
    is_news_event: bool = False


class LiveTrader:
    """
    Live paper trading orchestrator.

    Coordinates all components for live trading:
    - Data buffer for OHLCV and VWAP
    - Position manager for state and stop-loss monitoring
    - Order manager for order execution
    - Exit rule engine for exit decisions
    - Position sizer for sizing and strike selection
    - Multi-percentile executor for signal generation
    - Heuristic model for filtering

    Maintains parity with backtest logic:
    - Same VWAP breach detection
    - Same exit rules
    - Same position sizing
    - Same signal fusion
    """

    def __init__(
        self,
        executor: MultiPercentileExecutor,
        alpaca_client: AlpacaClient,
        alpaca_provider: AlpacaProvider,
        config: Optional[LiveTraderConfig] = None,
        exit_config: Optional[ExitConfig] = None,
        position_size_config: Optional[PositionSizeConfig] = None,
        order_config: Optional[OrderConfig] = None,
        heuristic_config: Optional[HeuristicConfig] = None,
        device: Optional[torch.device] = None,
    ):
        """
        Initialize the live trader.

        Args:
            executor: Multi-percentile signal executor
            alpaca_client: Alpaca trading client
            alpaca_provider: Alpaca data provider
            config: Trader configuration
            exit_config: Exit rule configuration
            position_size_config: Position sizing configuration
            order_config: Order execution configuration
            heuristic_config: Heuristic filter configuration
            device: Torch device for inference
        """
        self.executor = executor
        self.alpaca_client = alpaca_client
        self.alpaca_provider = alpaca_provider
        self.config = config or LiveTraderConfig()
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.et_tz = ZoneInfo("America/New_York")

        # Initialize components
        self.data_buffer = LiveDataBuffer()
        self.exit_engine = ExitRuleEngine(exit_config)
        self.position_sizer = PositionSizer(position_size_config)
        self.order_manager = OrderManager(alpaca_client, order_config)

        # Position manager with price monitoring (uses database for state)
        self.position_manager = PositionManager(
            alpaca_client=alpaca_client,
            get_option_price=self._get_option_price,
            session_id=self.config.session_id,
            exit_config=exit_config,
            price_check_interval=1.0,
        )

        # Heuristic model
        self.heuristic_model: Optional[HeuristicModel] = None
        if self.config.use_heuristic:
            self.heuristic_model = HeuristicModel(heuristic_config)

        # State
        self.pending_entry: Optional[PendingEntry] = None
        self.pending_breach: Optional[BreachSignal] = None

        # Statistics
        self.breach_immediate_entries: int = 0
        self.breach_continuation_entries: int = 0
        self.vwap_direction_rejections: int = 0
        self.timeframe_rejections: int = 0

        # Signal tracking (for live dashboard)
        self._last_signal_action: str = "HOLD"
        self._last_signal_source: str = "none"
        self._last_signal_confidence: float = 0.0
        self._last_signal_agreeing_models: tuple = ()

        # Graceful shutdown flag
        self._stop_requested: bool = False

        # Set stop-loss callback
        self.position_manager.set_stop_loss_callback(self._handle_stop_loss)

    def _get_option_price(self, option_symbol: str) -> Optional[float]:
        """Get current option price from Alpaca."""
        try:
            # In production, this would fetch real-time option quotes
            # For now, return None (will be implemented with options data API)
            quote = self.alpaca_provider.get_latest_quote("SPY")
            if quote:
                # This is a placeholder - real implementation needs option quotes
                return None
            return None
        except Exception as e:
            logger.error(f"Error getting option price for {option_symbol}: {e}")
            return None

    async def _handle_stop_loss(self, position: LivePosition, exit_reason: str) -> None:
        """Handle stop loss triggered by price monitor."""
        logger.warning(f"Stop loss triggered: {exit_reason} for {position.option_symbol}")

        # Get current price
        current_price = self._get_option_price(position.option_symbol)
        if current_price is None:
            # Use a conservative estimate
            current_price = position.entry_option_price * 0.9

        # Submit stop loss order
        order = await self.order_manager.submit_stop_loss_order(
            position.option_symbol,
            position.num_contracts,
            current_price,
        )

        # Wait for fill
        order = await self.order_manager.wait_for_fill(order, timeout=10.0)

        # Close position in manager
        exit_price = order.filled_avg_price if order.filled_avg_price > 0 else current_price
        underlying_price = self.data_buffer.get_latest_price() or position.entry_underlying_price

        self.position_manager.close_position(
            exit_time=datetime.now(self.et_tz),
            exit_option_price=exit_price,
            exit_underlying_price=underlying_price,
            exit_reason=exit_reason,
        )

    async def run(self) -> None:
        """
        Main trading loop.

        Runs continuously during market hours, processing each minute:
        1. Fetch latest bar
        2. Execute pending entry if exists
        3. Check exit conditions for active position
        4. Generate new signals (only if no position)
        """
        logger.info("Starting live trader")

        # Reconcile with broker
        self.position_manager.reconcile_with_broker()

        # Start price monitor
        self.position_manager.start_price_monitor()

        try:
            while not self._stop_requested:
                now = datetime.now(self.et_tz)

                # Check if market is open
                if not self._is_market_hours(now):
                    logger.debug("Outside market hours, waiting...")
                    await asyncio.sleep(60)
                    continue

                # Reset daily stats at market open
                if now.hour == 9 and now.minute == 30:
                    self.position_manager.reset_daily_stats()
                    self.data_buffer.reset_day()
                    self.pending_entry = None
                    self.pending_breach = None

                # Process current minute
                await self._process_minute()

                # Wait for next minute
                next_minute = now.replace(second=0, microsecond=0) + timedelta(minutes=1)
                sleep_seconds = (next_minute - datetime.now(self.et_tz)).total_seconds()
                if sleep_seconds > 0:
                    await asyncio.sleep(sleep_seconds)

        except KeyboardInterrupt:
            logger.info("Received shutdown signal")
        finally:
            self.position_manager.stop_price_monitor()
            logger.info("Live trader stopped")

    def request_stop(self) -> None:
        """Request graceful shutdown of the trading loop."""
        self._stop_requested = True

    def _is_market_hours(self, now: datetime) -> bool:
        """Check if within market hours."""
        market_open = time(9, 30)
        market_close = time(16, 0)
        current_time = now.time()
        return market_open <= current_time < market_close

    def _get_minutes_elapsed(self, now: datetime) -> int:
        """Get minutes elapsed since market open."""
        market_open = now.replace(hour=9, minute=30, second=0, microsecond=0)
        return int((now - market_open).total_seconds() / 60)

    async def _process_minute(self) -> None:
        """Process a single minute tick."""
        now = datetime.now(self.et_tz)
        minutes_elapsed = self._get_minutes_elapsed(now)

        # Fetch latest bar
        bar = self.alpaca_provider.get_latest_bar("SPY")
        if bar is None:
            logger.warning("Failed to get latest bar")
            return

        self.data_buffer.add_bar(bar)
        current_price = bar["close"]

        logger.debug(f"Processing minute {minutes_elapsed}: SPY @ ${current_price:.2f}")

        # Check time restrictions
        is_first_10_minutes = minutes_elapsed < self.config.trading_start_minutes
        is_last_15_minutes = minutes_elapsed >= self.config.trading_end_minutes

        # EOD handling
        if minutes_elapsed >= 390:  # 4:00 PM
            self.pending_entry = None
            position = self.position_manager.get_active_position()
            if position is not None:
                await self._close_position_market(position, "eod_close")
            return

        # Execute pending entry
        if self.pending_entry is not None and self.position_manager.get_active_position() is None:
            await self._execute_pending_entry(current_price, now)
            self.pending_entry = None

        # Check exit conditions for active position
        position = self.position_manager.get_active_position()
        if position is not None:
            await self._check_position_exit(position, current_price, now)

        # Generate new signals (only if no position and not restricted)
        if (
            position is None
            and self.pending_entry is None
            and not is_first_10_minutes
            and not is_last_15_minutes
        ):
            await self._generate_entry_signals(current_price, now, minutes_elapsed)

    async def _execute_pending_entry(self, current_price: float, now: datetime) -> None:
        """Execute a pending entry."""
        pe = self.pending_entry

        # VWAP direction filter at execution time
        # (Skip for reversals/bounces which trade at VWAP level by definition)
        vwap, _, _, _, _ = self.data_buffer.get_vwap_bands()
        if vwap is not None and not pe.is_bounce and not pe.is_news_event:
            if pe.position_type == PositionType.CALL and current_price < vwap:
                logger.info("Entry rejected: long below VWAP")
                return
            if pe.position_type == PositionType.PUT and current_price > vwap:
                logger.info("Entry rejected: short above VWAP")
                return

        # Determine strike and position size
        if pe.is_news_event:
            signal_type = "news_event"
        elif pe.is_bounce_engulfing:
            signal_type = "bounce_engulfing"
        elif pe.is_bounce:
            signal_type = "bounce"
        elif pe.is_breach:
            signal_type = "breach"
        elif pe.is_early_trade:
            signal_type = "early"
        else:
            signal_type = "normal"
        strike, _ = self.position_sizer.select_strike(
            current_price, pe.position_type, signal_type, pe.confidence, now
        )

        # Get option price (placeholder - needs real options data)
        # For now, estimate based on underlying
        estimated_option_price = self._estimate_option_price(current_price, strike, pe.position_type)
        if estimated_option_price is None:
            logger.warning("Could not estimate option price")
            return

        # Calculate position size
        size_result = self.position_sizer.calculate_position_size(
            self.position_manager.capital,
            estimated_option_price,
            signal_type,
            now,
            pe.position_size_bonus,
        )

        if size_result.num_contracts == 0:
            logger.warning("Position size is 0, skipping entry")
            return

        # Create option symbol
        expiry = now.date()  # 0DTE
        option_type = "C" if pe.position_type == PositionType.CALL else "P"
        option_symbol = format_option_symbol("SPY", now, strike, option_type)

        logger.info(
            f"Executing entry: {option_symbol} "
            f"x{size_result.num_contracts} @ ~${estimated_option_price:.2f}"
        )

        # Submit entry order
        order = await self.order_manager.submit_entry_order(
            option_symbol,
            size_result.num_contracts,
            estimated_option_price,
        )

        # Wait for fill
        order = await self.order_manager.wait_for_fill(order, timeout=5.0)

        if order.status != OrderStatus.FILLED:
            logger.warning(f"Entry order not filled: {order.status}")
            return

        fill_price = order.filled_avg_price if order.filled_avg_price > 0 else estimated_option_price

        # Open position in manager
        self.position_manager.open_position(
            position_type=pe.position_type,
            entry_time=now,
            entry_option_price=fill_price,
            option_symbol=option_symbol,
            strike=strike,
            num_contracts=size_result.num_contracts,
            entry_underlying_price=current_price,
            dominant_model=pe.dominant_model,
            max_hold_minutes=pe.max_hold_minutes,
            confluence_count=pe.confluence_count,
            is_breach=pe.is_breach,
            is_reversal=pe.is_reversal,
            is_bounce=pe.is_bounce,
            is_double_breakout=pe.is_double_breakout,
            is_news_event=pe.is_news_event,
        )

    def _estimate_option_price(
        self,
        underlying_price: float,
        strike: float,
        position_type: PositionType,
    ) -> Optional[float]:
        """
        Estimate option price based on underlying price and strike.

        This is a placeholder - in production, use real options quotes.
        """
        # Simple intrinsic + time value estimate
        if position_type == PositionType.CALL:
            intrinsic = max(0, underlying_price - strike)
        else:
            intrinsic = max(0, strike - underlying_price)

        # Add estimated time value (very rough for 0DTE)
        time_value = 0.30  # Rough estimate for near-ATM 0DTE

        return round(intrinsic + time_value, 2)

    async def _check_position_exit(
        self,
        position: LivePosition,
        current_price: float,
        now: datetime,
    ) -> None:
        """Check exit conditions for active position."""
        # Get current option price
        option_price = self._get_option_price(position.option_symbol)
        if option_price is None:
            # Estimate from underlying movement
            pnl_estimate = (current_price - position.entry_underlying_price) / position.entry_underlying_price
            if position.position_type == PositionType.PUT:
                pnl_estimate = -pnl_estimate
            option_price = position.entry_option_price * (1 + pnl_estimate * 3)  # ~3x leverage estimate

        # Build context for signal generation
        ohlcv_df = self.data_buffer.get_ohlcv_df()
        if len(ohlcv_df) < 60:
            # Not enough data for full signal generation
            fused_signal = FusedSignal(action=Signal.HOLD)
        else:
            # Get fused signal
            minutes_elapsed = self._get_minutes_elapsed(now)
            contexts = self._build_contexts(ohlcv_df)
            fused_signal = self.executor.get_signal(contexts, minutes_elapsed)

        # Get VWAP bands
        vwap, upper_1sd, lower_1sd, upper_2sd, lower_2sd = self.data_buffer.get_vwap_bands()

        # Create Position object for exit engine
        from src.backtest.types import Position
        bt_position = Position(
            position_type=position.position_type,
            entry_time=position.entry_time,
            entry_underlying_price=position.entry_underlying_price,
            entry_option_price=position.entry_option_price,
            strike=position.strike,
            entry_idx=0,
            position_value=position.position_value,
            dominant_model=position.dominant_model,
            max_hold_minutes=position.max_hold_minutes,
            barrier_start_time=position.barrier_start_time,
            peak_pnl_pct=position.peak_pnl_pct,
            breakeven_activated=position.breakeven_activated,
            is_runner=position.is_runner,
            runner_start_time=position.runner_start_time,
            runner_max_hold_minutes=position.runner_max_hold_minutes,
            runner_peak_pnl_pct=position.runner_peak_pnl_pct,
            runner_entry_pnl_pct=position.runner_entry_pnl_pct,
            is_breach=position.is_breach,
            is_reversal=position.is_reversal,
            num_contracts=position.num_contracts,
        )

        # Evaluate exit rules
        exit_decision = self.exit_engine.evaluate(
            bt_position,
            option_price,
            fused_signal,
            now,
            upper_2sd=upper_2sd,
            lower_2sd=lower_2sd,
            upper_1sd=upper_1sd,
            lower_1sd=lower_1sd,
        )

        if exit_decision.convert_to_runner:
            self.position_manager.convert_to_runner(now, option_price)
        elif exit_decision.should_exit:
            await self._close_position_market(position, exit_decision.exit_reason)
        elif exit_decision.should_renew:
            self.position_manager.renew_time_barrier(exit_decision.new_max_hold_minutes)

    async def _close_position_market(
        self,
        position: LivePosition,
        exit_reason: str,
    ) -> None:
        """Close position at market."""
        now = datetime.now(self.et_tz)
        current_price = self.data_buffer.get_latest_price()

        # Get option price
        option_price = self._get_option_price(position.option_symbol)
        if option_price is None:
            option_price = position.entry_option_price  # Fallback

        logger.info(f"Closing position: {position.option_symbol}, reason: {exit_reason}")

        # Submit exit order
        order = await self.order_manager.submit_exit_order(
            position.option_symbol,
            position.num_contracts,
            option_price,
        )

        # Wait for fill
        order = await self.order_manager.wait_for_fill(order, timeout=10.0)

        exit_price = order.filled_avg_price if order.filled_avg_price > 0 else option_price

        # Close in manager
        self.position_manager.close_position(
            exit_time=now,
            exit_option_price=exit_price,
            exit_underlying_price=current_price or position.entry_underlying_price,
            exit_reason=exit_reason,
        )

    async def _generate_entry_signals(
        self,
        current_price: float,
        now: datetime,
        minutes_elapsed: int,
    ) -> None:
        """Generate entry signals."""
        ohlcv_df = self.data_buffer.get_ohlcv_df()
        if len(ohlcv_df) < 15:
            return

        # Calculate VWAP bands
        vwap_series, upper_1sd, lower_1sd, upper_2sd, lower_2sd = calculate_vwap_with_bands(ohlcv_df)
        current_vwap = vwap_series.iloc[-1] if len(vwap_series) > 0 else None

        final_action = Signal.HOLD
        is_breach_detected = False
        is_bounce = False
        is_bounce_engulfing = False
        is_news_event = False
        is_early_trade = False
        agreeing_models = ()
        dominant_model = ""
        exit_horizon = 10
        confidence = 0.5
        position_size_bonus = 0.0

        # Get fused signal for confluence checks (needed before VWAP breach)
        fused_signal = None
        if len(ohlcv_df) >= 60:
            contexts = self._build_contexts(ohlcv_df)
            fused_signal = self.executor.get_signal(contexts, minutes_elapsed)

        # No VWAP-based signals before 10:00 AM - wait for structure to develop
        is_structure_ready = minutes_elapsed >= 30  # 30 min after 9:30 = 10:00 AM

        # PRIORITY 0.5: Check for news event (high-volume continuation after opening range)
        idx = len(ohlcv_df) - 1
        if is_structure_ready and self.pending_entry is None:
            ne_signal = detect_news_event(
                ohlcv_df, idx,
                upper_1sd=upper_1sd.iloc[idx] if len(upper_1sd) > idx else None,
                lower_1sd=lower_1sd.iloc[idx] if len(lower_1sd) > idx else None,
                upper_2sd=upper_2sd.iloc[idx] if len(upper_2sd) > idx else None,
                lower_2sd=lower_2sd.iloc[idx] if len(lower_2sd) > idx else None,
            )
            if ne_signal.event_type == NewsEventType.BULLISH:
                final_action = Signal.LONG
                is_news_event = True
                is_breach_detected = True  # Inherit breach exit rules
                dominant_model = "news_event"
                exit_horizon = 10
                confidence = 0.90
                agreeing_models = ("news_event",)
            elif ne_signal.event_type == NewsEventType.BEARISH:
                final_action = Signal.SHORT
                is_news_event = True
                is_breach_detected = True  # Inherit breach exit rules
                dominant_model = "news_event"
                exit_horizon = 10
                confidence = 0.90
                agreeing_models = ("news_event",)

        # PRIORITY 1: Check for VWAP breach
        vwap_breach = detect_breach(
            ohlcv_df, vwap_series, idx,
            upper_1sd_series=upper_1sd,
            lower_1sd_series=lower_1sd,
            upper_2sd_series=upper_2sd,
            lower_2sd_series=lower_2sd,
        )

        # Check pending continuation (only after 10:00 AM)
        if is_structure_ready and self.pending_breach is not None:
            if check_breach_continuation_confirmation(ohlcv_df, idx, self.pending_breach):
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
            self.pending_breach = None

        # Check for new VWAP breach (only after 10:00 AM)
        if is_structure_ready and not is_breach_detected:
            if vwap_breach.breach_type == BreachType.BULLISH:
                final_action = Signal.LONG
                is_breach_detected = True
                agreeing_models = ("vwap_breach_immediate",)
                dominant_model = "breach"
                exit_horizon = 15
                confidence = 0.8
                self.breach_immediate_entries += 1
            elif vwap_breach.breach_type == BreachType.BEARISH:
                final_action = Signal.SHORT
                is_breach_detected = True
                agreeing_models = ("vwap_breach_immediate",)
                dominant_model = "breach"
                exit_horizon = 15
                confidence = 0.8
                self.breach_immediate_entries += 1
            elif vwap_breach.breach_type in (BreachType.BULLISH_POTENTIAL, BreachType.BEARISH_POTENTIAL):
                self.pending_breach = vwap_breach

        # Model confluence check for VWAP breach: reject if contrasting, bonus if confirming
        # (News events bypass confluence — they're exogenous signals)
        if is_breach_detected and not is_news_event and final_action != Signal.HOLD and fused_signal is not None:
            confluence_ok, confluence_bonus = check_model_confluence(fused_signal, final_action)
            if not confluence_ok:
                logger.info("VWAP breach rejected: contrasting model signal")
                final_action = Signal.HOLD
                is_breach_detected = False
            else:
                position_size_bonus = confluence_bonus

        # PRIORITY 1.8: Check for VWAP bounce
        if is_structure_ready and not is_breach_detected and vwap_series is not None and len(ohlcv_df) > 0:
            bounce_signal = detect_vwap_bounce(ohlcv_df, vwap_series, idx)
            if bounce_signal.bounce_type != BounceType.NONE:
                if bounce_signal.bounce_type in (BounceType.BULLISH_HAMMER, BounceType.BULLISH_ENGULFING):
                    bounce_action = Signal.LONG
                else:
                    bounce_action = Signal.SHORT
                if fused_signal is not None:
                    confluence_ok, confluence_bonus = check_model_confluence(fused_signal, bounce_action)
                else:
                    confluence_ok, confluence_bonus = True, 0.0
                if confluence_ok:
                    final_action = bounce_action
                    is_bounce = True
                    is_bounce_engulfing = bounce_signal.is_engulfing
                    dominant_model = "bounce"
                    exit_horizon = 15
                    confidence = 0.75
                    agreeing_models = ("bounce",)
                    position_size_bonus = confluence_bonus
                else:
                    logger.info("Bounce rejected: contrasting model signal")

        # PRIORITY 2: Model fusion signals (if no breach or bounce)
        if not is_breach_detected and not is_bounce and fused_signal is not None:
            if fused_signal.action != Signal.HOLD:
                # Check heuristic filter
                if self.heuristic_model is not None:
                    lookback = min(30, len(ohlcv_df))
                    ohlcv_window = ohlcv_df.iloc[-lookback:]
                    heuristic_action = self.heuristic_model.get_signal(ohlcv_window, fused_signal.action)

                    if heuristic_action == fused_signal.action:
                        # Check VWAP direction
                        if current_vwap is not None:
                            vwap_ok, _ = self.heuristic_model.check_vwap_direction(
                                fused_signal.action, current_price, current_vwap
                            )
                            if vwap_ok:
                                final_action = fused_signal.action
                                agreeing_models = fused_signal.agreeing_models
                                dominant_model = fused_signal.dominant_model
                                exit_horizon = fused_signal.exit_horizon_minutes
                                confidence = fused_signal.confidence
                            else:
                                self.vwap_direction_rejections += 1
                else:
                    final_action = fused_signal.action
                    agreeing_models = fused_signal.agreeing_models
                    dominant_model = fused_signal.dominant_model
                    exit_horizon = fused_signal.exit_horizon_minutes
                    confidence = fused_signal.confidence

        # VWAP direction filter for non-breach/non-bounce trades
        if final_action != Signal.HOLD and not is_breach_detected and not is_bounce and current_vwap is not None:
            if final_action == Signal.LONG and current_price < current_vwap:
                final_action = Signal.HOLD
            elif final_action == Signal.SHORT and current_price > current_vwap:
                final_action = Signal.HOLD

        # Volume bonus: +1% position size if signal candle volume >= 1.25x previous candle
        if final_action != Signal.HOLD and len(ohlcv_df) >= 2:
            signal_vol = ohlcv_df.iloc[-1]['volume']
            prev_vol = ohlcv_df.iloc[-2]['volume']
            if prev_vol > 0 and signal_vol >= 1.25 * prev_vol:
                position_size_bonus += 0.01

        # Combine breach flags
        is_breach = is_breach_detected or is_bounce or is_news_event

        # Create pending entry
        if final_action == Signal.LONG:
            self.pending_entry = PendingEntry(
                position_type=PositionType.CALL,
                signal_time=now,
                underlying_price=current_price,
                dominant_model=dominant_model,
                max_hold_minutes=exit_horizon,
                confidence=confidence,
                confluence_count=len(agreeing_models),
                agreeing_models=agreeing_models,
                is_early_trade=is_early_trade,
                is_breach=is_breach,
                position_size_bonus=position_size_bonus,
                is_bounce=is_bounce,
                is_bounce_engulfing=is_bounce_engulfing,
                is_news_event=is_news_event,
            )
        elif final_action == Signal.SHORT:
            self.pending_entry = PendingEntry(
                position_type=PositionType.PUT,
                signal_time=now,
                underlying_price=current_price,
                dominant_model=dominant_model,
                max_hold_minutes=exit_horizon,
                confidence=confidence,
                confluence_count=len(agreeing_models),
                agreeing_models=agreeing_models,
                is_early_trade=is_early_trade,
                is_breach=is_breach,
                position_size_bonus=position_size_bonus,
                is_bounce=is_bounce,
                is_bounce_engulfing=is_bounce_engulfing,
                is_news_event=is_news_event,
            )

        # Track last signal for live dashboard
        self._last_signal_action = final_action.value if hasattr(final_action, 'value') else str(final_action)
        self._last_signal_source = dominant_model or ("breach" if is_breach_detected else "none")
        self._last_signal_confidence = confidence
        self._last_signal_agreeing_models = agreeing_models

    def _build_contexts(self, ohlcv_df: pd.DataFrame) -> dict:
        """Build context tensors for model inference."""
        # Convert OHLCV to features (simplified - real impl would match feature engineering)
        features = ohlcv_df[["open", "high", "low", "close", "volume"]].values
        feature_tensor = torch.tensor(features, dtype=torch.float32, device=self.device)

        # Build contexts for each model
        contexts = {}
        context_lengths = {"1m": 5, "5m": 15, "15m": 60}

        for model_name, ctx_len in context_lengths.items():
            if len(feature_tensor) >= ctx_len:
                contexts[model_name] = feature_tensor[-ctx_len:].unsqueeze(0)

        return contexts
