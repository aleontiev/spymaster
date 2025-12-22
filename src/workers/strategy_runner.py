"""
Strategy Runner for executing trading strategies on cached data.

Responsible for:
- Listening for "cache_updated" events from LoaderWorker
- Running active strategies using ModelGraphExecutor
- Maintaining position state per strategy
- Recording and emitting action decisions

Each strategy defines a model_graph DAG that specifies how models
connect and execute.
"""
import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime, date, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import pandas as pd
import torch

from src.workers.event_bus import EventBus, Event
from src.workers.model_registry import ModelRegistry, ModelType
from src.workers.graph_executor import ModelGraphExecutor, GraphNode
from src.workers.loader_worker import load_minute_for_strategy
from src.model.policy import EntryAction, ExitAction
from src.data.loader import NORMALIZED_CACHE_DIR

logger = logging.getLogger(__name__)


class ExitMode(str, Enum):
    """Exit policy modes."""
    RULE_BASED = "rule_based"
    NEURAL = "neural"
    CONTINUOUS_SIGNAL = "continuous_signal"


@dataclass
class PositionState:
    """State of an open position."""

    position_type: int  # +1 for call, -1 for put
    entry_price: float
    entry_time: datetime
    entry_action: EntryAction
    bars_held: int = 0
    unrealized_pnl: float = 0.0
    peak_pnl: float = 0.0

    def update(
        self,
        current_price: float,
        time_to_close: float,
    ) -> Dict[str, Any]:
        """
        Update position state and return context dict.

        Args:
            current_price: Current underlying price
            time_to_close: Hours until market close

        Returns:
            Position context for exit policy
        """
        self.bars_held += 1

        # Simplified P&L calculation (would need option price in real system)
        # For now, use underlying price change as proxy
        price_change_pct = (current_price - self.entry_price) / self.entry_price
        self.unrealized_pnl = price_change_pct * self.position_type * 100  # Simulate leverage

        if self.unrealized_pnl > self.peak_pnl:
            self.peak_pnl = self.unrealized_pnl

        return {
            "position_type": self.position_type,
            "unrealized_pnl": self.unrealized_pnl / 100.0,  # Normalize to [0, 1] range
            "bars_held": self.bars_held,
            "time_to_close": time_to_close,
            "entry_action": self.entry_action,
            "peak_pnl": self.peak_pnl,
        }


@dataclass
class StrategyAction:
    """Output of a strategy decision."""

    timestamp: datetime
    strategy_name: str
    action: EntryAction
    confidence: float
    exit_action: Optional[ExitAction] = None
    exit_reason: Optional[str] = None
    intermediate_outputs: Dict[str, Any] = field(default_factory=dict)
    position_type: Optional[int] = None  # +1 call, -1 put, None = no position


@dataclass
class StrategyConfig:
    """Configuration for a single strategy."""

    name: str
    entry_confidence: float = 0.4  # Minimum confidence to enter
    exit_mode: ExitMode = ExitMode.RULE_BASED
    model_graph: List[GraphNode] = field(default_factory=list)

    # Rule-based exit parameters
    take_profit_pct: Optional[float] = 50.0
    risk_reward_ratio: Optional[float] = 2.0
    stop_loss_pct: Optional[float] = None

    # Continuous signal parameters
    counter_signal_confirmation: int = 2
    plateau_window: int = 15
    plateau_ratio: float = 0.5

    # Patch parameters
    patch_length: int = 64

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "StrategyConfig":
        """Create StrategyConfig from dictionary."""
        # Parse model graph
        model_graph = []
        for node_dict in d.get("model_graph", []):
            model_graph.append(GraphNode.from_dict(node_dict))

        return cls(
            name=d["name"],
            entry_confidence=d.get("entry_confidence", 0.4),
            exit_mode=ExitMode(d.get("exit_mode", "rule_based")),
            model_graph=model_graph,
            take_profit_pct=d.get("take_profit_pct", 50.0),
            risk_reward_ratio=d.get("risk_reward_ratio", 2.0),
            stop_loss_pct=d.get("stop_loss_pct"),
            counter_signal_confirmation=d.get("counter_signal_confirmation", 2),
            plateau_window=d.get("plateau_window", 15),
            plateau_ratio=d.get("plateau_ratio", 0.5),
            patch_length=d.get("patch_length", 64),
        )


class StrategyRunner:
    """
    Worker that runs active strategies on cached data.

    Triggered when Loader emits "cache_updated" event.
    Produces (minute, strategy, action) decisions.

    Each strategy has a model_graph DAG that defines how models
    connect and execute.
    """

    def __init__(
        self,
        strategies: List[StrategyConfig],
        model_registry: ModelRegistry,
        event_bus: Optional[EventBus] = None,
        cache_dir: Path = NORMALIZED_CACHE_DIR,
        underlying: str = "SPY",
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ) -> None:
        """
        Initialize the StrategyRunner.

        Args:
            strategies: List of strategy configurations
            model_registry: Registry containing loaded models
            event_bus: Event bus for receiving and emitting events
            cache_dir: Directory for normalized cache
            underlying: Stock ticker
            device: Device for tensor operations
        """
        self.strategies = strategies
        self.model_registry = model_registry
        self.event_bus = event_bus
        self.cache_dir = Path(cache_dir)
        self.underlying = underlying
        self.device = device

        # Create executor for each strategy's model graph
        self.executors: Dict[str, ModelGraphExecutor] = {}
        for strategy in strategies:
            self.executors[strategy.name] = ModelGraphExecutor(
                model_registry=model_registry,
                graph=strategy.model_graph,
                patch_length=strategy.patch_length,
                device=device,
            )

        # Position state per strategy
        self.position_state: Dict[str, Optional[PositionState]] = {
            s.name: None for s in strategies
        }

        # Action history for continuous signal exit
        self.action_history: Dict[str, List[EntryAction]] = {
            s.name: [] for s in strategies
        }

        # State
        self._running = False
        self._current_date: Optional[date] = None

        # Decision log
        self.decisions: List[StrategyAction] = []

    def _reset_daily_state(self, target_date: date) -> None:
        """Reset state for a new trading day."""
        if self._current_date != target_date:
            logger.info(f"Resetting strategy state for {target_date}")
            self._current_date = target_date

            # Close all positions at day boundary
            for strategy_name in self.position_state:
                if self.position_state[strategy_name] is not None:
                    logger.info(f"Force closing position for {strategy_name} at day boundary")
                    self.position_state[strategy_name] = None

            # Clear action history
            for strategy_name in self.action_history:
                self.action_history[strategy_name].clear()

            # Clear resample cache in registry
            self.model_registry.clear_resample_cache()

    def _get_position_context(
        self,
        strategy_name: str,
        current_price: float,
        time_to_close: float,
    ) -> Optional[Dict[str, Any]]:
        """Get position context for a strategy if in trade."""
        position = self.position_state.get(strategy_name)
        if position is None:
            return None
        return position.update(current_price, time_to_close)

    def _should_enter(
        self,
        action: EntryAction,
        confidence: float,
        strategy: StrategyConfig,
    ) -> bool:
        """Check if we should enter a position."""
        if action == EntryAction.HOLD:
            return False
        return confidence >= strategy.entry_confidence

    def _open_position(
        self,
        strategy_name: str,
        action: EntryAction,
        price: float,
        timestamp: datetime,
    ) -> None:
        """Open a new position."""
        # Determine position type from action
        if action in (EntryAction.BUY_CALL_ATM, EntryAction.BUY_CALL_OTM):
            position_type = 1  # Long call
        else:
            position_type = -1  # Long put

        self.position_state[strategy_name] = PositionState(
            position_type=position_type,
            entry_price=price,
            entry_time=timestamp,
            entry_action=action,
        )

        logger.info(f"Opened {action.name} position for {strategy_name} at ${price:.2f}")

    def _close_position(
        self,
        strategy_name: str,
        price: float,
        reason: str,
    ) -> None:
        """Close an existing position."""
        position = self.position_state.get(strategy_name)
        if position is None:
            return

        final_pnl = position.unrealized_pnl
        logger.info(
            f"Closed position for {strategy_name}: "
            f"{position.entry_action.name} @ ${position.entry_price:.2f} → ${price:.2f}, "
            f"P&L: {final_pnl:+.1f}%, Reason: {reason}"
        )

        self.position_state[strategy_name] = None

    async def run_strategies(
        self,
        target_minute: datetime,
        target_date: date,
    ) -> List[StrategyAction]:
        """
        Run all strategies for a given minute.

        Args:
            target_minute: The minute to process
            target_date: The trading date

        Returns:
            List of StrategyAction for all strategies
        """
        # Reset state if new day
        self._reset_daily_state(target_date)

        # Load market data from cache
        df = await load_minute_for_strategy(
            target_minute=target_minute,
            underlying=self.underlying,
            cache_dir=self.cache_dir,
        )

        if df is None or df.empty:
            logger.warning(f"No cache data for {target_minute}")
            return []

        # Load full day's data for context (needed for patch creation)
        from src.data.loader import CACHE_VERSION
        cache_path = self.cache_dir / f"{self.underlying}_{CACHE_VERSION}_{target_date}.parquet"

        if not cache_path.exists():
            logger.warning(f"Full day cache not found: {cache_path}")
            return []

        full_df = pd.read_parquet(cache_path)
        if "timestamp" in full_df.columns:
            full_df = full_df.set_index("timestamp")

        # Find current index in full dataframe
        idx = len(full_df) - 1

        # Get current price and time to close
        current_price = df["close"].iloc[-1] if "close" in df.columns else 0.0
        time_to_close = df["time_to_close"].iloc[-1] if "time_to_close" in df.columns else 0.0

        actions = []

        for strategy in self.strategies:
            try:
                action = await self._run_single_strategy(
                    strategy=strategy,
                    df=full_df,
                    idx=idx,
                    current_price=current_price,
                    time_to_close=time_to_close,
                    timestamp=target_minute,
                )
                actions.append(action)

            except Exception as e:
                logger.error(f"Error running strategy {strategy.name}: {e}")
                # Create error action
                actions.append(StrategyAction(
                    timestamp=target_minute,
                    strategy_name=strategy.name,
                    action=EntryAction.HOLD,
                    confidence=0.0,
                    intermediate_outputs={"error": str(e)},
                ))

        # Store decisions
        self.decisions.extend(actions)

        return actions

    async def _run_single_strategy(
        self,
        strategy: StrategyConfig,
        df: pd.DataFrame,
        idx: int,
        current_price: float,
        time_to_close: float,
        timestamp: datetime,
    ) -> StrategyAction:
        """
        Run a single strategy.

        Args:
            strategy: Strategy configuration
            df: Full day's market data
            idx: Current index in dataframe
            current_price: Current underlying price
            time_to_close: Hours until market close
            timestamp: Current timestamp

        Returns:
            StrategyAction for this strategy
        """
        executor = self.executors[strategy.name]

        # Get position context
        position_context = self._get_position_context(
            strategy.name,
            current_price,
            time_to_close,
        )

        # Execute the model graph
        result = executor.execute(
            market_data=df,
            idx=idx,
            position_context=position_context,
        )

        # Extract entry action
        entry_action = result.get("entry_action", EntryAction.HOLD)
        confidence = result.get("confidence", 0.0)

        # Update action history
        self.action_history[strategy.name].append(entry_action)

        # Determine exit action
        exit_action = result.get("exit_action")
        exit_reason = result.get("exit_reason")

        # Handle position logic
        position = self.position_state.get(strategy.name)

        if position is None:
            # No position - check if we should enter
            if self._should_enter(entry_action, confidence, strategy):
                self._open_position(
                    strategy.name,
                    entry_action,
                    current_price,
                    timestamp,
                )
        else:
            # In position - check if we should exit
            if exit_action == ExitAction.CLOSE:
                self._close_position(strategy.name, current_price, exit_reason or "signal")

            # For continuous signal mode, check for counter-signal
            elif strategy.exit_mode == ExitMode.CONTINUOUS_SIGNAL:
                exit_decision = self._check_continuous_exit(
                    strategy,
                    entry_action,
                    position,
                    time_to_close,
                )
                if exit_decision[0] == ExitAction.CLOSE:
                    exit_action = exit_decision[0]
                    exit_reason = exit_decision[1]
                    self._close_position(strategy.name, current_price, exit_reason)

        return StrategyAction(
            timestamp=timestamp,
            strategy_name=strategy.name,
            action=entry_action,
            confidence=confidence,
            exit_action=exit_action,
            exit_reason=exit_reason,
            intermediate_outputs={
                "embeddings_shape": {k: v.shape for k, v in result.get("embeddings", {}).items()},
            },
            position_type=position.position_type if position else None,
        )

    def _check_continuous_exit(
        self,
        strategy: StrategyConfig,
        current_action: EntryAction,
        position: PositionState,
        time_to_close: float,
    ) -> Tuple[ExitAction, Optional[str]]:
        """
        Check continuous signal exit conditions.

        Args:
            strategy: Strategy config
            current_action: Current entry action signal
            position: Current position state
            time_to_close: Hours until market close

        Returns:
            (exit_action, reason) tuple
        """
        history = self.action_history[strategy.name]

        # Check for end-of-day
        if time_to_close < 5 / 60:  # < 5 minutes
            return (ExitAction.CLOSE, "eod_exit")

        # Determine if current action is counter-signal
        is_counter = False
        if position.position_type == 1:  # Long call
            is_counter = current_action in (EntryAction.BUY_PUT_ATM, EntryAction.BUY_PUT_OTM)
        else:  # Long put
            is_counter = current_action in (EntryAction.BUY_CALL_ATM, EntryAction.BUY_CALL_OTM)

        # Check counter-signal confirmation
        if len(history) >= strategy.counter_signal_confirmation:
            recent = history[-strategy.counter_signal_confirmation:]
            all_counter = all(
                self._is_counter_signal(a, position.position_type)
                for a in recent
            )
            if all_counter:
                return (ExitAction.CLOSE, "counter_signal")

        # Check plateau (momentum exhaustion)
        if len(history) >= strategy.plateau_window:
            recent = history[-strategy.plateau_window:]
            hold_or_counter = sum(
                1 for a in recent
                if a == EntryAction.HOLD or self._is_counter_signal(a, position.position_type)
            )
            ratio = hold_or_counter / strategy.plateau_window
            if ratio >= strategy.plateau_ratio:
                return (ExitAction.CLOSE, "plateau")

        return (ExitAction.HOLD_POSITION, None)

    def _is_counter_signal(self, action: EntryAction, position_type: int) -> bool:
        """Check if action is a counter-signal for the position."""
        if position_type == 1:  # Long call
            return action in (EntryAction.BUY_PUT_ATM, EntryAction.BUY_PUT_OTM)
        else:  # Long put
            return action in (EntryAction.BUY_CALL_ATM, EntryAction.BUY_CALL_OTM)

    async def on_cache_updated(self, event: Event) -> None:
        """
        Handle cache_updated event.

        Runs all strategies and emits action decisions.
        """
        event_data = event.data
        target_minute = event_data.get("timestamp")
        target_date = event_data.get("date")

        if target_minute is None or target_date is None:
            logger.error("Invalid cache_updated event: missing timestamp or date")
            return

        logger.debug(f"Running strategies for {target_minute}")

        try:
            actions = await self.run_strategies(target_minute, target_date)

            # Emit action decisions
            if self.event_bus is not None and actions:
                for action in actions:
                    await self.event_bus.emit(
                        "action_decided",
                        {
                            "timestamp": action.timestamp,
                            "strategy": action.strategy_name,
                            "action": action.action.name,
                            "confidence": action.confidence,
                            "exit_action": action.exit_action.name if action.exit_action else None,
                            "exit_reason": action.exit_reason,
                            "position_type": action.position_type,
                        },
                    )

            # Log summary
            for action in actions:
                if action.action != EntryAction.HOLD or action.exit_action == ExitAction.CLOSE:
                    logger.info(
                        f"{action.strategy_name}: {action.action.name} "
                        f"(conf={action.confidence:.2f})"
                        + (f" | EXIT: {action.exit_reason}" if action.exit_action == ExitAction.CLOSE else "")
                    )

        except Exception as e:
            logger.error(f"Error in on_cache_updated: {e}")

    async def run(self) -> None:
        """
        Main loop: listen for cache_updated events and run strategies.
        """
        if self.event_bus is None:
            logger.error("StrategyRunner requires an event bus")
            return

        logger.info(f"StrategyRunner started with {len(self.strategies)} strategies")
        self._running = True

        # Subscribe to events
        self.event_bus.subscribe("cache_updated", self.on_cache_updated)

        try:
            while self._running:
                await asyncio.sleep(1)

        except asyncio.CancelledError:
            logger.info("StrategyRunner cancelled")
        finally:
            self.event_bus.unsubscribe("cache_updated", self.on_cache_updated)
            logger.info("StrategyRunner stopped")

    def stop(self) -> None:
        """Stop the strategy runner."""
        self._running = False

    @property
    def is_running(self) -> bool:
        """Check if the runner is running."""
        return self._running

    def get_decisions(self) -> List[StrategyAction]:
        """Get all recorded decisions."""
        return self.decisions

    def get_position(self, strategy_name: str) -> Optional[PositionState]:
        """Get current position for a strategy."""
        return self.position_state.get(strategy_name)
