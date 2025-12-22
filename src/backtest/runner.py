"""
Strategy Runner for Backtesting.

Supports two modes:
1. Legacy mode: Single PolicyNetwork with 4 actions (HOLD, BUY_CALL, BUY_PUT, CLOSE)
2. Dual-policy mode: EntryPolicy (3 actions) + ExitPolicy (2 actions with position context)

Usage:
    # Dual-policy mode (recommended):
    uv run python -m src.backtest.runner \\
        --lejepa checkpoints/lejepa_best.pt \\
        --entry_policy checkpoints/entry_policy_best.pt \\
        --exit_policy checkpoints/exit_policy_best.pt

    # Legacy mode:
    uv run python -m src.backtest.runner \\
        --lejepa checkpoints/lejepa_best.pt \\
        --policy checkpoints/policy_best.pt
"""
import argparse
from datetime import datetime, time, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch

from src.backtest.engine import (
    BacktestConfig,
    BacktestEngine,
    OrderSide,
    PositionType,
)
from src.backtest.report import generate_html_report
from src.data.processing import MarketPatch
from src.data.loader import load_normalized_data
from src.model.lejepa import LeJEPA
from src.model.policy import (
    PolicyNetwork,
    TradingAction,
    EntryPolicy,
    ExitPolicy,
    RuleBasedExitPolicy,
    ContinuousSignalExitPolicy,
    EntryAction,
    ExitAction,
)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Run backtest with trained policy")

    # Strategy-based config (loads from strategies/<name>.json)
    parser.add_argument(
        "--strategy",
        type=str,
        default=None,
        help="Strategy name to load config from (e.g., 'my_strategy' loads strategies/my_strategy.json)",
    )

    parser.add_argument(
        "--lejepa",
        type=str,
        required=False,  # Not required if using --strategy
        help="Path to LeJEPA checkpoint",
    )

    # Dual-policy mode (recommended)
    parser.add_argument(
        "--entry_policy",
        type=str,
        default=None,
        help="Path to entry policy checkpoint (dual-policy mode)",
    )
    parser.add_argument(
        "--exit_policy",
        type=str,
        default=None,
        help="Path to exit policy checkpoint (dual-policy mode)",
    )

    # Legacy single-policy mode
    parser.add_argument(
        "--policy",
        type=str,
        default=None,
        help="Path to legacy policy checkpoint (single-policy mode)",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="data/polygon/stocks",
        help="Directory with stocks parquet files",
    )
    parser.add_argument(
        "--pattern",
        type=str,
        default="SPY_*.parquet",
        help="Glob pattern for parquet files",
    )
    parser.add_argument(
        "--options_dir",
        type=str,
        default="data/polygon/options",
        help="Directory with options parquet files",
    )
    parser.add_argument(
        "--oi_path",
        type=str,
        default="data/polygon/oi/spy_oi_2020-2025.parquet",
        help="Path to Open Interest parquet file for GEX features",
    )
    parser.add_argument(
        "--initial_capital",
        type=float,
        default=100_000,
        help="Initial capital",
    )
    parser.add_argument(
        "--max_position_pct",
        type=float,
        default=0.05,
        help="Max position size as percentage of capital",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Verbose output",
    )
    parser.add_argument(
        "--confidence_threshold",
        type=float,
        default=0.0,
        help="Minimum confidence to execute non-HOLD action (0.0 = always trade)",
    )
    parser.add_argument(
        "--stochastic",
        action="store_true",
        help="Use stochastic action selection instead of argmax",
    )
    parser.add_argument(
        "--force_trades",
        action="store_true",
        help="Force trades by using second-best action if HOLD is predicted",
    )
    parser.add_argument(
        "--action_cooldown",
        type=int,
        default=1,
        help="Minutes to wait between actions (default: 1)",
    )
    parser.add_argument(
        "--min_signal_threshold",
        type=float,
        default=0.20,
        help="Minimum probability for non-HOLD action when using --force_trades (default: 0.20)",
    )
    parser.add_argument(
        "--entry_confidence",
        type=float,
        default=0.4,
        help="Minimum confidence for entry policy to open position (dual-policy mode, default: 0.4)",
    )

    # Rule-based exit policy options
    parser.add_argument(
        "--rule_based_exit",
        action="store_true",
        help="Use rule-based exit policy instead of neural network exit policy",
    )
    parser.add_argument(
        "--take_profit_pct",
        type=float,
        default=50.0,
        help="Take profit threshold in percent (default: 50.0 = +50%%)",
    )
    parser.add_argument(
        "--risk_reward_ratio",
        type=float,
        default=2.0,
        help="Risk:Reward ratio (default: 2.0 = 1:2, so SL = TP/2)",
    )
    parser.add_argument(
        "--stop_loss_pct",
        type=float,
        default=None,
        help="Explicit stop loss percent (overrides R:R calculation if provided)",
    )
    parser.add_argument(
        "--time_stop_hours",
        type=float,
        default=None,
        help="Time-based stop: close if held longer than this (None = disabled)",
    )
    parser.add_argument(
        "--eod_exit_minutes",
        type=float,
        default=5.0,
        help="Minutes before close to force exit (default: 5)",
    )

    # Continuous signal exit mode options
    parser.add_argument(
        "--continuous_signal",
        action="store_true",
        help="Use continuous signal exit mode (entry policy determines exits)",
    )
    parser.add_argument(
        "--counter_signal_confirmation",
        type=int,
        default=2,
        help="Consecutive counter-signals needed to exit and flip (default: 2)",
    )
    parser.add_argument(
        "--plateau_window",
        type=int,
        default=15,
        help="Rolling window in minutes for plateau detection (default: 15)",
    )
    parser.add_argument(
        "--plateau_ratio",
        type=float,
        default=0.5,
        help="If same-direction ratio < (1 - this), exit (default: 0.5)",
    )
    parser.add_argument(
        "--flip_on_counter",
        action="store_true",
        default=True,
        help="Flip to counter-position on counter-signal exit (default: True)",
    )
    parser.add_argument(
        "--no_flip_on_counter",
        action="store_true",
        help="Disable position flipping on counter-signal",
    )

    parser.add_argument(
        "--start_date",
        type=str,
        default=None,
        help="Start date for backtest (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--end_date",
        type=str,
        default=None,
        help="End date for backtest (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--report",
        type=str,
        default=None,
        help="Path to save HTML report (e.g., reports/backtest.html)",
    )
    parser.add_argument(
        "--no_report",
        action="store_true",
        help="Disable HTML report generation",
    )
    parser.add_argument(
        "--normalized",
        action="store_true",
        help="Use pre-normalized cached data (23 features, matches LeJEPA training)",
    )
    parser.add_argument(
        "--underlying",
        type=str,
        default="SPY",
        help="Underlying symbol (default: SPY)",
    )

    return parser.parse_args()


def get_device(device_arg: str) -> torch.device:
    """Determine device."""
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)


class DualPolicyRunner:
    """
    Runs backtest using dual-policy architecture (EntryPolicy + ExitPolicy).

    - EntryPolicy: Supervised classifier for trade direction (HOLD/BUY_CALL/BUY_PUT)
    - ExitPolicy: RL-trained policy OR rule-based policy OR continuous signal exit

    Supports:
    - Neural network exit policies (ExitPolicy)
    - Rule-based exit policies (RuleBasedExitPolicy) with TP/SL thresholds
    - Continuous signal exit (ContinuousSignalExitPolicy) using entry policy for exits
    """

    def __init__(
        self,
        lejepa: LeJEPA,
        entry_policy: EntryPolicy,
        exit_policy: Union[ExitPolicy, RuleBasedExitPolicy, ContinuousSignalExitPolicy],
        config: BacktestConfig,
        device: torch.device,
        patch_length: int = 32,
        action_cooldown: int = 1,
        verbose: bool = False,
        entry_confidence: float = 0.4,
        use_rule_based_exit: bool = False,
        use_continuous_signal: bool = False,
    ) -> None:
        """
        Initialize dual-policy runner.

        Args:
            lejepa: Pre-trained LeJEPA model
            entry_policy: Trained entry policy (supervised)
            exit_policy: Exit policy - NN-based, rule-based, or continuous signal
            config: Backtest configuration
            device: Torch device
            patch_length: Length of input patches
            action_cooldown: Minutes between actions
            verbose: Verbose logging
            entry_confidence: Min confidence for entry policy
            use_rule_based_exit: Whether using rule-based exit policy
            use_continuous_signal: Whether using continuous signal exit mode
        """
        self.lejepa = lejepa
        self.entry_policy = entry_policy
        self.exit_policy = exit_policy
        self.use_rule_based_exit = use_rule_based_exit
        self.use_continuous_signal = use_continuous_signal or isinstance(exit_policy, ContinuousSignalExitPolicy)
        self.engine = BacktestEngine(config)
        self.device = device
        self.patch_length = patch_length
        self.action_cooldown = action_cooldown
        self.verbose = verbose
        self.entry_confidence = entry_confidence
        # Use explicit flag if provided, otherwise auto-detect
        self.use_rule_based_exit = use_rule_based_exit or isinstance(exit_policy, RuleBasedExitPolicy)

        self.patcher = None
        self._patch_length = patch_length
        self.use_normalized_data = False  # Set to True when using pre-normalized data
        self._data_tensor = None  # Pre-converted tensor for normalized data
        self._raw_close_prices = None  # Raw close prices for trading decisions
        self.use_3class_entry = False  # Set to True when using 3-class entry policy

        # State
        self.last_action_time: Optional[datetime] = None
        self.current_position_type: Optional[PositionType] = None
        self.position_entry_idx: Optional[int] = None
        self.position_entry_price: Optional[float] = None
        self.position_entry_time: Optional[datetime] = None
        self.position_entry_option_price: Optional[float] = None

    def set_normalized_data(self, df: pd.DataFrame, raw_close_prices: np.ndarray) -> None:
        """Pre-convert normalized DataFrame to tensor for efficient access.

        Args:
            df: Pre-normalized DataFrame (23 features)
            raw_close_prices: Array of raw close prices aligned with df index
        """
        self.use_normalized_data = True
        data = torch.tensor(df.values, dtype=torch.float32)
        data = torch.nan_to_num(data, nan=0.0, posinf=10.0, neginf=-10.0)
        data = torch.clamp(data, -10, 10)
        self._data_tensor = data.to(self.device)
        self._raw_close_prices = raw_close_prices

    @torch.no_grad()
    def get_embedding(self, df: pd.DataFrame, current_idx: int) -> Optional[torch.Tensor]:
        """Get LeJEPA embedding for current state."""
        if current_idx < self.patch_length:
            return None

        start_idx = current_idx - self.patch_length

        if self.use_normalized_data and self._data_tensor is not None:
            # Use pre-normalized data directly (no MarketPatch processing needed)
            patch_tensor = self._data_tensor[start_idx:current_idx].unsqueeze(0)
        else:
            # Legacy path: use MarketPatch for feature engineering
            try:
                patch = self.patcher.create_patch(df, start_idx)
            except ValueError:
                return None
            patch_tensor = patch.unsqueeze(0).to(self.device)

        with torch.amp.autocast(device_type=self.device.type, dtype=torch.bfloat16):
            embedding = self.lejepa.context_encoder(patch_tensor, return_all_tokens=False)

        return embedding.float()

    @torch.no_grad()
    def get_entry_action(
        self,
        embedding: torch.Tensor,
        deterministic: bool = True,
    ) -> Tuple[EntryAction, float]:
        """Get entry action from entry policy."""
        output = self.entry_policy(embedding)
        probs = output["action_probs"][0]

        if deterministic:
            action_idx = probs.argmax().item()
        else:
            action_idx = torch.multinomial(probs, 1).item()

        confidence = probs[action_idx].item()

        # Map 3-class output to 5-class EntryAction
        if self.use_3class_entry:
            # 3-class: 0=HOLD, 1=BUY_CALL, 2=BUY_PUT
            # Map to: HOLD, BUY_CALL_ATM, BUY_PUT_ATM
            if action_idx == 0:
                action = EntryAction.HOLD
            elif action_idx == 1:
                action = EntryAction.BUY_CALL_ATM
            else:  # action_idx == 2
                action = EntryAction.BUY_PUT_ATM
            return action, confidence
        else:
            return EntryAction(action_idx), confidence

    @torch.no_grad()
    def get_exit_action(
        self,
        embedding: torch.Tensor,
        position_context: torch.Tensor,
        current_time: datetime,
        unrealized_pnl_pct: float = 0.0,
        deterministic: bool = True,
    ) -> Tuple[ExitAction, float]:
        """Get exit action from exit policy (NN or rule-based)."""
        if self.use_rule_based_exit:
            # Rule-based exit policy
            market_close = datetime.combine(current_time.date(), time(16, 0))
            hours_to_close = max(0, (market_close - current_time).total_seconds() / 3600)

            # Calculate hours held
            hours_held = None
            if self.position_entry_time is not None:
                hours_held = (current_time - self.position_entry_time).total_seconds() / 3600

            action = self.exit_policy.get_action(
                unrealized_pnl_pct=unrealized_pnl_pct,
                time_to_close_hours=hours_to_close,
                hours_held=hours_held,
            )
            # Rule-based always has 100% confidence
            return action, 1.0
        else:
            # Neural network exit policy
            output = self.exit_policy(embedding, position_context.unsqueeze(0).to(self.device))
            probs = output["action_probs"][0]

            if deterministic:
                action_idx = probs.argmax().item()
            else:
                action_idx = torch.multinomial(probs, 1).item()

            confidence = probs[action_idx].item()
            return ExitAction(action_idx), confidence

    def get_position_context(
        self,
        current_idx: int,
        current_price: float,
        current_time: datetime,
    ) -> torch.Tensor:
        """
        Get position context for exit policy.

        Returns tensor with:
        - position_type: +1 (call) or -1 (put)
        - unrealized_pnl: Normalized current P&L
        - bars_held: Normalized holding period
        - time_to_close: Hours until close (normalized)
        """
        # Position type
        position_type = 1.0 if self.current_position_type == PositionType.CALL else -1.0

        # Unrealized P&L (simplified: just price change direction)
        if self.position_entry_price is not None:
            price_change = (current_price - self.position_entry_price) / self.position_entry_price
            if self.current_position_type == PositionType.PUT:
                price_change = -price_change  # Puts profit when price goes down
            # Normalize to reasonable range
            normalized_pnl = price_change * 10  # ~10% move -> 1.0
        else:
            normalized_pnl = 0.0

        # Bars held (normalized by ~6 hours = 360 bars)
        bars_held = (current_idx - self.position_entry_idx) / 360 if self.position_entry_idx else 0

        # Time to close (normalized by 6.5 hour trading day)
        market_close = datetime.combine(current_time.date(), time(16, 0))
        hours_to_close = max(0, (market_close - current_time).total_seconds() / 3600)
        normalized_time = hours_to_close / 6.5

        return torch.tensor(
            [position_type, normalized_pnl, bars_held, normalized_time],
            dtype=torch.float32,
        )

    def get_unrealized_pnl_pct(self, current_price: float) -> float:
        """
        Calculate unrealized P&L as a percentage for the current position.

        Uses actual option prices from the engine's position tracking, which
        properly accounts for time decay (theta), moneyness, and volatility.

        Returns:
            Unrealized P&L as a percentage (e.g., 50.0 = +50%)
        """
        if not self.engine.positions:
            return 0.0

        # Get the actual position from the engine
        # Engine tracks positions with proper option pricing via OptionsSimulator
        total_entry_value = 0.0
        total_current_value = 0.0

        for pos in self.engine.positions.values():
            entry_value = pos.entry_price * pos.quantity * self.engine.config.options_multiplier
            current_value = pos.current_price * pos.quantity * self.engine.config.options_multiplier
            total_entry_value += entry_value
            total_current_value += current_value

        if total_entry_value == 0:
            return 0.0

        # Calculate P&L percentage based on actual option prices
        unrealized_pnl_pct = ((total_current_value - total_entry_value) / total_entry_value) * 100

        return unrealized_pnl_pct

    def should_take_action(self, current_time: datetime) -> bool:
        """Check if enough time has passed since last action."""
        if self.last_action_time is None:
            return True
        elapsed = (current_time - self.last_action_time).total_seconds() / 60
        return elapsed >= self.action_cooldown

    def open_position(
        self,
        position_type: PositionType,
        current_price: float,
        current_time: datetime,
        current_idx: int,
        is_otm: bool = False,
        otm_offset: float = 2.0,
    ) -> bool:
        """
        Open a new position.

        Args:
            position_type: CALL or PUT
            current_price: Current underlying price
            current_time: Timestamp of entry
            current_idx: Index in dataset
            is_otm: Whether to buy OTM instead of ATM
            otm_offset: Dollar offset from ATM for OTM strikes (default $2)
        """
        # Determine strike based on ATM vs OTM
        base_strike = round(current_price)
        if is_otm:
            # OTM: higher strike for calls, lower for puts
            if position_type == PositionType.CALL:
                strike = base_strike + otm_offset
            else:
                strike = base_strike - otm_offset
        else:
            strike = base_strike

        equity = self.engine.get_total_equity()
        max_position_value = equity * self.engine.config.max_position_size
        option_price_estimate = current_price * 0.01
        contract_value = option_price_estimate * self.engine.config.options_multiplier
        quantity = max(1, int(max_position_value / contract_value))

        order = self.engine.submit_order(
            symbol="SPY",
            side=OrderSide.BUY,
            quantity=quantity,
            position_type=position_type,
            strike=strike,
        )

        if order.filled:
            self.current_position_type = position_type
            self.position_entry_idx = current_idx
            self.position_entry_price = current_price
            self.position_entry_time = current_time
            self.last_action_time = current_time

            if self.verbose:
                moneyness = "OTM" if is_otm else "ATM"
                print(f"  [{current_time.strftime('%H:%M')}] BUY {quantity} {position_type.value.upper()} {moneyness} @ strike {strike}, spot ${current_price:.2f}")
            return True

        return False

    def close_position(self, current_time: datetime, reason: str = "") -> None:
        """Close current position."""
        if self.engine.positions:
            self.engine.close_all_positions()
            if self.verbose:
                print(f"  [{current_time.strftime('%H:%M')}] CLOSE {self.current_position_type.value.upper()} {reason}")

        self.current_position_type = None
        self.position_entry_idx = None
        self.position_entry_price = None
        self.position_entry_time = None
        self.last_action_time = current_time

    def run_backtest(
        self,
        df: pd.DataFrame,
        deterministic: bool = True,
    ) -> Dict[str, float]:
        """Run full backtest with dual-policy architecture."""
        self.engine.reset()
        self.last_action_time = None
        self.current_position_type = None
        self.position_entry_idx = None
        self.position_entry_price = None
        self.position_entry_time = None

        self._backtest_df = df.copy()

        # Initialize patcher (only needed if not using normalized data)
        if not self.use_normalized_data:
            base_cols = {"open", "high", "low", "close", "volume"}
            extra_columns = [col for col in df.columns if col not in base_cols and col != "date"]
            self.patcher = MarketPatch(patch_length=self._patch_length, extra_columns=extra_columns)

        df = df.copy()
        df["date"] = df.index.date

        days = df["date"].unique()
        print(f"\nRunning dual-policy backtest on {len(days)} trading days...")

        entry_action_counts = {a: 0 for a in EntryAction}
        exit_action_counts = {a: 0 for a in ExitAction}

        for day_idx, day in enumerate(days):
            day_data = df[df["date"] == day]

            market_open = time(9, 30)
            market_close_time = time(16, 0)
            day_data = day_data[
                (day_data.index.time >= market_open) &
                (day_data.index.time <= market_close_time)
            ]

            if len(day_data) == 0:
                continue

            if self.verbose:
                print(f"\nDay {day_idx + 1}: {day} ({len(day_data)} bars)")

            if self.engine.positions:
                print(f"  WARNING: positions carried over - closing")
                self.engine.close_all_positions()

            # Reset state for new day
            self.last_action_time = None
            self.current_position_type = None
            self.position_entry_idx = None
            self.position_entry_price = None
            self.position_entry_time = None

            for i, (timestamp, row) in enumerate(day_data.iterrows()):
                overall_idx = df.index.get_loc(timestamp)

                # Get current price from raw prices if using normalized data
                if self.use_normalized_data and self._raw_close_prices is not None:
                    current_price = self._raw_close_prices[overall_idx]
                else:
                    current_price = row["close"]

                self.engine.update_market_state(timestamp, current_price)

                if overall_idx < self.patch_length:
                    continue

                # Don't trade in last 30 minutes
                market_close = datetime.combine(timestamp.date(), time(16, 0))
                minutes_to_close = (market_close - timestamp).total_seconds() / 60
                if minutes_to_close < 30:
                    continue

                if not self.should_take_action(timestamp):
                    continue

                # Get embedding
                embedding = self.get_embedding(df, overall_idx)
                if embedding is None:
                    continue

                # If we have a position, check exit conditions
                if self.current_position_type is not None:
                    # Calculate unrealized P&L percentage
                    unrealized_pnl_pct = self.get_unrealized_pnl_pct(current_price)

                    if self.use_continuous_signal:
                        # Continuous signal mode: use entry policy to check exit
                        entry_action, entry_conf = self.get_entry_action(embedding, deterministic)
                        entry_action_counts[entry_action] += 1

                        # Calculate time to close
                        hours_to_close = minutes_to_close / 60.0

                        # Check exit via continuous signal policy
                        exit_action, reason, flip_action = self.exit_policy.check_exit(
                            position_type=self.current_position_type,
                            latest_entry_action=entry_action,
                            unrealized_pnl_pct=unrealized_pnl_pct,
                            time_to_close_hours=hours_to_close,
                        )
                        exit_action_counts[exit_action] += 1

                        if exit_action == ExitAction.CLOSE:
                            self.close_position(timestamp, f"({reason})")

                            # Handle position flip if enabled
                            if flip_action is not None and flip_action != EntryAction.HOLD:
                                is_otm = False
                                if flip_action in (EntryAction.BUY_CALL_ATM, EntryAction.BUY_CALL_OTM):
                                    pos_type = PositionType.CALL
                                    is_otm = (flip_action == EntryAction.BUY_CALL_OTM)
                                elif flip_action in (EntryAction.BUY_PUT_ATM, EntryAction.BUY_PUT_OTM):
                                    pos_type = PositionType.PUT
                                    is_otm = (flip_action == EntryAction.BUY_PUT_OTM)
                                else:
                                    pos_type = PositionType.CALL if flip_action.value == 1 else PositionType.PUT

                                if self.verbose:
                                    print(f"  [{timestamp.strftime('%H:%M')}] FLIP to {pos_type.value.upper()}")
                                self.open_position(pos_type, current_price, timestamp, overall_idx, is_otm=is_otm)
                                # Reset the continuous signal policy for new position
                                self.exit_policy.reset()
                    else:
                        # Standard exit policy mode (NN or rule-based)
                        context = self.get_position_context(overall_idx, current_price, timestamp)

                        exit_action, exit_conf = self.get_exit_action(
                            embedding, context, timestamp, unrealized_pnl_pct, deterministic
                        )
                        exit_action_counts[exit_action] += 1

                        if exit_action == ExitAction.CLOSE:
                            self.close_position(timestamp, "(exit policy)")

                # If no position, use entry policy
                else:
                    entry_action, entry_conf = self.get_entry_action(embedding, deterministic)
                    entry_action_counts[entry_action] += 1

                    if entry_action != EntryAction.HOLD and entry_conf >= self.entry_confidence:
                        # Handle both 3-class (legacy) and 5-class entry actions
                        # 5-class: HOLD=0, CALL_ATM=1, CALL_OTM=2, PUT_ATM=3, PUT_OTM=4
                        # 3-class legacy: HOLD=0, BUY_CALL=1, BUY_PUT=2
                        is_otm = False
                        if entry_action in (EntryAction.BUY_CALL_ATM, EntryAction.BUY_CALL_OTM):
                            pos_type = PositionType.CALL
                            is_otm = (entry_action == EntryAction.BUY_CALL_OTM)
                        elif entry_action in (EntryAction.BUY_PUT_ATM, EntryAction.BUY_PUT_OTM):
                            pos_type = PositionType.PUT
                            is_otm = (entry_action == EntryAction.BUY_PUT_OTM)
                        else:
                            # Legacy 3-class support: BUY_CALL=1, BUY_PUT=2 (treat as ATM)
                            # Check if entry_action value is 1 (call) or 2 (put)
                            pos_type = PositionType.CALL if entry_action.value == 1 else PositionType.PUT
                        self.open_position(pos_type, current_price, timestamp, overall_idx, is_otm=is_otm)

                        # Reset continuous signal state for new position
                        if self.use_continuous_signal:
                            self.exit_policy.reset()

            # EOD close
            last_timestamp = day_data.index[-1]
            last_price = day_data["close"].iloc[-1]
            self.engine.update_market_state(last_timestamp, last_price)

            if self.engine.positions:
                if self.verbose:
                    print(f"  [EOD {last_timestamp.strftime('%H:%M')}] Closing position at EOD")
                self.engine.close_all_positions()
                self.current_position_type = None

            # Log trades for this day
            day_trades = [t for t in self.engine.trades if t.exit_timestamp.date() == day]
            if day_trades and self.verbose:
                for t in day_trades:
                    pnl_sign = "+" if t.pnl >= 0 else ""
                    print(f"    {t.position_type.value.upper()}: ${t.entry_price:.2f} -> ${t.exit_price:.2f} | P&L: {pnl_sign}${t.pnl:.2f}")

            # Daily summary
            daily_equity = self.engine.get_total_equity()
            daily_return = (daily_equity - self.engine.config.initial_capital) / self.engine.config.initial_capital * 100
            daily_trades = len(day_trades)
            day_open = day_data["close"].iloc[0]
            day_close = day_data["close"].iloc[-1]
            day_pct = (day_close - day_open) / day_open * 100
            print(f"  Day {day_idx + 1:3d} | {day} | SPY: ${day_open:.2f} -> ${day_close:.2f} ({day_pct:+.2f}%) | Equity: ${daily_equity:,.2f} | Return: {daily_return:+.2f}% | Trades: {daily_trades}")

        # Print action distributions
        print("\nEntry Policy Predictions:")
        total_entry = sum(entry_action_counts.values())
        for action, count in entry_action_counts.items():
            pct = count / total_entry * 100 if total_entry > 0 else 0
            print(f"  {action.name}: {count} ({pct:.1f}%)")

        print("\nExit Policy Predictions:")
        total_exit = sum(exit_action_counts.values())
        for action, count in exit_action_counts.items():
            pct = count / total_exit * 100 if total_exit > 0 else 0
            print(f"  {action.name}: {count} ({pct:.1f}%)")

        # Trade stats
        trades = self.engine.trades
        if trades:
            call_trades = len([t for t in trades if t.position_type == PositionType.CALL])
            put_trades = len([t for t in trades if t.position_type == PositionType.PUT])
            print(f"\nActual Trades:")
            print(f"  CALL: {call_trades}")
            print(f"  PUT: {put_trades}")
            print(f"  Total: {len(trades)}")

        metrics = self.engine.get_metrics()
        self.engine.print_summary()

        return metrics


class StrategyRunner:
    """
    Legacy strategy runner using single PolicyNetwork with 4 actions.

    For backward compatibility. Use DualPolicyRunner for new implementations.
    """

    def __init__(
        self,
        lejepa: LeJEPA,
        policy: PolicyNetwork,
        config: BacktestConfig,
        device: torch.device,
        patch_length: int = 32,
        action_cooldown: int = 5,
        verbose: bool = False,
        confidence_threshold: float = 0.0,
        force_trades: bool = False,
        min_signal_threshold: float = 0.20,
    ) -> None:
        """
        Initialize strategy runner.

        Args:
            lejepa: Pre-trained LeJEPA model
            policy: Trained policy network
            config: Backtest configuration
            device: Torch device
            patch_length: Length of input patches
            action_cooldown: Minutes to wait between actions
            verbose: Verbose logging
            confidence_threshold: Min confidence for non-HOLD actions
            force_trades: Use second-best action if HOLD is predicted
            min_signal_threshold: Min probability required for non-HOLD when force_trades enabled
        """
        self.lejepa = lejepa
        self.policy = policy
        self.engine = BacktestEngine(config)
        self.device = device
        self.patch_length = patch_length
        self.action_cooldown = action_cooldown
        self.verbose = verbose
        self.confidence_threshold = confidence_threshold
        self.force_trades = force_trades
        self.min_signal_threshold = min_signal_threshold

        self.patcher = None
        self._patch_length = patch_length

        # State
        self.last_action_time: Optional[datetime] = None
        self.current_position_type: Optional[PositionType] = None

    @torch.no_grad()
    def get_action(
        self,
        df: pd.DataFrame,
        current_idx: int,
        deterministic: bool = True,
    ) -> Tuple[TradingAction, float]:
        """Get action from policy for current market state."""
        if current_idx < self.patch_length:
            return TradingAction.HOLD, 0.0

        start_idx = current_idx - self.patch_length
        try:
            patch = self.patcher.create_patch(df, start_idx)
        except ValueError:
            return TradingAction.HOLD, 0.0

        patch_tensor = patch.unsqueeze(0).to(self.device)

        with torch.amp.autocast(device_type=self.device.type, dtype=torch.bfloat16):
            embedding = self.lejepa.context_encoder(patch_tensor, return_all_tokens=False)

        output = self.policy(embedding.float(), return_value=False)
        probs = output["action_probs"][0]

        if deterministic:
            action_idx = probs.argmax().item()
        else:
            action_idx = torch.multinomial(probs, 1).item()

        confidence = probs[action_idx].item()
        action = TradingAction(action_idx)

        if self.force_trades and action == TradingAction.HOLD:
            call_prob = probs[TradingAction.BUY_CALL.value].item()
            put_prob = probs[TradingAction.BUY_PUT.value].item()
            if call_prob > self.min_signal_threshold or put_prob > self.min_signal_threshold:
                if call_prob > put_prob:
                    action = TradingAction.BUY_CALL
                    confidence = call_prob
                else:
                    action = TradingAction.BUY_PUT
                    confidence = put_prob

        return action, confidence

    def should_take_action(self, current_time: datetime) -> bool:
        """Check if enough time has passed since last action."""
        if self.last_action_time is None:
            return True
        elapsed = (current_time - self.last_action_time).total_seconds() / 60
        return elapsed >= self.action_cooldown

    def execute_action(
        self,
        action: TradingAction,
        current_price: float,
        current_time: datetime,
    ) -> None:
        """Execute trading action."""
        strike = round(current_price)

        equity = self.engine.get_total_equity()
        max_position_value = equity * self.engine.config.max_position_size
        option_price_estimate = current_price * 0.01
        contract_value = option_price_estimate * self.engine.config.options_multiplier
        quantity = max(1, int(max_position_value / contract_value))

        if action == TradingAction.BUY_CALL:
            if self.current_position_type == PositionType.PUT:
                if self.verbose:
                    print(f"  [{current_time.strftime('%H:%M')}] CLOSE PUT (switching to CALL)")
                self.engine.close_all_positions()
                self.current_position_type = None

            if not self.engine.positions:
                order = self.engine.submit_order(
                    symbol="SPY",
                    side=OrderSide.BUY,
                    quantity=quantity,
                    position_type=PositionType.CALL,
                    strike=strike,
                )
                if order.filled:
                    self.current_position_type = PositionType.CALL
                    if self.verbose:
                        print(f"  [{current_time.strftime('%H:%M')}] BUY {quantity} CALL @ strike {strike}, spot ${current_price:.2f}")

        elif action == TradingAction.BUY_PUT:
            if self.current_position_type == PositionType.CALL:
                if self.verbose:
                    print(f"  [{current_time.strftime('%H:%M')}] CLOSE CALL (switching to PUT)")
                self.engine.close_all_positions()
                self.current_position_type = None

            if not self.engine.positions:
                order = self.engine.submit_order(
                    symbol="SPY",
                    side=OrderSide.BUY,
                    quantity=quantity,
                    position_type=PositionType.PUT,
                    strike=strike,
                )
                if order.filled:
                    self.current_position_type = PositionType.PUT
                    if self.verbose:
                        print(f"  [{current_time.strftime('%H:%M')}] BUY {quantity} PUT @ strike {strike}, spot ${current_price:.2f}")

        elif action == TradingAction.CLOSE_POSITION:
            if self.engine.positions:
                self.engine.close_all_positions()
                self.current_position_type = None
                if self.verbose:
                    print(f"  [{current_time.strftime('%H:%M')}] CLOSE all positions")

        if action != TradingAction.HOLD:
            self.last_action_time = current_time

    def run_backtest(
        self,
        df: pd.DataFrame,
        deterministic: bool = True,
    ) -> Dict[str, float]:
        """Run full backtest on historical data."""
        self.engine.reset()
        self.last_action_time = None
        self.current_position_type = None

        self._backtest_df = df.copy()

        base_cols = {"open", "high", "low", "close", "volume"}
        extra_columns = [col for col in df.columns if col not in base_cols and col != "date"]
        self.patcher = MarketPatch(patch_length=self._patch_length, extra_columns=extra_columns)

        df = df.copy()
        df["date"] = df.index.date

        days = df["date"].unique()
        print(f"\nRunning backtest on {len(days)} trading days...")

        action_counts = {a: 0 for a in TradingAction}

        for day_idx, day in enumerate(days):
            day_data = df[df["date"] == day]

            market_open = time(9, 30)
            market_close_time = time(16, 0)
            day_data = day_data[
                (day_data.index.time >= market_open) &
                (day_data.index.time <= market_close_time)
            ]

            if len(day_data) == 0:
                continue

            if self.verbose:
                print(f"\nDay {day_idx + 1}: {day} ({len(day_data)} bars)")

            if self.engine.positions:
                print(f"  WARNING: positions carried over - closing")
                self.engine.close_all_positions()

            self.last_action_time = None
            self.current_position_type = None

            for i, (timestamp, row) in enumerate(day_data.iterrows()):
                current_price = row["close"]
                self.engine.update_market_state(timestamp, current_price)

                overall_idx = df.index.get_loc(timestamp)
                if overall_idx < self.patch_length:
                    continue

                market_close = datetime.combine(timestamp.date(), time(16, 0))
                minutes_to_close = (market_close - timestamp).total_seconds() / 60
                if minutes_to_close < 30:
                    continue

                if not self.should_take_action(timestamp):
                    continue

                action, confidence = self.get_action(df, overall_idx, deterministic)
                action_counts[action] += 1

                if action != TradingAction.HOLD and confidence >= self.confidence_threshold:
                    self.execute_action(action, current_price, timestamp)

            last_timestamp = day_data.index[-1]
            last_price = day_data["close"].iloc[-1]
            self.engine.update_market_state(last_timestamp, last_price)

            if self.verbose and self.engine.positions:
                print(f"  [EOD] Closing positions")
            self.engine.close_all_positions()
            self.current_position_type = None

            day_trades = [t for t in self.engine.trades if t.exit_timestamp.date() == day]
            if day_trades and self.verbose:
                for t in day_trades:
                    pnl_sign = "+" if t.pnl >= 0 else ""
                    print(f"    {t.position_type.value.upper()}: ${t.entry_price:.2f} -> ${t.exit_price:.2f} | P&L: {pnl_sign}${t.pnl:.2f}")

            daily_equity = self.engine.get_total_equity()
            daily_return = (daily_equity - self.engine.config.initial_capital) / self.engine.config.initial_capital * 100
            daily_trades = len(day_trades)
            day_open = day_data["close"].iloc[0]
            day_close = day_data["close"].iloc[-1]
            day_pct = (day_close - day_open) / day_open * 100
            print(f"  Day {day_idx + 1:3d} | {day} | SPY: ${day_open:.2f} -> ${day_close:.2f} ({day_pct:+.2f}%) | Equity: ${daily_equity:,.2f} | Return: {daily_return:+.2f}% | Trades: {daily_trades}")

        print("\nPolicy Predictions:")
        total_actions = sum(action_counts.values())
        for action, count in action_counts.items():
            pct = count / total_actions * 100 if total_actions > 0 else 0
            print(f"  {action.name}: {count} ({pct:.1f}%)")

        trades = self.engine.trades
        if trades:
            call_trades = len([t for t in trades if t.position_type == PositionType.CALL])
            put_trades = len([t for t in trades if t.position_type == PositionType.PUT])
            print(f"\nActual Trades:")
            print(f"  CALL: {call_trades}")
            print(f"  PUT: {put_trades}")
            print(f"  Total: {len(trades)}")

        metrics = self.engine.get_metrics()
        self.engine.print_summary()

        return metrics


def load_lejepa(lejepa_path: str, device: torch.device) -> LeJEPA:
    """Load and freeze pre-trained LeJEPA model."""
    print(f"Loading LeJEPA from {lejepa_path}...")
    lejepa, _ = LeJEPA.load_checkpoint(lejepa_path, device=str(device))
    lejepa = lejepa.to(device)
    lejepa.eval()

    for param in lejepa.parameters():
        param.requires_grad = False

    return lejepa


def load_dual_policies(
    entry_path: str,
    exit_path: str,
    device: torch.device,
) -> Tuple[EntryPolicy, ExitPolicy]:
    """
    Load dual-policy models (entry + exit).

    Args:
        entry_path: Path to entry policy checkpoint
        exit_path: Path to exit policy checkpoint
        device: Torch device

    Returns:
        Tuple of (entry_policy, exit_policy)
    """
    # Load entry policy
    print(f"Loading entry policy from {entry_path}...")
    entry_ckpt = torch.load(entry_path, map_location=device, weights_only=False)
    entry_config = entry_ckpt.get("config", {})

    entry_policy = EntryPolicy(
        embedding_dim=entry_config.get("embedding_dim", 512),
        hidden_dim=entry_config.get("hidden_dim", 256),
        num_layers=entry_config.get("num_layers", 2),
    )
    entry_policy.load_state_dict(entry_ckpt["policy_state_dict"])
    entry_policy = entry_policy.to(device)
    entry_policy.eval()

    # Load exit policy
    print(f"Loading exit policy from {exit_path}...")
    exit_ckpt = torch.load(exit_path, map_location=device, weights_only=False)
    exit_config = exit_ckpt.get("config", {})

    exit_policy = ExitPolicy(
        embedding_dim=exit_config.get("embedding_dim", 512),
        hidden_dim=exit_config.get("hidden_dim", 256),
        num_layers=exit_config.get("num_layers", 2),
        context_dim=exit_config.get("context_dim", 4),
    )
    exit_policy.load_state_dict(exit_ckpt["policy_state_dict"])
    exit_policy = exit_policy.to(device)
    exit_policy.eval()

    return entry_policy, exit_policy


def load_legacy_policy(
    policy_path: str,
    lejepa: LeJEPA,
    device: torch.device,
) -> PolicyNetwork:
    """
    Load legacy single-policy model.

    Args:
        policy_path: Path to policy checkpoint
        lejepa: LeJEPA model (for embedding dim)
        device: Torch device

    Returns:
        PolicyNetwork
    """
    print(f"Loading legacy policy from {policy_path}...")
    policy_ckpt = torch.load(policy_path, map_location=device, weights_only=False)
    policy_config = policy_ckpt.get("config", {})

    # Infer num_layers from state dict if not in config
    num_layers = policy_config.get("num_layers", 2)
    if "shared_layers.8.weight" in policy_ckpt["policy_state_dict"]:
        num_layers = 3
    elif "shared_layers.4.weight" in policy_ckpt["policy_state_dict"]:
        num_layers = 2

    policy = PolicyNetwork(
        embedding_dim=policy_config.get("embedding_dim", lejepa.embedding_dim),
        hidden_dim=policy_config.get("hidden_dim", 256),
        num_layers=num_layers,
        num_actions=policy_config.get("num_actions", 4),
        use_value_head=False,
    )
    policy.load_state_dict(policy_ckpt["policy_state_dict"])
    policy = policy.to(device)
    policy.eval()

    return policy


# Legacy compatibility function
def load_models(
    lejepa_path: str,
    policy_path: str,
    device: torch.device,
) -> Tuple[LeJEPA, PolicyNetwork]:
    """
    Load pre-trained models (legacy interface).

    Args:
        lejepa_path: Path to LeJEPA checkpoint
        policy_path: Path to policy checkpoint
        device: Torch device

    Returns:
        Tuple of (lejepa, policy)
    """
    lejepa = load_lejepa(lejepa_path, device)
    policy = load_legacy_policy(policy_path, lejepa, device)
    return lejepa, policy


def main() -> None:
    """Main function."""
    args = parse_args()

    print("=" * 70)
    print("Strategy Backtest Runner")
    print("=" * 70)

    # Track strategy name for saving results
    strategy_name = None

    # Load strategy config if provided
    if args.strategy:
        from src.utils.strategy_manager import get_strategy, strategy_to_args, add_backtest_result, get_checkpoint_path

        strategy = get_strategy(args.strategy)
        if strategy is None:
            print(f"ERROR: Strategy '{args.strategy}' not found in strategies/")
            return

        strategy_name = args.strategy
        print(f"\nLoading strategy: {strategy_name}")
        print(f"Description: {strategy.get('description', 'N/A')}")

        # Convert strategy to args and re-parse
        strategy_args = strategy_to_args(strategy)

        # Merge: command-line args override strategy args
        # Re-parse with strategy args as base
        import sys
        original_argv = sys.argv
        sys.argv = [sys.argv[0]] + strategy_args

        # Add command-line args that were explicitly provided (except --strategy itself)
        for i, arg in enumerate(original_argv[1:]):
            if arg == "--strategy" or (i > 0 and original_argv[i] == "--strategy"):
                continue
            # Check if this arg was explicitly set (not default)
            if arg.startswith("--"):
                # Skip --strategy
                if arg == "--strategy":
                    continue
                # Add the arg and its value if it has one
                if i + 2 < len(original_argv) and not original_argv[i + 2].startswith("--"):
                    sys.argv.extend([arg, original_argv[i + 2]])
                elif arg.startswith("--no_") or arg in ["--verbose", "--stochastic", "--force_trades", "--rule_based_exit", "--continuous_signal", "--flip_on_counter", "--no_flip_on_counter", "--no_report"]:
                    sys.argv.append(arg)

        sys.argv = original_argv  # Restore
        print(f"Strategy args: {' '.join(strategy_args)}")

        # Apply strategy config to args
        checkpoints = strategy.get("checkpoints", {})
        if checkpoints.get("lejepa") and not args.lejepa:
            args.lejepa = get_checkpoint_path(checkpoints["lejepa"])
        if checkpoints.get("entry_policy") and not args.entry_policy:
            args.entry_policy = get_checkpoint_path(checkpoints["entry_policy"])
        if checkpoints.get("exit_policy") and not args.exit_policy:
            args.exit_policy = get_checkpoint_path(checkpoints["exit_policy"])

        exit_mode = strategy.get("exit_mode", "continuous_signal")
        exit_config = strategy.get("exit_config", {})
        backtest_config = strategy.get("backtest", {})

        # Apply exit mode
        if exit_mode == "continuous_signal" and not args.continuous_signal and not args.rule_based_exit:
            args.continuous_signal = True
        elif exit_mode == "rule_based" and not args.rule_based_exit and not args.continuous_signal:
            args.rule_based_exit = True

        # Apply exit config
        if exit_config.get("stop_loss_pct") and args.stop_loss_pct is None:
            args.stop_loss_pct = exit_config["stop_loss_pct"]
        if exit_config.get("take_profit_pct") and not args.take_profit_pct:
            args.take_profit_pct = exit_config["take_profit_pct"]
        if exit_config.get("eod_exit_minutes"):
            args.eod_exit_minutes = exit_config["eod_exit_minutes"]
        if exit_config.get("counter_signal_confirmation"):
            args.counter_signal_confirmation = exit_config["counter_signal_confirmation"]
        if exit_config.get("plateau_window"):
            args.plateau_window = exit_config["plateau_window"]
        if exit_config.get("plateau_ratio"):
            args.plateau_ratio = exit_config["plateau_ratio"]
        if exit_config.get("flip_on_counter") is not None:
            args.flip_on_counter = exit_config["flip_on_counter"]
        if exit_config.get("risk_reward_ratio"):
            args.risk_reward_ratio = exit_config["risk_reward_ratio"]
        if exit_config.get("time_stop_hours"):
            args.time_stop_hours = exit_config["time_stop_hours"]

        # Apply backtest config
        if backtest_config.get("initial_capital"):
            args.initial_capital = backtest_config["initial_capital"]
        if backtest_config.get("max_position_pct"):
            args.max_position_pct = backtest_config["max_position_pct"]
        if backtest_config.get("action_cooldown"):
            args.action_cooldown = backtest_config["action_cooldown"]
        if backtest_config.get("entry_confidence"):
            args.entry_confidence = backtest_config["entry_confidence"]

    # Validate arguments
    # Determine mode: dual-policy (with NN, rule-based, or continuous signal exit) or legacy
    rule_based_exit_mode = args.entry_policy is not None and args.rule_based_exit
    continuous_signal_mode = args.entry_policy is not None and args.continuous_signal
    dual_mode = args.entry_policy is not None and args.exit_policy is not None
    legacy_mode = args.policy is not None

    # Handle --no_flip_on_counter flag
    flip_on_counter = args.flip_on_counter
    if args.no_flip_on_counter:
        flip_on_counter = False

    if not dual_mode and not legacy_mode and not rule_based_exit_mode and not continuous_signal_mode:
        print("ERROR: Must provide either --entry_policy + --exit_policy (dual-policy mode)")
        print("       or --entry_policy + --rule_based_exit (rule-based exit mode)")
        print("       or --entry_policy + --continuous_signal (continuous signal exit mode)")
        print("       or --policy (legacy single-policy mode)")
        print("       or --strategy <name> (load from strategy file)")
        return

    if (dual_mode or rule_based_exit_mode or continuous_signal_mode) and legacy_mode:
        print("WARNING: Both dual-policy and legacy arguments provided. Using dual-policy mode.")
        legacy_mode = False

    if rule_based_exit_mode or continuous_signal_mode:
        dual_mode = True  # Treat both as variants of dual-mode

    # Device
    device = get_device(args.device)
    print(f"Device: {device}")

    # Set seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Create backtest config
    config = BacktestConfig(
        initial_capital=args.initial_capital,
        max_position_size=args.max_position_pct,
    )

    # Load models and create runner based on mode
    lejepa = load_lejepa(args.lejepa, device)

    if dual_mode:
        if continuous_signal_mode:
            # Continuous signal exit mode: load only entry policy, use it for exits too
            print("\nUsing CONTINUOUS SIGNAL mode (EntryPolicy for both entry + exit)")
            print(f"  Counter-signal confirmation: {args.counter_signal_confirmation} consecutive minutes")
            print(f"  Plateau window: {args.plateau_window} minutes")
            print(f"  Plateau ratio threshold: {args.plateau_ratio}")
            print(f"  Stop Loss: {args.stop_loss_pct}%")
            if args.take_profit_pct:
                print(f"  Take Profit: {args.take_profit_pct}%")
            print(f"  EOD Exit: {args.eod_exit_minutes} minutes before close")
            print(f"  Flip on counter-signal: {flip_on_counter}")

            # Load only entry policy (handle both checkpoint formats)
            entry_checkpoint = torch.load(args.entry_policy, map_location=device, weights_only=False)

            # Handle different checkpoint formats
            if "config" in entry_checkpoint:
                # New format from train_entry.py
                cfg = entry_checkpoint["config"]
                num_actions = cfg.get("num_actions", 3)  # Default 3 for legacy, 5 for new
                entry_policy = EntryPolicy(
                    embedding_dim=cfg["embedding_dim"],
                    hidden_dim=cfg["hidden_dim"],
                    num_layers=cfg["num_layers"],
                    num_actions=num_actions,
                ).to(device)
                entry_policy.load_state_dict(entry_checkpoint["policy_state_dict"])
            else:
                # Old format
                entry_policy = EntryPolicy(
                    embedding_dim=entry_checkpoint["embedding_dim"],
                    hidden_dim=entry_checkpoint["hidden_dim"],
                    num_layers=entry_checkpoint["num_layers"],
                ).to(device)
                entry_policy.load_state_dict(entry_checkpoint["model_state_dict"])
            entry_policy.eval()

            # Create continuous signal exit policy
            exit_policy = ContinuousSignalExitPolicy(
                counter_signal_confirmation=args.counter_signal_confirmation,
                plateau_window=args.plateau_window,
                plateau_ratio=args.plateau_ratio,
                stop_loss_pct=args.stop_loss_pct if args.stop_loss_pct else 25.0,
                take_profit_pct=args.take_profit_pct,
                eod_exit_minutes=args.eod_exit_minutes,
                flip_on_counter=flip_on_counter,
            )

            runner = DualPolicyRunner(
                lejepa=lejepa,
                entry_policy=entry_policy,
                exit_policy=exit_policy,
                config=config,
                device=device,
                verbose=args.verbose,
                action_cooldown=args.action_cooldown,
                entry_confidence=args.entry_confidence,
                use_continuous_signal=True,
            )
        elif rule_based_exit_mode:
            # Rule-based exit mode: load only entry policy, use rule-based exit
            print("\nUsing DUAL-POLICY mode (EntryPolicy + Rule-Based Exit)")
            print(f"  Take Profit: {args.take_profit_pct}%")
            print(f"  Risk:Reward Ratio: 1:{args.risk_reward_ratio}")
            stop_loss = args.stop_loss_pct if args.stop_loss_pct else args.take_profit_pct / args.risk_reward_ratio
            print(f"  Stop Loss: {stop_loss}%")
            if args.time_stop_hours:
                print(f"  Time Stop: {args.time_stop_hours} hours")
            print(f"  EOD Exit: {args.eod_exit_minutes} minutes before close")

            # Load only entry policy (handle both checkpoint formats)
            entry_checkpoint = torch.load(args.entry_policy, map_location=device, weights_only=False)

            # Handle different checkpoint formats
            num_actions = 5  # Default for old format
            if "config" in entry_checkpoint:
                # New format from train_entry.py
                cfg = entry_checkpoint["config"]
                num_actions = cfg.get("num_actions", 3)  # Default 3 for legacy, 5 for new
                entry_policy = EntryPolicy(
                    embedding_dim=cfg["embedding_dim"],
                    hidden_dim=cfg["hidden_dim"],
                    num_layers=cfg["num_layers"],
                    num_actions=num_actions,
                ).to(device)
                entry_policy.load_state_dict(entry_checkpoint["policy_state_dict"])
            else:
                # Old format
                entry_policy = EntryPolicy(
                    embedding_dim=entry_checkpoint["embedding_dim"],
                    hidden_dim=entry_checkpoint["hidden_dim"],
                    num_layers=entry_checkpoint["num_layers"],
                ).to(device)
                entry_policy.load_state_dict(entry_checkpoint["model_state_dict"])
            entry_policy.eval()

            # Create rule-based exit policy
            exit_policy = RuleBasedExitPolicy(
                take_profit_pct=args.take_profit_pct,
                risk_reward_ratio=args.risk_reward_ratio,
                stop_loss_pct=args.stop_loss_pct,
                time_stop_hours=args.time_stop_hours,
                eod_exit_minutes=args.eod_exit_minutes,
            )

            runner = DualPolicyRunner(
                lejepa=lejepa,
                entry_policy=entry_policy,
                exit_policy=exit_policy,
                config=config,
                device=device,
                verbose=args.verbose,
                action_cooldown=args.action_cooldown,
                entry_confidence=args.entry_confidence,
                use_rule_based_exit=True,
            )

            # Detect 3-class entry policy and set flag
            if num_actions == 3:
                print("  Using 3-class entry policy (HOLD, BUY_CALL, BUY_PUT)")
                runner.use_3class_entry = True
        else:
            # Standard dual-policy mode: load both entry and exit policies
            print("\nUsing DUAL-POLICY mode (EntryPolicy + ExitPolicy)")
            entry_policy, exit_policy = load_dual_policies(
                args.entry_policy, args.exit_policy, device
            )

            runner = DualPolicyRunner(
                lejepa=lejepa,
                entry_policy=entry_policy,
                exit_policy=exit_policy,
                config=config,
                device=device,
                verbose=args.verbose,
                action_cooldown=args.action_cooldown,
                entry_confidence=args.entry_confidence,
            )
    else:
        print("\nUsing LEGACY mode (single PolicyNetwork)")
        policy = load_legacy_policy(args.policy, lejepa, device)

        runner = StrategyRunner(
            lejepa=lejepa,
            policy=policy,
            config=config,
            device=device,
            verbose=args.verbose,
            confidence_threshold=args.confidence_threshold,
            force_trades=args.force_trades,
            action_cooldown=args.action_cooldown,
            min_signal_threshold=args.min_signal_threshold,
        )

    # Load data
    print("\n" + "=" * 70)
    print("Loading Data")
    print("=" * 70)

    if args.normalized:
        # Use pre-normalized cached data (23 features, matches LeJEPA training)
        print(f"Loading NORMALIZED data for {args.underlying}")
        print(f"Stocks dir: {args.data_dir}")
        print(f"Options dir: {args.options_dir}")
        df = load_normalized_data(
            stocks_dir=args.data_dir,
            options_dir=args.options_dir,
            underlying=args.underlying,
            start_date=args.start_date,
            end_date=args.end_date,
        )
        print(f"Loaded {len(df):,} bars with {len(df.columns)} features (normalized)")

        # Load raw close prices for trading decisions
        from src.data.loader import RAW_CACHE_DIR, parse_date
        from pathlib import Path
        raw_cache_dir = Path(RAW_CACHE_DIR)
        raw_files = sorted(raw_cache_dir.glob(f"{args.underlying}_raw_*.parquet"))

        start = parse_date(args.start_date) if args.start_date else None
        end = parse_date(args.end_date) if args.end_date else None

        raw_closes_list = []
        for rf in raw_files:
            date_str = rf.stem.split("_")[-1]
            try:
                from datetime import datetime as dt
                file_date = dt.strptime(date_str, "%Y-%m-%d").date()
                if start and file_date < start:
                    continue
                if end and file_date > end:
                    continue
                raw_df = pd.read_parquet(rf)
                raw_closes_list.append(raw_df["close"].values)
            except (ValueError, KeyError):
                continue
        raw_close_prices = np.concatenate(raw_closes_list) if raw_closes_list else np.zeros(len(df))
        print(f"Loaded {len(raw_close_prices):,} raw close prices for trading")

        # Set normalized data on runner for direct tensor access
        if dual_mode and hasattr(runner, 'set_normalized_data'):
            runner.set_normalized_data(df, raw_close_prices)
    else:
        # Legacy path: load normalized data without dual-mode
        print(f"Loading stocks data from: {args.data_dir}")
        print(f"Loading options data from: {args.options_dir}")

        df = load_normalized_data(
            stocks_dir=args.data_dir,
            options_dir=args.options_dir,
            start_date=args.start_date,
            end_date=args.end_date,
        )
        print(f"Loaded {len(df):,} bars with {len(df.columns)} features")

    if len(df) == 0:
        print("ERROR: No data in specified date range!")
        return

    # Print backtest period
    print(f"\nBacktest period: {df.index.min().date()} to {df.index.max().date()}")
    print(f"Total bars: {len(df):,}")
    num_days = len(df.index.normalize().unique())
    print(f"Trading days: {num_days}")

    # Handle NaN values
    nan_count = df.isnull().sum().sum()
    if nan_count > 0:
        print(f"Filling {nan_count} NaN values")
        df = df.ffill().bfill()

    # Run backtest
    metrics = runner.run_backtest(df, deterministic=not args.stochastic)

    print("\n" + "=" * 70)
    print("Backtest Complete!")
    print("=" * 70)

    # Generate HTML report
    if not args.no_report:
        report_path = args.report
        if report_path is None:
            # Generate default report path based on date range
            start_str = df.index.min().strftime("%Y%m%d")
            end_str = df.index.max().strftime("%Y%m%d")
            mode_suffix = "dual" if dual_mode else "legacy"
            report_path = f"reports/backtest_{mode_suffix}_{start_str}_{end_str}.html"

        print(f"\nGenerating HTML report...")
        generate_html_report(
            df=runner._backtest_df,
            trades=runner.engine.trades,
            metrics=metrics,
            config=config,
            equity_curve=runner.engine.equity_curve,
            output_path=report_path,
            title="Spymaster Backtest Report",
        )

        # Save backtest result to strategy if using --strategy
        if strategy_name:
            from src.utils.strategy_manager import add_backtest_result

            start_date = str(df.index.min().date())
            end_date = str(df.index.max().date())

            add_backtest_result(
                strategy_name=strategy_name,
                start_date=start_date,
                end_date=end_date,
                report_path=report_path,
                metrics=metrics,
            )
            print(f"\nBacktest result saved to strategy '{strategy_name}'")


if __name__ == "__main__":
    main()
