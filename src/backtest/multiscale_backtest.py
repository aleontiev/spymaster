"""
Multi-Scale Policy Backtest with Triple Barrier Method.

Features:
1. Uses MultiScalePolicy for entry signals (LONG/SHORT/NEUTRAL)
2. Triple Barrier exits with:
   - Take profit barrier (upper)
   - Stop loss barrier (lower) with trailing mechanism
   - Time barrier (max holding period)
3. Continuous signal checking:
   - Confirmation (same signal): Reset barriers for extended moves
   - Reversal (opposite signal): Exit immediately
4. Trailing stop-loss:
   - At +10% profit: Move stop to breakeven (0%)
   - Continue trailing by 10% (e.g., +20% -> stop at +10%)
5. Real options pricing from 1-minute options data

Usage:
    uv run python -m src.backtest.multiscale_backtest \
        --policy-checkpoint data/checkpoints/multiscale-mae-v2/policy_best.pt \
        --short-jepa-checkpoint checkpoints/lejepa-15-5/lejepa_best.pt \
        --med-jepa-checkpoint checkpoints/lejepa-45-15/lejepa_best.pt \
        --long-jepa-checkpoint checkpoints/lejepa-90-30/lejepa_best.pt \
        --start-date 2020-01-01 --end-date 2025-08-31
"""
import argparse
import re
from dataclasses import dataclass, field
from datetime import datetime, time, timedelta
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from src.model.lejepa import LeJEPA
from src.model.masked_gated_policy import MultiScalePolicy


class Signal(Enum):
    NEUTRAL = 0
    LONG = 1
    SHORT = 2


class PositionType(Enum):
    CALL = "call"
    PUT = "put"


@dataclass
class TripleBarrier:
    """Triple barrier configuration with trailing stop."""
    take_profit_pct: float = 50.0       # Exit at +50%
    initial_stop_loss_pct: float = 25.0  # Initial stop at -25%
    time_limit_minutes: int = 120        # Max 2 hours holding
    trailing_activation_pct: float = 10.0  # Start trailing at +10%
    trailing_offset_pct: float = 10.0     # Trail by 10%


@dataclass
class Position:
    """Active position tracking."""
    position_type: PositionType
    option_ticker: str
    strike: float
    entry_time: datetime
    entry_price: float  # Option price at entry
    entry_underlying_price: float
    quantity: int

    # Triple barrier state
    barrier_start_time: datetime = None  # Reset on confirmation
    current_stop_loss_pct: float = -25.0  # Current stop level (negative = loss)
    high_water_mark_pct: float = 0.0  # Highest profit seen

    def __post_init__(self):
        if self.barrier_start_time is None:
            self.barrier_start_time = self.entry_time


@dataclass
class Trade:
    """Completed trade record."""
    position_type: PositionType
    option_ticker: str
    strike: float
    entry_time: datetime
    exit_time: datetime
    entry_price: float
    exit_price: float
    entry_underlying: float
    exit_underlying: float
    quantity: int
    pnl: float
    pnl_pct: float
    exit_reason: str
    holding_minutes: float


@dataclass
class BacktestConfig:
    """Backtest configuration."""
    initial_capital: float = 100_000
    max_position_pct: float = 0.05  # 5% of capital per trade
    options_multiplier: int = 100
    min_confidence: float = 0.0  # Minimum model confidence to trade

    # Triple barrier settings
    take_profit_pct: float = 50.0
    initial_stop_loss_pct: float = 25.0
    time_limit_minutes: int = 120
    trailing_activation_pct: float = 10.0
    trailing_offset_pct: float = 10.0

    # Calibration options
    temperature: float = 1.0  # Temperature scaling (>1 = softer probs)
    directional_threshold: float = 0.0  # Min prob for LONG/SHORT (asymmetric threshold)
    calibration_mode: str = "none"  # "none", "confidence", "temperature", "threshold"

    # Signal alignment options
    require_all_align: bool = True  # Only trade when all 3 model heads agree
    individual_threshold: float = 0.4  # Min prob for individual head signal (lower = more signals)


class OptionsDataProvider:
    """Provides real options prices from 1-minute data."""

    def __init__(self, options_dir: str = "data/options-1m/SPY"):
        self.options_dir = Path(options_dir)
        self._cache: Dict[str, pd.DataFrame] = {}  # date -> DataFrame

    def _load_day(self, date: datetime) -> Optional[pd.DataFrame]:
        """Load options data for a specific day."""
        date_key = date.strftime("%Y-%m-%d")
        if date_key in self._cache:
            return self._cache[date_key]

        # Construct path: data/options-1m/SPY/YYYY-MM/DD.parquet
        month_dir = self.options_dir / date.strftime("%Y-%m")
        day_file = month_dir / f"{date.strftime('%d')}.parquet"

        if not day_file.exists():
            self._cache[date_key] = None
            return None

        try:
            df = pd.read_parquet(day_file)
            # Parse window_start as datetime if needed
            if not pd.api.types.is_datetime64_any_dtype(df['window_start']):
                df['window_start'] = pd.to_datetime(df['window_start'])
            self._cache[date_key] = df
            return df
        except Exception as e:
            print(f"Error loading options data for {date_key}: {e}")
            self._cache[date_key] = None
            return None

    def _parse_option_ticker(self, ticker: str) -> Optional[Tuple[datetime, str, float]]:
        """
        Parse option ticker to extract expiry, type, strike.

        Format: O:SPY240102C00402000
        - SPY: underlying
        - 240102: expiry YYMMDD
        - C: call (P for put)
        - 00402000: strike * 1000 (402.00)
        """
        match = re.match(r'O:SPY(\d{6})([CP])(\d{8})', ticker)
        if not match:
            return None

        expiry_str, opt_type, strike_str = match.groups()
        expiry = datetime.strptime(f"20{expiry_str}", "%Y%m%d")
        strike = int(strike_str) / 1000.0

        return expiry.date(), opt_type, strike

    def find_0dte_option(
        self,
        timestamp: datetime,
        position_type: PositionType,
        underlying_price: float,
        strike_offset: float = 0.0,  # ATM by default
    ) -> Optional[Tuple[str, float]]:
        """
        Find a 0DTE option and return its ticker and current price.

        Args:
            timestamp: Current time
            position_type: CALL or PUT
            underlying_price: Current SPY price
            strike_offset: Offset from ATM (positive = OTM)

        Returns:
            (ticker, price) or None if not found
        """
        df = self._load_day(timestamp)
        if df is None:
            return None

        # Target strike (rounded to nearest dollar)
        opt_type = "C" if position_type == PositionType.CALL else "P"
        if position_type == PositionType.CALL:
            target_strike = round(underlying_price + strike_offset)
        else:
            target_strike = round(underlying_price - strike_offset)

        # Filter for 0DTE options of the right type
        today = timestamp.date()
        candidates = []

        for ticker in df['ticker'].unique():
            parsed = self._parse_option_ticker(ticker)
            if parsed is None:
                continue

            expiry, ticker_type, strike = parsed

            # Must be 0DTE, correct type, and close to target strike
            if expiry == today and ticker_type == opt_type:
                if abs(strike - target_strike) <= 2:  # Within $2 of target
                    candidates.append((ticker, strike))

        if not candidates:
            return None

        # Find closest to target strike
        candidates.sort(key=lambda x: abs(x[1] - target_strike))
        best_ticker, best_strike = candidates[0]

        # Get price at this timestamp (or closest before)
        ticker_data = df[df['ticker'] == best_ticker].copy()
        ticker_data = ticker_data[ticker_data['window_start'] <= timestamp]

        if len(ticker_data) == 0:
            return None

        # Use most recent price
        ticker_data = ticker_data.sort_values('window_start')
        price = ticker_data.iloc[-1]['close']

        return best_ticker, price, best_strike

    def get_option_price(
        self,
        ticker: str,
        timestamp: datetime,
    ) -> Optional[float]:
        """Get the price of a specific option at a timestamp."""
        df = self._load_day(timestamp)
        if df is None:
            return None

        ticker_data = df[df['ticker'] == ticker].copy()
        if len(ticker_data) == 0:
            return None

        # Find price at or before timestamp
        ticker_data = ticker_data[ticker_data['window_start'] <= timestamp]
        if len(ticker_data) == 0:
            # Try to get first available price after
            ticker_data = df[df['ticker'] == ticker].copy()
            ticker_data = ticker_data[ticker_data['window_start'] >= timestamp]
            if len(ticker_data) == 0:
                return None

        ticker_data = ticker_data.sort_values('window_start')
        return ticker_data.iloc[-1]['close'] if len(ticker_data) > 0 else None


class MultiScaleBacktester:
    """
    Backtester for MultiScalePolicy with Triple Barrier exits.
    """

    def __init__(
        self,
        policy: MultiScalePolicy,
        short_jepa: LeJEPA,
        med_jepa: LeJEPA,
        long_jepa: LeJEPA,
        config: BacktestConfig,
        device: torch.device,
        options_provider: OptionsDataProvider,
        verbose: bool = False,
    ):
        self.policy = policy
        self.short_jepa = short_jepa
        self.med_jepa = med_jepa
        self.long_jepa = long_jepa
        self.config = config
        self.device = device
        self.options = options_provider
        self.verbose = verbose

        # Context lengths for each JEPA
        self.short_context = 15
        self.med_context = 45
        self.long_context = 90

        # State
        self.cash = config.initial_capital
        self.position: Optional[Position] = None
        self.trades: List[Trade] = []
        self.equity_curve: List[Tuple[datetime, float]] = []

        # Signal tracking
        self.last_signal: Optional[Signal] = None
        self.signal_history: List[Tuple[datetime, Signal, float]] = []

        # Individual model signal tracking (for HTML report)
        # Maps model name to list of (timestamp, signal_type, price)
        self.individual_signals: Dict[str, List[Tuple[datetime, str, float]]] = {
            "1m": [],  # short head
            "5m": [],  # med head
            "15m": [],  # long head
        }

    def reset(self):
        """Reset backtest state."""
        self.cash = self.config.initial_capital
        self.position = None
        self.trades = []
        self.equity_curve = []
        self.last_signal = None
        self.signal_history = []
        self.individual_signals = {"1m": [], "5m": [], "15m": []}

    @torch.no_grad()
    def get_embeddings(
        self,
        data_tensor: torch.Tensor,
        current_idx: int,
    ) -> Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        """
        Get embeddings from all three JEPAs.

        Args:
            data_tensor: Full normalized data tensor [T, features]
            current_idx: Current position in sequence

        Returns:
            (short_emb, med_emb, long_emb) or None if insufficient context
        """
        if current_idx < self.long_context:
            return None

        # Extract context windows
        short_ctx = data_tensor[current_idx - self.short_context:current_idx].unsqueeze(0)
        med_ctx = data_tensor[current_idx - self.med_context:current_idx].unsqueeze(0)
        long_ctx = data_tensor[current_idx - self.long_context:current_idx].unsqueeze(0)

        # Get embeddings
        short_emb = self.short_jepa.encode(short_ctx)
        med_emb = self.med_jepa.encode(med_ctx)
        long_emb = self.long_jepa.encode(long_ctx)

        return short_emb, med_emb, long_emb

    @torch.no_grad()
    def get_signal(
        self,
        short_emb: torch.Tensor,
        med_emb: torch.Tensor,
        long_emb: torch.Tensor,
        time_context: torch.Tensor,
    ) -> Tuple[Signal, float, Dict[str, any]]:
        """
        Get trading signal from MultiScalePolicy.

        Returns:
            (signal, confidence, details) where details contains:
                - combined_probs: Combined prediction probabilities
                - gate_weights: Gating weights for each scale
                - head_logits: Raw logits from each head
                - individual_signals: Dict mapping scale name to Signal
        """
        # Create mask (all available since we have full context)
        mask = torch.ones(1, 3, device=self.device)

        # Forward pass
        combined_logits, gate_weights, head_logits = self.policy(
            short_emb, med_emb, long_emb, time_context,
            mask=mask, return_head_logits=True
        )

        # Apply temperature scaling if configured
        if self.config.temperature != 1.0:
            combined_logits = combined_logits / self.config.temperature

        # Get combined prediction
        probs = torch.softmax(combined_logits, dim=-1)[0]

        # Extract individual head signals (for alignment check and recording)
        individual_signals = {}
        threshold = self.config.individual_threshold
        for name, logits in head_logits.items():
            head_probs = torch.softmax(logits, dim=-1)[0]
            # Use threshold-based signal extraction for more signals
            if head_probs[Signal.LONG.value].item() > threshold:
                individual_signals[name] = Signal.LONG
            elif head_probs[Signal.SHORT.value].item() > threshold:
                individual_signals[name] = Signal.SHORT
            else:
                individual_signals[name] = Signal.NEUTRAL

        # Check if all 3 signals align (all LONG or all SHORT)
        all_signals = list(individual_signals.values())
        all_long = all(s == Signal.LONG for s in all_signals)
        all_short = all(s == Signal.SHORT for s in all_signals)

        # Determine final signal
        if self.config.require_all_align:
            # Only trade when all 3 heads agree on direction
            if all_long:
                signal = Signal.LONG
                confidence = min(
                    torch.softmax(head_logits["short"], dim=-1)[0][Signal.LONG.value].item(),
                    torch.softmax(head_logits["med"], dim=-1)[0][Signal.LONG.value].item(),
                    torch.softmax(head_logits["long"], dim=-1)[0][Signal.LONG.value].item(),
                )
            elif all_short:
                signal = Signal.SHORT
                confidence = min(
                    torch.softmax(head_logits["short"], dim=-1)[0][Signal.SHORT.value].item(),
                    torch.softmax(head_logits["med"], dim=-1)[0][Signal.SHORT.value].item(),
                    torch.softmax(head_logits["long"], dim=-1)[0][Signal.SHORT.value].item(),
                )
            else:
                signal = Signal.NEUTRAL
                confidence = 0.0
        else:
            # Use combined signal (original behavior)
            mode = self.config.calibration_mode

            if mode == "confidence":
                pred_idx = probs.argmax().item()
                confidence = probs[pred_idx].item()
                if confidence < self.config.min_confidence:
                    signal = Signal.NEUTRAL
                else:
                    signal = Signal(pred_idx)

            elif mode == "temperature":
                pred_idx = probs.argmax().item()
                confidence = probs[pred_idx].item()
                signal = Signal(pred_idx)

            elif mode == "threshold":
                confidence = probs.max().item()
                thresh = self.config.directional_threshold
                if probs[Signal.LONG.value].item() > thresh:
                    signal = Signal.LONG
                    confidence = probs[Signal.LONG.value].item()
                elif probs[Signal.SHORT.value].item() > thresh:
                    signal = Signal.SHORT
                    confidence = probs[Signal.SHORT.value].item()
                else:
                    signal = Signal.NEUTRAL
                    confidence = probs[Signal.NEUTRAL.value].item()

            else:  # mode == "none" or default
                pred_idx = probs.argmax().item()
                confidence = probs[pred_idx].item()
                signal = Signal(pred_idx)

        return signal, confidence, {
            "combined_probs": probs,
            "gate_weights": gate_weights[0],
            "head_logits": head_logits,
            "individual_signals": individual_signals,
        }

    def get_equity(self, current_price: float, timestamp: datetime) -> float:
        """Calculate current total equity."""
        equity = self.cash

        if self.position is not None:
            # Get current option price
            opt_price = self.options.get_option_price(
                self.position.option_ticker, timestamp
            )
            if opt_price is None:
                # Fallback: estimate based on underlying move
                underlying_move = (current_price - self.position.entry_underlying_price)
                if self.position.position_type == PositionType.PUT:
                    underlying_move = -underlying_move
                # Rough delta approximation
                opt_price = max(0.01, self.position.entry_price + underlying_move * 0.5)

            position_value = opt_price * self.position.quantity * self.config.options_multiplier
            equity += position_value

        return equity

    def get_position_pnl_pct(self, timestamp: datetime) -> float:
        """Get current position P&L percentage."""
        if self.position is None:
            return 0.0

        current_opt_price = self.options.get_option_price(
            self.position.option_ticker, timestamp
        )
        if current_opt_price is None:
            return 0.0

        pnl_pct = (current_opt_price - self.position.entry_price) / self.position.entry_price * 100
        return pnl_pct

    def update_trailing_stop(self, current_pnl_pct: float) -> None:
        """Update trailing stop based on current P&L."""
        if self.position is None:
            return

        # Update high water mark
        if current_pnl_pct > self.position.high_water_mark_pct:
            self.position.high_water_mark_pct = current_pnl_pct

        # Check if we should activate trailing stop
        if current_pnl_pct >= self.config.trailing_activation_pct:
            # Calculate new stop level: HWM - trailing offset
            new_stop = self.position.high_water_mark_pct - self.config.trailing_offset_pct

            # Only move stop up, never down
            if new_stop > self.position.current_stop_loss_pct:
                self.position.current_stop_loss_pct = new_stop

    def check_exit_conditions(
        self,
        timestamp: datetime,
        current_pnl_pct: float,
        current_signal: Signal,
    ) -> Optional[str]:
        """
        Check all exit conditions.

        Returns exit reason or None if no exit.
        """
        if self.position is None:
            return None

        # 1. Signal reversal - exit immediately
        if self.position.position_type == PositionType.CALL and current_signal == Signal.SHORT:
            return "signal_reversal"
        if self.position.position_type == PositionType.PUT and current_signal == Signal.LONG:
            return "signal_reversal"

        # 2. Take profit
        if current_pnl_pct >= self.config.take_profit_pct:
            return "take_profit"

        # 3. Stop loss (including trailing)
        if current_pnl_pct <= self.position.current_stop_loss_pct:
            if self.position.current_stop_loss_pct >= 0:
                return "trailing_stop"
            else:
                return "stop_loss"

        # 4. Time limit
        minutes_held = (timestamp - self.position.barrier_start_time).total_seconds() / 60
        if minutes_held >= self.config.time_limit_minutes:
            return "time_limit"

        return None

    def check_signal_confirmation(self, current_signal: Signal) -> bool:
        """Check if current signal confirms position direction."""
        if self.position is None:
            return False

        if self.position.position_type == PositionType.CALL and current_signal == Signal.LONG:
            return True
        if self.position.position_type == PositionType.PUT and current_signal == Signal.SHORT:
            return True

        return False

    def open_position(
        self,
        signal: Signal,
        timestamp: datetime,
        underlying_price: float,
    ) -> bool:
        """Open a new position based on signal."""
        if signal == Signal.NEUTRAL:
            return False

        position_type = PositionType.CALL if signal == Signal.LONG else PositionType.PUT

        # Find 0DTE option
        result = self.options.find_0dte_option(timestamp, position_type, underlying_price)
        if result is None:
            if self.verbose:
                print(f"  [{timestamp.strftime('%H:%M')}] No 0DTE option found for {position_type.value}")
            return False

        ticker, price, strike = result

        # Calculate position size
        equity = self.get_equity(underlying_price, timestamp)
        max_position_value = equity * self.config.max_position_pct
        contract_value = price * self.config.options_multiplier
        quantity = max(1, int(max_position_value / contract_value))

        # Check if we have enough cash
        cost = price * quantity * self.config.options_multiplier
        if cost > self.cash:
            quantity = int(self.cash / (price * self.config.options_multiplier))
            if quantity < 1:
                return False
            cost = price * quantity * self.config.options_multiplier

        self.cash -= cost

        self.position = Position(
            position_type=position_type,
            option_ticker=ticker,
            strike=strike,
            entry_time=timestamp,
            entry_price=price,
            entry_underlying_price=underlying_price,
            quantity=quantity,
            barrier_start_time=timestamp,
            current_stop_loss_pct=-self.config.initial_stop_loss_pct,
            high_water_mark_pct=0.0,
        )

        if self.verbose:
            print(f"  [{timestamp.strftime('%H:%M')}] OPEN {quantity} {position_type.value.upper()} "
                  f"@ ${price:.2f} (strike {strike}, SPY ${underlying_price:.2f})")

        return True

    def close_position(
        self,
        timestamp: datetime,
        underlying_price: float,
        reason: str,
    ) -> Trade:
        """Close current position and record trade."""
        if self.position is None:
            return None

        # Get exit price
        exit_price = self.options.get_option_price(self.position.option_ticker, timestamp)
        if exit_price is None:
            # Fallback estimation
            underlying_move = (underlying_price - self.position.entry_underlying_price)
            if self.position.position_type == PositionType.PUT:
                underlying_move = -underlying_move
            exit_price = max(0.01, self.position.entry_price + underlying_move * 0.5)

        # Calculate P&L
        proceeds = exit_price * self.position.quantity * self.config.options_multiplier
        cost = self.position.entry_price * self.position.quantity * self.config.options_multiplier
        pnl = proceeds - cost
        pnl_pct = (exit_price - self.position.entry_price) / self.position.entry_price * 100

        self.cash += proceeds

        holding_minutes = (timestamp - self.position.entry_time).total_seconds() / 60

        trade = Trade(
            position_type=self.position.position_type,
            option_ticker=self.position.option_ticker,
            strike=self.position.strike,
            entry_time=self.position.entry_time,
            exit_time=timestamp,
            entry_price=self.position.entry_price,
            exit_price=exit_price,
            entry_underlying=self.position.entry_underlying_price,
            exit_underlying=underlying_price,
            quantity=self.position.quantity,
            pnl=pnl,
            pnl_pct=pnl_pct,
            exit_reason=reason,
            holding_minutes=holding_minutes,
        )

        self.trades.append(trade)

        if self.verbose:
            pnl_sign = "+" if pnl >= 0 else ""
            print(f"  [{timestamp.strftime('%H:%M')}] CLOSE {self.position.position_type.value.upper()} "
                  f"@ ${exit_price:.2f} | P&L: {pnl_sign}${pnl:.2f} ({pnl_pct:+.1f}%) | {reason}")

        self.position = None
        return trade

    def run_backtest(
        self,
        normalized_df: pd.DataFrame,
        raw_closes: np.ndarray,
        timestamps: pd.DatetimeIndex,
    ) -> Dict:
        """
        Run full backtest.

        Args:
            normalized_df: Normalized features DataFrame
            raw_closes: Raw close prices for underlying
            timestamps: Datetime index
        """
        self.reset()

        # Convert to tensor
        data_tensor = torch.tensor(normalized_df.values, dtype=torch.float32)
        data_tensor = torch.nan_to_num(data_tensor, nan=0.0, posinf=10.0, neginf=-10.0)
        data_tensor = torch.clamp(data_tensor, -10, 10).to(self.device)

        # Group by day
        dates = pd.Series(timestamps).dt.date.unique()
        print(f"\nRunning backtest on {len(dates)} trading days...")

        signal_counts = {s: 0 for s in Signal}

        # Convert to timezone-naive for easier handling
        if timestamps.tz is not None:
            timestamps = timestamps.tz_localize(None)

        for day_idx, day in enumerate(tqdm(dates, desc="Backtesting")):
            # Get indices for this day
            day_mask = pd.Series(timestamps).dt.date == day
            day_indices = np.where(day_mask)[0]

            if len(day_indices) == 0:
                continue

            # Market hours filter (9:30 - 16:00)
            market_open = time(9, 30)
            market_close = time(16, 0)

            for idx in day_indices:
                ts = timestamps[idx]

                # Skip pre/post market
                if ts.time() < market_open or ts.time() > market_close:
                    continue

                # Skip last 5 minutes (force close)
                close_dt = datetime.combine(ts.date(), market_close)
                minutes_to_close = (close_dt - ts).total_seconds() / 60
                if minutes_to_close < 5:
                    # Force close any position
                    if self.position is not None:
                        self.close_position(ts, raw_closes[idx], "eod_close")
                    continue

                # Get embeddings
                embeddings = self.get_embeddings(data_tensor, idx)
                if embeddings is None:
                    continue

                short_emb, med_emb, long_emb = embeddings

                # Create time context
                time_to_close = minutes_to_close / 390  # Normalized by trading day
                time_of_day = (ts.hour * 60 + ts.minute - 9 * 60 - 30) / 390
                time_context = torch.tensor([[time_of_day, time_to_close]],
                                           dtype=torch.float32, device=self.device)

                # Get signal
                signal, confidence, details = self.get_signal(short_emb, med_emb, long_emb, time_context)
                signal_counts[signal] += 1

                underlying_price = raw_closes[idx]

                # Record individual model signals for HTML report
                if "individual_signals" in details:
                    ind_sigs = details["individual_signals"]
                    # Map head names to model names
                    head_to_model = {"short": "1m", "med": "5m", "long": "15m"}
                    for head_name, head_signal in ind_sigs.items():
                        model_name = head_to_model.get(head_name, head_name)
                        if head_signal != Signal.NEUTRAL:
                            sig_type = "long" if head_signal == Signal.LONG else "short"
                            self.individual_signals[model_name].append(
                                (ts, sig_type, underlying_price)
                            )

                # Record equity
                self.equity_curve.append((ts, self.get_equity(underlying_price, ts)))

                # If we have a position
                if self.position is not None:
                    current_pnl_pct = self.get_position_pnl_pct(ts)

                    # Update trailing stop
                    self.update_trailing_stop(current_pnl_pct)

                    # Check for signal confirmation (reset barriers)
                    if self.check_signal_confirmation(signal):
                        self.position.barrier_start_time = ts
                        if self.verbose:
                            print(f"  [{ts.strftime('%H:%M')}] Signal confirmation - barriers reset")

                    # Check exit conditions
                    exit_reason = self.check_exit_conditions(ts, current_pnl_pct, signal)
                    if exit_reason:
                        self.close_position(ts, underlying_price, exit_reason)

                # If no position and non-neutral signal
                if self.position is None and signal != Signal.NEUTRAL:
                    if confidence >= self.config.min_confidence:
                        # Don't trade in last 30 minutes
                        if minutes_to_close >= 30:
                            self.open_position(signal, ts, underlying_price)

            # End of day - force close
            if self.position is not None:
                last_idx = day_indices[-1]
                last_ts = timestamps[last_idx]
                self.close_position(last_ts, raw_closes[last_idx], "eod_close")

            # Daily summary
            if self.verbose and (day_idx + 1) % 50 == 0:
                equity = self.get_equity(raw_closes[day_indices[-1]], timestamps[day_indices[-1]])
                ret_pct = (equity - self.config.initial_capital) / self.config.initial_capital * 100
                print(f"\n  Day {day_idx + 1}: Equity ${equity:,.2f} ({ret_pct:+.2f}%)")

        # Print summary
        print("\n" + "=" * 70)
        print("BACKTEST SUMMARY")
        print("=" * 70)

        print("\nSignal Distribution:")
        total_signals = sum(signal_counts.values())
        for s, count in signal_counts.items():
            pct = count / total_signals * 100 if total_signals > 0 else 0
            print(f"  {s.name}: {count:,} ({pct:.1f}%)")

        metrics = self.get_metrics()

        print(f"\nPerformance Metrics:")
        print(f"  Total Trades: {metrics['total_trades']}")
        print(f"  Win Rate: {metrics['win_rate']:.1f}%")
        print(f"  Avg P&L: ${metrics['avg_pnl']:.2f}")
        print(f"  Total P&L: ${metrics['total_pnl']:.2f}")
        print(f"  Return: {metrics['return_pct']:.2f}%")
        print(f"  Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
        print(f"  Max Drawdown: {metrics['max_drawdown']:.2f}%")
        print(f"  Profit Factor: {metrics['profit_factor']:.2f}")

        print(f"\nExit Reasons:")
        for reason, count in metrics['exit_reasons'].items():
            print(f"  {reason}: {count}")

        print(f"\nPosition Types:")
        print(f"  CALL: {metrics['call_trades']} trades")
        print(f"  PUT: {metrics['put_trades']} trades")

        return metrics

    def get_metrics(self) -> Dict:
        """Calculate performance metrics."""
        if not self.trades:
            return {
                'total_trades': 0,
                'win_rate': 0.0,
                'avg_pnl': 0.0,
                'total_pnl': 0.0,
                'return_pct': 0.0,
                'sharpe_ratio': 0.0,
                'max_drawdown': 0.0,
                'profit_factor': 0.0,
                'exit_reasons': {},
                'call_trades': 0,
                'put_trades': 0,
            }

        pnls = [t.pnl for t in self.trades]
        wins = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p <= 0]

        total_pnl = sum(pnls)
        avg_pnl = np.mean(pnls)
        std_pnl = np.std(pnls) if len(pnls) > 1 else 1.0

        win_rate = len(wins) / len(pnls) * 100

        # Sharpe (annualized, assuming ~250 trading days)
        sharpe = (avg_pnl / std_pnl) * np.sqrt(250) if std_pnl > 0 else 0

        # Profit factor
        gross_profit = sum(wins) if wins else 0
        gross_loss = abs(sum(losses)) if losses else 1
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0

        # Max drawdown from equity curve
        if self.equity_curve:
            equities = [e[1] for e in self.equity_curve]
            peak = equities[0]
            max_dd = 0
            for eq in equities:
                if eq > peak:
                    peak = eq
                dd = (peak - eq) / peak * 100
                max_dd = max(max_dd, dd)
        else:
            max_dd = 0

        # Exit reasons
        exit_reasons = {}
        for t in self.trades:
            exit_reasons[t.exit_reason] = exit_reasons.get(t.exit_reason, 0) + 1

        # Position types
        call_trades = len([t for t in self.trades if t.position_type == PositionType.CALL])
        put_trades = len([t for t in self.trades if t.position_type == PositionType.PUT])

        final_equity = self.equity_curve[-1][1] if self.equity_curve else self.cash
        return_pct = (final_equity - self.config.initial_capital) / self.config.initial_capital * 100

        return {
            'total_trades': len(self.trades),
            'win_rate': win_rate,
            'avg_pnl': avg_pnl,
            'total_pnl': total_pnl,
            'return_pct': return_pct,
            'sharpe_ratio': sharpe,
            'max_drawdown': max_dd,
            'profit_factor': profit_factor,
            'exit_reasons': exit_reasons,
            'call_trades': call_trades,
            'put_trades': put_trades,
            'final_equity': final_equity,
        }


def load_jepa(checkpoint_path: str, device: torch.device) -> LeJEPA:
    """Load and freeze LeJEPA model."""
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # Get config and filter to only accepted params
    config = checkpoint.get("config", {})
    accepted_params = {
        "input_dim", "d_model", "nhead", "num_layers", "embedding_dim",
        "max_context_len", "dropout", "lambda_reg", "reg_type"
    }
    filtered_config = {k: v for k, v in config.items() if k in accepted_params}

    model = LeJEPA(**filtered_config)

    # Load state dict, handling compiled model prefixes
    state_dict = checkpoint["state_dict"]
    cleaned_state_dict = {}
    for key, value in state_dict.items():
        new_key = key.replace("._orig_mod", "")
        cleaned_state_dict[new_key] = value

    # Use strict=False to handle models with extra modules (e.g., premarket encoders)
    model.load_state_dict(cleaned_state_dict, strict=False)
    model = model.to(device)
    model.eval()

    for param in model.parameters():
        param.requires_grad = False
    return model


def load_policy(checkpoint_path: str, device: torch.device) -> MultiScalePolicy:
    """Load MultiScalePolicy model."""
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # Handle different checkpoint formats
    if "config" in checkpoint:
        config = checkpoint["config"]
        model = MultiScalePolicy(
            emb_dim=config.get("emb_dim", 64),
            time_dim=config.get("time_dim", 2),
            hidden_dim=config.get("hidden_dim", 128),
            n_positions=config.get("n_positions", 3),
            dropout=config.get("dropout", 0.3),
        )
        model.load_state_dict(checkpoint["state_dict"])
    else:
        # Assume state dict only
        model = MultiScalePolicy()
        model.load_state_dict(checkpoint)

    model = model.to(device)
    model.eval()
    return model


def parse_args():
    parser = argparse.ArgumentParser(description="Multi-Scale Policy Backtest")

    # Model checkpoints
    parser.add_argument("--policy-checkpoint", type=str, required=True,
                        help="Path to MultiScalePolicy checkpoint")
    parser.add_argument("--short-jepa-checkpoint", type=str, required=True,
                        help="Path to short-context LeJEPA checkpoint")
    parser.add_argument("--med-jepa-checkpoint", type=str, required=True,
                        help="Path to medium-context LeJEPA checkpoint")
    parser.add_argument("--long-jepa-checkpoint", type=str, required=True,
                        help="Path to long-context LeJEPA checkpoint")

    # Data
    parser.add_argument("--data-dir", type=str, default="data/training-1m-normalized/SPY",
                        help="Directory with normalized data")
    parser.add_argument("--raw-dir", type=str, default="data/training-1m-raw/SPY",
                        help="Directory with raw price data")
    parser.add_argument("--options-dir", type=str, default="data/options-1m/SPY",
                        help="Directory with options data")
    parser.add_argument("--start-date", type=str, default="2020-01-01",
                        help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end-date", type=str, default="2025-08-31",
                        help="End date (YYYY-MM-DD)")

    # Backtest config
    parser.add_argument("--initial-capital", type=float, default=100_000,
                        help="Initial capital")
    parser.add_argument("--max-position-pct", type=float, default=0.05,
                        help="Max position as percent of equity")
    parser.add_argument("--take-profit-pct", type=float, default=50.0,
                        help="Take profit threshold (%%)")
    parser.add_argument("--stop-loss-pct", type=float, default=25.0,
                        help="Initial stop loss threshold (%%)")
    parser.add_argument("--time-limit-minutes", type=int, default=120,
                        help="Max holding time in minutes")
    parser.add_argument("--trailing-activation-pct", type=float, default=10.0,
                        help="Profit level to activate trailing stop (%%)")
    parser.add_argument("--trailing-offset-pct", type=float, default=10.0,
                        help="Trailing stop offset (%%)")

    # Calibration options
    parser.add_argument("--calibration-mode", type=str, default="none",
                        choices=["none", "confidence", "temperature", "threshold"],
                        help="Signal calibration mode")
    parser.add_argument("--min-confidence", type=float, default=0.0,
                        help="Min confidence to trade (default 0.0 = no filter)")
    parser.add_argument("--temperature", type=float, default=1.0,
                        help="Temperature for softmax (default 1.0 = no scaling)")
    parser.add_argument("--directional-threshold", type=float, default=0.0,
                        help="Min prob for LONG/SHORT (default 0.0 = use argmax)")

    # Signal alignment options
    parser.add_argument("--require-all-align", action="store_true", default=True,
                        help="Only trade when all 3 model heads agree (default: True)")
    parser.add_argument("--no-require-all-align", dest="require_all_align", action="store_false",
                        help="Trade based on combined signal (don't require alignment)")
    parser.add_argument("--individual-threshold", type=float, default=0.4,
                        help="Min prob for individual head signal (lower = more signals, default 0.4)")

    # Other
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--report", type=str, default=None,
                        help="Path to save HTML report")

    return parser.parse_args()


def main():
    args = parse_args()

    print("=" * 70)
    print("Multi-Scale Policy Backtest")
    print("=" * 70)

    # Device
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"Device: {device}")

    # Load models
    print("\nLoading models...")
    print(f"  Policy: {args.policy_checkpoint}")
    policy = load_policy(args.policy_checkpoint, device)

    print(f"  Short JEPA: {args.short_jepa_checkpoint}")
    short_jepa = load_jepa(args.short_jepa_checkpoint, device)

    print(f"  Med JEPA: {args.med_jepa_checkpoint}")
    med_jepa = load_jepa(args.med_jepa_checkpoint, device)

    print(f"  Long JEPA: {args.long_jepa_checkpoint}")
    long_jepa = load_jepa(args.long_jepa_checkpoint, device)

    # Create config
    config = BacktestConfig(
        initial_capital=args.initial_capital,
        max_position_pct=args.max_position_pct,
        take_profit_pct=args.take_profit_pct,
        initial_stop_loss_pct=args.stop_loss_pct,
        time_limit_minutes=args.time_limit_minutes,
        trailing_activation_pct=args.trailing_activation_pct,
        trailing_offset_pct=args.trailing_offset_pct,
        # Calibration
        calibration_mode=args.calibration_mode,
        min_confidence=args.min_confidence,
        temperature=args.temperature,
        directional_threshold=args.directional_threshold,
        # Signal alignment
        require_all_align=args.require_all_align,
        individual_threshold=args.individual_threshold,
    )

    print(f"\nBacktest Configuration:")
    print(f"  Initial Capital: ${config.initial_capital:,.2f}")
    print(f"  Max Position: {config.max_position_pct*100:.1f}%")
    print(f"  Take Profit: {config.take_profit_pct:.1f}%")
    print(f"  Stop Loss: {config.initial_stop_loss_pct:.1f}%")
    print(f"  Time Limit: {config.time_limit_minutes} minutes")
    print(f"  Trailing Activation: {config.trailing_activation_pct:.1f}%")
    print(f"  Trailing Offset: {config.trailing_offset_pct:.1f}%")

    # Print signal alignment settings
    print(f"\nSignal Alignment:")
    print(f"  Require All Align: {config.require_all_align}")
    print(f"  Individual Threshold: {config.individual_threshold:.2f}")

    # Print calibration settings
    print(f"\nCalibration Mode: {config.calibration_mode}")
    if config.calibration_mode == "confidence":
        print(f"  Min Confidence: {config.min_confidence:.2f}")
    elif config.calibration_mode == "temperature":
        print(f"  Temperature: {config.temperature:.2f}")
    elif config.calibration_mode == "threshold":
        print(f"  Directional Threshold: {config.directional_threshold:.2f}")

    # Options provider
    options = OptionsDataProvider(args.options_dir)

    # Create backtester
    backtester = MultiScaleBacktester(
        policy=policy,
        short_jepa=short_jepa,
        med_jepa=med_jepa,
        long_jepa=long_jepa,
        config=config,
        device=device,
        options_provider=options,
        verbose=args.verbose,
    )

    # Load data
    print(f"\nLoading data from {args.start_date} to {args.end_date}...")

    data_dir = Path(args.data_dir)
    raw_dir = Path(args.raw_dir)

    start_date = datetime.strptime(args.start_date, "%Y-%m-%d").date()
    end_date = datetime.strptime(args.end_date, "%Y-%m-%d").date()

    # Load normalized data
    normalized_dfs = []
    raw_closes = []
    all_timestamps = []

    # Iterate through months
    current = start_date
    file_count = 0
    while current <= end_date:
        month_str = current.strftime("%Y-%m")
        norm_month_dir = data_dir / month_str
        raw_month_dir = raw_dir / month_str

        if norm_month_dir.exists():
            for f in sorted(norm_month_dir.glob("*.parquet")):
                try:
                    # Try to parse filename as date (DD.parquet or YYYY-MM-DD.parquet)
                    stem = f.stem
                    if len(stem) == 2:  # DD format
                        file_date = datetime.strptime(f"{month_str}-{stem}", "%Y-%m-%d").date()
                    else:
                        file_date = datetime.strptime(stem, "%Y-%m-%d").date()

                    if start_date <= file_date <= end_date:
                        norm_df = pd.read_parquet(f)

                        # Load corresponding raw data
                        raw_file = raw_month_dir / f.name
                        if raw_file.exists():
                            raw_df = pd.read_parquet(raw_file)
                            if "close" in raw_df.columns:
                                # Extract timestamps
                                if "timestamp" in norm_df.columns:
                                    ts = pd.to_datetime(norm_df["timestamp"])
                                    all_timestamps.extend(ts.tolist())
                                    # Drop timestamp column for features
                                    norm_df = norm_df.drop(columns=["timestamp"])
                                elif "timestamp" in raw_df.columns:
                                    ts = pd.to_datetime(raw_df["timestamp"])
                                    all_timestamps.extend(ts.tolist())

                                normalized_dfs.append(norm_df)
                                raw_closes.append(raw_df["close"].values)
                                file_count += 1
                except ValueError as e:
                    continue

        # Move to next month
        if current.month == 12:
            current = current.replace(year=current.year + 1, month=1)
        else:
            current = current.replace(month=current.month + 1)

    if not normalized_dfs:
        print("ERROR: No data found in date range!")
        return

    # Combine data
    full_df = pd.concat(normalized_dfs, ignore_index=True)
    full_closes = np.concatenate(raw_closes)
    timestamps = pd.DatetimeIndex(all_timestamps)

    print(f"Loaded {len(full_df):,} bars from {file_count} files")
    print(f"Date range: {timestamps.min().date()} to {timestamps.max().date()}")
    print(f"Features: {full_df.shape[1]}")

    # Verify alignment
    if len(timestamps) != len(full_df) or len(timestamps) != len(full_closes):
        print(f"ERROR: Data alignment issue!")
        print(f"  Timestamps: {len(timestamps)}")
        print(f"  Features: {len(full_df)}")
        print(f"  Closes: {len(full_closes)}")
        return

    # Run backtest
    metrics = backtester.run_backtest(full_df, full_closes, timestamps)

    print("\n" + "=" * 70)
    print("Backtest Complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()
