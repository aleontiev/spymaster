"""
Backtesting Engine for 0DTE Options Trading.

Provides realistic simulation of options trading with:
- Real options price data from Polygon (when available)
- Fallback to Black-Scholes simulation
- Mock order execution with slippage
- Position tracking
- P&L calculation with theta decay
- Performance metrics (Sharpe, Drawdown, Win Rate)

Critical for 0DTE:
- Theta decay accelerates exponentially in final 2 hours
- Wide bid-ask spreads on options
- Slippage is significant for market orders
"""
from dataclasses import dataclass, field
from datetime import datetime, time
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import re
import glob

import numpy as np
import pandas as pd


class OrderSide(Enum):
    """Order side."""
    BUY = "buy"
    SELL = "sell"


class OrderType(Enum):
    """Order type."""
    LIMIT = "limit"
    MARKET = "market"  # Discouraged for options


class PositionType(Enum):
    """Position type for options."""
    CALL = "call"
    PUT = "put"


@dataclass
class Order:
    """Represents a trading order."""
    order_id: str
    symbol: str
    side: OrderSide
    order_type: OrderType
    quantity: int
    limit_price: Optional[float] = None
    position_type: Optional[PositionType] = None
    timestamp: Optional[datetime] = None
    filled: bool = False
    fill_price: Optional[float] = None
    fill_timestamp: Optional[datetime] = None


@dataclass
class Position:
    """Represents an open position."""
    position_id: str
    symbol: str
    position_type: PositionType
    quantity: int
    entry_price: float
    entry_timestamp: datetime
    strike: float
    expiry: datetime  # For 0DTE, same day
    current_price: float = 0.0
    unrealized_pnl: float = 0.0


@dataclass
class Trade:
    """Represents a completed trade (entry + exit)."""
    trade_id: str
    symbol: str
    position_type: PositionType
    quantity: int
    entry_price: float
    exit_price: float
    entry_timestamp: datetime
    exit_timestamp: datetime
    pnl: float
    holding_period_minutes: float


@dataclass
class BacktestConfig:
    """Configuration for backtesting."""
    initial_capital: float = 100_000.0
    max_position_size: float = 0.1  # 10% of capital per position
    slippage_bps: float = 10.0  # 10 basis points slippage
    commission_per_contract: float = 0.65  # Typical options commission
    options_multiplier: int = 100  # Standard options multiplier
    spread_pct: float = 0.5  # Bid-ask spread as % of mid price
    theta_decay_multiplier: float = 1.0  # Adjust theta decay speed


class OptionsSimulator:
    """
    Simulates options pricing with realistic dynamics.

    Models:
    - Intrinsic value
    - Time decay (theta) with exponential acceleration
    - Simplified volatility dynamics
    """

    def __init__(
        self,
        risk_free_rate: float = 0.05,
        base_volatility: float = 0.20,
    ) -> None:
        """
        Initialize options simulator.

        Args:
            risk_free_rate: Annual risk-free rate
            base_volatility: Base annual volatility
        """
        self.risk_free_rate = risk_free_rate
        self.base_volatility = base_volatility

    def estimate_option_price(
        self,
        underlying_price: float,
        strike: float,
        time_to_expiry_hours: float,
        position_type: PositionType,
        volatility: Optional[float] = None,
    ) -> Tuple[float, float, float]:
        """
        Estimate option price using simplified Black-Scholes-like model.

        Args:
            underlying_price: Current underlying price
            strike: Strike price
            time_to_expiry_hours: Hours until expiry
            position_type: CALL or PUT
            volatility: Optional implied volatility override

        Returns:
            Tuple of (mid_price, bid_price, ask_price)
        """
        vol = volatility if volatility else self.base_volatility

        # Time in years
        T = max(time_to_expiry_hours / (252 * 6.5), 1e-6)  # 6.5 trading hours per day

        # Intrinsic value
        if position_type == PositionType.CALL:
            intrinsic = max(0, underlying_price - strike)
        else:
            intrinsic = max(0, strike - underlying_price)

        # Time value (simplified exponential decay)
        # Accelerates dramatically in last 2 hours
        if time_to_expiry_hours <= 2:
            decay_factor = np.exp(-3 * (2 - time_to_expiry_hours))
        else:
            decay_factor = 1.0

        # Simplified time value based on ATM approximation
        atm_time_value = underlying_price * vol * np.sqrt(T) * 0.4 * decay_factor

        # Adjust for moneyness
        moneyness = underlying_price / strike
        if position_type == PositionType.CALL:
            if moneyness > 1.05:  # Deep ITM
                time_value = atm_time_value * 0.3
            elif moneyness < 0.95:  # Deep OTM
                time_value = atm_time_value * 0.3 * np.exp(-10 * (0.95 - moneyness))
            else:
                time_value = atm_time_value
        else:
            if moneyness < 0.95:  # Deep ITM for puts
                time_value = atm_time_value * 0.3
            elif moneyness > 1.05:  # Deep OTM for puts
                time_value = atm_time_value * 0.3 * np.exp(-10 * (moneyness - 1.05))
            else:
                time_value = atm_time_value

        mid_price = intrinsic + time_value

        # Ensure minimum price
        mid_price = max(mid_price, 0.01)

        # Bid-ask spread (wider for 0DTE)
        spread_factor = 1.0 + (2.0 - min(time_to_expiry_hours, 2.0)) * 0.5
        half_spread = mid_price * 0.02 * spread_factor  # 2% base spread

        bid_price = max(0.01, mid_price - half_spread)
        ask_price = mid_price + half_spread

        return mid_price, bid_price, ask_price


class OptionsDataProvider:
    """
    Provides real options prices from Polygon data files.

    Loads options data for specific dates and provides price lookups
    by timestamp, strike, and option type. Falls back to OptionsSimulator
    if real data is not available.

    Usage:
        provider = OptionsDataProvider(options_dir="data/polygon/options")
        provider.load_date("2025-10-17")
        price = provider.get_option_price(
            timestamp=datetime(2025, 10, 17, 10, 30),
            strike=580.0,
            position_type=PositionType.CALL,
            underlying_price=582.5
        )
    """

    def __init__(
        self,
        options_dir: str = "data/polygon/options",
        fallback_simulator: Optional[OptionsSimulator] = None,
    ) -> None:
        """
        Initialize options data provider.

        Args:
            options_dir: Directory containing options parquet files
            fallback_simulator: Simulator to use when real data unavailable
        """
        self.options_dir = Path(options_dir)
        self.fallback = fallback_simulator or OptionsSimulator()

        # Cache for loaded data: date_str -> DataFrame with processed options
        self._data_cache: Dict[str, pd.DataFrame] = {}

        # Track which dates have data available
        self._available_dates: set = set()
        self._scan_available_dates()

    def _scan_available_dates(self) -> None:
        """Scan options directory to find available dates."""
        if not self.options_dir.exists():
            return

        for file_path in self.options_dir.glob("SPY_*.parquet"):
            # Extract date from filename: SPY_2025-10-17.parquet
            match = re.search(r'SPY_(\d{4}-\d{2}-\d{2})\.parquet', file_path.name)
            if match:
                self._available_dates.add(match.group(1))

    def has_data_for_date(self, date: datetime) -> bool:
        """Check if real options data is available for a date."""
        date_str = date.strftime("%Y-%m-%d")
        return date_str in self._available_dates

    def load_date(self, date: datetime) -> bool:
        """
        Load options data for a specific date.

        Args:
            date: Date to load data for

        Returns:
            True if data was loaded, False if not available
        """
        date_str = date.strftime("%Y-%m-%d")

        # Already loaded
        if date_str in self._data_cache:
            return True

        # Check if data exists
        if date_str not in self._available_dates:
            return False

        # Load and process the file
        file_path = self.options_dir / f"SPY_{date_str}.parquet"
        try:
            df = pd.read_parquet(file_path)

            # Parse ticker to extract strike and option type
            # O:SPY251017C00580000 -> strike=580.0, type='C'
            def parse_ticker(ticker: str) -> Tuple[Optional[float], Optional[str]]:
                match = re.match(r'O:SPY\d{6}([CP])(\d{8})', ticker)
                if match:
                    opt_type = match.group(1)
                    strike = int(match.group(2)) / 1000
                    return strike, opt_type
                return None, None

            df['strike'], df['option_type'] = zip(*df['ticker'].apply(parse_ticker))

            # Convert timestamp and set as index
            df['window_start'] = pd.to_datetime(df['window_start'])

            # Remove rows with unparseable tickers
            df = df.dropna(subset=['strike', 'option_type'])

            # Create a lookup-friendly structure
            # Index by (option_type, strike) for fast access
            self._data_cache[date_str] = df

            return True

        except Exception as e:
            print(f"Warning: Failed to load options data for {date_str}: {e}")
            return False

    def get_option_price(
        self,
        timestamp: datetime,
        strike: float,
        position_type: PositionType,
        underlying_price: float,
    ) -> Tuple[float, float, float]:
        """
        Get option price at a specific timestamp.

        Args:
            timestamp: Time to look up price
            strike: Strike price
            position_type: CALL or PUT
            underlying_price: Current underlying price (for fallback)

        Returns:
            Tuple of (mid_price, bid_price, ask_price)
        """
        date_str = timestamp.strftime("%Y-%m-%d")

        # Try to load data if not cached
        if date_str not in self._data_cache:
            if not self.load_date(timestamp):
                # Fall back to simulator
                market_close = datetime.combine(timestamp.date(), time(16, 0))
                time_to_expiry = max(0, (market_close - timestamp).total_seconds() / 3600)
                return self.fallback.estimate_option_price(
                    underlying_price, strike, time_to_expiry, position_type
                )

        df = self._data_cache[date_str]
        opt_type = 'C' if position_type == PositionType.CALL else 'P'

        # Find the closest price for this strike and type
        # Look for data within 1 minute of the timestamp
        mask = (
            (df['option_type'] == opt_type) &
            (df['strike'] == strike) &
            (abs((df['window_start'] - timestamp).dt.total_seconds()) < 60)
        )

        matches = df[mask]

        if len(matches) > 0:
            # Use the closest match
            closest_idx = (matches['window_start'] - timestamp).abs().idxmin()
            row = matches.loc[closest_idx]

            mid_price = row['close']
            # Estimate bid/ask from high/low if available
            if 'high' in row and 'low' in row:
                half_spread = (row['high'] - row['low']) / 4
                bid_price = max(0.01, mid_price - half_spread)
                ask_price = mid_price + half_spread
            else:
                # Default 2% spread
                half_spread = mid_price * 0.01
                bid_price = max(0.01, mid_price - half_spread)
                ask_price = mid_price + half_spread

            return mid_price, bid_price, ask_price

        # Try nearby strikes if exact match not found
        # This handles cases where the exact strike wasn't traded at this minute
        nearby_mask = (
            (df['option_type'] == opt_type) &
            (abs(df['strike'] - strike) <= 2) &  # Within $2 of target strike
            (abs((df['window_start'] - timestamp).dt.total_seconds()) < 120)  # Within 2 minutes
        )

        nearby = df[nearby_mask]

        if len(nearby) > 0:
            # Use closest strike and time
            nearby = nearby.copy()
            nearby['strike_diff'] = abs(nearby['strike'] - strike)
            nearby['time_diff'] = abs((nearby['window_start'] - timestamp).dt.total_seconds())
            nearby = nearby.sort_values(['strike_diff', 'time_diff'])

            row = nearby.iloc[0]
            mid_price = row['close']

            # Adjust for strike difference using delta approximation
            # ATM options have ~0.50 delta, so $1 underlying move = $0.50 option move
            strike_diff = strike - row['strike']
            delta = 0.50 if position_type == PositionType.CALL else -0.50
            mid_price = mid_price - (strike_diff * delta)
            mid_price = max(0.01, mid_price)

            half_spread = mid_price * 0.02
            bid_price = max(0.01, mid_price - half_spread)
            ask_price = mid_price + half_spread

            return mid_price, bid_price, ask_price

        # Fall back to simulator if no real data found
        market_close = datetime.combine(timestamp.date(), time(16, 0))
        time_to_expiry = max(0, (market_close - timestamp).total_seconds() / 3600)
        return self.fallback.estimate_option_price(
            underlying_price, strike, time_to_expiry, position_type
        )

    def clear_cache(self) -> None:
        """Clear the data cache to free memory."""
        self._data_cache.clear()


class BacktestEngine:
    """
    Main backtesting engine for options trading strategies.

    Simulates:
    - Order execution with slippage
    - Position management
    - P&L tracking
    - Performance metrics

    Uses real options prices from Polygon data when available,
    falls back to Black-Scholes simulation otherwise.
    """

    def __init__(
        self,
        config: Optional[BacktestConfig] = None,
        options_dir: Optional[str] = None,
        use_real_prices: bool = True,
    ) -> None:
        """
        Initialize backtesting engine.

        Args:
            config: Backtest configuration
            options_dir: Directory containing options parquet files
            use_real_prices: Whether to use real options prices (default True)
        """
        self.config = config or BacktestConfig()
        self.options_sim = OptionsSimulator()
        self.use_real_prices = use_real_prices

        # Initialize options data provider for real prices
        if use_real_prices:
            options_path = options_dir or "data/polygon/options"
            self.options_data = OptionsDataProvider(
                options_dir=options_path,
                fallback_simulator=self.options_sim,
            )
        else:
            self.options_data = None

        # State
        self.cash = self.config.initial_capital
        self.positions: Dict[str, Position] = {}
        self.orders: List[Order] = []
        self.trades: List[Trade] = []
        self.equity_curve: List[Tuple[datetime, float]] = []

        # Counters
        self._order_counter = 0
        self._position_counter = 0
        self._trade_counter = 0

        # Current market state
        self.current_timestamp: Optional[datetime] = None
        self.current_underlying_price: float = 0.0

    def reset(self) -> None:
        """Reset engine state for new backtest."""
        self.cash = self.config.initial_capital
        self.positions.clear()
        self.orders.clear()
        self.trades.clear()
        self.equity_curve.clear()
        self._order_counter = 0
        self._position_counter = 0
        self._trade_counter = 0

    def _get_option_price(
        self,
        strike: float,
        position_type: PositionType,
    ) -> Tuple[float, float, float]:
        """
        Get option price using real data or simulator.

        Args:
            strike: Strike price
            position_type: CALL or PUT

        Returns:
            Tuple of (mid_price, bid_price, ask_price)
        """
        if self.use_real_prices and self.options_data is not None:
            return self.options_data.get_option_price(
                timestamp=self.current_timestamp,
                strike=strike,
                position_type=position_type,
                underlying_price=self.current_underlying_price,
            )
        else:
            market_close = datetime.combine(
                self.current_timestamp.date(),
                time(16, 0)
            )
            time_to_expiry = (market_close - self.current_timestamp).total_seconds() / 3600
            return self.options_sim.estimate_option_price(
                underlying_price=self.current_underlying_price,
                strike=strike,
                time_to_expiry_hours=time_to_expiry,
                position_type=position_type,
            )

    def update_market_state(
        self,
        timestamp: datetime,
        underlying_price: float,
    ) -> None:
        """
        Update current market state.

        Args:
            timestamp: Current timestamp
            underlying_price: Current underlying (SPY) price
        """
        self.current_timestamp = timestamp
        self.current_underlying_price = underlying_price

        # Update position values
        self._update_positions()

        # Record equity
        total_equity = self.get_total_equity()
        self.equity_curve.append((timestamp, total_equity))

    def _update_positions(self) -> None:
        """Update all position values based on current market."""
        if self.current_timestamp is None:
            return

        for pos_id, position in self.positions.items():
            mid_price, _, _ = self._get_option_price(
                strike=position.strike,
                position_type=position.position_type,
            )
            position.current_price = mid_price
            position.unrealized_pnl = (
                (mid_price - position.entry_price)
                * position.quantity
                * self.config.options_multiplier
            )

    def submit_order(
        self,
        symbol: str,
        side: OrderSide,
        quantity: int,
        position_type: PositionType,
        strike: float,
        limit_price: Optional[float] = None,
    ) -> Order:
        """
        Submit a new order.

        Args:
            symbol: Underlying symbol (e.g., "SPY")
            side: BUY or SELL
            quantity: Number of contracts
            position_type: CALL or PUT
            strike: Strike price
            limit_price: Limit price (required for limit orders)

        Returns:
            Submitted order
        """
        self._order_counter += 1
        order_id = f"ORD-{self._order_counter:06d}"

        order = Order(
            order_id=order_id,
            symbol=symbol,
            side=side,
            order_type=OrderType.LIMIT if limit_price else OrderType.MARKET,
            quantity=quantity,
            limit_price=limit_price,
            position_type=position_type,
            timestamp=self.current_timestamp,
        )

        # Immediate fill simulation
        self._fill_order(order, strike)

        self.orders.append(order)
        return order

    def _fill_order(self, order: Order, strike: float) -> None:
        """
        Simulate order fill with slippage.

        Args:
            order: Order to fill
            strike: Strike price for the option
        """
        if self.current_timestamp is None:
            return

        # Get option prices (uses real data if available)
        mid_price, bid_price, ask_price = self._get_option_price(
            strike=strike,
            position_type=order.position_type,
        )

        # Determine fill price with slippage
        slippage = mid_price * (self.config.slippage_bps / 10000)

        if order.side == OrderSide.BUY:
            fill_price = ask_price + slippage
        else:
            fill_price = max(0.01, bid_price - slippage)

        # Check if limit price allows fill
        if order.limit_price is not None:
            if order.side == OrderSide.BUY and fill_price > order.limit_price:
                return  # Order not filled
            if order.side == OrderSide.SELL and fill_price < order.limit_price:
                return  # Order not filled

        # Calculate cost
        contract_value = fill_price * self.config.options_multiplier * order.quantity
        commission = self.config.commission_per_contract * order.quantity

        if order.side == OrderSide.BUY:
            total_cost = contract_value + commission
            if total_cost > self.cash:
                return  # Insufficient funds

            self.cash -= total_cost

            # Create position
            self._position_counter += 1
            pos_id = f"POS-{self._position_counter:06d}"

            self.positions[pos_id] = Position(
                position_id=pos_id,
                symbol=order.symbol,
                position_type=order.position_type,
                quantity=order.quantity,
                entry_price=fill_price,
                entry_timestamp=self.current_timestamp,
                strike=strike,
                expiry=datetime.combine(self.current_timestamp.date(), time(16, 0)),
                current_price=fill_price,
            )

        else:  # SELL
            # Find and close position
            pos_to_close = None
            for pos_id, pos in self.positions.items():
                if (pos.position_type == order.position_type and
                    pos.quantity >= order.quantity):
                    pos_to_close = pos_id
                    break

            if pos_to_close is None:
                return  # No position to close

            position = self.positions[pos_to_close]

            # Calculate P&L
            proceeds = contract_value - commission
            self.cash += proceeds

            pnl = (fill_price - position.entry_price) * order.quantity * self.config.options_multiplier - (2 * commission)

            # Record trade
            self._trade_counter += 1
            trade = Trade(
                trade_id=f"TRD-{self._trade_counter:06d}",
                symbol=order.symbol,
                position_type=order.position_type,
                quantity=order.quantity,
                entry_price=position.entry_price,
                exit_price=fill_price,
                entry_timestamp=position.entry_timestamp,
                exit_timestamp=self.current_timestamp,
                pnl=pnl,
                holding_period_minutes=(
                    self.current_timestamp - position.entry_timestamp
                ).total_seconds() / 60,
            )
            self.trades.append(trade)

            # Remove or reduce position
            if position.quantity == order.quantity:
                del self.positions[pos_to_close]
            else:
                position.quantity -= order.quantity

        # Mark order as filled
        order.filled = True
        order.fill_price = fill_price
        order.fill_timestamp = self.current_timestamp

    def close_all_positions(self) -> None:
        """Close all open positions at current market prices."""
        positions_to_close = list(self.positions.values())

        for position in positions_to_close:
            self.submit_order(
                symbol=position.symbol,
                side=OrderSide.SELL,
                quantity=position.quantity,
                position_type=position.position_type,
                strike=position.strike,
            )

    def get_total_equity(self) -> float:
        """Get total equity (cash + position value)."""
        position_value = sum(
            pos.current_price * pos.quantity * self.config.options_multiplier
            for pos in self.positions.values()
        )
        return self.cash + position_value

    def get_metrics(self) -> Dict[str, float]:
        """
        Calculate performance metrics.

        Returns:
            Dictionary of performance metrics
        """
        if not self.trades:
            return {
                "total_trades": 0,
                "total_pnl": 0.0,
                "win_rate": 0.0,
                "avg_pnl": 0.0,
                "avg_win": 0.0,
                "avg_loss": 0.0,
                "expectancy": 0.0,
                "risk_reward": 0.0,
                "sharpe_ratio": 0.0,
                "max_drawdown": 0.0,
                "profit_factor": 0.0,
                "final_equity": self.get_total_equity(),
                "return_pct": (self.get_total_equity() - self.config.initial_capital) / self.config.initial_capital * 100,
            }

        # Basic metrics
        pnls = [t.pnl for t in self.trades]
        total_pnl = sum(pnls)
        wins = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p < 0]

        win_rate = len(wins) / len(pnls) if pnls else 0

        avg_pnl = np.mean(pnls) if pnls else 0.0
        std_pnl = np.std(pnls) if len(pnls) > 1 else 1.0

        # Sharpe ratio (annualized, assuming ~250 trading days)
        sharpe_ratio = (avg_pnl / std_pnl) * np.sqrt(250) if std_pnl > 0 else 0

        # Max drawdown from equity curve
        if self.equity_curve:
            equities = [e[1] for e in self.equity_curve]
            peak = equities[0]
            max_dd = 0
            for equity in equities:
                if equity > peak:
                    peak = equity
                dd = (peak - equity) / peak
                max_dd = max(max_dd, dd)
        else:
            max_dd = 0

        # Profit factor
        gross_profit = sum(wins) if wins else 0
        gross_loss = abs(sum(losses)) if losses else 1
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')

        # Expectancy: expected value per trade
        # E = (Win% * Avg Win) + (Loss% * Avg Loss)
        # Note: avg_loss is already negative
        avg_win = np.mean(wins) if wins else 0
        avg_loss = np.mean(losses) if losses else 0
        loss_rate = len(losses) / len(pnls) if pnls else 0
        expectancy = (win_rate * avg_win) + (loss_rate * avg_loss)

        # Risk/Reward ratio (avg win / |avg loss|)
        risk_reward = abs(avg_win / avg_loss) if avg_loss != 0 else float('inf')

        return {
            "total_trades": len(self.trades),
            "total_pnl": total_pnl,
            "win_rate": win_rate,
            "avg_pnl": avg_pnl,
            "avg_win": avg_win,
            "avg_loss": avg_loss,
            "expectancy": expectancy,
            "risk_reward": risk_reward,
            "sharpe_ratio": sharpe_ratio,
            "max_drawdown": max_dd,
            "profit_factor": profit_factor,
            "final_equity": self.get_total_equity(),
            "return_pct": (self.get_total_equity() - self.config.initial_capital) / self.config.initial_capital * 100,
        }

    def print_summary(self) -> None:
        """Print backtest summary."""
        metrics = self.get_metrics()

        print("\n" + "=" * 60)
        print("Backtest Summary")
        print("=" * 60)
        print(f"Initial Capital: ${self.config.initial_capital:,.2f}")
        print(f"Final Equity: ${metrics['final_equity']:,.2f}")
        print(f"Total Return: {metrics['return_pct']:.2f}%")
        print(f"\nTotal Trades: {metrics['total_trades']}")
        print(f"Win Rate: {metrics['win_rate']*100:.1f}%")
        print(f"Average P&L: ${metrics['avg_pnl']:.2f}")
        print(f"Average Win: ${metrics['avg_win']:.2f}")
        print(f"Average Loss: ${metrics['avg_loss']:.2f}")
        print(f"\nExpectancy (avg $ per trade): ${metrics['expectancy']:.2f}")
        print(f"Risk/Reward Ratio: {metrics['risk_reward']:.2f}")
        print(f"Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
        print(f"Max Drawdown: {metrics['max_drawdown']*100:.1f}%")
        print(f"Profit Factor: {metrics['profit_factor']:.2f}")
        print("=" * 60)


def test_backtest_engine() -> None:
    """Test the backtesting engine."""
    print("=" * 60)
    print("Testing Backtest Engine")
    print("=" * 60)

    # Create engine
    config = BacktestConfig(initial_capital=100_000)
    engine = BacktestEngine(config)

    # Simulate a trading day
    from datetime import timedelta

    base_date = datetime(2024, 1, 15, 9, 30)  # Market open
    spy_price = 470.0

    print(f"\nInitial capital: ${config.initial_capital:,.2f}")
    print(f"Starting SPY price: ${spy_price:.2f}")

    # Simulate market updates
    for i in range(390):  # 6.5 hours = 390 minutes
        timestamp = base_date + timedelta(minutes=i)

        # Random walk for SPY
        spy_price *= (1 + np.random.randn() * 0.0003)

        engine.update_market_state(timestamp, spy_price)

        # Simple strategy: buy call if we don't have positions and price is rising
        if i == 30:  # 30 minutes after open
            print(f"\nBuying call at {timestamp.strftime('%H:%M')}, SPY=${spy_price:.2f}")
            engine.submit_order(
                symbol="SPY",
                side=OrderSide.BUY,
                quantity=5,
                position_type=PositionType.CALL,
                strike=round(spy_price),
            )

        if i == 200:  # 3+ hours later
            print(f"Selling call at {timestamp.strftime('%H:%M')}, SPY=${spy_price:.2f}")
            engine.submit_order(
                symbol="SPY",
                side=OrderSide.SELL,
                quantity=5,
                position_type=PositionType.CALL,
                strike=round(spy_price - 1),  # Approximate strike
            )

    # Close any remaining positions at market close
    engine.close_all_positions()

    # Print summary
    engine.print_summary()

    print("\nTrade History:")
    for trade in engine.trades:
        print(f"  {trade.trade_id}: {trade.position_type.value} "
              f"Entry=${trade.entry_price:.2f} Exit=${trade.exit_price:.2f} "
              f"P&L=${trade.pnl:.2f}")


if __name__ == "__main__":
    test_backtest_engine()
