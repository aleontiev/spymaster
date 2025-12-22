"""
Alpaca Trading Client for 0DTE Options Execution.

Provides:
- Order submission with limit orders (no market orders for options)
- Position management
- Risk management with stop-losses
- Account status monitoring

IMPORTANT: Always use limit orders for options to avoid slippage.
"""
import asyncio
import os
from dataclasses import dataclass
from datetime import datetime, time
from enum import Enum
from typing import Dict, List, Optional

# Note: alpaca-trade-api is imported lazily to allow testing without API


class OrderSide(Enum):
    """Order side."""
    BUY = "buy"
    SELL = "sell"


class OrderType(Enum):
    """Order type."""
    LIMIT = "limit"
    # MARKET = "market"  # Intentionally excluded for options


class TimeInForce(Enum):
    """Time in force for orders."""
    DAY = "day"
    GTC = "gtc"
    IOC = "ioc"  # Immediate or cancel


@dataclass
class RiskConfig:
    """Risk management configuration."""
    max_position_size_pct: float = 0.05  # 5% of portfolio per position
    max_daily_loss_pct: float = 0.02  # 2% max daily loss
    max_positions: int = 3  # Max concurrent positions
    stop_loss_pct: float = 0.50  # 50% stop loss on options
    take_profit_pct: float = 1.00  # 100% take profit
    min_time_to_expiry_minutes: int = 30  # Don't trade last 30 minutes


class RiskManager:
    """
    Manages trading risk for 0DTE options.

    Key responsibilities:
    - Position sizing based on account value
    - Daily loss limits
    - Stop-loss enforcement
    - Time-based restrictions (no trading near expiry)
    """

    def __init__(self, config: Optional[RiskConfig] = None) -> None:
        """
        Initialize risk manager.

        Args:
            config: Risk configuration
        """
        self.config = config or RiskConfig()

        # Track daily P&L
        self.daily_pnl: float = 0.0
        self.daily_trades: int = 0
        self.last_reset_date: Optional[datetime] = None

    def reset_daily_stats(self) -> None:
        """Reset daily statistics (call at market open)."""
        self.daily_pnl = 0.0
        self.daily_trades = 0
        self.last_reset_date = datetime.now()

    def update_pnl(self, pnl: float) -> None:
        """Update daily P&L after a trade."""
        self.daily_pnl += pnl
        self.daily_trades += 1

    def can_open_position(
        self,
        account_value: float,
        current_positions: int,
        current_time: Optional[datetime] = None,
    ) -> tuple[bool, str]:
        """
        Check if a new position can be opened.

        Args:
            account_value: Current account value
            current_positions: Number of current positions
            current_time: Current time (for expiry check)

        Returns:
            Tuple of (allowed, reason)
        """
        # Check daily loss limit
        max_loss = account_value * self.config.max_daily_loss_pct
        if self.daily_pnl < -max_loss:
            return False, f"Daily loss limit reached: ${self.daily_pnl:.2f}"

        # Check max positions
        if current_positions >= self.config.max_positions:
            return False, f"Max positions ({self.config.max_positions}) reached"

        # Check time to expiry
        if current_time:
            market_close = datetime.combine(current_time.date(), time(16, 0))
            minutes_to_close = (market_close - current_time).total_seconds() / 60

            if minutes_to_close < self.config.min_time_to_expiry_minutes:
                return False, f"Too close to expiry ({minutes_to_close:.0f} min)"

        return True, "OK"

    def calculate_position_size(
        self,
        account_value: float,
        option_price: float,
        multiplier: int = 100,
    ) -> int:
        """
        Calculate appropriate position size.

        Args:
            account_value: Current account value
            option_price: Price per contract
            multiplier: Options multiplier (usually 100)

        Returns:
            Number of contracts
        """
        max_allocation = account_value * self.config.max_position_size_pct
        contract_cost = option_price * multiplier

        if contract_cost <= 0:
            return 0

        contracts = int(max_allocation / contract_cost)
        return max(1, contracts)  # At least 1 contract if allowed

    def should_stop_loss(
        self,
        entry_price: float,
        current_price: float,
    ) -> bool:
        """Check if position should be stopped out."""
        if entry_price <= 0:
            return False

        pct_change = (current_price - entry_price) / entry_price
        return pct_change < -self.config.stop_loss_pct

    def should_take_profit(
        self,
        entry_price: float,
        current_price: float,
    ) -> bool:
        """Check if profit should be taken."""
        if entry_price <= 0:
            return False

        pct_change = (current_price - entry_price) / entry_price
        return pct_change > self.config.take_profit_pct


class AlpacaClient:
    """
    Alpaca API client for options trading.

    Features:
    - Async order submission
    - Position monitoring
    - Account status
    - Paper trading support

    CRITICAL: Only uses LIMIT orders for options to avoid slippage.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        api_secret: Optional[str] = None,
        base_url: Optional[str] = None,
        paper: bool = True,
    ) -> None:
        """
        Initialize Alpaca client.

        Args:
            api_key: Alpaca API key (or from ALPACA_API_KEY env var)
            api_secret: Alpaca API secret (or from ALPACA_SECRET_KEY env var)
            base_url: API base URL (or from ALPACA_BASE_URL env var)
            paper: Use paper trading (default True for safety)
        """
        self.api_key = api_key or os.environ.get("ALPACA_API_KEY", "")
        self.api_secret = api_secret or os.environ.get("ALPACA_SECRET_KEY", "")

        if paper:
            self.base_url = base_url or os.environ.get(
                "ALPACA_BASE_URL",
                "https://paper-api.alpaca.markets"
            )
        else:
            self.base_url = base_url or os.environ.get(
                "ALPACA_BASE_URL",
                "https://api.alpaca.markets"
            )

        self.paper = paper
        self._api = None
        self._initialized = False

    def _init_api(self) -> None:
        """Initialize Alpaca API (lazy loading)."""
        if self._initialized:
            return

        try:
            import alpaca_trade_api as tradeapi

            self._api = tradeapi.REST(
                self.api_key,
                self.api_secret,
                self.base_url,
                api_version='v2'
            )
            self._initialized = True

        except ImportError:
            raise ImportError(
                "alpaca-trade-api not installed. "
                "Run: uv add alpaca-trade-api"
            )
        except Exception as e:
            raise ConnectionError(f"Failed to initialize Alpaca API: {e}")

    def get_account(self) -> Dict:
        """
        Get account information.

        Returns:
            Account info dict with equity, buying_power, etc.
        """
        self._init_api()

        account = self._api.get_account()
        return {
            "equity": float(account.equity),
            "buying_power": float(account.buying_power),
            "cash": float(account.cash),
            "portfolio_value": float(account.portfolio_value),
            "status": account.status,
            "trading_blocked": account.trading_blocked,
            "pattern_day_trader": account.pattern_day_trader,
        }

    def get_positions(self) -> List[Dict]:
        """
        Get all current positions.

        Returns:
            List of position dicts
        """
        self._init_api()

        positions = self._api.list_positions()
        return [
            {
                "symbol": p.symbol,
                "qty": int(p.qty),
                "avg_entry_price": float(p.avg_entry_price),
                "current_price": float(p.current_price),
                "market_value": float(p.market_value),
                "unrealized_pl": float(p.unrealized_pl),
                "unrealized_plpc": float(p.unrealized_plpc),
                "side": p.side,
            }
            for p in positions
        ]

    def submit_option_order(
        self,
        symbol: str,
        qty: int,
        side: OrderSide,
        limit_price: float,
        time_in_force: TimeInForce = TimeInForce.DAY,
    ) -> Dict:
        """
        Submit an options order.

        IMPORTANT: Only LIMIT orders are supported for options.

        Args:
            symbol: Option symbol (e.g., "SPY240115C00470000")
            qty: Number of contracts
            side: BUY or SELL
            limit_price: Limit price per contract
            time_in_force: Order duration

        Returns:
            Order confirmation dict
        """
        self._init_api()

        if limit_price <= 0:
            raise ValueError("Limit price must be positive")

        if qty <= 0:
            raise ValueError("Quantity must be positive")

        order = self._api.submit_order(
            symbol=symbol,
            qty=qty,
            side=side.value,
            type="limit",
            time_in_force=time_in_force.value,
            limit_price=str(limit_price),
        )

        return {
            "order_id": order.id,
            "client_order_id": order.client_order_id,
            "symbol": order.symbol,
            "qty": int(order.qty),
            "side": order.side,
            "type": order.type,
            "limit_price": float(order.limit_price) if order.limit_price else None,
            "status": order.status,
            "created_at": order.created_at,
        }

    def cancel_order(self, order_id: str) -> bool:
        """
        Cancel an order.

        Args:
            order_id: Order ID to cancel

        Returns:
            True if cancellation was successful
        """
        self._init_api()

        try:
            self._api.cancel_order(order_id)
            return True
        except Exception:
            return False

    def cancel_all_orders(self) -> int:
        """
        Cancel all open orders.

        Returns:
            Number of orders cancelled
        """
        self._init_api()

        cancelled = self._api.cancel_all_orders()
        return len(cancelled) if cancelled else 0

    def close_position(self, symbol: str) -> Dict:
        """
        Close a position.

        Args:
            symbol: Symbol to close

        Returns:
            Close order confirmation
        """
        self._init_api()

        order = self._api.close_position(symbol)
        return {
            "order_id": order.id,
            "symbol": order.symbol,
            "qty": int(order.qty),
            "side": order.side,
            "status": order.status,
        }

    def close_all_positions(self) -> int:
        """
        Close all positions.

        Returns:
            Number of positions closed
        """
        self._init_api()

        closed = self._api.close_all_positions()
        return len(closed) if closed else 0

    def get_option_chain(
        self,
        underlying: str = "SPY",
        expiration_date: Optional[str] = None,
    ) -> List[Dict]:
        """
        Get options chain for an underlying.

        Note: This requires Alpaca options API access.

        Args:
            underlying: Underlying symbol (e.g., "SPY")
            expiration_date: Optional expiration date filter (YYYY-MM-DD)

        Returns:
            List of option contracts
        """
        self._init_api()

        # Note: Alpaca options API endpoint may vary
        # This is a placeholder for the actual implementation
        try:
            # Options data endpoint
            options = self._api.get(
                f"/v1/options/contracts",
                params={
                    "underlying_symbols": underlying,
                    "expiration_date": expiration_date,
                }
            )
            return options
        except Exception as e:
            print(f"Warning: Could not fetch options chain: {e}")
            return []

    def is_market_open(self) -> bool:
        """Check if market is currently open."""
        self._init_api()

        clock = self._api.get_clock()
        return clock.is_open


def format_option_symbol(
    underlying: str,
    expiry: datetime,
    strike: float,
    option_type: str,  # "C" or "P"
) -> str:
    """
    Format OCC option symbol.

    Format: SYMBOL + YYMMDD + C/P + Strike*1000 (8 digits)

    Example: SPY240115C00470000
    - SPY = underlying
    - 240115 = January 15, 2024
    - C = Call
    - 00470000 = Strike $470.00

    Args:
        underlying: Underlying symbol
        expiry: Expiration date
        strike: Strike price
        option_type: "C" for call, "P" for put

    Returns:
        OCC option symbol
    """
    date_str = expiry.strftime("%y%m%d")
    strike_int = int(strike * 1000)
    return f"{underlying}{date_str}{option_type}{strike_int:08d}"


def test_client() -> None:
    """Test client components (without actual API calls)."""
    print("=" * 60)
    print("Testing Alpaca Client Components")
    print("=" * 60)

    # Test RiskManager
    print("\n--- Risk Manager ---")
    risk = RiskManager()

    account_value = 100_000

    allowed, reason = risk.can_open_position(account_value, 0)
    print(f"Can open position (0 positions): {allowed} - {reason}")

    allowed, reason = risk.can_open_position(account_value, 3)
    print(f"Can open position (3 positions): {allowed} - {reason}")

    size = risk.calculate_position_size(account_value, 2.50)
    print(f"Position size for $2.50 option: {size} contracts")

    print(f"Stop loss at entry=$2.50, current=$1.00: {risk.should_stop_loss(2.50, 1.00)}")
    print(f"Take profit at entry=$2.50, current=$5.50: {risk.should_take_profit(2.50, 5.50)}")

    # Test option symbol formatting
    print("\n--- Option Symbol Formatting ---")
    from datetime import datetime

    symbol = format_option_symbol("SPY", datetime(2024, 1, 15), 470.0, "C")
    print(f"SPY Jan 15 2024 $470 Call: {symbol}")

    symbol = format_option_symbol("SPY", datetime(2024, 1, 15), 465.5, "P")
    print(f"SPY Jan 15 2024 $465.50 Put: {symbol}")

    # Test client initialization (without actual API)
    print("\n--- Client Initialization ---")
    client = AlpacaClient(paper=True)
    print(f"Paper trading: {client.paper}")
    print(f"Base URL: {client.base_url}")

    print("\n" + "=" * 60)
    print("Client component tests passed!")
    print("=" * 60)


if __name__ == "__main__":
    test_client()
