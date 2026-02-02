"""
Order manager for handling order lifecycle.

Manages order submission, tracking, and cancellation with support for
limit orders only (as per CLAUDE.md: "Never use market orders for options").
"""
import asyncio
import logging
import time
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional
from zoneinfo import ZoneInfo

from src.execution.alpaca_client import AlpacaClient, OrderSide, TimeInForce


logger = logging.getLogger(__name__)


class OrderStatus(Enum):
    """Order status."""
    PENDING = "pending"
    SUBMITTED = "submitted"
    FILLED = "filled"
    PARTIALLY_FILLED = "partially_filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"
    EXPIRED = "expired"


@dataclass
class OrderConfig:
    """Order execution configuration."""
    max_slippage_pct: float = 1.0  # Max slippage from mid price
    order_timeout_seconds: float = 60.0  # Time before cancelling unfilled order
    use_ioc: bool = True  # Use IOC (Immediate or Cancel) to avoid stuck orders
    price_adjustment_pct: float = 0.5  # Adjust limit price by this % for faster fills


@dataclass
class PendingOrder:
    """Pending or submitted order."""
    order_id: Optional[str]
    client_order_id: str
    option_symbol: str
    side: OrderSide
    qty: int
    limit_price: float
    status: OrderStatus
    submitted_at: datetime
    filled_qty: int = 0
    filled_avg_price: float = 0.0
    last_status_check: Optional[datetime] = None


class OrderManager:
    """
    Manages order lifecycle for options trading.

    Features:
    - Limit orders only (no market orders for options)
    - Slippage control via limit price adjustment
    - IOC (Immediate or Cancel) support to avoid stuck orders
    - Order timeout and automatic cancellation
    - Order status tracking
    """

    def __init__(
        self,
        alpaca_client: AlpacaClient,
        config: Optional[OrderConfig] = None,
    ):
        """
        Initialize the order manager.

        Args:
            alpaca_client: Alpaca trading client
            config: Order execution configuration
        """
        self.alpaca_client = alpaca_client
        self.config = config or OrderConfig()
        self.et_tz = ZoneInfo("America/New_York")

        # Order tracking
        self._pending_orders: Dict[str, PendingOrder] = {}
        self._order_counter = 0

    def _generate_client_order_id(self) -> str:
        """Generate a unique client order ID."""
        self._order_counter += 1
        timestamp = datetime.now(self.et_tz).strftime("%Y%m%d%H%M%S")
        return f"spy0dte_{timestamp}_{self._order_counter:04d}"

    async def submit_entry_order(
        self,
        option_symbol: str,
        qty: int,
        mid_price: float,
    ) -> PendingOrder:
        """
        Submit an entry (buy) order.

        The limit price is set slightly above mid to increase fill probability
        while controlling slippage.

        Args:
            option_symbol: OCC option symbol
            qty: Number of contracts
            mid_price: Current mid price (bid/ask midpoint)

        Returns:
            PendingOrder with order details
        """
        # Calculate limit price: mid + adjustment (pay a bit more for faster fill)
        adjustment = mid_price * (self.config.price_adjustment_pct / 100)
        limit_price = round(mid_price + adjustment, 2)

        # Check slippage limit
        max_price = mid_price * (1 + self.config.max_slippage_pct / 100)
        limit_price = min(limit_price, max_price)

        client_order_id = self._generate_client_order_id()
        time_in_force = TimeInForce.IOC if self.config.use_ioc else TimeInForce.DAY

        logger.info(
            f"Submitting entry order: {option_symbol} "
            f"x{qty} @ ${limit_price:.2f} (mid: ${mid_price:.2f})"
        )

        try:
            result = self.alpaca_client.submit_option_order(
                symbol=option_symbol,
                qty=qty,
                side=OrderSide.BUY,
                limit_price=limit_price,
                time_in_force=time_in_force,
            )

            order = PendingOrder(
                order_id=result["order_id"],
                client_order_id=client_order_id,
                option_symbol=option_symbol,
                side=OrderSide.BUY,
                qty=qty,
                limit_price=limit_price,
                status=OrderStatus.SUBMITTED,
                submitted_at=datetime.now(self.et_tz),
            )

            self._pending_orders[client_order_id] = order
            logger.info(f"Entry order submitted: {result['order_id']}")
            return order

        except Exception as e:
            logger.error(f"Failed to submit entry order: {e}")
            return PendingOrder(
                order_id=None,
                client_order_id=client_order_id,
                option_symbol=option_symbol,
                side=OrderSide.BUY,
                qty=qty,
                limit_price=limit_price,
                status=OrderStatus.REJECTED,
                submitted_at=datetime.now(self.et_tz),
            )

    async def submit_exit_order(
        self,
        option_symbol: str,
        qty: int,
        mid_price: float,
    ) -> PendingOrder:
        """
        Submit an exit (sell) order.

        The limit price is set slightly below mid to increase fill probability.

        Args:
            option_symbol: OCC option symbol
            qty: Number of contracts
            mid_price: Current mid price (bid/ask midpoint)

        Returns:
            PendingOrder with order details
        """
        # Calculate limit price: mid - adjustment (accept a bit less for faster fill)
        adjustment = mid_price * (self.config.price_adjustment_pct / 100)
        limit_price = round(mid_price - adjustment, 2)
        limit_price = max(limit_price, 0.01)  # Minimum price

        # Check slippage limit
        min_price = mid_price * (1 - self.config.max_slippage_pct / 100)
        limit_price = max(limit_price, min_price)

        client_order_id = self._generate_client_order_id()
        time_in_force = TimeInForce.IOC if self.config.use_ioc else TimeInForce.DAY

        logger.info(
            f"Submitting exit order: {option_symbol} "
            f"x{qty} @ ${limit_price:.2f} (mid: ${mid_price:.2f})"
        )

        try:
            result = self.alpaca_client.submit_option_order(
                symbol=option_symbol,
                qty=qty,
                side=OrderSide.SELL,
                limit_price=limit_price,
                time_in_force=time_in_force,
            )

            order = PendingOrder(
                order_id=result["order_id"],
                client_order_id=client_order_id,
                option_symbol=option_symbol,
                side=OrderSide.SELL,
                qty=qty,
                limit_price=limit_price,
                status=OrderStatus.SUBMITTED,
                submitted_at=datetime.now(self.et_tz),
            )

            self._pending_orders[client_order_id] = order
            logger.info(f"Exit order submitted: {result['order_id']}")
            return order

        except Exception as e:
            logger.error(f"Failed to submit exit order: {e}")
            return PendingOrder(
                order_id=None,
                client_order_id=client_order_id,
                option_symbol=option_symbol,
                side=OrderSide.SELL,
                qty=qty,
                limit_price=limit_price,
                status=OrderStatus.REJECTED,
                submitted_at=datetime.now(self.et_tz),
            )

    async def submit_stop_loss_order(
        self,
        option_symbol: str,
        qty: int,
        current_price: float,
    ) -> PendingOrder:
        """
        Submit a stop loss exit order with aggressive pricing.

        For stop losses, we prioritize getting out quickly over price.

        Args:
            option_symbol: OCC option symbol
            qty: Number of contracts
            current_price: Current option price

        Returns:
            PendingOrder with order details
        """
        # For stop loss, use a more aggressive price (5% below current)
        limit_price = round(current_price * 0.95, 2)
        limit_price = max(limit_price, 0.01)

        client_order_id = self._generate_client_order_id()

        logger.warning(
            f"Submitting STOP LOSS order: {option_symbol} "
            f"x{qty} @ ${limit_price:.2f} (current: ${current_price:.2f})"
        )

        try:
            result = self.alpaca_client.submit_option_order(
                symbol=option_symbol,
                qty=qty,
                side=OrderSide.SELL,
                limit_price=limit_price,
                time_in_force=TimeInForce.IOC,  # Always IOC for stop loss
            )

            order = PendingOrder(
                order_id=result["order_id"],
                client_order_id=client_order_id,
                option_symbol=option_symbol,
                side=OrderSide.SELL,
                qty=qty,
                limit_price=limit_price,
                status=OrderStatus.SUBMITTED,
                submitted_at=datetime.now(self.et_tz),
            )

            self._pending_orders[client_order_id] = order
            logger.info(f"Stop loss order submitted: {result['order_id']}")
            return order

        except Exception as e:
            logger.error(f"Failed to submit stop loss order: {e}")
            return PendingOrder(
                order_id=None,
                client_order_id=client_order_id,
                option_symbol=option_symbol,
                side=OrderSide.SELL,
                qty=qty,
                limit_price=limit_price,
                status=OrderStatus.REJECTED,
                submitted_at=datetime.now(self.et_tz),
            )

    async def check_order_status(self, order: PendingOrder) -> PendingOrder:
        """
        Check and update order status.

        Args:
            order: Order to check

        Returns:
            Updated order with current status
        """
        if order.order_id is None:
            return order

        # In a real implementation, we would query the broker API
        # For now, this is a placeholder
        order.last_status_check = datetime.now(self.et_tz)

        # Check for timeout
        elapsed = (datetime.now(self.et_tz) - order.submitted_at).total_seconds()
        if elapsed > self.config.order_timeout_seconds and order.status == OrderStatus.SUBMITTED:
            logger.warning(f"Order {order.order_id} timed out, cancelling")
            await self.cancel_order(order)
            order.status = OrderStatus.EXPIRED

        return order

    async def cancel_order(self, order: PendingOrder) -> bool:
        """
        Cancel an order.

        Args:
            order: Order to cancel

        Returns:
            True if cancellation was successful
        """
        if order.order_id is None:
            return False

        try:
            result = self.alpaca_client.cancel_order(order.order_id)
            if result:
                order.status = OrderStatus.CANCELLED
                logger.info(f"Order {order.order_id} cancelled")
            return result
        except Exception as e:
            logger.error(f"Failed to cancel order {order.order_id}: {e}")
            return False

    async def cancel_all_orders(self) -> int:
        """
        Cancel all pending orders.

        Returns:
            Number of orders cancelled
        """
        try:
            count = self.alpaca_client.cancel_all_orders()
            logger.info(f"Cancelled {count} orders")

            # Update local tracking
            for order in self._pending_orders.values():
                if order.status == OrderStatus.SUBMITTED:
                    order.status = OrderStatus.CANCELLED

            return count
        except Exception as e:
            logger.error(f"Failed to cancel all orders: {e}")
            return 0

    async def wait_for_fill(
        self,
        order: PendingOrder,
        timeout: float = 5.0,
        poll_interval: float = 0.5,
    ) -> PendingOrder:
        """
        Wait for an order to fill.

        Args:
            order: Order to wait for
            timeout: Maximum time to wait in seconds
            poll_interval: Time between status checks

        Returns:
            Updated order with final status
        """
        start_time = time.time()

        while time.time() - start_time < timeout:
            order = await self.check_order_status(order)

            if order.status in (
                OrderStatus.FILLED,
                OrderStatus.CANCELLED,
                OrderStatus.REJECTED,
                OrderStatus.EXPIRED,
            ):
                return order

            await asyncio.sleep(poll_interval)

        # Timeout - try to cancel
        if order.status == OrderStatus.SUBMITTED:
            await self.cancel_order(order)
            order.status = OrderStatus.EXPIRED

        return order

    def get_pending_orders(self) -> List[PendingOrder]:
        """Get all pending orders."""
        return [
            order for order in self._pending_orders.values()
            if order.status in (OrderStatus.PENDING, OrderStatus.SUBMITTED)
        ]
