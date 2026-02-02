"""Execution module for trading operations."""
from src.execution.alpaca_client import (
    AlpacaClient,
    OrderSide,
    OrderType,
    RiskConfig,
    RiskManager,
    TimeInForce,
    format_option_symbol,
)
from src.execution.filtered_executor import FilteredExecutor
from src.execution.heuristic import HeuristicModel


def __getattr__(name: str):
    """Lazy import to avoid circular import with src.backtest."""
    if name == "RealOptionsProvider":
        from src.execution.options_provider import RealOptionsProvider
        return RealOptionsProvider
    if name == "PositionManager":
        from src.execution.position_manager import PositionManager
        return PositionManager
    if name == "LivePosition":
        from src.execution.position_manager import LivePosition
        return LivePosition
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    # Alpaca client
    "AlpacaClient",
    "OrderSide",
    "OrderType",
    "RiskConfig",
    "RiskManager",
    "TimeInForce",
    "format_option_symbol",
    # Position manager
    "LivePosition",
    "PositionManager",
    # Filtered executor
    "FilteredExecutor",
    # Heuristic model
    "HeuristicModel",
    # Options provider
    "RealOptionsProvider",
]
