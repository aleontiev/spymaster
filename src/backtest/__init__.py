"""Backtesting module for trading strategies."""
from src.backtest.types import (
    BacktestConfig,
    MonthlyMetrics,
    PendingEntry,
    Position,
    PositionType,
    Trade,
    TradingConfig,
)
from src.backtest.results import print_results


def __getattr__(name: str):
    """Lazy import to avoid circular import with src.execution.position_manager."""
    if name == "MultiPercentileBacktester":
        from src.backtest.multi_percentile import MultiPercentileBacktester
        return MultiPercentileBacktester
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "BacktestConfig",
    "MonthlyMetrics",
    "MultiPercentileBacktester",
    "PendingEntry",
    "Position",
    "PositionType",
    "Trade",
    "TradingConfig",
    "print_results",
]
