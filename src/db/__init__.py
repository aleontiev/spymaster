"""
Database layer for SpyMaster.

Provides SQLite/PostgreSQL-compatible storage for:
- Checkpoints (model configurations and training metadata)
- Strategies (trading strategy configurations)
- Backtests (backtest results and metrics)

Usage:
    from src.db import get_db, Checkpoint, Strategy, Backtest

    db = get_db()
    db.create_checkpoint(name="lejepa-v1", model_type="lejepa", ...)
    checkpoints = db.list_checkpoints()
"""

from src.db.database import (
    Database,
    get_db,
    init_db,
)
from src.db.models import (
    Checkpoint,
    Strategy,
    Backtest,
    MarketCalendar,
)

__all__ = [
    "Database",
    "get_db",
    "init_db",
    "Checkpoint",
    "Strategy",
    "Backtest",
    "MarketCalendar",
]
